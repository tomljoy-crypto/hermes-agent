"""A scoped worker re-binds its claim when its OWN owner restarts mid-turn.

The worker's process (its own systemd scope) and its claim lease (the board's) outlive the gateway,
but the turn does not: ``gateway/run_shutdown.py:864 stop_managed_turns`` interrupts it on a
restart, the admission settles ``terminal/interrupted`` and is never replayed
(``gateway/session_settlement_recovery.py:131-134``). Before this fix the client relayed that
settlement's exit code 130 as the run's final answer, so a card died because the host restarted.

These tests pin both arms and the boundary:

  R1  a restart inside the re-admission window: the worker asks the next owner to re-admit the same
      attempt, gets a fresh admission, keeps polling it and finishes the turn on it — exit 0, no
      relayed 130, an ``attempt``/``bound`` log line per attempt;
  R2  the window spent (or the owner refusing the re-admission): ``KanbanClaimUnrecoverable``, whose
      message is the specific, greppable ``worker claim unrecoverable after re-admission window:``;
  R3  a settlement that is NOT the owner's own stop (a user's Stop) is never re-admitted and its
      exit code is relayed unchanged: survival never widens into silence.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_client, kanban_worker_client
from hermes_cli.kanban_worker_client import KanbanClaimUnrecoverable

RESTART_CAUSE = "Gateway restarting"
USER_STOP_CAUSE = "Stopped by user"


class _Owner:
    """A profile owner whose restart interrupts the card's first admission.

    ``interrupt`` is the settled ``interrupt_message`` of admission ``a1``; ``readmit`` decides what
    the next owner answers to ``kanban.readmit`` (a fresh admission, a transient refusal, or a final
    one). ``a2``'s turn completes normally.
    """

    def __init__(self, *, interrupt=RESTART_CAUSE, readmit=None, first_status="terminal"):
        self.interrupt = interrupt
        self.readmit = readmit or (lambda **_kwargs: {"session_id": "s1",
                                                      "receipt": {"status": "pending", "admission_id": "a2"},
                                                      "re_admission": 1, "cause": f"owner stopped the turn ({RESTART_CAUSE})"})
        self.first_status = first_status
        self.methods = []
        self.admissions = []
        self.dials = 0
        self.last_terminal = None
        self.rebound = False

    def connect(self):
        owner = self

        @asynccontextmanager
        async def _connect(*_, **__):
            owner.dials += 1

            class _Client:
                events = asyncio.Queue()

                async def rpc(self, method, **params):
                    owner.methods.append(method)
                    if method == "kanban.run":
                        return {"session_id": "s1", "receipt": {"status": "pending", "admission_id": "a1"}}
                    if method == "kanban.readmit":
                        owner.rebound = True
                        return owner.readmit(**params)
                    assert method == "prompt.receipt", method
                    admission = params.get("admission_id")
                    if admission is None:  # the re-admission's own receipt
                        admission = "a2"
                    owner.admissions.append(admission)
                    if admission == "a1":
                        if params.get("include_result") is True:
                            return {"status": owner.first_status, "admission_id": "a1",
                                    "result": {"interrupted": True, "interrupt_message": owner.interrupt}}
                        # Pending until the restart is noticed, then the settlement the dying owner wrote.
                        if owner.first_status == "terminal":
                            owner.last_terminal = "a1"
                        return {"status": owner.first_status, "admission_id": "a1",
                                "outcome": "interrupted" if owner.first_status == "terminal" else None}
                    owner.last_terminal = "a2"
                    return {"status": "terminal", "admission_id": "a2"}

            yield _Client()

        return _connect


def _wire(monkeypatch, owner, *, clock=None, window=None):
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())
    if clock is None:
        async def _sleep(_seconds):
            return None
    else:
        async def _sleep(seconds):
            clock[0] += seconds
        monkeypatch.setattr(kanban_worker_client, "time", SimpleNamespace(
            monotonic=lambda: clock[0], strftime=__import__("time").strftime, gmtime=__import__("time").gmtime))
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _sleep)
    if window is not None:
        monkeypatch.setattr(kanban_worker_client, "readmission_window", lambda: window)
    from gateway import session_kanban
    # The run's booked result is the LAST worker_result: the rebound turn's, never the interrupt's.
    monkeypatch.setattr(session_kanban, "worker_result",
                        lambda *_args: {"exit_code": 0 if owner.last_terminal == "a2" else 130})


@pytest.mark.asyncio
async def test_R1_restart_within_window_rebinds_the_claim_and_finishes_the_turn(monkeypatch, tmp_path, capsys):
    owner = _Owner()
    _wire(monkeypatch, owner)

    assert await kanban_worker_client.run({"task_id": "t1", "run_id": 7, "claim_lock": "lock"},
                                          str(tmp_path / "kanban.db")) == 0
    # Same turn, same session: one re-admission, then the client polls the admission it was given.
    assert owner.methods.count("kanban.readmit") == 1
    assert "a2" in owner.admissions and owner.admissions[-1] == "a2"
    assert owner.rebound is True
    log = capsys.readouterr().err
    assert "kanban re-admission attempt 1 (1 inside the window) at " in log
    assert "kanban re-admission 1 bound: session s1 admission a2" in log
    # Never the interrupt's 130: the client's own answer is the completed turn's.
    assert owner.last_terminal == "a2"


@pytest.mark.asyncio
async def test_R2_spent_readmission_window_names_the_unrecoverable_claim(monkeypatch, tmp_path, capsys):
    clock = [1_000.0]

    def _defer(**_kwargs):
        raise gateway_client.GatewayRPCError("runtime_draining")
    owner = _Owner(readmit=_defer)
    _wire(monkeypatch, owner, clock=clock, window=30.0)

    with pytest.raises(KanbanClaimUnrecoverable) as raised:
        await kanban_worker_client.run({"task_id": "t1", "run_id": 7, "claim_lock": "lock"},
                                       str(tmp_path / "kanban.db"))
    message = str(raised.value)
    assert message.startswith("worker claim unrecoverable after re-admission window: ")
    assert RESTART_CAUSE in message
    assert "(gave up after " in message and "re-admission attempt(s))" in message
    # Every attempt inside the window was logged, none of them silent.
    log = capsys.readouterr().err
    assert "kanban re-admission attempt 1 (1 inside the window) at " in log
    assert "kanban re-admission attempt 1 (2 inside the window) at " in log
    assert "deferred by the owner: runtime_draining" in log
    assert clock[0] >= 1_030.0


@pytest.mark.asyncio
async def test_R2b_a_refused_readmission_names_the_refusal(monkeypatch, tmp_path):
    def _refuse(**_kwargs):
        raise gateway_client.GatewayRPCError("readmission_refused")
    owner = _Owner(readmit=_refuse)
    _wire(monkeypatch, owner)

    with pytest.raises(KanbanClaimUnrecoverable) as raised:
        await kanban_worker_client.run({"task_id": "t1", "run_id": 7, "claim_lock": "lock"},
                                       str(tmp_path / "kanban.db"))
    assert str(raised.value) == ("worker claim unrecoverable after re-admission window: "
                                 "admission refused: readmission_refused")


@pytest.mark.asyncio
async def test_R3_a_user_stop_is_never_re_admitted_and_relays_its_exit_code(monkeypatch, tmp_path):
    owner = _Owner(interrupt=USER_STOP_CAUSE)
    _wire(monkeypatch, owner)

    assert await kanban_worker_client.run({"task_id": "t1", "run_id": 7, "claim_lock": "lock"},
                                          str(tmp_path / "kanban.db")) == 130
    assert "kanban.readmit" not in owner.methods
    assert owner.last_terminal == "a1"


@pytest.mark.asyncio
async def test_R3b_an_unknown_admission_is_rebound_because_it_never_settled(monkeypatch, tmp_path):
    owner = _Owner(first_status="unknown")
    _wire(monkeypatch, owner)

    assert await kanban_worker_client.run({"task_id": "t1", "run_id": 7, "claim_lock": "lock"},
                                          str(tmp_path / "kanban.db")) == 0
    assert owner.methods.count("kanban.readmit") == 1
