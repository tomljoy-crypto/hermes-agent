"""Dispatcher subprocess: submit one claim to its profile owner; never run an agent."""
# Spawned as a bare ``sys.executable -m``: in a PM install that interpreter carries no dependencies
# until hermes_bootstrap selects the committed environment, so it must be the first import.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap" and not "hermes_bootstrap".startswith(f"{exc.name}."):
        raise

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Optional

# A worker that loses its owner (a dropped socket, a restart, `hermes update` draining it or a new
# owner still starting) waits for it instead of exiting: the condition is the host's, not the card's,
# and a failed exit here spends the card's failure budget and lets the owner end the turn. Asking
# again is safe: ``kanban.run`` is idempotent per attempt and ``prompt.receipt`` is a read.
# ``OWNER_WAIT_S`` is the longest the owner may stay silent between answers before the worker
# gives up (the first ``kanban.run`` answer has its own wait, ``admission_answer_budget``).
OWNER_WAIT_S = 180.0
_TRANSIENT_OWNER_STATES = frozenset({'draining', 'starting'})
# In-band owner refusals that leave the attempt intact: a draining owner admits nothing new, and a
# retired authority's successor holds the admission the next connection reads.
_TRANSIENT_RPC_REASONS = frozenset({'runtime_draining', 'runtime_retired'})
# Past the board lock wait: the owner's session setup and receipt before and after the write.
ADMISSION_ANSWER_MARGIN_S = 60.0
# A scoped worker outlives its owner: the process is its own systemd scope and its claim lease is the
# board's, so an owner restart that interrupts the turn (reason ``gateway/run.py``'s restart cause)
# must not end the card's attempt. Inside this window the worker asks the NEXT owner to re-admit the
# SAME attempt (a fresh admission on the same session, chained by request id) and keeps polling, so
# the turn continues and the run is never booked with the interrupt's 130. A settlement that is not
# the owner's own stop, or a window that runs out, stays final: the worker relays its exit code, or
# names the unrecoverable claim, rather than surviving silently.
READMISSION_WINDOW_S = 120.0


def readmission_window():
    """``READMISSION_WINDOW_S``, overridable per host (``HERMES_KANBAN_READMISSION_WINDOW_S``)."""
    raw = os.environ.get('HERMES_KANBAN_READMISSION_WINDOW_S')
    if raw is None:
        return READMISSION_WINDOW_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return READMISSION_WINDOW_S


class KanbanClaimUnrecoverable(Exception):
    """The attempt's claim could not be re-bound inside the re-admission window.

    Carries the specific, greppable cause (``worker claim unrecoverable after re-admission window:
    <cause>``) so the card's log names the condition instead of a bare ``settled interrupted``.
    """


def admission_answer_budget():
    """Seconds to wait for ``kanban.run``'s answer: the longest the owner's admission write can
    wait on the board lock (``write_txn`` retries ``BEGIN IMMEDIATE`` ``_BUSY_MAX_RETRIES`` times,
    each under the busy timeout) plus ``ADMISSION_ANSWER_MARGIN_S``."""
    from hermes_cli import kanban_db_connect
    lock_wait = (kanban_db_connect._BUSY_MAX_RETRIES + 1) * kanban_db_connect._resolve_busy_timeout_ms() / 1000
    return lock_wait + ADMISSION_ANSWER_MARGIN_S


@dataclass
class _Claim:
    """What the owner has said about this attempt, carried across connections."""
    answered: float  # monotonic time of the owner's last answer
    session_id: Optional[str] = None
    admission_id: Optional[str] = None
    readmits: int = 0  # re-admissions this attempt has been bound to (owner-side request-id chain)
    readmit_attempts: int = 0  # asks made inside the window, bound or deferred
    readmit_deadline: Optional[float] = None  # monotonic deadline of the re-admission window
    readmit_cause: str = ''

    def silent_for(self):
        return time.monotonic() - self.answered

    def expire_readmission(self):
        """Fail the attempt once its re-admission window is spent: survival stays bounded."""
        if self.readmit_deadline is not None and time.monotonic() >= self.readmit_deadline:
            raise KanbanClaimUnrecoverable(
                'worker claim unrecoverable after re-admission window: '
                f'{self.readmit_cause} (gave up after {self.readmit_attempts} re-admission attempt(s))')


async def run(params, board_db):
    from agent.retry_utils import jittered_backoff
    from hermes_cli.gateway_client import connect_gateway, GatewayClientError, outside_restart_wait
    from hermes_cli.gateway_owner_restart import restartable
    from websockets.exceptions import WebSocketException
    if os.environ.get('HERMES_TUI_GATEWAY_URL'):
        raise GatewayClientError('Kanban requires the assigned local profile owner')
    # A lost socket is not a lost attempt, and exiting here would let ``_watch_submitter`` end the
    # owner's in-flight turn, so the next connection resumes whatever the last one reached. The
    # wait is bounded from the owner's last answer: an owner that keeps answering keeps the
    # worker, and one that stays silent for ``OWNER_WAIT_S`` fails the worker honestly.
    claim = _Claim(answered=time.monotonic())
    failures = 0
    while True:
        # A spent re-admission window fails the attempt here, whatever kept it open (an owner that
        # keeps refusing, or one that never came back): the claim is unrecoverable, and says why.
        claim.expire_readmission()
        seen = claim.answered
        readmitted = False
        receipt = None
        try:
            # Never into a restart's after-turn wait: it cut this socket so the turn's submitter
            # waits for the NEXT owner, and a worker polling its receipt inside the wait keeps the
            # turn it waits on alive, holding `hermes update` until restart_after_turn_timeout.
            with outside_restart_wait():
                async with connect_gateway() as client:
                    # kanban.run subscribes this socket to the card's session, but only receipts
                    # matter here: discard the turn's events, or they fill the client's bounded
                    # queue and its reader stops, with the next receipt behind it.
                    drain = asyncio.create_task(_discard(client.events))
                    try:
                        receipt = await _await_receipt(client, params, board_db, claim)
                        readmitted = await _readmit(client, params, board_db, claim, receipt)
                    finally:
                        drain.cancel()
            if not readmitted:
                return _exit_code(receipt, params, board_db)
        # A restart also surfaces as a refused or reset dial, a failed handshake, or an answer
        # that never came.
        except (GatewayClientError, WebSocketException, OSError, TimeoutError) as exc:
            if not restartable(exc):
                raise
            if claim.silent_for() >= OWNER_WAIT_S:
                # WebSocket errors can embed remote bodies; only our own messages are quoted.
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                detail = exc if isinstance(exc, GatewayClientError) else type(exc).__name__ + (
                    f' (HTTP {status})' if status else '')
                raise GatewayClientError(f'Kanban owner silent for {claim.silent_for():.0f}s: {detail}') from exc
        failures = 1 if claim.answered != seen else failures + 1
        # Jittered (1-2 s first, up to 16 s): workers of one dead owner do not all redial, and so
        # start a replacement, in the same instant.
        await asyncio.sleep(jittered_backoff(failures, base_delay=1.0, max_delay=8.0, jitter_ratio=1.0))


async def _discard(events):
    while True:
        await events.get()


async def _await_receipt(client, params, board_db, claim):
    """Submit the claim (or, on a later connection, re-read the admission it got) and return its
    settled receipt, recording each owner answer on *claim*."""
    if claim.admission_id is None:
        db = str(await asyncio.to_thread(Path(board_db).resolve))
        # The owner answers only after it checks the claim and writes its ``owner_admitted`` marker
        # in one board write transaction. The board (the cross-profile dispatch bus) can make that
        # wait out its busy timeout, so the 30 s default left the owner admitting an attempt whose
        # submitter had exited (the later-bound worker's submitter-liveness watcher ended it, and the
        # admission was left ``unknown``). The wait is still finite: once the marker commits,
        # ``enforce_max_runtime`` and TTL reclaim skip the run (``owner_reclaim_paused``), so an owner
        # that stalls after admitting is bounded by this wait and ``OWNER_WAIT_S``.
        accepted = await client.rpc('kanban.run', _timeout=admission_answer_budget(), **dict(params, db=db))
        receipt = accepted['receipt']
        claim.session_id, claim.admission_id = accepted['session_id'], receipt['admission_id']
        print(f'Session: {claim.session_id}', file=sys.stderr)
    else:
        receipt = await _receipt(client, claim)
    claim.answered = time.monotonic()
    while receipt['status'] not in {'terminal', 'unknown'}:
        await asyncio.sleep(.2)
        receipt = await _receipt(client, claim)
        claim.answered = time.monotonic()
    return receipt


def _exit_code(receipt, params, board_db):
    from hermes_cli.gateway_client import GatewayClientError
    if receipt['status'] == 'unknown':
        raise GatewayClientError('Kanban execution unknown; retained by owner')
    from gateway.session_kanban import worker_result
    result = worker_result(board_db, params)
    # The attempt's reply ends the log, as a one-shot worker's did: crash diagnostics quote it.
    if result.get('last_output'):
        print(result['last_output'])
    return result.get('exit_code', 1)


async def _receipt(client, claim):
    return await client.rpc('prompt.receipt', session_id=claim.session_id, admission_id=claim.admission_id)


async def _readmit(client, params, board_db, claim, receipt):
    """Re-bind this attempt to a fresh admission after the owner's own restart cut its turn.

    True: the claim is bound again (or the owner is still coming back) and the caller must keep
    waiting on it. False: this settlement is final and its exit code is the card's answer — a user's
    Stop, a failed turn and a finished run keep the existing exit-code contract. Raises
    ``KanbanClaimUnrecoverable`` when the owner refuses the re-admission or the window is spent.
    """
    from hermes_cli.gateway_client import GatewayRPCError
    cause = await _owner_stop_cause(client, claim, receipt)
    if cause is None:
        return False
    claim.readmit_cause = cause
    now = time.monotonic()
    if claim.readmit_deadline is None:
        claim.readmit_deadline = now + readmission_window()
    claim.expire_readmission()
    seq = claim.readmits + 1
    claim.readmit_attempts += 1
    # One line per attempt, with the cause: a card's log then shows exactly how long and why the
    # worker held on. The sequence is not advanced until the owner answers, so a lost answer retries
    # the SAME request id and the owner returns the admission it already made.
    print(f'kanban re-admission attempt {seq} ({claim.readmit_attempts} inside the window) at {_stamp()} '
          f'for {params.get("task_id")} run {params.get("run_id")}: {cause}', file=sys.stderr, flush=True)
    try:
        accepted = await client.rpc('kanban.readmit', _timeout=admission_answer_budget(),
                                    re_admission=seq, **dict(params, db=str(Path(board_db).resolve())))
    except GatewayRPCError as exc:
        if str(exc) in _TRANSIENT_RPC_REASONS:
            # The owner is still draining or starting: keep the window, redial, ask again.
            print(f'kanban re-admission {seq} deferred by the owner: {exc}', file=sys.stderr, flush=True)
            return True
        raise KanbanClaimUnrecoverable(
            f'worker claim unrecoverable after re-admission window: admission refused: {exc}') from exc
    claim.session_id, claim.admission_id = accepted['session_id'], accepted['receipt']['admission_id']
    claim.readmits = seq
    claim.answered = time.monotonic()
    print(f'kanban re-admission {seq} bound: session {claim.session_id} admission '
          f'{claim.admission_id}', file=sys.stderr, flush=True)
    return True


async def _owner_stop_cause(client, claim, receipt):
    """The owner's own restart/shutdown cause for *receipt*'s interruption, else None.

    Only an admission the owner itself stopped for its restart is re-bindable. The cause comes from
    the settled result's ``interrupt_message`` (``gateway/session_bot_outcome.owner_stopped``), so a
    user's Stop and every other outcome stay final. An admission the owner lost unresolved
    (``unknown``) is re-bound too: it never reached a verdict of its own.
    """
    if receipt['status'] == 'unknown':
        return 'the owner lost the admission unresolved (unknown)'
    if receipt.get('outcome') != 'interrupted':
        # Not an interruption: a completed or failed turn is final, and asking for its result would
        # only add an RPC to the happy path.
        return None
    detail = await client.rpc('prompt.receipt', session_id=claim.session_id,
                              admission_id=claim.admission_id, include_result=True)
    result = detail.get('result') or {}
    from gateway.session_bot_outcome import owner_stopped
    if not result.get('interrupted') or not owner_stopped(result):
        return None
    return f'the owner stopped the turn ({result.get("interrupt_message")})'


def _stamp():
    try:
        return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    except AttributeError:  # a test's fake clock
        return f'{time.monotonic():.0f}s'


def main():
    # The dead-worker sweep books this process's exit from the trailer in its log (it never
    # reaps us), so every exit path writes it — the same contract as a quiet one-shot worker.
    from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER
    rc = 1
    try:
        rc = _submit()
    finally:
        print(f'\n{KANBAN_WORKER_EXIT_TRAILER}{int(rc)}', file=sys.stderr, flush=True)
    return rc


def _submit():
    try:
        params = {'board': os.environ['HERMES_KANBAN_BOARD'], 'task_id': os.environ['HERMES_KANBAN_TASK'],
            'run_id': int(os.environ['HERMES_KANBAN_RUN_ID']), 'claim_lock': os.environ['HERMES_KANBAN_CLAIM_LOCK']}
        board_db = os.environ['HERMES_KANBAN_DB']
        # Owner startup must never inherit this worker's task/tool/goal identity.
        for key in list(os.environ):
            # The busy timeout is a host tuning knob, not task identity: the answer budget reads it.
            if key.startswith('HERMES_KANBAN_') and key not in ('HERMES_KANBAN_HOME', 'HERMES_KANBAN_BUSY_TIMEOUT_MS'):
                os.environ.pop(key)
        for key in ('HERMES_SESSION_SOURCE', 'HERMES_TENANT', 'TERMINAL_CWD'):
            os.environ.pop(key, None)
        return asyncio.run(run(params, board_db))
    except KanbanClaimUnrecoverable as exc:
        # A specific, greppable reason (never a bare 'settled interrupted'): the claim this worker
        # held could not be re-bound, so the run's result is this failure, booked as one.
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    except (KeyError, ValueError, OSError) as exc:
        print(f'Kanban owner refused: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
