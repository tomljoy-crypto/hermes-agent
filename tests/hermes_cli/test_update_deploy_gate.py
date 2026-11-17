import subprocess

import pytest

from hermes_cli import update_deploy_gate as gate


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    _git(root, "init", "-q", "-b", "feature")
    (root / "a.txt").write_text("one\n")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-q", "-m", "init")
    return root


def test_clean_feature_branch_is_allowed(repo):
    assert gate.refusal(repo) is None


def test_deploy_live_is_refused(repo, monkeypatch):
    monkeypatch.delenv(gate.ALLOW_ENV, raising=False)
    _git(repo, "switch", "-q", "-c", "deploy/live")
    msg = gate.refusal(repo)
    assert msg and "deploy/live" in msg and "No gateway was stopped" in msg


def test_deploy_live_break_glass(repo, monkeypatch):
    _git(repo, "switch", "-q", "-c", "deploy/live")
    monkeypatch.setenv(gate.ALLOW_ENV, "1")
    assert gate.refusal(repo) is None


def test_skip_worktree_is_refused(repo):
    _git(repo, "update-index", "--skip-worktree", "a.txt")
    msg = gate.refusal(repo)
    assert msg and "a.txt" in msg and "skip-worktree" in msg


def test_assume_unchanged_is_refused(repo):
    _git(repo, "update-index", "--assume-unchanged", "a.txt")
    assert "a.txt" in (gate.refusal(repo) or "")


def test_non_git_dir_is_ignored(tmp_path):
    assert gate.refusal(tmp_path) is None


def test_dashboard_update_is_refused_before_spawning(monkeypatch):
    """POST /api/hermes/update must say no itself, not answer ok and fail silently in the background."""
    import asyncio

    from hermes_cli.web_routers import actions

    spawned = []
    monkeypatch.setattr(actions, "_spawn_hermes_action", lambda *a, **k: spawned.append(a))
    monkeypatch.setattr(actions, "is_commit_build", lambda root: False)
    monkeypatch.setattr(actions, "_dashboard_local_update_managed_externally", lambda: False)
    monkeypatch.setattr("hermes_cli.update_contract.evaluate_update_admission", lambda root: None)
    monkeypatch.setattr("hermes_cli.update_contract.record_refusal_receipt", lambda r: None)
    monkeypatch.setattr(gate, "refusal", lambda root: "✗ refused for the test. No gateway was stopped.")
    out = asyncio.run(actions.update_hermes())
    assert out["ok"] is False and "No gateway was stopped" in out["message"] and not spawned
