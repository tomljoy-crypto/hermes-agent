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
