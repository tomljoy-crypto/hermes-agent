"""Refuse ``hermes update`` BEFORE it pauses anything, on this fleet's install tree.

Why this exists (2026-11-16 22:20 and 2026-11-17 07:05, same cause twice): ``hermes update`` paused
the gateways, then ``git checkout --detach <release tag>`` aborted on a file that was only *hidden*
as clean (skip-worktree bit over a CRLF/LF phantom), the rollback's ``git restore`` pathspec skips
skip-worktree entries, and the updater exited leaving the fleet stopped. Separately, on a release
channel the update ALWAYS lands detached at the tag, abandoning the named branch the deploy policy
requires (``deploy/live``: vault 04-Infrastructure/Hermes-Install-Tree-Deploy-Policy.md, section 7:
promotion is a reviewed merge of origin/main into deploy/live, never ``hermes update``).

``refusal(root)`` returns a message when the update must not start, else None. It only READS git state
(no stop, no checkout, no stash), so a refusal leaves the fleet exactly as it was.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

DEPLOY_BRANCH = "deploy/live"
ALLOW_ENV = "HERMES_UPDATE_ALLOW_DEPLOY_LIVE"
POLICY = "vault 04-Infrastructure/Hermes-Install-Tree-Deploy-Policy.md (section 7: promote = merge, not `hermes update`)"


def _git(root: Path, *args: str) -> str | None:
    try:
        run = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=30, stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return run.stdout if run.returncode == 0 else None


def hidden_index_paths(root: Path) -> list[str]:
    """Tracked paths whose changes ``git status`` is told to ignore: skip-worktree ('S') or assume-unchanged (lowercase tag)."""
    out = _git(root, "ls-files", "-v")
    if not out:
        return []
    return [line[2:] for line in out.splitlines()
            if line[:1] == "S" or (line[:1].islower() and line[:1].isalpha())]


def refusal(root: Path) -> str | None:
    if not (Path(root) / ".git").exists():
        return None  # not a git checkout (zip/docker/nix installs have their own paths)
    hidden = hidden_index_paths(root)
    if hidden:
        shown = ", ".join(hidden[:5]) + (f" (+{len(hidden) - 5} more)" if len(hidden) > 5 else "")
        return ("✗ `hermes update` refused: git is hiding edits in this checkout (skip-worktree / assume-unchanged): "
                f"{shown}.\n  `git status` cannot see them, git would abort the checkout midway and the rollback would fail "
                "while the gateways are paused.\n  Clear the flag (`git update-index --no-skip-worktree <path>`) and commit "
                "or discard the change first. No gateway was stopped.")
    branch = (_git(root, "symbolic-ref", "-q", "--short", "HEAD") or "").strip()
    if branch == DEPLOY_BRANCH and os.environ.get(ALLOW_ENV) != "1":
        return (f"✗ `hermes update` refused: this tree runs on `{DEPLOY_BRANCH}`, and an update would detach it at the "
                "release commit (abandoning every local fix the fleet runs on).\n"
                f"  Promote by merging origin/main into {DEPLOY_BRANCH} and restarting the affected gateways: {POLICY}.\n"
                f"  Break-glass for a human only: {ALLOW_ENV}=1. No gateway was stopped.")
    return None
