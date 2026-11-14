"""Kanban decomposer — fan a triage task out into a graph of child tasks.

Invoked by ``hermes kanban decompose [task_id | --all]`` and the gateway
dispatcher's auto-decompose path. Reads the profile roster (with
descriptions), asks the auxiliary LLM for a task graph in JSON, then
atomically creates the children, links them under the root, and flips the
root ``triage -> todo``. The root stays alive as parent of every leaf child so
it wakes back up when the graph completes and its assignee (the orchestrator
profile) can judge completion and add more work.

Mirrors ``kanban_specify`` (lazy aux import, lenient parse, never raises on
expected failures). ``fanout=false`` collapses to the ``specify`` behaviour
(tighten + promote, no children), making ``decompose`` a strict superset.
Unknown assignees are rewritten to ``default_assignee`` — a child NEVER ends
up with ``assignee=None``.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import profiles as profiles_mod
from hermes_cli.kanban_specify import (
    _call_aux, _extract_json_blob, _load_triage_task, _task_prompt_fields, _title_body,
)
from hermes_cli.kanban_specify import _profile_author as _specify_author

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are the Kanban decomposer for the Hermes Agent board.

A user dropped a rough idea into the Triage column. Your job is to break it
into a small graph of concrete child tasks and route each one to the best-
matching profile from the available roster.

You will be given:
  - The original task title and body
  - The list of available profiles (each with name + description)
  - The fallback "default_assignee" used when no profile fits

Output a single JSON object with this exact shape:

  {
    "fanout": true,
    "rationale": "<one sentence on why this decomposition>",
    "tasks": [
      {
        "title": "<concrete task title, imperative voice, <= 80 chars>",
        "body":  "<detailed spec for the worker on this child task>",
        "assignee": "<profile name from the roster, or null for default>",
        "parents": [<int>, ...]
      },
      ...
    ]
  }

Rules:
  - "parents" is a list of INDICES (0-based) into this same "tasks" list,
    expressing actual data dependencies. Tasks with no parents run in
    PARALLEL. Tasks with parents wait until every parent completes.
  - Prefer parallelism. If two tasks can be done independently, give
    them no parents so the dispatcher fans them out at once.
  - Use 2-6 tasks for normal work. Don't create 20 tiny tasks. Don't
    cram everything into 1 task.
  - Pick assignees from the roster by matching the task to the profile's
    DESCRIPTION (not just the name). When nothing matches well, use null
    and the system will route to the default_assignee.
  - Each child task body is what a fresh worker will read with no other
    context — be specific about goal, approach, and acceptance criteria.

When the task is genuinely a single unit of work (no useful decomposition),
return:

  {
    "fanout": false,
    "rationale": "<one sentence>",
    "title": "<tightened title>",
    "body":  "<concrete spec for a single worker>",
    "assignee": "<profile name from the roster, or null for default>"
  }

In that case the task stays as one work item, just with a tightened spec and
a concrete assignee. If no profile fits, use null and the system will route to
the default_assignee.

No preamble, no closing remarks, no code fences. Output only the JSON object.
"""


_USER_TEMPLATE = """Task id: {task_id}
Title: {title}
Body:
{body}

Available profiles (assignees you may pick from):
{roster}

Default assignee (used when no profile fits a task): {default_assignee}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class DecomposeOutcome:
    """Result of decomposing a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    fanout: bool = False
    child_ids: list[str] | None = None
    new_title: Optional[str] = None
    dedup_hits: dict[str, str] = field(default_factory=dict)  # child title -> existing card id


def _profile_author() -> str:
    """Mirror of ``hermes_cli.kanban._profile_author``."""
    return _specify_author("decomposer")


def _resolve_profile_from_cfg(cfg: dict, key: str, *, fallback: Optional[str] = None) -> str:
    """``kanban.<key>`` if it names an existing profile, else ``fallback``
    (the root task's own assignee) if that does, else the active default
    profile — so a task is never stranded for lack of an owner.
    ``orchestrator_profile`` owns the root after fan-out; ``default_assignee``
    catches children the decomposer can't route.

    The root's assignee sits before the active profile because the decomposer
    runs inside whatever profile hosts the dispatcher — an operator's
    credential-less incognito profile, say — and that profile must never
    silently become the owner of work the card was assigned away from (#114294).
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get(key) or "").strip()
    for candidate in (explicit, (fallback or "").strip()):
        if candidate:
            try:
                if profiles_mod.profile_exists(candidate):
                    return candidate
            except Exception:
                pass
    try:
        return profiles_mod.get_active_profile_name() or "default"
    except Exception:
        return "default"


def _build_roster() -> tuple[list[dict], set[str]]:
    """``(roster_for_prompt, valid_assignee_names)``; entries are
    ``{name, description, has_description}``."""
    try:
        all_profiles = profiles_mod.list_profiles()
    except Exception as exc:
        logger.warning("decompose: failed to list profiles: %s", exc)
        return [], set()
    roster = []
    for p in all_profiles:
        desc = (p.description or "").strip()
        roster.append({
            "name": p.name,
            "description": desc or f"(no description; profile named {p.name!r})",
            "has_description": bool(desc),
        })
    return roster, {p.name for p in all_profiles}


def _format_roster(roster: list[dict]) -> str:
    if not roster:
        return "  (no profiles installed — decomposer cannot route work)"
    return "\n".join(
        f"  - {entry['name']}{'' if entry['has_description'] else ' ⚠ undescribed'}: {entry['description']}"
        for entry in roster
    )


def _normalize_assignee_choice(assignee: object, *, default_assignee: str, valid_names: set[str]) -> str:
    """A valid assignee, else ``default_assignee`` — promoted work is never
    left unassigned."""
    if not isinstance(assignee, str) or not assignee.strip():
        return default_assignee
    chosen = assignee.strip()
    return chosen if chosen in valid_names else default_assignee


@dataclass
class _Routing:
    """Config-derived routing context for one decomposition."""

    orchestrator: str
    default_assignee: str
    auto_promote: bool
    roster: list[dict]
    valid_names: set[str]


def _load_routing(*, root_assignee: Optional[str] = None) -> _Routing:
    from hermes_cli.config import load_config_readonly
    try:
        cfg = load_config_readonly()
    except Exception:  # decompose_task promises ok=False, never a raise, on config trouble
        cfg = {}
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    roster, valid_names = _build_roster()
    return _Routing(
        orchestrator=_resolve_profile_from_cfg(cfg, "orchestrator_profile", fallback=root_assignee),
        default_assignee=_resolve_profile_from_cfg(cfg, "default_assignee", fallback=root_assignee),
        auto_promote=bool(kanban_cfg.get("auto_promote_children", True)),
        roster=roster,
        valid_names=valid_names,
    )


def _apply_single(task: kb.Task, parsed: dict, routing: _Routing, author: str) -> DecomposeOutcome:
    """``fanout=false``: single-task spec promotion (same effect as specify)."""
    title_val, body_val = _title_body(parsed)
    assignee_val = None
    if not task.assignee:
        assignee_val = _normalize_assignee_choice(
            parsed.get("assignee"), default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
    if title_val is None and body_val is None:
        return DecomposeOutcome(task.id, False, "decomposer returned fanout=false with no title/body")
    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn, task.id, title=title_val, body=body_val, assignee=assignee_val, author=author,
        )
    if not ok:
        return DecomposeOutcome(task.id, False, "task moved out of triage before promotion")
    return DecomposeOutcome(task.id, True, "single task (no fanout)", fanout=False, new_title=title_val)


def _title_key(title: str) -> str:
    """Discriminator for duplicate siblings: case-, space- and trailing-punctuation-insensitive."""
    return re.sub(r"\s+", " ", title.strip().lower())[:200].rstrip(" .!?:;-")


# Window for the board-wide dedup gate: a card created (or last completed)
# within this many seconds blocks a would-be duplicate child.
_DEDUP_WINDOW_SECONDS = 24 * 3600


def _existing_card_for_title(
    conn, title: str, *, tenant: Optional[str], exclude_id: str,
    window_seconds: int = _DEDUP_WINDOW_SECONDS,
) -> Optional[str]:
    """Return the id of an open/done card whose normalised title matches
    ``title`` and was created within ``window_seconds`` — the dedup gate
    that stops a fix re-fanning out as twin cards across dispatches.

    Scope is the same tenant (children inherit the triage task's tenant, so
    one dispatch never dedupes across tenants) and excludes the triage root
    itself. ``created_at`` is an INTEGER epoch-seconds column; completed
    cards count too, so a finished fix is not re-opened within 24h.
    """
    cutoff = int(time.time()) - window_seconds
    rows = conn.execute(
        """SELECT id, title FROM tasks
           WHERE id != ?
             AND (tenant IS ? OR tenant = ?)
             AND created_at >= ?
             AND status IN ('todo', 'ready', 'running', 'review', 'blocked', 'done', 'failed')
           ORDER BY created_at ASC""",
        (exclude_id, tenant, tenant or "", cutoff),
    ).fetchall()
    target_key = _title_key(title)
    for row_id, row_title in rows:
        if _title_key(row_title) == target_key:
            return row_id
    return None


def _clean_children(
    task_id: str, raw_tasks: list, routing: _Routing, *, conn,
    tenant: Optional[str],
) -> tuple[list[dict], str, dict[str, str]]:
    """Validate/normalise the LLM's ``tasks`` list; ``(children, "", {})`` or
    ``([], reason, {})``.
    Unknown assignees route to the default; never assignee=None.

    Two dedup layers:
      1. Duplicate siblings within the reply collapse onto their first
         occurrence (one fix must not fan out as twin cards).
      2. A child whose normalised title matches an existing open/done card
         created in the last 24h on the same tenant is refused — the
         existing card's id is returned instead of a sibling, so repeated
         dispatches of the same fix converge on one tracker.
    Parent indices are remapped to the surviving positions, so dropping an
    entry never re-points a live child at the wrong card.
    """
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            return [], f"tasks[{idx}] is not an object", {}
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return [], f"tasks[{idx}].title is missing or empty", {}

    # Pass 1: collapse duplicate siblings within the reply.
    canonical: dict[int, int] = {}
    seen: dict[str, int] = {}
    for idx, entry in enumerate(raw_tasks):
        key = _title_key(entry["title"])
        if key in seen:
            canonical[idx] = seen[key]
            logger.info(
                "decompose: task %s child %d %r duplicates child %d — dropping the twin",
                task_id, idx, entry["title"].strip()[:200], seen[key],
            )
        else:
            seen[key] = idx
            canonical[idx] = idx

    kept = [idx for idx in range(len(raw_tasks)) if canonical[idx] == idx]

    # Pass 2: refuse children that duplicate an existing board card <24h old.
    dedup_hits: dict[str, str] = {}
    survivors: list[int] = []
    for idx in kept:
        entry = raw_tasks[idx]
        existing = _existing_card_for_title(
            conn, entry["title"], tenant=tenant, exclude_id=task_id,
        )
        if existing:
            dedup_hits[entry["title"].strip()[:200]] = existing
            logger.info(
                "decompose: task %s child %d %r duplicates existing card %s — refusing the duplicate",
                task_id, idx, entry["title"].strip()[:200], existing,
            )
            continue
        survivors.append(idx)
    kept = survivors
    position = {raw: pos for pos, raw in enumerate(kept)}

    children: list[dict] = []
    for pos, idx in enumerate(kept):
        entry = raw_tasks[idx]
        title = entry["title"]
        body = entry.get("body")
        assignee = entry.get("assignee")
        chosen = _normalize_assignee_choice(
            assignee, default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
        if isinstance(assignee, str) and assignee.strip() and assignee.strip() not in routing.valid_names:
            logger.info(
                "decompose: task %s child %d picked unknown assignee %r — "
                "routing to default_assignee %r",
                task_id, idx, assignee, routing.default_assignee,
            )
        parents = entry.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        remapped: list[int] = []
        for raw_parent in parents:
            # Drop non-int and out-of-range indices, then re-point onto the
            # surviving position (a twin's index resolves to the twin's keeper).
            if not isinstance(raw_parent, int) or not 0 <= raw_parent < len(raw_tasks):
                continue
            target = position[canonical[raw_parent]]
            if target == pos or target in remapped:
                continue
            remapped.append(target)
        children.append({
            "title": title.strip()[:200],
            "body": body.strip() if isinstance(body, str) else "",
            "assignee": chosen,
            "parents": remapped,
        })
    return children, "", dedup_hits


def _apply_fanout(task_id: str, parsed: dict, routing: _Routing, author: str) -> DecomposeOutcome:
    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(task_id, False, "decomposer returned fanout=true with empty tasks list")
    with kbc.connect_closing() as conn:
        root = kb.get_task(conn, task_id)
        if root is None:
            return DecomposeOutcome(task_id, False, "task vanished before fanout")
        children, reason, dedup_hits = _clean_children(
            task_id, raw_tasks, routing, conn=conn, tenant=root.tenant,
        )
        if reason:
            return DecomposeOutcome(task_id, False, reason)
        try:
            child_ids = decompose_triage_task(
                conn,
                task_id,
                root_assignee=routing.orchestrator,
                children=children,
                author=author,
                auto_promote=routing.auto_promote,
            )
        except ValueError as exc:
            return DecomposeOutcome(task_id, False, f"DB rejected graph: {exc}")
        except Exception as exc:
            logger.exception("decompose: DB error on task %s", task_id)
            return DecomposeOutcome(task_id, False, f"DB error: {type(exc).__name__}")
        if child_ids is None:
            return DecomposeOutcome(task_id, False, "task already decomposed or moved out of triage")
        return DecomposeOutcome(
            task_id, True,
            f"decomposed into {len(child_ids)} children"
            + (f"; {len(dedup_hits)} duplicates refused" if dedup_hits else ""),
            fanout=True, child_ids=child_ids, dedup_hits=dedup_hits,
        )


def decompose_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> DecomposeOutcome:
    """Decompose a triage task into a graph of child tasks. Expected failures
    (not in triage, no aux client, API error, malformed/empty reply) surface
    as ``ok=False``."""
    task, reason = _load_triage_task(task_id)
    if task is None:
        return DecomposeOutcome(task_id, False, reason)

    routing = _load_routing(root_assignee=task.assignee)
    raw, reason = _call_aux(
        "decompose", task_id, aux_task="kanban_decomposer", system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(
            **_task_prompt_fields(task),
            roster=_format_roster(routing.roster),
            default_assignee=routing.default_assignee,
        ),
        max_tokens=4000, timeout=timeout or 180, log=logger,
    )
    if raw is None:
        return DecomposeOutcome(task_id, False, reason)

    parsed = _extract_json_blob(raw, _FENCE_RE)
    if parsed is None:
        return DecomposeOutcome(task_id, False, "LLM returned malformed JSON")

    audit_author = author or _profile_author()
    if not parsed.get("fanout"):
        return _apply_single(task, parsed, routing, audit_author)
    return _apply_fanout(task_id, parsed, routing, audit_author)


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column."""
    with kbc.connect_closing() as conn:
        rows = kb.list_tasks(conn, status="triage", tenant=tenant, limit=1000)
    return [row.id for row in rows]
