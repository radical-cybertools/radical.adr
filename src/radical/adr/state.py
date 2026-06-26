"""RuntimeState and Snapshot — the operator's live and frozen state."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from typing import Any

from pydantic import BaseModel

_HISTORY_MAXLEN  = 64
_RESOLVED_MAXLEN = 1024   # bounded cache of terminal task results (for depends_on)


class Snapshot(BaseModel, frozen=True):
    """Frozen, serializable mirror of RuntimeState.

    Safe to pass across operator boundaries and into LLM prompts.
    No live asyncio.Future references.
    """

    operator_id:  str
    cycle:        int
    timestamp:    float
    source:       str             # "self" | "child:<id>" | "parent:<id>"

    observations: dict[str, Any]
    runtime:      dict[str, Any]  # uid → status string (no live futures)
    objectives:   dict[str, Any]
    artifacts:    dict[str, Any]
    directives:   dict[str, Any]
    task_context: dict[str, Any] = {}  # uid → {name: str, kwargs: dict}

    model_config = {"arbitrary_types_allowed": True}

    @property
    def task_summary(self) -> dict[str, str]:
        """uid → status for all tracked tasks."""
        return {k: v for k, v in self.runtime.items() if not k.startswith("cpu_")}


class RuntimeState:
    """Live, mutable state owned by one operator.

    Six semantic namespaces isolate concerns:

    observations  — what was learned from task results this cycle (cleared per cycle)
    runtime       — ephemeral execution state: UIDs, statuses
    objectives    — stable goals set externally; persist across cycles
    artifacts     — references to produced outputs (paths, identifiers)
    directives    — incoming instructions from parent/user; consumed each cycle
    task_context  — uid → {name, kwargs} for every spawned task; persists

    task_futures  — live asyncio.Future references (not serializable)
    history       — bounded ring of previous Snapshots
    """

    def __init__(self, operator_id: str) -> None:
        self.operator_id = operator_id
        self.cycle       = 0

        self.observations: dict[str, Any]             = {}
        self.runtime:      dict[str, Any]             = {}
        self.objectives:   dict[str, Any]             = {}
        self.artifacts:    dict[str, Any]             = {}
        self.directives:   dict[str, Any]             = {}
        self.task_context: dict[str, Any]             = {}

        self.task_futures: dict[str, asyncio.Future]  = {}
        self.history: deque[Snapshot]                 = deque(maxlen=_HISTORY_MAXLEN)

        # Terminal task results, kept after the live Future is pruned so that a
        # later depends_on can still resolve. Bounded (FIFO) — see remember_resolved.
        self.resolved: "OrderedDict[str, tuple[str, Any]]" = OrderedDict()

    def remember_resolved(self, uid: str, status: str, value: Any) -> None:
        """Cache a terminal task's result/error for later depends_on resolution.

        ``status`` is ``DONE``/``FAILED``/``CANCELED``; ``value`` is the result,
        the exception, or ``None``. Oldest entries are evicted past the cap — a
        depends_on older than the window is implausible (history is 64 cycles).
        """
        self.resolved[uid] = (status, value)
        self.resolved.move_to_end(uid)
        while len(self.resolved) > _RESOLVED_MAXLEN:
            self.resolved.popitem(last=False)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self, source: str = "self") -> Snapshot:
        """Produce a frozen Snapshot from current live state."""
        runtime_serializable = {
            uid: status
            for uid, status in self.runtime.items()
            if not isinstance(status, asyncio.Future)
        }
        snapshot = Snapshot(
            operator_id  = self.operator_id,
            cycle        = self.cycle,
            timestamp    = time.time(),
            source       = source,
            observations = dict(self.observations),
            runtime      = runtime_serializable,
            objectives   = dict(self.objectives),
            artifacts    = dict(self.artifacts),
            directives   = dict(self.directives),
            task_context = dict(self.task_context),
        )
        self.history.append(snapshot)
        return snapshot

    def clear_cycle(self) -> None:
        """Clear per-cycle namespaces after a snapshot has been taken."""
        self.observations.clear()
        self.directives.clear()
        # runtime is NOT cleared — task statuses persist until explicitly removed
