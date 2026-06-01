"""RuntimeState and Snapshot — the operator's live and frozen state."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

from pydantic import BaseModel

_HISTORY_MAXLEN = 64


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

    model_config = {"arbitrary_types_allowed": True}

    @property
    def task_summary(self) -> dict[str, str]:
        """uid → status for all tracked tasks."""
        return {k: v for k, v in self.runtime.items() if not k.startswith("cpu_")}


class RuntimeState:
    """Live, mutable state owned by one operator.

    Five semantic namespaces isolate concerns:

    observations  — what was learned from task results this cycle (cleared per cycle)
    runtime       — ephemeral execution state: UIDs, statuses, resource metrics
    objectives    — stable goals set externally; persist across cycles
    artifacts     — references to produced outputs (paths, identifiers)
    directives    — incoming instructions from parent/user; consumed each cycle

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

        self.task_futures: dict[str, asyncio.Future]  = {}
        self.history: deque[Snapshot]                 = deque(maxlen=_HISTORY_MAXLEN)

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
        )
        self.history.append(snapshot)
        return snapshot

    def clear_cycle(self) -> None:
        """Clear per-cycle namespaces after a snapshot has been taken."""
        self.observations.clear()
        self.directives.clear()
        # runtime is NOT cleared — task statuses persist until explicitly removed
