"""Observer protocol — optional lifecycle hooks for an operator run."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

if TYPE_CHECKING:
    from radical.adr.decision import Decision
    from radical.adr.state import Snapshot


@runtime_checkable
class ObserverBase(Protocol):
    """Optional lifecycle hooks for an operator run.

    All methods are synchronous — keep them fast. For async work, schedule
    a task inside the callback rather than awaiting directly.
    """

    def on_start(self, operator_id: str, metadata: dict) -> None: ...
    def on_cycle(
        self, snapshot: "Snapshot", obs: dict, decision: "Decision"
    ) -> None: ...
    def on_stop(self, final: "Snapshot | None", reason: str) -> None: ...
