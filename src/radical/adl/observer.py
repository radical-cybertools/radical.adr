"""Observer protocols and TelemetryObserver — optional observation hooks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

if TYPE_CHECKING:
    from radical.adl.decision import Decision
    from radical.adl.state import Snapshot

log = logging.getLogger(__name__)


@runtime_checkable
class ObserverBase(Protocol):
    """Optional lifecycle hooks for an operator run.

    All methods are synchronous — keep them fast. For async work, schedule
    a task inside the callback rather than awaiting directly.
    """

    def on_start(self, operator_id: str, metadata: dict) -> None: ...
    def on_cycle(self, snapshot: "Snapshot", decision: "Decision") -> None: ...
    def on_stop(self, final: "Snapshot | None", reason: str) -> None: ...


class TelemetryObserver:
    """Forwards Rhapsody TelemetrySubscriber events into an operator's inbox.

    This is a second, complementary observation channel alongside ``_watch_future``.
    It provides richer event types (lifecycle signals, resource metrics, custom domain
    events) that the future watcher cannot carry.

    The future watcher remains mandatory because ``TaskCompleted`` does not carry task
    return values — only ``await future`` delivers the result.

    Usage::

        telemetry = await flow.start_telemetry(...)
        observer  = TelemetryObserver(telemetry, operator)
        observer.start()
    """

    def __init__(self, telemetry: object, operator: object) -> None:
        # Import lazily so the core package does not hard-depend on rhapsody.
        from radical.adl.message import InboxMessage, MessageKind
        self._InboxMessage  = InboxMessage
        self._MessageKind   = MessageKind
        self._telemetry     = telemetry
        self._operator      = operator

    def start(self) -> None:
        """Register the async event handler with the TelemetrySubscriber."""
        self._telemetry.subscribe(self._on_event)

    async def _on_event(self, event: object) -> None:
        try:
            msg = self._InboxMessage(
                kind      = self._MessageKind.TASK_EVENT,
                sender_id = f"telemetry:{getattr(event, 'backend', 'unknown')}",
                payload   = event,
            )
            await self._operator.inbox.put(msg)
        except Exception:
            log.exception("TelemetryObserver: failed to forward event %r", event)
