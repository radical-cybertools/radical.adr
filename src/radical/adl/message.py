"""Inbox message types for inter-operator communication."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class MessageKind(str, Enum):
    TASK_EVENT = "task_event"
    SNAPSHOT   = "snapshot"
    DIRECTIVE  = "directive"
    STOP       = "stop"
    CUSTOM     = "custom"


class InboxMessage(BaseModel):
    """A single message delivered to an operator's inbox queue."""

    kind:      MessageKind
    sender_id: str
    payload:   Any
    timestamp: float = Field(default_factory=time.time)

    model_config = {"arbitrary_types_allowed": True}


class MessageBus:
    """Optional flat registry for sibling-to-sibling operator messaging.

    Not a global singleton — instantiate and wire where needed.
    Parent-child communication bypasses this entirely.
    """

    def __init__(self) -> None:
        self._queues: dict[str, Any] = {}  # operator_id → asyncio.Queue

    def register(self, operator: Any) -> None:
        self._queues[operator.operator_id] = operator.inbox

    def unregister(self, operator_id: str) -> None:
        self._queues.pop(operator_id, None)

    async def send(self, to_id: str, msg: InboxMessage) -> None:
        queue = self._queues.get(to_id)
        if queue is None:
            raise KeyError(f"MessageBus: unknown operator '{to_id}'")
        await queue.put(msg)
