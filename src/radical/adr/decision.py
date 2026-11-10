"""Decision and Action — the output of a policy cycle."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from radical.adr.goals import Goal


class ActionKind(str, Enum):
    SPAWN_TASK       = "spawn_task"
    CANCEL_TASK      = "cancel_task"
    RETRY_TASK       = "retry_task"
    SPAWN_BLOCK      = "spawn_block"
    SPAWN_OPERATOR   = "spawn_operator"
    RESTART_OPERATOR = "restart_operator"
    CANCEL_OPERATOR  = "cancel_operator"
    UPDATE_PARAM     = "update_param"


class Action(BaseModel):
    """A single, serializable workflow action instruction.

    With ``Operator(..., track_action_ids=True)``, ``action_id`` is assigned to
    each emitted occurrence before observer notification. Re-emitting an action
    gives it a fresh ID without mutating the original. Unset IDs default to
    ``None`` and are omitted from serialization.

    ``task_name`` is a string key into ``Operator.task_registry`` or
    ``Operator.workflow_registry``.  The callable lookup happens at execution
    time inside the operator — never here.

    ``depends_on`` is interpreted by ``kind``: for ``SPAWN_TASK``/``SPAWN_BLOCK``
    it lists upstream task **uids**; for ``SPAWN_OPERATOR``/``RESTART_OPERATOR`` it
    lists upstream child operator **names**.  Either way the resolved Futures gate
    the launch.

    For ``SPAWN_OPERATOR``: ``task_name`` is the child name, ``task_kwargs`` is
    the child's construction config, and ``operator_cls`` / ``policy`` carry live
    Python references (excluded from serialization).

    For ``RESTART_OPERATOR`` / ``CANCEL_OPERATOR``: ``task_name`` is the child name;
    ``operator_cls`` / ``policy`` are unused (the child is already registered).
    """

    action_id:   str | None       = None
    kind:        ActionKind
    target_uid:  str | None       = None
    task_name:   str | None       = None
    task_kwargs: dict[str, Any]   = {}
    depends_on:  list[str]        = []
    param_key:   str | None       = None
    param_value: Any              = None
    operator_cls: Any = Field(default=None, exclude=True)   # live ref; not serialized
    policy:       Any = Field(default=None, exclude=True)   # Policy instance OR class/factory

    model_config = {"arbitrary_types_allowed": True}

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler):
        data = handler(self)
        if self.action_id is None:
            data.pop("action_id", None)
        return data


class Decision(BaseModel):
    """The full output of one policy cycle."""

    actions:      list[Action]          = []
    directives:   dict[str, Any]        = {}
    stop:         bool                  = False
    delegate_to:  list[str]             = []
    notify:       list[tuple[str, Any]] = []
    goals:        list[Goal]            = []   # propose / upsert runtime goals
    remove_goals: list[str]             = []   # revoke runtime goals by name

    model_config = {"arbitrary_types_allowed": True}
