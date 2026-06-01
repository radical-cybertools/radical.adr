"""Decision and Action — the output of a policy cycle."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

from radical.adl.goals import Goal


class ActionKind(str, Enum):
    SPAWN_TASK   = "spawn_task"
    CANCEL_TASK  = "cancel_task"
    SPAWN_BLOCK  = "spawn_block"
    UPDATE_PARAM = "update_param"


class Action(BaseModel):
    """A single, serializable workflow action instruction.

    ``task_name`` is a string key into ``Operator.task_registry`` or
    ``Operator.workflow_registry``.  The callable lookup happens at execution
    time inside the operator — never here.
    """

    kind:        ActionKind
    target_uid:  str | None       = None
    task_name:   str | None       = None
    task_kwargs: dict[str, Any]   = {}
    depends_on:  list[str]        = []
    param_key:   str | None       = None
    param_value: Any              = None


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
