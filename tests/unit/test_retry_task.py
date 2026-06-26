"""Unit tests for acts.retry_task — action builder + pre-flight validation (no engine)."""

import pytest

from radical.adr import Operator, act, observe
from radical.adr.decision import Action, ActionKind
from radical.adr.errors import ActionError


class _Op(Operator):
    @act
    async def evaluate(self, x: float, seed: int = 0):
        return {"x": x}

    @observe
    def o(self, s):
        return {}


def test_retry_task_builds_action():
    op = _Op(engine=None)
    a = op.get_actions().retry_task("uid-1")
    assert a.kind == ActionKind.RETRY_TASK and a.target_uid == "uid-1"


def test_retry_task_rejects_unknown_uid():
    op = _Op(engine=None)
    with pytest.raises(ActionError, match="RETRY_TASK"):
        op._validate_actions([Action(kind=ActionKind.RETRY_TASK, target_uid="ghost")])


def test_retry_task_accepts_recorded_uid():
    op = _Op(engine=None)
    op.state.task_context["good"] = {"name": "evaluate", "kwargs": {"x": 1.0}}
    op._validate_actions([Action(kind=ActionKind.RETRY_TASK, target_uid="good")])  # no raise
