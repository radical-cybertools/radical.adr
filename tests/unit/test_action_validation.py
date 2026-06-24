"""Unit tests for Operator._validate_actions — no engine.

Covers Item 5: pre-flight validation collects all problems into one ActionError,
allows arbitrary update_param keys, and warns (not raises) on unresolvable
depends_on uids.
"""

import warnings

import pytest

from radical.adr import Action, ActionKind, Operator, act, observe
from radical.adr.errors import ActionError


class _ValOp(Operator):
    @act
    async def compute(self, x: float, seed: int = 0) -> dict:
        return {"val": x}

    @act
    async def chained(self, dep_future, y: float) -> dict:
        return {"y": y}

    @observe
    def extract(self, snapshot) -> dict:
        return {"cycle": snapshot.cycle}


def _op() -> _ValOp:
    # No live engine needed — _validate_actions only reads the act registry and
    # state.task_futures.
    return _ValOp(engine=None)


def _spawn(name, *, depends_on=None, **kwargs) -> Action:
    return Action(kind=ActionKind.SPAWN_TASK, task_name=name,
                  task_kwargs=kwargs, depends_on=depends_on or [])


def test_unknown_task_name_raises():
    op = _op()
    with pytest.raises(ActionError, match="unknown task 'nope'"):
        op._validate_actions([_spawn("nope")])


def test_unknown_kwarg_raises():
    op = _op()
    with pytest.raises(ActionError, match="unknown kwarg 'bogus'"):
        op._validate_actions([_spawn("compute", x=1.0, bogus=2)])


def test_missing_required_kwarg_raises():
    op = _op()
    with pytest.raises(ActionError, match="missing required parameter 'x'"):
        op._validate_actions([_spawn("compute", seed=3)])


def test_optional_kwarg_not_required():
    op = _op()
    op._validate_actions([_spawn("compute", x=1.0)])   # seed has a default → OK


def test_multiple_problems_collected():
    op = _op()
    with pytest.raises(ActionError) as ei:
        op._validate_actions([_spawn("nope"), _spawn("compute", bad=1)])
    msg = str(ei.value)
    assert "problem(s)" in msg
    # All distinct problems are reported in a single error, not just the first.
    assert "unknown task 'nope'" in msg
    assert "unknown kwarg 'bad'" in msg
    assert "missing required parameter 'x'" in msg


def test_cancel_unknown_uid_raises():
    op = _op()
    with pytest.raises(ActionError, match="unknown uid 'ghost'"):
        op._validate_actions([Action(kind=ActionKind.CANCEL_TASK, target_uid="ghost")])


def test_cancel_missing_target_uid_raises():
    op = _op()
    with pytest.raises(ActionError, match="target_uid is required"):
        op._validate_actions([Action(kind=ActionKind.CANCEL_TASK)])


def test_cancel_known_uid_ok():
    op = _op()
    op.state.task_futures["uid-1"] = object()
    op._validate_actions([Action(kind=ActionKind.CANCEL_TASK, target_uid="uid-1")])


def test_update_param_missing_key_raises():
    op = _op()
    with pytest.raises(ActionError, match="param_key is required"):
        op._validate_actions([Action(kind=ActionKind.UPDATE_PARAM, param_value=1)])


def test_update_param_arbitrary_key_allowed():
    op = _op()
    # cpu_percent is not an annotated attribute — deliberately allowed.
    op._validate_actions([
        Action(kind=ActionKind.UPDATE_PARAM, param_key="cpu_percent", param_value=80.0)
    ])


def test_depends_on_missing_uid_warns_not_raises():
    op = _op()
    with pytest.warns(UserWarning, match="depends_on 'missing'"):
        op._validate_actions([_spawn("chained", depends_on=["missing"], y=1.0)])


def test_depends_on_present_uid_no_warning():
    op = _op()
    op.state.task_futures["a"] = object()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # dep_future positional filled by the resolvable dep → no missing-param error either
        op._validate_actions([_spawn("chained", depends_on=["a"], y=1.0)])


def test_all_valid_actions_no_raise():
    op = _op()
    op.state.task_futures["uid-1"] = object()
    op._validate_actions([
        _spawn("compute", x=1.0),
        Action(kind=ActionKind.UPDATE_PARAM, param_key="anything", param_value=5),
        Action(kind=ActionKind.CANCEL_TASK, target_uid="uid-1"),
    ])
