"""Integration tests for the adaptation boundary at the UPDATE_PARAM seam:

  - out-of-bounds writes are rejected (not applied, not clamped) and surface as evidence
  - writes to a non-tunable key are rejected when a boundary is declared
  - in-bounds writes are applied (with int coercion)
  - a runtime goal that shadows a static @goals goal is rejected
"""

from __future__ import annotations

import pytest

from radical.adr import Decision, Goal, Operator, Policy, Tunable, decide, goals, observe


class _BoundOp(Operator):
    learning_rate = Tunable(3e-4, 1e-5, 1e-3)
    seen_rejection: bool = False

    @observe
    def o(self, snap) -> dict:
        if snap.rejections:
            self.seen_rejection = True
        return {"lr": self.learning_rate, "rejections": snap.rejections}


class _ProposeOnce(Policy):
    def __init__(self, op, key, value) -> None:
        super().__init__()
        self._acts = op.get_actions()
        self._key, self._value, self._done = key, value, False

    @decide
    async def run(self, obs) -> Decision:
        if not self._done:
            self._done = True
            return Decision(actions=[self._acts.update_param(self._key, self._value)])
        return Decision()


async def test_out_of_bounds_rejected_not_applied(engine):
    op = _BoundOp(engine, max_cycles=4)
    op.policy = _ProposeOnce(op, "learning_rate", 10.0)   # far above high=1e-3
    async for _ in op.run():
        pass
    assert op.learning_rate == pytest.approx(3e-4)         # unchanged (not clamped to 1e-3)
    assert op.seen_rejection is True                       # surfaced as evidence


async def test_non_tunable_key_rejected_when_boundary_declared(engine):
    op = _BoundOp(engine, max_cycles=4)
    op.policy = _ProposeOnce(op, "frozen_knob", 1.0)
    async for _ in op.run():
        pass
    assert "frozen_knob" not in op.state.objectives
    assert op.seen_rejection is True


class _IntOp(Operator):
    steps = Tunable(1000, 100, 5000, int=True)

    @observe
    def o(self, snap) -> dict:
        return {}


async def test_in_bounds_write_applied_with_int_coercion(engine):
    op = _IntOp(engine, max_cycles=3)
    op.policy = _ProposeOnce(op, "steps", 2000.0)          # integer-valued float, in bounds
    async for _ in op.run():
        pass
    assert op.steps == 2000 and isinstance(op.steps, int)


# ── goal shadow guard ─────────────────────────────────────────────────────────

class _ShadowOp(Operator):
    metric: float = 0.0

    @observe
    def o(self, snap) -> dict:
        self.metric = self.metric + 0.1
        return {"metric": self.metric, "other_metric": 0.0}

    @goals
    def g(self):
        return Goal("solved", metric="metric", threshold=1.0, direction="maximize")


class _AddGoalOnce(Policy):
    def __init__(self, goal) -> None:
        super().__init__()
        self._goal, self._done = goal, False

    @decide
    async def run(self, obs) -> Decision:
        if not self._done:
            self._done = True
            return Decision(goals=[self._goal])
        return Decision()


async def test_runtime_goal_shadowing_metric_rejected(engine):
    op = _ShadowOp(engine, max_cycles=3)
    op.policy = _AddGoalOnce(Goal("aux", metric="metric", threshold=0.5, direction="maximize"))
    async for _ in op.run():
        pass
    assert op.runtime_goals == []                          # shadowed the static metric → dropped


async def test_runtime_goal_shadowing_name_rejected(engine):
    op = _ShadowOp(engine, max_cycles=3)
    op.policy = _AddGoalOnce(Goal("solved", metric="other", threshold=0.5, direction="maximize"))
    async for _ in op.run():
        pass
    assert op.runtime_goals == []                          # name collides with static goal → dropped


async def test_non_shadowing_runtime_goal_accepted(engine):
    op = _ShadowOp(engine, max_cycles=3)
    op.policy = _AddGoalOnce(Goal("aux", metric="other_metric", threshold=0.5, direction="maximize"))
    async for _ in op.run():
        pass
    assert any(g.name == "aux" for g in op.runtime_goals)
