"""Integration test: Operator(budget=[Budget(...)]) is an independent stop.

A resource/deadline budget must terminate the run when the watched metric crosses
its ceiling, even when an unsatisfiable runtime goal is present — the exact failure
that motivated Budget: a budget expressed as a Goal is AND-combined with runtime
goals and can be blocked forever. Budget lives outside goal composition.
"""

from __future__ import annotations

from radical.adr import Budget, Decision, Goal, Operator, Policy, decide, goals, observe


class _CountingOp(Operator):
    count: int = 0

    @observe
    def o(self, snap) -> dict:
        self.count = self.count + 1
        return {"count": self.count, "down": 100 - self.count}


class _AddsImpossibleGoal(Policy):
    """Adds one unsatisfiable runtime goal, then idles — never stops on its own."""

    def __init__(self) -> None:
        super().__init__()
        self._added = False

    @decide
    async def run(self, obs) -> Decision:
        if not self._added:
            self._added = True
            return Decision(goals=[
                Goal("impossible", metric="count", threshold=1e9, direction="maximize")
            ])
        return Decision()


async def test_budget_stops_despite_unsatisfiable_goal(engine):
    op = _CountingOp(engine, max_cycles=100_000, budget=[Budget("count", 5, "max")])
    op.policy = _AddsImpossibleGoal()
    final = None
    async for snap in op.run():
        final = snap
    # stopped by the budget, not by max_cycles or the (never-met) goal
    assert op.count == 5
    assert final is not None and final.cycle < 100
    # the impossible goal really was in effect → we genuinely tested independence
    assert any(g.name == "impossible" for g in op.runtime_goals)


async def test_budget_direction_min(engine):
    # "down" starts at 99 and decreases; stop when it reaches the floor.
    op = _CountingOp(engine, max_cycles=100_000, budget=[Budget("down", 96, "min")])
    op.policy = _AddsImpossibleGoal()
    async for _ in op.run():
        pass
    # down = 100 - count; 100 - 4 = 96 <= 96 → stop at count == 4
    assert op.count == 4


class _Idle(Policy):
    @decide
    async def run(self, obs) -> Decision:
        return Decision()


class _GoalOp(Operator):
    count: int = 0

    @observe
    def o(self, snap) -> dict:
        self.count = self.count + 1
        return {"count": self.count}

    @goals
    def g(self):
        return Goal("done", metric="count", threshold=3, direction="maximize", inclusive=True)


async def test_budget_does_not_block_normal_goal_stop(engine):
    # Budget is generous and never reached; the satisfiable goal stops the run first
    # (count == 3), proving Budget does not interfere with ordinary convergence.
    op = _GoalOp(engine, max_cycles=100_000, budget=[Budget("count", 1000, "max")])
    op.policy = _Idle()
    async for _ in op.run():
        pass
    assert op.count == 3
