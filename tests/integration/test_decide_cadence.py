"""Integration test: decide_when gates policy consultation without breaking the
loop — a decide_when="idle" operator still self-stops on its goal.

(The gate's inflight/new_result semantics are unit-tested in
tests/unit/test_evidence_helpers.py; here we prove it integrates cleanly.)
"""

from __future__ import annotations

from radical.adr import Decision, Goal, Operator, Policy, decide, goals, observe


class _Idle(Policy):
    @decide
    async def run(self, obs) -> Decision:
        return Decision()


class _IdleGoalOp(Operator):
    decide_when = "idle"
    count: int = 0

    @observe
    def o(self, snap) -> dict:
        self.count = self.count + 1
        return {"count": self.count}

    @goals
    def g(self):
        return Goal("done", metric="count", threshold=3, direction="maximize", inclusive=True)


async def test_decide_when_idle_still_stops_on_goal(engine):
    # No tasks are dispatched, so inflight is always 0 → the policy is consulted every
    # cycle exactly as without a gate, and the goal stops the run at count == 3.
    op = _IdleGoalOp(engine, max_cycles=1000)
    op.policy = _Idle()
    async for _ in op.run():
        pass
    assert op.count == 3
