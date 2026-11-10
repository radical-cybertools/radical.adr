"""Integration test: Operator(max_runtime=...) stops a goalless run by wall-clock.

An operator with no goals and a never-stopping policy must still terminate via the
wall-clock cap, well before the (large) max_cycles bound.
"""

from __future__ import annotations

from radical.adr import Decision, Operator, Policy, decide, observe


class _IdleOp(Operator):
    @observe
    def o(self, snap) -> dict:
        return {"cycle": snap.cycle}


class _IdlePolicy(Policy):
    @decide
    async def run(self, obs) -> Decision:
        return Decision()        # never stops on its own; no goals


async def test_max_runtime_stops_goalless_run(engine):
    op = _IdleOp(engine, max_cycles=100_000, max_runtime=0.3)
    op.policy = _IdlePolicy()
    final = None
    async for snap in op.run():
        final = snap
    # stopped by wall-clock, far below max_cycles
    assert final is not None and final.cycle < 1000
