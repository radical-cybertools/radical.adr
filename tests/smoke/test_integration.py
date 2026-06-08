"""Smoke test: full adaptive search campaign end-to-end.

Runs a simplified version of example 01 and asserts that:
- The operator completes without error
- The convergence goal is reached within the cycle budget
- The final radius is below the goal threshold
"""

from __future__ import annotations

import random

import pytest

from radical.adl import Decision, Operator, Policy, decide, goals, observe, act
from radical.adl.goals import Goal

TARGET     = 3.14159
MIN_RADIUS = 0.005
MAX_CYCLES = 20


class SearchOperator(Operator):
    center:  float = 0.0
    radius:  float = 5.0
    history: list  = []

    @goals
    def convergence_criteria(self):
        return Goal(
            name="converged", metric="radius",
            threshold=MIN_RADIUS, direction="minimize"
        )

    @act
    async def evaluate(self, x: float, seed: int = 0) -> dict:
        rng = random.Random(seed)
        return {"x": x, "value": (x - TARGET) ** 2 + rng.gauss(0, 0.01)}

    @observe
    def extract(self, snap) -> dict:
        new_results = [
            v for k, v in snap.observations.items()
            if k.startswith("result.") and isinstance(v, dict) and "x" in v
        ]
        history = list(snap.objectives.get("history", [])) + new_results
        best    = min(history, key=lambda r: r["value"]) if history else None
        return {
            "new_results": new_results,
            "history":     history,
            "center":      snap.objectives.get("center", 0.0),
            "radius":      snap.objectives.get("radius", 5.0),
            "best":        best,
            "cycle":       snap.cycle,
        }


class SearchPolicy(Policy):
    def __init__(self, op: SearchOperator) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] >= MAX_CYCLES:
            return Decision(stop=True)

        if obs["best"] is not None:
            new_c = obs["best"]["x"]
            new_r = obs["radius"] * 0.6
            return Decision(actions=[
                self._act.update_param("history", obs["history"]),
                self._act.update_param("center",  new_c),
                self._act.update_param("radius",  new_r),
                *[self._act.evaluate(x=new_c + new_r * (2*i/3 - 1), seed=obs["cycle"]*4+i)
                  for i in range(4)],
            ])

        return Decision(actions=[
            self._act.evaluate(x=obs["center"] + obs["radius"] * (2*i/3 - 1),
                             seed=obs["cycle"]*4+i)
            for i in range(4)
        ])


async def test_adaptive_search_converges(engine):
    op = SearchOperator(engine)
    op.policy = SearchPolicy(op)

    snap = await op.run_to_completion()

    assert snap is not None, "operator returned no snapshot"
    assert snap.cycle <= MAX_CYCLES, f"exceeded cycle budget: {snap.cycle}"
    assert op.state.objectives.get("radius", 5.0) < MIN_RADIUS, (
        f"goal not reached: radius={op.state.objectives.get('radius')}"
    )
