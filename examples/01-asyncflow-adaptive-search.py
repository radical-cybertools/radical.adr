"""01-asyncflow-adaptive-search.py

Adaptive search for the minimum of f(x) = (x - TARGET)² + noise.

ADR concepts demonstrated
  - Operator: @act, @observe, @goals
  - Policy:   @decide — the single explicit entry point
  - op.policy = SearchPolicy(op) — one clean assign

Usage:
    python 01-asyncflow-adaptive-search.py
"""

import asyncio
import logging
import random
from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from radical.adr import Decision, Operator, Policy, decide, goals, observe, act
from radical.adr import enable_logging
from radical.adr.goals import Goal

TARGET     = 3.14159
MIN_RADIUS = 0.005

async def evaluate_task(x: float, seed: int = 0) -> dict:
    rng = random.Random(seed)
    return {"x": x, "value": (x - TARGET) ** 2 + rng.gauss(0, 0.03)}


class SearchOperator(Operator):

    center:  float = 0.0
    radius:  float = 5.0
    history: list  = []

    @goals
    def convergence_criteria(self):
        return Goal(name="converged", metric="radius",
                              threshold=MIN_RADIUS, direction="minimize")

    @act
    async def evaluate(self, x: float, seed: int = 0) -> dict:
        return self.engine.function_task(evaluate_task)(x=x, seed=seed)

    @observe
    def extract(self, snapshot) -> dict:
        new_results = [v for k, v in snapshot.observations.items()
                       if k.startswith("result.") and isinstance(v, dict) and "x" in v]
        history = list(snapshot.objectives.get("history", [])) + new_results
        best    = min(history, key=lambda r: r["value"]) if history else None
        return {"new_results": new_results, "history": history,
                "center": snapshot.objectives.get("center", 0.0),
                "radius": snapshot.objectives.get("radius", 5.0),
                "best": best, "cycle": snapshot.cycle}


class SearchPolicy(Policy):

    def __init__(self, op: SearchOperator) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] >= 15:
            return Decision(stop=True)

        if obs["best"] is not None:
            new_c, new_r = obs["best"]["x"], obs["radius"] * 0.5
            return Decision(actions=[
                self._act.update_param("history", obs["history"]),
                self._act.update_param("center",  new_c),
                self._act.update_param("radius",  new_r),
                *self._spawn(new_c, new_r, obs["cycle"]),
            ])

        return Decision(actions=self._spawn(obs["center"], obs["radius"], obs["cycle"]))

    def _spawn(self, c, r, cyc):
        return [self._act.spawn_task("evaluate", x=c + r*(2*i/3-1), seed=cyc*4+i)
                for i in range(4)]


async def main():
    backend = await LocalExecutionBackend(ProcessPoolExecutor())
    engine  = await WorkflowEngine.create(backend)

    op = SearchOperator(engine)
    op.policy = SearchPolicy(op)

    async for snapshot in op.run():
        print(f"cycle {snapshot.cycle:2d}  center={op.center:+.5f}  radius={op.radius:.5f}")

    best = min(op.history, key=lambda r: r["value"])
    print(f"\nbest x={best['x']:.6f}  error={abs(best['x'] - TARGET):.6f}  cycles={snapshot.cycle + 1}")
    await engine.shutdown()


if __name__ == "__main__":
    enable_logging(logging.INFO)
    asyncio.run(main())
