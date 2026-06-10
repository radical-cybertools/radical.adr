"""06-multi-operator.py

N operators running concurrently via asyncio.gather, each with its own policy.

ADR concepts demonstrated
  - N operators via asyncio.gather — no DAG, no coordination
  - Each operator has its own state and @decide policy
  - Operators are isolated: no shared mutable state

Usage:
    python 06-multi-operator.py
"""

from __future__ import annotations

import asyncio
import random
from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from radical.adr import Decision, Operator, Policy, decide, goals, observe, act
from radical.adr.goals import Goal

TARGET     = 3.14159
MIN_RADIUS = 0.005


async def evaluate_task(x: float, seed: int = 0) -> dict:
    import random
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
    async def evaluate(self, x: float, seed: int = 0):
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


async def run_one(engine, start_center: float, op_id: str) -> dict:
    op        = SearchOperator(engine, operator_id=op_id)
    op.center = start_center
    op.policy = SearchPolicy(op)
    await op.run_to_completion()
    best = min(op.history, key=lambda r: r["value"]) if op.history else None
    return {"id": op_id, "start": start_center,
            "best_x": best["x"] if best else None,
            "error":  abs(best["x"] - TARGET) if best else None,
            "cycles": op.state.cycle}


async def main():
    backend = await LocalExecutionBackend(ProcessPoolExecutor(max_workers=8))
    engine  = await WorkflowEngine.create(backend)

    starts  = [-5.0, 0.0, 5.0, 10.0]
    results = await asyncio.gather(*[
        run_one(engine, c, f"search-{i}") for i, c in enumerate(starts)
    ])

    print(f"{'Operator':<12} {'Start':>7} {'Best x':>10} {'Error':>10} {'Cycles':>7}")
    print("-" * 50)
    for r in results:
        if r["best_x"] is not None:
            print(f"{r['id']:<12} {r['start']:>+7.1f} {r['best_x']:>+10.6f} "
                  f"{r['error']:>10.6f} {r['cycles']:>7}")

    best = min((r for r in results if r["error"] is not None), key=lambda r: r["error"])
    print(f"\nBest: {best['id']}  x={best['best_x']:.6f}  error={best['error']:.6f}")
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
