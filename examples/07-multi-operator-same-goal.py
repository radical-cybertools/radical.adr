"""07-multi-operator-same-goal.py

Two operators racing to satisfy the same convergence goal concurrently.
The first to satisfy the goal cancels the other via a done-callback —
no shared state, no new primitives, pure asyncio.

ADL concepts demonstrated
  - Req 2: Multi-Operators (concurrent) same goal
  - run_until_first_satisfied — coordination via add_done_callback
  - Each operator is fully independent (own state, own tasks)
  - Clean shutdown: the winner's run() finishes naturally; the loser's
    run_to_completion task is cancelled, propagating to its in-flight
    AsyncFlow futures via the run() finally block

Usage:
    python 07-multi-operator-same-goal.py
"""

from __future__ import annotations

import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from radical.adl import Decision, Operator, Policy, decide, goals, observe, tool
from radical.adl.goals import ScientificGoal

TARGET     = 3.14159
MIN_RADIUS = 0.005


async def evaluate_task(x: float, seed: int = 0) -> dict:
    import random
    rng = random.Random(seed)
    await asyncio.sleep(0.01)   # simulate lightweight compute
    return {"x": x, "value": (x - TARGET) ** 2 + rng.gauss(0, 0.03)}


# ── Operator ──────────────────────────────────────────────────────────────────

class SearchOperator(Operator):
    """Adaptive search operator — identical to example 06."""

    center:  float = 0.0
    radius:  float = 5.0
    history: list  = []

    @goals
    def convergence_criteria(self):
        return ScientificGoal(
            name="converged", metric="radius",
            threshold=MIN_RADIUS, direction="minimize",
        )

    @tool
    async def evaluate(self, x: float, seed: int = 0):
        return self.engine.function_task(evaluate_task)(x=x, seed=seed)

    @observe
    def extract(self, snapshot) -> dict:
        new_results = [
            v for k, v in snapshot.observations.items()
            if k.startswith("result.") and isinstance(v, dict) and "x" in v
        ]
        history = list(snapshot.objectives.get("history", [])) + new_results
        best    = min(history, key=lambda r: r["value"]) if history else None
        return {
            "new_results": new_results,
            "history":     history,
            "center":      snapshot.objectives.get("center", 0.0),
            "radius":      snapshot.objectives.get("radius", 5.0),
            "best":        best,
            "cycle":       snapshot.cycle,
        }


class SearchPolicy(Policy):

    def __init__(self, op: SearchOperator) -> None:
        super().__init__()
        self._tools = op.get_tools()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] >= 30:
            return Decision(stop=True)

        if obs["best"] is not None:
            new_c, new_r = obs["best"]["x"], obs["radius"] * 0.6
            return Decision(actions=[
                self._tools.update_state("history", obs["history"]),
                self._tools.update_state("center",  new_c),
                self._tools.update_state("radius",  new_r),
                *self._spawn(new_c, new_r, obs["cycle"]),
            ])

        return Decision(actions=self._spawn(obs["center"], obs["radius"], obs["cycle"]))

    def _spawn(self, c, r, cyc):
        return [
            self._tools.spawn_task("evaluate", x=c + r * (2 * i / 3 - 1), seed=cyc * 4 + i)
            for i in range(4)
        ]


# ── Multi-operator coordination ───────────────────────────────────────────────

async def _consume(op: Operator) -> None:
    """Drain the run() generator to completion."""
    async for _ in op.run():
        pass


async def run_until_first_satisfied(*operators: Operator) -> list[Operator]:
    """Run N operators concurrently; stop all as soon as any one finishes.

    Returns the list in completion order (winner first).
    Uses add_done_callback — zero polling, zero extra coroutines.
    """
    tasks: list[asyncio.Task] = [
        asyncio.create_task(_consume(op), name=op.operator_id)
        for op in operators
    ]

    finished_order: list[str] = []

    def _on_any_done(done_task: asyncio.Task) -> None:
        finished_order.append(done_task.get_name())
        for t in tasks:
            if not t.done():
                t.cancel()

    for t in tasks:
        t.add_done_callback(_on_any_done)

    await asyncio.gather(*tasks, return_exceptions=True)

    op_by_id = {op.operator_id: op for op in operators}
    return [op_by_id[oid] for oid in finished_order if oid in op_by_id]


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    backend = await LocalExecutionBackend(ThreadPoolExecutor(max_workers=8))
    engine  = await WorkflowEngine.create(backend)

    # Two operators: same goal, different starting positions.
    # One is placed close to TARGET (fast), one is far (slow).
    starts = [
        ("search-fast", 3.5),   # close to TARGET=3.14159 — likely wins
        ("search-slow", -8.0),  # far from TARGET — likely loses
    ]

    operators = []
    for op_id, start_center in starts:
        op        = SearchOperator(engine, operator_id=op_id)
        op.center = start_center
        op.policy = SearchPolicy(op)
        operators.append(op)

    t0 = time.perf_counter()
    completion_order = await run_until_first_satisfied(*operators)
    elapsed = time.perf_counter() - t0

    print(f"\n{'Operator':<14} {'Start':>7} {'Final radius':>13} "
          f"{'Cycles':>7} {'Outcome'}")
    print("-" * 60)
    winner_id = completion_order[0].operator_id if completion_order else None
    for op in operators:
        best    = min(op.history, key=lambda r: r["value"]) if op.history else None
        outcome = "SATISFIED GOAL" if op.operator_id == winner_id else "cancelled"
        print(
            f"{op.operator_id:<14} "
            f"{dict(starts)[op.operator_id]:>+7.1f} "
            f"{op.radius:>13.6f} "
            f"{op.state.cycle:>7} "
            f"{outcome}"
        )
        if best:
            print(f"  └─ best x={best['x']:.6f}  error={abs(best['x'] - TARGET):.6f}")

    print(f"\nTotal wall time: {elapsed:.2f}s")
    if winner_id:
        print(f"Winner: {winner_id}")

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
