"""05-spawn-block.py

Spawning an AsyncFlow workflow block from an ADL operator using @tool.

ADL concepts demonstrated
  - @tool returning a block Future — method body orchestrates sub-tasks
  - tools.prepare_and_score(...) dispatches to the @tool which calls engine.block
  - No register_workflow_fn needed — the @tool method IS the workflow entry point

Usage:
    python 05-spawn-block.py
"""

from __future__ import annotations

import asyncio
import random
from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from radical.adl import Decision, Operator, Policy, decide, goals, observe, tool
from radical.adl.goals import ScientificGoal

CANDIDATES = [f"mol_{i:04d}" for i in range(30)]


def _mock_score(candidate: str) -> float:
    return -random.Random(hash(candidate)).uniform(3.0, 10.0)


# ── Module-level task implementations ────────────────────────────────────────

async def _preprocess_impl(c: str) -> dict:
    return {"candidate": c, "processed": c.upper()}


async def _score_impl(processed: str) -> dict:
    return {"processed": processed, "score": _mock_score(processed)}


# ── Operator ──────────────────────────────────────────────────────────────────

class PipelineOperator(Operator):

    best_score:       float = 0.0
    batch_size:       int   = 3
    candidates_tried: int   = 0

    @goals
    def convergence_criteria(self):
        return ScientificGoal(name="good_score", metric="best_score",
                              threshold=-8.0, direction="minimize")

    @tool
    async def prepare_and_score(self, candidate: str):
        """Orchestrate preprocess → score as a sequential block."""
        async def _pipeline(c: str) -> dict:
            pre   = await self.engine.function_task(_preprocess_impl)(c)
            score = self.engine.function_task(_score_impl)(pre["processed"])
            return await score
        return self.engine.block(_pipeline)(candidate)   # returns Future

    @observe
    def extract(self, snapshot) -> dict:
        results = [v for k, v in snapshot.observations.items()
                   if k.startswith("result.") and isinstance(v, dict) and "score" in v]
        if results:
            self.best_score = min(r["score"] for r in results)
        return {"results": results, "best_score": self.best_score, "cycle": snapshot.cycle}


class PipelinePolicy(Policy):

    def __init__(self, op: PipelineOperator) -> None:
        super().__init__()
        self._op    = op
        self._tools = op.get_tools()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] >= 8:
            return Decision(stop=True)
        actions = []
        for _ in range(self._op.batch_size):
            c = CANDIDATES[self._op.candidates_tried % len(CANDIDATES)]
            self._op.candidates_tried += 1
            actions.append(self._tools.prepare_and_score(candidate=c))
        return Decision(actions=actions)


async def main():
    backend = await LocalExecutionBackend(ThreadPoolExecutor(max_workers=4))
    engine  = await WorkflowEngine.create(backend)

    op        = PipelineOperator(engine)
    op.policy = PipelinePolicy(op)

    print("Pipeline operator  |  goal: best_score < -8.0 kcal/mol\n")
    async for snapshot in op.run():
        print(f"  cycle {snapshot.cycle:2d}  best={op.best_score:+.3f}  "
              f"results={len(snapshot.observations)}")

    print(f"\nFinal best: {op.best_score:+.3f}  cycles={op.state.cycle}")
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
