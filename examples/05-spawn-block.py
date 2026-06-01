"""05-spawn-block.py

Spawning an AsyncFlow workflow block (SPAWN_BLOCK) from an ADL operator.

ADL concepts demonstrated
  - register_workflow_fn() to register an @engine.block sub-workflow
  - tools.spawn_workflow() → Action(kind=SPAWN_BLOCK)
  - @decide for explicit policy entry point

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


class PipelineOperator(Operator):

    best_score:      float = 0.0
    batch_size:      int   = 3
    candidates_tried: int  = 0

    @goals
    def convergence_criteria(self):
        return ScientificGoal(name="good_score", metric="best_score",
                              threshold=-8.0, direction="minimize")

    @tool
    async def preprocess(self, candidate: str) -> dict:
        return {"candidate": candidate, "processed": candidate.upper()}

    @tool
    async def dock(self, processed: str) -> dict:
        return {"processed": processed, "score": _mock_score(processed)}

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
            actions.append(self._tools.spawn_workflow("prepare_and_score", candidate=c))
        return Decision(actions=actions)


async def main():
    backend = await LocalExecutionBackend(ThreadPoolExecutor(max_workers=4))
    engine  = await WorkflowEngine.create(backend)

    op = PipelineOperator(engine)

    @engine.block
    async def prepare_and_score(candidate: str) -> dict:
        pre   = await engine.function_task(
            lambda c: {"candidate": c, "processed": c.upper()})(candidate)
        score = await engine.function_task(
            lambda p: {"processed": p, "score": _mock_score(p)})(pre["processed"])
        return score

    op.register_workflow_fn("prepare_and_score", prepare_and_score)
    op.policy = PipelinePolicy(op)

    print("Pipeline operator  |  goal: best_score < -8.0 kcal/mol\n")
    async for snapshot in op.run():
        print(f"  cycle {snapshot.cycle:2d}  best={op.best_score:+.3f}  "
              f"results={len(snapshot.observations)}")

    print(f"\nFinal best: {op.best_score:+.3f}  cycles={op.state.cycle}")
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
