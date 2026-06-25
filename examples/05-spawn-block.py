"""05-spawn-block.py

Spawning an AsyncFlow workflow block from an ADR operator using @act.

ADR concepts demonstrated
  - @act returning a block Future — method body orchestrates sub-tasks
  - acts.prepare_and_score(...) dispatches to the @act which calls engine.block
  - No register_workflow_fn needed — the @act method IS the workflow entry point

Usage:
    python 05-spawn-block.py
"""

from __future__ import annotations

import asyncio
import logging
import random
from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from radical.adr import Decision, Operator, Policy, decide, goals, observe, act
from radical.adr import enable_logging
from radical.adr.goals import Goal

CANDIDATES = [f"mol_{i:04d}" for i in range(30)]

# ── Module-level task implementations ────────────────────────────────────────

async def workflow(engine: WorkflowEngine, candidate: str) -> dict:
    @engine.function_task
    async def _preprocess_impl(candidate: str) -> dict:
        return {"candidate": candidate, "processed": candidate.upper()}

    @engine.function_task
    async def _score_impl(processed) -> dict:
        print(processed)
        return {"processed": processed,
                "score": -random.Random(hash(processed)).uniform(3.0, 10.0)}
    
    pre_future = await _preprocess_impl(candidate)
    
    score_result = await _score_impl(pre_future)

    print(score_result)

    

    return score_result



# ── Operator ──────────────────────────────────────────────────────────────────

class PipelineOperator(Operator):

    best_score:       float = 0.0
    batch_size:       int   = 3
    candidates_tried: int   = 0

    @goals
    def convergence_criteria(self):
        return Goal(name="good_score", metric="best_score",
                              threshold=-8.0, direction="minimize")

    @act
    async def prepare_and_score(self, candidate: str):
        """Orchestrate preprocess → score as a sequential block."""
        return self.engine.block(workflow)(self.engine, candidate)   # returns Future

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
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] >= 8:
            return Decision(stop=True)
        actions = []
        for _ in range(self._op.batch_size):
            c = CANDIDATES[self._op.candidates_tried % len(CANDIDATES)]
            self._op.candidates_tried += 1
            actions.append(self._act.prepare_and_score(candidate=c))
        return Decision(actions=actions)


async def main():
    backend = await LocalExecutionBackend(ProcessPoolExecutor(max_workers=4))
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
    enable_logging(logging.INFO)
    asyncio.run(main())
