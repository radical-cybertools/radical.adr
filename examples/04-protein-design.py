"""04-protein-design.py

Protein design pipeline steered by an ADR operator.

ADR concepts demonstrated
  - Multiple @act methods (dock, md_refine)
  - @observe, @goals
  - Policy(primary=MockLLMPolicy(), fallback=ProteinDesignPolicy(op))
  - Both policies use @decide — same shape regardless of type

NOTE: Uses mock docking — replace @act bodies with real calls for production.

Usage:
    python 04-protein-design.py
"""

from __future__ import annotations

import asyncio
import random
from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from radical.adr import Decision, LLMPolicy, Operator, Policy, decide, goals, observe, act
from radical.adr.goals import Goal

CANDIDATES = [f"mol_{i:04d}" for i in range(100)]


def _mock_score(candidate: str, seed: int = 0) -> float:
    return -random.Random(hash(candidate) ^ seed).uniform(4.0, 10.0)


async def dock_task(candidate: str, seed: int = 0) -> dict:
    return {"candidate": candidate, "score": _mock_score(candidate, seed)}


async def md_refine_task(candidate: str) -> dict:
    return {"candidate": candidate,
            "score_refined": _mock_score(candidate) - random.uniform(0.1, 0.5)}


class MockLLMPolicy(LLMPolicy):
    """Placeholder — replace @decide body with a real LLM call."""
    system_prompt = "You are a protein design expert."

    @decide
    async def run(self, obs: dict) -> Decision:
        raise NotImplementedError("Replace with real LLM vendor call")


class ProteinDesignOperator(Operator):

    best_score:      float = 0.0
    batch_size:      int   = 4
    candidates_tried: int  = 0

    @goals
    def convergence_criteria(self):
        return Goal(name="good_score", metric="best_score",
                              threshold=-8.0, direction="minimize")

    @act
    async def dock(self, candidate: str, seed: int = 0):
        return self.engine.function_task(dock_task)(candidate=candidate, seed=seed)

    @act
    async def md_refine(self, candidate: str):
        return self.engine.function_task(md_refine_task)(candidate=candidate)

    @observe
    def extract(self, snapshot) -> dict:
        dock_results = [v for k, v in snapshot.observations.items()
                        if isinstance(v, dict) and "score" in v and "score_refined" not in v]
        refine_results = [v for k, v in snapshot.observations.items()
                          if isinstance(v, dict) and "score_refined" in v]
        all_scores = ([r["score"] for r in dock_results]
                      + [r["score_refined"] for r in refine_results])
        if all_scores:
            self.best_score = min(all_scores)
        return {"dock_results":    dock_results,
                "high_performers": [r["candidate"] for r in dock_results if r["score"] < -7.0],
                "best_score":      self.best_score,
                "cycle":           snapshot.cycle}


class ProteinDesignPolicy(Policy):

    def __init__(self, op: ProteinDesignOperator) -> None:
        super().__init__()
        self._op    = op
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        actions = [self._act.md_refine(candidate=c)
                   for c in obs["high_performers"][:2]]
        for _ in range(self._op.batch_size):
            c = CANDIDATES[self._op.candidates_tried % len(CANDIDATES)]
            self._op.candidates_tried += 1
            actions.append(self._act.dock(candidate=c, seed=obs["cycle"]))
        return Decision(actions=actions)


async def main():
    backend = await LocalExecutionBackend(ProcessPoolExecutor(max_workers=4))
    engine  = await WorkflowEngine.create(backend)

    op        = ProteinDesignOperator(engine)
    op.policy = Policy(
        primary  = MockLLMPolicy(),           # tries LLM; raises → fallback
        fallback = ProteinDesignPolicy(op),
    )

    print("Protein design  |  goal: best_score < -8.0 kcal/mol\n")
    async for snapshot in op.run():
        print(f"  cycle {snapshot.cycle:2d}  best={op.best_score:+.3f}")
        if snapshot.cycle >= 10:
            break

    print(f"\nFinal best: {op.best_score:+.3f}  cycles={op.state.cycle}")
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
