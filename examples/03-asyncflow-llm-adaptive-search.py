"""03-asyncflow-llm-adaptive-search.py

Same adaptive search as example 01, but the policy is an LLMPolicy backed
by OpenRouter.  Uses `instructor` to parse structured output.

ADR concepts demonstrated
  - LLMPolicy subclass with @decide — same shape as any other Policy
  - Vendor-agnostic: LLM call is inline in @decide, no hidden hooks
  - op.policy = OpenRouterPolicy(api_key)

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python 03-asyncflow-llm-adaptive-search.py

Dependencies:
    pip install openai instructor
"""

from __future__ import annotations

import asyncio
import os
import random
from concurrent.futures import ProcessPoolExecutor

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from radical.adr import Decision, LLMPolicy, Operator, decide, goals, observe, act
from radical.adr.goals import Goal

TARGET     = 3.14159
MIN_RADIUS = 0.005
MODEL      = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")


async def evaluate_task(x: float, seed: int = 0) -> dict:
    import random
    rng = random.Random(seed)
    return {"x": x, "value": (x - TARGET) ** 2 + rng.gauss(0, 0.03)}

SYSTEM_PROMPT = f"""\
Minimize f(x) = (x - {TARGET})² + noise on the real line.
Each cycle: pick best x seen so far, halve the search radius, spawn 4 evaluations.
Stop when radius < {MIN_RADIUS}.
"""


class SearchDecision(BaseModel):
    new_center:  float
    new_radius:  float
    evaluations: list[float] = Field(description="x values to evaluate next")
    stop:        bool = False


class OpenRouterPolicy(LLMPolicy):
    system_prompt = SYSTEM_PROMPT

    def __init__(self, api_key: str, op: "LLMSearchOperator") -> None:
        super().__init__()
        self._act = op.get_actions()
        raw         = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.client = instructor.from_openai(raw)

    @decide
    async def run(self, obs: dict) -> Decision:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": self.render_observation(obs)},
        ]
        sd = await self.client.chat.completions.create(
            model=MODEL, response_model=SearchDecision, messages=messages)
        return Decision(
            stop=sd.stop,
            actions=[
                self._act.update_param("center", sd.new_center),
                self._act.update_param("radius", sd.new_radius),
                *[self._act.evaluate(x=x, seed=i)
                  for i, x in enumerate(sd.evaluations)],
            ],
        )


class LLMSearchOperator(Operator):

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
        if new_results:
            self.history = history
        best = min(history, key=lambda r: r["value"]) if history else None
        return {"new_results": new_results, "history": history,
                "center": snapshot.objectives.get("center", 0.0),
                "radius": snapshot.objectives.get("radius", 5.0),
                "best": best, "cycle": snapshot.cycle}


async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY before running.")

    backend   = await LocalExecutionBackend(ProcessPoolExecutor())
    engine    = await WorkflowEngine.create(backend)
    op        = LLMSearchOperator(engine)
    op.policy = OpenRouterPolicy(api_key, op)

    async for snapshot in op.run():
        print(f"cycle {snapshot.cycle:2d}  center={op.center:+.5f}  radius={op.radius:.5f}")
        if snapshot.cycle >= 12:
            break

    if op.history:
        best = min(op.history, key=lambda r: r["value"])
        print(f"\nbest x={best['x']:.6f}  error={abs(best['x'] - TARGET):.6f}")
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
