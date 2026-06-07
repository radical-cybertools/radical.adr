"""09-langgraph-policy.py

Adaptive search steered by a LangGraph policy.

Extends the minimal LangGraph pattern (StateGraph + structured output) to show
what ADL adds on top:

  LangGraph  — reasons about observations and decides what to compute next
  ADL        — executes decisions on AsyncFlow, tracks goals, stops autonomously

ADL concepts demonstrated
  - LangGraph graph as the @decide policy
  - ScientificGoal: operator stops when converged, not when LLM says to
  - Runtime goal proposals: LangGraph agent tightens goals mid-campaign

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python 09-langgraph-policy.py

Dependencies:
    pip install langgraph langchain-openai
"""

from __future__ import annotations

import asyncio
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from radical.adl import Decision, LLMPolicy, Operator, act, decide, goals, observe
from radical.adl.goals import ScientificGoal

TARGET     = 3.14159
MIN_RADIUS = 0.005
MODEL      = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

SYSTEM_PROMPT = f"""\
You are steering an adaptive search for the minimum of f(x) = (x − π)² + noise.
The true minimum is near x ≈ {TARGET:.5f}.

Each cycle:
1. Set new_center to the best x seen so far.
2. Shrink new_radius by 0.5–0.7× the current radius.
3. Spread exactly 4 x_values around new_center within ±new_radius.
4. If the radius is already below 0.02 and clearly converging, propose
   tighter_threshold = 0.001. Otherwise null."""


# ── Module-level impl ─────────────────────────────────────────────────────────

async def _evaluate_impl(x: float, seed: int = 0) -> dict:
    rng = random.Random(seed)
    return {"x": x, "value": (x - TARGET) ** 2 + rng.gauss(0, 0.03)}


# ── Operator ──────────────────────────────────────────────────────────────────

class SearchOperator(Operator):

    center:  float = 0.0
    radius:  float = 5.0
    history: list  = []

    @goals # ADL: stops when ALL goals satisfied
    def convergence_criteria(self):
        return ScientificGoal(
            name="converged", metric="radius",
            threshold=MIN_RADIUS, direction="minimize",
        )

    @act                                            # ADL: dispatches to AsyncFlow backend
    async def evaluate(self, x: float, seed: int = 0):
        return self.engine.function_task(_evaluate_impl)(x=x, seed=seed)

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


# ── LangGraph structured output ───────────────────────────────────────────────

class EvaluationPlan(BaseModel):
    x_values:           list[float] = Field(description="Exactly 4 x values to evaluate next")
    new_center:         float       = Field(description="Best x seen so far — new search center")
    new_radius:         float       = Field(description="Shrunk radius for next round (multiply current by 0.5–0.7)")
    tighter_threshold:  float | None = Field(
        None,
        description="Propose a tighter radius goal (e.g. 0.001) if already below 0.02 and converging. Otherwise null.",
    )
    reasoning: str = Field(description="One sentence explaining this cycle's strategy")


# ── LangGraph policy ──────────────────────────────────────────────────────────

class LangGraphPolicy(LLMPolicy):
    """LangGraph reasoning graph used as the ADL @decide policy.

    LangGraph handles: LLM reasoning + structured output each cycle.
    ADL handles:       @act dispatch to AsyncFlow, goal evaluation,
                       runtime goal proposals, autonomous stopping.
    """

    system_prompt = SYSTEM_PROMPT

    def __init__(self, op: SearchOperator, api_key: str) -> None:
        super().__init__()
        self._act  = op.get_actions()
        llm        = ChatOpenAI(
            model    = MODEL,
            api_key  = api_key,
            base_url = "https://openrouter.ai/api/v1",
        )
        self._graph = self._build_graph(llm)

    def _build_graph(self, llm: ChatOpenAI):
        """Minimal LangGraph: one reasoning node, structured output."""

        class State(TypedDict):
            obs:  dict
            plan: Optional[dict]

        structured = llm.with_structured_output(EvaluationPlan)

        async def reason(state: State) -> dict:
            obs  = state["obs"]
            best = obs.get("best")
            msg  = (
                f"Cycle {obs['cycle']}  |  "
                f"center={obs['center']:.4f}  radius={obs['radius']:.6f}  "
                f"best={'x=%.5f val=%.5f' % (best['x'], best['value']) if best else 'none'}"
            )
            plan = await structured.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=msg),
            ])
            return {"plan": plan.model_dump()}

        g = StateGraph(State)
        g.add_node("reason", reason)
        g.add_edge(START, "reason")
        g.add_edge("reason", END)
        return g.compile()

    @decide
    async def run(self, obs: dict) -> Decision:
        # Cycle 0: seed without LLM — no data to reason about yet
        if obs["cycle"] == 0:
            return Decision(actions=[
                self._act.evaluate(x=obs["center"] + obs["radius"] * (2*i/3 - 1), seed=i)
                for i in range(4)
            ])

        # LangGraph reasons about the observation and returns a plan
        result = await self._graph.ainvoke({"obs": obs, "plan": None})
        plan   = EvaluationPlan.model_validate(result["plan"])

        print(f"  [cycle {obs['cycle']:2d}] {plan.reasoning}")

        actions = [
            self._act.update_param("history", obs["history"]),
            self._act.update_param("center",  plan.new_center),
            self._act.update_param("radius",  plan.new_radius),
            *[self._act.evaluate(x=x, seed=obs["cycle"] * 4 + i)
              for i, x in enumerate(plan.x_values)],
        ]

        # ADL VALUE: the LangGraph agent can propose a new ScientificGoal
        # at runtime. ADL adds it to the active set and evaluates it every
        # cycle alongside the original goal — no restart, no code change.
        new_goals = []
        if plan.tighter_threshold is not None:
            print(f"             → proposing tighter goal: radius < {plan.tighter_threshold}")
            new_goals.append(ScientificGoal(
                name="converged_tight", metric="radius",
                threshold=plan.tighter_threshold, direction="minimize",
            ))

        return Decision(actions=actions, goals=new_goals)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY before running.")

    backend = await LocalExecutionBackend(ThreadPoolExecutor(max_workers=4))
    engine  = await WorkflowEngine.create(backend)

    op        = SearchOperator(engine)
    op.policy = LangGraphPolicy(op, api_key)

    print("LangGraph + ADL  |  adaptive search")
    print(f"  model={MODEL}  target≈{TARGET}  goal: radius < {MIN_RADIUS}")
    print("  LangGraph decides what to compute  →  ADL executes + checks goals\n")

    async for _ in op.run():
        pass

    # The operator stopped because the ScientificGoal converged —
    # not because the LLM returned stop=True.
    best = min(op.history, key=lambda r: r["value"]) if op.history else None
    print(f"\nStopped by: ScientificGoal (autonomous) — not the LLM")
    print(f"Cycles: {op.state.cycle}  |  Final radius: {op.radius:.6f}")
    if best:
        print(f"Best:   x={best['x']:.6f}  error={abs(best['x'] - TARGET):.6f}")

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
