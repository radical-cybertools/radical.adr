"""radical.adl — Agentic Adaptive Decision Layer for AsyncFlow.

Core loop (Run → Observe → Decide → Act)::

    1. RUN     — drain inbox: task results → RuntimeState
    2. OBSERVE — operator.observe(snapshot) → obs dict (clean domain data)
    3. DECIDE  — policy.decide(obs) → Decision
    4. ACT     — apply Decision.actions → AsyncFlow calls
               check goals → stop if all satisfied

Sacred boundary:
    ADL NEVER: schedule, manage execution lifecycle, resolve dependencies,
               own resources, coordinate distributed systems.
    ADL ONLY:  observe (@observe assembles metrics from any source),
               reason  (policy.decide),
               act     (@act calls, future.cancel()).

Quick-start::

    from radical.adl import Operator, act, Decision, goals, observe
    from radical.adl.goals import Goal
    from concurrent.futures import ProcessPoolExecutor

    class SearchOperator(Operator):
        center: float = 0.0
        radius: float = 10.0

        @goals
        def criteria(self):
            return Goal(name="converged", metric="radius",
                        threshold=0.005, direction="minimize")

        @act
        async def evaluate(self, x: float, seed: int = 0) -> dict:
            import random
            rng = random.Random(seed)
            return {"x": x, "value": (x - 3.14159) ** 2 + rng.gauss(0, 0.03)}

        @observe
        def extract(self, snapshot) -> dict:
            results = [v for k, v in snapshot.observations.items()
                       if k.startswith("result.") and isinstance(v, dict)]
            return {"radius": snapshot.objectives.get("radius", self.radius),
                    "results": results}

    async def main():
        from radical.asyncflow import WorkflowEngine, LocalExecutionBackend
        backend = await LocalExecutionBackend(ProcessPoolExecutor())
        engine  = await WorkflowEngine.create(backend)
        op = SearchOperator(engine)
        async for snapshot in op.run():
            print(snapshot.cycle, op.radius)
        await engine.shutdown()
"""

from radical.adl.decision import Action
from radical.adl.decision import ActionKind
from radical.adl.decision import Decision
from radical.adl.errors import ActionError
from radical.adl.errors import ADLError
from radical.adl.errors import PolicyError
from radical.adl.errors import RegistryError
from radical.adl.goals import Goal
from radical.adl.message import InboxMessage
from radical.adl.message import MessageBus
from radical.adl.message import MessageKind
from radical.adl.observer import ObserverBase
from radical.adl.operator import Operator
from radical.adl.operator import goals
from radical.adl.operator import observe
from radical.adl.operator import act
from radical.adl.policy.base import Policy
from radical.adl.policy.base import decide
from radical.adl.policy.combinators import ChainPolicy
from radical.adl.policy.llm import LLMPolicy
from radical.adl.state import RuntimeState
from radical.adl.state import Snapshot
from radical.adl.tools import ActionSet

__all__ = [
    # Operator + decorators
    "Operator",
    "act",
    "observe",
    "goals",
    # State
    "RuntimeState",
    "Snapshot",
    # Decision
    "Decision",
    "Action",
    "ActionKind",
    # Goals
    "Goal",
    # Messages
    "InboxMessage",
    "MessageKind",
    "MessageBus",
    # Policies
    "decide",
    "Policy",
    "LLMPolicy",
    "ChainPolicy",
    # Tools
    "ActionSet",
    # Observers
    "ObserverBase",
    # Errors
    "ADLError",
    "PolicyError",
    "ActionError",
    "RegistryError",
]
