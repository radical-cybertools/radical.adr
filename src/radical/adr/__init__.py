"""radical.adr — Autonomous Decision Runtime for AsyncFlow.

Core loop (Run → Observe → Decide → Act)::

    1. RUN     — drain inbox: task results → RuntimeState
    2. OBSERVE — operator.observe(snapshot) → obs dict (clean domain data)
    3. DECIDE  — policy.decide(obs) → Decision
    4. ACT     — apply Decision.actions → AsyncFlow calls
               check goals → stop if all satisfied

Sacred boundary:
    ADR NEVER: schedule, manage execution lifecycle, resolve dependencies,
               own resources, coordinate distributed systems.
    ADR ONLY:  observe (@observe assembles metrics from any source),
               reason  (policy.decide),
               act     (@act calls, future.cancel()).

Quick-start::

    from radical.adr import Operator, act, Decision, goals, observe
    from radical.adr.goals import Goal
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

from radical.adr.decision import Action
from radical.adr.decision import ActionKind
from radical.adr.decision import Decision
from radical.adr.errors import ActionError
from radical.adr.errors import ADLError
from radical.adr.errors import PolicyError
from radical.adr.errors import RegistryError
from radical.adr.goals import Goal
from radical.adr.logging import enable_logging
from radical.adr.message import InboxMessage
from radical.adr.message import MessageBus
from radical.adr.message import MessageKind
from radical.adr.observer import ObserverBase
from radical.adr.recording import RecordingObserver
from radical.adr.recording import replay_trace
from radical.adr.operator import Operator
from radical.adr.operator import ChildHandle
from radical.adr.operator import goals
from radical.adr.operator import observe
from radical.adr.operator import act
from radical.adr.policy.base import Policy
from radical.adr.policy.base import decide
from radical.adr.policy.combinators import ChainPolicy
from radical.adr.policy.llm import LLMPolicy
from radical.adr.state import RuntimeState
from radical.adr.state import Snapshot
from radical.adr.tools import ActionSet

__all__ = [
    # Operator + decorators
    "Operator",
    "ChildHandle",
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
    "RecordingObserver",
    "replay_trace",
    # Logging
    "enable_logging",
    # Errors
    "ADLError",
    "PolicyError",
    "ActionError",
    "RegistryError",
]
