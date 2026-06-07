"""radical.adl — Agentic Adaptive Decision Layer for AsyncFlow.

Core loop (Run → Observe → Decide → Act)::

    1. RUN     — drain inbox: task results + telemetry → RuntimeState
    2. OBSERVE — operator.observe(snapshot) → obs dict (clean domain data)
    3. DECIDE  — policy.decide(obs) OR @loop fallback method → Decision
    4. ACT     — apply Decision.actions → AsyncFlow calls
               check goals → stop if all satisfied

Sacred boundary:
    ADL NEVER: schedule, manage execution lifecycle, resolve dependencies,
               own resources, coordinate distributed systems.
    ADL ONLY:  observe (await future, TelemetryObserver),
               reason  (policy.decide),
               act     (factory(), future.cancel()).

Quick-start::

    from radical.adl import Operator, act, loop, Decision, Action, ActionKind
    from radical.adl.goals import ScientificGoal
    from radical.adl.policy import RulePolicy

    class SearchOperator(Operator):
        goals  = [ScientificGoal(name="converged", metric="radius",
                                 threshold=0.005, direction="minimize")]
        center: float = 0.0
        radius: float = 10.0
        history: list = []

        @act
        async def evaluate(self, x: float, seed: int = 0) -> dict:
            import random
            rng = random.Random(seed)
            return {"x": x, "value": (x - 3.14159) ** 2 + rng.gauss(0, 0.03)}

        @loop(policy=RulePolicy([
            (lambda obs: bool(obs["new_results"]),
             lambda obs: Decision(actions=[
                 Action(kind=ActionKind.UPDATE_PARAM, param_key="radius",
                        param_value=obs.get("radius", 1.0) * 0.5),
             ])),
            (lambda obs: True, lambda obs: Decision()),
        ]))
        async def adaptive_cycle(self, obs: dict) -> Decision:
            return Decision(stop=True)

    async def main():
        from radical.asyncflow import WorkflowEngine, LocalExecutionBackend
        from concurrent.futures import ThreadPoolExecutor
        backend = await LocalExecutionBackend(ThreadPoolExecutor())
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
from radical.adl.goals import ResourceGoal
from radical.adl.goals import ScientificGoal
from radical.adl.message import InboxMessage
from radical.adl.message import MessageBus
from radical.adl.message import MessageKind
from radical.adl.observer import ObserverBase
from radical.adl.observer import TelemetryObserver
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
    "ScientificGoal",
    "ResourceGoal",
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
    "TelemetryObserver",
    # Errors
    "ADLError",
    "PolicyError",
    "ActionError",
    "RegistryError",
]
