"""Operator — the central ADL runtime class.

Usage (subclass-based)::

    from radical.adl import Operator, tool, loop, Decision, Action, ActionKind
    from radical.adl.goals import ScientificGoal
    from radical.adl.policy import RulePolicy

    class SearchOperator(Operator):
        goals = [ScientificGoal(name="converged", metric="radius",
                                threshold=0.005, direction="minimize")]

        center: float = 0.0
        radius: float = 10.0
        history: list = []

        @tool
        async def evaluate(self, x: float, seed: int = 0) -> dict: ...

        @loop(policy=RulePolicy([
            (lambda obs: bool(obs["new_results"]),
             lambda obs: Decision(actions=[...])),
            (lambda obs: True,
             lambda obs: Decision(stop=True)),
        ]))
        async def adaptive_cycle(self, obs: dict) -> Decision:
            return Decision(stop=True)

    op = SearchOperator(engine)
    async for snapshot in op.run():
        print(snapshot.cycle)
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from radical.adl.decision import Action, ActionKind, Decision
from radical.adl.errors import ActionError, PolicyError
from radical.adl.message import InboxMessage, MessageKind
from radical.adl.state import RuntimeState, Snapshot
from radical.adl.tools import ToolSet

from radical.adl.goals import Goal

if TYPE_CHECKING:
    from radical.adl.observer import ObserverBase
    from radical.adl.policy.base import Policy

log = logging.getLogger(__name__)

_DRAIN_BATCH   = 64
_DRAIN_TIMEOUT = 0.05   # seconds
_MISSING       = object()  # sentinel for "no default value"

# Instance attributes that must NOT be proxied to state.objectives
_RESERVED = frozenset({
    "operator_id", "engine", "parent", "observer",
    "state", "inbox", "children",
    "task_registry", "workflow_registry",
    "_stop_event", "_watcher_tasks", "_adl_toolset", "_policy", "_runtime_goals",
})


# ── Module-level decorators ───────────────────────────────────────────────────

def tool(fn: Callable) -> Callable:
    """Mark a method as an ADL tool (spawnable as an AsyncFlow task).

    The method body is the task implementation.  When spawned, ``self`` is
    excluded — the method must not depend on operator instance state.

    Example::

        @tool
        async def evaluate(self, x: float, seed: int = 0) -> dict:
            import random
            rng = random.Random(seed)
            return {"x": x, "value": (x - 3.14) ** 2 + rng.gauss(0, 0.03)}
    """
    fn._adl_tool = True  # type: ignore[attr-defined]
    return fn


def observe(fn: Callable) -> Callable:
    """Mark a method as the operator's observation extractor.

    The method receives a ``Snapshot`` and returns a plain ``dict`` of
    domain observations.  There can be at most one ``@observe`` method per
    operator class.

    Example::

        @observe
        def extract(self, snapshot) -> dict:
            scores = [v["score"] for v in snapshot.observations.values()
                      if isinstance(v, dict) and "score" in v]
            return {"best_score": min(scores, default=0.0),
                    "cycle":      snapshot.cycle}
    """
    fn._adl_observe = True  # type: ignore[attr-defined]
    return fn


def goals(fn: Callable) -> Callable:
    """Mark a method as the operator's convergence criteria.

    The method returns a single ``Goal`` or a list of ``Goal`` objects.
    The operator stops when ALL goals are satisfied.

    Example::

        @goals
        def convergence_criteria(self):
            return [
                ScientificGoal(name="converged", metric="radius", threshold=0.005),
                ResourceGoal(name="cpu", metric="cpu_percent", threshold=85.0),
            ]
    """
    fn._adl_goals = True  # type: ignore[attr-defined]
    return fn


# ── Future watcher ────────────────────────────────────────────────────────────

async def _watch_future(uid: str, future: asyncio.Future, inbox: asyncio.Queue) -> None:
    """Background task: await a single AsyncFlow future and deliver its outcome."""
    try:
        result = await future
        await inbox.put(InboxMessage(
            kind      = MessageKind.TASK_EVENT,
            sender_id = "asyncflow",
            payload   = {"uid": uid, "state": "DONE", "result": result},
        ))
    except asyncio.CancelledError:
        await inbox.put(InboxMessage(
            kind      = MessageKind.TASK_EVENT,
            sender_id = "asyncflow",
            payload   = {"uid": uid, "state": "CANCELED"},
        ))
    except Exception as exc:
        await inbox.put(InboxMessage(
            kind      = MessageKind.TASK_EVENT,
            sender_id = "asyncflow",
            payload   = {"uid": uid, "state": "FAILED", "error": exc},
        ))


# ── Operator ─────────────────────────────────────────────────────────────────

class Operator:
    """Agentic runtime operator.

    Subclass to define tools, goals, and the adaptive cycle::

        class MyOp(Operator):
            goals = [ScientificGoal(...)]
            best:  float = float("inf")      # persists across cycles via state.objectives

            @tool
            async def compute(self, x: float) -> dict: ...

            @decide(policy=RulePolicy([...]))
            async def adaptive_cycle(self, obs: dict) -> Decision:
                return Decision(stop=True)   # fallback

        op = MyOp(engine)
        async for snapshot in op.run(): ...

    The operator ONLY: observes (``_watch_future``, telemetry),
    reasons (``policy.decide``), and acts (``factory()``, ``future.cancel()``).
    It never owns scheduling, execution lifecycle, or distributed coordination.
    """

    # ── Class-level registries (populated by __init_subclass__) ──────────────

    _adl_tools_registry: dict[str, Callable] = {}
    _adl_observe_fn:     Callable | None      = None
    _adl_goals_fn:       Callable | None      = None
    _adl_state_keys:     set[str]             = set()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Fresh registries per subclass (no shared mutable dicts)
        cls._adl_tools_registry = {}
        cls._adl_observe_fn     = None
        cls._adl_goals_fn       = None

        for name, val in vars(cls).items():
            if callable(val) and getattr(val, "_adl_tool", False):
                cls._adl_tools_registry[name] = val
            if callable(val) and getattr(val, "_adl_observe", False):
                cls._adl_observe_fn = val
            if callable(val) and getattr(val, "_adl_goals", False):
                cls._adl_goals_fn = val

        # Collect annotated class attributes that represent operator state
        state_keys: set[str] = set()
        for klass in cls.__mro__:
            for aname in vars(klass).get("__annotations__", {}):
                raw = vars(klass).get(aname, _MISSING)
                if (not aname.startswith("_")
                        and aname not in _RESERVED
                        and not isinstance(raw, (classmethod, staticmethod, property))):
                    state_keys.add(aname)
        cls._adl_state_keys = state_keys

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        engine:        Any,
        operator_id:   str | None               = None,
        *,
        policy:        "Policy | None"      = None,
        parent:        "Operator | None"        = None,
        inbox_maxsize: int                       = 0,
        observer:      "ObserverBase | None"    = None,
    ) -> None:
        object.__setattr__(self, "operator_id",  operator_id or type(self).__name__)
        object.__setattr__(self, "engine",       engine)
        object.__setattr__(self, "parent",       parent)
        object.__setattr__(self, "observer",     observer)

        object.__setattr__(self, "state",         RuntimeState(self.operator_id))
        object.__setattr__(self, "inbox",         asyncio.Queue(maxsize=inbox_maxsize))
        object.__setattr__(self, "children",      {})
        object.__setattr__(self, "task_registry", {})
        object.__setattr__(self, "workflow_registry", {})
        object.__setattr__(self, "_stop_event",   asyncio.Event())
        object.__setattr__(self, "_watcher_tasks", [])

        object.__setattr__(self, "_policy",        policy)
        object.__setattr__(self, "_runtime_goals", [])

        # Seed state.objectives from class-level annotated defaults
        for key in type(self)._adl_state_keys:
            for klass in type(self).__mro__:
                raw = vars(klass).get(key, _MISSING)
                if raw is not _MISSING and not callable(raw):
                    self.state.objectives[key] = raw
                    break

        # Auto-register @tool methods with the AsyncFlow engine
        for name, fn in type(self)._adl_tools_registry.items():
            self._register_tool_as_task(name, fn)

        # Build ToolSet
        object.__setattr__(self, "_adl_toolset", ToolSet(self))

    # ── State proxy ───────────────────────────────────────────────────────────

    def __getattribute__(self, name: str) -> Any:
        # Intercept reads for annotated state keys → state.objectives.
        # Class-level defaults (e.g. center: float = 0.0) would shadow __getattr__,
        # so we must override __getattribute__ to take priority.
        state_keys: set = type(self)._adl_state_keys
        if name in state_keys:
            try:
                objs = object.__getattribute__(self, "state").objectives
                if name in objs:
                    return objs[name]
            except AttributeError:
                pass
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self)._adl_state_keys:
            try:
                state = object.__getattribute__(self, "state")
                state.objectives[name] = value
                return
            except AttributeError:
                pass
        object.__setattr__(self, name, value)

    # ── Tool registration ─────────────────────────────────────────────────────

    def _register_tool_as_task(self, name: str, fn: Callable) -> None:
        """Wrap a @tool method (drop self) and register with engine.function_task."""
        sig = inspect.signature(fn)
        new_params = [p for pname, p in sig.parameters.items() if pname != "self"]

        @functools.wraps(fn)
        async def _task_fn(**kwargs: Any) -> Any:
            return await fn(None, **kwargs)

        _task_fn.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]

        registered = self.engine.function_task(_task_fn)
        self.task_registry[name] = registered

    def register_task_fn(self, name: str, fn: Callable) -> None:
        """Explicitly register an AsyncFlow task factory by name.

        Use this when the task is defined outside the class body (e.g., for
        ProcessPoolExecutor compatibility where the function must be at module level).
        """
        self.task_registry[name] = fn

    def register_workflow_fn(self, name: str, fn: Callable) -> None:
        """Register an AsyncFlow workflow block factory by name."""
        self.workflow_registry[name] = fn

    # ── Policy ───────────────────────────────────────────────────────────────

    @property
    def policy(self) -> "Policy | None":
        """Active decision policy.  Overrides the ``@loop``-level default."""
        return object.__getattribute__(self, "_policy")

    @policy.setter
    def policy(self, value: "Policy | None") -> None:
        object.__setattr__(self, "_policy", value)

    # ── Tools ─────────────────────────────────────────────────────────────────

    def get_tools(self, format: str | None = None) -> "ToolSet | list":
        """Return the operator's ToolSet, optionally rendered as LLM schemas.

        ``get_tools()``              → ToolSet (Python callables, usable by any policy)
        ``get_tools("openai")``      → list[dict] OpenAI tool spec
        ``get_tools("anthropic")``   → list[dict] Anthropic tool spec
        ``get_tools("langgraph")``   → list[BaseTool] LangGraph tools
        """
        toolset: ToolSet = object.__getattribute__(self, "_adl_toolset")
        if format is None:
            return toolset
        return toolset.get_schema(format)

    # ── Decide ────────────────────────────────────────────────────────────────

    async def _decide(self, obs: dict) -> Decision:
        policy = object.__getattribute__(self, "_policy")
        if policy is None:
            raise PolicyError(
                f"Operator '{self.operator_id}' has no policy set. "
                "Assign one via op.policy = ... before calling run()."
            )
        return await policy.decide(obs)

    def _check_goals(self, obs: dict, snapshot: Snapshot, decision: Decision) -> None:
        """Force stop if all active goals (static + runtime) are satisfied."""
        goals_fn = type(self)._adl_goals_fn
        static: list = []
        if goals_fn is not None:
            raw = goals_fn(self)
            static = raw if isinstance(raw, list) else [raw]
            for g in static:
                if not isinstance(g, Goal):
                    raise TypeError(
                        f"@goals method '{goals_fn.__name__}' on {type(self).__name__} "
                        f"must return Goal or list[Goal], got {type(g).__name__}"
                    )
        runtime: list = object.__getattribute__(self, "_runtime_goals")
        all_goals = static + runtime
        if all_goals and all(g.satisfied(obs, snapshot) for g in all_goals):
            decision.stop = True

    def _apply_goal_changes(self, decision: Decision) -> None:
        """Upsert / remove runtime goals proposed by the policy."""
        runtime: list = object.__getattribute__(self, "_runtime_goals")
        if decision.remove_goals:
            names = set(decision.remove_goals)
            runtime[:] = [g for g in runtime if g.name not in names]
        for new_goal in decision.goals:
            idx = next((i for i, g in enumerate(runtime) if g.name == new_goal.name), None)
            if idx is not None:
                runtime[idx] = new_goal   # update threshold / config in place
            else:
                runtime.append(new_goal)  # new goal

    @property
    def runtime_goals(self) -> list:
        """Runtime goals currently proposed by the policy."""
        return object.__getattribute__(self, "_runtime_goals")

    # ── Hierarchy ─────────────────────────────────────────────────────────────

    def spawn_child(
        self,
        child_id: str,
        policy:   "Policy | None" = None,
        engine:   Any = None,
        **kwargs: Any,
    ) -> "Operator":
        """Create and register a child operator."""
        child = Operator(
            engine or self.engine,
            child_id,
            policy = policy,
            parent = self,
            **kwargs,
        )
        self.children[child_id] = child
        return child

    def detach_child(self, child_id: str) -> "Operator | None":
        child = self.children.pop(child_id, None)
        if child is not None:
            child.parent = None
        return child

    def all_descendants(self) -> list["Operator"]:
        result = [self]
        for child in self.children.values():
            result.extend(child.all_descendants())
        return result

    # ── Inbox / messaging ────────────────────────────────────────────────────

    async def send(self, msg: InboxMessage) -> None:
        await self.inbox.put(msg)

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self) -> AsyncIterator[Snapshot]:
        """Async generator: one Snapshot per cycle until stop.

        Each cycle:
          1. RUN     — drain inbox → RuntimeState
          2. OBSERVE — self.observe(snapshot) → obs dict
          3. DECIDE  — policy.decide(obs) OR fallback method → Decision
          4. ACT     — apply(Decision.actions) → AsyncFlow calls
                       check goals → may force stop
        """
        if self.observer is not None:
            self.observer.on_start(self.operator_id, {"objectives": self.state.objectives})

        try:
            while not self._stop_event.is_set():
                await self._drain_inbox()                           # 1. RUN

                snapshot   = self.state.snapshot()
                observe_fn = type(self)._adl_observe_fn             # 2. OBSERVE
                if observe_fn is None:
                    raise TypeError(
                        f"Operator '{self.operator_id}' ({type(self).__name__}) has no "
                        "@observe method. Every Operator subclass must define one — "
                        "ADL has no default observation."
                    )
                obs = observe_fn(self, snapshot)
                if not isinstance(obs, dict):
                    raise TypeError(
                        f"@observe method '{observe_fn.__name__}' on "
                        f"{type(self).__name__} must return dict, "
                        f"got {type(obs).__name__}"
                    )

                try:
                    decision = await self._decide(obs)              # 3. DECIDE
                except Exception as exc:
                    raise PolicyError(
                        f"Operator '{self.operator_id}' policy raised on cycle {self.state.cycle}"
                    ) from exc

                self._check_goals(obs, snapshot, decision)
                self._apply_goal_changes(decision)

                if self.observer is not None:
                    self.observer.on_cycle(snapshot, decision)

                await self._apply_actions(decision.actions)         # 4. ACT
                await self._dispatch(decision)

                if decision.stop:
                    self._stop_event.set()

                yield snapshot

                self.state.clear_cycle()
                self.state.cycle += 1

                if decision.stop:
                    break

        finally:
            for future in self.state.task_futures.values():
                if not future.done():
                    future.cancel()
            for task in self._watcher_tasks:
                task.cancel()
            if self._watcher_tasks:
                await asyncio.gather(*self._watcher_tasks, return_exceptions=True)
            self._watcher_tasks.clear()

            final_snap = self.state.snapshot() if self.state.cycle > 0 else None
            if self.observer is not None:
                reason = "stop" if self._stop_event.is_set() else "shutdown"
                self.observer.on_stop(final_snap, reason)

    async def run_to_completion(self) -> Snapshot | None:
        """Consume the run generator to completion and return the last snapshot."""
        last: Snapshot | None = None
        async for snapshot in self.run():
            last = snapshot
        return last

    async def shutdown(self) -> None:
        """Signal stop and cancel outstanding watcher tasks."""
        self._stop_event.set()
        await self.inbox.put(InboxMessage(
            kind=MessageKind.STOP, sender_id="system", payload=None
        ))
        for task in self._watcher_tasks:
            task.cancel()
        if self._watcher_tasks:
            await asyncio.gather(*self._watcher_tasks, return_exceptions=True)
        self._watcher_tasks.clear()

    # ── Inbox drain ──────────────────────────────────────────────────────────

    async def _drain_inbox(self) -> None:
        drained = 0
        while drained < _DRAIN_BATCH:
            try:
                msg = self.inbox.get_nowait()
            except asyncio.QueueEmpty:
                if drained == 0:
                    try:
                        msg = await asyncio.wait_for(
                            self.inbox.get(), timeout=_DRAIN_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        break
                else:
                    break
            self._apply_message(msg)
            drained += 1

    def _apply_message(self, msg: InboxMessage) -> None:
        if msg.kind == MessageKind.STOP:
            self._stop_event.set()
            return

        if msg.kind == MessageKind.TASK_EVENT:
            payload = msg.payload
            uid     = payload.get("uid") if isinstance(payload, dict) else None
            if uid:
                status = payload.get("state", "UNKNOWN")
                self.state.runtime[uid] = status
                if status == "DONE":
                    self.state.observations[f"result.{uid}"] = payload.get("result")
            else:
                event_type = getattr(payload, "event_type", "unknown")
                self.state.observations[f"telemetry.{event_type}"] = payload
                if event_type == "ResourceUpdate":
                    for attr in ("cpu_percent", "memory_percent", "gpu_percent"):
                        val = getattr(payload, attr, None)
                        if val is not None:
                            self.state.runtime[attr] = val

        elif msg.kind == MessageKind.DIRECTIVE:
            payload = msg.payload
            if isinstance(payload, dict):
                self.state.directives.update(payload.get("directives", payload))

        elif msg.kind == MessageKind.SNAPSHOT:
            snapshot: Snapshot = msg.payload
            child_id = msg.sender_id
            for k, v in snapshot.observations.items():
                self.state.observations[f"child:{child_id}:{k}"] = v
            for k, v in snapshot.runtime.items():
                self.state.observations[f"child:{child_id}:runtime:{k}"] = v

    # ── Action application ───────────────────────────────────────────────────

    async def _apply_actions(self, actions: list[Action]) -> None:
        for act in actions:
            try:
                await self._apply_one(act)
            except Exception as exc:
                raise ActionError(
                    f"Operator '{self.operator_id}': action {act.kind} failed"
                ) from exc

    async def _apply_one(self, act: Action) -> None:
        if act.kind == ActionKind.SPAWN_TASK:
            if act.task_name is None or act.task_name not in self.task_registry:
                raise ActionError(
                    f"SPAWN_TASK: unknown task_name '{act.task_name}'. "
                    f"Registered: {list(self.task_registry)}"
                )
            factory    = self.task_registry[act.task_name]
            dep_futures = [
                self.state.task_futures[uid]
                for uid in act.depends_on
                if uid in self.state.task_futures
            ]
            future = factory(*dep_futures, **act.task_kwargs)
            uid    = str(getattr(future, "id", id(future)))
            self.state.task_futures[uid] = future
            self.state.runtime[uid]      = "PENDING"
            task = asyncio.create_task(
                _watch_future(uid, future, self.inbox),
                name=f"watcher.{self.operator_id}.{uid}",
            )
            self._watcher_tasks.append(task)

        elif act.kind == ActionKind.CANCEL_TASK:
            if act.target_uid is None:
                raise ActionError("CANCEL_TASK requires target_uid")
            future = self.state.task_futures.get(act.target_uid)
            if future is None:
                raise ActionError(f"CANCEL_TASK: unknown uid '{act.target_uid}'")
            future.cancel()

        elif act.kind == ActionKind.SPAWN_BLOCK:
            if act.task_name is None or act.task_name not in self.workflow_registry:
                raise ActionError(
                    f"SPAWN_BLOCK: unknown workflow '{act.task_name}'. "
                    f"Registered: {list(self.workflow_registry)}"
                )
            factory    = self.workflow_registry[act.task_name]
            dep_futures = [
                self.state.task_futures[uid]
                for uid in act.depends_on
                if uid in self.state.task_futures
            ]
            future = factory(*dep_futures, **act.task_kwargs)
            uid    = str(getattr(future, "id", id(future)))
            self.state.task_futures[uid] = future
            self.state.runtime[uid]      = "PENDING"
            task = asyncio.create_task(
                _watch_future(uid, future, self.inbox),
                name=f"watcher.{self.operator_id}.{uid}",
            )
            self._watcher_tasks.append(task)

        elif act.kind == ActionKind.UPDATE_PARAM:
            if act.param_key is None:
                raise ActionError("UPDATE_PARAM requires param_key")
            self.state.objectives[act.param_key] = act.param_value

    # ── Dispatch ─────────────────────────────────────────────────────────────

    async def _dispatch(self, decision: Decision) -> None:
        for child_id in decision.delegate_to:
            child = self.children.get(child_id)
            if child is not None:
                await child.inbox.put(InboxMessage(
                    kind      = MessageKind.DIRECTIVE,
                    sender_id = self.operator_id,
                    payload   = {"directives": decision.directives},
                ))

        for (target_id, payload) in decision.notify:
            child = self.children.get(target_id)
            if child is not None:
                await child.inbox.put(InboxMessage(
                    kind      = MessageKind.CUSTOM,
                    sender_id = self.operator_id,
                    payload   = payload,
                ))

        if self.parent is not None:
            snapshot = self.state.snapshot(source=f"child:{self.operator_id}")
            await self.parent.inbox.put(InboxMessage(
                kind      = MessageKind.SNAPSHOT,
                sender_id = self.operator_id,
                payload   = snapshot,
            ))


