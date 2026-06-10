"""Operator — the central ADR runtime class.

Usage (subclass-based)::

    from radical.adr import Operator, act, Decision
    from radical.adr.goals import Goal

    class SearchOperator(Operator):
        center: float = 0.0
        radius: float = 10.0
        history: list = []

        @goals
        def convergence_criteria(self):
            return Goal(name="converged", metric="radius",
                        threshold=0.005, direction="minimize")

        @act
        async def evaluate(self, x: float, seed: int = 0) -> dict:
            # Local execution — returns plain dict
            import random
            rng = random.Random(seed)
            return {"x": x, "value": (x - 3.14) ** 2 + rng.gauss(0, 0.03)}

        # Or dispatch to AsyncFlow backend:
        #   @act
        #   async def evaluate(self, x: float, seed: int = 0) -> dict:
        #       return self.engine.function_task(_eval_impl)(x=x, seed=seed)

    op = SearchOperator(engine)
    async for snapshot in op.run():
        print(snapshot.cycle)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from radical.adr.decision import Action, ActionKind, Decision
from radical.adr.errors import ActionError, PolicyError
from radical.adr.message import InboxMessage, MessageKind
from radical.adr.state import RuntimeState, Snapshot
from radical.adr.tools import ActionSet

from radical.adr.goals import Goal

if TYPE_CHECKING:
    from radical.adr.observer import ObserverBase
    from radical.adr.policy.base import Policy

log = logging.getLogger(__name__)

_DRAIN_BATCH   = 64
_DRAIN_TIMEOUT = 0.05   # seconds
_MISSING       = object()  # sentinel for "no default value"

# Instance attributes that must NOT be proxied to state.objectives
_RESERVED = frozenset({
    "operator_id", "engine", "parent", "observer",
    "state", "inbox", "children",
    "_stop_event", "_adl_actionset", "_policy", "_runtime_goals", "_block_factory",
    "_max_cycles",
})


# ── Module-level decorators ───────────────────────────────────────────────────

def act(fn: Callable) -> Callable:
    """Mark a method as an ADR act — the operator's execution primitive.

    The method runs in the operator's event loop with ``self`` available.
    Return a plain value for local execution, or return an ``asyncio.Future``
    to dispatch work to an AsyncFlow backend:

    Local (runs inline)::

        @act
        async def evaluate(self, x: float, seed: int = 0) -> dict:
            import random
            rng = random.Random(seed)
            return {"x": x, "value": (x - 3.14) ** 2 + rng.gauss(0, 0.03)}

    Backend task (dispatches to engine.function_task)::

        @act
        async def evaluate(self, x: float, seed: int = 0) -> dict:
            return self.engine.function_task(_eval_impl)(x=x, seed=seed)

    Block (orchestrates sub-tasks)::

        @act
        async def pipeline(self, candidate: str) -> dict:
            async def _block(c: str) -> dict:
                prepped = await self.engine.function_task(_prep)(c)
                return await self.engine.function_task(_score)(prepped)
            return self.engine.block(_block)(candidate)

    Methods that accept ``*dep_futures`` receive resolved AsyncFlow futures
    from ``depends_on`` UIDs specified in the Action and can forward them
    directly to engine calls for native DAG chaining.
    """
    fn._adl_act = True  # type: ignore[attr-defined]
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
                Goal(name="converged",   metric="radius",      threshold=0.005),
                Goal(name="cpu_healthy", metric="cpu_percent", threshold=90.0),
            ]
    """
    fn._adl_goals = True  # type: ignore[attr-defined]
    return fn


# ── Future event delivery ─────────────────────────────────────────────────────

def _deliver_task_event(
    future: asyncio.Future, uid: str, inbox: asyncio.Queue
) -> None:
    """Synchronous done-callback: deliver a TASK_EVENT when a future settles.

    Registered via future.add_done_callback — zero coroutines, O(1) per task
    at registration, fires immediately on settlement.
    """
    if future.cancelled():
        payload: dict = {"uid": uid, "state": "CANCELED"}
    elif (exc := future.exception()) is not None:
        payload = {"uid": uid, "state": "FAILED", "error": exc}
    else:
        payload = {"uid": uid, "state": "DONE", "result": future.result()}
    inbox.put_nowait(InboxMessage(
        kind      = MessageKind.TASK_EVENT,
        sender_id = "asyncflow",
        payload   = payload,
    ))


# ── Operator ─────────────────────────────────────────────────────────────────

class Operator:
    """Autonomous runtime operator.

    Subclass to define acts, goals, and the adaptive cycle::

        class MyOp(Operator):
            best: float = float("inf")   # persists across cycles via state.objectives

            @goals
            def convergence_criteria(self):
                return Goal(name="converged", metric="best", threshold=0.01)

            @act
            async def compute(self, x: float) -> dict:
                # local execution — or return self.engine.function_task(fn)(x=x)
                return {"value": x ** 2}

            @observe
            def extract(self, snapshot) -> dict:
                results = [v for k, v in snapshot.observations.items()
                           if k.startswith("result.") and "value" in v]
                return {"best": min((r["value"] for r in results), default=float("inf"))}

        op = MyOp(engine)
        op.policy = MyPolicy(op)
        async for snapshot in op.run(): ...

    The operator ONLY: observes (done-callbacks, external subscriptions),
    reasons (``policy.decide``), and acts (``@act`` calls, ``future.cancel()``).
    It never owns scheduling, execution lifecycle, or distributed coordination.
    """

    # ── Class-level registries (populated by __init_subclass__) ──────────────

    _adl_act_registry: dict[str, Callable] = {}
    _adl_observe_fn:     Callable | None      = None
    _adl_goals_fn:       Callable | None      = None
    _adl_state_keys:     set[str]             = set()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Fresh registries per subclass (no shared mutable dicts)
        cls._adl_act_registry = {}
        cls._adl_observe_fn     = None
        cls._adl_goals_fn       = None

        for name, val in vars(cls).items():
            if callable(val) and getattr(val, "_adl_act", False):
                cls._adl_act_registry[name] = val
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
        operator_id:   str | None            = None,
        *,
        policy:        "Policy | None"       = None,
        parent:        "Operator | None"     = None,
        inbox_maxsize: int                   = 0,
        observer:      "ObserverBase | None" = None,
        max_cycles:    int | None            = None,
    ) -> None:
        object.__setattr__(self, "operator_id",  operator_id or type(self).__name__)
        object.__setattr__(self, "engine",       engine)
        object.__setattr__(self, "parent",       parent)
        object.__setattr__(self, "observer",     observer)

        object.__setattr__(self, "state",        RuntimeState(self.operator_id))
        object.__setattr__(self, "inbox",        asyncio.Queue(maxsize=inbox_maxsize))
        object.__setattr__(self, "children",     {})
        object.__setattr__(self, "_stop_event",    asyncio.Event())
        object.__setattr__(self, "_block_factory", None)
        object.__setattr__(self, "_max_cycles",    max_cycles)

        object.__setattr__(self, "_policy",        policy)
        object.__setattr__(self, "_runtime_goals", [])

        # Seed state.objectives from class-level annotated defaults
        for key in type(self)._adl_state_keys:
            for klass in type(self).__mro__:
                raw = vars(klass).get(key, _MISSING)
                if raw is not _MISSING and not callable(raw):
                    self.state.objectives[key] = raw
                    break

        # Build ActionSet
        object.__setattr__(self, "_adl_actionset", ActionSet(self))

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

    # ── Policy ───────────────────────────────────────────────────────────────

    @property
    def policy(self) -> "Policy | None":
        """Active decision policy."""
        return object.__getattribute__(self, "_policy")

    @policy.setter
    def policy(self, value: "Policy | None") -> None:
        object.__setattr__(self, "_policy", value)

    # ── Actions ───────────────────────────────────────────────────────────────

    def get_actions(self) -> "ActionSet":
        """Return the operator's ActionSet for use in policy code.

        ``get_actions()`` → ActionSet (Python callables, one per ``@act`` method
        plus built-ins: spawn_task, cancel_task, update_param, stop)
        """
        return object.__getattribute__(self, "_adl_actionset")

    def get_schema(self, format: str = "openai") -> list:
        """Render all ``@act`` methods as LLM-callable schemas.

        ``get_schema("openai")``      → list[dict] OpenAI tool spec
        ``get_schema("anthropic")``   → list[dict] Anthropic tool spec
        ``get_schema("langgraph")``   → list[BaseTool] LangGraph tools
        """
        actionset: ActionSet = object.__getattribute__(self, "_adl_actionset")
        return actionset.get_schema(format)

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
                runtime[idx] = new_goal
            else:
                runtime.append(new_goal)

    @property
    def runtime_goals(self) -> list:
        """Runtime goals currently proposed by the policy."""
        return object.__getattribute__(self, "_runtime_goals")

    @property
    def cycle_history(self) -> "list[Snapshot]":
        """Read-only list of previous cycle snapshots (oldest first, up to 64).

        Accessible inside ``@observe`` as ``self.cycle_history``. The current
        cycle's snapshot is NOT included — only completed past cycles.

        Example usage in ``@observe``::

            @observe
            def extract(self, snapshot) -> dict:
                prev   = self.cycle_history
                radii  = [s.objectives.get("radius") for s in prev[-5:]]
                stale  = len(set(radii)) == 1 and len(radii) == 5
                return {..., "stagnating": stale}
        """
        history = object.__getattribute__(self, "state").history
        # history[-1] is the current cycle's snapshot (appended by snapshot() before
        # @observe runs) — exclude it so @observe can compare past cycles to the present.
        return list(history)[:-1] if history else []

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
          3. DECIDE  — policy.decide(obs) → Decision
          4. ACT     — apply(Decision.actions) → @act calls / future tracking
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
                        "ADR has no default observation."
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

                mc = object.__getattribute__(self, "_max_cycles")
                if mc is not None and self.state.cycle >= mc:
                    self._stop_event.set()

                if decision.stop:
                    break

        finally:
            for future in self.state.task_futures.values():
                if not future.done():
                    future.cancel()

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

    def start(self, *dep_futures: asyncio.Future) -> asyncio.Future:
        """Register and start this operator as an AsyncFlow block.

        Returns a block Future that resolves to the final Snapshot when the
        operator's run loop finishes.  Passing upstream Futures as
        ``dep_futures`` chains this operator's start to their completion —
        the same native dependency model as ``engine.block`` and
        ``engine.function_task``.

        Example — sequential DAG expressed in a parent @act::

            @act
            async def launch_b(self, dep_future) -> asyncio.Future:
                return self.child_b.start(dep_future)   # B starts when dep is DONE
        """
        factory = object.__getattribute__(self, "_block_factory")
        if factory is None:
            async def _run_block(*_dep_results: Any) -> "Snapshot | None":
                return await self.run_to_completion()
            _run_block.__name__     = f"operator.{self.operator_id}"
            _run_block.__qualname__ = f"operator.{self.operator_id}"
            factory = self.engine.block(_run_block)
            object.__setattr__(self, "_block_factory", factory)
        return factory(*dep_futures)

    async def shutdown(self) -> None:
        """Signal stop and cancel outstanding task futures."""
        self._stop_event.set()
        await self.inbox.put(InboxMessage(
            kind=MessageKind.STOP, sender_id="system", payload=None
        ))

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
        self._sync_running_states()

    def _sync_running_states(self) -> None:
        """Promote futures whose AsyncFlow state is RUNNING but not yet reflected."""
        for uid, fut in self.state.task_futures.items():
            if (getattr(fut, "state", None) == "RUNNING"
                    and self.state.runtime.get(uid) == "PENDING"):
                self.state.runtime[uid] = "RUNNING"

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
                elif status == "FAILED":
                    error = payload.get("error")
                    ctx   = self.state.task_context.get(uid, {})
                    self.state.observations[f"error.{uid}"] = {
                        "task_name": ctx.get("name", "unknown"),
                        "message":   str(error) if error else "unknown",
                        "type":      type(error).__name__ if error else "unknown",
                    }

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
            for k, v in snapshot.objectives.items():
                self.state.observations[f"child:{child_id}:obj:{k}"] = v

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
        if act.kind in (ActionKind.SPAWN_TASK, ActionKind.SPAWN_BLOCK):
            if act.task_name is None:
                raise ActionError(f"{act.kind}: task_name is required")
            fn = type(self)._adl_act_registry.get(act.task_name)
            if fn is None:
                raise ActionError(
                    f"Unknown act '{act.task_name}'. "
                    f"Registered: {list(type(self)._adl_act_registry)}"
                )
            dep_futures = [
                self.state.task_futures[uid]
                for uid in act.depends_on
                if uid in self.state.task_futures
            ]
            result = await fn(self, *dep_futures, **act.task_kwargs)
            if isinstance(result, asyncio.Future):
                uid = str(getattr(result, "id", id(result)))
                self.state.task_futures[uid] = result
                self.state.runtime[uid]      = "PENDING"
                self.state.task_context[uid] = {"name": act.task_name, "kwargs": act.task_kwargs}
                result.add_done_callback(
                    lambda fut, _uid=uid: _deliver_task_event(fut, _uid, self.inbox)
                )
            elif result is not None:
                # Local result — deliver to inbox so @observe sees it next cycle
                uid = f"local.{act.task_name}.{id(result)}"
                self.state.task_context[uid] = {"name": act.task_name, "kwargs": act.task_kwargs}
                self.inbox.put_nowait(InboxMessage(
                    kind      = MessageKind.TASK_EVENT,
                    sender_id = "adl.local",
                    payload   = {"uid": uid, "state": "DONE", "result": result},
                ))

        elif act.kind == ActionKind.CANCEL_TASK:
            if act.target_uid is None:
                raise ActionError("CANCEL_TASK requires target_uid")
            future = self.state.task_futures.get(act.target_uid)
            if future is None:
                raise ActionError(f"CANCEL_TASK: unknown uid '{act.target_uid}'")
            future.cancel()

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
