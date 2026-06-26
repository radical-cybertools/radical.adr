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
import inspect
import logging
import time
import warnings
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from radical.adr.decision import Action, ActionKind, Decision
from radical.adr.errors import ActionError, PolicyError
from radical.adr.goals import AllGoal, AnyGoal, _GoalGroup, goal_from_dict, goal_to_dict
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
    "state", "inbox", "children", "_child_starts", "_child_config",
    "_stop_event", "_adl_actionset", "_policy", "_runtime_goals", "_block_factory",
    "_max_cycles", "_max_runtime", "_started_at", "_goal_streaks",
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
    log.debug("task uid=%s -> %s", uid, payload["state"])
    inbox.put_nowait(InboxMessage(
        kind      = MessageKind.TASK_EVENT,
        sender_id = "asyncflow",
        payload   = payload,
    ))


# ── Child handle ─────────────────────────────────────────────────────────────

class ChildHandle:
    """Live, read-only view of a spawned child operator.

    Returned by ``Operator.child(name)``; surfaced to a policy through ``@observe``
    so decisions read child state by intent — ``obs["explorer"].done``,
    ``obs["explorer"].radius`` — without touching Futures, uids, or ``child:*`` keys.

    - ``status``  → NOT_STARTED / PENDING / RUNNING / DONE / FAILED
    - ``done`` / ``failed`` → convenience booleans
    - any other attribute → the child's reported objective of that name (mirrored
      into the parent's persistent objectives under ``child:{name}:obj:{metric}`` by
      the SNAPSHOT merge, so the latest value survives across parent cycles)
    """

    def __init__(self, parent: "Operator", name: str) -> None:
        object.__setattr__(self, "_p", parent)
        object.__setattr__(self, "_n", name)

    @property
    def status(self) -> str:
        p = object.__getattribute__(self, "_p")
        n = object.__getattribute__(self, "_n")
        uid = p._child_starts.get(n)
        return p.state.runtime.get(uid, "PENDING") if uid else "NOT_STARTED"

    @property
    def started(self) -> bool:
        """True once the child has ever been launched (incl. terminal states).

        The clean status gate for fan-out policies: ``if not child.started: spawn``
        replaces a user-managed spawn-once flag.  Idempotent ``spawn_operator``
        keys off the same fact, so a policy that gates on ``started`` never hits the
        no-op warning path.
        """
        return self.status != "NOT_STARTED"

    @property
    def done(self) -> bool:
        return self.status == "DONE"

    @property
    def failed(self) -> bool:
        return self.status == "FAILED"

    def get(self, metric: str, default: Any = None) -> Any:
        p = object.__getattribute__(self, "_p")
        n = object.__getattribute__(self, "_n")
        return p.state.objectives.get(f"child:{n}:obj:{metric}", default)

    def __getattr__(self, metric: str) -> Any:
        # Only reached for names that are not real attributes/properties.
        val = self.get(metric, _MISSING)
        if val is _MISSING:
            n = object.__getattribute__(self, "_n")
            raise AttributeError(f"child '{n}' has no reported metric '{metric}'")
        return val

    def __repr__(self) -> str:
        n = object.__getattribute__(self, "_n")
        return f"ChildHandle({n!r}, status={self.status})"


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
        max_runtime:   float | None          = None,
    ) -> None:
        object.__setattr__(self, "operator_id",  operator_id or type(self).__name__)
        object.__setattr__(self, "engine",       engine)
        object.__setattr__(self, "parent",       parent)
        object.__setattr__(self, "observer",     observer)

        object.__setattr__(self, "state",        RuntimeState(self.operator_id))
        object.__setattr__(self, "inbox",        asyncio.Queue(maxsize=inbox_maxsize))
        object.__setattr__(self, "children",     {})
        object.__setattr__(self, "_child_starts", {})   # child name → start-future uid
        object.__setattr__(self, "_child_config", {})   # child name → initial config (restart)
        object.__setattr__(self, "_stop_event",    asyncio.Event())
        object.__setattr__(self, "_block_factory", None)
        object.__setattr__(self, "_max_cycles",    max_cycles)
        object.__setattr__(self, "_max_runtime",   max_runtime)   # wall-clock cap (seconds)
        object.__setattr__(self, "_started_at",    None)          # set at run() start
        object.__setattr__(self, "_goal_streaks",  {})            # goal name → consecutive-satisfied

        object.__setattr__(self, "_policy",        policy)
        object.__setattr__(self, "_runtime_goals", [])

        self._seed_default_objectives()

        # Build ActionSet
        object.__setattr__(self, "_adl_actionset", ActionSet(self))

    def _seed_default_objectives(self) -> None:
        """Seed state.objectives from class-level annotated defaults.

        Called at construction and again from the restart reset path, so a
        restarted child starts from the same declared baseline.
        """
        for key in type(self)._adl_state_keys:
            for klass in type(self).__mro__:
                raw = vars(klass).get(key, _MISSING)
                if raw is not _MISSING and not callable(raw):
                    self.state.objectives[key] = raw
                    break

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
                if not isinstance(g, (Goal, _GoalGroup)):
                    raise TypeError(
                        f"@goals method '{goals_fn.__name__}' on {type(self).__name__} "
                        f"must return Goal/AnyGoal/AllGoal (or a list), got {type(g).__name__}"
                    )
        runtime: list = object.__getattribute__(self, "_runtime_goals")
        all_goals = static + runtime
        if not all_goals:
            return

        # Evaluate the whole goal tree (no short-circuit) so every leaf goal's
        # consecutive-satisfied streak (for_cycles debounce) updates each cycle.
        streaks: dict = object.__getattribute__(self, "_goal_streaks")
        seen: set = set()
        results = [self._eval_goal(g, obs, snapshot, streaks, seen) for g in all_goals]
        for name in [k for k in streaks if k not in seen]:
            del streaks[name]                       # drop streaks for removed goals

        if all(results):
            if not decision.stop:
                log.info("cycle %d: all %d goal(s) satisfied -> stopping",
                         self.state.cycle, len(all_goals))
            decision.stop = True

    def _eval_goal(self, goal, obs: dict, snapshot: Snapshot,
                   streaks: dict, seen: set) -> bool:
        """Evaluate a goal (leaf or composite), applying for_cycles debounce.

        Uses list comprehensions (not generators) so *all* leaves are evaluated
        every cycle and their debounce streaks stay correct.
        """
        if isinstance(goal, AllGoal):
            return all([self._eval_goal(g, obs, snapshot, streaks, seen) for g in goal.goals])
        if isinstance(goal, AnyGoal):
            return any([self._eval_goal(g, obs, snapshot, streaks, seen) for g in goal.goals])
        # leaf Goal
        seen.add(goal.name)
        instant = goal.satisfied(obs, snapshot)
        streak  = streaks.get(goal.name, 0) + 1 if instant else 0
        streaks[goal.name] = streak
        return streak >= getattr(goal, "for_cycles", 1)

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
    def inflight(self) -> int:
        """Number of in-flight tasks (status PENDING or RUNNING).

        Read it in ``@observe`` (e.g. ``"inflight": self.inflight``) to let a
        policy throttle its own spawn rate, or target it with a ``Goal``. ADR
        never drops spawns — throttling is always the policy's choice.
        """
        return sum(1 for s in self.state.runtime.values()
                   if s in ("PENDING", "RUNNING"))

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

    def add_child(
        self,
        name:     str,
        operator: "Operator | type[Operator]",
        *,
        policy:   Any = None,
        **config: Any,
    ) -> "Operator":
        """Register a child operator under ``name`` (auto-wired to this parent).

        ``operator`` may be an ``Operator`` *class* — instantiated as
        ``operator(self.engine, name, parent=self)`` — or an existing instance,
        which is re-parented to ``self``.

        ``config`` seeds the child's objectives: each key is assigned on the child
        after construction (an annotated attribute like ``center`` proxies into
        ``child.state.objectives``), so a fan-out can vary per-child tuning in one
        call — ``add_child("refiner_0", SearchOperator, center=3.1)``.

        ``policy`` may be a ``Policy`` instance (attached as-is) or a class/factory
        called with the child (``child.policy = policy(child)``) — the latter lets a
        single line provision policies for a homogeneous fan-out of children.

        Returns the child operator. The child is observable via ``self.child(name)``.
        """
        if isinstance(operator, type):
            child = operator(self.engine, name, parent=self)
        else:
            child = operator
            object.__setattr__(child, "parent", self)

        for key, value in config.items():
            setattr(child, key, value)   # annotated keys proxy into child objectives

        if policy is not None:
            from radical.adr.policy.base import Policy
            child.policy = policy if isinstance(policy, Policy) else policy(child)

        self.children[name] = child
        self._child_config[name] = dict(config)   # initial config snapshot for restart
        return child

    def child(self, name: str) -> "ChildHandle":
        """Return a live, read-only :class:`ChildHandle` for a registered child."""
        return ChildHandle(self, name)

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
            self.observer.on_start(self.operator_id, {
                "objectives": self.state.objectives,
                "children":   {n: type(c).__name__ for n, c in self.children.items()},
            })

        object.__setattr__(self, "_started_at", time.monotonic())
        log.info("operator '%s' starting (max_cycles=%s, max_runtime=%s)",
                 self.operator_id, object.__getattribute__(self, "_max_cycles"),
                 object.__getattribute__(self, "_max_runtime"))

        try:
            while not self._stop_event.is_set():
                await self._drain_inbox()                           # 1. RUN
                log.debug("cycle %d: drained inbox", self.state.cycle)

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
                log.debug("cycle %d: observed %d keys", self.state.cycle, len(obs))

                try:
                    decision = await self._decide(obs)              # 3. DECIDE
                except Exception as exc:
                    raise PolicyError(
                        f"Operator '{self.operator_id}' policy raised on cycle {self.state.cycle}"
                    ) from exc

                self._check_goals(obs, snapshot, decision)
                self._apply_goal_changes(decision)

                log.info("cycle %d: decision — %d actions, %d goal proposals, stop=%s",
                         self.state.cycle, len(decision.actions),
                         len(decision.goals), decision.stop)

                if self.observer is not None:
                    self.observer.on_cycle(snapshot, obs, decision)

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

                mr = object.__getattribute__(self, "_max_runtime")
                started = object.__getattribute__(self, "_started_at")
                if mr is not None and started is not None \
                        and (time.monotonic() - started) >= mr:
                    log.info("operator '%s': max_runtime %.1fs reached -> stopping",
                             self.operator_id, mr)
                    self._stop_event.set()

                if decision.stop:
                    break

        finally:
            for future in self.state.task_futures.values():
                if not future.done():
                    future.cancel()

            final_snap = self.state.snapshot() if self.state.cycle > 0 else None
            reason = "stop" if self._stop_event.is_set() else "shutdown"
            log.info("operator '%s' stopped after %d cycles (reason=%s)",
                     self.operator_id, self.state.cycle, reason)
            if self.observer is not None:
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

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        """Serialize this operator's *steering* state to JSON for later resume.

        Persists exactly the state ADR owns — ``cycle``, ``objectives``, task
        ``runtime`` statuses, ``artifacts``, ``task_context``, runtime goals,
        goal-debounce streaks, and child lifecycle metadata. Live task Futures
        are **not** persisted (they belong to AsyncFlow/RHAPSODY — the sacred
        boundary); any task still in flight at save time is recorded as ``LOST``
        so a resumed policy can ``retry_task`` it from ``task_context``.

        Child *operator instances* are not serialized — re-declare them with
        ``add_child`` in the operator's constructor as usual; their lifecycle
        metadata (``_child_starts`` / ``_child_config``) is restored on top.
        """
        import json
        runtime = {uid: ("LOST" if st in ("PENDING", "RUNNING") else st)
                   for uid, st in self.state.runtime.items()}
        data = {
            "version":       1,
            "operator_id":   self.operator_id,
            "cycle":         self.state.cycle,
            "objectives":    self.state.objectives,
            "runtime":       runtime,
            "artifacts":     self.state.artifacts,
            "task_context":  self.state.task_context,
            "runtime_goals": [goal_to_dict(g)
                              for g in object.__getattribute__(self, "_runtime_goals")],
            "goal_streaks":  object.__getattribute__(self, "_goal_streaks"),
            "child_starts":  dict(self._child_starts),
            "child_config":  dict(self._child_config),
        }
        with open(path, "w") as f:
            json.dump(data, f, default=str, indent=2)
        log.info("operator '%s': checkpoint saved to %s (cycle=%d)",
                 self.operator_id, path, self.state.cycle)

    def load_checkpoint(self, path: str) -> None:
        """Restore steering state saved by :meth:`save_checkpoint` onto this operator.

        Call on a freshly constructed operator before ``run()``; the loop then
        resumes from the saved ``cycle`` with the saved objectives/goals. Tasks
        recorded as ``LOST`` surface in ``runtime`` for the policy to re-spawn.
        """
        import json
        with open(path) as f:
            data = json.load(f)
        self.state.cycle = data.get("cycle", 0)
        self.state.objectives.update(data.get("objectives", {}))
        self.state.runtime.update(data.get("runtime", {}))
        self.state.artifacts.update(data.get("artifacts", {}))
        self.state.task_context.update(data.get("task_context", {}))
        object.__setattr__(self, "_runtime_goals",
                           [goal_from_dict(d) for d in data.get("runtime_goals", [])])
        object.__getattribute__(self, "_goal_streaks").update(data.get("goal_streaks", {}))
        self._child_starts.update(data.get("child_starts", {}))
        self._child_config.update(data.get("child_config", {}))
        log.info("operator '%s': checkpoint loaded from %s (cycle=%d)",
                 self.operator_id, path, self.state.cycle)

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
                # Prune the now-terminal task Future so task_futures stays bounded
                # over long runs; keep the resolved value so a later depends_on can
                # still resolve. Child-operator block Futures are exempt (few; their
                # liveness drives cancel/restart semantics).
                if status in ("DONE", "FAILED", "CANCELED") \
                        and uid not in set(self._child_starts.values()):
                    self.state.task_futures.pop(uid, None)
                    self.state.remember_resolved(uid, status, payload.get("result")
                                                 if status == "DONE" else payload.get("error"))

        elif msg.kind == MessageKind.DIRECTIVE:
            payload = msg.payload
            if isinstance(payload, dict):
                self.state.directives.update(payload.get("directives", payload))

        elif msg.kind == MessageKind.SNAPSHOT:
            snapshot: Snapshot = msg.payload
            child_id = msg.sender_id
            # A child's observations are per-cycle on the child → mirror them in the
            # parent's per-cycle observations. Its runtime/objectives are *persistent*
            # child state → mirror them in the parent's persistent objectives store so
            # the latest values survive across parent cycles (latest-wins). This keeps
            # ChildHandle metric reads stable even when the child's final SNAPSHOT and
            # its block-future DONE event arrive on different parent cycles.
            for k, v in snapshot.observations.items():
                self.state.observations[f"child:{child_id}:{k}"] = v
            for k, v in snapshot.runtime.items():
                self.state.objectives[f"child:{child_id}:runtime:{k}"] = v
            for k, v in snapshot.objectives.items():
                self.state.objectives[f"child:{child_id}:obj:{k}"] = v

    # ── Action application ───────────────────────────────────────────────────

    async def _apply_actions(self, actions: list[Action]) -> None:
        # Validate the whole batch first, then apply. A single invalid action
        # aborts the batch before any action in it is applied — no partial state.
        self._validate_actions(actions)
        for act in actions:
            try:
                await self._apply_one(act)
            except Exception as exc:
                raise ActionError(
                    f"Operator '{self.operator_id}': action {act.kind} failed"
                ) from exc

    def _validate_actions(self, actions: list[Action]) -> None:
        """Pre-flight check of every action in a Decision.

        Raises a single ``ActionError`` listing *all* problems found, so one bad
        action can no longer leave earlier actions in the same Decision applied.
        Unresolvable ``depends_on`` uids warn (not raise) — they may legitimately
        reference a future on another operator's state.
        """
        registry = type(self)._adl_act_registry
        problems: list[str] = []

        for i, act in enumerate(actions):
            if act.kind in (ActionKind.SPAWN_TASK, ActionKind.SPAWN_BLOCK):
                if act.task_name is None:
                    problems.append(f"action[{i}] {act.kind.value}: task_name is required")
                    continue
                fn = registry.get(act.task_name)
                if fn is None:
                    problems.append(
                        f"action[{i}]: unknown task '{act.task_name}'; "
                        f"registered: {list(registry)}"
                    )
                    continue
                self._validate_spawn_params(i, act, fn, problems)

            elif act.kind == ActionKind.SPAWN_OPERATOR:
                name = act.task_name
                if not name:
                    problems.append(f"action[{i}] SPAWN_OPERATOR: task_name (child name) is required")
                elif name not in self.children and act.operator_cls is None:
                    problems.append(
                        f"action[{i}] SPAWN_OPERATOR '{name}': child not registered and no "
                        "operator_cls provided"
                    )
                # depends_on holds child NAMES here; an unknown name may be spawned earlier
                # in the same batch, so warn (do not raise) rather than block.
                for dep_name in act.depends_on:
                    if dep_name not in self.children and dep_name not in self._child_starts:
                        warnings.warn(
                            f"Operator '{self.operator_id}': action[{i}] SPAWN_OPERATOR '{name}' "
                            f"depends_on child '{dep_name}' which is not a known child; "
                            "the dependency will be dropped if still unknown at apply time.",
                            UserWarning,
                            stacklevel=2,
                        )

            elif act.kind == ActionKind.RESTART_OPERATOR:
                name = act.task_name
                if not name:
                    problems.append(
                        f"action[{i}] RESTART_OPERATOR: task_name (child name) is required"
                    )
                elif name not in self.children:
                    problems.append(
                        f"action[{i}] RESTART_OPERATOR '{name}': unknown child; spawn it first"
                    )
                # depends_on holds child NAMES; unknown name → warn (do not raise).
                for dep_name in act.depends_on:
                    if dep_name not in self.children and dep_name not in self._child_starts:
                        warnings.warn(
                            f"Operator '{self.operator_id}': action[{i}] RESTART_OPERATOR "
                            f"'{name}' depends_on child '{dep_name}' which is not a known "
                            "child; the dependency will be dropped if still unknown at "
                            "apply time.",
                            UserWarning,
                            stacklevel=2,
                        )

            elif act.kind == ActionKind.CANCEL_OPERATOR:
                # Cancel is safe / idempotent: an unknown or unstarted name does not
                # raise — the no-op is logged at apply time.
                if not act.task_name:
                    problems.append(
                        f"action[{i}] CANCEL_OPERATOR: task_name (child name) is required"
                    )

            elif act.kind == ActionKind.CANCEL_TASK:
                if act.target_uid is None:
                    problems.append(f"action[{i}] CANCEL_TASK: target_uid is required")
                elif act.target_uid not in self.state.task_futures \
                        and act.target_uid not in self.state.resolved:
                    problems.append(
                        f"action[{i}] CANCEL_TASK: unknown uid '{act.target_uid}'"
                    )

            elif act.kind == ActionKind.RETRY_TASK:
                if act.target_uid is None:
                    problems.append(f"action[{i}] RETRY_TASK: target_uid is required")
                elif act.target_uid not in self.state.task_context:
                    problems.append(
                        f"action[{i}] RETRY_TASK: unknown uid '{act.target_uid}' "
                        "(no recorded task to retry)"
                    )

            elif act.kind == ActionKind.UPDATE_PARAM:
                # param_key only — objectives deliberately accept arbitrary dynamic keys.
                if act.param_key is None:
                    problems.append(f"action[{i}] UPDATE_PARAM: param_key is required")

        if problems:
            raise ActionError(
                f"Operator '{self.operator_id}': {len(problems)} action problem(s):\n  - "
                + "\n  - ".join(problems)
            )

    def _validate_spawn_params(
        self, i: int, act: Action, fn: Callable, problems: list[str]
    ) -> None:
        try:
            params = [
                (name, p)
                for name, p in inspect.signature(fn).parameters.items()
                if name != "self"
            ]
        except (ValueError, TypeError):
            return   # un-introspectable signature → skip param checks

        has_var_kw = any(p.kind == p.VAR_KEYWORD for _, p in params)
        named = [
            (name, p) for name, p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        valid_names = {name for name, _ in named}

        # Unknown kwargs (unless the act takes **kwargs)
        if not has_var_kw:
            for k in act.task_kwargs:
                if k not in valid_names:
                    problems.append(
                        f"action[{i}] '{act.task_name}': unknown kwarg '{k}'; "
                        f"accepts {sorted(valid_names)}"
                    )

        # Declared depends_on futures fill leading positional params. Use the
        # declared count (intent), not the resolvable count — an unresolvable dep
        # is reported once, as a warning below, not doubled as a missing-param error.
        positional = [
            name for name, p in named
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        filled_by_deps = set(positional[:len(act.depends_on)])

        for name, p in named:
            if p.default is inspect.Parameter.empty and name not in filled_by_deps \
                    and name not in act.task_kwargs:
                problems.append(
                    f"action[{i}] '{act.task_name}': missing required parameter '{name}'"
                )

        # Unresolvable depends_on → warn, do not raise. A pruned-but-cached
        # terminal task (state.resolved) is still resolvable, so don't warn on it.
        for uid in act.depends_on:
            if uid not in self.state.task_futures and uid not in self.state.resolved:
                warnings.warn(
                    f"Operator '{self.operator_id}': action[{i}] '{act.task_name}' "
                    f"depends_on '{uid}' which is not a known task future on this "
                    "operator; it will be dropped at apply time.",
                    UserWarning,
                    stacklevel=2,
                )

    def _track_future(self, result: asyncio.Future, name: str, kwargs: dict) -> None:
        """Register a spawned Future for runtime tracking (shared by tasks and operators)."""
        uid = str(getattr(result, "id", id(result)))
        self.state.task_futures[uid] = result
        self.state.runtime[uid]      = "PENDING"
        self.state.task_context[uid] = {"name": name, "kwargs": kwargs}
        log.debug("spawned '%s' uid=%s", name, uid)
        result.add_done_callback(
            lambda fut, _uid=uid: _deliver_task_event(fut, _uid, self.inbox)
        )

    def _resolve_task_deps(self, uids: list[str]) -> list[asyncio.Future]:
        """Resolve depends_on task UIDs to Futures (live or reconstructed).

        A still-live Future is passed directly. A terminal task whose Future was
        pruned is reconstructed from ``state.resolved`` as a completed Future, so
        DAG chaining survives pruning. Unknown/evicted UIDs are skipped (already
        warned at validation).
        """
        loop = asyncio.get_event_loop()
        deps: list[asyncio.Future] = []
        for uid in uids:
            live = self.state.task_futures.get(uid)
            if live is not None:
                deps.append(live)
                continue
            cached = self.state.resolved.get(uid)
            if cached is not None:
                status, value = cached
                fut = loop.create_future()
                if status == "DONE":
                    fut.set_result(value)
                elif status == "FAILED":
                    fut.set_exception(value if isinstance(value, BaseException)
                                      else Exception(str(value)))
                else:  # CANCELED
                    fut.cancel()
                deps.append(fut)
        return deps

    async def _spawn_act_result(self, name: str, kwargs: dict) -> None:
        """Invoke a registered @act and route its return (shared by spawn/retry)."""
        result = await type(self)._adl_act_registry[name](self, **kwargs)
        if isinstance(result, asyncio.Future):
            self._track_future(result, name, kwargs)
        elif result is not None:
            uid = f"local.{name}.{id(result)}"
            self.state.task_context[uid] = {"name": name, "kwargs": kwargs}
            self.inbox.put_nowait(InboxMessage(
                kind      = MessageKind.TASK_EVENT,
                sender_id = "adl.local",
                payload   = {"uid": uid, "state": "DONE", "result": result},
            ))

    async def _retry_task(self, act: Action) -> None:
        """Re-spawn a finished task from its recorded name + kwargs."""
        uid = act.target_uid
        ctx = self.state.task_context.get(uid) if uid else None
        if ctx is None:
            raise ActionError(f"RETRY_TASK: unknown uid '{uid}'")
        name = ctx["name"]
        if name not in type(self)._adl_act_registry:
            raise ActionError(f"RETRY_TASK '{uid}': '{name}' is not a registered act")
        log.info("operator '%s': retrying task '%s' (was uid=%s)",
                 self.operator_id, name, uid)
        await self._spawn_act_result(name, ctx.get("kwargs", {}))

    async def _apply_one(self, act: Action) -> None:
        log.debug("cycle %d: apply %s '%s'",
                  self.state.cycle, act.kind, act.task_name or act.target_uid or act.param_key)
        if act.kind in (ActionKind.SPAWN_TASK, ActionKind.SPAWN_BLOCK):
            if act.task_name is None:
                raise ActionError(f"{act.kind}: task_name is required")
            fn = type(self)._adl_act_registry.get(act.task_name)
            if fn is None:
                raise ActionError(
                    f"Unknown act '{act.task_name}'. "
                    f"Registered: {list(type(self)._adl_act_registry)}"
                )
            dep_futures = self._resolve_task_deps(act.depends_on)
            result = await fn(self, *dep_futures, **act.task_kwargs)
            if isinstance(result, asyncio.Future):
                self._track_future(result, act.task_name, act.task_kwargs)
            elif result is not None:
                # Local result — deliver to inbox so @observe sees it next cycle
                uid = f"local.{act.task_name}.{id(result)}"
                self.state.task_context[uid] = {"name": act.task_name, "kwargs": act.task_kwargs}
                self.inbox.put_nowait(InboxMessage(
                    kind      = MessageKind.TASK_EVENT,
                    sender_id = "adl.local",
                    payload   = {"uid": uid, "state": "DONE", "result": result},
                ))

        elif act.kind == ActionKind.SPAWN_OPERATOR:
            result = await self._spawn_operator(act)
            if result is not None:   # None → idempotent no-op (already started)
                self._track_future(result, act.task_name, act.task_kwargs)

        elif act.kind == ActionKind.RESTART_OPERATOR:
            result = await self._restart_operator(act)
            self._track_future(result, act.task_name, {})

        elif act.kind == ActionKind.CANCEL_OPERATOR:
            await self._cancel_operator(act)

        elif act.kind == ActionKind.CANCEL_TASK:
            if act.target_uid is None:
                raise ActionError("CANCEL_TASK requires target_uid")
            future = self.state.task_futures.get(act.target_uid)
            if future is None:
                if act.target_uid in self.state.resolved:
                    return   # already terminal (pruned) — cancel is a no-op
                raise ActionError(f"CANCEL_TASK: unknown uid '{act.target_uid}'")
            future.cancel()

        elif act.kind == ActionKind.RETRY_TASK:
            await self._retry_task(act)

        elif act.kind == ActionKind.UPDATE_PARAM:
            if act.param_key is None:
                raise ActionError("UPDATE_PARAM requires param_key")
            self.state.objectives[act.param_key] = act.param_value

    async def _spawn_operator(self, act: Action) -> asyncio.Future | None:
        """Launch a child operator as a block, registering it on first use.

        ``act.task_name`` is the child name; ``act.depends_on`` lists upstream child
        **names** whose start-Futures gate this launch (resolved internally — the
        policy never handles Futures or uids). Returns the child's block Future, or
        ``None`` when the child was already started (idempotent no-op).
        """
        name = act.task_name
        if not name:
            raise ActionError("SPAWN_OPERATOR: task_name (child name) is required")

        # Idempotent: once a child has ever been started (including terminal DONE/
        # FAILED/CANCELED), re-emitting spawn_operator is a no-op. Re-running is
        # restart_operator's job. Well-written policies gate on `child.started`, so
        # this warning fires only on a genuine double-spawn — replacing the silent
        # "N overlapping runs" bug a re-emit used to cause.
        if name in self._child_starts:
            log.warning(
                "Operator '%s': spawn_operator '%s' is a no-op — child already "
                "started (status=%s); use restart_operator('%s') to re-run.",
                self.operator_id, name, self.child(name).status, name,
            )
            return None

        child = self.children.get(name)
        if child is None:
            if act.operator_cls is None:
                raise ActionError(
                    f"SPAWN_OPERATOR '{name}': child is not registered and no "
                    "operator_cls was provided to spawn_operator(...)"
                )
            child = self.add_child(name, act.operator_cls, policy=act.policy,
                                   **act.task_kwargs)

        log.info("operator '%s': spawning child operator '%s'", self.operator_id, name)
        fut = child.start(*self._resolve_child_deps(act.depends_on))
        self._child_starts[name] = str(getattr(fut, "id", id(fut)))
        return fut

    def _resolve_child_deps(self, dep_names: list[str]) -> list[asyncio.Future]:
        """Map upstream child **names** to their live start-Futures (skip unknown)."""
        dep_futures = []
        for dep_name in dep_names:
            dep_uid = self._child_starts.get(dep_name)
            dep_fut = self.state.task_futures.get(dep_uid) if dep_uid else None
            if dep_fut is not None:
                dep_futures.append(dep_fut)
        return dep_futures

    def _reset_run_state(self, name: str) -> None:
        """Wipe a child's run state back to its initial declared config.

        Swaps a fresh :class:`RuntimeState` onto the child, re-seeds annotated
        defaults, re-applies the stored ``add_child`` config, clears the stop event,
        and drops the cached block factory so the next ``start()`` rebuilds it.
        Runtime/observations/cycle are wiped; the seeded config (e.g. ``center``) is
        preserved — restart replays the *initial* config, not evolved state.
        """
        child = self.children[name]
        object.__setattr__(child, "state", RuntimeState(child.operator_id))
        child._seed_default_objectives()
        for key, value in self._child_config.get(name, {}).items():
            setattr(child, key, value)
        child._stop_event.clear()
        object.__setattr__(child, "_block_factory", None)

    async def _restart_operator(self, act: Action) -> asyncio.Future:
        """Cancel any in-flight run of a registered child and re-launch it fresh.

        ``act.task_name`` is the child name (must already be registered);
        ``act.depends_on`` lists upstream child names to gate the new launch.
        Returns the child's new block Future.
        """
        name = act.task_name
        if not name:
            raise ActionError("RESTART_OPERATOR: task_name (child name) is required")
        child = self.children.get(name)
        if child is None:
            raise ActionError(
                f"RESTART_OPERATOR '{name}': unknown child; spawn it first"
            )

        # Cancel the in-flight run if still live.
        old_uid = self._child_starts.get(name)
        old_fut = self.state.task_futures.get(old_uid) if old_uid else None
        if old_fut is not None and not old_fut.done():
            old_fut.cancel()

        self._reset_run_state(name)
        self._child_starts.pop(name, None)

        log.info("operator '%s': restarting child operator '%s'", self.operator_id, name)
        fut = child.start(*self._resolve_child_deps(act.depends_on))
        self._child_starts[name] = str(getattr(fut, "id", id(fut)))
        return fut

    async def _cancel_operator(self, act: Action) -> None:
        """Cancel a child's in-flight block Future, leaving a CANCELED tombstone.

        ``name`` stays in ``_child_starts`` so a later ``spawn_operator`` is still a
        no-op; re-running after cancel is ``restart_operator``'s job. Cancelling an
        unstarted (or already-finished) child is a logged no-op — cancel is safe.
        """
        name = act.task_name
        if not name:
            raise ActionError("CANCEL_OPERATOR: task_name (child name) is required")

        uid = self._child_starts.get(name)
        fut = self.state.task_futures.get(uid) if uid else None
        if fut is None:
            log.warning(
                "Operator '%s': cancel_operator '%s' is a no-op — child not started.",
                self.operator_id, name,
            )
            return
        if not fut.done():
            log.info("operator '%s': cancelling child operator '%s'", self.operator_id, name)
            fut.cancel()   # done-callback delivers CANCELED

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
