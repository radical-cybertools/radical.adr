"""Integration tests for the ADR operator cycle.

All tests go through the public API only (run_to_completion, get_actions).
The engine fixture in conftest.py uses ProcessPoolExecutor.
"""

from __future__ import annotations

import asyncio

import pytest

from radical.adr import Decision, Operator, Policy, decide, goals, observe, act
from radical.adr.errors import ActionError, PolicyError
from radical.adr.goals import Goal


# ── Module-level sync impl for function_task dispatch (must be picklable) ─────

async def _double_impl(x: float) -> dict:
    return {"result": x * 2}


# ── 4a: Local tool dispatch ───────────────────────────────────────────────────

class _LocalOp(Operator):
    got: int = 0

    @goals
    def g(self):
        return Goal("done", "got", 1, "maximize")

    @act
    async def local_fn(self, x: float) -> dict:
        return {"result": x * 2}

    @observe
    def ext(self, snap) -> dict:
        results = [v for k, v in snap.observations.items()
                   if k.startswith("result.") and isinstance(v, dict)]
        return {"got": 1 if results else 0}


class _LocalPolicy(Policy):
    def __init__(self, op: _LocalOp) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["got"]:
            return Decision(stop=True)
        return Decision(actions=[self._act.local_fn(x=21.0)])


async def test_local_tool_dispatch(engine):
    op = _LocalOp(engine)
    op.policy = _LocalPolicy(op)
    snap = await op.run_to_completion()
    assert snap is not None
    assert snap.cycle >= 1


# ── 4b: Async tool dispatch via function_task ─────────────────────────────────

class _AsyncOp(Operator):
    got: int = 0

    @goals
    def g(self):
        return Goal("done", "got", 1, "maximize")

    @act
    async def async_fn(self, x: float) -> asyncio.Future:
        return self.engine.function_task(_double_impl)(x=x)

    @observe
    def ext(self, snap) -> dict:
        results = [v for k, v in snap.observations.items()
                   if k.startswith("result.") and isinstance(v, dict)]
        return {"got": 1 if results else 0}


class _AsyncPolicy(Policy):
    def __init__(self, op: _AsyncOp) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["got"]:
            return Decision(stop=True)
        return Decision(actions=[self._act.async_fn(x=21.0)])


async def test_async_tool_dispatch(engine):
    op = _AsyncOp(engine)
    op.policy = _AsyncPolicy(op)
    snap = await op.run_to_completion()
    assert snap is not None
    assert snap.cycle >= 1


# ── 4c: update_param writes to objectives ────────────────────────────────────

class _UpdateOp(Operator):
    center: float = 0.0

    @observe
    def ext(self, snap) -> dict:
        return {"cycle": snap.cycle}


class _UpdatePolicy(Policy):
    def __init__(self, op: _UpdateOp) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] == 0:
            return Decision(actions=[self._act.update_param("center", 42.0)])
        return Decision(stop=True)


async def test_update_param_writes_objectives(engine):
    op = _UpdateOp(engine)
    op.policy = _UpdatePolicy(op)
    await op.run_to_completion()
    assert op.state.objectives["center"] == 42.0


# ── 4d: Goal convergence stops the operator ───────────────────────────────────

class _GoalOp(Operator):
    radius: float = 1.0

    @goals
    def g(self):
        return Goal("done", "radius", 0.01, "minimize")

    @observe
    def ext(self, snap) -> dict:
        return {"radius": snap.objectives.get("radius", 1.0), "cycle": snap.cycle}


class _GoalPolicy(Policy):
    def __init__(self, op: _GoalOp) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] >= 10:
            return Decision(stop=True)
        if obs["cycle"] == 0:
            return Decision(actions=[self._act.update_param("radius", 0.001)])
        return Decision()


async def test_goal_convergence_stops_operator(engine):
    op = _GoalOp(engine)
    op.policy = _GoalPolicy(op)
    snap = await op.run_to_completion()
    assert snap is not None
    # Operator stopped because goal was satisfied, not because cycle limit hit
    assert op.state.objectives.get("radius", 1.0) < 0.01


# ── 4e: SNAPSHOT — child objectives visible to parent ────────────────────────

class _ChildOp(Operator):
    radius: float = 0.5

    @observe
    def ext(self, snap) -> dict:
        return {"cycle": snap.cycle}


class _ChildPolicy(Policy):
    @decide
    async def run(self, obs: dict) -> Decision:
        # Run two cycles then stop so parent has time to receive the SNAPSHOT
        if obs["cycle"] >= 1:
            return Decision(stop=True)
        return Decision()


class _ParentOp(Operator):
    last_child_radius: float = 0.0

    @observe
    def ext(self, snap) -> dict:
        child_radius = snap.objectives.get("child:test-child:obj:radius")
        return {
            "child_radius": child_radius,
            "cycle": snap.cycle,
        }


class _ParentPolicy(Policy):
    def __init__(self, op: _ParentOp) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["child_radius"] is not None:
            return Decision(actions=[
                self._act.update_param("last_child_radius", obs["child_radius"])
            ], stop=True)
        if obs["cycle"] >= 8:  # safety stop
            return Decision(stop=True)
        return Decision()


async def test_snapshot_child_objectives_visible_to_parent(engine):
    parent = _ParentOp(engine, "test-parent")
    parent.policy = _ParentPolicy(parent)

    child = _ChildOp(engine, "test-child", parent=parent)
    child.policy = _ChildPolicy()

    await asyncio.gather(parent.run_to_completion(), child.run_to_completion())

    assert parent.state.objectives.get("last_child_radius") == pytest.approx(0.5)


# ── G3: max_cycles safety valve ───────────────────────────────────────────────

class _NoCycleOp(Operator):
    @observe
    def extract(self, snapshot) -> dict:
        return {"cycle": snapshot.cycle}


class _NoCyclePolicy(Policy):
    @decide
    async def run(self, obs: dict) -> Decision:
        return Decision()


async def test_max_cycles_stops_operator(engine):
    op = _NoCycleOp(engine, max_cycles=3)
    op.policy = _NoCyclePolicy()
    await op.run_to_completion()
    assert op.state.cycle == 3


# ── G1: cycle_history visible in @observe ─────────────────────────────────────

class _HistOp(Operator):
    hist_len: int = -1

    @observe
    def extract(self, snapshot) -> dict:
        return {"hist_len": len(self.cycle_history)}


class _HistPolicy(Policy):
    def __init__(self, op):
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        return Decision(actions=[self._act.update_param("hist_len", obs["hist_len"])])


async def test_cycle_history_grows_across_cycles(engine):
    op = _HistOp(engine, max_cycles=4)
    op.policy = _HistPolicy(op)
    await op.run_to_completion()
    # After 4 cycles the history should have grown: final recorded hist_len == 3
    assert op.hist_len == 3


# ── G2: task_context visible in snapshot ──────────────────────────────────────

class _CtxOp(Operator):
    found_ctx: dict = {}

    @act
    async def compute(self, x: float) -> dict:
        return {"val": x * 2}

    @observe
    def extract(self, snapshot) -> dict:
        # On cycle 1 the task spawned in cycle 0 is done; task_context persists
        for uid, ctx in snapshot.task_context.items():
            if ctx.get("name") == "compute":
                return {"ctx": ctx, "cycle": snapshot.cycle}
        return {"ctx": {}, "cycle": snapshot.cycle}


class _CtxPolicy(Policy):
    def __init__(self, op):
        super().__init__()
        self._act = op.get_actions()
        self._spawned = False

    @decide
    async def run(self, obs: dict) -> Decision:
        if not self._spawned:
            self._spawned = True
            return Decision(actions=[
                self._act.compute(x=42.0),
            ])
        ctx = obs.get("ctx", {})
        if ctx:
            return Decision(
                actions=[self._act.update_param("found_ctx", ctx)],
                stop=True,
            )
        return Decision()


async def test_task_context_in_snapshot(engine):
    op = _CtxOp(engine)
    op.policy = _CtxPolicy(op)
    await op.run_to_completion()
    assert op.found_ctx.get("name") == "compute"
    assert op.found_ctx.get("kwargs") == {"x": 42.0}


# ── G4: failed task delivers structured error to @observe ─────────────────────

async def _failing_impl(**kwargs) -> dict:
    raise ValueError("deliberate test failure")


class _ErrOp(Operator):
    error_info: dict = {}

    @act
    async def failing_act(self, x: float) -> dict:
        from radical.asyncflow import WorkflowEngine
        return self.engine.function_task(_failing_impl)(x=x)

    @observe
    def extract(self, snapshot) -> dict:
        errors = {
            k.removeprefix("error."): v
            for k, v in snapshot.observations.items()
            if k.startswith("error.")
        }
        return {"errors": errors, "cycle": snapshot.cycle}


class _ErrPolicy(Policy):
    def __init__(self, op):
        super().__init__()
        self._act = op.get_actions()
        self._spawned = False

    @decide
    async def run(self, obs: dict) -> Decision:
        if not self._spawned:
            self._spawned = True
            return Decision(actions=[self._act.failing_act(x=1.0)])
        if obs["errors"]:
            info = next(iter(obs["errors"].values()))
            return Decision(
                actions=[self._act.update_param("error_info", info)],
                stop=True,
            )
        return Decision()


async def test_failed_task_error_in_observations(engine):
    op = _ErrOp(engine)
    op.policy = _ErrPolicy(op)
    await op.run_to_completion()
    assert op.error_info.get("task_name") == "failing_act"
    assert op.error_info.get("type") == "ValueError"
    assert "deliberate test failure" in op.error_info.get("message", "")


# ── Item 3: observer.on_cycle receives the obs dict ───────────────────────────

class _ObsCaptureOp(Operator):
    @observe
    def extract(self, snapshot) -> dict:
        return {"cycle": snapshot.cycle, "marker": snapshot.cycle * 10}


class _ObsCapturePolicy(Policy):
    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] >= 2:
            return Decision(stop=True)
        return Decision()


class _CapturingObserver:
    def __init__(self) -> None:
        self.cycles: list = []

    def on_start(self, operator_id, metadata) -> None:
        pass

    def on_cycle(self, snapshot, obs, decision) -> None:
        self.cycles.append((snapshot, obs, decision))

    def on_stop(self, final, reason) -> None:
        pass


async def test_observer_on_cycle_receives_obs(engine):
    obs_rec = _CapturingObserver()
    op = _ObsCaptureOp(engine, observer=obs_rec)
    op.policy = _ObsCapturePolicy()
    await op.run_to_completion()

    assert obs_rec.cycles, "observer never saw a cycle"
    for snapshot, obs, decision in obs_rec.cycles:
        assert isinstance(obs, dict)
        # obs is exactly what @observe returned for that cycle
        assert obs["cycle"] == snapshot.cycle
        assert obs["marker"] == snapshot.cycle * 10


# ── Item 5: validate-all-then-apply-all prevents partial application ──────────

class _PartialOp(Operator):
    flag: int = 0

    @act
    async def real_task(self, x: float) -> dict:
        return {"x": x}

    @observe
    def extract(self, snapshot) -> dict:
        return {"cycle": snapshot.cycle}


class _PartialPolicy(Policy):
    """Emits a valid update_param followed by an invalid spawn (bad task_name)."""

    def __init__(self, op) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        return Decision(actions=[
            self._act.update_param("flag", 1),       # valid — must NOT be applied
            self._act.spawn_task("does_not_exist"),  # invalid — aborts the batch
        ])


async def test_invalid_action_prevents_partial_application(engine):
    op = _PartialOp(engine)
    op.policy = _PartialPolicy(op)
    with pytest.raises((ActionError, PolicyError)):
        await op.run_to_completion()
    # The valid update_param that preceded the invalid action must NOT have applied.
    assert op.state.objectives.get("flag", 0) == 0


class _ValidBatchPolicy(Policy):
    def __init__(self, op) -> None:
        super().__init__()
        self._act = op.get_actions()
        self._done = False

    @decide
    async def run(self, obs: dict) -> Decision:
        if self._done:
            return Decision(stop=True)
        self._done = True
        return Decision(actions=[self._act.update_param("flag", 7)])


async def test_valid_actions_still_apply(engine):
    op = _PartialOp(engine)
    op.policy = _ValidBatchPolicy(op)
    await op.run_to_completion()
    assert op.state.objectives.get("flag") == 7
