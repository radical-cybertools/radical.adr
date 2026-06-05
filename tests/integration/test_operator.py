"""Integration tests for the ADL operator cycle.

All tests go through the public API only (run_to_completion, get_tools).
The engine fixture in conftest.py uses ProcessPoolExecutor.
"""

from __future__ import annotations

import asyncio

import pytest

from radical.adl import Decision, Operator, Policy, decide, goals, observe, tool
from radical.adl.goals import ScientificGoal


# ── Module-level sync impl for function_task dispatch (must be picklable) ─────

async def _double_impl(x: float) -> dict:
    return {"result": x * 2}


# ── 4a: Local tool dispatch ───────────────────────────────────────────────────

class _LocalOp(Operator):
    got: int = 0

    @goals
    def g(self):
        return ScientificGoal("done", "got", 1, "maximize")

    @tool
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
        self._t = op.get_tools()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["got"]:
            return Decision(stop=True)
        return Decision(actions=[self._t.local_fn(x=21.0)])


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
        return ScientificGoal("done", "got", 1, "maximize")

    @tool
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
        self._t = op.get_tools()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["got"]:
            return Decision(stop=True)
        return Decision(actions=[self._t.async_fn(x=21.0)])


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
        self._t = op.get_tools()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] == 0:
            return Decision(actions=[self._t.update_param("center", 42.0)])
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
        return ScientificGoal("done", "radius", 0.01, "minimize")

    @observe
    def ext(self, snap) -> dict:
        return {"radius": snap.objectives.get("radius", 1.0), "cycle": snap.cycle}


class _GoalPolicy(Policy):
    def __init__(self, op: _GoalOp) -> None:
        super().__init__()
        self._t = op.get_tools()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] >= 10:
            return Decision(stop=True)
        if obs["cycle"] == 0:
            return Decision(actions=[self._t.update_param("radius", 0.001)])
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
        child_radius = snap.observations.get("child:test-child:obj:radius")
        return {
            "child_radius": child_radius,
            "cycle": snap.cycle,
        }


class _ParentPolicy(Policy):
    def __init__(self, op: _ParentOp) -> None:
        super().__init__()
        self._t = op.get_tools()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["child_radius"] is not None:
            return Decision(actions=[
                self._t.update_param("last_child_radius", obs["child_radius"])
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
