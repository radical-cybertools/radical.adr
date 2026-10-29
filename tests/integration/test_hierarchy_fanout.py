"""Integration test for adaptive fan-out across child operators.

A parent runs its own ODA loop and spawns children via the first-class
SPAWN_OPERATOR surface only — no Futures, uids, or artifacts in user code:

    explorer
     ├── refiner_0  depends_on=explorer
     ├── refiner_1  depends_on=explorer
     └── merge      depends_on=[refiner_0, refiner_1]

Asserts the native AsyncFlow gating holds: refiners reach DONE only after the
explorer, and merge only after both refiners — observed purely through
ChildHandle.done, with no artifact/uid bookkeeping anywhere.
"""

from __future__ import annotations

import pytest

from radical.adr import Decision, Operator, Policy, act, decide, observe

K = 2


# ── Module-level task impl (picklable for function_task dispatch) ─────────────

async def _quick_impl(x: float) -> dict:
    return {"x": x, "value": x}


# ── Child operator (reused for explorer, refiners, merge) ────────────────────

class _FanChild(Operator):
    @act
    async def evaluate(self, x: float):
        return self.engine.function_task(_quick_impl)(x=x)

    @observe
    def ext(self, snap) -> dict:
        have = any(k.startswith("result.") for k in snap.observations)
        return {"cycle": snap.cycle, "have": have}


class _FanChildPolicy(Policy):
    def __init__(self, op: _FanChild) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["have"]:
            return Decision(stop=True)
        if obs["cycle"] == 0:
            return Decision(actions=[self._act.evaluate(x=1.0)])
        # No cycle cap: evaluate waits for a new worker process to start, which can
        # outlast a fixed number of idle cycles. The parent's max_runtime bounds it.
        return Decision()


# ── Parent operator ───────────────────────────────────────────────────────────

class _FanParent(Operator):
    def __init__(self, engine, operator_id=None, **kw):
        super().__init__(engine, operator_id, **kw)
        self.add_child("explorer", _FanChild, policy=_FanChildPolicy)

    @observe
    def watch(self, snap) -> dict:
        obs = {
            "cycle":    snap.cycle,
            "explorer": self.child("explorer"),
            "merge":    self.child("merge"),
        }
        for i in range(K):
            obs[f"refiner_{i}"] = self.child(f"refiner_{i}")
        return obs


class _FanParentPolicy(Policy):
    def __init__(self, op: _FanParent) -> None:
        super().__init__()
        self._act = op.get_actions()
        self.done_cycle: dict[str, int] = {}   # child name → first cycle seen DONE

    @decide
    async def run(self, obs: dict) -> Decision:
        # Status-gated, no user-managed spawn-once flags: each spawn is emitted
        # exactly once because we gate on the child's own observed `.started`.
        names = ["explorer", *[f"refiner_{i}" for i in range(K)], "merge"]
        for nm in names:
            if obs[nm].done and nm not in self.done_cycle:
                self.done_cycle[nm] = obs["cycle"]

        if not obs["explorer"].started:
            return Decision(actions=[self._act.spawn_operator("explorer")])

        if obs["explorer"].done and not obs["refiner_0"].started:
            return Decision(actions=[
                self._act.spawn_operator(f"refiner_{i}", _FanChild,
                                         policy=_FanChildPolicy, depends_on="explorer")
                for i in range(K)
            ])

        refiners_done = all(obs[f"refiner_{i}"].done for i in range(K))
        if obs["refiner_0"].started and refiners_done and not obs["merge"].started:
            return Decision(actions=[
                self._act.spawn_operator("merge", _FanChild, policy=_FanChildPolicy,
                                         depends_on=[f"refiner_{i}" for i in range(K)]),
            ])

        if obs["merge"].done:
            return Decision(stop=True)
        return Decision()


async def test_fanout_dependency_ordering(engine):
    parent = _FanParent(engine, max_runtime=30.0)    # bounds a regression
    parent.policy = _FanParentPolicy(parent)
    await parent.run_to_completion()

    dc = parent.policy.done_cycle

    # Every child completed.
    assert "explorer" in dc
    assert "merge" in dc
    for i in range(K):
        assert f"refiner_{i}" in dc

    # Native AsyncFlow gating: refiners only after explorer; merge only after both.
    for i in range(K):
        assert dc[f"refiner_{i}"] > dc["explorer"], "refiner finished before explorer"
        assert dc["merge"] > dc[f"refiner_{i}"], "merge finished before a refiner"

    # No runtime-machinery bookkeeping leaked into the parent: the first-class API
    # never touches artifacts; all launches are tracked by name in _child_starts.
    assert parent.state.artifacts == {}
    assert set(parent._child_starts) == {"explorer", "refiner_0", "refiner_1", "merge"}


# ── restart_operator ──────────────────────────────────────────────────────────

class _RestartParent(Operator):
    def __init__(self, engine, operator_id=None, **kw):
        super().__init__(engine, operator_id, **kw)
        self.add_child("worker", _FanChild, policy=_FanChildPolicy)

    @observe
    def watch(self, snap) -> dict:
        return {"cycle": snap.cycle, "worker": self.child("worker")}


class _RestartPolicy(Policy):
    def __init__(self, op: _RestartParent) -> None:
        super().__init__()
        self._act = op.get_actions()
        self._op  = op
        self.restarted = False
        self.run_uids: list[str] = []   # distinct start uids observed at DONE

    @decide
    async def run(self, obs: dict) -> Decision:
        worker = obs["worker"]
        if not worker.started:
            return Decision(actions=[self._act.spawn_operator("worker")])

        if worker.done:
            uid = self._op._child_starts.get("worker")
            if uid not in self.run_uids:
                self.run_uids.append(uid)
            if not self.restarted:
                self.restarted = True
                return Decision(actions=[self._act.restart_operator("worker")])
            return Decision(stop=True)
        return Decision()


async def test_restart_operator_runs_child_again(engine):
    parent = _RestartParent(engine, max_runtime=30.0)    # bounds a regression
    parent.policy = _RestartPolicy(parent)
    await parent.run_to_completion()

    # The child ran twice — two distinct start-Futures under the same name.
    uids = parent.policy.run_uids
    assert len(uids) == 2, f"expected 2 runs, saw {uids}"
    assert uids[0] != uids[1]
    assert parent.policy.restarted


# ── cancel_operator ───────────────────────────────────────────────────────────

async def _idle_impl() -> dict:
    import asyncio as _a
    await _a.sleep(30)        # long enough to be cancelled mid-flight
    return {"done": True}


class _SlowChild(Operator):
    @act
    async def block(self):
        return self.engine.function_task(_idle_impl)()

    @observe
    def ext(self, snap) -> dict:
        return {"cycle": snap.cycle}


class _SlowChildPolicy(Policy):
    def __init__(self, op: _SlowChild) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] == 0:
            return Decision(actions=[self._act.block()])
        return Decision()        # idle forever (until cancelled from the parent)


class _CancelParent(Operator):
    def __init__(self, engine, operator_id=None, **kw):
        super().__init__(engine, operator_id, **kw)
        self.add_child("slow", _SlowChild, policy=_SlowChildPolicy)

    @observe
    def watch(self, snap) -> dict:
        return {"cycle": snap.cycle, "slow": self.child("slow")}


class _CancelPolicy(Policy):
    def __init__(self, op: _CancelParent) -> None:
        super().__init__()
        self._act = op.get_actions()
        self.cancelled = False
        self.final_status: str | None = None

    @decide
    async def run(self, obs: dict) -> Decision:
        slow = obs["slow"]
        if not slow.started:
            return Decision(actions=[self._act.spawn_operator("slow")])

        if not self.cancelled and obs["cycle"] >= 3:
            self.cancelled = True
            return Decision(actions=[self._act.cancel_operator("slow")])

        if self.cancelled and slow.status in ("CANCELED", "FAILED"):
            self.final_status = slow.status
            return Decision(stop=True)

        if obs["cycle"] >= 60:        # safety stop
            self.final_status = slow.status
            return Decision(stop=True)
        return Decision()


async def test_cancel_operator_leaves_canceled_tombstone(engine):
    parent = _CancelParent(engine)
    parent.policy = _CancelPolicy(parent)
    await parent.run_to_completion()

    assert parent.policy.final_status == "CANCELED"
    # Tombstone: the name stays in _child_starts so a later spawn would no-op.
    assert "slow" in parent._child_starts
