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
        if obs["cycle"] >= 10:        # safety
            return Decision(stop=True)
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
            "fanned":   snap.objectives.get("fanned", False),
            "merged":   snap.objectives.get("merged", False),
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
        names = ["explorer", *[f"refiner_{i}" for i in range(K)], "merge"]
        for nm in names:
            if obs[nm].done and nm not in self.done_cycle:
                self.done_cycle[nm] = obs["cycle"]

        if obs["explorer"].status == "NOT_STARTED":
            return Decision(actions=[self._act.spawn_operator("explorer")])

        if obs["explorer"].done and not obs["fanned"]:
            return Decision(actions=[
                self._act.update_param("fanned", True),
                *[self._act.spawn_operator(f"refiner_{i}", _FanChild,
                                           policy=_FanChildPolicy, depends_on="explorer")
                  for i in range(K)],
            ])

        refiners_done = all(obs[f"refiner_{i}"].done for i in range(K))
        if obs["fanned"] and refiners_done and not obs["merged"]:
            return Decision(actions=[
                self._act.update_param("merged", True),
                self._act.spawn_operator("merge", _FanChild, policy=_FanChildPolicy,
                                         depends_on=[f"refiner_{i}" for i in range(K)]),
            ])

        if obs["merged"] and obs["merge"].done:
            return Decision(stop=True)

        if obs["cycle"] >= 80:        # safety stop
            return Decision(stop=True)
        return Decision()


async def test_fanout_dependency_ordering(engine):
    parent = _FanParent(engine)
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
