"""08-hierarchical-operator.py

Adaptive fan-out across child operators — the parent runs its own
observe/decide/act loop and spawns a *runtime-determined* set of children,
each launched relative to what an earlier child discovered.

ADR concepts demonstrated
  - add_child(name, OperatorCls, policy=…) — declare a child as first-class
  - spawn_operator(name, …) — launch a child operator as a block (steering, not
    scheduling); the runtime count/names/config are all decided live
  - depends_on=<child names> — gate a launch on upstream children completing,
    expressed in domain terms (names), never Futures/uids
  - obs["explorer"].done / obs["explorer"].best_x — observe a child through a
    live ChildHandle; no runtime machinery (Future, UID, artifacts) in user code

DAG (decided at runtime, not hard-coded):
    campaign
     └── explorer                      (broad search; finds a best region)
          ├── refiner_0  depends_on=explorer   (narrow search around best - Δ)
          ├── refiner_1  depends_on=explorer   (narrow search around best)
          └── refiner_2  depends_on=explorer   (narrow search around best + Δ)

Usage:
    python 08-hierarchical-operator.py
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from radical.adr import Decision, Operator, Policy, act, decide, goals, observe
from radical.adr.goals import Goal

TARGET     = 3.14159
MIN_RADIUS = 0.02
K          = 3        # refiners fanned out around the explorer's best region


async def evaluate_task(x: float, seed: int = 0) -> dict:
    import random
    rng = random.Random(seed)
    await asyncio.sleep(0.005)   # simulate light compute
    return {"x": x, "value": (x - TARGET) ** 2 + rng.gauss(0, 0.03)}


# ── Child operator (reused for explorer + every refiner) ─────────────────────

class SearchOperator(Operator):

    center:  float = 0.0
    radius:  float = 5.0
    best_x:  float = 0.0   # surfaced to the parent via the ChildHandle

    @goals
    def convergence_criteria(self):
        return Goal(name="converged", metric="radius",
                    threshold=MIN_RADIUS, direction="minimize")

    @act
    async def evaluate(self, x: float, seed: int = 0):
        return self.engine.function_task(evaluate_task)(x=x, seed=seed)

    @observe
    def extract(self, snapshot) -> dict:
        new_results = [v for k, v in snapshot.observations.items()
                       if k.startswith("result.") and isinstance(v, dict) and "x" in v]
        history = list(snapshot.objectives.get("history", [])) + new_results
        best    = min(history, key=lambda r: r["value"]) if history else None
        return {"new_results": new_results, "history": history,
                "center": snapshot.objectives.get("center", 0.0),
                "radius": snapshot.objectives.get("radius", 5.0),
                "best": best, "cycle": snapshot.cycle}


class SearchPolicy(Policy):

    def __init__(self, op: SearchOperator) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] >= 30:
            return Decision(stop=True)
        if obs["best"] is not None:
            new_c, new_r = obs["best"]["x"], obs["radius"] * 0.6
            return Decision(actions=[
                self._act.update_param("history", obs["history"]),
                self._act.update_param("center",  new_c),
                self._act.update_param("radius",  new_r),
                self._act.update_param("best_x",  new_c),   # report best to parent
                *[self._act.evaluate(x=new_c + new_r * (2*i/3-1), seed=obs["cycle"]*4+i)
                  for i in range(4)],
            ])
        return Decision(actions=[
            self._act.evaluate(x=obs["center"] + obs["radius"]*(2*i/3-1),
                               seed=obs["cycle"]*4+i)
            for i in range(4)
        ])


# ── Parent operator ───────────────────────────────────────────────────────────

class CampaignOperator(Operator):

    def __init__(self, engine, operator_id=None, **kwargs):
        super().__init__(engine, operator_id, **kwargs)
        # The explorer is declared up front; refiners are added adaptively at
        # runtime (in the policy) once the explorer reports its best region.
        self.add_child("explorer", SearchOperator, policy=SearchPolicy)

    @observe
    def watch(self, snapshot) -> dict:
        obs = {
            "explorer":   self.child("explorer"),
            "fanned_out": snapshot.objectives.get("fanned_out", False),
            "cycle":      snapshot.cycle,
        }
        for i in range(K):
            obs[f"refiner_{i}"] = self.child(f"refiner_{i}")
        return obs


class CampaignPolicy(Policy):

    def __init__(self, op: CampaignOperator) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        explorer = obs["explorer"]

        # Cycle 0: launch the explorer (already registered in __init__).
        if explorer.status == "NOT_STARTED":
            print(f"  [campaign c{obs['cycle']:2d}] launching explorer")
            return Decision(actions=[self._act.spawn_operator("explorer")])

        # Explorer done → fan out K refiners around its best region. Count, names,
        # and per-child config are all decided here, at runtime. `fanned_out` is a
        # user-managed spawn-once guard (a first-class idempotency API is a planned
        # follow-up).
        if explorer.done and not obs["fanned_out"]:
            best = explorer.best_x
            print(f"  [campaign c{obs['cycle']:2d}] explorer done "
                  f"(best_x={best:+.4f}) → fanning out {K} refiners")
            return Decision(actions=[
                self._act.update_param("fanned_out", True),
                *[self._act.spawn_operator(
                      f"refiner_{i}", SearchOperator, policy=SearchPolicy,
                      depends_on="explorer", center=best + (i - 1) * 0.5)
                  for i in range(K)],
            ])

        # All refiners done → stop.
        if obs["fanned_out"] and all(obs[f"refiner_{i}"].done for i in range(K)):
            print(f"  [campaign c{obs['cycle']:2d}] all {K} refiners done → stopping")
            return Decision(stop=True)

        # Idle cycle — waiting on children. Surface their live status periodically.
        if obs["fanned_out"] and obs["cycle"] % 5 == 0:
            states = "  ".join(f"refiner_{i}={obs[f'refiner_{i}'].status}"
                               for i in range(K))
            print(f"  [campaign c{obs['cycle']:2d}] {states}")
        return Decision()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    backend = await LocalExecutionBackend(ProcessPoolExecutor(max_workers=8))
    engine  = await WorkflowEngine.create(backend)

    campaign        = CampaignOperator(engine, "campaign")
    campaign.policy = CampaignPolicy(campaign)

    print(f"Adaptive fan-out: explorer → {K} refiners (depends_on=explorer)\n")
    t0 = time.perf_counter()
    await campaign.run_to_completion()
    elapsed = time.perf_counter() - t0

    explorer = campaign.children["explorer"]
    print(f"\nDone in {elapsed:.2f}s  |  campaign cycles={campaign.state.cycle}")
    print(f"  explorer: cycles={explorer.state.cycle}  "
          f"radius={explorer.radius:.5f}  best_x={explorer.best_x:+.5f}")
    for i in range(K):
        refiner = campaign.children.get(f"refiner_{i}")
        if refiner is not None:
            print(f"  refiner_{i}: cycles={refiner.state.cycle}  "
                  f"radius={refiner.radius:.5f}  center={refiner.center:+.5f}")

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
