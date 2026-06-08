"""08-hierarchical-operator.py

Dynamic parent-child DAG: parent operator sequences two child operators
through its own observe/decide/act loop.

ADL concepts demonstrated
  - @tool returning a Future from child_op.start() — operator as block
  - Parent drives child lifecycle: launch A, then B only after A finishes
  - dep_futures passed through @tool to child.start() for native AsyncFlow DAG
  - SNAPSHOT messaging: children report back to parent each cycle
  - No external scaffolding — the parent's run() loop drives the entire DAG

DAG:
    parent
     ├── child-a  (starts cycle 0, converges on TARGET=3.14)
     └── child-b  (starts only after child-a is DONE, converges from opposite side)

Usage:
    python 08-hierarchical-operator.py
"""

from __future__ import annotations

import asyncio
import random
import time
from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from radical.adl import Decision, Operator, Policy, decide, goals, observe, act
from radical.adl.goals import Goal

TARGET     = 3.14159
MIN_RADIUS = 0.005


async def evaluate_task(x: float, seed: int = 0) -> dict:
    import random
    rng = random.Random(seed)
    await asyncio.sleep(0.005)   # simulate light compute
    return {"x": x, "value": (x - TARGET) ** 2 + rng.gauss(0, 0.03)}


# ── Child operator (reused for both children) ────────────────────────────────

class SearchOperator(Operator):

    center:  float = 0.0
    radius:  float = 5.0
    history: list  = []

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
                *[self._act.evaluate(x=new_c + new_r * (2*i/3-1), seed=obs["cycle"]*4+i)
                  for i in range(4)],
            ])
        return Decision(actions=[
            self._act.evaluate(x=obs["center"] + obs["radius"]*(2*i/3-1),
                                  seed=obs["cycle"]*4+i)
            for i in range(4)
        ])


# ── Parent operator ───────────────────────────────────────────────────────────

class ParentOperator(Operator):

    def __init__(self, engine, operator_id=None, **kwargs):
        super().__init__(engine, operator_id, **kwargs)
        # Children are instance attributes — accessible via self in @tool methods.
        # parent=self ensures _dispatch sends SNAPSHOT messages to this operator.
        self.child_a = SearchOperator(engine, "child-a", parent=self)
        self.child_b = SearchOperator(engine, "child-b", parent=self)
        self.child_a.policy = SearchPolicy(self.child_a)
        self.child_b.policy = SearchPolicy(self.child_b)
        # Child-b starts far from TARGET (slower convergence)
        self.child_b.center = -8.0

    @act
    async def launch_child_a(self) -> asyncio.Future:
        """Start child-a as an AsyncFlow block; returns its block Future."""
        fut = self.child_a.start()
        uid = str(getattr(fut, "id", id(fut)))
        self.state.artifacts["child_start_uid.child-a"] = uid
        return fut

    @act
    async def launch_child_b(self, dep_future) -> asyncio.Future:
        """Start child-b after child-a completes (native AsyncFlow dependency)."""
        fut = self.child_b.start(dep_future)
        uid = str(getattr(fut, "id", id(fut)))
        self.state.artifacts["child_start_uid.child-b"] = uid
        return fut

    @observe
    def watch(self, snapshot) -> dict:
        a_uid     = snapshot.artifacts.get("child_start_uid.child-a")
        b_uid     = snapshot.artifacts.get("child_start_uid.child-b")
        a_status  = snapshot.runtime.get(a_uid, "NOT_STARTED") if a_uid else "NOT_STARTED"
        b_status  = snapshot.runtime.get(b_uid, "NOT_STARTED") if b_uid else "NOT_STARTED"
        # Child objectives (radius, center) arrive via SNAPSHOT messaging each cycle
        a_radius  = snapshot.observations.get("child:child-a:obj:radius", float("inf"))
        b_radius  = snapshot.observations.get("child:child-b:obj:radius", float("inf"))
        return {
            "child_a_uid":    a_uid,
            "child_b_uid":    b_uid,
            "child_a_status": a_status,
            "child_b_status": b_status,
            "child_a_radius": a_radius,
            "child_b_radius": b_radius,
            "cycle":          snapshot.cycle,
        }


class ParentPolicy(Policy):

    def __init__(self, op: ParentOperator) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        # Cycle 0: launch child-a
        if obs["cycle"] == 0:
            print(f"  [parent c{obs['cycle']:2d}] launching child-a")
            return Decision(actions=[self._act.launch_child_a()])

        a_uid    = obs["child_a_uid"]
        a_status = obs["child_a_status"]
        b_uid    = obs["child_b_uid"]
        b_status = obs["child_b_status"]

        # When child-a is DONE: launch child-b with child-a's Future as dependency
        if a_status == "DONE" and b_uid is None:
            print(f"  [parent c{obs['cycle']:2d}] child-a done "
                  f"(radius={obs['child_a_radius']:.5f}) → launching child-b")
            return Decision(actions=[
                self._act.launch_child_b(depends_on=[a_uid])
            ])

        # When child-b is DONE: stop
        if b_status == "DONE":
            print(f"  [parent c{obs['cycle']:2d}] child-b done "
                  f"(radius={obs['child_b_radius']:.5f}) → stopping")
            return Decision(stop=True)

        # Idle cycle — waiting for children
        if obs["cycle"] % 5 == 0:
            print(f"  [parent c{obs['cycle']:2d}] "
                  f"child-a={a_status} (r={obs['child_a_radius']:.4f})  "
                  f"child-b={b_status} (r={obs['child_b_radius']:.4f})")
        return Decision()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    backend = await LocalExecutionBackend(ProcessPoolExecutor(max_workers=8))
    engine  = await WorkflowEngine.create(backend)

    parent        = ParentOperator(engine, "parent")
    parent.policy = ParentPolicy(parent)

    print("DAG of operators: parent → child-a → child-b (sequential)\n")
    t0 = time.perf_counter()
    await parent.run_to_completion()
    elapsed = time.perf_counter() - t0

    print(f"\nDone in {elapsed:.2f}s  |  parent cycles={parent.state.cycle}")
    print(f"  child-a: cycles={parent.child_a.state.cycle}  "
          f"radius={parent.child_a.radius:.5f}  "
          f"center={parent.child_a.center:+.5f}")
    print(f"  child-b: cycles={parent.child_b.state.cycle}  "
          f"radius={parent.child_b.radius:.5f}  "
          f"center={parent.child_b.center:+.5f}")

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
