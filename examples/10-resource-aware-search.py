"""10-resource-aware-search.py

Adaptive search with resource-aware goals.

Demonstrates that ``Goal`` is source-agnostic: the same type expresses both
a scientific convergence criterion and a resource constraint. ``@observe`` is
the single assembly point for all metrics — ADR has no knowledge of where
they originate.

ADR concepts demonstrated
  - Single ``Goal`` type for scientific AND resource metrics
  - ``@observe`` assembles metrics from task results AND a user-managed
    telemetry subscription — no ADR telemetry classes involved
  - Policy throttles task count when ``cpu_percent`` is high
  - ``--local`` mode: policy simulates CPU via ``update_param``; same Goals,
    same operator code, different metric source

Usage:
    python 10-resource-aware-search.py           # requires ConcurrentExecutionBackend
    python 10-resource-aware-search.py --local   # no RHAPSODY needed
"""

from __future__ import annotations

import asyncio
import random
import sys
from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from radical.adr import Decision, Operator, act, decide, goals, observe
from radical.adr.goals import Goal
from radical.adr.policy.base import Policy

TARGET     = 3.14159
MIN_RADIUS = 0.005
CPU_LIMIT  = 95.0   # Goal: cpu_percent must be below this to be "satisfied"

# ── Module-level impl ─────────────────────────────────────────────────────────

async def _evaluate_impl(x: float, seed: int = 0) -> dict:
    rng = random.Random(seed)
    return {"x": x, "value": (x - TARGET) ** 2 + rng.gauss(0, 0.03)}


# ── User-managed resource state (updated outside ADR) ─────────────────────────
# The user subscribes to RHAPSODY (or any external stream) and stores the latest
# resource metrics here. ADR has no knowledge of this dict or its source.

_resource: dict = {}


async def _on_resource_event(event: object) -> None:
    """User-defined RHAPSODY subscriber — no ADR types used."""
    if (getattr(event, "event_type", None) == "ResourceUpdate"
            and getattr(event, "resource_scope", "per_node") == "per_node"):
        val = getattr(event, "cpu_percent", None)
        if val is not None:
            _resource["cpu_percent"] = val


# ── Operator ──────────────────────────────────────────────────────────────────

class SearchOperator(Operator):

    center:  float = 0.0
    radius:  float = 5.0
    history: list  = []

    @goals
    def criteria(self):
        return [
            # Scientific: stop when search has converged
            Goal(name="converged",   metric="radius",
                 threshold=MIN_RADIUS, direction="minimize"),
            # Resource: stop only when CPU is healthy (< CPU_LIMIT)
            # With --local, the policy injects a simulated value via update_param
            # so this Goal reads it from snapshot.objectives (the fallback path).
            Goal(name="cpu_healthy", metric="cpu_percent",
                 threshold=CPU_LIMIT,  direction="minimize"),
        ]

    @act
    async def evaluate(self, x: float, seed: int = 0):
        return self.engine.function_task(_evaluate_impl)(x=x, seed=seed)

    @observe
    def extract(self, snapshot) -> dict:
        results = [
            v for k, v in snapshot.observations.items()
            if k.startswith("result.") and isinstance(v, dict) and "x" in v
        ]
        history = list(snapshot.objectives.get("history", [])) + results
        best    = min(history, key=lambda r: r["value"]) if history else None
        obs = {
            "results": results,
            "history": history,
            "center":  snapshot.objectives.get("center", 0.0),
            "radius":  snapshot.objectives.get("radius", 5.0),
            "best":    best,
            "cycle":   snapshot.cycle,
        }
        # Surface resource metric.
        # Priority: live telemetry dict → objectives (--local simulate_cpu path).
        # Default to 0.0 when telemetry has not yet fired — 0.0 < CPU_LIMIT so the
        # goal is satisfied and convergence is not blocked by a missing metric.
        cpu = _resource.get("cpu_percent")
        if cpu is None:
            cpu = snapshot.objectives.get("cpu_percent", 0.0)
        obs["cpu_percent"] = cpu
        return obs


# ── Policy ────────────────────────────────────────────────────────────────────

class SearchPolicy(Policy):
    """Simple policy that shrinks the search radius each cycle.

    Throttles task count when cpu_percent is high.
    In --local mode, simulates a CPU value that decreases over time.
    """

    def __init__(self, op: SearchOperator, simulate_cpu: bool = False) -> None:
        self._act         = op.get_actions()
        self._simulate    = simulate_cpu
        self._sim_cpu     = 99.0  # starts high, decreases toward 0

    @decide
    async def run(self, obs: dict) -> Decision:
        cpu     = obs.get("cpu_percent", 0.0)
        center  = obs["center"]
        radius  = obs["radius"]
        cycle   = obs["cycle"]

        print(f"  cycle {cycle:3d} | radius={radius:.6f} | cpu={cpu:.1f}%")

        # Seed cycle: no data yet
        if cycle == 0:
            actions = [
                self._act.evaluate(x=center + radius * (2*i/3 - 1), seed=i)
                for i in range(4)
            ]
            return self._maybe_sim_cpu(Decision(actions=actions), cycle)

        # Shrink radius
        new_center = obs["best"]["x"] if obs["best"] else center
        new_radius = radius * 0.6
        n_tasks    = 1 if cpu > 80.0 else 4  # throttle under CPU pressure

        actions = [
            self._act.update_param("history", obs["history"]),
            self._act.update_param("center",  new_center),
            self._act.update_param("radius",  new_radius),
            *[
                self._act.evaluate(x=new_center + new_radius * (2*i/(n_tasks-1 or 1) - 1),
                                   seed=cycle * 4 + i)
                for i in range(n_tasks)
            ],
        ]
        return self._maybe_sim_cpu(Decision(actions=actions), cycle)

    def _maybe_sim_cpu(self, decision: Decision, cycle: int) -> Decision:
        if not self._simulate:
            return decision
        # Simulate CPU decreasing over time (starts high, crosses threshold ~cycle 15)
        self._sim_cpu = max(0.0, self._sim_cpu - 6.5)
        decision.actions.append(
            self._act.update_param("cpu_percent", self._sim_cpu)
        )
        return decision


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_rhapsody() -> None:
    """Run with ConcurrentExecutionBackend + live RHAPSODY telemetry."""
    try:
        from rhapsody.backends import ConcurrentExecutionBackend
    except ImportError:
        print("rhapsody not installed — use --local flag.")
        sys.exit(1)

    backend   = await ConcurrentExecutionBackend()
    engine    = await WorkflowEngine.create(backend)
    telemetry = await engine.start_telemetry(resource_poll_interval=1.0)

    # User subscribes to telemetry directly — no ADR class involved
    telemetry.subscribe(_on_resource_event)

    op        = SearchOperator(engine, max_cycles=100)
    op.policy = SearchPolicy(op, simulate_cpu=False)

    print("Resource-aware search  |  RHAPSODY backend")
    print(f"  Goals: radius < {MIN_RADIUS}  AND  cpu_percent < {CPU_LIMIT}")
    print(f"  Telemetry poll: 1.0 s  |  Safety valve: max_cycles=100\n")

    async for _ in op.run():
        pass

    best = min(op.history, key=lambda r: r["value"]) if op.history else None
    print(f"\nStopped autonomously — both goals satisfied")
    print(f"Cycles: {op.state.cycle}  |  Final radius: {op.radius:.6f}")
    if best:
        print(f"Best:   x={best['x']:.6f}  error={abs(best['x'] - TARGET):.6f}")

    await engine.shutdown()


async def main_local() -> None:
    """Run with LocalExecutionBackend + simulated CPU from policy."""
    backend = await LocalExecutionBackend(ProcessPoolExecutor(max_workers=4))
    engine  = await WorkflowEngine.create(backend)

    op        = SearchOperator(engine, max_cycles=100)
    op.policy = SearchPolicy(op, simulate_cpu=True)

    print("Resource-aware search  |  local backend (simulated CPU)")
    print(f"  Goals: radius < {MIN_RADIUS}  AND  cpu_percent < {CPU_LIMIT}")
    print(f"  CPU starts at 99%, drops ~6.5% per cycle  |  Safety valve: max_cycles=100\n")

    async for _ in op.run():
        pass

    best = min(op.history, key=lambda r: r["value"]) if op.history else None
    print(f"\nStopped autonomously — both goals satisfied")
    print(f"Cycles: {op.state.cycle}  |  Final radius: {op.radius:.6f}")
    if best:
        print(f"Best:   x={best['x']:.6f}  error={abs(best['x'] - TARGET):.6f}")

    await engine.shutdown()


if __name__ == "__main__":
    if "--local" in sys.argv:
        asyncio.run(main_local())
    else:
        asyncio.run(main_rhapsody())
