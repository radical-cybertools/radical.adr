"""11-record-replay.py

Record a live run's observation trace, then replay it offline against a
*different* policy — deterministic A/B evaluation with no live execution.

The operator surfaces a real evolving metric (`best_score`) and the live policy
emits actual actions each cycle, so the recorded JSONL carries meaningful obs and
decisions — not just a cycle counter. The candidate policy reasons over that
recorded metric (a score threshold) and diverges from the live policy on the
exact same observations.

ADR concepts demonstrated
  - RecordingObserver — captures the per-cycle obs dict + decision to JSONL
  - replay_trace      — re-drives a different policy against the recorded obs,
                        offline (no engine, no Operator, no task dispatch)

Usage:
    python 11-record-replay.py
"""

import asyncio
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from radical.adr import (
    Decision,
    Operator,
    Policy,
    RecordingObserver,
    decide,
    observe,
    replay_trace,
)
from radical.adr import enable_logging


class SearchOperator(Operator):
    """Tracks a best-so-far score that the policy drives down each cycle."""

    best_score: float = 1.0   # annotated → proxied to state.objectives

    @observe
    def extract(self, snapshot) -> dict:
        return {
            "cycle":      snapshot.cycle,
            "best_score": snapshot.objectives.get("best_score", 1.0),
        }


class LivePolicy(Policy):
    """Improves the score each cycle (emits a real update_param action), then
    stops after a fixed cycle budget."""

    def __init__(self, op: SearchOperator) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] >= 5:
            return Decision(stop=True)
        improved = obs["best_score"] * 0.6
        return Decision(actions=[self._act.update_param("best_score", improved)])


class CandidatePolicy(Policy):
    """A different policy we want to evaluate on the *same* recorded conditions:
    it stops on a score threshold rather than a cycle budget, so it converges
    earlier on identical observations."""

    THRESHOLD = 0.25

    @decide
    async def run(self, obs: dict) -> Decision:
        return Decision(stop=obs["best_score"] <= self.THRESHOLD)


async def main():
    trace_path = Path(tempfile.gettempdir()) / "adr_record_replay_trace.jsonl"

    backend = await LocalExecutionBackend(ProcessPoolExecutor())
    engine  = await WorkflowEngine.create(backend)

    # 1. Live run — record the observation trace.
    op = SearchOperator(engine, observer=RecordingObserver(str(trace_path)))
    op.policy = LivePolicy(op)
    async for snapshot in op.run():
        print(f"live   cycle {snapshot.cycle}  "
              f"best_score={snapshot.objectives.get('best_score', 1.0):.4f}")
    await engine.shutdown()

    recorded = trace_path.read_text().strip().splitlines()
    print(f"\nrecorded {len(recorded)} cycles to {trace_path}")
    print(f"  sample line (cycle 2): {recorded[2]}")

    # 2. Offline replay — drive a different policy against the same obs trace.
    decisions = await replay_trace(str(trace_path), CandidatePolicy())
    stop_cycle = next((i for i, d in enumerate(decisions) if d.stop), None)
    print(f"\nreplayed {len(decisions)} decisions; LivePolicy ran all "
          f"{len(recorded)} cycles, but CandidatePolicy (stop when "
          f"best_score <= {CandidatePolicy.THRESHOLD}) would have stopped "
          f"at cycle {stop_cycle}")

    assert len(decisions) == len(recorded)
    assert stop_cycle == 3


if __name__ == "__main__":
    enable_logging(logging.INFO)
    asyncio.run(main())
