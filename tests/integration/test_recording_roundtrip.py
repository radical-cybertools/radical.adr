"""Integration test for record-then-replay round trip — uses the engine fixture.

Covers Item 4 end-to-end: run a real operator with RecordingObserver, then
replay_trace the resulting file against a *different* policy instance offline.
"""

from __future__ import annotations

from radical.adr import Decision, Operator, Policy, decide, observe
from radical.adr.recording import RecordingObserver, replay_trace


class _CountOp(Operator):
    @observe
    def extract(self, snapshot) -> dict:
        return {"cycle": snapshot.cycle}


class _LivePolicy(Policy):
    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] >= 3:
            return Decision(stop=True)
        return Decision()


class _ReplayPolicy(Policy):
    """A different policy: stops one cycle earlier than the live one."""

    @decide
    async def run(self, obs: dict) -> Decision:
        return Decision(stop=obs["cycle"] >= 2)


async def test_record_then_replay_round_trip(engine, tmp_path):
    path = tmp_path / "trace.jsonl"

    op = _CountOp(engine, observer=RecordingObserver(str(path)))
    op.policy = _LivePolicy()
    await op.run_to_completion()

    n_recorded = len(path.read_text().strip().splitlines())
    assert n_recorded >= 3        # live policy ran several cycles before stopping

    decisions = await replay_trace(str(path), _ReplayPolicy())

    # One replayed decision per recorded cycle
    assert len(decisions) == n_recorded
    # The replay policy applied its own (different) logic: stop at cycle >= 2
    assert decisions[2].stop is True
    assert decisions[0].stop is False
