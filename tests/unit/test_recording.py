"""Unit tests for RecordingObserver and replay_trace — no engine.

Covers Item 4: recording writes one {cycle, obs, decision} line per cycle, and
replay_trace re-drives a policy against each recorded obs in order, offline.
"""

import json
import time

import pytest

from radical.adr.decision import Decision
from radical.adr.policy.base import Policy, decide
from radical.adr.recording import RecordingObserver, replay_trace
from radical.adr.state import Snapshot


def _snap(cycle: int) -> Snapshot:
    return Snapshot(
        operator_id="test",
        cycle=cycle,
        timestamp=time.time(),
        source="self",
        observations={},
        runtime={},
        objectives={},
        artifacts={},
        directives={},
    )


def test_recording_writes_one_line_per_cycle(tmp_path):
    path = tmp_path / "trace.jsonl"
    obs_seq = [{"cycle": 0, "x": 1}, {"cycle": 1, "x": 2}, {"cycle": 2, "x": 3}]

    rec = RecordingObserver(str(path))
    rec.on_start("op", {})
    for i, obs in enumerate(obs_seq):
        rec.on_cycle(_snap(i), obs, Decision(stop=(i == 2)))
    rec.on_stop(_snap(2), "done")

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        record = json.loads(line)
        assert set(record.keys()) == {"cycle", "obs", "decision"}
        assert record["cycle"] == i
        assert record["obs"] == obs_seq[i]


class _EchoPolicy(Policy):
    """Records each obs it was driven with and returns a trivial Decision."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list = []

    @decide
    async def run(self, obs: dict) -> Decision:
        self.seen.append(obs)
        return Decision(stop=bool(obs.get("stop")))


async def test_replay_trace_drives_policy_in_order(tmp_path):
    path = tmp_path / "trace.jsonl"
    obs_seq = [{"cycle": 0}, {"cycle": 1}, {"cycle": 2, "stop": True}]
    with open(path, "w") as f:
        for i, obs in enumerate(obs_seq):
            f.write(json.dumps({"cycle": i, "obs": obs, "decision": {}}) + "\n")

    policy = _EchoPolicy()
    decisions = await replay_trace(str(path), policy)

    assert len(decisions) == 3
    assert policy.seen == obs_seq                 # exact obs pass-through, in order
    assert decisions[-1].stop is True


async def test_replay_trace_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        await replay_trace(str(tmp_path / "nope.jsonl"), _EchoPolicy())
