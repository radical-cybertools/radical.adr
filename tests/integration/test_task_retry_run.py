"""Integration test: acts.retry_task re-spawns a failed task over a real engine.

A task whose backend impl always raises is retried once from its recorded
task_context, producing a second tracked 'flaky' task (same name + kwargs).
"""

from __future__ import annotations

import json

import pytest

from radical.adr import Decision, Operator, Policy, act, decide, observe
from radical.adr.recording import RecordingObserver


async def _always_fail_impl(x: float) -> dict:
    raise ValueError("boom")


class _RetryOp(Operator):
    @act
    async def flaky(self, x: float):
        return self.engine.function_task(_always_fail_impl)(x=x)

    @observe
    def o(self, snap) -> dict:
        errors = [k.removeprefix("error.") for k in snap.observations
                  if k.startswith("error.")]
        n_flaky = sum(1 for c in self.state.task_context.values() if c["name"] == "flaky")
        return {"cycle": snap.cycle, "errors": errors, "n_flaky": n_flaky}


class _RetryPolicy(Policy):
    def __init__(self, op):
        super().__init__()
        self._act = op.get_actions()
        self.retried = False

    @decide
    async def run(self, obs) -> Decision:
        if obs["cycle"] == 0:
            return Decision(actions=[self._act.flaky(x=1.0)])
        if obs["errors"] and not self.retried:
            self.retried = True
            return Decision(actions=[self._act.retry_task(obs["errors"][0])])
        if self.retried and obs["n_flaky"] >= 2:
            return Decision(stop=True)
        return Decision()


@pytest.mark.parametrize("track_action_ids", [False, True])
async def test_retry_task_respawns_failed_task(engine, tmp_path, track_action_ids):
    path = tmp_path / "trace.jsonl"
    # The policy stops on the retry, not a cycle count: the first error waits for
    # a new worker process to import this module, which can outlast a fixed number
    # of idle cycles (notably on macOS). max_runtime only bounds a regression.
    op = _RetryOp(engine, observer=RecordingObserver(str(path)),
                  track_action_ids=track_action_ids, max_runtime=30.0)
    op.policy = _RetryPolicy(op)
    async for _ in op.run():
        pass
    # original flaky + one retry → two 'flaky' entries recorded in task_context
    n_flaky = sum(1 for c in op.state.task_context.values() if c["name"] == "flaky")
    assert n_flaky >= 2

    recorded = [json.loads(line) for line in path.read_text().splitlines()]
    emitted = [a for r in recorded for a in r["decision"]["actions"]]
    original_uid, retry_uid = list(op.state.task_context)
    assert emitted[1]["target_uid"] == original_uid
    if track_action_ids:
        assert op.state.task_context[original_uid]["action_id"] == emitted[0]["action_id"]
        assert op.state.task_context[retry_uid]["action_id"] == emitted[1]["action_id"]
        assert emitted[0]["action_id"] != emitted[1]["action_id"]
    else:
        assert all("action_id" not in action for action in emitted)
        assert all(ctx == {"name": "flaky", "kwargs": {"x": 1.0}}
                   for ctx in op.state.task_context.values())
