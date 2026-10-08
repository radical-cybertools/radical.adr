"""Trace emitted instructions through recording, task completion and retries."""

import asyncio
import json

import pytest

from radical.adr import Action, ActionKind, Decision, Operator, Policy, act, decide, observe
from radical.adr.recording import RecordingObserver


class _ScriptPolicy(Policy):
    def __init__(self, script):
        super().__init__()
        self.script = script

    @decide
    async def step(self, obs):
        return self.script(obs)


class _LocalOp(Operator):
    @act
    async def constant(self, value: int = 7):
        return value

    @act
    async def nothing(self):
        return None

    @observe
    def extract(self, snapshot):
        return {"cycle": snapshot.cycle, "context": snapshot.task_context,
                "observations": snapshot.observations}


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("kind", [ActionKind.SPAWN_TASK, ActionKind.SPAWN_BLOCK])
async def test_reused_actions_have_distinct_recorded_links(kind, tmp_path):
    path = tmp_path / "trace.jsonl"
    op = _LocalOp(None, observer=RecordingObserver(str(path)))
    # Same object in two positions, and the same decision returned twice. A
    # caller-supplied ID must not turn distinct occurrences into one instruction.
    action = Action(kind=kind, task_name="constant", action_id="policy-template")
    decision = Decision(actions=[action, action])
    op.policy = _ScriptPolicy(
        lambda obs: decision if obs["cycle"] < 2 else Decision(stop=True)
    )
    snapshots = [snapshot async for snapshot in op.run()]

    records = _records(path)
    ids = [a["action_id"] for r in records for a in r["decision"]["actions"]]
    assert len(ids) == len(set(ids)) == 4
    assert all(ids) and "policy-template" not in ids
    assert action.action_id == "policy-template"
    assert decision.actions[0] is action and decision.actions[1] is action
    # All four return the same interned integer; none may overwrite a task.
    context = snapshots[-1].task_context
    assert len(context) == 4
    assert {ctx["action_id"] for ctx in context.values()} == set(ids)
    assert all(snapshots[-1].runtime[uid] == "DONE" for uid in context)
    assert len(op.state.resolved) == 4
    assert len(records[1]["obs"]["context"]) == 2
    assert len(records[2]["obs"]["context"]) == 4

    checkpoint = tmp_path / "checkpoint.json"
    op.save_checkpoint(str(checkpoint))
    restored = _LocalOp(None)
    restored.load_checkpoint(str(checkpoint))
    assert restored.state.task_context == context


async def test_local_retry_links_to_retry_action_and_previous_task(tmp_path):
    path = tmp_path / "trace.jsonl"
    op = _LocalOp(None, observer=RecordingObserver(str(path)))
    acts = op.get_actions()

    def script(obs):
        if obs["cycle"] == 0:
            return Decision(actions=[acts.constant()])
        if obs["cycle"] == 1:
            return Decision(actions=[acts.retry_task(next(iter(obs["context"])))])
        return Decision(stop=True)

    op.policy = _ScriptPolicy(script)
    await op.run_to_completion()
    records = _records(path)
    original, retried = list(op.state.task_context)
    retry_action = records[1]["decision"]["actions"][0]
    assert retry_action["target_uid"] == original
    assert op.state.task_context[original]["action_id"] == records[0]["decision"]["actions"][0]["action_id"]
    assert op.state.task_context[retried]["action_id"] == retry_action["action_id"]
    assert op.state.task_context[original]["action_id"] != retry_action["action_id"]
    assert op.state.runtime[retried] == "DONE"


async def test_delayed_results_keep_their_origin_after_policy_changes(tmp_path):
    futures = []

    class FutureOp(Operator):
        @act
        async def evaluate(self):
            future = asyncio.get_running_loop().create_future()
            futures.append(future)
            return future

        @observe
        def extract(self, snapshot):
            return {"cycle": snapshot.cycle}

    path = tmp_path / "trace.jsonl"
    op = FutureOp(None, observer=RecordingObserver(str(path)))
    op.policy = _ScriptPolicy(lambda obs: Decision(actions=[op.get_actions().evaluate()]))
    snapshots = []
    async for snapshot in op.run():
        snapshots.append(snapshot)
        if snapshot.cycle == 0:
            # Switch before either result arrives. This policy submits one more
            # identical call, then waits for the results and stops.
            op.policy = _ScriptPolicy(lambda obs: (
                Decision(actions=[op.get_actions().evaluate()]) if obs["cycle"] == 1
                else Decision(stop=obs["cycle"] >= 3)
            ))
        elif snapshot.cycle == 1:
            futures[1].set_result("second")
            await asyncio.sleep(0)  # deliver callbacks before the next cycle
        elif snapshot.cycle == 2:
            futures[0].set_result("first")
            await asyncio.sleep(0)

    records = _records(path)
    first_uid, second_uid = list(op.state.task_context)
    assert snapshots[2].observations[f"result.{second_uid}"] == "second"
    assert snapshots[3].observations[f"result.{first_uid}"] == "first"
    assert op.state.task_context[first_uid]["action_id"] == records[0]["decision"]["actions"][0]["action_id"]
    assert op.state.task_context[second_uid]["action_id"] == records[1]["decision"]["actions"][0]["action_id"]
    assert not op.state.task_futures  # links survive terminal Future pruning


async def test_actions_without_tasks_still_get_recorded_ids(tmp_path):
    path = tmp_path / "trace.jsonl"
    op = _LocalOp(None, observer=RecordingObserver(str(path)))
    acts = op.get_actions()
    op.policy = _ScriptPolicy(lambda obs: Decision(
        actions=[acts.nothing(), acts.update_param("value", 2)], stop=True,
    ))
    await op.run_to_completion()
    ids = [a["action_id"] for a in _records(path)[0]["decision"]["actions"]]
    assert all(ids) and len(set(ids)) == 2
    assert op.state.task_context == {}
    assert op.state.objectives["value"] == 2


async def test_child_spawn_and_restart_keep_action_links(engine, tmp_path):
    path = tmp_path / "trace.jsonl"
    parent = _LocalOp(engine, observer=RecordingObserver(str(path)), max_cycles=20)
    parent.add_child("child", _LocalOp,
                     policy=_ScriptPolicy(lambda obs: Decision(stop=True)))
    acts = parent.get_actions()
    restarted = False

    def script(obs):
        nonlocal restarted
        if obs["cycle"] == 0:
            # The second spawn is an idempotent no-op with its own action ID.
            return Decision(actions=[acts.spawn_operator("child"), acts.spawn_operator("child")])
        if parent.child("child").done:
            if restarted:
                return Decision(stop=True)
            restarted = True
            return Decision(actions=[acts.restart_operator("child")])
        return Decision()

    parent.policy = _ScriptPolicy(script)
    await parent.run_to_completion()
    emitted = [a for r in _records(path) for a in r["decision"]["actions"]]
    assert len(emitted) == 3
    assert len({a["action_id"] for a in emitted}) == 3
    assert {ctx["action_id"] for ctx in parent.state.task_context.values()} == {
        emitted[0]["action_id"], emitted[2]["action_id"],
    }
    assert parent.child("child").done
