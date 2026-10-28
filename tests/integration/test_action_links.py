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


@pytest.mark.parametrize("options", [{}, {"track_action_ids": False}])
async def test_tracking_disabled_preserves_context_and_trace(options, tmp_path, monkeypatch):
    def unexpected_uuid():
        raise AssertionError("tracking is disabled: no IDs should be generated")

    monkeypatch.setattr("radical.adr.operator.uuid4", unexpected_uuid)
    path = tmp_path / "trace.jsonl"
    op = _LocalOp(None, observer=RecordingObserver(str(path)), **options)
    action = op.get_actions().constant(value=7)
    original = Decision(actions=[action])
    retry = Decision(actions=[op.get_actions().retry_task(f"local.constant.{id(7)}")])
    op.policy = _ScriptPolicy(lambda obs: (
        original if obs["cycle"] == 0 else retry if obs["cycle"] == 1
        else Decision(stop=True)
    ))
    await op.run_to_completion()

    # Legacy local UIDs, task context, and serialized decision fields are intact.
    assert op.state.task_context == {
        f"local.constant.{id(7)}": {"name": "constant", "kwargs": {"value": 7}},
    }
    recorded_action = _records(path)[0]["decision"]["actions"][0]
    assert recorded_action == {
        "kind": "spawn_task", "target_uid": None, "task_name": "constant",
        "task_kwargs": {"value": 7}, "depends_on": [],
        "param_key": None, "param_value": None,
    }
    assert "action_id" not in _records(path)[1]["decision"]["actions"][0]
    assert original.actions[0] is action and action.action_id is None


@pytest.mark.parametrize("kind", [ActionKind.SPAWN_TASK, ActionKind.SPAWN_BLOCK])
async def test_reused_actions_have_distinct_recorded_links(kind, tmp_path):
    path = tmp_path / "trace.jsonl"
    op = _LocalOp(None, observer=RecordingObserver(str(path)), track_action_ids=True)
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
    op = _LocalOp(None, observer=RecordingObserver(str(path)), track_action_ids=True)
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
    op = FutureOp(None, observer=RecordingObserver(str(path)), track_action_ids=True)
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


@pytest.mark.parametrize("track_action_ids", [False, True])
@pytest.mark.parametrize("native_ids", [None, "id", "uid"])
async def test_future_identity_survives_pruning(tmp_path, monkeypatch,
                                              track_action_ids, native_ids):
    # Simulate address reuse deterministically, without retaining completed Futures.
    monkeypatch.setattr("radical.adr.operator.id", lambda _: 7, raising=False)
    if not track_action_ids:
        def unexpected_uuid():
            raise AssertionError("tracking is disabled: no IDs should be generated")
        monkeypatch.setattr("radical.adr.operator.uuid4", unexpected_uuid)

    class FutureOp(Operator):
        @act
        async def evaluate(self, value: int):
            future = asyncio.get_running_loop().create_future()
            if native_ids:
                future.id = f"backend.{value}"
            if native_ids == "uid":  # AsyncFlow's full uid takes precedence over id
                future.uid = f"task.{value}"
            future.set_result(value)
            return future

        @observe
        def extract(self, snapshot):
            return {"cycle": snapshot.cycle}

    path = tmp_path / "trace.jsonl"
    op = FutureOp(None, observer=RecordingObserver(str(path)),
                  track_action_ids=track_action_ids)

    def script(obs):
        if obs["cycle"] < 3:
            return Decision(actions=[op.get_actions().evaluate(value=obs["cycle"])])
        return Decision(stop=True)

    op.policy = _ScriptPolicy(script)
    results = []
    async for snapshot in op.run():
        results.extend(snapshot.results)

    assert results == [0, 1, 2]
    assert not op.state.task_futures
    context = op.state.task_context
    if native_ids == "uid":
        assert set(context) == {"task.0", "task.1", "task.2"}
    elif native_ids:
        assert set(context) == {"backend.0", "backend.1", "backend.2"}
    elif not track_action_ids:
        assert set(context) == {"7"}  # legacy fallback is unchanged
    if track_action_ids:
        emitted = [a for r in _records(path) for a in r["decision"]["actions"]]
        assert {ctx["action_id"]: ctx["kwargs"]["value"] for ctx in context.values()} == {
            action["action_id"]: i for i, action in enumerate(emitted)
        }
    else:
        assert all("action_id" not in ctx for ctx in context.values())


async def test_actions_without_tasks_still_get_recorded_ids(tmp_path):
    path = tmp_path / "trace.jsonl"
    op = _LocalOp(None, observer=RecordingObserver(str(path)), track_action_ids=True)
    acts = op.get_actions()
    op.policy = _ScriptPolicy(lambda obs: Decision(
        actions=[acts.nothing(), acts.update_param("value", 2)], stop=True,
    ))
    await op.run_to_completion()
    ids = [a["action_id"] for a in _records(path)[0]["decision"]["actions"]]
    assert all(ids) and len(set(ids)) == 2
    assert op.state.task_context == {}
    assert op.state.objectives["value"] == 2


@pytest.mark.parametrize("child_tracking", [False, True])
@pytest.mark.parametrize("native_ids", [False, True])
async def test_child_spawn_and_restart_keep_action_links(engine, tmp_path, monkeypatch,
                                                       child_tracking, native_ids):
    path = tmp_path / "trace.jsonl"
    parent = _LocalOp(engine, observer=RecordingObserver(str(path)),
                      max_cycles=20, track_action_ids=True)
    child = _LocalOp(engine, track_action_ids=child_tracking)
    if not native_ids:
        monkeypatch.setattr(child, "start", lambda: asyncio.create_task(child.run_to_completion()))
    parent.add_child("child", child, policy=_ScriptPolicy(lambda obs: Decision(
        actions=[child.get_actions().constant()], stop=True,
    )))
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
    # Each operator opts in independently; restarting retains its configuration.
    child_context, = child.state.task_context.values()
    assert ("action_id" in child_context) is child_tracking


async def test_retry_from_legacy_checkpoint_adds_link_only_to_new_task(tmp_path):
    checkpoint = tmp_path / "legacy.json"
    old = _LocalOp(None)
    old.state.task_context["old-task"] = {"name": "constant", "kwargs": {"value": 7}}
    old.state.runtime["old-task"] = "DONE"
    old.save_checkpoint(str(checkpoint))

    path = tmp_path / "trace.jsonl"
    op = _LocalOp(None, observer=RecordingObserver(str(path)), track_action_ids=True)
    op.load_checkpoint(str(checkpoint))
    op.policy = _ScriptPolicy(lambda obs: Decision(
        actions=[op.get_actions().retry_task("old-task")], stop=True,
    ))
    await op.run_to_completion()
    assert "action_id" not in op.state.task_context["old-task"]
    new_context, = [ctx for uid, ctx in op.state.task_context.items() if uid != "old-task"]
    retry_action = _records(path)[0]["decision"]["actions"][0]
    assert new_context["action_id"] == retry_action["action_id"]
    assert retry_action["target_uid"] == "old-task"


async def _double_impl(x: int) -> int:
    return 2 * x


async def test_asyncflow_keys_are_call_time_uids(engine, tmp_path, monkeypatch):
    # Task and block keys must be AsyncFlow's full uids (``task.000001``,
    # ``block.000002``), which its telemetry and trace events use, not the
    # bare ``future.id`` suffix.
    tasks, blocks = [], []

    class TaskOp(Operator):
        @act
        async def double(self, x: int):
            future = self.engine.function_task(_double_impl)(x=x)
            tasks.append((future, getattr(future, "uid", None)))
            return future

        @observe
        def extract(self, snapshot):
            return {"cycle": snapshot.cycle}

    path = tmp_path / "trace.jsonl"
    parent = TaskOp(engine, observer=RecordingObserver(str(path)),
                    max_runtime=60.0, track_action_ids=True)
    child = TaskOp(engine)
    start = child.start

    def capture_start(*deps):
        block = start(*deps)
        blocks.append(block)
        return block

    monkeypatch.setattr(child, "start", capture_start)
    parent.add_child("child", child, policy=_ScriptPolicy(lambda obs: Decision(stop=True)))
    acts = parent.get_actions()
    parent.policy = _ScriptPolicy(lambda obs: (
        Decision(actions=[acts.double(x=3), acts.spawn_operator("child")])
        if obs["cycle"] == 0
        else Decision(stop=parent.child("child").done and not parent.inflight)
    ))
    observations = {}
    async for snapshot in parent.run():
        observations.update(snapshot.observations)

    (task, call_time_uid), = tasks
    if call_time_uid is None:
        pytest.skip("this AsyncFlow sets no future.uid at call time "
                    "(radical-cybertools/radical.asyncflow#106 is unreleased)")
    block, = blocks
    emitted = [a for r in _records(path) for a in r["decision"]["actions"]]
    task_uid, block_uid = list(parent.state.task_context)
    assert task_uid == call_time_uid == task.task["uid"]
    assert block_uid == parent._child_starts["child"] == block.block["uid"]
    assert observations[f"result.{task_uid}"] == 6
    assert parent.child("child").done
    assert [parent.state.task_context[uid]["action_id"] for uid in (task_uid, block_uid)] == [
        action["action_id"] for action in emitted[:2]
    ]
