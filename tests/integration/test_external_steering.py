"""External steering: CUSTOM {"mutation": ...} inbox messages are applied by the operator.

This is the path ADLMCPServer uses. The mutation joins the cycle's decision, so it goes
through the same validation, Tunable boundary, and action-ID assignment as the policy's
own actions — and an invalid one is dropped without aborting the run.
"""

from __future__ import annotations

import pytest

from radical.adr import (
    Decision, InboxMessage, MessageKind, Operator, Policy, Tunable, act, decide, observe,
)


class _SteeredOp(Operator):
    center: float = 0.0
    lr = Tunable(0.1, 0.01, 1.0)

    @act
    async def evaluate(self, x: float) -> dict:
        return {"x": x}

    @act
    async def boom(self) -> dict:
        raise RuntimeError("act failed")

    @observe
    def o(self, snap) -> dict:
        return {"results": snap.results, "rejections": snap.rejections}


class _Idle(Policy):
    """Never acts on its own; records what it observed."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[dict] = []

    @decide
    async def run(self, obs) -> Decision:
        self.seen.append(obs)
        return Decision()


class _Capture:
    def __init__(self) -> None:
        self.decisions: list[Decision] = []

    def on_start(self, operator_id, metadata) -> None: ...
    def on_cycle(self, snapshot, obs, decision) -> None:
        self.decisions.append(decision)
    def on_stop(self, final, reason) -> None: ...


def _mutation(mutation) -> InboxMessage:
    return InboxMessage(kind=MessageKind.CUSTOM, sender_id="mcp", payload={"mutation": mutation})


async def test_custom_mutation_is_applied():
    op = _SteeredOp(None, max_cycles=3)
    op.policy = _Idle()
    acts = op.get_actions()
    op.inbox.put_nowait(_mutation(acts.update_param("lr", 0.5)))
    op.inbox.put_nowait(_mutation(acts.evaluate(x=2.0)))
    await op.run_to_completion()

    assert op.lr == pytest.approx(0.5)
    assert any(r == {"x": 2.0} for obs in op.policy.seen for r in obs["results"])


async def test_custom_mutation_respects_tunable_bounds():
    op = _SteeredOp(None, max_cycles=3)
    op.policy = _Idle()
    op.inbox.put_nowait(_mutation(op.get_actions().update_param("lr", 10.0)))   # high=1.0
    await op.run_to_completion()

    assert op.lr == pytest.approx(0.1)                       # unchanged, not clamped
    assert any(obs["rejections"] for obs in op.policy.seen)  # surfaced as evidence


async def test_custom_mutation_gets_action_id_and_is_observed():
    capture = _Capture()
    op = _SteeredOp(None, max_cycles=2, observer=capture, track_action_ids=True)
    op.policy = _Idle()
    op.inbox.put_nowait(_mutation(op.get_actions().evaluate(x=1.0)))
    await op.run_to_completion()

    applied = [a for d in capture.decisions for a in d.actions]
    assert [a.task_name for a in applied] == ["evaluate"]
    assert applied[0].action_id is not None
    ctx = [c for c in op.state.task_context.values() if c["name"] == "evaluate"]
    assert ctx[0]["action_id"] == applied[0].action_id


async def test_invalid_custom_mutation_is_dropped_without_aborting_run(caplog):
    op = _SteeredOp(None, max_cycles=3)
    op.policy = _Idle()
    acts = op.get_actions()
    op.inbox.put_nowait(_mutation(acts.spawn_task("no_such_act")))
    op.inbox.put_nowait(_mutation(acts.update_param("lr", 0.5)))
    with caplog.at_level("WARNING", logger="radical.adr.operator"):
        await op.run_to_completion()

    assert op.state.cycle == 3                                 # run was not aborted
    assert op.lr == pytest.approx(0.5)                         # the valid mutation still applied
    assert "no_such_act" in caplog.text


async def test_custom_mutation_failing_at_apply_is_logged_not_raised(caplog):
    op = _SteeredOp(None, max_cycles=3)
    op.policy = _Idle()
    acts = op.get_actions()
    op.inbox.put_nowait(_mutation(acts.boom()))
    op.inbox.put_nowait(_mutation(acts.update_param("lr", 0.5)))
    with caplog.at_level("WARNING", logger="radical.adr.operator"):
        await op.run_to_completion()

    assert op.state.cycle == 3                                 # run was not aborted
    assert op.lr == pytest.approx(0.5)                         # later mutation still applied
    assert "act failed" in caplog.text


async def test_custom_stop_mutation_stops_the_run():
    op = _SteeredOp(None, max_cycles=10)
    op.policy = _Idle()
    op.inbox.put_nowait(_mutation(op.get_actions().stop()))
    await op.run_to_completion()

    assert op.state.cycle == 1


async def test_custom_payload_without_mutation_is_ignored():
    op = _SteeredOp(None, max_cycles=2)
    op.policy = _Idle()
    op.inbox.put_nowait(InboxMessage(kind=MessageKind.CUSTOM, sender_id="parent",
                                     payload={"note": "hello"}))
    await op.run_to_completion()

    assert op.state.cycle == 2
    assert op.center == 0.0
