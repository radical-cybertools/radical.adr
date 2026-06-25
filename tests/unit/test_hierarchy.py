"""Unit tests for first-class operators — no engine.

Covers the SPAWN_OPERATOR / ChildHandle / add_child surface:
  - ChildHandle status transitions (NOT_STARTED → PENDING/DONE/FAILED) driven by
    the parent's _child_starts + state.runtime
  - ChildHandle metric reads from the parent's persistent objectives merge
    (child:{name}:obj:{metric}); AttributeError / .get(default) when absent
  - ActionSet.spawn_operator(...) builds a SPAWN_OPERATOR Action carrying
    operator_cls/policy as live refs, excluded from model_dump (JSON-safe)
  - _validate_actions rejects a SPAWN_OPERATOR with no registration and no
    operator_cls, and warns (not raises) on unknown depends_on child names
"""

import json
import logging
import warnings

import pytest

from radical.adr import (
    Action,
    ActionKind,
    ChildHandle,
    Operator,
    Policy,
    decide,
    observe,
)
from radical.adr.errors import ActionError


class _Child(Operator):
    radius: float = 5.0

    @observe
    def extract(self, snapshot) -> dict:
        return {"cycle": snapshot.cycle}


class _ChildPolicy(Policy):
    # Factory contract: add_child(..., policy=_ChildPolicy) calls _ChildPolicy(child).
    def __init__(self, op=None) -> None:
        super().__init__()
        self._op = op

    @decide
    async def run(self, obs: dict):
        from radical.adr import Decision
        return Decision()


class _Parent(Operator):
    @observe
    def watch(self, snapshot) -> dict:
        return {"explorer": self.child("explorer")}


def _parent() -> _Parent:
    # No live engine needed — these paths only touch the children registry,
    # _child_starts, state.runtime and state.objectives.
    return _Parent(engine=None)


# ── ChildHandle status ───────────────────────────────────────────────────────

def test_status_not_started_before_spawn():
    p = _parent()
    h = p.child("explorer")
    assert h.status == "NOT_STARTED"
    assert not h.done
    assert not h.failed


def test_status_pending_after_start_recorded():
    p = _parent()
    p._child_starts["explorer"] = "uid-1"
    p.state.runtime["uid-1"] = "PENDING"
    assert p.child("explorer").status == "PENDING"


def test_status_done_and_failed():
    p = _parent()
    p._child_starts["explorer"] = "uid-1"
    p.state.runtime["uid-1"] = "DONE"
    assert p.child("explorer").done
    p.state.runtime["uid-1"] = "FAILED"
    assert p.child("explorer").failed


def test_status_pending_default_when_uid_unknown_in_runtime():
    # start recorded but no runtime entry yet → PENDING (not NOT_STARTED)
    p = _parent()
    p._child_starts["explorer"] = "uid-1"
    assert p.child("explorer").status == "PENDING"


# ── ChildHandle.started ──────────────────────────────────────────────────────

def test_started_false_before_spawn():
    p = _parent()
    assert p.child("explorer").started is False


def test_started_true_after_start_recorded():
    p = _parent()
    p._child_starts["explorer"] = "uid-1"
    assert p.child("explorer").started is True


def test_started_true_in_terminal_states():
    # started stays True for DONE / FAILED / CANCELED — "ever started", not "live".
    p = _parent()
    p._child_starts["explorer"] = "uid-1"
    for terminal in ("DONE", "FAILED", "CANCELED"):
        p.state.runtime["uid-1"] = terminal
        assert p.child("explorer").started is True


# ── ChildHandle metric reads ─────────────────────────────────────────────────

def test_metric_read_from_persistent_objectives():
    p = _parent()
    p.state.objectives["child:explorer:obj:radius"] = 0.01
    p.state.objectives["child:explorer:obj:best_x"] = 3.14
    h = p.child("explorer")
    assert h.radius == 0.01
    assert h.best_x == 3.14
    assert h.get("radius") == 0.01


def test_metric_missing_attribute_raises():
    p = _parent()
    h = p.child("explorer")
    with pytest.raises(AttributeError, match="no reported metric 'radius'"):
        _ = h.radius


def test_metric_missing_get_returns_default():
    p = _parent()
    h = p.child("explorer")
    assert h.get("radius") is None
    assert h.get("radius", 9.9) == 9.9


def test_child_returns_childhandle():
    p = _parent()
    assert isinstance(p.child("explorer"), ChildHandle)


# ── ActionSet.spawn_operator builder ─────────────────────────────────────────

def test_spawn_operator_builds_action():
    p = _parent()
    acts = p.get_actions()
    a = acts.spawn_operator("refiner_0", _Child, policy=_ChildPolicy,
                            depends_on="explorer", center=3.1)
    assert a.kind == ActionKind.SPAWN_OPERATOR
    assert a.task_name == "refiner_0"
    assert a.depends_on == ["explorer"]          # single name normalized to a list
    assert a.operator_cls is _Child
    assert a.policy is _ChildPolicy
    assert a.task_kwargs == {"center": 3.1}


def test_spawn_operator_depends_on_list():
    p = _parent()
    a = p.get_actions().spawn_operator("merge", _Child,
                                       depends_on=["refiner_0", "refiner_1"])
    assert a.depends_on == ["refiner_0", "refiner_1"]


def test_spawn_operator_action_model_dump_is_json_safe():
    p = _parent()
    a = p.get_actions().spawn_operator("refiner_0", _Child, policy=_ChildPolicy,
                                       depends_on="explorer", center=3.1)
    dumped = a.model_dump()
    # operator_cls/policy are live refs excluded from serialization
    assert "operator_cls" not in dumped
    assert "policy" not in dumped
    # the rest round-trips through JSON cleanly
    json.dumps(dumped)
    assert dumped["kind"] == "spawn_operator"
    assert dumped["task_name"] == "refiner_0"
    assert dumped["depends_on"] == ["explorer"]


# ── _validate_actions ────────────────────────────────────────────────────────

def test_validate_rejects_unregistered_without_operator_cls():
    p = _parent()
    a = Action(kind=ActionKind.SPAWN_OPERATOR, task_name="ghost")
    with pytest.raises(ActionError, match="child not registered and no operator_cls"):
        p._validate_actions([a])


def test_validate_requires_task_name():
    p = _parent()
    a = Action(kind=ActionKind.SPAWN_OPERATOR)
    with pytest.raises(ActionError, match="child name. is required"):
        p._validate_actions([a])


def test_validate_accepts_registered_child_without_cls():
    p = _parent()
    p.add_child("explorer", _Child)
    a = Action(kind=ActionKind.SPAWN_OPERATOR, task_name="explorer")
    p._validate_actions([a])   # registered → no operator_cls needed


def test_validate_accepts_operator_cls_for_new_name():
    p = _parent()
    a = Action(kind=ActionKind.SPAWN_OPERATOR, task_name="refiner_0",
               operator_cls=_Child)
    p._validate_actions([a])   # not registered but cls provided → OK


def test_validate_unknown_depends_on_name_warns_not_raises():
    p = _parent()
    a = Action(kind=ActionKind.SPAWN_OPERATOR, task_name="refiner_0",
               operator_cls=_Child, depends_on=["explorer"])
    with pytest.warns(UserWarning, match="depends_on child 'explorer'"):
        p._validate_actions([a])


def test_validate_known_depends_on_name_no_warning():
    p = _parent()
    p.add_child("explorer", _Child)
    a = Action(kind=ActionKind.SPAWN_OPERATOR, task_name="refiner_0",
               operator_cls=_Child, depends_on=["explorer"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p._validate_actions([a])


# ── add_child ────────────────────────────────────────────────────────────────

def test_add_child_seeds_config_into_objectives():
    p = _parent()
    child = p.add_child("refiner_0", _Child, center=2.5)
    # center is not an annotated attr on _Child → plain instance attr; radius is.
    assert p.children["refiner_0"] is child
    assert child.center == 2.5


def test_add_child_attaches_policy_factory():
    p = _parent()
    child = p.add_child("explorer", _Child, policy=_ChildPolicy)
    assert isinstance(child.policy, _ChildPolicy)


def test_add_child_attaches_policy_instance():
    p = _parent()
    inst = _ChildPolicy()
    child = p.add_child("explorer", _Child, policy=inst)
    assert child.policy is inst


def test_add_child_seeds_annotated_attr_into_objectives():
    p = _parent()
    child = p.add_child("refiner_0", _Child, radius=0.3)
    assert child.state.objectives["radius"] == 0.3


def test_add_child_records_initial_config_for_restart():
    p = _parent()
    p.add_child("refiner_0", _Child, radius=0.3, center=2.5)
    assert p._child_config["refiner_0"] == {"radius": 0.3, "center": 2.5}


# ── restart_operator / cancel_operator builders ──────────────────────────────

def test_restart_operator_builds_action():
    p = _parent()
    a = p.get_actions().restart_operator("refiner_0", depends_on="explorer")
    assert a.kind == ActionKind.RESTART_OPERATOR
    assert a.task_name == "refiner_0"
    assert a.depends_on == ["explorer"]            # single name normalized to a list


def test_restart_operator_depends_on_list():
    p = _parent()
    a = p.get_actions().restart_operator("merge", depends_on=["refiner_0", "refiner_1"])
    assert a.depends_on == ["refiner_0", "refiner_1"]


def test_cancel_operator_builds_action():
    p = _parent()
    a = p.get_actions().cancel_operator("refiner_0")
    assert a.kind == ActionKind.CANCEL_OPERATOR
    assert a.task_name == "refiner_0"


# ── spawn_operator idempotency (no-op + warning) ─────────────────────────────

async def test_spawn_operator_noop_when_already_started(caplog):
    p = _parent()
    p.add_child("explorer", _Child)
    p._child_starts["explorer"] = "uid-1"          # simulate "already started"
    p.state.runtime["uid-1"] = "RUNNING"
    a = Action(kind=ActionKind.SPAWN_OPERATOR, task_name="explorer")
    with caplog.at_level(logging.WARNING):
        result = await p._spawn_operator(a)
    assert result is None                          # no-op → no new launch
    assert "no-op" in caplog.text
    assert p._child_starts["explorer"] == "uid-1"  # untouched


async def test_cancel_operator_noop_when_unstarted(caplog):
    p = _parent()
    p.add_child("explorer", _Child)                # registered but never started
    a = Action(kind=ActionKind.CANCEL_OPERATOR, task_name="explorer")
    with caplog.at_level(logging.WARNING):
        await p._cancel_operator(a)
    assert "no-op" in caplog.text


# ── _reset_run_state (restart's wipe-to-initial-config) ──────────────────────

def test_reset_run_state_restores_initial_config():
    p = _parent()
    child = p.add_child("refiner_0", _Child, radius=0.3)   # initial radius=0.3
    # Evolve the child's state, then reset.
    child.radius = 9.9
    child.state.cycle = 7
    child.state.runtime["x"] = "DONE"
    p._reset_run_state("refiner_0")
    assert child.radius == 0.3            # back to the seeded config
    assert child.state.cycle == 0         # fresh RuntimeState
    assert child.state.runtime == {}
    assert child._block_factory is None   # next start() rebuilds the block


# ── _validate_actions: restart / cancel ──────────────────────────────────────

def test_validate_restart_unknown_child_raises():
    p = _parent()
    a = Action(kind=ActionKind.RESTART_OPERATOR, task_name="ghost")
    with pytest.raises(ActionError, match="unknown child"):
        p._validate_actions([a])


def test_validate_restart_registered_child_ok():
    p = _parent()
    p.add_child("explorer", _Child)
    a = Action(kind=ActionKind.RESTART_OPERATOR, task_name="explorer")
    p._validate_actions([a])


def test_validate_restart_requires_task_name():
    p = _parent()
    a = Action(kind=ActionKind.RESTART_OPERATOR)
    with pytest.raises(ActionError, match="task_name"):
        p._validate_actions([a])


def test_validate_cancel_unknown_child_does_not_raise():
    # cancel is safe / idempotent — unknown name validates (no-op logged at apply).
    p = _parent()
    a = Action(kind=ActionKind.CANCEL_OPERATOR, task_name="ghost")
    p._validate_actions([a])


def test_validate_cancel_requires_task_name():
    p = _parent()
    a = Action(kind=ActionKind.CANCEL_OPERATOR)
    with pytest.raises(ActionError, match="task_name"):
        p._validate_actions([a])
