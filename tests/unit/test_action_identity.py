"""Action IDs are optional in legacy decisions and survive serialization."""

import json

import pytest

from radical.adr import Action, ActionKind, Decision


def test_action_without_id_still_loads():
    decision = Decision.model_validate({"actions": [{"kind": "spawn_task", "task_name": "run"}]})
    assert decision.actions[0].action_id is None


def test_action_id_round_trips_in_decision():
    decision = Decision.model_validate({"actions": [
        {"kind": "spawn_task", "task_name": "run", "action_id": "recorded-id"},
    ]})
    restored = Decision.model_validate_json(decision.model_dump_json())
    assert restored.actions[0].action_id == "recorded-id"


def test_serialization_schema_preserves_action_fields():
    schemas = [
        Action.model_json_schema(mode="serialization"),
        Decision.model_json_schema(mode="serialization")["$defs"]["Action"],
    ]
    for schema in schemas:
        assert {"kind", "task_name", "task_kwargs", "action_id"} <= schema["properties"].keys()
        assert schema["required"] == ["kind"]
        assert not {"operator_cls", "policy"} & schema["properties"].keys()


@pytest.mark.parametrize("action_id", [None, "recorded-id"])
def test_serialization_only_omits_unset_action_id(action_id):
    action = Action(kind=ActionKind.SPAWN_TASK, task_name="run", action_id=action_id)
    expected = {
        "kind": "spawn_task", "target_uid": None, "task_name": "run",
        "task_kwargs": {}, "depends_on": [], "param_key": None, "param_value": None,
    }
    if action_id is not None:
        expected["action_id"] = action_id
    assert action.model_dump() == expected
    assert json.loads(action.model_dump_json()) == expected
    assert Decision(actions=[action]).model_dump()["actions"] == [expected]
    assert json.loads(Decision(actions=[action]).model_dump_json())["actions"] == [expected]
    assert action.model_dump(include={"task_name"}) == {"task_name": "run"}
    assert "action_id" not in action.model_dump(exclude={"action_id"})
