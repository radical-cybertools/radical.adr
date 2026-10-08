"""Action IDs are optional in legacy decisions and survive serialization."""

from radical.adr import Decision


def test_action_without_id_still_loads():
    decision = Decision.model_validate({"actions": [{"kind": "spawn_task", "task_name": "run"}]})
    assert decision.actions[0].action_id is None


def test_action_id_round_trips_in_decision():
    decision = Decision.model_validate({"actions": [
        {"kind": "spawn_task", "task_name": "run", "action_id": "recorded-id"},
    ]})
    restored = Decision.model_validate_json(decision.model_dump_json())
    assert restored.actions[0].action_id == "recorded-id"
