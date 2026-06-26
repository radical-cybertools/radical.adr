"""Unit tests for ActionSet.get_schema completeness (LLM tool schema).

Closes gap G7: every spawnable tool exposes an optional depends_on, and the
child-operator lifecycle verbs + retry_task are rendered alongside the @acts.
"""

from radical.adr import Operator, act, observe


class _Op(Operator):
    @act
    async def evaluate(self, x: float, seed: int = 0):
        return {"x": x}

    @observe
    def o(self, s):
        return {}


def test_schema_renders_lifecycle_and_retry_verbs():
    schema = _Op(engine=None).get_actions().get_schema("openai")
    names = {s["function"]["name"] for s in schema}
    assert {"spawn_operator", "restart_operator", "cancel_operator", "retry_task"} <= names


def test_act_tool_exposes_optional_depends_on():
    schema = _Op(engine=None).get_actions().get_schema("openai")
    ev = next(s for s in schema if s["function"]["name"] == "evaluate")
    params = ev["function"]["parameters"]
    assert "depends_on" in params["properties"]
    assert "depends_on" not in params.get("required", [])     # optional, not required
