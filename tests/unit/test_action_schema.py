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


# ── @act parameter rendering ──────────────────────────────────────────────────

class _ParamOp(Operator):
    @act
    async def chained(self, *dep_futures, scale: float = 1.0):
        return {"n": len(dep_futures)}

    @act
    async def gated(self, *dep_futures, tag: str):
        return {"tag": tag}

    @act
    async def flexible(self, **options):
        return options

    # String annotations are what `from __future__ import annotations` produces.
    @act
    async def scaled(self, x: "float", n: "int", tag: "str" = "a"):
        return {"x": x}

    @act
    async def opaque(self, x: "NotDefinedAnywhere"):   # noqa: F821
        return {"x": x}

    @observe
    def o(self, s):
        return {}


def _tool(fmt: str, name: str) -> dict:
    for s in _ParamOp(engine=None).get_actions().get_schema(fmt):
        if fmt == "openai" and s["function"]["name"] == name:
            return s["function"]["parameters"]
        if fmt == "anthropic" and s["name"] == name:
            return s["input_schema"]
    raise KeyError(name)


def test_defaulted_act_param_is_optional():
    for fmt in ("openai", "anthropic"):
        params = _tool(fmt, "scaled")
        assert set(params["properties"]) == {"x", "n", "tag", "depends_on"}
        assert params["required"] == ["x", "n"]


def test_var_positional_param_not_rendered():
    for fmt in ("openai", "anthropic"):
        params = _tool(fmt, "chained")
        assert set(params["properties"]) == {"scale", "depends_on"}
        assert "required" not in params                       # nothing is required
        assert "additionalProperties" not in params
        assert _tool(fmt, "gated")["required"] == ["tag"]     # keyword-only, no default


def test_var_keyword_param_allows_extra_keys():
    for fmt in ("openai", "anthropic"):
        params = _tool(fmt, "flexible")
        assert set(params["properties"]) == {"depends_on"}    # no bogus 'options' param
        assert params["additionalProperties"] is True


def test_string_annotations_render_their_real_types():
    for fmt in ("openai", "anthropic"):
        props = _tool(fmt, "scaled")["properties"]
        assert props["x"] == {"type": "number"}
        assert props["n"] == {"type": "integer"}
        assert props["tag"] == {"type": "string"}


def test_unresolvable_annotation_renders_untyped():
    for fmt in ("openai", "anthropic"):
        params = _tool(fmt, "opaque")
        assert params["properties"]["x"] == {}
        assert params["required"] == ["x"]
