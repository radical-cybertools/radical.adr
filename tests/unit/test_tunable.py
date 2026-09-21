"""Unit tests for the Tunable adaptation boundary: validation semantics and the
agent-schema rendering (one declaration → validation + schema)."""

from __future__ import annotations

import pytest

from radical.adr import Operator, Tunable, observe


def test_validate_in_bounds():
    t = Tunable(3e-4, 1e-5, 1e-3)
    val, reason = t.validate(5e-4)
    assert reason is None and val == pytest.approx(5e-4)


def test_validate_out_of_bounds_is_rejected_not_clamped():
    t = Tunable(3e-4, 1e-5, 1e-3)
    val, reason = t.validate(10.0)
    assert val is None and reason == "out_of_bounds"     # never returns a clamped value


def test_validate_int_accepts_integer_valued_float_rejects_fraction():
    t = Tunable(256, 64, 1024, int=True)
    val, reason = t.validate(256.0)
    assert reason is None and val == 256 and isinstance(val, int)
    val, reason = t.validate(256.7)
    assert val is None and reason == "not_integer"


def test_validate_step_grid():
    t = Tunable(0.0, 0.0, 1.0, step=0.25)
    assert t.validate(0.5) == (0.5, None)
    assert t.validate(0.3)[1] == "off_step"


def test_validate_non_number():
    assert Tunable(0.0, 0.0, 1.0).validate("nope")[1] == "not_a_number"


def test_construction_errors():
    with pytest.raises(ValueError):
        Tunable(0.5, 1.0, 0.0)              # low > high
    with pytest.raises(ValueError):
        Tunable(10.0, 0.0, 1.0)            # default out of bounds
    with pytest.raises(ValueError):
        Tunable(0.0, 0.0, 1.0, step=-1)   # bad step


class _SchemaOp(Operator):
    learning_rate = Tunable(3e-4, 1e-5, 1e-3)
    grad_steps    = Tunable(25_000, 2_000, 100_000, int=True)

    @observe
    def o(self, snap) -> dict:
        return {}


def test_get_schema_renders_boundary_into_update_param():
    op = _SchemaOp(None)
    schema = op.get_actions().get_schema("openai")
    up = next(s for s in schema if s["function"]["name"] == "update_param")
    desc = up["function"]["description"]
    assert "learning_rate in [1e-05, 0.001]" in desc
    assert "grad_steps in [2000.0, 100000.0] (integer)" in desc


def test_tunables_seed_as_scalars_not_descriptors():
    op = _SchemaOp(None)
    assert op.learning_rate == pytest.approx(3e-4)   # proxy returns the scalar
    assert op.tunables == {"learning_rate": pytest.approx(3e-4), "grad_steps": 25_000}
