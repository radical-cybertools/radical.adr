"""Annotated operator attributes are proxied into state.objectives on every Python.

This module deliberately has no ``from __future__ import annotations``: on Python 3.14
(PEP 649/749) class annotations are then evaluated lazily and are not stored in the
class ``__dict__``, which is where ``Operator.__init_subclass__`` used to look.
"""

import sys

import pytest

from radical.adr import Operator, act, observe


class _StateOp(Operator):
    radius: float = 5.0
    center: float = 0.0

    @observe
    def o(self, snap) -> dict:
        return {}


class _ChildStateOp(_StateOp):
    width: int = 3


def test_annotated_attrs_are_state_keys_and_seeded():
    op = _StateOp(None)
    assert {"radius", "center"} <= _StateOp._adl_state_keys
    assert op.state.objectives == {"radius": 5.0, "center": 0.0}


def test_attribute_and_objectives_stay_in_sync():
    op = _StateOp(None)
    op.radius = 1.25                              # attribute write → objectives
    assert op.state.objectives["radius"] == 1.25
    op.state.objectives["center"] = 2.0           # objectives write → attribute read
    assert op.center == 2.0


def test_inherited_annotations_are_collected():
    op = _ChildStateOp(None)
    assert {"radius", "center", "width"} <= _ChildStateOp._adl_state_keys
    assert op.state.objectives == {"radius": 5.0, "center": 0.0, "width": 3}


@pytest.mark.skipif(sys.version_info < (3, 14),
                    reason="before 3.14 an undefined annotation fails at class creation")
def test_annotation_naming_an_undefined_type_is_not_evaluated():
    class _LazyOp(Operator):
        mol: UndefinedMolecule = None             # noqa: F821  (never evaluated)

        @observe
        def o(self, snap) -> dict:
            return {}

    assert "mol" in _LazyOp._adl_state_keys
    assert _LazyOp(None).state.objectives == {"mol": None}


@pytest.mark.skipif(sys.version_info < (3, 14),
                    reason="before 3.14 an undefined annotation fails at definition")
def test_act_annotated_with_an_undefined_type_can_be_rendered_and_spawned():
    # On 3.14 this is the TYPE_CHECKING-only import idiom: the name is never bound.
    class _DockOp(Operator):
        @act
        async def dock(self, mol: UndefinedMolecule, score: float = 0.0):   # noqa: F821
            return {"mol": mol}

        @observe
        def o(self, snap) -> dict:
            return {}

    op = _DockOp(None)
    schema = op.get_schema("openai")
    dock = next(s for s in schema if s["function"]["name"] == "dock")
    assert dock["function"]["parameters"]["properties"]["mol"] == {}      # untyped
    op._validate_actions([op.get_actions().dock(mol="m.pdb", score=1.0)])  # does not raise
