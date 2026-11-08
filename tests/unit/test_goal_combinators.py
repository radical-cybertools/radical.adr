"""Unit tests for AnyGoal/AllGoal + Goal(for_cycles) — no engine (Item 5).

Covers:
  - AnyGoal (OR), AllGoal (AND), nesting
  - bare Goal / list unchanged (implicit AND)
  - for_cycles debounce: a goal must hold N consecutive cycles to stop
  - all leaves evaluate every cycle (streaks stay correct under short-circuit)
  - checkpoint round-trip of composite runtime goals
"""

import time

from radical.adr import AllGoal, AnyGoal, Decision, Goal, Operator, goals, observe
from radical.adr.goals import goal_from_dict, goal_to_dict
from radical.adr.state import Snapshot


def _snap(**objectives) -> Snapshot:
    return Snapshot(operator_id="t", cycle=0, timestamp=time.time(), source="self",
                    observations={}, runtime={}, objectives=objectives,
                    artifacts={}, directives={})


class _Op(Operator):
    @observe
    def o(self, s):
        return {}


def _check(op, obs, snap):
    """Run the operator's goal check and return whether it forced stop."""
    d = Decision()
    op._check_goals(obs, snap, d)
    return d.stop


# ── composites ───────────────────────────────────────────────────────────────

def test_anygoal_or_satisfied_on_either():
    g = AnyGoal([Goal("a", "x", 1.0, direction="maximize"),
                 Goal("b", "y", 1.0, direction="maximize")])
    assert g.satisfied({"x": 5.0, "y": 0.0}, _snap()) is True     # only x
    assert g.satisfied({"x": 0.0, "y": 0.0}, _snap()) is False    # neither


def test_allgoal_and_requires_both():
    g = AllGoal([Goal("a", "x", 1.0, direction="maximize"),
                 Goal("b", "y", 1.0, direction="maximize")])
    assert g.satisfied({"x": 5.0, "y": 5.0}, _snap()) is True
    assert g.satisfied({"x": 5.0, "y": 0.0}, _snap()) is False


def test_operator_anygoal_stops_on_either():
    class Op(_Op):
        @goals
        def g(self):
            return AnyGoal([Goal("conv", "radius", 0.5),
                            Goal("budget", "spent", 100, direction="maximize")])
    op = Op(engine=None)
    assert _check(op, {"radius": 1.0, "spent": 150}, _snap()) is True   # budget hit
    op2 = Op(engine=None)
    assert _check(op2, {"radius": 1.0, "spent": 0}, _snap()) is False


def test_bare_list_is_implicit_and():
    class Op(_Op):
        @goals
        def g(self):
            return [Goal("a", "x", 1.0, direction="maximize"),
                    Goal("b", "y", 1.0, direction="maximize")]
    op = Op(engine=None)
    assert _check(op, {"x": 5, "y": 0}, _snap()) is False
    assert _check(op, {"x": 5, "y": 5}, _snap()) is True


# ── for_cycles debounce ──────────────────────────────────────────────────────

def test_for_cycles_requires_consecutive():
    class Op(_Op):
        @goals
        def g(self):
            return Goal("conv", "radius", 0.5, for_cycles=3)
    op = Op(engine=None)
    assert _check(op, {"radius": 0.1}, _snap()) is False   # streak 1
    assert _check(op, {"radius": 0.1}, _snap()) is False   # streak 2
    assert _check(op, {"radius": 0.1}, _snap()) is True    # streak 3 → stop


def test_for_cycles_streak_resets_on_miss():
    class Op(_Op):
        @goals
        def g(self):
            return Goal("conv", "radius", 0.5, for_cycles=2)
    op = Op(engine=None)
    assert _check(op, {"radius": 0.1}, _snap()) is False   # streak 1
    assert _check(op, {"radius": 9.0}, _snap()) is False   # miss → reset
    assert _check(op, {"radius": 0.1}, _snap()) is False   # streak 1 again
    assert _check(op, {"radius": 0.1}, _snap()) is True    # streak 2 → stop


def test_anygoal_streaks_all_leaves_update():
    # Under AnyGoal, a satisfied early member must not stop later members'
    # streaks from advancing (full evaluation, no short-circuit).
    class Op(_Op):
        @goals
        def g(self):
            return AnyGoal([Goal("a", "x", 1.0, direction="maximize", for_cycles=1),
                            Goal("b", "y", 1.0, direction="maximize", for_cycles=2)])
    op = Op(engine=None)
    # a satisfied immediately → AnyGoal already true, but b's streak still tracks
    assert _check(op, {"x": 5, "y": 5}, _snap()) is True
    assert op._goal_streaks["b"] == 1     # b's streak advanced despite a being true


# ── checkpoint round-trip of composite goals ─────────────────────────────────

def test_goal_dict_roundtrip_nested():
    g = AnyGoal([Goal("a", "x", 1.0, for_cycles=2),
                 AllGoal([Goal("b", "y", 2.0), Goal("c", "z", 3.0, direction="maximize")])],
                name="top")
    back = goal_from_dict(goal_to_dict(g))
    assert type(back).__name__ == "AnyGoal" and back.name == "top"
    assert back.goals[0].for_cycles == 2
    assert type(back.goals[1]).__name__ == "AllGoal"
    assert back.goals[1].goals[1].direction == "maximize"
