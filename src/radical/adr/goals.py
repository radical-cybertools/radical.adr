"""Goal — the only convergence/constraint abstraction in ADR."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radical.adr.state import Snapshot


@dataclass
class Goal:
    """Satisfied when ``obs[metric]`` crosses the threshold.

    direction="minimize" → satisfied when obs[metric] <  threshold
    direction="maximize" → satisfied when obs[metric] >  threshold

    ``inclusive=True`` flips the comparison to ``<=`` / ``>=`` so a discrete
    goal like "reach 5 hits" fires at exactly 5 instead of requiring 6. The
    default ``inclusive=False`` preserves the strict-inequality behavior exactly.

    Metric lookup order:
      1. ``obs[metric]``                 — primary: what @observe explicitly returns
      2. ``snapshot.objectives[metric]`` — fallback: operator state (e.g. radius, center)

    Goals are source-agnostic. Whether the metric comes from task results, operator
    state, a telemetry subscription, or any other external source is entirely
    determined by what the ``@observe`` method puts into obs.

    Dynamic goals proposed by a policy via ``Decision(goals=[Goal(...)])`` are
    evaluated each cycle alongside statically declared ``@goals`` — same type,
    same contract, same runtime upsert/remove semantics.

    ``for_cycles=N`` (default 1) debounces the goal: it counts as met only after
    it has been satisfied for N *consecutive* cycles, so a one-cycle fluke does
    not stop the operator. ``satisfied()`` itself is instantaneous — the
    consecutive-cycle streak is tracked by the operator's goal-check loop.
    """
    name:       str
    metric:     str
    threshold:  float
    direction:  str  = "minimize"
    inclusive:  bool = False
    for_cycles: int  = 1
    _warn_once: bool = field(default=True, init=False, repr=False, compare=False)

    def satisfied(self, obs: dict[str, Any], snapshot: "Snapshot") -> bool:
        val = obs.get(self.metric)
        if val is None:
            val = snapshot.objectives.get(self.metric)
        if val is None:
            if self._warn_once:
                warnings.warn(
                    f"Goal '{self.name}': metric '{self.metric}' not found in obs "
                    "or snapshot.objectives — returning False. "
                    "Ensure @observe surfaces this metric.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warn_once = False
            return False
        if self.direction == "minimize":
            return float(val) <= self.threshold if self.inclusive else float(val) < self.threshold
        return float(val) >= self.threshold if self.inclusive else float(val) > self.threshold


class _GoalGroup:
    """Base for composite goals — combine member goals with any/all semantics.

    Members may be plain ``Goal`` objects or nested groups, so arbitrary
    AND/OR convergence trees are expressible. ``name`` participates in runtime
    goal upsert/remove just like a plain ``Goal``.
    """

    _combine = staticmethod(all)        # overridden by subclasses
    _default_name = "goal_group"

    def __init__(self, goals: "list", name: str | None = None) -> None:
        self.goals = list(goals)
        self.name  = name or self._default_name

    def satisfied(self, obs: dict[str, Any], snapshot: "Snapshot") -> bool:
        """Instantaneous combine of members (debounce is applied by the loop)."""
        return type(self)._combine([g.satisfied(obs, snapshot) for g in self.goals])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, goals={self.goals!r})"


class AllGoal(_GoalGroup):
    """Satisfied when **all** member goals are satisfied (explicit AND).

    The same semantics as a bare list returned from ``@goals``, but composable —
    use it as a member of an ``AnyGoal`` to express mixed AND/OR convergence.
    """
    _combine = staticmethod(all)
    _default_name = "all_goals"


class AnyGoal(_GoalGroup):
    """Satisfied when **any** member goal is satisfied (OR).

    Lets the operator stop on the first of several alternative conditions —
    e.g. ``AnyGoal([converged, budget_exhausted])``.
    """
    _combine = staticmethod(any)
    _default_name = "any_goals"


# ── (De)serialization — used by Operator checkpointing ───────────────────────

def goal_to_dict(goal) -> dict:
    """Serialize a Goal / AnyGoal / AllGoal tree to a JSON-safe dict."""
    if isinstance(goal, _GoalGroup):
        return {"kind": type(goal).__name__, "name": goal.name,
                "goals": [goal_to_dict(g) for g in goal.goals]}
    return {"kind": "Goal", "name": goal.name, "metric": goal.metric,
            "threshold": goal.threshold, "direction": goal.direction,
            "inclusive": goal.inclusive, "for_cycles": goal.for_cycles}


def goal_from_dict(d: dict):
    """Reconstruct a Goal / AnyGoal / AllGoal tree from :func:`goal_to_dict`."""
    kind = d.get("kind", "Goal")
    if kind == "Goal":
        return Goal(name=d["name"], metric=d["metric"], threshold=d["threshold"],
                    direction=d.get("direction", "minimize"),
                    inclusive=d.get("inclusive", False),
                    for_cycles=d.get("for_cycles", 1))
    members = [goal_from_dict(g) for g in d.get("goals", [])]
    cls = AllGoal if kind == "AllGoal" else AnyGoal
    return cls(members, name=d.get("name"))
