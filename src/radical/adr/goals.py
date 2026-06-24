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
    """
    name:       str
    metric:     str
    threshold:  float
    direction:  str  = "minimize"
    inclusive:  bool = False
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
