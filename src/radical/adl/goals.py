"""Goal types for operator convergence criteria."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radical.adl.state import Snapshot


@dataclass
class Goal:
    """Base class for operator goals.

    Subclasses implement ``satisfied(obs, snapshot) → bool``.
    The operator stops when ALL its goals are satisfied.
    """
    name:      str
    metric:    str
    threshold: float

    def satisfied(self, obs: dict[str, Any], snapshot: "Snapshot") -> bool:
        raise NotImplementedError


@dataclass
class ScientificGoal(Goal):
    """Satisfied when ``obs[metric]`` crosses the threshold.

    direction="minimize" → satisfied when obs[metric] < threshold
    direction="maximize" → satisfied when obs[metric] > threshold
    """
    direction: str = "minimize"

    def satisfied(self, obs: dict[str, Any], snapshot: "Snapshot") -> bool:
        # Check obs first; fall back to snapshot.objectives so goals work even
        # when @observe doesn't explicitly surface the metric.
        val = obs.get(self.metric)
        if val is None:
            val = snapshot.objectives.get(self.metric)
        if val is None:
            return False
        if self.direction == "minimize":
            return float(val) < self.threshold
        return float(val) > self.threshold


@dataclass
class ResourceGoal(Goal):
    """Satisfied when ``snapshot.runtime[metric]`` stays within bound.

    bound="below" → satisfied when runtime[metric] < threshold
    bound="above" → satisfied when runtime[metric] > threshold
    """
    bound: str = "below"

    def satisfied(self, obs: dict[str, Any], snapshot: "Snapshot") -> bool:
        val = snapshot.runtime.get(self.metric)
        if val is None:
            return False
        if self.bound == "below":
            return float(val) < self.threshold
        return float(val) > self.threshold
