"""Unit tests for goal types — no AsyncFlow, no fixtures."""

import time

import pytest

from radical.adl.goals import ResourceGoal, ScientificGoal
from radical.adl.state import Snapshot


def _snap(**runtime) -> Snapshot:
    """Minimal Snapshot for goal evaluation tests."""
    return Snapshot(
        operator_id="test",
        cycle=0,
        timestamp=time.time(),
        source="self",
        observations={},
        runtime=runtime,
        objectives={},
        artifacts={},
        directives={},
    )


def _snap_with_objectives(**objectives) -> Snapshot:
    return Snapshot(
        operator_id="test",
        cycle=0,
        timestamp=time.time(),
        source="self",
        observations={},
        runtime={},
        objectives=objectives,
        artifacts={},
        directives={},
    )


# ── ScientificGoal ─────────────────────────────────────────────────────────────

class TestScientificGoalMaximize:

    def test_satisfied_at_threshold(self):
        g = ScientificGoal("g", "metric", 10.0, "maximize")
        assert g.satisfied({"metric": 10.1}, _snap())

    def test_satisfied_above_threshold(self):
        g = ScientificGoal("g", "metric", 10.0, "maximize")
        assert g.satisfied({"metric": 100.0}, _snap())

    def test_not_satisfied_at_exact_threshold(self):
        g = ScientificGoal("g", "metric", 10.0, "maximize")
        assert not g.satisfied({"metric": 10.0}, _snap())

    def test_not_satisfied_below_threshold(self):
        g = ScientificGoal("g", "metric", 10.0, "maximize")
        assert not g.satisfied({"metric": 9.99}, _snap())


class TestScientificGoalMinimize:

    def test_satisfied_at_threshold(self):
        g = ScientificGoal("g", "metric", 0.01, "minimize")
        assert g.satisfied({"metric": 0.009}, _snap())

    def test_satisfied_well_below_threshold(self):
        g = ScientificGoal("g", "metric", 0.01, "minimize")
        assert g.satisfied({"metric": 0.0001}, _snap())

    def test_not_satisfied_at_exact_threshold(self):
        g = ScientificGoal("g", "metric", 0.01, "minimize")
        assert not g.satisfied({"metric": 0.01}, _snap())

    def test_not_satisfied_above_threshold(self):
        g = ScientificGoal("g", "metric", 0.01, "minimize")
        assert not g.satisfied({"metric": 0.02}, _snap())


class TestScientificGoalMissingMetric:

    def test_missing_from_obs_and_objectives(self):
        g = ScientificGoal("g", "metric", 1.0, "maximize")
        assert not g.satisfied({}, _snap())

    def test_missing_from_obs_but_in_objectives(self):
        g = ScientificGoal("g", "metric", 1.0, "maximize")
        snap = _snap_with_objectives(metric=2.0)
        assert g.satisfied({}, snap)

    def test_obs_takes_priority_over_objectives(self):
        g = ScientificGoal("g", "metric", 1.0, "maximize")
        snap = _snap_with_objectives(metric=0.5)  # not satisfied via objectives
        assert g.satisfied({"metric": 2.0}, snap)  # obs says satisfied → True


# ── ResourceGoal ───────────────────────────────────────────────────────────────

class TestResourceGoalBelow:

    def test_satisfied_below_threshold(self):
        g = ResourceGoal("cpu", "cpu_percent", 90.0, "below")
        snap = _snap(cpu_percent=85.0)
        assert g.satisfied({}, snap)

    def test_not_satisfied_at_threshold(self):
        g = ResourceGoal("cpu", "cpu_percent", 90.0, "below")
        snap = _snap(cpu_percent=90.0)
        assert not g.satisfied({}, snap)

    def test_not_satisfied_above_threshold(self):
        g = ResourceGoal("cpu", "cpu_percent", 90.0, "below")
        snap = _snap(cpu_percent=95.0)
        assert not g.satisfied({}, snap)

    def test_missing_metric_not_satisfied(self):
        g = ResourceGoal("cpu", "cpu_percent", 90.0, "below")
        assert not g.satisfied({}, _snap())


class TestResourceGoalAbove:

    def test_satisfied_above_threshold(self):
        g = ResourceGoal("gpu", "gpu_percent", 50.0, "above")
        snap = _snap(gpu_percent=60.0)
        assert g.satisfied({}, snap)

    def test_not_satisfied_below_threshold(self):
        g = ResourceGoal("gpu", "gpu_percent", 50.0, "above")
        snap = _snap(gpu_percent=40.0)
        assert not g.satisfied({}, snap)
