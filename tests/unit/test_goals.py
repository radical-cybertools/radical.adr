"""Unit tests for the unified Goal — no AsyncFlow, no fixtures."""

import time
import warnings

import pytest

from radical.adr.goals import Goal
from radical.adr.state import Snapshot


def _snap(**runtime) -> Snapshot:
    """Minimal Snapshot with optional runtime dict."""
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
    """Minimal Snapshot with objectives populated."""
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


# ── Goal ───────────────────────────────────────────────────────────────────────

def test_goal_minimize_satisfied():
    assert Goal("g", "val", 10.0).satisfied({"val": 5.0}, _snap()) is True


def test_goal_minimize_not_satisfied():
    assert Goal("g", "val", 10.0).satisfied({"val": 15.0}, _snap()) is False


def test_goal_maximize_satisfied():
    assert Goal("g", "val", 10.0, direction="maximize").satisfied({"val": 15.0}, _snap()) is True


def test_goal_maximize_not_satisfied():
    assert Goal("g", "val", 10.0, direction="maximize").satisfied({"val": 5.0}, _snap()) is False


def test_goal_fallback_to_objectives():
    # Metric absent from obs → fallback to snapshot.objectives
    snap = _snap_with_objectives(radius=0.5)
    assert Goal("g", "radius", 1.0).satisfied({}, snap) is True


def test_goal_missing_warns():
    # Metric absent from both obs and objectives → False + UserWarning
    g = Goal("g", "missing_metric", 1.0)
    with pytest.warns(UserWarning, match="missing_metric"):
        result = g.satisfied({}, _snap())
    assert result is False


def test_goal_warn_fires_once():
    # _warn_once: warning fires exactly once; subsequent calls are silent
    g = Goal("g", "missing_metric", 1.0)
    with pytest.warns(UserWarning):
        g.satisfied({}, _snap())
    # Second call must NOT fire another warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g.satisfied({}, _snap())   # would raise AssertionError if warning fired


def test_goal_runtime_not_checked_automatically():
    # snapshot.runtime is NOT a lookup fallback — @observe must surface it explicitly.
    # This test documents and enforces the @observe contract.
    snap = _snap(cpu_percent=80.0)  # runtime has cpu_percent, but obs does not
    g = Goal("g", "cpu_percent", 90.0)
    with pytest.warns(UserWarning):
        result = g.satisfied({}, snap)
    assert result is False


