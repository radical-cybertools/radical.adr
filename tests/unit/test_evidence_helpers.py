"""Unit tests for Phase-2 evidence helpers: Snapshot.results / rejections,
Operator.trend / stalled, and the decide_when gate (_should_decide).

Pure logic — no engine. Operators are built with a dummy engine (None); the
helpers touch only state, not execution.
"""

from __future__ import annotations

import pytest

from radical.adr import Operator, observe
from radical.adr.state import Snapshot


class _Op(Operator):
    metric: float = 0.0

    @observe
    def o(self, snap) -> dict:
        return {}


def _snap(cycle: int, objectives: dict, observations: dict | None = None) -> Snapshot:
    return Snapshot(
        operator_id="x", cycle=cycle, timestamp=0.0, source="self",
        observations=observations or {}, runtime={}, objectives=objectives,
        artifacts={}, directives={},
    )


# ── Snapshot.results / rejections ─────────────────────────────────────────────

def test_snapshot_results_and_rejections_filter_by_prefix():
    s = _snap(1, {}, observations={
        "result.a": {"v": 1},
        "error.b": {"e": "boom"},
        "child:c:k": 9,
        "rejected_param.lr": {"requested": 10.0, "reason": "out_of_bounds"},
    })
    assert s.results == [{"v": 1}]
    assert s.rejections == [{"requested": 10.0, "reason": "out_of_bounds"}]


# ── trend / stalled ───────────────────────────────────────────────────────────

def _push(op: _Op, values: list[float], live: float) -> None:
    """Seed cycle_history with `values` (each a distinct cycle) + a trailing
    'current' snapshot (excluded by cycle_history), and set the live value."""
    h = op.state.history
    h.clear()
    for i, v in enumerate(values):
        h.append(_snap(i, {"metric": v}))
    h.append(_snap(99, {"metric": -999.0}))     # current cycle → excluded
    op.state.objectives["metric"] = live


def test_trend_improving_vs_regressing_by_direction():
    op = _Op(None)
    _push(op, [0.0, 0.1], live=0.2)             # distinct series 0.0, 0.1, 0.2
    assert op.trend("metric", "maximize") == "improving"
    assert op.trend("metric", "minimize") == "regressing"


def test_trend_flat_with_too_few_points():
    op = _Op(None)
    _push(op, [], live=0.5)                      # single distinct value
    assert op.trend("metric", "maximize") == "flat"


def test_stalled_counts_distinct_values_since_best():
    op = _Op(None)
    _push(op, [0.0, 0.2], live=0.1)             # best 0.2 then 0.1 → 1 since best
    assert op.stalled("metric", "maximize", 1) is True
    assert op.stalled("metric", "maximize", 2) is False
    # improving run never stalls
    _push(op, [0.0, 0.1], live=0.2)
    assert op.stalled("metric", "maximize", 1) is False


def test_trend_stalled_reject_bad_direction():
    op = _Op(None)
    _push(op, [0.0], live=0.1)
    with pytest.raises(ValueError):
        op.trend("metric", "up")
    with pytest.raises(ValueError):
        op.stalled("metric", "up", 1)
    with pytest.raises(ValueError):
        op.stalled("metric", "maximize", 0)


# ── decide_when gate ──────────────────────────────────────────────────────────

def test_should_decide_modes():
    op = _Op(None)
    no_res = _snap(0, {})
    with_res = _snap(0, {}, observations={"result.a": {}})

    op.decide_when = None
    assert op._should_decide(no_res) is True

    op.decide_when = "idle"
    assert op._should_decide(no_res) is True        # nothing in flight
    op.state.runtime["u1"] = "RUNNING"
    assert op._should_decide(no_res) is False        # inflight == 1
    op.state.runtime.clear()

    op.decide_when = "new_result"
    assert op._should_decide(no_res) is False
    assert op._should_decide(with_res) is True

    op.decide_when = "bogus"
    with pytest.raises(ValueError):
        op._should_decide(no_res)
