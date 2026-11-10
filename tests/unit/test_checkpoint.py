"""Unit tests for Operator.save_checkpoint / load_checkpoint round-trip (no engine).

Steering state (cycle, objectives, runtime, task_context, runtime goals, goal
streaks) restores onto a fresh operator; in-flight tasks are recorded as LOST.
"""

from radical.adr import Goal, Operator, goals, observe


class _CkptOp(Operator):
    radius: float = 5.0

    @observe
    def o(self, s):
        return {}

    @goals
    def g(self):
        return Goal("c", "radius", 0.5)


def test_checkpoint_restores_steering_state(tmp_path):
    path = tmp_path / "ck.json"
    op = _CkptOp(engine=None)
    op.radius = 1.25
    op.state.cycle = 9
    op.state.runtime.update({"t1": "RUNNING", "t2": "DONE"})
    op.state.task_context["t1"] = {"name": "evaluate", "kwargs": {"x": 1}}
    op.runtime_goals.append(Goal("rg", "m", 2.0, for_cycles=3))
    op._goal_streaks["c"] = 2
    op.save_checkpoint(str(path))

    fresh = _CkptOp(engine=None)
    fresh.load_checkpoint(str(path))
    assert fresh.state.cycle == 9
    assert fresh.radius == 1.25
    assert fresh.state.runtime["t2"] == "DONE"
    assert fresh.runtime_goals[0].name == "rg" and fresh.runtime_goals[0].for_cycles == 3
    assert fresh._goal_streaks["c"] == 2


def test_checkpoint_marks_inflight_tasks_lost(tmp_path):
    path = tmp_path / "ck.json"
    op = _CkptOp(engine=None)
    op.state.runtime.update({"t1": "RUNNING", "t2": "PENDING", "t3": "DONE"})
    op.save_checkpoint(str(path))

    fresh = _CkptOp(engine=None)
    fresh.load_checkpoint(str(path))
    assert fresh.state.runtime["t1"] == "LOST"     # in-flight at save time
    assert fresh.state.runtime["t2"] == "LOST"
    assert fresh.state.runtime["t3"] == "DONE"      # already terminal
