"""Unit test for Operator.inflight — count of PENDING + RUNNING tasks (no engine)."""

from radical.adr import Operator, observe


class _Op(Operator):
    @observe
    def o(self, s):
        return {}


def test_inflight_counts_only_pending_and_running():
    op = _Op(engine=None)
    op.state.runtime.update({"a": "PENDING", "b": "RUNNING", "c": "DONE", "d": "FAILED"})
    assert op.inflight == 2
