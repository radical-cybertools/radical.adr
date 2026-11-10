"""Unit tests for replay_trace(operators=) — offline replay of hierarchical traces.

A SPAWN_OPERATOR's operator_cls is excluded from the recorded trace, so replay
back-fills it from the operators={name: Cls} map; without the map, replay is
unchanged.
"""

import json

from radical.adr import Decision, Operator, decide, observe, replay_trace
from radical.adr.decision import Action, ActionKind
from radical.adr.policy.base import Policy


class _Op(Operator):
    @observe
    def o(self, s):
        return {}


async def test_operators_map_backfills_operator_cls(tmp_path):
    trace = tmp_path / "t.jsonl"
    trace.write_text(json.dumps({"cycle": 0, "obs": {}, "decision": {}}) + "\n")

    class SpawningPolicy(Policy):
        @decide
        async def run(self, obs):
            return Decision(actions=[Action(kind=ActionKind.SPAWN_OPERATOR,
                                            task_name="refiner")])

    decisions = await replay_trace(str(trace), SpawningPolicy(),
                                   operators={"refiner": _Op})
    assert decisions[0].actions[0].operator_cls is _Op


async def test_without_operators_map_is_unchanged(tmp_path):
    trace = tmp_path / "t.jsonl"
    trace.write_text(json.dumps({"cycle": 0, "obs": {"x": 1}, "decision": {}}) + "\n")

    class FlatPolicy(Policy):
        @decide
        async def run(self, obs):
            return Decision()

    decisions = await replay_trace(str(trace), FlatPolicy())
    assert len(decisions) == 1
