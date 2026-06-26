"""Integration test: terminal task Futures are pruned, yet depends_on survives.

A two-stage DAG (produce -> consume depends_on produce) over a real engine, where
produce settles and is pruned from state.task_futures before consume runs. The
consumer must still receive produce's result (reconstructed from the bounded
state.resolved cache), and task_futures must not accumulate.
"""

from __future__ import annotations

from radical.adr import Decision, Operator, Policy, act, decide, observe


async def _produce_impl(x: float) -> dict:
    return {"v": x}


class _DagOp(Operator):
    @act
    async def produce(self, x: float):
        return self.engine.function_task(_produce_impl)(x=x)

    @act
    async def consume(self, dep):
        val = await dep                       # resolves even if produce was pruned
        return {"consumed": val}

    @observe
    def o(self, snap) -> dict:
        produced = {k.removeprefix("result."): v for k, v in snap.observations.items()
                    if k.startswith("result.") and isinstance(v, dict) and "v" in v}
        consumed = [v for k, v in snap.observations.items()
                    if k.startswith("result.") and isinstance(v, dict) and "consumed" in v]
        return {"cycle": snap.cycle, "produced": produced, "consumed": consumed}


class _DagPolicy(Policy):
    def __init__(self, op):
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs) -> Decision:
        if obs["consumed"]:
            return Decision(stop=True)
        if obs["produced"]:
            uid = next(iter(obs["produced"]))          # produce's uid (now terminal/pruned)
            return Decision(actions=[self._act.consume(depends_on=[uid])])
        if obs["cycle"] == 0:
            return Decision(actions=[self._act.produce(x=7.0)])
        if obs["cycle"] >= 12:
            return Decision(stop=True)
        return Decision()


async def test_consume_resolves_dep_after_produce_is_pruned(engine):
    op = _DagOp(engine)
    op.policy = _DagPolicy(op)
    final = None
    async for snap in op.run():
        final = snap

    consumed = [v for k, v in final.observations.items()
                if k.startswith("result.") and isinstance(v, dict) and "consumed" in v]
    assert consumed and consumed[0]["consumed"] == {"v": 7.0}
    # both tasks terminal → pruned; task_futures bounded, not accumulating
    assert len(op.state.task_futures) == 0
    assert any(s == "DONE" for s in op.state.runtime.values())
