"""Integration test: acts.retry_task re-spawns a failed task over a real engine.

A task whose backend impl always raises is retried once from its recorded
task_context, producing a second tracked 'flaky' task (same name + kwargs).
"""

from __future__ import annotations

from radical.adr import Decision, Operator, Policy, act, decide, observe


async def _always_fail_impl(x: float) -> dict:
    raise ValueError("boom")


class _RetryOp(Operator):
    @act
    async def flaky(self, x: float):
        return self.engine.function_task(_always_fail_impl)(x=x)

    @observe
    def o(self, snap) -> dict:
        errors = [k.removeprefix("error.") for k in snap.observations
                  if k.startswith("error.")]
        n_flaky = sum(1 for c in self.state.task_context.values() if c["name"] == "flaky")
        return {"cycle": snap.cycle, "errors": errors, "n_flaky": n_flaky}


class _RetryPolicy(Policy):
    def __init__(self, op):
        super().__init__()
        self._act = op.get_actions()
        self.retried = False

    @decide
    async def run(self, obs) -> Decision:
        if obs["cycle"] == 0:
            return Decision(actions=[self._act.flaky(x=1.0)])
        if obs["errors"] and not self.retried:
            self.retried = True
            return Decision(actions=[self._act.retry_task(obs["errors"][0])])
        if obs["cycle"] >= 12 or (self.retried and obs["n_flaky"] >= 2):
            return Decision(stop=True)
        return Decision()


async def test_retry_task_respawns_failed_task(engine):
    op = _RetryOp(engine)
    op.policy = _RetryPolicy(op)
    async for _ in op.run():
        pass
    # original flaky + one retry → two 'flaky' entries recorded in task_context
    n_flaky = sum(1 for c in op.state.task_context.values() if c["name"] == "flaky")
    assert n_flaky >= 2
