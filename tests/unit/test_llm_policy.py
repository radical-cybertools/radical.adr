"""Unit tests for LLMPolicy.usage_history / record_usage — no engine.

Covers Item 6b: bounded local usage bookkeeping, partial fields, FIFO eviction,
@decide still dispatches after the new __init__, and composition with Item 1's
timeout constructor.
"""

from radical.adr.decision import Decision
from radical.adr.policy.base import decide
from radical.adr.policy.llm import LLMPolicy, _USAGE_HISTORY_MAXLEN


def test_usage_history_starts_empty():
    assert len(LLMPolicy().usage_history) == 0


def test_record_usage_appends_fields():
    pol = LLMPolicy()
    pol.record_usage(tokens=1200, latency=0.8, cost=0.003)
    assert list(pol.usage_history) == [{"tokens": 1200, "latency": 0.8, "cost": 0.003}]


def test_record_usage_partial_fields():
    pol = LLMPolicy()
    pol.record_usage(tokens=500)
    assert pol.usage_history[-1] == {"tokens": 500, "latency": None, "cost": None}


def test_usage_history_evicts_oldest():
    pol = LLMPolicy()
    for i in range(_USAGE_HISTORY_MAXLEN + 10):
        pol.record_usage(tokens=i)
    assert len(pol.usage_history) == _USAGE_HISTORY_MAXLEN
    # oldest 10 evicted → first remaining is token=10
    assert pol.usage_history[0]["tokens"] == 10
    assert pol.usage_history[-1]["tokens"] == _USAGE_HISTORY_MAXLEN + 9


class _SubLLM(LLMPolicy):
    @decide
    async def run(self, obs: dict) -> Decision:
        self.record_usage(tokens=1)
        return Decision(stop=True)


async def test_subclass_decide_still_dispatches():
    pol = _SubLLM()
    result = await pol.decide({"cycle": 0})
    assert result.stop is True
    assert len(pol.usage_history) == 1


def test_composes_with_timeout_constructor():
    # Item 6b forwards primary/fallback/timeout to Policy.__init__ (Item 1).
    pol = LLMPolicy(primary=LLMPolicy(), fallback=LLMPolicy(), timeout=0.1)
    assert pol._timeout == 0.1
    assert len(pol.usage_history) == 0
