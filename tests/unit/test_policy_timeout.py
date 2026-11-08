"""Unit tests for Policy(primary, fallback, timeout) composition — no engine.

Covers Item 1: a hung primary call must trigger the fallback via timeout,
distinct from the raise-path fallback, and default timeout=None preserves the
original behavior exactly.
"""

import asyncio
import logging


from radical.adr.decision import Decision
from radical.adr.policy.base import Policy, decide


class _HungPrimary(Policy):
    @decide
    async def run(self, obs: dict) -> Decision:
        await asyncio.sleep(10)            # never completes within the test timeout
        return Decision(stop=True)


class _SlowPrimary(Policy):
    @decide
    async def run(self, obs: dict) -> Decision:
        await asyncio.sleep(0.05)          # slow but finite
        return Decision(stop=True, directives={"who": "primary"})


class _RaisingPrimary(Policy):
    @decide
    async def run(self, obs: dict) -> Decision:
        raise RuntimeError("primary boom")


class _Fallback(Policy):
    @decide
    async def run(self, obs: dict) -> Decision:
        return Decision(stop=False, directives={"who": "fallback"})


async def test_hung_primary_falls_back_via_timeout():
    pol = Policy(primary=_HungPrimary(), fallback=_Fallback(), timeout=0.05)
    result = await asyncio.wait_for(pol.decide({"cycle": 0}), timeout=2.0)
    assert result.directives["who"] == "fallback"


async def test_default_timeout_none_returns_primary():
    # No timeout → a slow-but-finite primary still wins; default behavior unchanged.
    pol = Policy(primary=_SlowPrimary(), fallback=_Fallback())
    result = await pol.decide({"cycle": 0})
    assert result.directives["who"] == "primary"


async def test_raising_primary_falls_back_with_timeout_set():
    # Regression: raise-path fallback still works even when a timeout is configured.
    pol = Policy(primary=_RaisingPrimary(), fallback=_Fallback(), timeout=5.0)
    result = await pol.decide({"cycle": 0})
    assert result.directives["who"] == "fallback"


async def test_timeout_and_raise_log_distinguishably(caplog):
    with caplog.at_level(logging.WARNING, logger="radical.adr.policy.base"):
        await Policy(primary=_HungPrimary(), fallback=_Fallback(), timeout=0.05).decide({})
    timeout_msgs = [r for r in caplog.records if "timed out" in r.getMessage()]
    assert timeout_msgs, "timeout path should log a distinct 'timed out' warning"
    assert all(r.levelno == logging.WARNING for r in timeout_msgs)
    assert not any(r.exc_info for r in timeout_msgs), "timeout log must carry no traceback"

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="radical.adr.policy.base"):
        await Policy(primary=_RaisingPrimary(), fallback=_Fallback(), timeout=5.0).decide({})
    raised_msgs = [r for r in caplog.records if "raised" in r.getMessage()]
    assert raised_msgs, "raise path should log a distinct 'raised' message"
    assert any(r.exc_info for r in raised_msgs), "raise log should carry a traceback"
