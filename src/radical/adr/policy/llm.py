"""LLMPolicy — thin base class for LLM-driven policies."""

from __future__ import annotations

import json
from collections import deque

from radical.adr.policy.base import Policy

# Bounded history of per-call usage entries. Mirrors state.py's history deque
# pattern; sized larger because usage is recorded per LLM call (often >1/cycle).
_USAGE_HISTORY_MAXLEN = 256


class LLMPolicy(Policy):
    """Thin base class for LLM-driven policies.

    Provides helpers — subclass and add ``@decide``::

        class OpenRouterPolicy(LLMPolicy):
            system_prompt = "You are an optimization expert."

            def __init__(self, api_key: str) -> None:
                super().__init__()
                self.client = instructor.from_openai(AsyncOpenAI(api_key=api_key, ...))

            @decide
            async def run(self, obs: dict) -> Decision:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": self.render_observation(obs)},
                ]
                sd = await self.client.chat.completions.create(
                    model=MODEL, response_model=SearchDecision, messages=messages)
                self.record_usage(tokens=sd.usage.total_tokens, cost=...)
                return Decision(stop=sd.stop, actions=[...])

    ``usage_history`` is purely local bookkeeping — it never flows through
    ``state``, ``snapshot``, or ``@observe``. ADR grows no telemetry pipeline;
    this is only a standard field so tooling can read per-call usage uniformly.
    """

    system_prompt: str = ""

    def __init__(
        self,
        *,
        primary:  "Policy | None" = None,
        fallback: "Policy | None" = None,
        timeout:  float | None    = None,
    ) -> None:
        super().__init__(primary=primary, fallback=fallback, timeout=timeout)
        self.usage_history: deque = deque(maxlen=_USAGE_HISTORY_MAXLEN)

    def render_observation(self, obs: dict) -> str:
        """Convert obs dict to a user-turn prompt string. Default: JSON dump."""
        return json.dumps(obs, default=str)

    def record_usage(
        self,
        *,
        tokens:  int | None   = None,
        latency: float | None = None,
        cost:    float | None = None,
    ) -> None:
        """Record one LLM call's usage stats.

        Call manually after your vendor call inside ``@decide``. Any subset of
        fields may be provided. Purely local — never flows through state,
        snapshot, or ``@observe``.
        """
        self.usage_history.append({"tokens": tokens, "latency": latency, "cost": cost})
