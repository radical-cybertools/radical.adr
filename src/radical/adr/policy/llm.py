"""LLMPolicy — thin base class for LLM-driven policies."""

from __future__ import annotations

import json

from radical.adr.policy.base import Policy


class LLMPolicy(Policy):
    """Thin base class for LLM-driven policies.

    Provides two helpers — subclass and add ``@decide``::

        class OpenRouterPolicy(LLMPolicy):
            system_prompt = "You are an optimization expert."

            def __init__(self, api_key: str) -> None:
                self.client = instructor.from_openai(AsyncOpenAI(api_key=api_key, ...))

            @decide
            async def run(self, obs: dict) -> Decision:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": self.render_observation(obs)},
                ]
                sd = await self.client.chat.completions.create(
                    model=MODEL, response_model=SearchDecision, messages=messages)
                return Decision(stop=sd.stop, actions=[...])
    """

    system_prompt: str = ""

    def render_observation(self, obs: dict) -> str:
        """Convert obs dict to a user-turn prompt string. Default: JSON dump."""
        return json.dumps(obs, default=str)
