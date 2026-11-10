"""ChainPolicy — merge Decisions from multiple policies."""

from __future__ import annotations

from radical.adr.decision import Decision
from radical.adr.policy.base import Policy


class ChainPolicy(Policy):
    """Run all policies and merge their Decisions.

    Actions are unioned; ``stop=True`` if ANY policy requests stop.
    Directives are merged left-to-right (later policies win on key conflicts).
    """

    def __init__(self, policies: list[Policy]) -> None:
        super().__init__()
        self._policies = policies

    async def decide(self, obs: dict) -> Decision:
        all_actions:    list = []
        all_directives: dict = {}
        stop                 = False

        for policy in self._policies:
            d = await policy.decide(obs)
            all_actions.extend(d.actions)
            all_directives.update(d.directives)
            if d.stop:
                stop = True

        return Decision(actions=all_actions, directives=all_directives, stop=stop)
