"""Policy — unified base, composition class, and @decide decorator."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from radical.adl.decision import Decision

log = logging.getLogger(__name__)


def decide(fn: Callable) -> Callable:
    """Mark a method as the policy's decision entry point.

    Mirrors ``@tool``, ``@observe``, ``@goals`` on the Operator side.
    The decorated method receives ``obs: dict`` and returns a ``Decision``.
    The method may be named anything.

    Example::

        class SearchPolicy(Policy):

            @decide
            async def run(self, obs: dict) -> Decision:
                if obs["cycle"] >= 15:
                    return Decision(stop=True)
                ...
    """
    fn._adl_decide = True  # type: ignore[attr-defined]
    return fn


class Policy:
    """Base class for all ADL policies, and a composition class.

    **Subclass with @decide** (primary usage)::

        class SearchPolicy(Policy):
            def __init__(self, op):
                self._tools = op.get_tools()

            @decide
            async def run(self, obs: dict) -> Decision:
                if obs["cycle"] >= 15:
                    return Decision(stop=True)
                ...

    **Composition** — combine primary and fallback::

        op.policy = Policy(
            primary  = MyLLMPolicy(),
            fallback = MyRulePolicy(),
        )

    When used as composition, ``decide()`` tries ``primary`` first.
    If it raises or returns ``None``, ``fallback`` is tried.
    """

    _adl_decide_fn: Callable | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._adl_decide_fn = None
        for name, val in vars(cls).items():
            if callable(val) and getattr(val, "_adl_decide", False):
                cls._adl_decide_fn = val

    def __init__(
        self,
        *,
        primary:  "Policy | None" = None,
        fallback: "Policy | None" = None,
    ) -> None:
        self._primary  = primary
        self._fallback = fallback

    async def decide(self, obs: dict) -> Decision:
        # Dispatch to @decide method if the subclass defines one
        fn = type(self)._adl_decide_fn
        if fn is not None:
            result = await fn(self, obs)
            if not isinstance(result, Decision):
                raise TypeError(
                    f"@decide method '{fn.__name__}' on {type(self).__name__} "
                    f"must return Decision, got {type(result).__name__}"
                )
            return result

        # Composition: try primary, then fallback
        if self._primary is not None:
            try:
                result = await self._primary.decide(obs)
                if result is not None:
                    return result
            except Exception:
                log.exception("Policy: primary raised; trying fallback")
        if self._fallback is not None:
            return await self._fallback.decide(obs)
        return Decision()
