"""Policy — unified base, composition class, and @decide decorator."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

from radical.adr.decision import Decision
from radical.adr.errors import PolicyError

log = logging.getLogger(__name__)


def decide(fn: Callable) -> Callable:
    """Mark a method as the policy's decision entry point.

    Mirrors ``@act``, ``@observe``, ``@goals`` on the Operator side.
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
    """Base class for all ADR policies, and a composition class.

    **Subclass with @decide** (primary usage)::

        class SearchPolicy(Policy):
            def __init__(self, op):
                self._act = op.get_actions()

            @decide
            async def run(self, obs: dict) -> Decision:
                if obs["cycle"] >= 15:
                    return Decision(stop=True)
                ...

    **Composition** — combine primary and fallback::

        op.policy = Policy(
            primary  = MyLLMPolicy(),
            fallback = MyRulePolicy(),
            timeout  = 30.0,
        )

    When used as composition, ``decide()`` tries ``primary`` first.
    If it raises, returns ``None``, or (when ``timeout`` is set) takes longer
    than ``timeout`` seconds, ``fallback`` is tried. A hung primary call no
    longer blocks the decision loop forever.
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
        timeout:  float | None    = None,
    ) -> None:
        self._primary  = primary
        self._fallback = fallback
        self._timeout  = timeout

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
                if self._timeout is not None:
                    result = await asyncio.wait_for(
                        self._primary.decide(obs), timeout=self._timeout
                    )
                else:
                    result = await self._primary.decide(obs)
                if result is not None:
                    return result
            except asyncio.TimeoutError:
                log.warning(
                    "Policy: primary timed out after %.1fs; trying fallback",
                    self._timeout,
                )
            except Exception:
                log.exception("Policy: primary raised; trying fallback")
        if self._fallback is not None:
            return await self._fallback.decide(obs)
        raise PolicyError(
            f"Policy '{type(self).__name__}' has no @decide method and no "
            "primary/fallback composition. Either decorate a method with @decide "
            "or use Policy(primary=..., fallback=...)."
        )
