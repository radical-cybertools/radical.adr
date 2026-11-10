"""Adaptation and resource boundaries for ADR operators.

Small, domain-agnostic value types that let an operator declare *limits* on the
autonomous loop:

  Tunable  — a bounded parameter an agent may adapt (the adaptation boundary).
  Budget   — a hard resource/deadline stop, evaluated independently of goal state.

Nothing here is RL- or domain-specific; these express only "this knob is legal
within these bounds" and "watch this observed metric, and stop at this ceiling."
"""

from __future__ import annotations

from dataclasses import dataclass


class Tunable:
    """A bounded parameter an agent is permitted to adapt — the adaptation boundary.

    Declare tunables as class attributes on an ``Operator``::

        class Run(Operator):
            learning_rate = Tunable(3e-4, 1e-5, 1e-3)
            grad_steps    = Tunable(25_000, 2_000, 100_000, int=True)
            discount      = 0.99          # plain attr → frozen (agent can't write it)

    One declaration is the single source of truth: ADR seeds ``state.objectives``
    with ``default``, **validates** every agent ``update_param`` against the bounds
    (see :meth:`validate`), surfaces the current values via ``operator.tunables``,
    and renders the boundary into the agent's tool schema. An out-of-bounds proposal
    is **rejected** (surfaced as evidence), never silently clamped — the agent asked
    for a specific experiment, so ADR either runs it or tells the agent why it didn't.

    Parameters
    ----------
    default: the initial value (must be within bounds).
    low, high: inclusive bounds.
    int: if True, only integer-valued numbers are accepted (256.0 is fine, 256.7 is
        rejected — no silent rounding of the requested value).
    step: if set, accepted values must lie on the grid ``low + k*step``.
    """

    __slots__ = ("default", "low", "high", "is_int", "step")

    def __init__(self, default: float, low: float, high: float,
                 *, int: bool = False, step: "float | None" = None) -> None:
        if low > high:
            raise ValueError(f"Tunable: low ({low}) must be <= high ({high})")
        if step is not None and step <= 0:
            raise ValueError(f"Tunable: step must be > 0, got {step}")
        self.low    = float(low)
        self.high   = float(high)
        self.is_int = bool(int)
        self.step   = float(step) if step is not None else None
        accepted, reason = self.validate(default)
        if reason is not None:
            raise ValueError(f"Tunable: default {default!r} is invalid ({reason})")
        self.default = accepted

    def validate(self, value: object) -> "tuple[float | int | None, str | None]":
        """Return ``(accepted_value, None)`` if ``value`` is legal, else ``(None, reason)``.

        ``reason`` is one of ``"not_a_number"``, ``"out_of_bounds"``, ``"not_integer"``,
        ``"off_step"``. An accepted value is returned as ``int`` when ``int=True``.
        """
        try:
            v = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None, "not_a_number"
        if v < self.low or v > self.high:
            return None, "out_of_bounds"
        if self.is_int and v != round(v):
            return None, "not_integer"
        if self.step is not None:
            k = round((v - self.low) / self.step)
            if abs((self.low + k * self.step) - v) > 1e-9 * max(1.0, abs(v)):
                return None, "off_step"
        return (int(round(v)) if self.is_int else v), None

    def __repr__(self) -> str:
        extra = (", int=True" if self.is_int else "") + (f", step={self.step}" if self.step else "")
        return f"Tunable({self.default}, {self.low}, {self.high}{extra})"


@dataclass(frozen=True)
class Budget:
    """A resource/deadline ceiling that terminates a run when crossed.

    Unlike a :class:`~radical.adr.goals.Goal`, a ``Budget`` is **not** part of the
    convergence condition. It is an independent stop condition evaluated every
    cycle, so no goal composition — including a runtime goal an agent adds, whether
    satisfiable or not — can block it. It guarantees the run *terminates* when the
    resource is exhausted; it does **not** change what "converged" means.

    Parameters
    ----------
    metric:
        The observed key to watch. Read from the ``@observe`` dict first, then from
        ``snapshot.objectives``. Examples: ``"trials"``, ``"total_grad_steps"``,
        ``"usd_cost"``, ``"tokens"``.
    limit:
        The ceiling value.
    direction:
        ``"max"`` (default) stops when ``value >= limit`` (a spend ceiling).
        ``"min"`` stops when ``value <= limit`` (a floor / remaining-budget countdown).
    """

    metric:    str
    limit:     float
    direction: str = "max"

    def __post_init__(self) -> None:
        if self.direction not in ("max", "min"):
            raise ValueError(
                f"Budget.direction must be 'max' or 'min', got {self.direction!r}"
            )

    def exhausted(self, value: object) -> bool:
        """True when ``value`` has crossed the ceiling. A missing value never trips."""
        if value is None:
            return False
        v = float(value)  # type: ignore[arg-type]
        return v >= self.limit if self.direction == "max" else v <= self.limit
