"""LLM decision policy for the ADR MiniGrid-DoorKey AutoRL arm (the PPO counterpart with AMBIGUOUS
failure modes — the setting built to test whether reasoning beats a fixed rule).

Same observe→decide→act loop and same bounded `Tunable` search space as the rule policy, but the
decision is made by an LLM reasoning over the PPO training diagnostics (KL, entropy, clip-frac, value
loss, explained variance, grad norm) — the AgentHPO-style "LLM as hyperparameter optimizer" direction
(arXiv:2402.01881). The LLM gets EXACTLY the same evidence, bounds, budget and action surface as the
rule; ONLY the decision mechanism differs. ADR enforces the bounds (out-of-range proposals are rejected
and fed back as evidence) and the hard budget.

`DryTuningPolicy` is a no-LLM stub with the identical action surface, for plumbing tests without a key.
"""

from __future__ import annotations

import json

from radical.adr import Decision, LLMPolicy, Policy, decide

# USD per 1M tokens (input, output) — app-owned; ADR core carries no provider prices.
_PRICES = {"openai/gpt-4o-mini": (0.15, 0.60), "openai/gpt-4o": (2.50, 10.0),
           "anthropic/claude-3.5-sonnet": (3.00, 15.0), "default": (0.15, 0.60)}


def _cost(model, pt, ct):
    pin, pout = _PRICES.get(model, _PRICES["default"])
    return pt / 1e6 * pin + ct / 1e6 * pout


def _schema():
    from pydantic import BaseModel, Field

    class Change(BaseModel):
        name:  str = Field(description="one of the tunable hyperparameter names")
        value: float

    class TuneDecision(BaseModel):
        reasoning:     str = Field(description="1-2 sentences grounded in the diagnostics")
        param_changes: list[Change] = Field(default_factory=list)
        jump:          bool = Field(default=False, description="abandon the current model and start a "
                                    "fresh config with the given param_changes")
        stop:          bool = Field(default=False)
    return TuneDecision


class LLMTuningPolicy(LLMPolicy):
    def __init__(self, op, api_key: str, model: str):
        super().__init__()
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY required for the LLM arm (or use --dry-agent).")
        import instructor
        from openai import AsyncOpenAI
        self._act = op.get_actions()
        self._raw = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self._client = instructor.from_openai(self._raw)
        self._model = model
        self._bounds = {k: [t.low, t.high] for k, t in type(op)._adl_tunables.items()}
        self.last_decision, self._last_tokens, self._last_cost = {}, 0, 0.0
        self._last_latency = 0.0

    async def aclose(self):
        # Close the httpx/AsyncOpenAI client while the event loop is still open, so its connection
        # teardown doesn't fire after asyncio.run() closes the loop (the "Event loop is closed" noise).
        try:
            await self._raw.close()
        except Exception:
            pass

    @decide
    async def run(self, obs) -> Decision:
        act = self._act
        if not obs["has_evidence"]:
            self.last_decision = {"reasoning": "bootstrap: train the starting config"}
            return Decision(actions=[act.train_step()])

        evidence = {k: obs[k] for k in ("return", "best_return", "trend", "stalled",
                                        "diagnostics", "hyperparameters", "budget_frac",
                                        "rejected_changes")}
        prompt = (
            "You are tuning PPO hyperparameters on MiniGrid-DoorKey-8x8 (a SPARSE task: find the key, open "
            "the door, reach the goal; episode return is 0 until solved, then in (0,1] scaled by speed) to "
            f"reach a greedy eval return of {obs.get('best_return', 0):.2f}→target in as FEW environment "
            f"steps as possible. You may only change these hyperparameters, within bounds: "
            f"{json.dumps(self._bounds)}. Out-of-range proposals are rejected. Decide from the TRAINING "
            "DIAGNOSTICS (this is your edge over black-box search), and note the failure modes are "
            "AMBIGUOUS — the right fix depends on which one you see:\n"
            "  • entropy has COLLAPSED (low, e.g. <0.8 vs ~1.95 uniform) while return≈0 = the policy went "
            "deterministic BEFORE finding the sparse reward → raise entropy_coef (do NOT raise lr);\n"
            "  • high KL or high clip_frac = unstable update → lower learning_rate;\n"
            "  • tiny KL (≈0) with entropy still healthy but return≈0 = learning too slowly → raise "
            "learning_rate;\n"
            "  • low explained_var = the value function is behind;\n"
            "  • a persistently stuck lineage → set jump=true to restart with a fresh config.\n"
            "Prefer continuing (no changes) when the return is climbing.\n\n"
            f"EVIDENCE:\n{json.dumps(evidence, indent=2, default=str)}"
        )
        import time
        t0 = time.time()
        try:
            d, completion = await self._client.chat.completions.create_with_completion(
                model=self._model, response_model=_schema(),
                messages=[{"role": "user", "content": prompt}], max_retries=2)
        except Exception as exc:
            self._last_latency = time.time() - t0
            self.last_decision = {"reasoning": f"[LLM error: {type(exc).__name__}] continue"}
            return Decision(actions=[act.train_step()])
        self._last_latency = time.time() - t0     # API round-trip; subtracted from compute wall-clock

        u = getattr(completion, "usage", None)
        pt = int(getattr(u, "prompt_tokens", 0) or 0)
        ct = int(getattr(u, "completion_tokens", 0) or 0)
        self._last_tokens = int(getattr(u, "total_tokens", pt + ct) or (pt + ct))
        self._last_cost = float(getattr(u, "cost", None) or _cost(self._model, pt, ct))
        self.record_usage(tokens=self._last_tokens, cost=self._last_cost, latency=self._last_latency)

        actions = [act.update_param(c.name, c.value) for c in d.param_changes]
        if d.jump:
            actions.append(act.restart())
        actions.append(act.train_step())
        self.last_decision = {"reasoning": d.reasoning, "jump": bool(d.jump),
                              "param_changes": [{"name": c.name, "value": round(c.value, 6)}
                                                for c in d.param_changes]}
        return Decision(actions=actions, stop=bool(d.stop and obs["best_return"] >= obs.get("return", 0)))


class DryTuningPolicy(Policy):
    """No-LLM stub with the same action surface (for --dry-agent plumbing tests, no key)."""
    def __init__(self, op):
        super().__init__()
        self._act = op.get_actions()
        self.last_decision, self._last_tokens, self._last_cost = {}, 0, 0.0

    @decide
    async def run(self, obs) -> Decision:
        act = self._act
        if not obs["has_evidence"]:
            self.last_decision = {"reasoning": "dry bootstrap"}
            return Decision(actions=[act.train_step()])
        acts, note = [], "dry: continue"
        if obs["return"] < 0.1 and obs["diagnostics"].get("entropy", 9) < 0.8:
            acts.append(act.update_param("entropy_coef", min(obs["hyperparameters"]["entropy_coef"] + 0.02, 0.1)))
            note = "dry: exploration collapse → raise entropy_coef"
        acts.append(act.train_step())
        self.last_decision = {"reasoning": note}
        return Decision(actions=acts)
