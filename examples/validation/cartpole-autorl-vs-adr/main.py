"""ADR steering PPO hyperparameters on CartPole — the treatment arm(s) of the AutoRL study.

ADR expresses hyperparameter tuning as its observe -> decide -> act loop over the SAME PPO primitive
(`cartpole_ppo.train_and_eval`), the SAME `Tunable` search space, and the SAME env-step `Budget` as the
baselines (`baselines.py`). The distinction is WHAT the decision sees: the full PPO training diagnostics
(KL, entropy, clip-frac, value loss, explained variance, grad norm), not just a scalar score. The rule
policy acts on the *mechanism* — underfitting (low explained-variance + tiny KL + flat return) -> raise
lr; unstable update (high KL / clip-frac) -> lower lr; stalled -> perturb for stability — which
Random Search / Hyperband / PBT cannot do (they see only outcomes).

Two arms: `--arm rule` (deterministic, no LLM, no key) and `--arm llm` (an LLM reasons over the same
diagnostics; `--dry-agent` runs the plumbing with no key). All arms reuse the shipped ADR public API:
`Tunable`, `Budget`, `decide_when`, `trend`/`stalled`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine
import numpy as np

from radical.adr import (
    Budget, Decision, Goal, Operator, Policy, Tunable, act, decide, enable_logging, goals, observe,
)

import cartpole_ppo as cp
import baselines           # reuse the SAME config sampler as the baselines (equal starting distribution)

HERE    = Path(__file__).resolve().parent
RESULTS = HERE / "experiment_results"
TARGETS = json.loads((RESULTS / "targets.json").read_text())
STEP_CHUNK  = TARGETS["step_chunk"]
EVAL_EP     = TARGETS["eval_episodes"]
TOTAL_BUDGET = TARGETS["total_env_step_budget"]
R_HIGH, R_MED = TARGETS["R_high"], TARGETS["R_medium"]


# The @act dispatches PPO to an AsyncFlow worker. Module-level + async, per the sacred boundary.
async def _train_step_impl(*, config, ckpt_in, step_budget, seed, eval_episodes):
    return cp.train_and_eval(config=config, ckpt_in=ckpt_in, step_budget=step_budget,
                             seed=seed, eval_episodes=eval_episodes)


class CartPoleTuningOperator(Operator):
    """One adaptive config lineage: ADR trains the current config incrementally and steers its knobs."""

    decide_when = "idle"                 # a trial spans many fast cycles — decide only between trials

    # ── the shared adaptation boundary (identical ranges to the baselines' search space) ──
    learning_rate = Tunable(1e-3, *cp.SEARCH_SPACE["learning_rate"])
    clip_coef     = Tunable(0.2,  *cp.SEARCH_SPACE["clip_coef"])
    gae_lambda    = Tunable(0.95, *cp.SEARCH_SPACE["gae_lambda"])
    entropy_coef  = Tunable(0.01, *cp.SEARCH_SPACE["entropy_coef"])
    n_epochs      = Tunable(4,    *cp.SEARCH_SPACE["n_epochs"], int=True)
    num_steps     = Tunable(1024, *cp.SEARCH_SPACE["num_steps"], int=True)

    # ── evidence folded from each trial (persists → history/trend) ──
    best_return: float = -1e9
    last_return: float = 0.0
    trials:      int   = 0
    env_steps:   int   = 0                # cumulative TRAINING steps — the budget metric
    ckpt_path:   str   = ""               # model handle (a path); operator-written, never agent-written

    seed = 0                              # set per run

    @goals
    def solved(self):
        return Goal("solved", metric="best_return", threshold=R_HIGH, direction="maximize", inclusive=True)

    @act
    async def train_step(self):
        cfg = self.tunables                                   # current config = the Tunable values
        return self.engine.function_task(_train_step_impl)(
            config=cfg, ckpt_in=(self.ckpt_path or None), step_budget=STEP_CHUNK,
            seed=self.seed, eval_episodes=EVAL_EP)

    @act
    async def restart(self):
        # Operational reset: drop the current model so the next train_step starts a FRESH lineage.
        # The new config's hyperparameters are set by the policy via update_param (bounded); this only
        # resets the checkpoint. Local no-op act (returns None → nothing tracked).
        self.ckpt_path = ""
        return None

    @observe
    def extract(self, snapshot) -> dict:
        result = snapshot.results[0] if snapshot.results else None
        if result is not None and "return" in result:
            self.ckpt_path  = result["ckpt"]
            self.last_return = result["return"]
            self.env_steps  = self.env_steps + result["env_steps"]
            self.trials     = self.trials + 1
            self.best_return = max(self.best_return, result["return"])
            self._diag = result["diagnostics"]
            self._last_trial_steps = result["env_steps"]
        return {
            "has_evidence":   self.trials > 0,
            "inflight":       self.inflight,
            "trials":         self.trials,
            "return":         self.last_return,
            "best_return":    self.best_return,
            "env_steps":      self.env_steps,                 # Budget("env_steps", …) watches this
            "trend":          self.trend("best_return", "maximize"),
            "stalled":        self.stalled("best_return", "maximize", 3),
            "diagnostics":    getattr(self, "_diag", {}),
            "hyperparameters": self.tunables,
            "rejected_changes": self.rejections,
            "budget_frac":    round(self.env_steps / TOTAL_BUDGET, 3),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Rule policy — mechanism-aware, deterministic (no LLM). Conditions on the diagnostics.
# ══════════════════════════════════════════════════════════════════════════════

class RulePolicy(Policy):
    """Mechanism-aware, deterministic. Reads the PPO diagnostics to either ADAPT the current config
    (nudge a knob whose diagnostic says why it's failing) or JUMP to a fresh config when a lineage is
    persistently stuck — the exploration Random/Hyperband/PBT get by sampling many configs, but here
    directed by evidence rather than blind sampling / random perturbation."""

    def __init__(self, op):
        super().__init__()
        self._act = op.get_actions()
        self._rng = np.random.RandomState(op.seed * 31 + 7)
        self._best, self._no_improve = -1e9, 0
        self.last_decision = {}

    @decide
    async def run(self, obs) -> Decision:
        act = self._act
        if not obs["has_evidence"]:
            self.last_decision = {"reasoning": "bootstrap: train the starting config"}
            return Decision(actions=[act.train_step()])

        # track how long since the global best improved
        if obs["best_return"] > self._best + 1e-6:
            self._best, self._no_improve = obs["best_return"], 0
        else:
            self._no_improve += 1

        d, hp = obs["diagnostics"], obs["hyperparameters"]
        ret = obs["return"]
        changes, reason = [], "improving — continue current config"

        if self._no_improve >= 5 and obs["best_return"] < R_HIGH:
            # persistently stuck → JUMP: fresh config (bounded) + reset the model
            cfg = baselines.sample_config(self._rng)
            actions = [act.update_param(k, v) for k, v in cfg.items()]
            actions += [act.restart(), act.train_step()]
            self._no_improve = 0
            self.last_decision = {"reasoning": f"stuck {5}+ trials at best={obs['best_return']:.0f} "
                                  f"→ jump to a fresh config", "param_changes":
                                  [{"name": k, "value": round(v, 6)} for k, v in cfg.items()],
                                  "jump": True}
            return Decision(actions=actions)

        if d.get("kl", 0) > 0.05 or d.get("clip_frac", 0) > 0.30:
            changes.append(("learning_rate", hp["learning_rate"] * 0.5))
            reason = f"unstable update (KL={d.get('kl'):.3f}, clip={d.get('clip_frac'):.2f}) → halve lr"
        elif ret < R_MED and d.get("kl", 1) < 0.004:
            # learning too slowly: updates are tiny (KL≈0) yet still far from target → raise lr.
            # Proactive (not gated on a hard stall) — this is the diagnostic-driven acceleration the
            # baselines cannot do: they'd just spend budget on the slow config or resample blindly.
            changes.append(("learning_rate", hp["learning_rate"] * 1.7))
            reason = f"slow learning (KL={d.get('kl'):.3f}, ret={ret:.0f}≪target) → raise lr"
        elif self._no_improve >= 2 and ret < R_HIGH:
            changes.append(("num_steps", hp["num_steps"] * 1.5))       # steadier gradients
            changes.append(("entropy_coef", hp["entropy_coef"] + 0.005))
            reason = "stalled below target → larger rollout + more exploration"

        actions = [act.update_param(k, v) for k, v in changes]
        actions.append(act.train_step())
        self.last_decision = {"reasoning": reason,
                              "param_changes": [{"name": k, "value": round(v, 6)} for k, v in changes]}
        return Decision(actions=actions)


# ══════════════════════════════════════════════════════════════════════════════
# Run harness + logging (writes the same per-record schema as the baselines)
# ══════════════════════════════════════════════════════════════════════════════

class _Logger:
    """Observer-free logger: reads the operator after each trial and appends a raw record."""
    def __init__(self, arm, seed):
        self.arm, self.seed, self.path = arm, seed, RESULTS / f"{arm}.jsonl"
        self.trial, self.t0, self.rows = 0, time.time(), []

    def maybe_log(self, op):
        if op.trials <= self.trial:
            return
        self.trial = op.trials
        pol = op.policy
        row = {
            "arm": self.arm, "seed": self.seed,
            "config_id": f"c{abs(hash(tuple(sorted(op.tunables.items())))) % (10**8):08d}",
            "config": op.tunables, "trial": op.trials,
            "env_steps": int(getattr(op, "_last_trial_steps", 0)),
            "cum_env_steps": int(op.env_steps),
            "eval_steps": 0,
            "wall_time": round(time.time() - self.t0, 3),
            "return": round(op.last_return, 2), "best_return_so_far": round(op.best_return, 2),
            "solved": bool(op.best_return >= TARGETS["solve_threshold"]),
            "diagnostics": {k: round(v, 4) for k, v in getattr(op, "_diag", {}).items()},
            "decision": getattr(pol, "last_decision", {}) or {},
            "tokens": int(getattr(pol, "_last_tokens", 0)),
            "cost": float(getattr(pol, "_last_cost", 0.0)),
            "latency": float(getattr(pol, "_last_latency", 0.0)),   # LLM API round-trip (s), 0 otherwise
        }
        self.rows.append(row)

    def flush(self):
        with self.path.open("a") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")


def _make_policy(arm, op, model, key, dry):
    if arm == "rule":
        return RulePolicy(op)
    from llm_policy import LLMTuningPolicy, DryTuningPolicy      # Stage 4
    return DryTuningPolicy(op) if dry else LLMTuningPolicy(op, key, model)


async def _run_seed(engine, arm, seed, model, key, dry):
    """Steer ONE seed over the already-running engine. The backend + AsyncFlow engine are created ONCE
    by the caller (`_run_all`) and shared across every seed — per the sacred boundary, ADR launches the
    execution stack once and only steers over it. A fresh Operator per seed keeps the seeds isolated."""
    logger = _Logger("adr_" + arm, seed)
    op = None
    try:
        op = CartPoleTuningOperator(engine, f"cartpole-tune-{seed}",
                                    max_cycles=1_000_000, budget=[Budget("env_steps", TOTAL_BUDGET)])
        op.seed = seed
        # Start from a RANDOM config drawn from the same distribution the baselines sample — so ADR is
        # adapting an arbitrary start, not coasting from a hand-picked good one (fairness). These are
        # operator-internal writes (not agent update_param), so the boundary is not involved.
        cfg0 = baselines.sample_config(np.random.RandomState(1_000 + seed))
        for k, v in cfg0.items():
            setattr(op, k, v)
        op.policy = _make_policy(arm, op, model, key, dry)
        async for _ in op.run():
            logger.maybe_log(op)
        logger.flush()
    finally:
        if op is not None and hasattr(op.policy, "aclose"):
            await op.policy.aclose()          # close the per-seed LLM http client (engine stays up)
    hit = next((r["cum_env_steps"] for r in logger.rows if r["best_return_so_far"] >= R_HIGH), None)
    return op.trials, op.best_return, hit


async def _run_all(arm, seeds, model, key, dry):
    """Create the backend + AsyncFlow engine ONCE, then steer every seed over it, and shut down once."""
    enable_logging(logging.WARNING)
    backend = await LocalExecutionBackend(ProcessPoolExecutor(max_workers=2))
    engine = await WorkflowEngine.create(backend)
    try:
        for seed in range(seeds):
            t0 = time.time()
            trials, best, hit = await _run_seed(engine, arm, seed, model, key, dry)
            print(f"  seed {seed}: trials={trials} best={best:.0f} steps_to_R_high={hit} "
                  f"wall={time.time()-t0:.0f}s", flush=True)
    finally:
        await engine.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["rule", "llm"], default="rule")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
    ap.add_argument("--dry-agent", action="store_true", help="LLM arm plumbing with no API call")
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")

    print(f"ADR arm '{a.arm}'{' (dry)' if a.dry_agent else ''} · {a.seeds} seeds · budget="
          f"{TOTAL_BUDGET} train steps/seed · target R_high={R_HIGH}")
    asyncio.run(_run_all(a.arm, a.seeds, a.model, key, a.dry_agent))
    print(f"raw rows -> {RESULTS / ('adr_' + a.arm + '.jsonl')}")


if __name__ == "__main__":
    main()
