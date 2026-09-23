"""ADR steering DQN hyperparameters on MinAtar — the treatment arm of the SECOND AutoRL study.

Same observe→decide→act loop, same `Tunable` boundary + env-step `Budget` as the baselines, over the
same DQN primitive (`minatar_dqn.train_and_eval`). The distinction is WHAT the decision sees: the DQN
training diagnostics (TD-error, grad norm, dormant-neuron ratio, Q-overestimation) — not just a scalar
score. The rule policy acts on the mechanism (unstable update → lower lr; tiny TD-error far from target
→ raise lr; stalled → adjust target-sync / exploration or jump), which Random/Hyperband/PBT cannot do.

Only the `rule` arm here (the "prove ADR works without an LLM" case). Mirrors cartpole-autorl/main.py.
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

import minatar_dqn as cp
import baselines           # reuse the SAME config sampler as the baselines (equal starting distribution)

HERE    = Path(__file__).resolve().parent
RESULTS = HERE / "experiment_results"
TARGETS = json.loads((RESULTS / "targets.json").read_text())
STEP_CHUNK, EVAL_EP = TARGETS["step_chunk"], TARGETS["eval_episodes"]
TOTAL_BUDGET = TARGETS["total_env_step_budget"]
R_HIGH, R_MED = TARGETS["R_high"], TARGETS["R_medium"]


async def _train_step_impl(*, config, ckpt_in, step_budget, seed, eval_episodes):
    return cp.train_and_eval(config=config, ckpt_in=ckpt_in, step_budget=step_budget,
                             seed=seed, eval_episodes=eval_episodes)


class MinAtarTuningOperator(Operator):
    decide_when = "idle"

    learning_rate = Tunable(1e-3,  *cp.SEARCH_SPACE["learning_rate"])
    target_update = Tunable(1000,  *cp.SEARCH_SPACE["target_update"], int=True)
    epsilon_final = Tunable(0.1,   *cp.SEARCH_SPACE["epsilon_final"])
    batch_size    = Tunable(32,    *cp.SEARCH_SPACE["batch_size"], int=True)

    best_return: float = -1e9
    last_return: float = 0.0
    trials:      int   = 0
    env_steps:   int   = 0
    ckpt_path:   str   = ""
    seed = 0

    @goals
    def solved(self):
        return Goal("solved", metric="best_return", threshold=R_HIGH, direction="maximize", inclusive=True)

    @act
    async def train_step(self):
        return self.engine.function_task(_train_step_impl)(
            config=self.tunables, ckpt_in=(self.ckpt_path or None), step_budget=STEP_CHUNK,
            seed=self.seed, eval_episodes=EVAL_EP)

    @act
    async def restart(self):
        self.ckpt_path = ""
        return None

    @observe
    def extract(self, snapshot) -> dict:
        result = snapshot.results[0] if snapshot.results else None
        if result is not None and "return" in result:
            self.ckpt_path = result["ckpt"]; self.last_return = result["return"]
            self.env_steps = self.env_steps + result["env_steps"]; self.trials = self.trials + 1
            self.best_return = max(self.best_return, result["return"])
            self._diag = result["diagnostics"]; self._last_trial_steps = result["env_steps"]
        return {
            "has_evidence": self.trials > 0, "inflight": self.inflight, "trials": self.trials,
            "return": self.last_return, "best_return": self.best_return, "env_steps": self.env_steps,
            "trend": self.trend("best_return", "maximize"),
            "stalled": self.stalled("best_return", "maximize", 3),
            "diagnostics": getattr(self, "_diag", {}), "hyperparameters": self.tunables,
            "rejected_changes": self.rejections, "budget_frac": round(self.env_steps / TOTAL_BUDGET, 3),
        }


class RulePolicy(Policy):
    """Mechanism-aware, deterministic. Reads the DQN diagnostics to adapt the current config or jump."""
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
            self.last_decision = {"reasoning": "bootstrap"}
            return Decision(actions=[act.train_step()])
        if obs["best_return"] > self._best + 1e-6:
            self._best, self._no_improve = obs["best_return"], 0
        else:
            self._no_improve += 1

        d, hp, ret = obs["diagnostics"], obs["hyperparameters"], obs["return"]
        changes, reason = [], "improving — continue"

        if self._no_improve >= 5 and obs["best_return"] < R_HIGH:
            cfg = baselines.sample_config(self._rng)
            actions = [act.update_param(k, v) for k, v in cfg.items()] + [act.restart(), act.train_step()]
            self._no_improve = 0
            self.last_decision = {"reasoning": f"stuck at best={obs['best_return']:.1f} → jump", "jump": True,
                                  "param_changes": [{"name": k, "value": round(v, 6)} for k, v in cfg.items()]}
            return Decision(actions=actions)

        if d.get("grad_norm", 0) > 5.0 or d.get("td_error", 0) > 0.5:
            changes.append(("learning_rate", hp["learning_rate"] * 0.5))
            reason = f"unstable (td={d.get('td_error'):.3f}, grad={d.get('grad_norm'):.2f}) → halve lr"
        elif ret < R_MED and d.get("td_error", 1) < 0.03:
            changes.append(("learning_rate", hp["learning_rate"] * 1.7))
            reason = f"slow (td={d.get('td_error'):.3f}, ret={ret:.1f}≪target) → raise lr"
        elif self._no_improve >= 2 and ret < R_HIGH:
            changes.append(("target_update", max(200, hp["target_update"] * 0.6)))   # sync more often
            changes.append(("epsilon_final", max(0.02, hp["epsilon_final"] * 0.7)))  # less noise
            reason = "stalled → faster target sync + less exploration"

        actions = [act.update_param(k, v) for k, v in changes] + [act.train_step()]
        self.last_decision = {"reasoning": reason,
                              "param_changes": [{"name": k, "value": round(v, 6)} for k, v in changes]}
        return Decision(actions=actions)


class _Logger:
    def __init__(self, arm, seed):
        self.arm, self.seed, self.path = arm, seed, RESULTS / f"{arm}.jsonl"
        self.trial, self.t0, self.rows = 0, time.time(), []

    def maybe_log(self, op):
        if op.trials <= self.trial:
            return
        self.trial = op.trials; pol = op.policy
        self.rows.append({
            "arm": self.arm, "seed": self.seed,
            "config_id": f"c{abs(hash(tuple(sorted(op.tunables.items())))) % (10**8):08d}",
            "config": op.tunables, "trial": op.trials,
            "env_steps": int(getattr(op, "_last_trial_steps", 0)), "cum_env_steps": int(op.env_steps),
            "eval_steps": 0, "wall_time": round(time.time() - self.t0, 3),
            "return": round(op.last_return, 2), "best_return_so_far": round(op.best_return, 2),
            "solved": bool(op.best_return >= TARGETS["solve_threshold"]),
            "diagnostics": {k: round(v, 4) for k, v in getattr(op, "_diag", {}).items()},
            "decision": getattr(pol, "last_decision", {}) or {},
            "tokens": int(getattr(pol, "_last_tokens", 0)),
            "cost": float(getattr(pol, "_last_cost", 0.0)),
            "latency": float(getattr(pol, "_last_latency", 0.0))})   # LLM round-trip (s), 0 for rule

    def flush(self):
        with self.path.open("a") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")


def _make_policy(arm, op, model, key, dry):
    if arm == "rule":
        return RulePolicy(op)
    from llm_policy import LLMTuningPolicy, DryTuningPolicy      # DQN LLM arm
    return DryTuningPolicy(op) if dry else LLMTuningPolicy(op, key, model)


async def _run_seed(engine, arm, seed, model, key, dry):
    """Steer ONE seed over the already-running engine (created once by `_run_all`). A fresh Operator per
    seed keeps the seeds isolated; the backend + AsyncFlow engine are shared, not rebuilt per seed."""
    logger = _Logger("adr_" + arm, seed); op = None
    try:
        op = MinAtarTuningOperator(engine, f"minatar-tune-{seed}",
                                   max_cycles=1_000_000, budget=[Budget("env_steps", TOTAL_BUDGET)])
        op.seed = seed
        for k, v in baselines.sample_config(np.random.RandomState(1_000 + seed)).items():
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
            print(f"  seed {seed}: trials={trials} best={best:.1f} steps_to_R_high={hit} "
                  f"wall={time.time()-t0:.0f}s", flush=True)
    finally:
        await engine.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["rule", "llm"], default="rule")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
    ap.add_argument("--dry-agent", action="store_true", help="LLM arm plumbing with no API call")
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    print(f"ADR arm '{a.arm}'{' (dry)' if a.dry_agent else ''} · {a.seeds} seeds · "
          f"budget={TOTAL_BUDGET} steps/seed · R_high={R_HIGH}", flush=True)
    asyncio.run(_run_all(a.arm, a.seeds, a.model, key, a.dry_agent))
    print(f"raw rows -> {RESULTS / ('adr_' + a.arm + '.jsonl')}", flush=True)


if __name__ == "__main__":
    main()
