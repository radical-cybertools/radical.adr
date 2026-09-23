"""Conventional AutoRL baselines for the MiniGrid PPO study — the arms ADR is benchmarked against.

Three standard hyperparameter-optimization methods (the ARLBench baseline set), all driving the SAME
PPO primitive (`minigrid_ppo.train_and_eval`) over the SAME `Tunable` search space and the SAME total
environment-step budget, with K seeds and raw per-seed logging:

  random     — Random Search: sample configs, train each to a fixed per-config budget, keep the best.
  hyperband  — Successive Halving (Hyperband-lite): many configs briefly, promote survivors with more
               budget (multi-fidelity, early stopping).
  pbt        — Population-Based Training: a population trains; laggards copy a leader's weights+config
               (exploit) and perturb hyperparameters (explore).

These are OUTCOME-driven: they see a scalar return (random/hyperband) or a ranking (pbt). They do NOT
read the training diagnostics — that is the ADR arms' distinction (see main.py). Everything here is
deliberately dependency-free (numpy only) so the baselines are self-contained and reproducible.

Budget accounting is identical to the ADR arms: only TRAINING env-steps count; a run stops when the
cumulative training steps reach the frozen `total_env_step_budget`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import minigrid_ppo as cp

HERE     = Path(__file__).resolve().parent
RESULTS  = HERE / "experiment_results"
TARGETS  = json.loads((RESULTS / "targets.json").read_text())
if TARGETS.get("env_id"):
    cp.ENV_ID = TARGETS["env_id"]        # match the env the reference phase froze (e.g. DoorKey-6x6)


# ── config sampling / perturbation (uses an ISOLATED RandomState — train_and_eval reseeds the global
#    numpy/torch RNGs internally, so the sampler must not depend on the globals) ──────────────────────

def sample_config(rng: np.random.RandomState) -> dict:
    cfg = {}
    for name, (lo, hi) in cp.SEARCH_SPACE.items():
        if name == "learning_rate":
            v = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))     # log-uniform for lr
        else:
            v = float(rng.uniform(lo, hi))
        cfg[name] = int(round(v)) if name in cp._INT else v
    return cfg


def perturb_config(cfg: dict, rng: np.random.RandomState) -> dict:
    """PBT 'explore': multiply each hyperparameter by 0.8 or 1.2, clamped to its bounds."""
    out = {}
    for name, (lo, hi) in cp.SEARCH_SPACE.items():
        v = cfg[name] * rng.choice([0.8, 1.2])
        v = min(max(v, lo), hi)
        out[name] = int(round(v)) if name in cp._INT else float(v)
    return out


def config_id(cfg: dict) -> str:
    return f"c{abs(hash(tuple(sorted(cfg.items())))) % (10 ** 8):08d}"


# ── raw per-seed recorder ────────────────────────────────────────────────────────────────────────

class Recorder:
    def __init__(self, arm: str, seed: int):
        self.arm, self.seed = arm, seed
        self.path = RESULTS / f"{arm}.jsonl"
        self.cum_steps, self.best, self.trial, self.t0 = 0, -1e9, 0, time.time()
        self.rows: list[dict] = []

    def log(self, cfg: dict, r: dict, decision: dict | None = None) -> dict:
        self.cum_steps += r["env_steps"]
        self.best = max(self.best, r["return"])
        self.trial += 1
        row = {
            "arm": self.arm, "seed": self.seed, "config_id": config_id(cfg), "config": cfg,
            "trial": self.trial, "env_steps": r["env_steps"], "cum_env_steps": self.cum_steps,
            "eval_steps": r["eval_steps"], "wall_time": round(time.time() - self.t0, 3),
            "return": round(r["return"], 4), "best_return_so_far": round(self.best, 4),
            "solved": bool(r["solved"]), "diagnostics": {k: round(v, 4) for k, v in r["diagnostics"].items()},
            "decision": decision or {}, "tokens": 0, "cost": 0.0,
        }
        self.rows.append(row)
        print(f"  [{self.arm} s{self.seed}] trial {self.trial:>3} cum {self.cum_steps:>7} "
              f"ret {r['return']:.2f} best {self.best:.2f} "
              f"({time.time() - self.t0:.0f}s)", flush=True)   # LIVE progress
        return row

    def budget_left(self) -> int:
        return TARGETS["total_env_step_budget"] - self.cum_steps

    def flush(self):
        with self.path.open("a") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")


def _cfg_seed(base_seed: int, k: int) -> int:
    return base_seed * 100_000 + k        # distinct-but-reproducible init per config lineage


# ── the three baseline arms ───────────────────────────────────────────────────────────────────────

def run_random(seed: int, rec: Recorder):
    """Sample a config, train it to a fixed per-config budget in chunks, then move on. Repeat."""
    rng = np.random.RandomState(seed)
    step_chunk = TARGETS["step_chunk"]
    per_config = 6 * step_chunk           # each config gets a fixed slice before we move on
    k = 0
    while rec.budget_left() > 0:
        cfg = sample_config(rng); cseed = _cfg_seed(seed, k); k += 1
        ckpt, used = None, 0
        while used < per_config and rec.budget_left() > 0:
            r = cp.train_and_eval(config=cfg, ckpt_in=ckpt, step_budget=min(step_chunk, rec.budget_left()),
                                  seed=cseed, eval_episodes=TARGETS["eval_episodes"])
            ckpt = r["ckpt"]; used += r["env_steps"]; rec.log(cfg, r)


def run_hyperband(seed: int, rec: Recorder, eta: int = 3):
    """Successive Halving: n0 configs at rung 0; keep the top 1/eta, give them eta× more budget; repeat."""
    rng = np.random.RandomState(seed)
    step_chunk = TARGETS["step_chunk"]
    budget = TARGETS["total_env_step_budget"]
    n_rungs = 3
    r0 = step_chunk
    n0 = int(max(eta ** (n_rungs - 1), budget // (r0 * n_rungs)))   # size the bracket to the budget

    # rung 0: run n0 fresh configs for r0 steps
    pop = []
    for k in range(n0):
        if rec.budget_left() <= 0:
            break
        cfg = sample_config(rng); cseed = _cfg_seed(seed, k)
        r = cp.train_and_eval(config=cfg, ckpt_in=None, step_budget=min(r0, rec.budget_left()), seed=cseed,
                              eval_episodes=TARGETS["eval_episodes"])
        rec.log(cfg, r)
        pop.append({"cfg": cfg, "cseed": cseed, "ckpt": r["ckpt"], "ret": r["return"]})

    # promote survivors through the rungs; a rung-i config reaches cumulative budget r0*eta^i
    cum_prev = r0
    for i in range(1, n_rungs):
        pop.sort(key=lambda m: m["ret"], reverse=True)
        pop = pop[:max(1, len(pop) // eta)]     # keep the top 1/eta
        cum_now = r0 * (eta ** i)
        add = cum_now - cum_prev                # additional steps for survivors this rung
        for m in pop:
            if rec.budget_left() <= 0:
                break
            r = cp.train_and_eval(config=m["cfg"], ckpt_in=m["ckpt"], step_budget=min(add, rec.budget_left()),
                                  seed=m["cseed"], eval_episodes=TARGETS["eval_episodes"])
            m["ckpt"], m["ret"] = r["ckpt"], r["return"]
            rec.log(m["cfg"], r)
        cum_prev = cum_now

    # spend any leftover budget continuing the single best (fair use of the equal budget)
    if pop:
        best = max(pop, key=lambda m: m["ret"])
        while rec.budget_left() > 0:
            r = cp.train_and_eval(config=best["cfg"], ckpt_in=best["ckpt"], step_budget=min(step_chunk, rec.budget_left()),
                                  seed=best["cseed"], eval_episodes=TARGETS["eval_episodes"])
            best["ckpt"] = r["ckpt"]; rec.log(best["cfg"], r)


def run_pbt(seed: int, rec: Recorder, pop_size: int = 4):
    """Population-Based Training: each member trains one chunk per generation; bottom members copy a top
    member's weights+config (exploit) and perturb the config (explore)."""
    rng = np.random.RandomState(seed)
    step_chunk = TARGETS["step_chunk"]
    members = []
    for k in range(pop_size):
        cfg = sample_config(rng)
        members.append({"cfg": cfg, "cseed": _cfg_seed(seed, k), "ckpt": None, "ret": -1e9})

    while rec.budget_left() > 0:
        # one training step for each member (a generation)
        for m in members:
            if rec.budget_left() <= 0:
                break
            r = cp.train_and_eval(config=m["cfg"], ckpt_in=m["ckpt"], step_budget=min(step_chunk, rec.budget_left()),
                                  seed=m["cseed"], eval_episodes=TARGETS["eval_episodes"])
            m["ckpt"], m["ret"] = r["ckpt"], r["return"]
            rec.log(m["cfg"], r, decision={"pbt": "train"})
        # exploit + explore: bottom quartile copies a random top-quartile member, then perturbs
        order = sorted(range(len(members)), key=lambda i: members[i]["ret"])
        q = max(1, pop_size // 4)
        top, bottom = order[-q:], order[:q]
        for bi in bottom:
            ti = top[rng.randint(len(top))]
            members[bi]["ckpt"] = members[ti]["ckpt"]                 # copy weights (exploit)
            members[bi]["cfg"] = perturb_config(members[ti]["cfg"], rng)  # perturb config (explore)


ARMS = {"random": run_random, "hyperband": run_hyperband, "pbt": run_pbt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()

    print(f"Baseline arm '{a.arm}' · {a.seeds} seeds · budget="
          f"{TARGETS['total_env_step_budget']} train steps/seed · targets R="
          f"{TARGETS['R_low']}/{TARGETS['R_medium']}/{TARGETS['R_high']}")
    for seed in range(a.seeds):
        rec = Recorder(a.arm, seed)
        t0 = time.time()
        ARMS[a.arm](seed, rec)
        rec.flush()
        # steps to reach R_high (first record with best >= R_high), for a quick console read
        hit = next((row["cum_env_steps"] for row in rec.rows
                    if row["best_return_so_far"] >= TARGETS["R_high"]), None)
        print(f"  seed {seed}: trials={rec.trial} best={rec.best:.3f} "
              f"steps_to_R_high={hit} wall={time.time()-t0:.0f}s")
    print(f"raw rows -> {RESULTS / (a.arm + '.jsonl')}")


if __name__ == "__main__":
    main()
