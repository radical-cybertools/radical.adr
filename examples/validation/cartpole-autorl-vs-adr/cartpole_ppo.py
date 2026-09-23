"""RL workload for the ADR CartPole AutoRL study — pure, self-contained, ADR-agnostic.

From-scratch PPO (clipped surrogate + GAE) with a small actor-critic MLP on Gymnasium
`CartPole-v1`. Nothing here imports or knows about ADR; every experimental arm — the AutoRL baselines
(`baselines.py`: Random Search, Hyperband, PBT) and the ADR arms (`main.py`) — calls the SAME primitive:

    train_and_eval(config, ckpt_in, step_budget, seed) -> {return, env_steps, ckpt_out, solved,
                                                            diagnostics{kl, entropy, clip_frac, ...}}

so the ONLY thing that differs between arms is *which configs get evaluated and for how long* (the
search strategy). Everything is module-level and picklable so AsyncFlow can run it on workers.

The returned `diagnostics` (KL, entropy, clip fraction, value loss, explained variance, grad norm) are
the white-box evidence that distinguishes ADR from the baselines: Random/Hyperband/BO see only a scalar
score and PBT only a ranking, whereas an ADR policy decides on *why* a run is failing (lr too high → KL
blows up; entropy collapsed → raise entropy coef), not merely *that* it scored low.

CartPole-v1: obs=(4,), 2 discrete actions, +1 reward/step, up to 500 steps (max return 500).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import gymnasium as gym

# Single-threaded torch → bit-reproducible results whether train_and_eval runs in the main process or
# on an AsyncFlow worker (multi-threaded CPU reductions have process-dependent float order). Also
# avoids CPU oversubscription when arms run in parallel. Applied on import, so workers inherit it.
torch.set_num_threads(1)

HERE   = Path(__file__).resolve().parent
CACHE  = HERE / ".cache"; CACHE.mkdir(exist_ok=True)
ENV_ID = "CartPole-v1"

# Fixed (not searched): the study tunes the sensitive PPO knobs, holds the rest constant.
GAMMA          = 0.99
HIDDEN         = 64
VF_COEF        = 0.5
MAX_GRAD_NORM  = 0.5
NUM_MINIBATCH  = 4
TARGET_KL      = 0.03           # early-stop the epoch loop if the update grows too large (stabilizer)
SOLVE_RETURN   = 475.0          # CartPole-v1 is "solved" at mean greedy return >= 475

# The shared search space (identical for every arm). Bounds are the tuning ranges.
SEARCH_SPACE = {
    "learning_rate": (1e-4, 4e-3),   # sweet spot ~1e-3; low end too slow, high end KL-unstable
    "clip_coef":     (0.1, 0.3),
    "gae_lambda":    (0.80, 0.99),
    "entropy_coef":  (0.0, 0.05),
    "n_epochs":      (1, 10),        # int
    "num_steps":     (256, 2048),    # int — rollout length per PPO update
}
_INT = {"n_epochs", "num_steps"}


def default_config() -> dict:
    return {"learning_rate": 1e-3, "clip_coef": 0.2, "gae_lambda": 0.95,
            "entropy_coef": 0.01, "n_epochs": 4, "num_steps": 1024}


def _init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs with orthogonal init (the CleanRL-standard recipe) — a shared
    trunk with default init left the value function unable to learn (explained variance ~0)."""

    def __init__(self, obs_dim: int, n_act: int, hidden: int):
        super().__init__()
        self.critic = nn.Sequential(
            _init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            _init(nn.Linear(hidden, hidden)), nn.Tanh(),
            _init(nn.Linear(hidden, 1), std=1.0))
        self.actor = nn.Sequential(
            _init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            _init(nn.Linear(hidden, hidden)), nn.Tanh(),
            _init(nn.Linear(hidden, n_act), std=0.01))

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def act(self, obs, greedy=False):
        logits, value = self.forward(obs)
        if greedy:
            return torch.argmax(logits, dim=-1), None, value
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), value


def _evaluate(env, ac: ActorCritic, episodes: int):
    rets, steps, solved_eps = [], 0, 0
    for i in range(episodes):
        o, _ = env.reset(seed=10_000_000 + i)
        done, ep_ret = False, 0.0
        while not done:
            with torch.no_grad():
                a, _, _ = ac.act(torch.as_tensor(o, dtype=torch.float32), greedy=True)
            o, r, term, trunc, _ = env.step(int(a))
            ep_ret += r; steps += 1; done = term or trunc
        rets.append(ep_ret); solved_eps += int(ep_ret >= SOLVE_RETURN)
    return float(np.mean(rets)), steps, solved_eps / episodes


def train_and_eval(*, config: dict, ckpt_in: str | None, step_budget: int, seed: int,
                   eval_episodes: int = 10) -> dict:
    """One trial: continue/kick off PPO for ~`step_budget` training env steps, then greedy-eval.

    Incremental + checkpoint-threaded: `ckpt_in=None` starts a fresh config; a path continues the same
    config (weights, optimizer, RNG restored). Only TRAINING steps count toward `step_budget` (the
    budget unit); evaluation steps are fixed overhead reported separately.
    """
    lr        = float(config["learning_rate"])
    clip      = float(config["clip_coef"])
    lam       = float(config["gae_lambda"])
    ent_coef  = float(config["entropy_coef"])
    n_epochs  = int(config["n_epochs"])
    num_steps = int(config["num_steps"])

    fresh = not (ckpt_in and Path(ckpt_in).exists())
    if fresh:
        # Seed BEFORE building the net — orthogonal init consumes the global torch RNG, so seeding
        # after construction would make the initial weights (and the whole run) irreproducible.
        torch.manual_seed(seed); np.random.seed(seed)

    env = gym.make(ENV_ID)
    obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
    ac = ActorCritic(obs_dim, n_act, HIDDEN)
    opt = torch.optim.Adam(ac.parameters(), lr=lr, eps=1e-5)

    if fresh:
        total_steps = 0
    else:
        c = torch.load(ckpt_in, weights_only=False)
        ac.load_state_dict(c["ac"]); opt.load_state_dict(c["opt"])
        torch.set_rng_state(c["torch_rng"]); np.random.set_state(c["np_rng"])
        total_steps = c["total_steps"]
    for g in opt.param_groups:
        g["lr"] = lr

    o, _ = env.reset(seed=seed * 7919 + total_steps)   # seed once; internal resets advance the RNG
    train_steps = 0
    diag = {"kl": 0.0, "entropy": 0.0, "clip_frac": 0.0, "value_loss": 0.0,
            "explained_var": 0.0, "grad_norm": 0.0}

    while train_steps < step_budget:
        # ── collect a rollout of `num_steps` transitions ──
        obs_b  = torch.zeros((num_steps, obs_dim))
        act_b  = torch.zeros(num_steps, dtype=torch.long)
        logp_b = torch.zeros(num_steps)
        val_b  = torch.zeros(num_steps)
        rew_b  = torch.zeros(num_steps)
        done_b = torch.zeros(num_steps)
        for t in range(num_steps):
            ot = torch.as_tensor(o, dtype=torch.float32)
            with torch.no_grad():
                a, logp, v = ac.act(ot, greedy=False)
            o2, r, term, trunc, _ = env.step(int(a))
            obs_b[t] = ot; act_b[t] = a; logp_b[t] = logp; val_b[t] = v
            rew_b[t] = r; done_b[t] = float(term or trunc)
            o = o2
            if term or trunc:
                o, _ = env.reset()          # no seed — let start states vary naturally
        train_steps += num_steps; total_steps += num_steps

        # ── GAE(λ) ──
        with torch.no_grad():
            last_v = ac.forward(torch.as_tensor(o, dtype=torch.float32))[1]
        adv = torch.zeros(num_steps); gae = 0.0
        for t in range(num_steps - 1, -1, -1):
            next_v = last_v if t == num_steps - 1 else val_b[t + 1]
            nonterm = 1.0 - done_b[t]
            delta = rew_b[t] + GAMMA * next_v * nonterm - val_b[t]
            gae = delta + GAMMA * lam * nonterm * gae
            adv[t] = gae
        ret_b = adv + val_b
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── PPO update: K epochs over minibatches ──
        idx = np.arange(num_steps); mb = max(1, num_steps // NUM_MINIBATCH)
        kls, ents, cfs, vls, gns = [], [], [], [], []
        for _ in range(n_epochs):
            np.random.shuffle(idx)
            epoch_kls = []
            for s in range(0, num_steps, mb):
                b = idx[s:s + mb]
                logits, v = ac.forward(obs_b[b])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act_b[b])
                logratio = new_logp - logp_b[b]
                ratio = logratio.exp()
                a_mb = adv[b]
                pg1 = -a_mb * ratio
                pg2 = -a_mb * torch.clamp(ratio, 1 - clip, 1 + clip)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((v - ret_b[b]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg_loss + VF_COEF * v_loss - ent_coef * ent
                opt.zero_grad(set_to_none=True); loss.backward()
                gn = nn.utils.clip_grad_norm_(ac.parameters(), MAX_GRAD_NORM)
                opt.step()
                with torch.no_grad():
                    k = ((ratio - 1) - logratio).mean().item()              # approx KL (k3)
                    kls.append(k); epoch_kls.append(k)
                    ents.append(ent.item())
                    cfs.append((((ratio - 1).abs() > clip).float().mean()).item())
                    vls.append(v_loss.item()); gns.append(float(gn))
            if np.mean(epoch_kls) > TARGET_KL:      # early-stop: don't over-update a good policy
                break
        with torch.no_grad():
            y = ret_b.numpy(); yhat = val_b.numpy()
            ev = 1.0 - np.var(y - yhat) / (np.var(y) + 1e-8)
        diag = {"kl": float(np.mean(kls)), "entropy": float(np.mean(ents)),
                "clip_frac": float(np.mean(cfs)), "value_loss": float(np.mean(vls)),
                "explained_var": float(ev), "grad_norm": float(np.mean(gns))}

    # ── greedy evaluation (fixed overhead — not budgeted) ──
    mean_ret, eval_steps, solve_rate = _evaluate(env, ac, eval_episodes)
    env.close()

    ckpt_out = str(CACHE / f"ckpt_{seed}_{abs(hash(tuple(sorted(config.items())))) % (10**8)}.pt")
    torch.save({"ac": ac.state_dict(), "opt": opt.state_dict(),
                "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                "total_steps": total_steps, "config": config}, ckpt_out)

    return {
        "return": mean_ret,
        "solved": mean_ret >= SOLVE_RETURN,
        "solve_rate": solve_rate,
        "env_steps": int(train_steps),
        "eval_steps": int(eval_steps),
        "total_steps": int(total_steps),
        "ckpt": ckpt_out,
        "diagnostics": diag,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Reference phase (Stage 1): confirm PPO learns, then FREEZE the targets.
# Run:  python cartpole_ppo.py --reference
# ══════════════════════════════════════════════════════════════════════════════

def _reference_phase(step_chunk: int, per_config_budget: int, eval_episodes: int, seed: int):
    import json, time

    # A hand-picked spread across the sensitive knobs — good configs should reach ~500 fast; bad ones
    # (lr too high → KL blow-up; no exploration) should visibly fail. Sanity sweep, NOT an arm.
    configs = {
        "good":      {"learning_rate": 1e-3, "clip_coef": 0.2, "gae_lambda": 0.95,
                      "entropy_coef": 0.01, "n_epochs": 4, "num_steps": 1024},
        "small_lr":  {"learning_rate": 1e-4, "clip_coef": 0.2, "gae_lambda": 0.95,
                      "entropy_coef": 0.01, "n_epochs": 4, "num_steps": 1024},
        "big_lr":    {"learning_rate": 4e-3, "clip_coef": 0.2, "gae_lambda": 0.95,
                      "entropy_coef": 0.01, "n_epochs": 8, "num_steps": 512},
        "small_roll":{"learning_rate": 1e-3, "clip_coef": 0.2, "gae_lambda": 0.95,
                      "entropy_coef": 0.01, "n_epochs": 4, "num_steps": 256},
    }
    print(f"Reference phase · env={ENV_ID} · step_chunk={step_chunk} · per_config_budget="
          f"{per_config_budget} · eval_episodes={eval_episodes} · seed={seed}\n")
    print(f"{'config':<10} {'steps':>7} {'best':>6} {'final':>6} {'solved':>7}  return / KL per chunk")
    print("-" * 100)

    finals, steps_to_solve = {}, {}
    for name, cfg in configs.items():
        ckpt, cum, traj, solved_at = None, 0, [], None
        t0 = time.time()
        while cum < per_config_budget:
            r = train_and_eval(config=cfg, ckpt_in=ckpt, step_budget=step_chunk,
                               seed=seed, eval_episodes=eval_episodes)
            ckpt = r["ckpt"]; cum += r["env_steps"]; traj.append((r["return"], r["diagnostics"]["kl"]))
            if solved_at is None and r["return"] >= SOLVE_RETURN:
                solved_at = cum
        finals[name] = max(v for v, _ in traj)
        steps_to_solve[name] = solved_at
        traj_s = " ".join(f"{v:3.0f}/{kl:.2f}" for v, kl in traj)
        print(f"{name:<10} {cum:>7} {max(v for v,_ in traj):>6.0f} {traj[-1][0]:>6.0f} "
              f"{str(traj[-1][0] >= SOLVE_RETURN):>7}  {traj_s}")
        print(f"{'':10} wall={time.time()-t0:.1f}s  steps_to_solve(>= {SOLVE_RETURN:.0f}): {solved_at}")

    best_final = max(finals.values())
    solved_steps = [s for s in steps_to_solve.values() if s is not None]
    print("\n" + "=" * 100)
    if not solved_steps:
        print(f"WARNING: no config reached return>={SOLVE_RETURN:.0f} (best={best_final:.0f}). Increase "
              "per_config_budget or revisit the PPO implementation before freezing targets.")
        return

    typical_solve = int(np.median(solved_steps))
    # 5x median-solve: fast configs solve with margin, but slow ones (e.g. lr too low) do NOT unless
    # they are adapted — so search / adaptation actually matters within the budget (not brute patience).
    total_budget = 5 * typical_solve
    targets = {
        "R_low": 100.0, "R_medium": 300.0, "R_high": 475.0,
        "solve_threshold": SOLVE_RETURN,
        "total_env_step_budget": int(total_budget),
        "step_chunk": int(step_chunk),
        "eval_episodes": int(eval_episodes),
        "frozen": True,
        "notes": (f"Frozen from reference phase (seed={seed}). best_final={best_final:.0f}; "
                  f"median training steps to solve = {typical_solve}. total budget = 8 * median-solve."),
    }
    out = HERE / "experiment_results" / "targets.json"
    out.write_text(json.dumps(targets, indent=2))
    print(f"FROZEN targets -> {out}")
    print(json.dumps(targets, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true", help="run the reference phase and freeze targets")
    ap.add_argument("--step-chunk", type=int, default=2000)
    ap.add_argument("--per-config-budget", type=int, default=30000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.reference:
        _reference_phase(a.step_chunk, a.per_config_budget, a.eval_episodes, a.seed)
    else:
        r = train_and_eval(config=default_config(), ckpt_in=None, step_budget=2000, seed=0)
        print("self-test trial:", {k: r[k] for k in ("return", "solved", "env_steps")}, r["diagnostics"])
