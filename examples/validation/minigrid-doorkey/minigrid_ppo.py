"""RL workload for the MiniGrid AutoRL study — pure, self-contained, ADR-agnostic. THIRD domain.

From-scratch PPO (clipped surrogate + GAE) with a small conv actor-critic on MiniGrid
`DoorKey-8x8` — a SPARSE, STRUCTURED task (find key → open door → reach goal) chosen precisely because it
has MULTIPLE, AMBIGUOUS failure modes: exploration/entropy collapse (never finds the reward), unstable
update (KL blow-up), and slow learning. That ambiguity is what CartPole and MinAtar lacked, and it is the
setting in which an LLM policy's *reasoning* (pick the context-dependent fix) can beat a fixed rule's
priority ordering — the open question in `ADR_AUTORL_FINDINGS.md`.

Same public contract as `cartpole-autorl-vs-adr/cartpole_ppo.py` so `baselines.py` / `main.py` /
`analyze.py` are reused unchanged (only the import target differs):

    train_and_eval(config, ckpt_in, step_budget, seed) -> {return, env_steps, ckpt, solved,
                                                            diagnostics{kl, entropy, clip_frac, ...}}

The returned `diagnostics` (KL, entropy, clip fraction, value loss, explained variance, grad norm) are
the white-box evidence: Random/Hyperband see only the scalar return, PBT only a ranking, whereas an ADR
policy decides on *why* a run is failing (entropy≈0 with return≈0 → raise entropy_coef; KL blows up →
lower lr) — not merely *that* it scored low.

MiniGrid-DoorKey-8x8-v0: partial 7x7x3 egocentric obs, 7 discrete actions, sparse reward
`1 − 0.9·(steps/max_steps)` on success (episode return ∈ (0,1]), max 640 steps.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid-* envs with gymnasium)
from minigrid.wrappers import ImgObsWrapper

# Single-threaded torch → bit-reproducible whether train_and_eval runs in the main process or on an
# AsyncFlow worker (multi-threaded CPU reductions have process-dependent float order). Applied on import.
torch.set_num_threads(1)

HERE   = Path(__file__).resolve().parent
CACHE  = HERE / ".cache"; CACHE.mkdir(exist_ok=True)
# ENV_ID is read from an env var so AsyncFlow WORKER processes (which re-import this module) use the same
# grid size the reference froze. Default 8x8; documented fallback DoorKey-6x6/5x5 if 8x8 is unlearnable
# at a CPU budget. The arms set MINIGRID_DOORKEY_ENV from targets.json before spawning workers.
ENV_ID = os.environ.get("MINIGRID_DOORKEY_ENV", "MiniGrid-DoorKey-8x8-v0")

# Fixed (not searched): the study tunes the sensitive PPO knobs, holds the rest constant.
GAMMA          = 0.99
HIDDEN         = 128
VF_COEF        = 0.5
MAX_GRAD_NORM  = 0.5
NUM_MINIBATCH  = 4
TARGET_KL      = 0.03           # early-stop the epoch loop if the update grows too large (stabilizer)
OBS_SCALE      = 10.0           # MiniGrid encodes object/color/state as small ints (0..10) → normalize

SOLVE_RETURN   = 0.5            # provisional; FROZEN by the reference phase (relative to best_final)

# The shared search space (identical for every arm). Bounds are the tuning ranges. entropy_coef is
# widened vs CartPole because exploration is the decisive DoorKey lever (the ambiguous failure mode).
SEARCH_SPACE = {
    "learning_rate": (1e-4, 3e-3),
    "clip_coef":     (0.1, 0.3),
    "gae_lambda":    (0.90, 0.99),
    "entropy_coef":  (0.0, 0.1),
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


def _make_env():
    return ImgObsWrapper(gym.make(ENV_ID))


def _prep(o) -> torch.Tensor:
    """(7,7,3) uint8 egocentric grid → (3,7,7) float in [0,1]."""
    x = np.transpose(np.asarray(o, np.float32), (2, 0, 1)) / OBS_SCALE
    return torch.as_tensor(x)


class ActorCritic(nn.Module):
    """Small conv trunk + separate actor/critic heads (orthogonal init, CleanRL-standard MiniGrid PPO
    recipe). A shared conv encoder with separate heads is standard for MiniGrid's small egocentric grid;
    the orthogonal-init scheme (trunk √2, actor 0.01, critic 1.0) matches the CartPole/MinAtar workloads."""

    def __init__(self, in_ch: int, n_act: int, hidden: int):
        super().__init__()
        self.enc = nn.Sequential(
            _init(nn.Conv2d(in_ch, 16, 2)), nn.ReLU(),
            nn.Flatten(),
            _init(nn.Linear(16 * 6 * 6, hidden)), nn.ReLU())
        self.actor  = _init(nn.Linear(hidden, n_act), std=0.01)
        self.critic = _init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, obs):
        h = self.enc(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

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
                a, _, _ = ac.act(_prep(o).unsqueeze(0), greedy=True)
            o, r, term, trunc, _ = env.step(int(a[0]))
            ep_ret += r; steps += 1; done = term or trunc
        rets.append(ep_ret); solved_eps += int(ep_ret >= SOLVE_RETURN)
    return float(np.mean(rets)), steps, solved_eps / episodes


def train_and_eval(*, config: dict, ckpt_in: str | None, step_budget: int, seed: int,
                   eval_episodes: int = 10) -> dict:
    """One trial: continue/kick off PPO for ~`step_budget` training env steps, then greedy-eval.

    Incremental + checkpoint-threaded: `ckpt_in=None` starts a fresh config; a path continues the same
    config (weights, optimizer, torch+numpy RNG restored). Only TRAINING steps count toward `step_budget`.
    """
    lr        = float(config["learning_rate"])
    clip      = float(config["clip_coef"])
    lam       = float(config["gae_lambda"])
    ent_coef  = float(config["entropy_coef"])
    n_epochs  = int(config["n_epochs"])
    num_steps = int(config["num_steps"])

    fresh = not (ckpt_in and Path(ckpt_in).exists())
    if fresh:
        # Seed BEFORE building the net — orthogonal init consumes the global torch RNG, so seeding after
        # construction would make the initial weights (and the whole run) irreproducible.
        torch.manual_seed(seed); np.random.seed(seed)

    env = _make_env()
    in_ch = env.observation_space.shape[2]        # (7,7,3) → 3 channels after transpose
    n_act = env.action_space.n
    ac = ActorCritic(in_ch, n_act, HIDDEN)
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
        obs_b  = torch.zeros((num_steps, in_ch, 7, 7))
        act_b  = torch.zeros(num_steps, dtype=torch.long)
        logp_b = torch.zeros(num_steps)
        val_b  = torch.zeros(num_steps)
        rew_b  = torch.zeros(num_steps)
        done_b = torch.zeros(num_steps)
        for t in range(num_steps):
            ot = _prep(o)
            with torch.no_grad():
                a, logp, v = ac.act(ot.unsqueeze(0), greedy=False)
            o2, r, term, trunc, _ = env.step(int(a[0]))
            obs_b[t] = ot; act_b[t] = a[0]; logp_b[t] = logp[0]; val_b[t] = v[0]
            rew_b[t] = r; done_b[t] = float(term or trunc)
            o = o2
            if term or trunc:
                o, _ = env.reset()          # no seed — let start states vary naturally
        train_steps += num_steps; total_steps += num_steps

        # ── GAE(λ) ──
        with torch.no_grad():
            last_v = ac.forward(_prep(o).unsqueeze(0))[1][0]
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
        "diagnostics": {k: round(v, 4) for k, v in diag.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Reference phase (Stage 1): confirm PPO learns, then FREEZE the targets.
# Run:  python minigrid_ppo.py --reference
# ══════════════════════════════════════════════════════════════════════════════

def _reference_phase(step_chunk: int, per_config_budget: int, eval_episodes: int, seed: int):
    import json, time
    global SOLVE_RETURN

    # A hand-picked spread across the sensitive knobs — `good` should learn; `no_entropy` should collapse
    # exploration (return≈0, entropy→0); `high_lr` should be KL-unstable; `low_lr` slow. Sanity sweep that
    # also demonstrates the AMBIGUOUS failure modes the study is about. NOT an arm.
    configs = {
        "good":       {"learning_rate": 1e-3, "clip_coef": 0.2, "gae_lambda": 0.95,
                       "entropy_coef": 0.02, "n_epochs": 4, "num_steps": 1024},
        "no_entropy": {"learning_rate": 1e-3, "clip_coef": 0.2, "gae_lambda": 0.95,
                       "entropy_coef": 0.0,  "n_epochs": 4, "num_steps": 1024},
        "low_lr":     {"learning_rate": 1e-4, "clip_coef": 0.2, "gae_lambda": 0.95,
                       "entropy_coef": 0.02, "n_epochs": 4, "num_steps": 1024},
        "high_lr":    {"learning_rate": 3e-3, "clip_coef": 0.2, "gae_lambda": 0.95,
                       "entropy_coef": 0.02, "n_epochs": 8, "num_steps": 512},
    }
    print(f"Reference phase · env={ENV_ID} · step_chunk={step_chunk} · per_config_budget="
          f"{per_config_budget} · eval_episodes={eval_episodes} · seed={seed}\n", flush=True)
    print(f"{'config':<11} {'steps':>7} {'best':>6} {'final':>6} {'solved':>7}  ret/ent/kl per chunk")
    print("-" * 110)

    finals, steps_to_solve = {}, {}
    for name, cfg in configs.items():
        ckpt, cum, traj, solved_at, t0 = None, 0, [], None, time.time()
        while cum < per_config_budget:
            r = train_and_eval(config=cfg, ckpt_in=ckpt, step_budget=step_chunk,
                               seed=seed, eval_episodes=eval_episodes)
            ckpt = r["ckpt"]; cum += r["env_steps"]
            dg = r["diagnostics"]
            traj.append((r["return"], dg["entropy"], dg["kl"]))
            if solved_at is None and r["return"] >= 0.5:
                solved_at = cum
            # LIVE per-chunk log (flushed) so progress is visible during the run, not only at the end.
            print(f"  [{name:<10}] cum {cum:>7} ret {r['return']:.2f} best {max(v for v,_,_ in traj):.2f} "
                  f"ent {dg['entropy']:.2f} kl {dg['kl']:.3f} ev {dg['explained_var']:>6.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        finals[name] = max(v for v, _, _ in traj)
        steps_to_solve[name] = solved_at
        traj_s = " ".join(f"{v:.2f}/{e:.2f}/{k:.2f}" for v, e, k in traj)
        print(f"{name:<11} {cum:>7} {max(v for v,_,_ in traj):>6.2f} {traj[-1][0]:>6.2f} "
              f"{str(traj[-1][0] >= 0.5):>7}  {traj_s}")
        print(f"{'':11} wall={time.time()-t0:.1f}s  steps_to_solve(>=0.5): {solved_at}")

    best_final = max(finals.values())
    solved_steps = [s for s in steps_to_solve.values() if s is not None]
    print("\n" + "=" * 110)
    if not solved_steps or best_final <= 0.0:
        print(f"WARNING: no config solved DoorKey-8x8 within {per_config_budget} steps "
              f"(best_final={best_final:.2f}). Increase per_config_budget, or switch ENV_ID to "
              "DoorKey-6x6 (documented fallback), before freezing targets.")
        return

    # Relative targets (env scale unknown a priori), mirroring the MinAtar study: R_high = 0.7·best_final.
    SOLVE_RETURN = round(0.7 * best_final, 3)
    typical_solve = int(np.median(solved_steps))
    total_budget = 5 * typical_solve      # fast configs solve with margin; slow ones need adaptation
    targets = {
        "env_id": ENV_ID,          # arms (baselines.py / main.py) read this so they match the frozen env
        "R_low": round(0.25 * best_final, 3), "R_medium": round(0.5 * best_final, 3),
        "R_high": SOLVE_RETURN, "solve_threshold": SOLVE_RETURN,
        "total_env_step_budget": int(total_budget),
        "step_chunk": int(step_chunk), "eval_episodes": int(eval_episodes), "frozen": True,
        "notes": (f"MiniGrid/{ENV_ID} PPO. Frozen from reference (seed={seed}). best_final="
                  f"{best_final:.3f}; median steps-to-solve(>=0.5)={typical_solve}. "
                  f"R_high=0.7*best_final; budget=5*median-solve."),
    }
    (HERE / "experiment_results").mkdir(exist_ok=True)
    out = HERE / "experiment_results" / "targets.json"
    out.write_text(json.dumps(targets, indent=2))
    print(f"FROZEN targets -> {out}")
    print(json.dumps(targets, indent=2))
    print("per-config best:", {k: round(v, 3) for k, v in finals.items()})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true", help="run the reference phase and freeze targets")
    ap.add_argument("--env", default=None, help="override ENV_ID, e.g. MiniGrid-DoorKey-6x6-v0 (8x8 is "
                    "too hard for vanilla PPO at a CPU budget; 6x6/5x5 are the learnable fallbacks)")
    ap.add_argument("--step-chunk", type=int, default=4000)
    ap.add_argument("--per-config-budget", type=int, default=100000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.env:
        ENV_ID = a.env                          # module global read by _make_env() (single-process here)
        os.environ["MINIGRID_DOORKEY_ENV"] = a.env   # and for any child process, kept consistent
    if a.reference:
        _reference_phase(a.step_chunk, a.per_config_budget, a.eval_episodes, a.seed)
    else:
        import time; t0 = time.time()
        r = train_and_eval(config=default_config(), ckpt_in=None, step_budget=4000, seed=0, eval_episodes=5)
        print(f"self-test trial (env={ENV_ID}):", {k: r[k] for k in ("return", "solved", "env_steps")},
              r["diagnostics"], f"({time.time()-t0:.0f}s)")
