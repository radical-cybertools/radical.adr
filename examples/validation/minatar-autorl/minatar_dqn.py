"""RL workload for the MinAtar AutoRL study — pure, ADR-agnostic. Same interface contract as
`cartpole-autorl/cartpole_ppo.py`, so `baselines.py` / `main.py` / `analyze.py` are reused unchanged
(only the import target differs). This is the SECOND domain (pixel-ish, off-policy DQN) used to test
whether the ADR-substrate result generalizes beyond the CartPole/PPO case.

DQN on MinAtar (Breakout) at replay ratio 1 (kept cheap for CPU). The tunable knobs are the classic DQN
sensitivities — learning rate, target-update frequency, epsilon-final, batch size. The white-box
diagnostic panel (td_error, grad_norm, dormant_ratio, q_overestimation) is what the ADR policy conditions
on and the black-box baselines cannot see.

Incremental + checkpoint-threaded (net, target, optimizer, replay buffer, RNG) with the determinism fixes
validated on CartPole (single-thread torch, seed-before-construction, RNG state in the checkpoint).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from minatar import Environment

torch.set_num_threads(1)

HERE   = Path(__file__).resolve().parent
GAME   = "breakout"
GAMMA  = 0.99
BUF_CAP = 50_000
DORMANT_TAU = 0.025

SEARCH_SPACE = {
    "learning_rate": (1e-4, 5e-3),
    "target_update": (200, 4000),    # int — steps between target-net syncs
    "epsilon_final": (0.02, 0.30),
    "batch_size":    (16, 128),      # int
}
_INT = {"target_update", "batch_size"}


def default_config() -> dict:
    return {"learning_rate": 1e-3, "target_update": 1000, "epsilon_final": 0.1, "batch_size": 32}


def _init(layer, std=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, std); nn.init.constant_(layer.bias, 0.0); return layer


class QNet(nn.Module):
    def __init__(self, in_ch, n_act):
        super().__init__()
        self.conv = _init(nn.Conv2d(in_ch, 16, 3, 1))
        self.fc = _init(nn.Linear(16 * 8 * 8, 128))
        self.head = _init(nn.Linear(128, n_act), std=1.0)

    def features(self, x):
        h = F.relu(self.conv(x)); h = h.reshape(h.size(0), -1); return F.relu(self.fc(h))

    def forward(self, x):
        return self.head(self.features(x))


def _state(env):
    return np.transpose(np.asarray(env.state(), np.float32), (2, 0, 1))


class Replay:
    def __init__(self, cap, shape):
        self.cap, self.i, self.full = cap, 0, False
        self.s = np.zeros((cap, *shape), np.uint8); self.s2 = np.zeros((cap, *shape), np.uint8)
        self.a = np.zeros(cap, np.int64); self.r = np.zeros(cap, np.float32); self.d = np.zeros(cap, np.float32)

    def add(self, s, a, r, s2, d):
        i = self.i; self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.i = (i + 1) % self.cap; self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, n):
        idx = np.random.randint(0, len(self), n)
        return (torch.as_tensor(self.s[idx]).float(), torch.as_tensor(self.a[idx]),
                torch.as_tensor(self.r[idx]), torch.as_tensor(self.s2[idx]).float(),
                torch.as_tensor(self.d[idx]))


def _dormant(net, states):
    with torch.no_grad():
        h = net.features(states); sc = h.abs().mean(0); sc = sc / (sc.mean() + 1e-9)
        return float((sc < DORMANT_TAU).float().mean())


def _evaluate(net, n_ep, seed, max_steps=1000):
    env = Environment(GAME); env.seed(9_000_000 + seed)
    rets, q0, g0 = [], [], []
    for i in range(n_ep):
        env.reset(); s = _state(env); done, ep, steps, disc = False, 0.0, 0, []
        with torch.no_grad():
            q0.append(float(net(torch.as_tensor(s).unsqueeze(0)).max()))
        while not done and steps < max_steps:
            with torch.no_grad():
                a = int(net(torch.as_tensor(_state(env)).unsqueeze(0)).argmax())
            r, done = env.act(a); disc.append(r); ep += r; steps += 1
        rets.append(ep); g0.append(sum((GAMMA ** t) * rr for t, rr in enumerate(disc)))
    return float(np.mean(rets)), float(np.mean(q0) - np.mean(g0))


def train_and_eval(*, config, ckpt_in, step_budget, seed, eval_episodes=10):
    lr = float(config["learning_rate"]); tup = int(config["target_update"])
    eps_f = float(config["epsilon_final"]); bs = int(config["batch_size"])

    fresh = not (ckpt_in and Path(ckpt_in).exists())
    if fresh:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = Environment(GAME); env.seed(seed)
    env.reset(); in_ch, n_act = env.state_shape()[2], env.num_actions()
    net = QNet(in_ch, n_act); tgt = QNet(in_ch, n_act)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    buf = Replay(BUF_CAP, (in_ch, 10, 10))

    if fresh:
        tgt.load_state_dict(net.state_dict()); total = 0
    else:
        c = torch.load(ckpt_in, weights_only=False)
        net.load_state_dict(c["net"]); tgt.load_state_dict(c["tgt"]); opt.load_state_dict(c["opt"])
        torch.set_rng_state(c["trng"]); np.random.set_state(c["nrng"]); total = c["total"]
        n = c["buf_n"]; buf.i, buf.full = c["buf_i"], c["buf_full"]
        buf.s[:n], buf.s2[:n] = c["bs"], c["bs2"]; buf.a[:n], buf.r[:n], buf.d[:n] = c["ba"], c["br"], c["bd"]
    for g in opt.param_groups:
        g["lr"] = lr

    s = _state(env); train_steps = 0
    diag = {"td_error": 0.0, "grad_norm": 0.0, "dormant_ratio": 0.0, "q_overest": 0.0}
    while train_steps < step_budget:
        eps = max(eps_f, 1.0 - total / 20000.0)
        a = np.random.randint(n_act) if np.random.rand() < eps else \
            int(net(torch.as_tensor(s).unsqueeze(0)).argmax())
        r, done = env.act(a); s2 = _state(env); buf.add(s, a, r, s2, float(done)); s = s2
        if done:
            env.reset(); s = _state(env)
        train_steps += 1; total += 1
        if len(buf) >= bs:
            bs_, ba, br, bs2, bd = buf.sample(bs)
            with torch.no_grad():
                y = br + GAMMA * (1 - bd) * tgt(bs2).max(1).values
            q = net(bs_).gather(1, ba.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
            diag["td_error"], diag["grad_norm"] = float((q - y).abs().mean()), float(gn)
        if total % tup == 0:
            tgt.load_state_dict(net.state_dict())

    if len(buf) >= 512:
        probe = buf.sample(512)[0]; diag["dormant_ratio"] = _dormant(net, probe)
    ret, overest = _evaluate(net, eval_episodes, seed); diag["q_overest"] = overest

    n = len(buf)
    ckpt_out = str((HERE / ".cache")); Path(ckpt_out).mkdir(exist_ok=True)
    ckpt_out = f"{ckpt_out}/ck_{seed}_{abs(hash(tuple(sorted(config.items())))) % 10**8}.pt"
    torch.save({"net": net.state_dict(), "tgt": tgt.state_dict(), "opt": opt.state_dict(),
                "trng": torch.get_rng_state(), "nrng": np.random.get_state(), "total": total,
                "buf_n": n, "buf_i": buf.i, "buf_full": buf.full, "bs": buf.s[:n].copy(),
                "bs2": buf.s2[:n].copy(), "ba": buf.a[:n].copy(), "br": buf.r[:n].copy(),
                "bd": buf.d[:n].copy()}, ckpt_out)

    return {"return": float(ret), "solved": float(ret) >= SOLVE_RETURN, "env_steps": int(train_steps),
            "eval_steps": 0, "total_steps": int(total), "ckpt": ckpt_out,
            "diagnostics": {k: round(v, 4) for k, v in diag.items()}}


SOLVE_RETURN = 8.0        # provisional; frozen by the reference phase


def _reference(step_chunk, per_config_budget, eval_episodes, seed):
    import json, time
    global SOLVE_RETURN
    configs = {
        "good":   {"learning_rate": 1e-3, "target_update": 1000, "epsilon_final": 0.1, "batch_size": 32},
        "low_lr": {"learning_rate": 2e-4, "target_update": 1000, "epsilon_final": 0.1, "batch_size": 32},
        "high_lr":{"learning_rate": 4e-3, "target_update": 500,  "epsilon_final": 0.1, "batch_size": 32},
        "slow_tgt":{"learning_rate": 1e-3, "target_update": 4000, "epsilon_final": 0.05, "batch_size": 32},
    }
    print(f"MinAtar/{GAME} DQN reference · step_chunk={step_chunk} · per_config={per_config_budget} · seed={seed}\n",
          flush=True)
    finals = {}
    for name, cfg in configs.items():
        ckpt, cum, best, t0 = None, 0, -1e9, time.time()
        while cum < per_config_budget:
            r = train_and_eval(config=cfg, ckpt_in=ckpt, step_budget=step_chunk, seed=seed,
                               eval_episodes=eval_episodes)
            ckpt = r["ckpt"]; cum += r["env_steps"]; best = max(best, r["return"])
            print(f"  [{name}] cum {cum:>6} ret {r['return']:>5.1f} best {best:>5.1f} "
                  f"td {r['diagnostics']['td_error']:.3f} dormant {r['diagnostics']['dormant_ratio']:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        finals[name] = best
    best_all = max(finals.values())
    SOLVE_RETURN = round(0.7 * best_all, 1)
    typical = per_config_budget
    targets = {"R_low": round(0.25 * best_all, 1), "R_medium": round(0.5 * best_all, 1),
               "R_high": SOLVE_RETURN, "solve_threshold": SOLVE_RETURN,
               "total_env_step_budget": int(4 * typical), "step_chunk": int(step_chunk),
               "eval_episodes": int(eval_episodes), "frozen": True,
               "notes": f"MinAtar/breakout DQN RR=1. best_all={best_all}. R_high=0.7*best."}
    (HERE / "experiment_results").mkdir(exist_ok=True)
    (HERE / "experiment_results" / "targets.json").write_text(json.dumps(targets, indent=2))
    print("\nFROZEN targets:", json.dumps(targets), flush=True)
    print("per-config best:", finals, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--step-chunk", type=int, default=15000)
    ap.add_argument("--per-config-budget", type=int, default=60000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.reference:
        _reference(a.step_chunk, a.per_config_budget, a.eval_episodes, a.seed)
    else:
        import time; t0 = time.time()
        r = train_and_eval(config=default_config(), ckpt_in=None, step_budget=8000, seed=0, eval_episodes=3)
        print("self-test:", {k: r[k] for k in ("return", "env_steps")}, r["diagnostics"], f"({time.time()-t0:.0f}s)")
