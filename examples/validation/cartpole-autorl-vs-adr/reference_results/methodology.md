# Methodology reference (Patterson et al. 2024) + testbed rationale

**Citation:** Patterson, Neumann, White, White (2024). *Empirical Design in Reinforcement Learning.*
JMLR 25(318):1-63. `literature-problems/23-0183.pdf`.

Not an experiment we reproduce — the methodology standard this study follows.

| Rule | How this study honors it |
|---|---|
| **Budget in environment steps** | Every arm gets an identical **total training env-step budget**. Trials + wall-clock are *derived*, not the budget unit. Evaluation steps are fixed measurement overhead, equal per trial, and are NOT charged to the budget. |
| **Equal tuning budget per competitor** | The `Tunable` search space is shared by every arm; all arms get the same total env-step budget. Early stopping (Hyperband, ADR) is legitimate adaptive allocation — total steps stay equal. |
| **Uncertainty-aware selection** | Report **mean ± 95% CI over 10 seeds**; never a single run. |
| **Pre-registered targets** | A reference phase freezes `R_low/med/high` on our env BEFORE any arm runs (`experiment_results/targets.json`). Metric = env-steps/trials/wall-clock to reach each R. |
| **Preserve raw per-seed results** | Every arm writes raw per-record JSONL; aggregation only in `analyze.py`. |
| **Maximization bias** | Best config selected under `mean - std`, reported with its own CI; time-to-target uses the pre-registered thresholds, not the best-observed value. |

## Why PPO + CartPole (and why NOT REINFORCE + MountainCar)

An earlier attempt used REINFORCE on `MountainCarContinuous-v0`. It was abandoned (scaffolding deleted)
because **vanilla REINFORCE cannot solve MountainCarContinuous in a short budget**: the env gives no
signal until the goal is first reached, Gaussian exploration essentially never produces the coordinated
maneuver, and reward shaping either collapses to passivity/max-throttle (−100) or gets gamed. Even
Duan et al.'s own REINFORCE only reached −67 on Mountain Car (best method −61.7, random −415) — i.e. the
reference's REINFORCE is *weak* there too. It therefore yields no clean time-to-target signal.

**PPO + CartPole-v1** is the [ARLBench](https://arxiv.org/abs/2409.18827) quickstart and the AutoRL
domain standard: PPO is the on-policy default, CartPole is cheap (seconds/run) and genuinely sensitive
to lr / clip / entropy / GAE-λ / epochs, so good configs reach ~500 quickly while bad ones (e.g. lr too
high → KL blow-up) visibly fail — exactly the hyperparameter sensitivity a tuning-method study needs.

## Provenance statement (for the final report)
- Domain-standard conventions + baselines: ARLBench (Becktepe et al. 2024/2026).
- Experimental methodology: Patterson et al. 2024.
- Wall-clock baseline: our own local Random Search / Hyperband / PBT implementations.
- Experimental treatment: ADR (rule-based, and LLM).
