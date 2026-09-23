# ADR validation — is bounded, mechanism-aware steering better than black-box AutoRL?

This folder holds the **scientific validation** of ADR as an **AutoRL / hyperparameter-steering
method**. Each study benchmarks ADR against the standard AutoRL baselines (Random Search, Hyperband,
PBT) on a real RL algorithm, under **identical seeds, search spaces, and environment-step budgets**, and
reports the result with domain-standard statistics — honestly, including where ADR only ties.

> **The question these examples answer:** given the same knobs and the same budget, does ADR reach the
> target in **fewer environment steps** than black-box search — and *why*?

---

## What ADR is (and is not) here

ADR is **not a new RL algorithm and not a new optimizer.** It is a bounded **observe → decide → act**
steering substrate. In these studies it plays the role of an AutoRL method: it tunes a training run's
hyperparameters online. The one thing it does that the baselines cannot is **condition on the training
diagnostics** (the white-box signals *inside* a run) rather than on the final score alone.

| | What it sees each trial | What it can do |
|---|---|---|
| **Random Search** | final scalar return | sample a new config |
| **Hyperband (SHA)** | return at rungs | prune, promote survivors |
| **PBT** | performance *ranking* | copy a leader's weights + perturb |
| **ADR (rule / LLM)** | **full diagnostic panel** (PPO: KL, entropy, clip-frac, value-loss, explained-var, grad-norm; DQN: TD-error, grad-norm, dormant-ratio, Q-overestimation) | **diagnose the failure mode**, fix the running config in place, or jump — guided by evidence |

---

## The studies

| Study | Dir | Domain / algorithm | Arms |
|---|---|---|---|
| **CartPole** | [`cartpole-autorl-vs-adr/`](cartpole-autorl-vs-adr/) | PPO (on-policy) on `CartPole-v1` | Random · Hyperband · PBT · ADR-rule · ADR-LLM |
| **MinAtar** | [`minatar-autorl/`](minatar-autorl/) | DQN (off-policy) on MinAtar-Breakout | Random · Hyperband · PBT · ADR-rule · ADR-LLM |

A third, in-progress domain (PPO on sparse **MiniGrid-DoorKey**, built to stress the *LLM-vs-rule*
question) lives at [`../minigrid-doorkey/`](../minigrid-doorkey/). The consolidated write-up of results
and critique is [`../ADR_AUTORL_FINDINGS.md`](../ADR_AUTORL_FINDINGS.md).

Every study has the **same file layout**, so once you know one you know them all:

```
<study>/
├── <workload>.py     # pure, ADR-agnostic RL primitive: train_and_eval(config, ckpt, step_budget, seed)
│                     #   → {return, env_steps, ckpt, solved, diagnostics{...}}   (cartpole_ppo / minatar_dqn)
├── baselines.py      # Random / Hyperband / PBT — the AutoRL arms ADR is measured against
├── main.py           # the ADR arms: an Operator + rule policy (+ LLM dispatch), engine created ONCE
├── llm_policy.py     # LLM decision policy (AgentHPO-style) + a no-key dry stub
├── analyze.py        # figures + summary.json with 95% bootstrap CIs
└── experiment_results/
    ├── targets.json  # FROZEN before any arm runs (budget + R_low/med/high, pre-registered)
    ├── *.jsonl       # raw per-seed, per-trial records (one arm per file)
    └── fig_*.png     # fig_curves / fig_summary / fig_solvetime
```

The workload module is the **only** thing that differs between domains. `baselines.py`, `main.py`,
`analyze.py` are the *same* harness and the *same* ADR public API (`Tunable`, `Budget`, `decide_when`,
`trend`/`stalled`) driving both — which is itself the domain-agnosticism claim.

---

## How to run

```bash
PYTHON=/home/aymen/RADICAL/AGENTIC-ADAPTIVE-DECISION-LAYER/adl_ve/bin/python
cd cartpole-autorl-vs-adr           # or minatar-autorl

# (once) freeze the pre-registered targets from a reference sweep — already committed in targets.json
$PYTHON <workload>.py --reference

# black-box baselines (no API key)
$PYTHON baselines.py --arm random    --seeds 10
$PYTHON baselines.py --arm hyperband --seeds 10
$PYTHON baselines.py --arm pbt       --seeds 10

# ADR arms
$PYTHON main.py --arm rule --seeds 10                       # deterministic, no key
export OPENROUTER_API_KEY="sk-or-..."                       # LLM arm only
$PYTHON main.py --arm llm  --seeds 10 --model openai/gpt-4o-mini
mv experiment_results/adr_llm.jsonl experiment_results/adr_llm_gpt-4o-mini.jsonl   # keep per-model files

# figures + confidence intervals
$PYTHON analyze.py
```

Notes:
- **Budget is environment steps, not wall-clock** — the primary metric is hardware- and
  parallelism-immune (per Patterson et al.). `analyze.py` also reports wall-clock but subtracts LLM API
  latency so the comparison stays fair.
- The `.jsonl` loggers **append**; delete an arm's file before re-running it.
- `main.py` creates the AsyncFlow/RHAPSODY execution stack **once** and steers every seed over it (the
  sacred boundary: ADR steers, it does not own execution). `--dry-agent` exercises the LLM plumbing with
  no key.

---

## How to read the results — `fig_summary.png`

Two panels per study: **solve-rate** (% of seeds that reach the frozen `R_high`) and **steps-to-R_high**
(with 95% CIs, lower = better).

**CartPole (10 seeds, budget 76,800, R_high=475):**

| Arm | Solve rate | Steps to R_high [95% CI] |
|---|---|---|
| Random Search | 60% | 39,105 [20,996–57,691] |
| Hyperband | 100% | 61,734 [58,637–63,604] |
| PBT | 80% | 43,653 [32,103–56,136] |
| **ADR (rule)** | **100%** | **17,385 [13,051–22,795]** |
| **ADR (LLM)** | **100%** | **17,094 [12,710–22,255]** |

→ ADR reaches the target in **2.3–3.6× fewer env-steps** at 100% reliability, with CIs that do **not**
overlap Hyperband's or PBT's. The rule and the LLM tie (CartPole has one unambiguous failure mode).

**MinAtar (5 seeds, budget 120,000, R_high=3.8):** all arms tie (~25–30k, overlapping CIs) because the
target is reachable in 1–2 trials — there is **no hard tuning problem to solve**. ADR's only edge here is
**self-stopping** at the goal (~120 s vs ~510–580 s wall-clock). Reported, not hidden.

---

## Why ADR is better (when it is)

The baselines are **outcome-driven**: they see a finished run as a scalar (or a ranking) and can only
**sample, prune, or perturb** the *config space*. That means they must **spend budget to discover a
config is bad, then throw the run away**. ADR is **mechanism-driven**:

1. **Diagnose, don't just score.** Low KL + low return and high KL produce the *same bad score* but need
   *opposite* fixes. ADR distinguishes "learning too slowly → raise lr" from "unstable update → lower
   lr"; the baselines cannot tell them apart.
2. **Fix in place.** It corrects the *running* config instead of restarting a fresh one — no budget
   wasted re-discovering.
3. **Directed jumps.** When a lineage is truly stuck, it jumps to a new config *guided by the evidence*,
   not by blind sampling (Random/PBT).
4. **Bounded & auditable.** A declared `Tunable` space rejects illegal proposals (surfaced as evidence,
   never silently clamped), a hard `Budget` caps compute, and every decision is logged with its
   rationale. The operator self-stops on the goal.

**Same knobs, same budget — ADR just doesn't waste steps learning what the diagnostics already reveal.**

### Honest boundary condition

ADR's efficiency edge appears **only when the tuning problem is non-trivial** (real hyperparameter
sensitivity under a budget where bad configs cannot brute-force the target). On a trivially-tunable task
(MinAtar at a low target) every method ties and ADR's benefit narrows to autonomous early-stopping. And
that an **LLM** beats a hand-written **rule** remains **unproven** — it tied on both domains so far; it
needs a task with genuinely *ambiguous, competing* failure modes (the motivation for the MiniGrid study).

---

## Methodology & references

- Budget in **environment steps**; pre-registered frozen targets (`targets.json`); **95% bootstrap CIs**
  over seeds; raw per-seed data preserved.
- Conventions & baselines follow **ARLBench** (arXiv:2409.18827); empirical-design methodology follows
  **Patterson et al.**, *Empirical Design in RL* (JMLR 2024). LLM-as-optimizer arm follows **AgentHPO**
  (arXiv:2402.01881). AutoRL framing: survey arXiv:2201.05000.
