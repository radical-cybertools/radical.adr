"""Analysis for the MiniGrid-DoorKey AutoRL study — Patterson-compliant stats + figures.

Loads the raw per-seed records each arm produced, computes time-to-threshold and final-performance
metrics with 95% bootstrap confidence intervals over seeds, prints the decomposition table
(Random → Hyperband → PBT → ADR-rule → ADR-LLM), and writes three figures:

  fig_curves.png    — best return vs environment steps (mean over seeds + CI band); the efficiency view.
  fig_summary.png   — solve-rate and steps-to-R_high per arm; "does it find good configs, fast".
  fig_solvetime.png — wall-clock (left axis) vs env-steps (right axis) to R_high; latency broken out.

Reads only experiment_results/*.jsonl (measured) + targets.json (frozen). MiniGrid returns are in (0,1],
so the return axis is scaled accordingly (vs CartPole's 0–500).
"""

from __future__ import annotations

import json
from pathlib import Path

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean over fully-stopped columns

HERE    = Path(__file__).resolve().parent
RESULTS = HERE / "experiment_results"
T       = json.loads((RESULTS / "targets.json").read_text())
BUDGET  = T["total_env_step_budget"]
R_LOW, R_MED, R_HIGH = T["R_low"], T["R_medium"], T["R_high"]
RET_TOP = 1.05                                          # MiniGrid episode return ceiling (≈1)

ARM_ORDER = ["random", "hyperband", "pbt", "adr_rule"]   # fixed arms; LLM arms are auto-discovered
LABEL = {"random": "Random Search", "hyperband": "Hyperband (SHA)", "pbt": "PBT",
         "adr_rule": "ADR (rule)"}
COLOR = {"random": "#94a3b8", "hyperband": "#3b82f6", "pbt": "#a855f7", "adr_rule": "#f97316"}
_LLM_COLORS = ["#ef4444", "#db2777", "#7c3aed", "#0ea5e9", "#16a34a", "#d97706"]


def discover_llm_arms():
    """Any experiment_results/adr_llm*.jsonl → its own arm (compare multiple LLMs). Registers a label
    and color for each (label derived from the filename suffix, e.g. adr_llm_gpt4o_mini → 'gpt4o mini')."""
    stems = [p.stem for p in sorted(RESULTS.glob("adr_llm*.jsonl"))]
    for i, stem in enumerate(stems):
        suffix = stem[len("adr_llm"):].strip("_")
        LABEL[stem] = f"ADR-LLM ({suffix.replace('_', ' ')})" if suffix else "ADR (LLM)"
        COLOR[stem] = _LLM_COLORS[i % len(_LLM_COLORS)]
    return stems


def load(arm):
    p = RESULTS / f"{arm}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open()]
    seeds = {}
    for r in rows:
        seeds.setdefault(r["seed"], []).append(r)
    for s in seeds:
        seeds[s].sort(key=lambda r: r["trial"])
    return seeds


def steps_to(records, R):
    for r in records:
        if r["best_return_so_far"] >= R:
            return r["cum_env_steps"]
    return None                       # censored — never reached within budget


def wall_to(records, R, subtract_latency=False):
    lat = 0.0
    for r in records:
        lat += r.get("latency", 0.0)
        if r["best_return_so_far"] >= R:
            # cumulative wall-clock at that trial; optionally remove the LLM API latency spent so far,
            # leaving RL compute time only (baselines/rule have latency 0, so unaffected).
            return r["wall_time"] - (lat if subtract_latency else 0.0)
    return None


def boot_ci(vals, n=2000):
    vals = np.asarray([v for v in vals if v is not None], dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    means = [np.mean(vals[np.random.randint(0, len(vals), len(vals))]) for _ in range(n)]
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def curve(records, grid):
    # best-return-so-far as a function of steps; NaN AFTER this seed's last step, because an arm that
    # self-stops early (ADR) is not running past its stop — forward-filling to the full budget would
    # falsely imply it kept spending steps.
    cums = np.array([r["cum_env_steps"] for r in records], dtype=float)
    best = np.array([r["best_return_so_far"] for r in records], dtype=float)
    idx = np.clip(np.searchsorted(cums, grid, side="right") - 1, 0, len(records) - 1)
    y = best[idx].astype(float)
    y[grid < cums[0]] = best[0]
    y[grid > cums[-1]] = np.nan          # stopped — no data beyond the last trial
    return y


def main():
    np.random.seed(0)
    arms = [a for a in ARM_ORDER if (RESULTS / f"{a}.jsonl").exists()] + discover_llm_arms()
    if not arms:
        print("No arm results found in experiment_results/ (need at least one *.jsonl).\n"
              "Run the arms first, e.g.:\n"
              "  python baselines.py --arm random --seeds 10\n"
              "  python baselines.py --arm hyperband --seeds 10\n"
              "  python baselines.py --arm pbt --seeds 10\n"
              "  python main.py --arm rule --seeds 10\n"
              "  python main.py --arm llm  --seeds 10   # needs OPENROUTER_API_KEY\n"
              "then re-run: python analyze.py")
        return
    grid = np.linspace(0, BUDGET, 80)
    summary = {}

    print(f"\nMiniGrid-DoorKey PPO AutoRL — {len(arms)} arms · budget={BUDGET} train steps/seed · "
          f"targets R={R_LOW}/{R_MED}/{R_HIGH}\n")
    hdr = f"{'arm':<18} {'seeds':>5} {'solve%':>7} {'steps→R_high (mean[95CI])':>28} {'best return':>18}"
    print(hdr); print("-" * len(hdr))

    curves, sth_by_arm = {}, {}
    for arm in arms:
        seeds = load(arm)
        per = list(seeds.values())
        sth = [steps_to(rs, R_HIGH) for rs in per]
        sth_by_arm[arm] = sth
        solved = [s is not None for s in sth]
        solve_rate = float(np.mean(solved))
        best_finals = [rs[-1]["best_return_so_far"] for rs in per]
        sm, slo, shi = boot_ci(sth)           # among solved seeds
        bm, blo, bhi = boot_ci(best_finals)
        curves[arm] = np.vstack([curve(rs, grid) for rs in per])
        summary[arm] = {"seeds": len(per), "solve_rate": solve_rate,
                        "steps_to_R_high": {"mean": sm, "ci": [slo, shi], "n_solved": int(sum(solved))},
                        "best_return": {"mean": bm, "ci": [blo, bhi]},
                        "steps_to_R_low": boot_ci([steps_to(rs, R_LOW) for rs in per])[0],
                        "steps_to_R_med": boot_ci([steps_to(rs, R_MED) for rs in per])[0]}
        sth_s = f"{sm:.0f} [{slo:.0f},{shi:.0f}]" if not np.isnan(sm) else "— (none solved)"
        print(f"{LABEL[arm]:<18} {len(per):>5} {solve_rate*100:>6.0f}% {sth_s:>28} "
              f"{bm:>6.3f} [{blo:.3f},{bhi:.3f}]")

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── fig 1: best return vs env steps (mean + CI band); each line ENDS where the arm stops ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in arms:
        C = curves[arm]
        mean = np.nanmean(C, 0)
        boot = np.array([np.nanmean(C[np.random.randint(0, len(C), len(C))], 0) for _ in range(400)])
        lo, hi = np.nanpercentile(boot, 2.5, 0), np.nanpercentile(boot, 97.5, 0)
        ax.plot(grid, mean, color=COLOR[arm], lw=2.2, label=LABEL[arm])
        ax.fill_between(grid, lo, hi, color=COLOR[arm], alpha=0.15)
        fin = np.where(np.isfinite(mean))[0]                 # mark where the arm's curve ends
        if len(fin):
            ax.plot(grid[fin[-1]], mean[fin[-1]], "o", color=COLOR[arm], ms=6, zorder=5)
    for R, name in [(R_LOW, "R_low"), (R_MED, "R_med"), (R_HIGH, "R_high")]:
        ax.axhline(R, color="#cbd5e1", ls="--", lw=1)
        ax.text(BUDGET, R + 0.01, name, color="#64748b", fontsize=8, ha="right")
    ax.set_xlabel("environment steps (training)"); ax.set_ylabel("best greedy return so far")
    ax.set_title("MiniGrid-DoorKey PPO tuning — efficiency (line ends where the arm stops)")
    ax.legend(frameon=False, fontsize=9); ax.set_ylim(0, RET_TOP); fig.tight_layout()
    fig.savefig(RESULTS / "fig_curves.png", dpi=130)

    # ── fig 2: solve-rate + steps-to-R_high per arm ──
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    xs = np.arange(len(arms)); cols = [COLOR[a] for a in arms]; labs = [LABEL[a] for a in arms]
    a1.bar(xs, [summary[a]["solve_rate"] * 100 for a in arms], color=cols)
    a1.set_xticks(xs); a1.set_xticklabels(labs, rotation=20, ha="right", fontsize=8)
    a1.set_ylabel("% seeds reaching R_high"); a1.set_ylim(0, 105); a1.set_title("Solve rate")
    means = [summary[a]["steps_to_R_high"]["mean"] for a in arms]
    errs = [[max(0, (summary[a]["steps_to_R_high"]["mean"] - summary[a]["steps_to_R_high"]["ci"][0]))
             if not np.isnan(summary[a]["steps_to_R_high"]["mean"]) else 0 for a in arms],
            [max(0, (summary[a]["steps_to_R_high"]["ci"][1] - summary[a]["steps_to_R_high"]["mean"]))
             if not np.isnan(summary[a]["steps_to_R_high"]["mean"]) else 0 for a in arms]]
    a2.bar(xs, [0 if np.isnan(m) else m for m in means], yerr=errs, color=cols, capsize=4)
    a2.set_xticks(xs); a2.set_xticklabels(labs, rotation=20, ha="right", fontsize=8)
    a2.set_ylabel("steps to R_high (among solved)"); a2.set_title("Time-to-target (lower = better)")
    fig2.suptitle("ADR vs standard AutoRL baselines (equal budget)", fontsize=11)
    fig2.tight_layout(); fig2.savefig(RESULTS / "fig_summary.png", dpi=130)

    # ── fig 3: time to completion — dual axis (left = wall-clock seconds, right = env-steps) ──
    from matplotlib.patches import Patch
    fig3, aL = plt.subplots(figsize=(9, 5.2))
    aR = aL.twinx()
    xs = np.arange(len(arms)); w = 0.38
    tcomp, tlat, smean, serr = [], [], [], [[], []]
    for arm in arms:
        per = list(load(arm).values())
        c = boot_ci([wall_to(rs, R_HIGH, subtract_latency=True) for rs in per])[0]    # RL compute time
        e = boot_ci([wall_to(rs, R_HIGH, subtract_latency=False) for rs in per])[0]   # end-to-end
        tcomp.append(0 if np.isnan(c) else c)
        tlat.append(0 if (np.isnan(c) or np.isnan(e)) else max(0, e - c))             # LLM latency portion
        sm, slo, shi = summary[arm]["steps_to_R_high"]["mean"], *summary[arm]["steps_to_R_high"]["ci"]
        smean.append(0 if np.isnan(sm) else sm)
        serr[0].append(0 if np.isnan(sm) else max(0, sm - slo)); serr[1].append(0 if np.isnan(sm) else max(0, shi - sm))
    cols = [COLOR[a] for a in arms]
    # left axis: stacked TIME bar — RL compute (solid) + LLM latency (light top). total height = end-to-end.
    aL.bar(xs - w / 2, tcomp, w, color=cols)
    aL.bar(xs - w / 2, tlat, w, bottom=tcomp, color=cols, alpha=0.30, hatch="....")
    # right axis: ENV-STEPS bar (hatched) with 95% CI — the latency-immune primary metric.
    aR.bar(xs + w / 2, smean, w, yerr=serr, color=cols, alpha=0.45, hatch="////", capsize=3)
    for x, c, l in zip(xs, tcomp, tlat):
        lab = f" {c:.0f}s" + (f"\n+{l:.0f}s LLM" if l > 0.5 else "")
        aL.text(x - w / 2, c + l, lab, ha="center", va="bottom", fontsize=7)
    for x, m in zip(xs, smean):
        aR.text(x + w / 2, m, f" {m/1000:.0f}k", ha="center", va="bottom", fontsize=7.5)
    aL.set_ylabel("wall-clock time to reach R_high  (seconds)")
    aR.set_ylabel("environment steps to reach R_high")
    if any(t + l > 0 for t, l in zip(tcomp, tlat)):
        aL.set_ylim(0, max(t + l for t, l in zip(tcomp, tlat)) * 1.35)
    if max(smean) > 0:
        aR.set_ylim(0, max(smean) * 1.30)
    aL.set_xticks(xs)
    aL.set_xticklabels([f"{LABEL[a]}\n{summary[a]['solve_rate']*100:.0f}% solved" for a in arms], fontsize=8.5)
    aL.set_title("Time to completion — lower = faster  (single-machine CPU · among solved seeds)")
    legend = [Patch(facecolor="#64748b", label="RL compute time (left)"),
              Patch(facecolor="#64748b", alpha=0.30, hatch="....", label="LLM latency (left)"),
              Patch(facecolor="#64748b", alpha=0.45, hatch="////", label="env-steps (right, ±95% CI)")]
    aL.legend(handles=legend, loc="upper right", fontsize=8, frameon=True)
    fig3.tight_layout(); fig3.savefig(RESULTS / "fig_solvetime.png", dpi=130)

    print(f"\nfigures -> {RESULTS/'fig_curves.png'} , {RESULTS/'fig_summary.png'} , "
          f"{RESULTS/'fig_solvetime.png'}")
    print(f"summary -> {RESULTS/'summary.json'}")


if __name__ == "__main__":
    main()
