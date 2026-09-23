# experiment_results/ — raw measured data (NEVER mixed with reference_results/)

Each arm appends **raw per-record** rows (one per trial = one `train_and_eval` call) to its own JSONL:
`random.jsonl`, `hyperband.jsonl`, `pbt.jsonl`, `adr_rule.jsonl`, `adr_llm.jsonl`. Aggregation and
statistics happen only in `analyze.py`.

## Per-record fields (one JSON object per line)

| Field | Type | Meaning |
|---|---|---|
| `arm` | str | `random` / `hyperband` / `pbt` / `adr_rule` / `adr_llm` |
| `seed` | int | Experiment seed (0..K-1); fixes env + torch RNG. |
| `config_id` | str | Identity of the hyperparameter config this trial trained. |
| `config` | object | `learning_rate`, `clip_coef`, `gae_lambda`, `entropy_coef`, `n_epochs`, `num_steps`. |
| `trial` | int | Global trial index within this (arm, seed) run, in execution order. |
| `env_steps` | int | TRAINING env steps consumed by this trial (the budget unit). |
| `cum_env_steps` | int | Cumulative training env steps after this trial — the **budget clock**. |
| `eval_steps` | int | Eval env steps (fixed overhead; not budgeted). |
| `wall_time` | float | Cumulative wall-clock seconds after this trial. |
| `return` | float | Greedy eval return after this trial (CartPole-v1: max 500). |
| `best_return_so_far` | float | Max `return` seen so far in this (arm, seed) run. |
| `solved` | bool | Greedy eval mean return >= solve threshold. |
| `diagnostics` | object | `kl`, `entropy`, `clip_frac`, `value_loss`, `explained_var`, `grad_norm` from the last PPO update — the **white-box evidence** the ADR policies condition on. |
| `decision` | object | (ADR arms) the policy's reasoning + param changes that led to this trial. Empty for baselines. |
| `tokens` | int | LLM tokens for that decision (0 for baselines / rule). |
| `cost` | float | Estimated USD cost of that decision (0 for baselines / rule). |

## Derived in analyze.py (not stored raw)

- `steps_to_R`, `trials_to_R`, `walltime_to_R` for `R in {R_low, R_medium, R_high}` — first record in an
  (arm, seed) run with `best_return_so_far >= R`; censored if never reached within budget.
- Per-arm mean ± 95% CI over the K seeds.

## targets.json (written by the reference phase, then frozen)

`{ "R_low", "R_medium", "R_high", "solve_threshold", "total_env_step_budget", "step_chunk",
   "eval_episodes", "frozen": true, "notes" }`
