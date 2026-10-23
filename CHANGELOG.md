# Changelog

All notable changes to `radical.adr` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Six validated gaps from downstream (Campaign Manager) integration feedback, each
assessed against source and landed as small, additive changes, plus first-class child
operators (`SPAWN_OPERATOR` / `add_child` / `ChildHandle`) with declarative,
runtime-owned lifecycle verbs (idempotent `spawn_operator`, `restart_operator`,
`cancel_operator`), user-facing `enable_logging(...)`, a 7-item API-hardening pass
(task pruning, `retry_task`, `max_runtime`/`inflight`, `AnyGoal`/`AllGoal`/`for_cycles`, fuller LLM
schema, hierarchical replay, checkpoint/resume), and a domain-agnostic adaptation-boundary layer
(`Tunable`, `Budget`, `decide_when`, evidence helpers) distilled from a real offline-RL steering
use case.

### Added

- **Optional action-to-task links.** `Operator(..., track_action_ids=True)` assigns
  a fresh `Action.action_id` per emitted occurrence before observer notification
  and retains it in task context for tasks, blocks, retries, and child-operator
  launches/restarts. Tracking defaults to `False`; unset IDs are omitted from
  serialization. Checkpoints preserve task links; recording them requires a
  custom observer or explicit inclusion in `@observe`.

- **Adaptation boundary, budgets & evidence helpers (additive / opt-in; new module
  `radical.adr.boundary`, exported from the package root).** Distilled from a real offline-RL steering
  workload (`examples/antmaze-offline-rl/`), all domain-agnostic.
  - **`Tunable(default, low, high, *, int=False, step=None)`** — declares a bounded, agent-tunable
    parameter as an operator class attribute. One declaration is the single source of truth: ADR seeds
    the scalar default, **validates** every agent `update_param` against the bounds at the
    `UPDATE_PARAM` seam (out-of-bounds / non-`int` / off-`step`, or a write to a non-`Tunable` key, is
    **rejected and surfaced as evidence** — never silently clamped or dropped, never aborts the cycle),
    exposes `self.tunables`, and renders the bounds into the agent tool schema (`get_schema`).
    Operator-internal writes bypass the boundary — only agent `UPDATE_PARAM` is gated.
  - **`Budget(metric, limit, direction="max")`** via `Operator(..., budget=[...])` — a
    resource/deadline stop checked beside `max_runtime`, **outside** goal checking, so no goal
    composition (an unsatisfiable agent-added runtime goal) can block termination. It is not a `Goal`
    and does not change convergence semantics.
  - **`decide_when = "idle" | "new_result"`** (operator class attr) — gates *when* the policy is
    consulted; skipped cycles still run `@observe`, goals, and budgets. Not a loop construct.
  - **Evidence helpers** — `Snapshot.results` / `Snapshot.rejections` (read-only views over
    `observations`); `Operator.trend(metric, direction)` / `stalled(metric, direction, n)`
    (direction-aware, over the *distinct* value series so idle-cycle repeats don't skew them).
  - **Static-goal shadow guard** — a runtime goal reusing a static `@goals` goal's `name` or `metric`
    is rejected, keeping the authoritative objective outside the agent's reach (no `Goal(source=…)`).
  - Deferred (documented, not built): a cross-run outcome hook, and usage/cost rollup (provider prices
    stay in the app, not core).

- **API hardening (7 additive/opt-in items; no existing behavior changes unless opted in).**
  - **Bounded task tracking** — terminal task Futures are now pruned from `state.task_futures`
    (kept in a bounded `state.resolved` cache), so long campaigns no longer leak memory or slow
    `_sync_running_states`. `depends_on` on a pruned task still resolves (the result is replayed as
    a completed Future) — DAG chaining is unaffected.
  - **`acts.retry_task(uid)`** (new `ActionKind.RETRY_TASK`) — re-spawn a finished/failed task from
    its recorded `task_context` (same `@act` + kwargs, fresh uid). Policy-driven recovery.
  - **`Operator(..., max_runtime=<seconds>)`** — opt-in wall-clock cap, a sibling to `max_cycles`;
    plus a read-only **`Operator.inflight`** (PENDING+RUNNING count) to surface in `@observe` for
    self-throttling. (A hard inflight cap was deliberately *not* added — it would drop spawns.)
  - **`AnyGoal([...])` / `AllGoal([...])`** composite goals (OR / explicit AND, nestable) and
    **`Goal(..., for_cycles=N)`** debounce (must hold N consecutive cycles). A bare `Goal`/list is
    still implicit AND — unchanged.
  - **`get_schema()` now renders `depends_on`** (optional) on every spawnable tool and the
    child-operator lifecycle verbs + `retry_task` — closes the documented LLM-schema gap (G7).
  - **`replay_trace(path, policy, operators={name: Cls})`** — hierarchical traces are now
    replayable offline (the `operators` map back-fills `operator_cls` on `SPAWN_OPERATOR`
    decisions). `operators=None` ⇒ unchanged.
  - **`Operator.save_checkpoint(path)` / `load_checkpoint(path)`** — serialize/restore *steering*
    state (cycle, objectives, runtime, task_context, runtime goals, goal streaks, child lifecycle
    metadata) so a preempted campaign can resume. Live execution is not snapshotted (boundary-safe);
    in-flight tasks are marked `LOST` for the resumed policy to `retry_task`.
- **User-facing logging** — new `enable_logging(...)` (exported from the package root) sets up
  leveled, human-readable narration of the Observe→Decide→Act loop under the `radical.adr`
  namespace. **INFO** gives a one-line decision summary per cycle plus operator start /
  goal-satisfied stop / child spawn-restart-cancel / final-stop milestones; **DEBUG** adds
  per-action and per-task detail; the existing WARNING/exception sites are unchanged.
  Format-compatible with AsyncFlow and RHAPSODY (same pipe-delimited line, `propagate=False` so ADR
  logs never duplicate and survive a sibling reconfiguring root), with an opt-in
  `configure_stack=True` to configure the whole stack at one level in a single call. Programmatic
  only (no env vars); no structured/JSON output — logging is a side channel and never flows through
  `state`/`snapshot`/`@observe` (use a `RecordingObserver` for structured per-cycle data).
- **First-class child operators** — child operators are now composed without runtime
  plumbing. New `ActionKind.SPAWN_OPERATOR` + `acts.spawn_operator(name, OperatorCls,
  policy=…, depends_on=[names], **config)` *register-or-reuse then launch* a child as a
  block; names, count, config, and dependencies are all runtime values, so adaptive
  fan-out ("explorer finds a region → spawn N refiners around it") needs no escape
  hatch. `Operator.add_child(name, OperatorCls, policy=…, **config)` declares a child
  (`policy=` accepts a `Policy` instance OR a class/factory called with the child).
  `Operator.child(name)` returns a read-only `ChildHandle` (`obs["explorer"].done`,
  `obs["explorer"].radius`). `Future`/`UID`/`artifacts`/`child:*` string keys no longer
  appear in user-facing operator or policy code. `depends_on` is shared across action
  kinds — task **uids** for `SPAWN_TASK`, child **names** for `SPAWN_OPERATOR`. See the
  rewritten `examples/08-hierarchical-operator.py` (adaptive fan-out).
- **Declarative operator-lifecycle verbs** — child lifecycle is now runtime-owned, with
  no user-managed spawn-once flag. `acts.spawn_operator(...)` is **idempotent**: once a
  child has ever been started (including terminal `DONE`/`FAILED`/`CANCELED`), a re-emit
  is a no-op + a logged warning — replacing the silent "N overlapping runs" bug an
  unguarded re-emit used to cause. New `acts.restart_operator(name, depends_on=…)` re-runs
  a registered child from its **initial** config (cancel in-flight → reset → relaunch), and
  `acts.cancel_operator(name)` cancels a child's block Future, leaving a `CANCELED`
  tombstone. New `ChildHandle.started` (`status != "NOT_STARTED"`) is the status gate
  policies use instead of a boolean flag — `if not child.started: spawn`. `ActionKind` is
  now 7 (`RESTART_OPERATOR`, `CANCEL_OPERATOR` added).
- **Record / replay** — new `RecordingObserver` and `replay_trace()` (in
  `radical.adr.recording`, exported from the package root) to record a live run's
  per-cycle observation stream to JSONL and replay it offline — no engine, no tasks —
  against any other policy. Enables deterministic decision-level A/B comparison,
  regression testing, and debugging. Replays the policy only (open-loop); it does not
  re-execute actions. See `examples/11-record-replay.py`.
- **`Goal(inclusive=...)`** — optional boundary-inclusive comparison. `inclusive=True`
  turns `<`/`>` into `<=`/`>=`, so a goal like "reach 5 hits" fires at exactly 5.
  Defaults to `False` (strict), preserving existing behavior.
- **`Policy(..., timeout=...)`** — optional per-call timeout on the composed primary
  policy. A primary that hangs now triggers the fallback instead of blocking the
  Observe→Decide→Act cycle forever.
- **`LLMPolicy.record_usage(...)`** — bounded local bookkeeping of per-call
  token/latency/cost (deque, maxlen 256). Purely local to the policy — never flows
  through `state`, `snapshot`, or `@observe`.

### Changed

- **Pre-flight action validation** — `Decision` actions are now validated as a batch
  *before* any are applied (validate-all-then-apply-all). A single invalid action
  raises one `ActionError` listing every problem and aborts the batch, so an invalid
  action can no longer leave earlier actions in the same `Decision` partially applied.
- **`examples/08-hierarchical-operator.py` is now status-gated** — the `fanned_out`
  spawn-once guard flag is gone; the campaign policy gates fan-out on the children's own
  observed `ChildHandle.started`, relying on idempotent `spawn_operator`.
- **Docs** — `ADR.md` gains a multi-goal AND-convergence example; `SYSTEM.md` documents
  the 3-arg `on_cycle` protocol, the record/replay workflow, and the `timeout` /
  `LLMPolicy` helpers.

### Fixed

- **`@act` tool schemas match the method signatures.** `ActionSet.get_schema("openai")` and
  `get_schema("anthropic")` marked every `@act` parameter required, including ones with defaults.
  They rendered `*dep_futures` and `**kwargs` as ordinary parameters, so an `@act` taking
  upstream futures could not be called from a tool call: its `dep_futures` argument was rejected
  as an unknown kwarg. They also left string annotations (`from __future__ import annotations`)
  untyped. Defaulted parameters are now optional, variadic parameters are not rendered, and
  string annotations are resolved with `typing.get_type_hints`; an unresolvable one still
  renders untyped. The `langgraph` format builds tools from the `ActionSet` callables and is
  unchanged.

- With action tracking enabled, local results and Futures without a backend `id`
  receive fresh task UIDs, preventing reused Python object addresses from
  overwriting context. Backend-assigned IDs are retained; task UID behavior with
  tracking disabled is unchanged.

- A hung primary policy in a `Policy(primary, fallback)` composition previously blocked
  the ODA loop indefinitely — fallback only triggered on raise/`None`, never on a hang.
  Now bounded by `timeout`.
- A goal whose metric landed exactly on its threshold never fired under the hardcoded
  strict `<`/`>`. Addressable via the new `inclusive` flag.

### Breaking

- **`ObserverBase.on_cycle`** signature changed from `(self, snapshot, decision)` to
  `(self, snapshot, obs, decision)`. Observers now receive the `@observe` output
  directly. Custom observers must add the `obs` parameter. No compatibility shim
  (project's MVP no-backward-compat stance).
- **`Operator.spawn_child()` method removed**, replaced by `add_child(name,
  OperatorCls, policy=…, **config)` (the old method hardcoded the base `Operator`
  class and had no callers). Use `add_child` + `acts.spawn_operator(...)`.
- **Child SNAPSHOT propagation destination changed.** A child's per-cycle observations
  still mirror into the parent's per-cycle `observations` (`child:{id}:{k}`), but its
  *persistent* runtime/objectives now mirror into the parent's persistent `objectives`
  (`child:{id}:runtime:{k}`, `child:{id}:obj:{k}`) instead of the per-cycle
  `observations` — so the latest child metrics survive across parent cycles. Code that
  read `snapshot.observations["child:{id}:obj:..."]` must read
  `snapshot.objectives["child:{id}:obj:..."]` (or, preferably, use `ChildHandle`).
