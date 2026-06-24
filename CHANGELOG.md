# Changelog

All notable changes to `radical.adr` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Six validated gaps from downstream (Campaign Manager) integration feedback, each
assessed against source and landed as small, additive changes, plus first-class child
operators (`SPAWN_OPERATOR` / `add_child` / `ChildHandle`). Full suite: 74 passing.

### Added

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
  rewritten `examples/08-hierarchical-operator.py` (adaptive fan-out). Spawn-once
  idempotency remains a user-managed guard flag pending a first-class follow-up.
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
- **Docs** — `ADR.md` gains a multi-goal AND-convergence example; `SYSTEM.md` documents
  the 3-arg `on_cycle` protocol, the record/replay workflow, and the `timeout` /
  `LLMPolicy` helpers.

### Fixed

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
