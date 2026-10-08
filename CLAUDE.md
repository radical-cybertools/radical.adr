# CLAUDE.md

## Project

`radical.adr` — Autonomous Decision Runtime. An `Operator` runs an
Observe → Decide → Act loop over long-running HPC tasks, tracks convergence
goals as first-class objects, and stops itself when all goals are satisfied.

Key documents:
- `ADR.md` — motivation, autonomy story, positioning vs LangGraph/Academy
- `SYSTEM.md` — full technical reference: decorators, state model, messaging, internals

---

## Environment

```bash
PYTHON=/home/aymen/RADICAL/AGENTIC-ADAPTIVE-DECISION-LAYER/adl_ve/bin/python

cd /home/aymen/RADICAL/AGENTIC-ADAPTIVE-DECISION-LAYER/radical.adl
$PYTHON -m pip install -e .
$PYTHON -m pytest tests/ -v
```

Related repos at `/home/aymen/RADICAL/AGENTIC-ADAPTIVE-DECISION-LAYER/`:
- `radical.asyncflow/` — `WorkflowEngine`, `LocalExecutionBackend`
- `rhapsody/`          — `ConcurrentExecutionBackend`, `engine.start_telemetry()`

---

## Source layout

```
src/radical/adr/
├── operator.py        Operator class; @act / @observe / @goals decorators; run loop
├── goals.py           Goal(name, metric, threshold, direction)
├── boundary.py        Tunable (adaptation boundary), Budget (resource/deadline stop)
├── decision.py        Decision, Action, ActionKind
├── state.py           RuntimeState, Snapshot
├── message.py         MessageKind, InboxMessage, MessageBus
├── observer.py        ObserverBase protocol (lifecycle hooks — unrelated to telemetry)
├── policy/
│   ├── base.py        Policy base, @decide
│   ├── llm.py         LLMPolicy
│   └── combinators.py ChainPolicy
└── tools/
    └── __init__.py    ActionSet (spawn_task, cancel_task, update_param, stop, @act shorthands)

tests/
├── conftest.py        engine fixture
├── unit/              no fixtures, no engine — pure logic
├── integration/       engine fixture, full operator cycles
└── smoke/             end-to-end

examples/              01–10 + demo/
docs/
├── index.mdx          Mintlify tutorial (single page)
└── docs.json          Mintlify config
```

---

## Sacred boundary — the one rule above all

**The `Operator` ONLY: observes, decides, acts.**
It NEVER: schedules work, manages execution lifecycle, resolves task dependencies,
owns resources, or coordinates distributed systems.

AsyncFlow owns workflow managmenet layer.
RHAPSODY owns execution and runtime layer.
ADR steers them.

This boundary is structural — the policy cannot call AsyncFlow directly,
cannot cancel a future without going through a CANCEL_TASK `Action`,
and cannot block waiting for results. Everything flows through the ODA cycle.

---

## Design constraints established for this project

These are non-obvious decisions that must NOT be reversed without explicit discussion.

### `@act` decorator — ADR's own execution primitive
`@act` marks an operator method as an execution primitive. It is NOT the same as
`@tool` in LangGraph or other agentic SDKs. `@tool` is deliberately reserved for
the external agentic SDK the user brings in (e.g. LangGraph). ADR does not define `@tool`.

### `@act` return semantics — three modes, strict contract
- Return `asyncio.Future`  → AsyncFlow dispatch; future tracked, result arrives next cycle
- Return plain value        → local execution; delivered to inbox via `put_nowait` next cycle
- Return `None`             → no-op; method ran but nothing tracked

Mixing these in one method is valid. Which path runs depends on what the method returns
at runtime — not a decorator flag.

### Module-level task implementations
Functions passed to `engine.function_task(fn)` MUST be:
- `async def` at module level (not methods, not nested functions, not lambdas)
- No `self` parameter — AsyncFlow only passes kwargs
- Wrong or extra params → task silently FAILS with no error raised at spawn time

### Always use `ProcessPoolExecutor`, never `ThreadPoolExecutor` for tests
`engine.function_task` serializes callables to worker processes. `ThreadPoolExecutor`
causes subtle test failures. The `conftest.py` fixture uses `ProcessPoolExecutor(max_workers=2)`.

### `engine.start_telemetry()` — on `WorkflowEngine`, not on `Session`
`WorkflowEngine.create(backend)` registers a callback on the backend.
`Session(backends=[backend])` registers a DIFFERENT callback on the same slot.
**They conflict — only one can own the backend.** Always use `engine.start_telemetry()`.
Never construct a `Session` alongside a `WorkflowEngine` on the same backend.

### Telemetry is the user's responsibility — ADR has no telemetry classes
`TelemetryObserver` was deliberately removed. Users subscribe to the telemetry stream
directly in `main()`, update a shared dict, and surface values in `@observe`.
ADR has no telemetry infrastructure and should not grow any.

### API-hardening invariants (items 1–7) — additive / opt-in, never reducing
- **Task-future pruning.** Terminal tasks are popped from `state.task_futures` into a bounded
  `state.resolved` (OrderedDict, `_RESOLVED_MAXLEN`). `_resolve_task_deps` reconstructs a completed
  Future from `resolved` so `depends_on` survives pruning — do NOT make pruning drop a dep silently.
  Child-operator block Futures (uids in `_child_starts.values()`) are EXEMPT from pruning (their
  liveness drives cancel/restart). `CANCEL_TASK`/validation tolerate a pruned uid (no-op, not error).
- **`retry_task`.** `RETRY_TASK` re-spawns via `task_context[uid]` (`_retry_task` → `_spawn_act_result`,
  shared with the SPAWN_TASK local path). Explicit + policy-driven; NO silent auto-retry/`max_retries`
  (would hide failures). Validation requires uid ∈ `task_context`.
- **`max_runtime` + `inflight`.** `max_runtime` is a pure stop condition beside `max_cycles`
  (`_started_at` = `time.monotonic()` at run start). `Operator.inflight` (PENDING+RUNNING count) is a
  read-only property surfaced via `@observe` — NOT auto-injected into the snapshot (keep `@observe`
  the single assembly point). A hard inflight cap was deliberately omitted (would drop spawns).
- **`AnyGoal`/`AllGoal` + `Goal(for_cycles=N)`.** Composites (`_GoalGroup` base) implement
  `.satisfied`; `_check_goals` uses `_eval_goal` with LIST comprehensions (no short-circuit) so every
  leaf's debounce streak (`_goal_streaks`, keyed by goal name on the OPERATOR — goals are
  reconstructed each cycle) updates each cycle. Bare `Goal`/list stays implicit AND.
- **Checkpoint.** `save_checkpoint`/`load_checkpoint` persist STEERING state only (cycle, objectives,
  runtime, task_context, runtime goals via `goal_to_dict`/`goal_from_dict`, `_goal_streaks`,
  `_child_starts`/`_child_config`). Live Futures are NOT serialized — in-flight → `LOST`. Boundary-safe.
- **Hierarchical replay.** `replay_trace(..., operators={name: Cls})` back-fills `operator_cls` on
  `SPAWN_OPERATOR` decisions (still open-loop). `on_start` metadata now includes `children` names.

### Adaptation & resource boundaries (`boundary.py`) — additive, opt-in
New value types in `src/radical/adr/boundary.py` (`Tunable`, `Budget`), exported from the package
root. All of the following reproduce today's behavior exactly when unused.
- **`Tunable(default, low, high, *, int=False, step=None)`** — a declared adaptation boundary as a
  class attribute (annotated or not). `__init_subclass__` collects Tunable attrs into
  `_adl_tunables` and adds their names to `_adl_state_keys`; `_seed_default_objectives` seeds the
  **scalar** `.default` (not the descriptor). Enforcement lives ONLY at the `UPDATE_PARAM` apply seam
  (`_apply_param_update`): with ≥1 Tunable declared, an out-of-bounds / non-`int` / off-`step` write,
  or a write to a non-Tunable key, is **rejected — never silently clamped or dropped**. Rejection is
  soft (NOT via `_validate_actions`, which raises and would abort the cycle): it records
  `rejected_param.<name>` evidence via a persistent `_pending_rejections` buffer that is flushed into
  each cycle's `observations` and cleared once the policy is next consulted. Operator-internal writes
  (`self.x = v`, e.g. `@observe` folds, `build_operator` config) bypass the boundary — only agent
  `UPDATE_PARAM` is gated. `self.tunables` surfaces `{name: value}`; `ActionSet.get_schema` renders the
  bounds into the `update_param` tool description (one declaration → validation + observation + schema).
- **`Budget(metric, limit, direction="max")`** — a resource/deadline stop checked in the run loop
  beside `max_runtime` (`_check_budgets`), reading the metric from `obs` then `snapshot.objectives`.
  It is **not** a `Goal` and is evaluated OUTSIDE `_check_goals`, so no goal composition (an
  unsatisfiable agent-added runtime goal) can block termination. It guarantees the run *stops*; it does
  NOT change runtime-goal composition semantics. Passed via `Operator(..., budget=[Budget(...)])`.
- **`decide_when`** (class attr; `None` = every cycle) — cadence gate applied between OBSERVE and DECIDE
  (`_should_decide`). `"idle"` consults the policy only when `inflight == 0`; `"new_result"` only when
  `snapshot.results` is non-empty. On skipped cycles ADR applies an empty `Decision()` but STILL runs
  `@observe`, goal checks, and budgets. No callable form yet (intent-based, kept small). It is a filter
  on *when* the implicit loop consults the policy — NOT a new loop construct (no `@loop`).
- **Evidence helpers** — `Snapshot.results` / `Snapshot.rejections` (read-only views over
  `observations` `result.*` / `rejected_param.*`; no `latest_result` — dict order is recording order,
  not a semantic latest). `Operator.trend(metric, direction, *, eps)` /
  `stalled(metric, direction, n, *, eps)` compute over `cycle_history` with **consecutive duplicates
  collapsed** (a trial spans many idle cycles that repeat the objective), so they reflect per-change
  trend. `direction` ("maximize"/"minimize") is REQUIRED — numeric change ≠ scientific improvement, and
  it is deliberately NOT inferred from goals (auxiliary metrics need not have one).
- **Static-goal shadow guard** — `_apply_goal_changes` rejects a runtime goal whose `name` or `metric`
  matches a static `@goals` goal (via `_static_goal_identity`). This is collision prevention, NOT goal
  provenance: no `Goal(source=…)` field, no subtype (authoritative-vs-agent stays *derived* from
  `@goals` vs `runtime_goals`). A "must be strictly tighter" comparator is intentionally not built.

### Logging — stdlib, `radical.adr` namespace, side channel only
`enable_logging(...)` (in `src/radical/adr/logging.py`, exported from the package root) is the public
setup. ADR uses Python **stdlib `logging`** (NOT `radical.utils`), per-module
`log = logging.getLogger(__name__)` → the `radical.adr` namespace. The setup configures that
namespace logger with `propagate=False` (RHAPSODY's coexistence pattern) and the shared pipe format
`"%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s"` so ADR/AsyncFlow/RHAPSODY logs
interleave uniformly. **No env vars** (match both siblings — programmatic only). `configure_stack=True`
cascades to `radical.asyncflow.logging.init_default_logger` + `rhapsody.enable_logging` (guarded).
INFO = one line/cycle lifecycle narration; DEBUG = per-action/message; the 4 existing
WARNING/exception sites are unchanged. **Logging is a side channel** — like `LLMPolicy.record_usage`
it must NEVER flow through `state`/`snapshot`/`@observe`; the Observer (`observer.py`) remains the
structured channel. No structured/JSON log output (use a `RecordingObserver` for that).

### `Goal` is the only goal type
`Goal(name, metric, threshold, direction)` — source-agnostic, checks `obs[metric]`.
There is no `ScientificGoal`, no `ResourceGoal`, no goal taxonomy.
Do not re-introduce subtypes under any framing.

### `@observe` is the single assembly point for all metrics
The only data the policy ever sees is the `dict` returned by `@observe`.
Task results, resource metrics, telemetry readings, operator state — everything
must flow through `@observe`. There is no other channel.

### First-class child operators — `add_child` / `spawn_operator` / `ChildHandle`
Child operators are first-class; runtime machinery (`Future`, `UID`, `artifacts`,
`child:*` string keys) must stay OUT of user-facing operator/policy code.
- `add_child(name, OperatorCls, policy=…, **config)` registers + parent-wires a child.
  `policy=` may be a `Policy` instance OR a class/factory called with the child
  (`policy(child)`). `config` is applied as `setattr(child, k, v)` after construction
  (annotated keys proxy into `child.state.objectives`) — NOT passed to `__init__`,
  whose signature is fixed.
- `ActionKind` is now **8**: `SPAWN_TASK`, `CANCEL_TASK`, `RETRY_TASK`, `SPAWN_BLOCK`,
  `SPAWN_OPERATOR`, `RESTART_OPERATOR`, `CANCEL_OPERATOR`, `UPDATE_PARAM`. The three
  operator verbs are dedicated `_apply_one` branches (NOT registry-routed sentinels —
  `__init_subclass__` builds the act registry from `vars(cls)` only, so they could not
  be inherited anyway). `SPAWN_OPERATOR`/`RESTART_OPERATOR` reuse `_track_future`
  (shared with `SPAWN_TASK`); `CANCEL_OPERATOR` tracks nothing (like `CANCEL_TASK`).
- `Action.depends_on` is interpreted by `kind`: task **uids** for `SPAWN_TASK`,
  child **names** for `SPAWN_OPERATOR`/`RESTART_OPERATOR` (resolved via the shared
  `_resolve_child_deps`, `_child_starts[name]`). Do NOT add a separate `after` field —
  the shared `depends_on` is intentional.
- `Action.operator_cls` / `Action.policy` are `Field(exclude=True)` live refs, so a
  `SPAWN_OPERATOR` decision does NOT round-trip through record/replay (`model_dump`
  stays JSON-safe). That path targets flat policies.
- `ChildHandle` (from `self.child(name)`) reads `status`/`started`/`done`/`failed` from
  the parent's own `_child_starts`+`runtime` (the child block Future), and metrics from
  the parent's persistent `objectives` (`child:{name}:obj:{metric}`). `.started` is
  `status != "NOT_STARTED"` — the status gate policies use instead of a spawn-once flag.
  Child runtime/objectives propagate to the parent's PERSISTENT `objectives` (not the
  per-cycle `observations`) precisely so a metric survives the cycle gap between a
  child's final SNAPSHOT and its block-Future `DONE` event.
- `start()` is now an internal mechanic (`_spawn_operator`/`_restart_operator` call it);
  it is NOT a documented user path. There is no two-tier "drop to raw start()" story.
- **Spawn-once / idempotency is runtime-owned, not a user flag.** `spawn_operator` is a
  no-op + `log.warning` when `name in _child_starts` (idempotency key = "ever started",
  INCLUDING terminal `DONE`/`FAILED`/`CANCELED`); `_spawn_operator` returns `None` and
  `_apply_one` skips `_track_future`. `restart_operator(name)` re-runs: cancel in-flight
  → `_reset_run_state` (fresh `RuntimeState` + `_seed_default_objectives()` + re-apply
  `_child_config[name]` + clear `_block_factory`) → relaunch. `cancel_operator(name)`
  cancels the block Future, leaving a `CANCELED` tombstone (`name` stays in
  `_child_starts`). Do NOT reintroduce a `fanned_out`-style user guard or an
  `if_exists=`/`once=` param — intent lives in the verb.
- `_child_config[name]` is the initial-config snapshot captured by `add_child` (a copy
  of `**config`); it is what `restart` replays. Keep it in `_RESERVED` so it is not
  proxied to objectives.

### `update_param` writes to `state.objectives`
`Action(kind=UPDATE_PARAM, param_key=k, param_value=v)` sets `state.objectives[k] = v`.
This is visible in the next cycle's `@observe` via `snapshot.objectives.get(k)` AND
via `self.k` (the proxy). It is NOT in `snapshot.observations`.

### Goal checking fires after DECIDE, before ACT
`_check_goals(obs, snapshot, decision)` sets `decision.stop = True` if all goals are
satisfied. The policy cannot prevent this. Actions in that cycle still execute (ACT
still runs on the stop cycle).

### Static goals are re-evaluated every cycle, not cached
`_check_goals` calls `_adl_goals_fn(self)` fresh each cycle. Goal objects are
reconstructed each call. Do not rely on object identity across cycles.

### `cycle_history` excludes the current cycle
`snapshot()` appends to `state.history` BEFORE `@observe` runs. So `state.history[-1]`
is always the current cycle's snapshot. `cycle_history` returns `list(history)[:-1]`
to give the policy only completed past cycles.

### `_runtime_goals` upserts by name
`Decision.goals` triggers `_apply_goal_changes` which upserts by `goal.name`.
Same name → replaces in-place. New name → appends. The list is mutable and persists
across cycles. `Decision.remove_goals` removes by name.

### `state.task_futures` accumulates and is never cleaned up per-task
Futures are stored on task spawn and never removed when the task completes.
`_sync_running_states` iterates over all of them every cycle. For very long runs
this grows without bound. Known limitation — do not add workarounds silently.

### `_RESERVED` frozenset controls the proxy
Any instance attribute in `_RESERVED` is NOT proxied to `state.objectives`.
Any annotated class attribute NOT in `_RESERVED` IS proxied (read/write).
When adding infrastructure attributes to `Operator.__init__`, add them to `_RESERVED`
and use `object.__setattr__` to set them.

### `depends_on` IS now in the LLM schema (gap G7 — closed)
`get_schema(...)` renders an optional `depends_on` parameter on every spawnable tool, plus the
child-operator lifecycle verbs and `retry_task`. `depends_on` is marked optional via
`_OPTIONAL_PARAMS` in `tools/__init__.py` (not in the JSON-schema `required` list).

### IMPRESS integration — run pipelines locally inside `@act`
IMPRESS uses AsyncFlow internally. Nesting `engine.function_task` inside an IMPRESS
pipeline call creates a conflict. The correct pattern is: run IMPRESS pipelines **locally**
inside an `@act` method using `asyncio.gather` for batch parallelism. Do not use
`function_task` to dispatch IMPRESS pipelines.

---

## Testing conventions

```bash
$PYTHON -m pytest tests/ -v              # all tiers
$PYTHON -m pytest tests/unit/ -v         # pure logic, no engine
$PYTHON -m pytest tests/integration/ -v  # operator cycles
$PYTHON -m pytest tests/smoke/ -v        # end-to-end
```

- `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed
- `conftest.py` provides a single `engine` fixture (async, scoped to function)
- Integration tests define inline `_Op` / `_Policy` classes, one `async def test_*`
- Unit tests build `Snapshot` objects directly using `_snap()` / `_snap_with_objectives()`
  helpers defined at the top of each test file

---

## Documentation conventions

- **`ADR.md`** — motivation, the autonomy story (6 properties), positioning vs LangGraph/Academy.
  No API details, no implementation mechanics. Written for someone deciding whether to use ADR.

- **`SYSTEM.md`** — technical reference: decorators, state namespaces, messaging protocol,
  action execution path, goal checking internals, runtime internals. Written for someone
  implementing against ADR.

- **`docs/index.mdx`** — Mintlify single-page tutorial. Shows a complete working example
  from scratch. Keep it runnable. No API details not shown in examples.

- **`CLAUDE.md`** (this file) — development guide for Claude. What you need to know that
  is not obvious from reading the source.

When updating docs, keep the separation: ADR.md stays conceptual, SYSTEM.md stays technical.
Do not let implementation details leak into ADR.md.

---

## Snapshot state namespaces (quick reference)

| Namespace | Contents | Lifetime |
|---|---|---|
| `observations` | `result.{uid}` (done), `error.{uid}` (failed), child per-cycle obs (`child:{id}:{k}`) | Cleared each cycle |
| `runtime` | `uid → PENDING/RUNNING/DONE/FAILED/CANCELED` | Persists |
| `objectives` | annotated operator attrs, `update_param` writes, child runtime/objectives mirror (`child:{id}:runtime:{k}`, `child:{id}:obj:{k}`) | Persists |
| `artifacts` | output paths / identifiers | Persists |
| `directives` | parent instructions | Cleared each cycle |
| `task_context` | `uid → {name, kwargs}`, plus `action_id` with `track_action_ids=True` | Persists |

Not in `Snapshot` (live references on `RuntimeState`):
- `task_futures: dict[str, asyncio.Future]`
- `history: deque[Snapshot]` (maxlen=64)

---

## Running examples

For each example, and specifically for LocalExecutionBackend or
RHAPSODY ConcurrentExecutionBackend use ProcessPoolExecutor and
not ThreadPoolExecutor. 

```bash
PYTHON=/home/aymen/RADICAL/AGENTIC-ADAPTIVE-DECISION-LAYER/adl_ve/bin/python
cd /home/aymen/RADICAL/AGENTIC-ADAPTIVE-DECISION-LAYER/radical.adl

$PYTHON examples/01-asyncflow-adaptive-search.py          # minimal, no API key
$PYTHON examples/08-hierarchical-operator.py              # parent-child DAG
$PYTHON examples/10-resource-aware-search.py --local      # resource goals, no RHAPSODY

# LLM examples require OPENROUTER_API_KEY
export OPENROUTER_API_KEY="sk-or-..."
$PYTHON examples/03-asyncflow-llm-adaptive-search.py
$PYTHON examples/09-langgraph-policy.py   # pip install langgraph langchain-openai
```