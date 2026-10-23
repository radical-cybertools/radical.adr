# ADR System Overview

Technical reference for the Autonomous Decision Runtime. For motivation and scientific context, see [ADR.md](ADR.md).

---

## 1. Stack Position

```
┌──────────────────────────────────────────────────────────────┐
│  Domain Science                                              │
│  IMPRESS · ROSE · Campaign Manager · custom scripts         │
│                                                              │
│    ┌────────────────────────────────────────────────────┐   │
│    │  radical.adr  (ADR — Autonomous Decision Runtime)  │   │
│    │  Operator · Policy · Goals · Decision · ActionSet  │   │
│    └────────────────────────┬───────────────────────────┘   │
└─────────────────────────────│────────────────────────────────┘
                              │  AsyncFlow API
┌─────────────────────────────▼────────────────────────────────┐
│  radical.asyncflow                                           │
│  WorkflowEngine · function_task · engine.block              │
└─────────────────────────────┬────────────────────────────────┘
                              │  execution backend
┌─────────────────────────────▼────────────────────────────────┐
│  RADICAL-Pilot · LocalExecutionBackend · HPC cluster        │
└──────────────────────────────────────────────────────────────┘
```

**Sacred boundary:** the Operator only Observes, Decides, and Acts. It never owns scheduling, execution lifecycle, or distributed resources.

---

## 2. The Adaptive Cycle

```
┌──────┐    ┌─────────┐    ┌────────┐    ┌─────┐
│  RUN │ →  │ OBSERVE │ →  │ DECIDE │ →  │ ACT │ → (loop)
└──────┘    └─────────┘    └────────┘    └─────┘
```

| Step | What happens |
|------|-------------|
| **RUN** | Drain inbox (up to 64 messages): task completions, child snapshots, directives. Poll `future.state` to promote PENDING→RUNNING. |
| **OBSERVE** | Call `@observe(snapshot)` → plain `dict` of domain observations. |
| **DECIDE** | Call `policy.decide(obs)` → `Decision`. Check all goals; force `stop=True` if all satisfied. |
| **ACT** | Apply each `Action`: call `@act` methods, cancel futures, update state. Dispatch directives/snapshots. |

After each cycle, a `Snapshot` is yielded. `observations` and `directives` are cleared; `runtime`, `objectives`, and `artifacts` persist.

---

## 3. Core Abstractions

### 3.1 Operator

Central runtime class. Subclass to define a campaign:

```python
class SearchOperator(Operator):
    center: float = 0.0   # annotated attrs → state.objectives (persist)
    radius: float = 5.0

    @goals
    def convergence_criteria(self):
        return Goal(name="converged", metric="radius",
                    threshold=0.005, direction="minimize")

    @act
    async def evaluate(self, x: float, seed: int = 0) -> dict:
        rng = random.Random(seed)
        return {"x": x, "value": (x - TARGET) ** 2 + rng.gauss(0, 0.03)}

    @observe
    def extract(self, snapshot) -> dict:
        ...
```

**Lifecycle methods:**

| Method | Description |
|--------|-------------|
| `run()` | Async generator — yields one `Snapshot` per cycle |
| `run_to_completion()` | Consumes `run()`, returns the final `Snapshot` |
| `start(*dep_futures)` | Registers `run_to_completion()` as an `engine.block`; returns the block `Future`. Used for operator-level DAG chaining. |
| `shutdown()` | Sets stop event, posts STOP to inbox |

**Adaptation boundary, cadence & budgets** (all opt-in — see §3.9):

- Declare `Tunable(default, low, high)` class attributes; ADR bounds every agent `update_param` against them.
- Set `decide_when = "idle" | "new_result"` (class attr) to consult the policy only on cycles that carry new evidence.
- Pass `Operator(..., budget=[Budget("trials", 12)])` for a hard stop that goal state can never block.
- Read evidence with `self.trend(metric, direction)`, `self.stalled(metric, direction, n)`, `self.tunables`, `self.rejections`.

---

### 3.2 Decorators

Registered once at class definition time via `__init_subclass__`.

#### `@observe` (mandatory)

Receives the cycle's `Snapshot`, returns a `dict`. Called every cycle before DECIDE. If missing, `run()` raises `TypeError` immediately — there is no default observation.

`@observe` is the **single assembly point for all metrics** — task results from `snapshot.observations`, operator state from `snapshot.objectives`, or any external source (e.g. a telemetry subscription updated by the user in `main()`). `Goal.satisfied()` reads exclusively from the `obs` dict that `@observe` returns.

#### `@goals` (optional)

Returns a `Goal` or `list[Goal]`. These are **static goals** evaluated alongside runtime goals every cycle.

#### `@act`

Marks an async method as an operator capability. The method **runs in the operator's event loop with `self` available**. The return value determines execution mode:

| Return type | Execution mode |
|-------------|---------------|
| `asyncio.Future` | Async — future tracked via done-callback, result arrives in next cycle |
| Plain value (`dict`, etc.) | Local — value delivered to inbox via `put_nowait`, visible next cycle |
| `None` | No-op — method ran but produced no trackable result |

**Local execution** (computation inline, no AsyncFlow overhead):
```python
@act
async def evaluate(self, x: float, seed: int = 0) -> dict:
    rng = random.Random(seed)
    return {"x": x, "value": (x - TARGET) ** 2 + rng.gauss(0, 0.03)}
```

**Backend task** (dispatches to `engine.function_task`; non-blocking, returns Future):
```python
@act
async def dock(self, candidate: str) -> dict:
    return self.engine.function_task(_dock_impl)(candidate=candidate)
```

**Orchestration block** (sequences sub-tasks; returns block Future):
```python
@act
async def md_refine(self, candidate: str):
    async def _block(c: str) -> dict:
        prepped = await self.engine.function_task(_prep)(c)
        return await self.engine.function_task(_simulate)(prepped)
    return self.engine.block(_block)(candidate)
```

**Child operators** are first-class — you do not write a launch `@act`. Declare a child with `add_child`, launch it from the policy with `acts.spawn_operator(name, …)`, and observe it via `self.child(name)`. `SPAWN_OPERATOR` reuses the same Future-tracking path as `SPAWN_TASK` internally (`child.start(*dep_futs)` wrapped as an `engine.block`), so the `Future`/`UID`/`artifacts` machinery stays out of user code. See §5.3.

Module-level implementations (`_dock_impl`, `_prep`, etc.) are plain async functions — pure, testable, independent of ADR. The `@act` method is the dispatch point; the implementation is separate.

`@act` methods that need upstream results accept `*dep_futures` as positional args. `_apply_one` resolves `act.depends_on` UIDs to live futures and passes them through.

---

### 3.3 Policy

```python
class SearchPolicy(Policy):
    def __init__(self, op):
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        ...
```

`@decide` is mandatory — an empty `Policy` with no composition raises `PolicyError` on first call. Composition: `Policy(primary=LLMPolicy(), fallback=RulePolicy(), timeout=30.0)`. When `timeout` is set, a primary call that hangs longer than `timeout` seconds triggers the fallback (alongside the existing raise / `None` triggers).

**`LLMPolicy` helpers.** `LLMPolicy` is a thin `Policy` subclass providing `render_observation(obs)` (default: JSON dump of the obs dict for the user turn) and `record_usage(tokens=, latency=, cost=)`, which appends one entry to a bounded `usage_history` deque (maxlen 256). Usage tracking is purely local bookkeeping — call `record_usage` manually after your vendor call inside `@decide`. It never flows through `state`, `snapshot`, or `@observe`; ADR grows no telemetry pipeline.

---

### 3.4 Goals

```python
Goal(name="mse",        metric="model_mse",   threshold=0.005, direction="minimize")
Goal(name="gap",        metric="band_gap_ev", threshold=1.5,   direction="maximize")
Goal(name="cpu_health", metric="cpu_percent", threshold=90.0,  direction="minimize")
```


`satisfied(obs, snapshot)` checks `obs[metric]` first, then `snapshot.objectives[metric]` as fallback. **`snapshot.runtime` is not checked automatically** — resource or telemetry metrics must be surfaced explicitly in `@observe`.

**Goals are source-agnostic.** Whether a metric comes from a task result, operator state, or an external telemetry subscription is determined entirely by what `@observe` puts into `obs`.

**Runtime goals** — proposed by policy via `Decision.goals`. Same `name` upserts in-place; new `name` adds independent requirement. Removed via `Decision.remove_goals`.

Convergence check fires after DECIDE: if all `static_goals + runtime_goals` are satisfied, `decision.stop` is forced `True`.

---

### 3.5 Decision and Actions

```python
class Decision:
    actions:      list[Action]          = []
    goals:        list[Goal]            = []
    remove_goals: list[str]             = []
    stop:         bool                  = False
    delegate_to:  list[str]             = []   # child IDs to send DIRECTIVE
    notify:       list[tuple[str, Any]] = []   # (child_id, payload)
    directives:   dict[str, Any]        = {}   # payload for delegate_to
```

**ActionKind:**

| Kind | Effect |
|------|--------|
| `SPAWN_TASK` | Look up `@act` in `_adl_act_registry`, `await fn(self, *dep_futures, **kwargs)`, handle return |
| `SPAWN_BLOCK` | Alias for `SPAWN_TASK` — same lookup and execution path |
| `SPAWN_OPERATOR` | Register-or-reuse a child operator by name, then launch it as a block (`child.start(*dep_futs)`); track its block Future like any spawned task. **Idempotent**: if the child has ever been started (`name in _child_starts`), it is a no-op + a logged warning |
| `RESTART_OPERATOR` | Re-run a registered child: cancel any in-flight run, reset the child to its **initial** config (`_child_config[name]`), then launch fresh. Not idempotent |
| `CANCEL_OPERATOR` | `future.cancel()` on the child's block Future, leaving a `CANCELED` tombstone (name stays in `_child_starts`) |
| `CANCEL_TASK` | `future.cancel()` on tracked future |
| `UPDATE_PARAM` | `state.objectives[key] = value` |

`depends_on: list[str]` on every `Action`, interpreted by `kind`: for `SPAWN_TASK`/`SPAWN_BLOCK` it lists upstream task **UIDs** in `state.task_futures`; for `SPAWN_OPERATOR` it lists upstream child operator **names** (resolved internally via `_child_starts[name] → task_futures[uid]`). Either way the resolved Futures gate the launch — enabling native DAG chaining without the policy ever handling a Future or uid.

For `SPAWN_OPERATOR`, `task_name` is the child **name**, `task_kwargs` is the child's **config** (objective seeds), and `operator_cls` / `policy` carry live Python references that are `exclude=True` (so `model_dump()` stays JSON-safe — a `SPAWN_OPERATOR` decision does not round-trip through record/replay).

---

### 3.6 ActionSet

```python
acts = op.get_actions()

acts.evaluate(x=1.0)                           # @act shorthand → SPAWN_TASK
acts.spawn_task("evaluate", x=1.0)             # explicit → SPAWN_TASK
acts.spawn_workflow("md_refine", candidate=c)  # alias → SPAWN_TASK
acts.cancel_task(uid)                          # CANCEL_TASK
acts.update_param("radius", 2.5)               # UPDATE_PARAM
acts.stop()                                    # Decision(stop=True)

# spawn_operator — launch a child operator by name (register-or-reuse, idempotent):
acts.spawn_operator("explorer")                            # launch a declared child (no-op if started)
acts.spawn_operator("refiner_0", SearchOperator,           # register + launch a new one
                    policy=SearchPolicy, center=3.1)

# operator lifecycle verbs — re-run / stop a registered child:
acts.restart_operator("explorer")                          # cancel + reset-to-initial + relaunch
acts.cancel_operator("explorer")                           # cancel in-flight → CANCELED tombstone

# depends_on — task UIDs for SPAWN_TASK, child names for SPAWN_OPERATOR/RESTART_OPERATOR:
acts.spawn_task("refine", depends_on=[task_uid])           # gate on an upstream task
acts.spawn_operator("merge", MergeOp, depends_on=["refiner_0", "refiner_1"])
```

`depends_on` is a first-class parameter on all spawning methods — it is extracted from kwargs and set on `act.depends_on`, not passed as a task kwarg. `spawn_operator` / `restart_operator` accept a single name or a list of names.

LLM schema: `op.get_actions("openai")` / `"anthropic"` / `"langgraph"` — renders all `@act` signatures and docstrings in vendor format.

---

### 3.7 State Model

| Namespace | Contents | Lifetime |
|-----------|----------|----------|
| `observations` | Task results (`result.{uid}`), task errors (`error.{uid}`), child per-cycle obs (`child:{id}:{k}`) | Cleared each cycle |
| `runtime` | UID → `PENDING`/`RUNNING`/`DONE`/`FAILED`/`CANCELED` | Persists |
| `objectives` | Annotated operator attrs, policy params, child runtime/objectives mirror (`child:{id}:runtime:{k}`, `child:{id}:obj:{k}`) | Persists |
| `artifacts` | Stable identifiers, file paths | Persists |
| `directives` | Instructions from parent | Cleared each cycle |
| `task_context` | `uid → {name, kwargs}`, plus `action_id` with tracking enabled, for every spawned task | Persists |

Live references (not in `Snapshot`): `task_futures: dict[str, Future]`, `history: deque[Snapshot]` (maxlen=64).

**Optional action links** — `Operator(..., track_action_ids=True)` assigns a fresh
`Action.action_id` to each emitted occurrence before observer notification, using
copies so policy-owned action IDs are unchanged. Context retains the ID for the
Future or local result returned by an `@act`, and for child launches/restarts;
backend subtasks hidden inside an action are outside this scope. A retry's new
task points to the retry action, whose `target_uid` identifies the preceding task.

With tracking enabled, local results and Futures without a backend `id` receive
fresh UUID-based task UIDs; backend-assigned IDs are retained. Subsequent snapshots
expose new context, and checkpoints preserve it. Tracking defaults to `False`,
retaining legacy task context and UID behavior; unset action IDs are omitted from
serialization. Each operator opts in independently. Applications own persistence
and links to domain-specific decisions; observers provide the debugging channel.

**Snapshot** — frozen Pydantic model. Safe to log, serialize, or pass across coroutines.

**Task errors** — when a task fails, `snapshot.observations["error.{uid}"]` contains:
```python
{"task_name": str, "message": str, "type": str}   # type = exception class name
```

Pattern for error-aware `@observe`:
```python
@observe
def extract(self, snapshot) -> dict:
    errors = {
        uid.removeprefix("error."): info
        for uid, info in snapshot.observations.items()
        if uid.startswith("error.")
    }
    return {..., "errors": errors}
```

**Cycle history** — accessible inside `@observe` via `self.cycle_history`. Returns previous
completed snapshots (oldest first, current cycle excluded, up to 64):
```python
@observe
def extract(self, snapshot) -> dict:
    prev    = self.cycle_history           # list of past Snapshots
    radii   = [s.objectives.get("radius") for s in prev[-5:]]
    stale   = len(set(radii)) == 1 and len(radii) == 5
    return {..., "stagnating": stale}
```

**Result & rejection views** — `snapshot.results` returns this cycle's task-result payloads
(`result.{uid}`, in completion-drain order — not a semantic "latest"); `snapshot.rejections` returns
this cycle's boundary rejections (`rejected_param.{name}`). Both are read-only conveniences over
`observations`. See §3.9.

**Stop conditions** — three, all on `Operator.__init__` and all independent of goal state:
```python
op = SearchOperator(engine, max_cycles=500, max_runtime=3600,
                    budget=[Budget("trials", 12)])   # cycles · wall-clock · resource/deadline
```

---

### 3.8 Messaging

All inter-component communication enters through `inbox: asyncio.Queue`.

| MessageKind | Direction | Payload |
|-------------|-----------|---------|
| `TASK_EVENT` | AsyncFlow/local → Operator | `{uid, state, result/error}` |
| `SNAPSHOT` | Child → Parent | child's `Snapshot` |
| `DIRECTIVE` | Parent → Child | `{directives: dict}` |
| `CUSTOM` | Parent → Child (`notify`), MCP → Operator | arbitrary; `{"mutation": Action \| Decision}` is applied |
| `STOP` | System → Operator | `None` |

**Async tasks:** `future.add_done_callback(_deliver_task_event)` — synchronous, zero watcher coroutines, O(1) registration.

**Local acts:** result delivered via `inbox.put_nowait(TASK_EVENT{DONE, result})` — same 1-cycle latency as async tasks, same visibility in next cycle's `@observe`.

**External steering:** a `CUSTOM` message whose payload is `{"mutation": Action | Decision}` (what `ADLMCPServer` sends) is buffered at drain time. Its actions are validated one by one (an invalid one is logged and dropped, never aborting the run) and appended after the policy's actions in that cycle's `Decision`, so they pass the same `Tunable` boundary, action-ID assignment and `on_cycle` notification. They are applied after the policy's batch, one by one; one that fails (e.g. its `@act` raises) is logged and dropped, never raised. They apply on cadence-skipped cycles too. A mutation with `stop=True` sets the stop event, like `STOP`. Any other `CUSTOM` payload is not read by ADR.

---

### 3.9 Adaptation Boundary, Budgets & Evidence Helpers

All additive and opt-in — an operator that declares none of these behaves exactly as before. The
value types live in `radical.adr.boundary` (`Tunable`, `Budget`), exported from the package root.

**`Tunable(default, low, high, *, int=False, step=None)`** — a bounded parameter the agent may adapt.
Declared as a class attribute:

```python
class Run(Operator):
    learning_rate = Tunable(3e-4, 1e-5, 1e-3)
    grad_steps    = Tunable(25_000, 2_000, 100_000, int=True)
    discount      = 0.99          # plain attr → frozen; the agent cannot write it
```

One declaration is the single source of truth. `__init_subclass__` collects Tunables into
`_adl_tunables` and adds their names to the objective proxy; ADR seeds `state.objectives` with the
scalar `default`. At the `UPDATE_PARAM` apply seam **only**, every agent write is **validated** against
the bounds. An out-of-bounds / non-`int` / off-`step` value — or a write to a non-`Tunable` key once
any boundary is declared — is **rejected, never silently clamped or dropped**: the requested value is
never altered, and the refusal is recorded as evidence (`observations["rejected_param.{name}"]`,
surfaced via `self.rejections` / `Snapshot.rejections`) so the agent re-decides next cycle. Rejection
is soft — it never raises or aborts the cycle (unlike `_validate_actions`). Operator-internal writes
(`self.lr = v`, `@observe` folds, construction config) bypass the boundary; only agent `UPDATE_PARAM`
is gated. `self.tunables` returns `{name: value}`, and `get_schema(...)` renders the bounds into the
`update_param` tool description so the agent sees the same boundary that validates it.

**`Budget(metric, limit, direction="max")`** — a resource/deadline stop, passed as
`Operator(..., budget=[Budget("trials", 12)])`. Checked in the run loop beside `max_runtime`
(`_check_budgets`), reading the metric from `obs` then `snapshot.objectives`; on breach it sets the
stop event. It is **not** a `Goal` and is evaluated **outside** `_check_goals`, so no goal composition
— including an unsatisfiable runtime goal an agent adds — can block termination. A `Budget` guarantees
the run *stops* when the resource is exhausted; it does not change what "converged" means.

**`decide_when`** (class attribute; `None` = every cycle) — a cadence gate applied between OBSERVE and
DECIDE (`_should_decide`). `"idle"` consults the policy only when `inflight == 0` (captures bootstrap
and post-result, skips mid-task cycles); `"new_result"` only when `snapshot.results` is non-empty. On
a skipped cycle ADR applies an empty `Decision()`, but `@observe`, goal checks, and budgets still run.
It gates *when the implicit loop consults the policy* — it is not a second loop (no `@loop`).

**Evidence helpers** (read-only):

| Helper | Returns | Notes |
|--------|---------|-------|
| `Snapshot.results` | `list` | Task-result payloads recorded this cycle (completion-drain order; no "latest" guarantee). |
| `Snapshot.rejections` | `list` | Boundary rejections surfaced this cycle. |
| `self.trend(metric, direction, *, eps=1e-9)` | `"improving"/"flat"/"regressing"` | `direction` (`"maximize"`/`"minimize"`) is **required** — numeric change ≠ scientific improvement, and it is not inferred from goals (auxiliary metrics need not have one). Compares the two most recent **distinct** values (consecutive duplicates collapsed, since a trial spans many idle cycles). |
| `self.stalled(metric, direction, n, *, eps=1e-9)` | `bool` | No improvement across the last `n` distinct values. |
| `self.tunables` | `dict` | Current `{name: value}` of the declared boundary. |
| `self.rejections` | `list` | Rejections pending until the next decision consumes them. |

**Static-goal shadow guard** — a runtime goal proposed via `Decision(goals=[...])` whose `name` or
`metric` collides with a static `@goals` goal is rejected (dropped + logged). The agent may add
stricter goals on *other* metrics but can never shadow or relax the authoritative objective. This is
collision prevention, not goal provenance — there is no `Goal(source=…)` field or subtype; the
authoritative-vs-agent distinction stays *derived* from `@goals` vs `runtime_goals`.

---

## 4. Runtime Internals

### 4.1 Act Execution Path

```
policy decides → Action(SPAWN_TASK, "dock", kwargs) → _apply_one
                                                           │
                 fn = _adl_act_registry["dock"]         │
                 dep_futures = resolve(act.depends_on)     │
                 result = await fn(self, *dep_futures, **kwargs)
                                                           │
                     ┌─────────────────────────────────────┤
                     │ Future         │ plain value        │
                     ▼                ▼                    │
              track future      inbox.put_nowait           │
              add_done_callback  (TASK_EVENT{DONE})        │
```

The `@act` method decides execution strategy. `_apply_one` is execution-strategy-agnostic — it handles the return.

### 4.2 Inbox Drain

`_drain_inbox` processes up to 64 messages per cycle. First message: blocked wait up to 0.05s. Subsequent: `get_nowait()`. After draining: `_sync_running_states()` polls `future.state` for PENDING→RUNNING promotions.

`_apply_message` merges incoming data:
- `TASK_EVENT` → `state.runtime[uid]`
  - DONE: `state.observations["result.{uid}"] = result`
  - FAILED: `state.observations["error.{uid}"] = {task_name, message, type}` — structured error visible to `@observe`
- `DIRECTIVE` → `state.directives`
- `SNAPSHOT` (child) → three key families. A child's *observations* are per-cycle, so they mirror into the parent's per-cycle `observations`; its *runtime* and *objectives* are persistent child state, so they mirror into the parent's **persistent `objectives`** store (latest-wins) — keeping the latest child metrics available across parent cycles, even when a child's final SNAPSHOT and its block-Future `DONE` event land on different parent cycles:
  - `state.observations["child:{id}:{k}"]` — child's task results (per-cycle)
  - `state.objectives["child:{id}:runtime:{k}"]` — child's task statuses (persistent)
  - `state.objectives["child:{id}:obj:{k}"]` — child's objectives (radius, center, etc.; persistent)

  These keys are read for you by `ChildHandle` (`obs["explorer"].radius`) — user code rarely touches them directly.

### 4.3 Goal Checking

```python
all_goals = static_goals + runtime_goals
if all_goals and all(g.satisfied(obs, snapshot) for g in all_goals):
    decision.stop = True
```

Fires after DECIDE, before ACT. The ACT phase still runs in the stop cycle — any actions the policy returned are applied before the operator exits.

---

## 5. Multi-Operator Patterns

### 5.1 Independent Operators — different goals

Fully isolated operators run concurrently via `asyncio.gather`. Each has its own state, policy, tasks, goals. See `examples/06-multi-operator.py`.

### 5.2 Concurrent Same-Goal Operators

First operator to satisfy the goal cancels the others via `add_done_callback` on the wrapping asyncio.Task:

```python
async def run_until_first_satisfied(*operators):
    tasks = [asyncio.create_task(_consume(op)) for op in operators]
    def _on_any_done(t):
        for other in tasks:
            if not other.done(): other.cancel()
    for t in tasks: t.add_done_callback(_on_any_done)
    await asyncio.gather(*tasks, return_exceptions=True)
```

See `examples/07-multi-operator-same-goal.py`.

### 5.3 Parent-Child Hierarchy — dynamic DAG

The parent drives child lifecycle entirely from its observe/decide/act loop. Child operators are **first-class**: declared with `add_child`, launched with `spawn_operator`, and observed through a live `ChildHandle` — `Future`/`UID`/`artifacts` and `child:*` string keys never appear in user code.

```python
class CampaignOperator(Operator):

    def __init__(self, engine, operator_id=None, **kwargs):
        super().__init__(engine, operator_id, **kwargs)
        # add_child(name, OperatorCls, policy=…, **config) — registers a child,
        # auto-wires parent=self, attaches a policy (instance OR class/factory
        # called with the child), and seeds config as the child's objectives.
        self.add_child("explorer", SearchOperator, policy=SearchPolicy)

    @observe
    def watch(self, snapshot) -> dict:
        # self.child(name) → a live ChildHandle (read-only view of the child).
        return {"explorer": self.child("explorer"), "cycle": snapshot.cycle}
```

**Parent observes children** through the `ChildHandle` returned by `self.child(name)`:
- `obs["explorer"].status` — `NOT_STARTED` / `PENDING` / `RUNNING` / `DONE` / `FAILED` / `CANCELED`
- `obs["explorer"].started` — `True` once ever launched (incl. terminal states); the status gate for spawn-once
- `obs["explorer"].done` / `.failed` — convenience booleans
- `obs["explorer"].radius` — any reported objective by name (raises `AttributeError` if the child never reported it; `.get("radius", default)` for a soft read)

**Launch and sequencing** are expressed with `spawn_operator` in the policy — `depends_on` lists upstream child **names** (not uids):
```python
# launch the declared explorer
Decision(actions=[acts.spawn_operator("explorer")])

# gate a child on another child by name — AsyncFlow resolves the upstream
# block Future before starting this child's block; no external scaffolding:
Decision(actions=[acts.spawn_operator("merge", MergeOp, depends_on=["explorer"])])
```

#### Adaptive fan-out

Because names, count, and config are all runtime values, "explorer finds a region → spawn N refiners around it" is expressible in one list-comprehension — no escape hatch, no `start()`/`fut.id` plumbing:

```python
class CampaignPolicy(Policy):
    K = 3

    @decide
    async def run(self, obs):
        if not obs["explorer"].started:                      # runtime-owned spawn-once gate
            return Decision(actions=[self._act.spawn_operator("explorer")])

        if obs["explorer"].done and not obs["refiner_0"].started:
            best = obs["explorer"].best_x
            return Decision(actions=[
                self._act.spawn_operator(f"refiner_{i}", SearchOperator,
                                         policy=SearchPolicy, depends_on="explorer",
                                         center=best + (i - 1) * 0.5)
                for i in range(self.K)
            ])

        if obs["refiner_0"].started and all(obs[f"refiner_{i}"].done for i in range(self.K)):
            return Decision(stop=True)
        return Decision()
```

> **Spawn-once is runtime-owned, not a user flag.** `spawn_operator` is idempotent: once a
> child has ever been started (`name in _child_starts`, *including* terminal `DONE`/`FAILED`/
> `CANCELED`), re-emitting it is a no-op + a logged warning. Well-written policies gate on
> `child.started` (above), so they never hit the warning path — it is a safety net for a genuine
> double-spawn, replacing the silent "N overlapping runs" bug an unguarded re-emit used to cause.
>
> **`restart_operator(name)`** re-runs a child: it cancels any in-flight run, resets the child to
> its **initial** declared config (the kwargs captured by `add_child` in `_child_config`), and
> relaunches. **`cancel_operator(name)`** cancels the child's block Future, leaving a `CANCELED`
> tombstone — the name stays known, so a later `spawn_operator` is still a no-op; re-running after
> a cancel is `restart_operator`'s job.

**Parent steers children** via `Decision(delegate_to=["explorer"], directives={...})`.

See `examples/08-hierarchical-operator.py`.

---

## 6. Observer and Replay

`ObserverBase` defines three synchronous lifecycle hooks. `on_cycle` receives both the
`Snapshot` and the `obs` dict the policy actually reasoned over that cycle, plus the resulting
`Decision`:

```python
class ObserverBase(Protocol):
    def on_start(self, operator_id: str, metadata: dict) -> None: ...
    def on_cycle(self, snapshot: Snapshot, obs: dict, decision: Decision) -> None: ...
    def on_stop(self, final: Snapshot | None, reason: str) -> None: ...
```

`JsonlObserver` emits one JSON line per event:

| Event | When |
|-------|------|
| `operator_start` | Before first cycle |
| `cycle_start` / `decision` | Each cycle |
| `task_spawn` / `task_complete` / `task_failed` / `task_cancelled` | Task lifecycle |
| `goal_add` / `goal_update` / `goal_remove` / `goals_satisfied` | Goal changes |
| `operator_stop` | After final cycle |

HTML replay dashboards: `demo-visualization.html` (force-directed) and `demo-visualization-timeline.html` (Gantt).

### Record / replay for offline policy evaluation

`RecordingObserver` captures the raw per-cycle `obs` dict to a JSONL trace during a live run.
`replay_trace` then re-drives a *different* policy against that exact recorded `obs` sequence —
fully offline (no engine, no Operator, no task dispatch) — so policies can be A/B compared on
identical conditions without re-running the campaign.

```python
from radical.adr import RecordingObserver, replay_trace

# 1. live run — record the observation trace
op = MyOperator(engine, observer=RecordingObserver("trace.jsonl"))
await op.run_to_completion()

# 2. offline — replay against a different policy
decisions = await replay_trace("trace.jsonl", CandidatePolicy())
```

`replay_trace` reads only the recorded `obs` (the recorded `decision` is for inspection).
See `examples/11-record-replay.py`.

---

## 7. AsyncFlow Integration

### Engine construction

```python
backend = await LocalExecutionBackend(ThreadPoolExecutor(max_workers=8))
engine  = await WorkflowEngine.create(backend)
op      = MyOperator(engine)
```

### `@act` and `engine.block`

ADR does **not** pre-register `@act` methods with AsyncFlow. Each `@act` method calls `self.engine.function_task(fn)(...)` or `self.engine.block(fn)(...)` at invocation time — each call creates a fresh component with a new UID, which is correct by AsyncFlow's design.

Module-level functions passed to `engine.function_task` must be importable (picklable) when using `ProcessPoolExecutor`. When using `ThreadPoolExecutor` or running locally, any async function works.

### `start()` and operator blocks

`op.start(*dep_futures)` wraps `run_to_completion()` as an `engine.block` (lazy, cached on first call). The returned block Future integrates the operator into AsyncFlow's native dependency graph. This is an **internal mechanic** — `SPAWN_OPERATOR` calls it for you (resolving `depends_on` child names to their start-Futures). User code launches children with `acts.spawn_operator(...)`, not `start()` directly.

### Feature branch dependency

Requires `feature/enable_block_cancellation` in AsyncFlow:

| Feature | ADR usage |
|---------|-----------|
| `future.state` attribute | `_sync_running_states` polls for RUNNING transitions |
| Block cancellation propagation | Cancelling an operator's block also cancels its spawned tasks |
| `workflow_id` kwarg | Task attribution (passed by `@act` methods, not `_apply_one`) |
| `patched_cancel` fix | `future.cancel()` returns bool, not method reference |
| Idempotent `handle_task_cancellation` | Safe repeated cancellation |

---

## 8. Examples

| File | Demonstrates |
|------|-------------|
| `01-asyncflow-adaptive-search.py` | Minimal operator: `@act` local, `@observe`, `@goals`, single run |
| `04-protein-design.py` | Static + runtime goals; LLM-steered variant |
| `05-spawn-block.py` | `@act` returning block Future — sequential sub-task orchestration |
| `06-multi-operator.py` | N independent operators, different goals (Req 1) |
| `07-multi-operator-same-goal.py` | `run_until_first_satisfied` — concurrent same goal (Req 2) |
| `08-hierarchical-operator.py` | Adaptive fan-out: `add_child` / `spawn_operator` / `ChildHandle` — explorer → N refiners `depends_on` it (Req 3) |
| `demo/demo_protein_design.py` | 30-candidate campaign: 4-phase goals, `@act` dispatching to `function_task` and `block` |
| `demo/demo_protein_design_llm.py` | Same campaign steered by LLM via OpenRouter |

---

## 9. Design Invariants

**`@observe` is mandatory.** `run()` raises `TypeError` before the first cycle if undefined. No default observation exists — the operator is domain-specific by definition.

**`@decide` is mandatory.** Empty `Policy` raises `PolicyError` on first call — no silent empty-`Decision` fallthrough.

**`@act` runs in the event loop.** The method body runs in the operator's asyncio event loop with `self` available. It is responsible for deciding execution strategy: local computation, `engine.function_task`, or `engine.block`. The operator never pre-registers tasks with AsyncFlow; registration happens at invocation time.

**`_apply_one` is execution-agnostic.** It calls the `@act` method, inspects the return, and routes accordingly — Future or plain value. A single code path handles all execution modes.

**Inbox is the only input channel.** All inputs — task completions (done-callback → `put_nowait`), local results (`put_nowait`), child snapshots, parent directives — arrive through `inbox` and are processed in `_drain_inbox`. No out-of-band state mutations.

**State proxying is transparent.** `self.radius = 2.5` writes to `state.objectives["radius"]`. Visible in `snapshot.objectives`, readable in `@observe`. No serialization needed.

**Goals check before act.** `_check_goals` runs after DECIDE but before ACT. The stop cycle's actions still apply.

**SNAPSHOT propagates all child state.** When a child sends a SNAPSHOT to its parent, the parent merges child observations (`child:{id}:{k}`, per-cycle), runtime statuses (`child:{id}:runtime:{k}`, persistent), and objectives (`child:{id}:obj:{k}`, persistent) — giving the parent full visibility into each child's state. Persistent families survive across parent cycles (latest-wins) so the latest child metrics stay readable even when a child's final SNAPSHOT and its block-Future `DONE` event land on different cycles.

**Child operators are first-class.** A parent declares children with `add_child(name, OperatorCls, policy=…, **config)`, launches them with the `SPAWN_OPERATOR` action (`acts.spawn_operator(name, …, depends_on=[child_names])`), and observes them through `self.child(name)` → a read-only `ChildHandle` (`.status`/`.started`/`.done`/`.failed` and metric-by-name). `Future`, `UID`, `artifacts`, and `child:*` string keys are internal — they never appear in policy or operator code. Names, count, config, and dependencies are all runtime values, so adaptive fan-out needs no escape hatch. (`policy=` may be a `Policy` instance or a class/factory called with the child.)

**Child lifecycle is runtime-owned, declarative.** `spawn_operator` is idempotent ("ensure launched"): a re-emit on an already-started child (`name in _child_starts`, incl. terminal states) is a no-op + a logged warning — there is no user-managed spawn-once flag. `restart_operator(name)` re-runs a child from its **initial** config (`_child_config`); `cancel_operator(name)` cancels its block Future, leaving a `CANCELED` tombstone. All three are *steering* verbs — they delegate gating to `child.start(*futs)` / `future.cancel()` and never schedule, resolve dependencies, or block.

**The adaptation boundary rejects, it never silently clamps.** With `Tunable`s declared, an
out-of-bounds agent `update_param` (or a write to a non-`Tunable` key) is refused and surfaced as
evidence (`rejected_param.*`) — the requested value is never altered and the cycle never aborts. The
agent runs the experiment it asked for, or learns why it could not.

**Budgets stop independently of goals.** `Budget` is a stop condition, not a `Goal`; it is checked
outside `_check_goals`, so no goal composition can keep a run alive past its resource/deadline cap. It
guarantees termination — it does not change runtime-goal composition semantics.

**Cadence gates consultation, not the loop.** `decide_when` only changes *when the policy is
consulted*; ADR still owns the single implicit loop, and `@observe`/goals/budgets run every cycle.

**Trend is direction-aware.** `trend`/`stalled` require an explicit `direction` and operate on the
*distinct* value series (idle-cycle repeats collapsed); ADR never guesses whether up or down is
"better", and never infers it from a goal.

**Runtime goals cannot shadow static ones.** A proposed runtime goal colliding with a static `@goals`
goal by `name` or `metric` is rejected, keeping the authoritative objective outside the agent's reach.
