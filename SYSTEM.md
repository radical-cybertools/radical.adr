# ADL System Overview

Technical reference for the Agentic Adaptive Decision Layer. For motivation and scientific context, see [ADL.md](ADL.md).

---

## 1. Stack Position

```
┌─────────────────────────────────────────────────────┐
│  Domain Science                                      │
│  IMPRESS · ROSE · Campaign Manager · custom scripts │
└───────────────────────┬─────────────────────────────┘
                        │  ADL boundary (this layer)
┌───────────────────────▼─────────────────────────────┐
│  radical.adl                                         │
│  Operator · Policy · Goals · Decision · ActionSet     │
└───────────────────────┬─────────────────────────────┘
                        │  AsyncFlow API
┌───────────────────────▼─────────────────────────────┐
│  radical.asyncflow                                   │
│  WorkflowEngine · function_task · engine.block      │
└───────────────────────┬─────────────────────────────┘
                        │  execution backend
┌───────────────────────▼─────────────────────────────┐
│  RADICAL-Pilot · LocalExecutionBackend · HPC cluster│
└─────────────────────────────────────────────────────┘
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

**Operator lifecycle tool** (starts a child operator as a block, enables parent-driven DAG):
```python
@act
async def launch_child(self, dep_future=None) -> asyncio.Future:
    fut = self.child_op.start(dep_future) if dep_future else self.child_op.start()
    self.state.artifacts["child_uid"] = str(getattr(fut, "id", id(fut)))
    return fut
```

Module-level implementations (`_dock_impl`, `_prep`, etc.) are plain async functions — pure, testable, independent of ADL. The `@act` method is the dispatch point; the implementation is separate.

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

`@decide` is mandatory — an empty `Policy` with no composition raises `PolicyError` on first call. Composition: `Policy(primary=LLMPolicy(), fallback=RulePolicy())`.

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
| `CANCEL_TASK` | `future.cancel()` on tracked future |
| `UPDATE_PARAM` | `state.objectives[key] = value` |

`depends_on: list[str]` on every `Action` — UIDs of futures in `state.task_futures`. Resolved to live futures before calling the `@act` method, enabling native intra-operator DAG chaining.

---

### 3.6 ActionSet

```python
act = op.get_actions()

tools.evaluate(x=1.0)                           # @act shorthand → SPAWN_TASK
tools.spawn_task("evaluate", x=1.0)             # explicit → SPAWN_TASK
tools.spawn_workflow("md_refine", candidate=c)  # alias → SPAWN_TASK
tools.cancel_task(uid)                          # CANCEL_TASK
tools.update_param("radius", 2.5)              # UPDATE_PARAM
tools.stop()                                   # Decision(stop=True)

# depends_on — resolved to dep_futures before @act is called:
tools.launch_child_b(depends_on=[child_a_uid])
```

`depends_on` is a first-class parameter on all spawning methods — it is extracted from kwargs and set on `act.depends_on`, not passed as a task kwarg.

LLM schema: `op.get_actions("openai")` / `"anthropic"` / `"langgraph"` — renders all `@act` signatures and docstrings in vendor format.

---

### 3.7 State Model

| Namespace | Contents | Lifetime |
|-----------|----------|----------|
| `observations` | Task results (`result.{uid}`), child data, telemetry | Cleared each cycle |
| `runtime` | UID → `PENDING`/`RUNNING`/`DONE`/`FAILED`/`CANCELED` | Persists |
| `objectives` | Annotated operator attrs, policy params | Persists |
| `artifacts` | Stable identifiers, file paths | Persists |
| `directives` | Instructions from parent | Cleared each cycle |

Live references (not in `Snapshot`): `task_futures: dict[str, Future]`, `history: deque[Snapshot]` (maxlen=64).

**Snapshot** — frozen Pydantic model. Safe to log, serialize, or pass across coroutines.

---

### 3.8 Messaging

All inter-component communication enters through `inbox: asyncio.Queue`.

| MessageKind | Direction | Payload |
|-------------|-----------|---------|
| `TASK_EVENT` | AsyncFlow/local → Operator | `{uid, state, result/error}` |
| `SNAPSHOT` | Child → Parent | child's `Snapshot` |
| `DIRECTIVE` | Parent → Child | `{directives: dict}` |
| `CUSTOM` | Parent → Child | arbitrary |
| `STOP` | System → Operator | `None` |

**Async tasks:** `future.add_done_callback(_deliver_task_event)` — synchronous, zero watcher coroutines, O(1) registration.

**Local tools:** result delivered via `inbox.put_nowait(TASK_EVENT{DONE, result})` — same 1-cycle latency as async tasks, same visibility in next cycle's `@observe`.

---

## 4. Runtime Internals

### 4.1 Tool Execution Path

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
- `TASK_EVENT` → `state.runtime[uid]`, `state.observations["result.{uid}"]`
- `DIRECTIVE` → `state.directives`
- `SNAPSHOT` (child) → three key families:
  - `state.observations["child:{id}:{k}"]` — child's task results
  - `state.observations["child:{id}:runtime:{k}"]` — child's task statuses
  - `state.observations["child:{id}:obj:{k}"]` — child's objectives (radius, center, etc.)

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

The parent drives child lifecycle entirely from its observe/decide/act loop. Children are created with `parent=self` so `_dispatch` sends SNAPSHOT messages upward each cycle.

```python
class ParentOperator(Operator):

    def __init__(self, engine, operator_id=None, **kwargs):
        super().__init__(engine, operator_id, **kwargs)
        self.child_a = SearchOperator(engine, "child-a", parent=self)
        self.child_b = SearchOperator(engine, "child-b", parent=self)
        self.child_a.policy = SearchPolicy(self.child_a)
        self.child_b.policy = SearchPolicy(self.child_b)

    @act
    async def launch_child_a(self) -> asyncio.Future:
        fut = self.child_a.start()
        self.state.artifacts["child_start_uid.child-a"] = str(getattr(fut, "id", id(fut)))
        return fut

    @act
    async def launch_child_b(self, dep_future) -> asyncio.Future:
        fut = self.child_b.start(dep_future)   # B starts when A is DONE
        self.state.artifacts["child_start_uid.child-b"] = str(getattr(fut, "id", id(fut)))
        return fut
```

**Parent observes children** each cycle via SNAPSHOT:
- `obs["child:child-a:obj:radius"]` — child's objectives
- `obs["child:child-a:result.uid"]` — child's task results
- `obs["child:child-a:runtime.uid"]` — child's task statuses

**Sequencing** is expressed via `depends_on` in the policy:
```python
Decision(actions=[tools.launch_child_b(depends_on=[obs["child_a_uid"]])])
```
AsyncFlow resolves child-a's block Future before starting child-b's block — no external scaffolding.

**Parent steers children** via `Decision(delegate_to=["child-a"], directives={...})`.

See `examples/08-hierarchical-operator.py`.

---

## 6. Observer and Replay

`ObserverBase` defines `on_start`, `on_cycle`, `on_stop`. `JsonlObserver` emits one JSON line per event:

| Event | When |
|-------|------|
| `operator_start` | Before first cycle |
| `cycle_start` / `decision` | Each cycle |
| `task_spawn` / `task_complete` / `task_failed` / `task_cancelled` | Task lifecycle |
| `goal_add` / `goal_update` / `goal_remove` / `goals_satisfied` | Goal changes |
| `operator_stop` | After final cycle |

HTML replay dashboards: `demo-visualization.html` (force-directed) and `demo-visualization-timeline.html` (Gantt).

---

## 7. AsyncFlow Integration

### Engine construction

```python
backend = await LocalExecutionBackend(ThreadPoolExecutor(max_workers=8))
engine  = await WorkflowEngine.create(backend)
op      = MyOperator(engine)
```

### `@act` and `engine.block`

ADL does **not** pre-register `@act` methods with AsyncFlow. Each `@act` method calls `self.engine.function_task(fn)(...)` or `self.engine.block(fn)(...)` at invocation time — each call creates a fresh component with a new UID, which is correct by AsyncFlow's design.

Module-level functions passed to `engine.function_task` must be importable (picklable) when using `ProcessPoolExecutor`. When using `ThreadPoolExecutor` or running locally, any async function works.

### `start()` and operator blocks

`op.start(*dep_futures)` wraps `run_to_completion()` as an `engine.block` (lazy, cached on first call). The returned block Future integrates the operator into AsyncFlow's native dependency graph — it can be passed as `dep_future` to another `start()` call, producing sequential operator DAGs without external sequencing code.

### Feature branch dependency

Requires `feature/enable_block_cancellation` in AsyncFlow:

| Feature | ADL usage |
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
| `08-hierarchical-operator.py` | Parent-child sequential DAG via `start()` + `@act` + `depends_on` (Req 3) |
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

**SNAPSHOT propagates all child state.** When a child sends a SNAPSHOT to its parent, the parent merges child observations (`child:{id}:{k}`), runtime statuses (`child:{id}:runtime:{k}`), and objectives (`child:{id}:obj:{k}`) — giving the parent full visibility into each child's state each cycle.
