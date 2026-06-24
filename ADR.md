# Autonomous Decision Runtime (ADR)

## Why ADR Exists

Scientific computing is no longer about running a fixed set of simulations and collecting results. Modern HPC scientific workflows are iterative: submit experiments to a cluster, observe partial results, decide what to run next, and repeat until something converges. This pattern underlies every serious computational science application — drug discovery, materials design, protein engineering, climate modeling, active learning.

The loop that governs this process — **observe what came back, decide what to do next, change the running computation** — has been implemented ad-hoc inside every application. Systems like IMPRESS, ROSE, and Campaign Manager each solve this problem from scratch, in incompatible ways, every time. The execution layer beneath them (RADICAL-Pilot, AsyncFlow) is standardized. The adaptive steering layer above is not.

**ADR does more than standardize the adaptive steering loop.** It provides the full runtime infrastructure for autonomous HPC scientific workflows: a principled Observe → Decide → Act cycle, first-class goals that can be set, tracked, and dynamically expanded at runtime, and explicit awareness of the computational environment — because in HPC, what the science needs and what the infrastructure allows are both convergence criteria.

---

## The Architectural Gap

Modern scientific computing stacks look like this:

```
┌─────────────────────────────────────────────────────────────┐
│  Domain Science  (IMPRESS · ROSE · Campaign Manager · ...)  │
│                                                             │
│    ┌─────────────────────────────────────────┐             │
│    │  ADR  ←  this component is always ad-hoc│             │
│    │  Observe → Decide → Act                 │             │
│    └─────────────────┬───────────────────────┘             │
└──────────────────────│──────────────────────────────────────┘
                       │  standardized
              [AsyncFlow — HPC Execution]
                       │  standardized
              [RADICAL-Pilot · HPC Backends]
```

The execution layer is well-studied and standardized. The domain science layer is application-specific and always will be. The adaptive steering layer — the loop that connects science to computation — is the gap. It is written differently in every project, tested differently in every project, and cannot be reused, composed, or extended across projects.

ADR fills that gap with a single abstraction: the **Operator**, which runs a principled Observe → Decide → Act cycle over long-running HPC tasks, tracks scientific and computational goals as first-class runtime objects, and stops autonomously when both the science is done and the resource constraints are satisfied.

---

## What the Loop Actually Requires

The adaptive steering loop in scientific computing has specific requirements that distinguish it from general agent frameworks:

**Observations are sparse.** HPC tasks take minutes to hours. A steering decision is made once per cycle, not thousands of times per second. The framework must be designed for this cadence.

**Tasks are distributed.** Work runs on remote compute resources managed by RADICAL-Pilot or equivalent. The steering runtime never owns execution — it only decides what to submit and when.

**Convergence is a scientific criterion.** "Stop when band gap exceeds 1.5 eV" or "stop when model uncertainty drops below 5%" are scientific hypotheses, not program counters. They must be first-class, not embedded as magic constants in loop conditions.

**Convergence is also a computational criterion.** In an HPC environment, resource availability is a constraint that evolves at runtime. "Throttle spawning when GPU utilization exceeds 90%" or "pause when queue depth exceeds capacity" are as real as any scientific stopping condition. The steering runtime must be able to express, evaluate, and act on both kinds of criteria simultaneously.

**The policy — what to compute next — must be replaceable.** An HPC scientific workflow should run identically whether steered by domain rules, a machine learning model, or a large language model. Switching the decision logic must not require rewriting the workflow.

No existing framework was designed with all five of these requirements in mind.

---

## What ADR Adds

### Formal Separation of Concerns

ADR enforces a strict three-step cycle each iteration:

1. **Observe** — extract domain metrics from task results (MSE, docking scores, uncertainties, resource utilization)
2. **Decide** — a pluggable policy maps the observation to a structured decision
3. **Act** — the decision is applied: new tasks are spawned, running tasks are cancelled, parameters are updated

This separation is not obvious. Most systems conflate observation with decision logic, or decision logic with execution management. ADR makes each step an explicit, independently testable, independently replaceable unit.

### LLM as a Scientific Steering Policy

ADR introduces a new use case for large language models: **scientific decision-maker**. Not LLM as an orchestrator (the LangGraph model), and not LLM as a distributed actor in a multi-agent system (the Academy model) — LLM as a reasoning engine that answers the question:

> *Given these simulation results, which experiments should I run next?*

This is a fundamentally different use of language models. The LLM sees domain observations — MSE values, docking scores, uncertainty estimates — and produces a structured decision about which HPC tasks to spawn next. A researcher can connect a language model to an HPC scientific workflow with a single policy class. ADR provides the infrastructure that makes this safe, reproducible, and composable.

### ADR and LangGraph Compose — They Do Not Compete

LangGraph and ADR solve different problems and fit naturally together.

**LangGraph** is a reasoning engine. It defines how a language model thinks: which tools it calls, how it chains reasoning steps, how it produces structured decisions. LangGraph is excellent at this and should be used for it.

**ADR** is a workflow execution and governance runtime. It defines what computation runs on HPC infrastructure, how results flow back to the policy, when the workflow stops, and how the operator recovers from infrastructure failures. ADR knows nothing about how the policy reasons — only that it returns a structured decision each cycle.

These two concerns are orthogonal. **A LangGraph graph is a valid ADR policy.** The graph contributes reasoning and structured output. ADR contributes execution dispatch, goal evaluation, autonomous stopping, and failure recovery. Neither needs to know what the other is doing. Neither replaces the other.

This means a researcher can use LangGraph to express the scientific decision-making logic — what to run next, how to tighten the search, what new criteria to propose — while relying on ADR to ensure the workflow executes correctly on HPC infrastructure, terminates autonomously when the science is done, and does not require human supervision.

See `examples/09-langgraph-policy.py` for a complete working integration: a LangGraph agent steers an adaptive search and proposes new goals at runtime, while ADR handles execution, convergence checking, and autonomous stopping.

### Convergence Goals as a First-Class Abstraction

ADR introduces **Goals** as a first-class runtime concept. A goal is a named, evaluable convergence criterion:

```python
Goal(name="band_gap_target", metric="band_gap",    threshold=1.5,  direction="maximize")
Goal(name="model_converged", metric="model_mse",   threshold=0.005, direction="minimize")
Goal(name="cpu_healthy",     metric="cpu_percent", threshold=90.0,  direction="minimize")
```

Goals are source-agnostic. Whether `cpu_percent` comes from a live telemetry subscription or
`band_gap` from a task result is determined entirely by what `@observe` puts into `obs` — not
by the goal type. There is one goal type.

Goals are evaluated against live observations every cycle. When all goals are satisfied, the
operator stops — not because a counter expired, but because both the science is done and the
computational constraints are met.

This is a qualitatively different model from stop conditions embedded in loop logic. Goals are:
- **Named** — they can be tracked, logged, and reported
- **Composable** — multiple goals must all be satisfied before stopping
- **Agnostic** — metric origin (task result, telemetry, external sensor) is an `@observe` concern
- **First-class** — they are not control flow; they are data

### Runtime Goal Modification

ADR supports **dynamic goal proposals at runtime**. A policy — including an LLM — can propose new goals, update existing ones, or revoke them entirely during execution.

A concrete scenario: an LLM steering a materials screening workflow observes in cycle 3 that candidates with high electrical conductivity are also showing unexpectedly favorable mechanical properties. It proposes a new goal:

```
"I should now also track tensile strength. Stop when it exceeds 400 MPa."
```

This goal is added to the active set. From that cycle forward, the workflow monitors two convergence criteria: the original (conductivity) and the newly discovered one (tensile strength). The workflow restructures itself around what the LLM learned — without a human in the loop, without restarting, without code changes.

This is **online problem reformulation** — the ability to change what "done" means as the workflow runs. No prior HPC workflow system has infrastructure for this.

### Multi-Goal Convergence

ADR evaluates all active goals simultaneously, including those proposed at runtime. The stopping condition is:

```
stop ← all(goal.satisfied(obs) for goal in static_goals + runtime_goals)
```

This supports genuinely multi-objective HPC scientific workflows. A protein design operator can require simultaneously:
- best docking score below −8.0 kcal/mol (scientific goal)
- GPU utilization below 90% (resource goal)
- uncertainty estimate below 0.05 (model confidence goal)

Any combination of goals from any mix of sources (static, LLM-proposed, or runtime-updated) is supported. The operator does not stop until all of them are satisfied.

**Example — hits AND budget AND deadline:**

```python
Goal(name="hits_found",   metric="n_hits",        threshold=5,    direction="maximize")
Goal(name="under_budget", metric="compute_hours", threshold=200,  direction="minimize")
Goal(name="cpu_healthy",  metric="cpu_percent",   threshold=90.0, direction="minimize")
```

All three must be satisfied simultaneously — there is no special multi-objective API, just more entries in the list returned by `@goals`. See `examples/10-resource-aware-search.py` for a working two-goal (scientific + resource) version of this pattern.

### Swappable Policy Without Touching the Workflow

Because the policy is a separate object assigned to the operator, the same HPC scientific workflow can be steered by:

- A deterministic rule policy — fast, reproducible, auditable
- An LLM policy — adaptive, reasoning over natural-language context
- A hybrid — LLM as primary, rules as fallback when the LLM fails
- Multiple policies in composition

Switching requires one assignment. The operator, its acts, its observations, and its goals are untouched.

### N Independent Operators

Ensemble methods, multi-start optimization, and population-based search map directly onto N operators running concurrently via `asyncio.gather`. Each steers its own HPC sub-workflow with its own state, its own policy, and its own goals — no coordination protocol, no shared mutable state.

This matches how computational science actually works: run multiple strategies in parallel, compare results, keep the best.

### Autonomous Operation

An ADR operator is autonomous in a precise, verifiable sense. Six properties hold simultaneously:

**Self-directed stopping.** The operator decides when the workflow is finished. Every cycle, it evaluates all active goals — scientific and computational — against the latest observations. When every goal is satisfied, the operator stops itself — no counter expiry, no human signal, no external monitor required. The stopping condition is a judgment encoded as data, not a magic number embedded in a loop.

**Self-modifying objectives.** The operator can change what it is trying to achieve while running. A policy — including an LLM — can propose new goals, tighten existing thresholds, or remove goals that are no longer relevant. This happens inside the same running loop, without restart, without human intervention. The workflow restructures itself around what it learns.

**Self-scheduling work.** Every cycle, the operator decides what to compute next: which tasks to spawn, which to cancel, which candidates to prioritize, how many parallel workloads to maintain. No scheduler, no queue manager, no human allocation step sits between the policy decision and the execution.

**Self-aware of its own trajectory.** The operator maintains a record of its past cycles. The policy can inspect this history to detect when the workflow has stalled, compute rates of improvement, or switch strategy when convergence slows — without an external monitoring process or logging infrastructure.

**Self-protecting.** The operator can be given a maximum cycle budget. If the workflow has not converged within that budget — because a goal metric was never populated, because the search space is exhausted, or because the problem is genuinely intractable — the operator stops cleanly. This is a structural guarantee, not a responsibility placed on the user.

**Self-diagnostic.** When a task fails, the operator surfaces the failure as a structured observation through the same channel as successful results. The policy can classify the error, decide whether to retry, reduce parallelism, skip the failing region, or escalate — without parsing raw exception strings and without human diagnosis.

These six properties are enforced structurally, not by convention. The policy has no way to stop the operator externally — it returns a `Decision`, and the operator acts on it. Goals are evaluated by the operator, not the policy. Execution is delegated to AsyncFlow; the operator never owns it. The autonomy is built into the architecture, not added as convention.

---

## What an Operator Can Do

The capabilities of an ADR operator fall into three categories: execution, goal management, and coordination.

### Execution

An operator can spawn and cancel any unit of work that AsyncFlow can express:

- **Spawn a task** — dispatch a function to a backend executor (thread pool, process pool, HPC node). The task runs asynchronously; the result arrives in the operator's inbox when complete.
- **Spawn a workflow** — dispatch a multi-step block that orchestrates sub-tasks in sequence or in parallel. The workflow runs as a single schedulable unit with its own lifecycle.
- **Spawn a child operator** — start another full ADR operator as a schedulable block. The child runs its own independent observe/decide/act loop. Its lifecycle — when it starts, what it depends on, when it stops — is controlled by the parent through the same primitives used for tasks and workflows.
- **Cancel any of the above** — cancel any tracked future by UID. Cancellation is synchronous and immediate. When a workflow or operator block is cancelled, all tasks it spawned are cancelled automatically.
- **Execute locally** — run lightweight computation inline in the operator's event loop without involving AsyncFlow. Results are delivered to the inbox with the same 1-cycle latency as remote tasks, keeping the observation model uniform.

### Goal Management

An operator's convergence criteria are not fixed at startup:

- **Propose a new goal** — add an independent convergence requirement at any point during the workflow. Goals can be scientific (domain metric thresholds) or computational (resource utilization bounds).
- **Update an existing goal** — tighten or relax a threshold in-place. The same goal name upserts; no accumulation of stale entries.
- **Remove a goal** — revoke a criterion that has been met or is no longer relevant to the science or the infrastructure.

All goal changes take effect immediately in the same cycle they are issued. The stopping condition is always the conjunction of all currently active goals — static (defined in the class) and runtime (proposed by the policy).

### Coordination

An operator does not act in isolation:

- **Observe child operators** — every child sends its full state (task results, task statuses, operator-level objectives) to the parent's inbox each cycle via a SNAPSHOT message. The parent sees the child's convergence progress, resource usage, and intermediate results without any polling.
- **Steer child operators** — the parent issues directives to children each cycle. A directive is a structured instruction that the child reads from its snapshot and acts upon in its own decide phase.
- **Sequence child operators** — a child can be declared dependent on another child's completion. The second child does not start until the first finishes, expressed as a native AsyncFlow dependency with no external orchestration code.
- **Cancel a subtree** — cancelling a child operator's block cancels all tasks the child has spawned, recursively, through AsyncFlow's block membership tracking.

What an operator cannot do natively: communicate directly with a peer operator at the same level. Two siblings have no shared channel. Coordination between peers requires a common parent to relay messages.

---

## Comparison to Related Systems

| Capability | LangGraph | Academy | ADR |
|---|---|---|---|
| Designed for HPC task execution | ✗ | ✗ | ✓ |
| Task duration | milliseconds | always-on | minutes–hours |
| LLM as scientific steering policy | ✗ | ✗ | ✓ |
| Convergence goals as first-class | ✗ | ✗ | ✓ |
| Runtime goal proposals from policy | ✗ | ✗ | ✓ |
| Multi-goal convergence | ✗ | ✗ | ✓ |
| Self-directed stopping (autonomous) | ✗ | ✗ | ✓ |
| Self-modifying objectives at runtime | ✗ | ✗ | ✓ |
| Self-aware of execution trajectory | ✗ | ✗ | ✓ |
| Self-diagnostic on task failure | ✗ | ✗ | ✓ |
| Spawn child operators with dependencies | ✗ | ✗ | ✓ |
| Policy swappable without loop rewrite | partial | ✗ | ✓ |
| N independent operators | subgraphs | actor peers | ✓ |
| Resource-aware stopping | ✗ | ✗ | ✓ |
| Separation of observe/decide/act | ✗ | ✗ | ✓ (enforced) |
| Usable as an ADR policy | ✓ | ✗ | — |

**LangGraph** is designed for LLM-orchestrated interactive workflows. It has no model for HPC execution, long-running tasks, or convergence goals. It is not a competitor to ADR — it is a reasoning engine that can serve as an ADR policy. LangGraph reasons about what to compute next; ADR executes it on HPC infrastructure, tracks goals, and stops autonomously.

**Academy** provides a model for distributed stateful actors coordinating across federated infrastructure. Its agents are compute nodes; ADR operators are steering controllers. These are different abstractions for different problems at different layers of the stack.

---

## Representative Use Cases

**LLM-steered protein design** — An HPC scientific workflow runs docking simulations on candidate molecules. Each hour, an LLM reviews partial results and decides which candidates to refine further with molecular dynamics. A resource goal automatically throttles spawning when GPU utilization exceeds 90%. The LLM proposes a new goal mid-workflow when it observes an unexpected binding mode.

**Ensemble active learning** — Two Gaussian process learners with different kernels run in parallel, each as an independent operator. An ADR operator observes their MSE each round and allocates more compute budget to whichever is converging faster. Both stop when their shared convergence goal is satisfied.

**Adaptive materials screening** — An operator searches a chemical parameter space. An LLM steering the workflow proposes a new convergence criterion after observing an unexpected electronic property in cycle 4. The workflow continues with both the original goal and the newly discovered one, stopping only when both are satisfied.

**Multi-start optimization** — Four operators start from different initial conditions and search independently. Each has its own policy, its own state, and its own convergence goal. They run concurrently and report their best results at completion.

**Resource-aware HPC workflow** — A large-scale screening workflow monitors live cluster telemetry. When GPU utilization exceeds a threshold, the operator automatically pauses new spawning. When resources free up, spawning resumes. The workflow self-regulates around both the science and the infrastructure — no human intervention required.

---

## Limitations and Known Gaps

The following gaps exist in the current implementation. Each entry names the gap, its practical consequence, and what specifically is blocking resolution.

### G1 — AsyncFlow cancellation feature branch not merged

**Gap:** Block cancellation, `future.state`, `workflow_id` kwarg, and `patched_cancel` live on the `feature/enable_block_cancellation` branch of AsyncFlow. The AsyncFlow main branch does not include these.

**Consequence:** Running ADR with operator-as-block (`start()`), task cancellation, or `RUNNING` state detection requires the feature branch. A team cannot install from main and get full ADR behavior.

**Blocker:** AsyncFlow maintainers must review and merge `feature/enable_block_cancellation`. This is external to ADR. Until merged, teams must explicitly install from the feature branch and accept instability.

---

### G2 — No peer-to-peer operator communication

**Gap:** Two operators at the same level in a hierarchy have no shared channel. All inter-sibling messages must be relayed through a common parent.

**Consequence:** The concurrent multi-operator pattern (multiple operators sharing a joint objective) requires a coordinator parent, which serializes the coordination signal through one event loop. True peer-to-peer decision sharing is not expressible.

**Blocker:** No native AsyncFlow primitive for operator-to-operator message passing exists. Resolution requires either (a) a shared queue primitive in AsyncFlow, or (b) an ADR messaging bus — a shared inbox or broker that multiple operators can write to and read from. Neither is designed yet.

---

### G3 — No test suite ✅ Resolved

**Previously:** No automated tests. All verification was manual via example scripts covering the happy path only.

**Resolution:** Full test suite added across three tiers:
- `tests/unit/` — goal evaluation, edge cases, warn-once behavior
- `tests/integration/` — operator cycle, goal convergence, parent-child SNAPSHOT, task dispatch, max_cycles, cycle_history, task_context, error delivery
- `tests/smoke/` — end-to-end adaptive search

Run: `python -m pytest tests/ -v`

---

### G4 — ResourceGoal has no telemetry pipeline ✅ Resolved

**Previously:** `ResourceGoal` checked only `snapshot.runtime`, which was never populated.
It was effectively inert.

**Resolution:** `ResourceGoal` and `ScientificGoal` are replaced by a single `Goal` type.
`Goal.satisfied()` checks `obs[metric]` (populated by `@observe`) with a fallback to
`snapshot.objectives`. The `@observe` method is the explicit assembly point for all metrics —
task results, telemetry, external sources alike.

**Pattern for resource metrics:**

```python
# 1. Subscribe to RHAPSODY telemetry directly (no ADR classes involved):
_resource: dict = {}
async def on_event(event):
    if getattr(event, "event_type", None) == "ResourceUpdate":
        _resource["cpu_percent"] = event.cpu_percent
telemetry = await engine.start_telemetry(resource_poll_interval=1.0)
telemetry.subscribe(on_event)

# 2. Surface the metric in @observe:
@observe
def extract(self, snapshot) -> dict:
    return {
        "radius":      snapshot.objectives.get("radius", 5.0),
        "cpu_percent": _resource.get("cpu_percent", 0.0),
    }

# 3. Declare the goal — identical to any other goal:
@goals
def criteria(self):
    return [
        Goal("converged",   "radius",      0.005, direction="minimize"),
        Goal("cpu_healthy", "cpu_percent", 90.0,  direction="minimize"),
    ]
```

See `examples/10-resource-aware-search.py` for a complete working example.

---

### G5 — No checkpointing or resumption

**Gap:** All operator state — objectives, task futures, cycle count, observations — is in-memory and ephemeral. If the process dies for any reason (node failure, OOM, timeout), the workflow state is lost.

**Consequence:** ADR workflows are not survivable. This is acceptable for short experiments (minutes to an hour) but unacceptable for realistic HPC scientific workflows that may run for hours or days on shared resources with wall-time limits.

**Blocker:** No persistence layer exists in ADR or AsyncFlow. Resolution requires one of: (a) periodic `Snapshot` serialization to disk, or (b) an AsyncFlow checkpoint-and-restore API. The `Snapshot` dataclass is already serialization-friendly; a `pickle` or JSON dump of `operator.state` is technically straightforward but untested.

---

### G6 — `start()` not validated at scale

**Gap:** The `op.start()` operator-as-block pattern has been tested with one parent and two sequential children. Behavior with deep nesting (3+ levels), wide fan-out (50+ children launched concurrently), or mixed serial/parallel topologies has not been verified.

**Consequence:** A team attempting a large hierarchical workflow may encounter uncharted bugs in AsyncFlow block nesting, SNAPSHOT message floods, or inbox backpressure.

**Blocker:** Requires a dedicated scale test. No known architectural barrier, but performance characteristics are unknown.

---

### G7 — `depends_on` is invisible to LLM schema

**Gap:** The `depends_on` parameter — a list of task UIDs expressing execution dependencies — is supported in `_apply_one` and `ActionSet`, but is not reflected in the schema rendered by `get_schema(format="openai")` or `get_schema(format="anthropic")`. LLM-backed policies using the tool schema cannot learn that dependency chaining is available.

**Consequence:** LLM policies must be manually prompted about `depends_on` if they need to chain tasks. The schema-driven interaction model is incomplete.

**Blocker:** The schema renderer (`_collect_specs`) does not inject `depends_on` into any tool spec. Resolution requires either injecting it as a universal parameter on every spawnable tool or defining a schema-level convention for expressing it.

---

### G8 — Goals are binary; policy has no progress signal

**Gap:** `Goal.satisfied()` returns `True` or `False`. The policy has no way to ask how close a goal is to being satisfied, whether it is improving, or at what rate. To make decisions like "reduce step size as we approach convergence," the policy must re-implement the same threshold arithmetic that `Goal` already encapsulates.

**Consequence:** Adaptive policies routinely duplicate goal logic. Rate-of-improvement decisions (e.g., switch from exploration to exploitation when convergence is 80% complete) require bespoke observation tracking and cannot be expressed against the goal abstraction.

**Blocker:** `Goal` needs a `progress(obs, snapshot) → float` method (0.0 = no progress, 1.0 = satisfied) and a `delta(obs, prev_obs) → float` method. Requires agreed semantics for non-monotonic metrics and defining "progress" for maximize-direction goals.

---

### G9 — No conditional action sequences

**Gap:** A `Decision` holds a flat list of actions that all execute unconditionally in order. The policy cannot express "run task A, then if its result exceeds threshold, run task B, otherwise run task C" — any condition requires waiting for results in the next cycle.

**Consequence:** Simple conditional logic that would be a single if-statement in procedural code costs a full HPC cycle (potentially hours) in ADR. Policies that need tight feedback between tasks must either use the `depends_on` chaining pattern (which only sequences, not conditionally branches) or accept the extra-cycle overhead.

**Blocker:** Conditional branching would require either (a) a new `ConditionalAction` wrapper that binds a predicate to downstream actions, or (b) a richer `@act` orchestration pattern where the method itself contains the condition using `await` on futures inline. Option (b) is already expressible via `engine.block` but is undiscoverable and bypasses the Decision model.

---

## MVP Solidification Checklist

The following items are required to harden the current ADR implementation for handoff to a team testing it on real use cases. Items are ordered by priority.

### P0 — Blocking for any use

These must be resolved before the system is usable at all.

- [ ] **AsyncFlow branch:** Document exactly which AsyncFlow branch to install and pin. Until `feature/enable_block_cancellation` is merged, every install guide must reference the branch explicitly with the exact commit or tag. A team following default install instructions will silently get broken behavior.
- [ ] **Install guide:** Write a step-by-step setup document: virtualenv creation, `pip install -e .`, the correct AsyncFlow branch, Python version requirement, and a smoke test command that confirms the environment is correct.
- [ ] **Working hello-world:** Verify that `examples/01-asyncflow-adaptive-search.py` runs to completion out of the box on a fresh install. This is the first thing a new team will try.

### P1 — Required for realistic use cases

These must be in place for a team to work productively.

- [x] **Test suite:** Unit, integration, and smoke tests covering the core cycle, goal evaluation, task dispatch, error delivery, cycle history, task context, and max_cycles. Run: `python -m pytest tests/ -v`
- [ ] **Capability boundary document:** A concise (one-page) reference that a team reads before writing their first operator. Must cover: the sacred boundary (Operator never owns execution), the four-step cycle, `@act` return semantics (local/task/block), `depends_on` mechanics, SNAPSHOT protocol, and what is *not* supported (peer-to-peer, checkpointing).
- [x] **Error handling guide:** Failed tasks deliver `snapshot.observations["error.{uid}"] = {task_name, message, type}`. Pattern documented in SYSTEM.md §3.7. Policy reads `errors = {k.removeprefix("error."): v for k, v in snapshot.observations.items() if k.startswith("error.")}`.
- [x] **Resource-aware goals:** `ResourceGoal` is replaced by the unified `Goal` type. Resource metrics are surfaced via `@observe` from any external subscription (e.g. RHAPSODY telemetry). See `examples/10-resource-aware-search.py` and the pattern in G4 above.

### P2 — Important for longer or more complex workflows

- [ ] **`depends_on` in LLM schema:** Add `depends_on` to the OpenAI and Anthropic schema output so LLM-backed policies can use dependency chaining without extra prompting.
- [ ] **LLM policy guide:** Document how to write an LLM-backed policy end-to-end: schema rendering, calling the model, parsing the tool call response, returning a `Decision`. Point to `demo_protein_design_llm.py` as the reference implementation and note its dependencies.
- [ ] **Known limitations notice:** Add a prominent notice to the README listing G1–G9 with one-line summaries. A team needs to know what the system cannot do before they invest in it.

### P3 — High value but deferrable

- [ ] **Checkpointing (minimal):** A `save_state(path)` / `restore_state(path)` pair on `Operator` using `pickle` or `json` on `operator.state`. Even a best-effort dump (no future resurrection, only objectives and observations) would let teams recover from crashes and resume from the last checkpoint.
- [ ] **Peer-to-peer pattern:** Document the parent-relay workaround explicitly. Until native peer messaging is built, teams need a clear recipe for coordinating sibling operators through a parent.
- [ ] **`@workflow` decorator sugar:** Currently, dispatching a multi-step block requires writing a module-level async function and calling `self.engine.block(fn)(...)` inside a `@act`. A `@workflow` shorthand that wraps this pattern would improve ergonomics and discoverability.
- [ ] **Scale validation:** Run `08-hierarchical-operator.py` with 10+ sequential children and a 3-level hierarchy. Document observed performance and any bottlenecks.

---

## Summary

ADR addresses a gap in the scientific computing software stack that has existed as long as adaptive HPC scientific workflows have. The systems it connects — domain science above and HPC execution below — are well-studied. The steering loop between them has been reimplemented from scratch in every project.

ADR makes it:
- **Reusable** — the same operator pattern works for molecular dynamics, active learning, and optimization
- **Composable** — policies, goals, and operators combine cleanly
- **Autonomous** — the operator stops itself when science is done, restructures its own objectives as it learns, and schedules its own work without a human in the loop
- **LLM-ready** — a language model can steer any ADR operator without modifying the workflow
- **Goal-directed** — convergence spans both scientific criteria and computational resource bounds, is not a loop counter, and can evolve as the workflow runs

The result is a system where an operator — whether steered by rules, an ML model, or an LLM — runs a complete HPC scientific workflow from start to finish without human intervention: spawning work, observing results, restructuring what it is trying to achieve, and stopping when both the science and the infrastructure say it is done. That capability, at scale, on real supercomputer infrastructure, is what ADR uniquely enables.
