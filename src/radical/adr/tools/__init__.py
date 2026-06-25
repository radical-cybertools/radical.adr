"""ActionSet — callable-first action interface for ADR operators.

Actions are Python callables first; LLM schema rendering is a view on top.
Any reasoning backend (RulePolicy, LLMPolicy, LangGraph) can use the same actions.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable

from radical.adr.decision import Action, ActionKind, Decision

if TYPE_CHECKING:
    from radical.adr.operator import Operator

__all__ = ["ActionSet"]


class ActionSet:
    """Python-callable actions for an operator.

    Built-in actions (always present)::

        acts.spawn_task("evaluate", x=1.0)    → Action(SPAWN_TASK, ...)
        acts.spawn_workflow("refine_block")   → Action(SPAWN_BLOCK, ...)
        acts.cancel_task("uid-abc")           → Action(CANCEL_TASK, ...)
        acts.update_param("center", 3.14)     → Action(UPDATE_PARAM, ...)
        acts.stop()                           → Decision(stop=True)

    Plus one shorthand per ``@act`` method::

        acts.dock(candidate="mol.pdb", target="6LU7")
            → Action(SPAWN_TASK, "dock", {"candidate": ..., "target": ...})

    LLM schema rendering::

        op.get_schema("openai")      # → list[dict]
        op.get_schema("anthropic")   # → list[dict]
        op.get_schema("langgraph")   # → list[BaseTool]
    """

    def __init__(self, operator: "Operator") -> None:
        object.__setattr__(self, "_operator", operator)

    # ── Built-in actions ─────────────────────────────────────────────────────

    def spawn_task(self, name: str, depends_on: "list[str] | None" = None, **kwargs: Any) -> Action:
        """Spawn a registered @act by name.

        ``depends_on`` is a list of task UIDs that must be DONE before this
        action is invoked — the resolved futures are passed as positional args.
        """
        return Action(kind=ActionKind.SPAWN_TASK, task_name=name,
                      task_kwargs=kwargs, depends_on=depends_on or [])

    def spawn_workflow(self, name: str, depends_on: "list[str] | None" = None, **kwargs: Any) -> Action:
        """Spawn a @act method that returns a block Future (alias for spawn_task)."""
        return Action(kind=ActionKind.SPAWN_TASK, task_name=name,
                      task_kwargs=kwargs, depends_on=depends_on or [])

    def spawn_operator(
        self,
        name:         str,
        operator_cls: Any = None,
        *,
        policy:       Any = None,
        depends_on:   "str | list[str] | None" = None,
        **config:     Any,
    ) -> Action:
        """Spawn a child operator (registering it on first use) as a block.

        ``operator_cls`` is required the first time a ``name`` is spawned (it is
        instantiated and parent-wired); on a re-spawn of an already-registered
        child it may be omitted. ``policy`` provisions the child's decision policy
        (a ``Policy`` instance, or a class/factory called with the child). ``config``
        is forwarded as construction kwargs.

        ``depends_on`` lists upstream child **names** whose completion gates this
        launch — accepts a single name or a list. The name→Future resolution happens
        internally; the policy never handles Futures or uids.
        """
        deps = [depends_on] if isinstance(depends_on, str) else list(depends_on or [])
        return Action(kind=ActionKind.SPAWN_OPERATOR, task_name=name, depends_on=deps,
                      operator_cls=operator_cls, policy=policy, task_kwargs=config)

    def restart_operator(
        self,
        name:       str,
        *,
        depends_on: "str | list[str] | None" = None,
    ) -> Action:
        """Re-run an already-registered child from the start.

        Cancels any in-flight run, resets the child to its **initial** declared
        config (the kwargs given to ``add_child`` / first ``spawn_operator``), and
        re-launches it. ``depends_on`` lists upstream child **names** gating the new
        launch (single name or list). Unlike ``spawn_operator`` this is *not*
        idempotent — it always starts a fresh run.
        """
        deps = [depends_on] if isinstance(depends_on, str) else list(depends_on or [])
        return Action(kind=ActionKind.RESTART_OPERATOR, task_name=name, depends_on=deps)

    def cancel_operator(self, name: str) -> Action:
        """Cancel a child operator's in-flight run by name.

        Cancels the child's block Future (propagating to its spawned tasks); the
        child then shows ``CANCELED``. The name stays known to the runtime, so a
        later ``spawn_operator`` is still a no-op — re-running is ``restart_operator``.
        Cancelling an unstarted/finished child is a logged no-op.
        """
        return Action(kind=ActionKind.CANCEL_OPERATOR, task_name=name)

    def cancel_task(self, uid: str) -> Action:
        """Cancel a running or pending task by its UID."""
        return Action(kind=ActionKind.CANCEL_TASK, target_uid=uid)

    def update_param(self, key: str, value: Any) -> Action:
        """Write a key-value pair into operator objectives."""
        return Action(kind=ActionKind.UPDATE_PARAM, param_key=key, param_value=value)

    def stop(self) -> Decision:
        """Signal the operator loop to terminate cleanly after this cycle."""
        return Decision(stop=True)

    # ── @act shorthand ───────────────────────────────────────────────────────

    def __getattr__(self, name: str) -> Callable[..., Action]:
        op = object.__getattribute__(self, "_operator")
        act_fns = type(op)._adl_act_registry  # type: ignore[attr-defined]
        fn = act_fns.get(name)
        if fn is not None:
            def _spawn(**kwargs: Any) -> Action:
                deps = kwargs.pop("depends_on", [])
                return Action(kind=ActionKind.SPAWN_TASK, task_name=name,
                              task_kwargs=kwargs, depends_on=deps)
            _spawn.__name__ = name
            _spawn.__doc__  = fn.__doc__ or f"Spawn the '{name}' task."
            return _spawn
        raise AttributeError(f"ActionSet has no act '{name}'")

    # ── Schema rendering ──────────────────────────────────────────────────────

    def get_schema(self, format: str = "openai") -> list:
        """Render all @act methods in the requested vendor format."""
        specs = self._collect_specs()
        if format == "openai":
            return [_to_openai(s) for s in specs]
        elif format == "anthropic":
            return [_to_anthropic(s) for s in specs]
        elif format == "langgraph":
            return _to_langgraph(self, specs)
        else:
            raise ValueError(
                f"Unknown format '{format}'. Supported: openai, anthropic, langgraph"
            )

    def _collect_specs(self) -> list[dict]:
        op = object.__getattribute__(self, "_operator")
        specs: list[dict] = []

        for name, desc, params in _BUILTIN_SPECS:
            specs.append({"name": name, "description": desc, "params": params})

        for name, fn in type(op)._adl_act_registry.items():  # type: ignore[attr-defined]
            try:
                sig = inspect.signature(fn)
                params = [
                    (pname, p.annotation)
                    for pname, p in sig.parameters.items()
                    if pname != "self"
                ]
            except (ValueError, TypeError):
                params = []
            specs.append({
                "name":        name,
                "description": (fn.__doc__ or f"Spawn the '{name}' task.").strip(),
                "params":      params,
            })

        return specs


# ── Schema helpers ─────────────────────────────────────────────────────────────

_BUILTIN_SPECS = [
    ("spawn_task",     "Spawn a registered AsyncFlow task by name with given kwargs.",
     [("name", str), ("kwargs", dict)]),
    ("spawn_workflow", "Spawn a registered AsyncFlow workflow block by name.",
     [("name", str), ("kwargs", dict)]),
    ("spawn_operator", "Spawn a child operator by name; depends_on lists upstream child names.",
     [("name", str), ("depends_on", list)]),
    ("restart_operator", "Re-run a registered child from the start (resets to initial config).",
     [("name", str), ("depends_on", list)]),
    ("cancel_operator", "Cancel a child operator's in-flight run by name (leaves a CANCELED tombstone).",
     [("name", str)]),
    ("cancel_task",    "Cancel a running or pending task by its UID.",
     [("uid", str)]),
    ("update_param",   "Write a key-value pair into operator objectives.",
     [("key", str), ("value", Any)]),
    ("stop",           "Signal the operator loop to terminate cleanly.", []),
]

_TYPE_MAP: dict[Any, str] = {
    str:   "string",
    int:   "integer",
    float: "number",
    bool:  "boolean",
    dict:  "object",
    list:  "array",
}


def _param_schema(params: list[tuple]) -> dict:
    props: dict = {}
    required: list[str] = []
    for pname, ptype in params:
        json_type = _TYPE_MAP.get(ptype)
        props[pname] = {"type": json_type} if json_type else {}
        required.append(pname)
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _to_openai(spec: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name":        spec["name"],
            "description": spec["description"],
            "parameters":  _param_schema(spec["params"]),
        },
    }


def _to_anthropic(spec: dict) -> dict:
    return {
        "name":         spec["name"],
        "description":  spec["description"],
        "input_schema": _param_schema(spec["params"]),
    }


def _to_langgraph(toolset: ActionSet, specs: list[dict]) -> list:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as e:
        raise ImportError(
            "langchain_core is required for format='langgraph'. "
            "Install: pip install langchain-core"
        ) from e

    result = []
    for spec in specs:
        name   = spec["name"]
        desc   = spec["description"]
        fn_ref = getattr(toolset, name, None)
        if callable(fn_ref):
            result.append(StructuredTool.from_function(fn_ref, name=name, description=desc))
    return result
