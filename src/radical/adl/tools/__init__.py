"""ToolSet — callable-first tool interface for ADL operators.

Tools are Python callables first; LLM schema rendering is a view on top.
Any reasoning backend (RulePolicy, LLMPolicy, LangGraph) can use the same tools.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable

from radical.adl.decision import Action, ActionKind, Decision

if TYPE_CHECKING:
    from radical.adl.operator import Operator

__all__ = ["ToolSet"]


class ToolSet:
    """Python-callable tools for an operator.

    Built-in tools (always present)::

        tools.spawn_task("evaluate", x=1.0)    → Action(SPAWN_TASK, ...)
        tools.spawn_workflow("refine_block")   → Action(SPAWN_BLOCK, ...)
        tools.cancel_task("uid-abc")           → Action(CANCEL_TASK, ...)
        tools.update_state("center", 3.14)     → Action(UPDATE_PARAM, ...)
        tools.stop()                           → Decision(stop=True)

    Plus one shorthand per ``@tool`` method::

        tools.dock(candidate="mol.pdb", target="6LU7")
            → Action(SPAWN_TASK, "dock", {"candidate": ..., "target": ...})

    LLM schema rendering::

        op.get_tools(format="openai")      # → list[dict]
        op.get_tools(format="anthropic")   # → list[dict]
        op.get_tools(format="langgraph")   # → list[BaseTool]
    """

    def __init__(self, operator: "Operator") -> None:
        object.__setattr__(self, "_operator", operator)

    # ── Built-in tools ────────────────────────────────────────────────────────

    def spawn_task(self, name: str, depends_on: "list[str] | None" = None, **kwargs: Any) -> Action:
        """Spawn a registered @tool by name.

        ``depends_on`` is a list of task UIDs that must be DONE before this
        tool is invoked — the resolved futures are passed as positional args.
        """
        return Action(kind=ActionKind.SPAWN_TASK, task_name=name,
                      task_kwargs=kwargs, depends_on=depends_on or [])

    def spawn_workflow(self, name: str, depends_on: "list[str] | None" = None, **kwargs: Any) -> Action:
        """Spawn a @tool method that returns a block Future (alias for spawn_task)."""
        return Action(kind=ActionKind.SPAWN_TASK, task_name=name,
                      task_kwargs=kwargs, depends_on=depends_on or [])

    def cancel_task(self, uid: str) -> Action:
        """Cancel a running or pending task by its UID."""
        return Action(kind=ActionKind.CANCEL_TASK, target_uid=uid)

    def update_state(self, key: str, value: Any) -> Action:
        """Write a key-value pair into operator state (objectives)."""
        return Action(kind=ActionKind.UPDATE_PARAM, param_key=key, param_value=value)

    def stop(self) -> Decision:
        """Signal the operator loop to terminate cleanly after this cycle."""
        return Decision(stop=True)

    # ── @tool shorthand ───────────────────────────────────────────────────────

    def __getattr__(self, name: str) -> Callable[..., Action]:
        op = object.__getattribute__(self, "_operator")
        tool_fns = type(op)._adl_tools_registry  # type: ignore[attr-defined]
        fn = tool_fns.get(name)
        if fn is not None:
            def _spawn(**kwargs: Any) -> Action:
                deps = kwargs.pop("depends_on", [])
                return Action(kind=ActionKind.SPAWN_TASK, task_name=name,
                              task_kwargs=kwargs, depends_on=deps)
            _spawn.__name__ = name
            _spawn.__doc__  = fn.__doc__ or f"Spawn the '{name}' task."
            return _spawn
        raise AttributeError(f"ToolSet has no tool '{name}'")

    # ── Schema rendering ──────────────────────────────────────────────────────

    def get_schema(self, format: str = "openai") -> list:
        """Render all tools in the requested vendor format."""
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

        for name, fn in type(op)._adl_tools_registry.items():  # type: ignore[attr-defined]
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
    ("cancel_task",    "Cancel a running or pending task by its UID.",
     [("uid", str)]),
    ("update_state",   "Write a key-value pair into operator state.",
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


def _to_langgraph(toolset: ToolSet, specs: list[dict]) -> list:
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
