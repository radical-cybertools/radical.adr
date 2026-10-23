"""ADLMCPServer — exposes an Operator's ActionSet over the MCP protocol.

Optional dependency: ``pip install "radical.adr[mcp]"``

This module is a thin transport adapter.  The same ActionSet and tool schemas
used by LLM policies (``operator.get_schema``) are reused here — no duplication.

Usage::

    from radical.adr.mcp.server import ADLMCPServer

    server = ADLMCPServer(operator)
    await server.serve(transport="stdio")
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING
from typing import Any

from radical.adr.decision import Decision

if TYPE_CHECKING:
    from radical.adr.operator import Operator
    from radical.adr.tools import ActionSet

log = logging.getLogger(__name__)

# JSON-schema types rendered by ActionSet.get_schema → Python annotations for FastMCP.
_JSON_TYPES: dict[str, Any] = {
    "string":  str,
    "integer": int,
    "number":  float,
    "boolean": bool,
    "object":  dict,
    "array":   list,
}


class ADLMCPServer:
    """Wraps an Operator's ActionSet as an MCP server via FastMCP.

    Requires the ``mcp`` package, 1.x (``pip install "radical.adr[mcp]"``).

    When a tool is called via MCP, the resulting Action is placed on the
    operator's inbox as a CUSTOM ``{"mutation": ...}`` message.  The operator's
    run loop picks it up on the next cycle and applies it with that cycle's
    decision.
    """

    def __init__(
        self,
        operator:  "Operator",
        tools:     "ActionSet | None" = None,
        name:      str                = "ADR Workflow Operator",
    ) -> None:
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:
            raise ImportError(
                "ADLMCPServer requires the 'mcp' package (1.x). "
                'Install it with: pip install "radical.adr[mcp]"'
            ) from exc

        from radical.adr.message import InboxMessage
        from radical.adr.message import MessageKind

        self._operator    = operator
        self._tools       = tools if tools is not None else operator.get_actions()
        self._InboxMessage = InboxMessage
        self._MessageKind  = MessageKind

        self.mcp = FastMCP(name)
        for spec in self._tools.get_schema("anthropic"):
            self.mcp.add_tool(
                fn          = self._make_handler(operator, spec),
                name        = spec["name"],
                description = spec["description"],
            )

    def _make_handler(self, operator: "Operator", spec: dict) -> Any:
        tools    = self._tools
        MsgKind  = self._MessageKind
        InboxMsg = self._InboxMessage
        name     = spec["name"]
        schema   = spec["input_schema"]
        required = set(schema.get("required", []))

        async def handler(**kwargs: Any) -> dict:
            try:
                # FastMCP passes omitted optional params as None — drop them.
                args = {k: v for k, v in kwargs.items() if k in required or v is not None}
                mutation = _parse(tools, name, args)
                actions  = mutation.actions if isinstance(mutation, Decision) else [mutation]
                operator._validate_actions(actions)   # reject now; re-checked when applied
                await operator.inbox.put(InboxMsg(
                    kind      = MsgKind.CUSTOM,
                    sender_id = "mcp",
                    payload   = {"mutation": mutation},
                ))
                return {"status": "queued", "tool": name}
            except Exception as exc:
                log.exception("ADLMCPServer: tool '%s' failed", name)
                return {"status": "error", "detail": str(exc)}

        handler.__name__      = name
        handler.__signature__ = _signature(schema, required)  # type: ignore[attr-defined]
        return handler

    async def serve(self, transport: str = "stdio") -> None:
        """Start the MCP server.

        Args:
            transport: "stdio" (default) or "sse"
        """
        if transport == "stdio":
            await self.mcp.run_stdio_async()
        elif transport == "sse":
            await self.mcp.run_sse_async()
        else:
            raise ValueError(f"Unknown transport '{transport}'. Supported: stdio, sse")


def _signature(schema: dict, required: set[str]) -> inspect.Signature:
    """Keyword-only signature mirroring a tool's input schema, so FastMCP renders it."""
    params = []
    for pname, prop in schema.get("properties", {}).items():
        ann = _JSON_TYPES.get(prop.get("type"), Any)
        if pname in required:
            params.append(inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY,
                                            annotation=ann))
        else:
            params.append(inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY,
                                            default=None, annotation=ann | None))
    return inspect.Signature(params)


def _parse(tools: "ActionSet", name: str, args: dict) -> Any:
    """Turn a tool call into the Action (or Decision, for ``stop``) the ActionSet builds."""
    if name in ("spawn_task", "spawn_workflow"):
        # Rendered as {name, kwargs, depends_on}; the callables take the task kwargs flat.
        return getattr(tools, name)(args["name"], depends_on=args.get("depends_on"),
                                    **(args.get("kwargs") or {}))
    return getattr(tools, name)(**args)
