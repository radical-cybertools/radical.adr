"""ADLMCPServer — exposes an Operator's ToolRegistry over the MCP protocol.

Optional dependency: ``pip install radical-adl[mcp]``

This module is a thin transport adapter.  The same ToolRegistry and Pydantic
schemas used by LLMPolicy are reused here — no duplication.

Usage::

    from radical.adl.mcp.server import ADLMCPServer

    server = ADLMCPServer(operator)
    await server.serve(transport="stdio")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from radical.adl.operator import Operator
    from radical.adl.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


class ADLMCPServer:
    """Wraps an Operator's ToolRegistry as an MCP server via FastMCP.

    Requires ``mcp`` package (``pip install mcp``).

    When a tool is called via MCP, the mutation is placed on the operator's
    inbox as a CUSTOM message.  The operator's run loop picks it up on the next
    cycle and applies the mutation.
    """

    def __init__(
        self,
        operator:  "Operator",
        tools:     "ToolRegistry | None" = None,
        name:      str                   = "ADL Workflow Operator",
    ) -> None:
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:
            raise ImportError(
                "ADLMCPServer requires the 'mcp' package. "
                "Install it with: pip install radical-adl[mcp]"
            ) from exc

        from radical.adl.message import InboxMessage
        from radical.adl.message import MessageKind
        from radical.adl.tools.asyncflow_tools import ASYNCFLOW_TOOLS

        self._operator    = operator
        self._tools       = tools or ASYNCFLOW_TOOLS
        self._InboxMessage = InboxMessage
        self._MessageKind  = MessageKind

        self.mcp = FastMCP(name)
        for tool in self._tools.all():
            self.mcp.add_tool(
                fn          = self._make_handler(operator, tool),
                name        = tool.name,
                description = tool.description,
            )

    def _make_handler(self, operator: "Operator", tool: Any) -> Any:
        tools    = self._tools
        MsgKind  = self._MessageKind
        InboxMsg = self._InboxMessage

        async def handler(**kwargs: Any) -> dict:
            try:
                mutation = tools.parse(operator, tool.name, kwargs)
                if mutation is not None:
                    await operator.inbox.put(InboxMsg(
                        kind      = MsgKind.CUSTOM,
                        sender_id = "mcp",
                        payload   = {"mutation": mutation},
                    ))
                return {"status": "queued", "tool": tool.name}
            except Exception as exc:
                log.exception("ADLMCPServer: tool '%s' failed", tool.name)
                return {"status": "error", "detail": str(exc)}

        handler.__name__ = tool.name
        return handler

    async def serve(self, transport: str = "stdio") -> None:
        """Start the MCP server.

        Args:
            transport: "stdio" (default) or "sse"
        """
        await self.mcp.run(transport)
