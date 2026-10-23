"""ADLMCPServer: tools come from the operator's ActionSet, and a tool call reaches the operator.

Requires the optional ``mcp`` extra; the module is skipped cleanly without it.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mcp.server.fastmcp")

from radical.adr import Decision, Operator, Policy, act, decide, observe   # noqa: E402
from radical.adr.mcp.server import ADLMCPServer                             # noqa: E402


class _Op(Operator):
    center: float = 0.0

    @act
    async def evaluate(self, x: float) -> dict:
        """Evaluate the objective at x."""
        return {"x": x}

    @observe
    def o(self, snap) -> dict:
        return {"results": snap.results}


class _Idle(Policy):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[dict] = []

    @decide
    async def run(self, obs) -> Decision:
        self.seen.append(obs)
        return Decision()


async def _call(server: ADLMCPServer, tool: str, args: dict) -> dict:
    out = await server.mcp.call_tool(tool, args)
    if isinstance(out, tuple):          # (content, structured) on newer mcp 1.x
        out = out[0]
    return out if isinstance(out, dict) else json.loads(out[0].text)


async def test_server_constructs_with_operator_actions():
    op = _Op(None)
    server = ADLMCPServer(op)

    tools = {t.name: t for t in await server.mcp.list_tools()}
    assert {"spawn_task", "update_param", "cancel_task", "stop", "evaluate"} <= set(tools)
    assert tools["evaluate"].description == "Evaluate the objective at x."
    assert set(tools["evaluate"].inputSchema["properties"]) == {"x", "depends_on"}
    assert tools["evaluate"].inputSchema["required"] == ["x"]


async def test_tool_call_is_applied_by_operator():
    op = _Op(None, max_cycles=3)
    op.policy = _Idle()
    server = ADLMCPServer(op)

    assert await _call(server, "update_param", {"key": "center", "value": 2.5}) \
        == {"status": "queued", "tool": "update_param"}
    assert (await _call(server, "evaluate", {"x": 1.5}))["status"] == "queued"
    assert (await _call(server, "spawn_task",
                        {"name": "evaluate", "kwargs": {"x": 3.0}}))["status"] == "queued"
    await op.run_to_completion()

    assert op.center == 2.5
    results = [r for obs in op.policy.seen for r in obs["results"]]
    assert {"x": 1.5} in results and {"x": 3.0} in results


async def test_invalid_tool_call_returns_error_and_queues_nothing():
    op = _Op(None)
    server = ADLMCPServer(op)

    reply = await _call(server, "spawn_task", {"name": "no_such_act", "kwargs": {}})
    assert reply["status"] == "error" and "no_such_act" in reply["detail"]
    assert op.inbox.empty()


async def test_stop_tool_stops_the_run():
    op = _Op(None, max_cycles=10)
    op.policy = _Idle()
    server = ADLMCPServer(op)

    assert (await _call(server, "stop", {}))["status"] == "queued"
    await op.run_to_completion()
    assert op.state.cycle == 1


async def test_serve_awaits_the_async_transport(monkeypatch):
    server = ADLMCPServer(_Op(None))
    ran: list[str] = []

    async def fake_stdio() -> None:
        ran.append("stdio")

    async def fake_sse() -> None:
        ran.append("sse")

    monkeypatch.setattr(server.mcp, "run_stdio_async", fake_stdio)
    monkeypatch.setattr(server.mcp, "run_sse_async", fake_sse)
    await server.serve("stdio")
    await server.serve("sse")
    assert ran == ["stdio", "sse"]

    with pytest.raises(ValueError):
        await server.serve("carrier-pigeon")
