"""MCP client — loads servers from config.yaml and exposes their tools.

Uses `langchain-mcp-adapters` so MCP tools appear as ordinary LangChain
`BaseTool` objects that any agent can bind. If the adapters package or any
configured server is unavailable, the helpers degrade gracefully (return None
/ empty list) — agents must check.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any

from assistant.config import get_config

log = logging.getLogger(__name__)


def _build_connections() -> dict[str, dict[str, Any]]:
    cfg = get_config().mcp
    connections: dict[str, dict[str, Any]] = {}
    for s in cfg.servers:
        if s.transport == "stdio":
            if not s.command:
                log.warning("MCP server %r has stdio transport but no command; skipping.", s.name)
                continue
            connections[s.name] = {
                "transport": "stdio",
                "command": s.command[0],
                "args": list(s.command[1:]),
                "env": dict(s.env),
            }
        elif s.transport in ("http", "sse") and s.url:
            connections[s.name] = {"transport": s.transport, "url": s.url}
    return connections


@lru_cache(maxsize=1)
def get_mcp_client():
    """Return a MultiServerMCPClient, or None if no servers / adapters missing."""
    connections = _build_connections()
    if not connections:
        return None
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        log.warning("langchain-mcp-adapters not installed (%s); MCP tools disabled.", e)
        return None
    return MultiServerMCPClient(connections)


async def aget_tools() -> list[Any]:
    client = get_mcp_client()
    if client is None:
        return []
    try:
        return await client.get_tools()
    except Exception as e:  # noqa: BLE001
        log.warning("MCP get_tools failed: %s", e)
        return []


def get_tools_sync() -> list[Any]:
    """Sync wrapper. Safe to call from sync LangGraph nodes."""
    try:
        return asyncio.run(aget_tools())
    except RuntimeError:
        # already inside an event loop (e.g. async node) — caller should use aget_tools
        return []
