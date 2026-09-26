"""Information Gatherer — invokes MCP tools when local knowledge is insufficient.

In MVP this is a lean wrapper: it loads MCP tools (if any), lets a small LLM
agent decide which to call, and writes results to state.mcp_results. The QA
agent invokes this node when its confidence is low (wired in Phase 8).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from assistant.llm import get_router
from assistant.mcp.client import get_tools_sync
from assistant.state import GraphState

log = logging.getLogger(__name__)


def info_gatherer_node(state: GraphState) -> GraphState:
    query = state.get("question") or (state.get("scratch") or {}).get("query")
    if not query:
        return {"mcp_results": []}

    tools = get_tools_sync()
    if not tools:
        log.info("Info Gatherer: no MCP tools available; skipping.")
        return {"mcp_results": []}

    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"type": "progress", "stage": "gathering"})
    except (LookupError, RuntimeError):
        pass

    llm = get_router().chat("info_gatherer")
    try:
        llm_with_tools = llm.bind_tools(tools)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not bind MCP tools to LLM: %s", e)
        return {"mcp_results": []}

    msg = llm_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "Use the available tools to gather information that helps answer the "
                    "user's question. Prefer narrow, targeted tool calls. Return a short "
                    "summary of findings."
                )
            ),
            HumanMessage(content=str(query)),
        ]
    )

    results: list[dict[str, Any]] = []
    tool_calls = getattr(msg, "tool_calls", None) or []
    for tc in tool_calls:
        results.append({"tool": tc.get("name"), "args": tc.get("args"), "raw": True})
    if not results:
        results.append({"summary": getattr(msg, "content", str(msg))})
    return {"mcp_results": results, "did_gather": True}
