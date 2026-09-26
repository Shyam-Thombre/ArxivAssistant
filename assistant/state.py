"""Shared LangGraph state schema.

A single state object flows through the multi-agent graph. Fields are optional —
each agent populates the ones it owns and leaves the rest untouched. Using a
TypedDict (rather than a class) is what LangGraph expects for state.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

Intent = Literal["ask", "ingest", "criteria_update", "monitor_tick", "unknown"]


class GraphState(TypedDict, total=False):
    # Routing
    intent: Intent
    domain: str | None
    paper_ids: list[str]
    conversation_id: str | None

    # QA flow
    question: str
    answer: str
    citations: list[str]            # chunk IDs
    confidence: float | None        # 0..1 self-reported

    # Retrieval
    retrieved_chunks: list[dict[str, Any]]
    sub_queries: list[str]
    agentic_iter: int

    # Ingestion / curation
    candidates: list[dict[str, Any]]     # arxiv search results
    accepted: list[dict[str, Any]]
    judged: list[dict[str, Any]]
    ingested: list[str]                  # paper IDs

    # Criteria update
    user_comment: str
    new_criteria_version: int | None

    # MCP / info gathering
    mcp_results: list[dict[str, Any]]
    did_gather: bool

    # Conversation messages (for chat-style use later)
    messages: Annotated[list[Any], add_messages]

    # Free-form scratch for agents that need to share intermediates
    scratch: dict[str, Any]
