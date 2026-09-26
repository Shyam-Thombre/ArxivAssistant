"""LangGraph wiring — assembles all agents into the top-level state machine.

Compiled with a SQLite checkpointer so state is durable across runs.
Per-invocation `thread_id` (set by the CLI) keys the checkpoint row.
"""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from assistant.agents.criteria import criteria_node
from assistant.agents.curator import curator_node
from assistant.agents.info_gatherer import info_gatherer_node
from assistant.agents.ingestion import ingestion_node
from assistant.agents.memory import memory_node
from assistant.agents.monitor import monitor_tick_node
from assistant.agents.orchestrator import orchestrator_node, route
from assistant.agents.qa import qa_node
from assistant.agents.retrieval import retrieval_node
from assistant.config import get_config
from assistant.mcp.client import get_mcp_client
from assistant.state import GraphState

log = logging.getLogger(__name__)


def _build_checkpointer():
    """Return a checkpointer; SQLite if available, else MemorySaver."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        log.info("langgraph-checkpoint-sqlite not installed; using MemorySaver.")
        return MemorySaver()
    cfg = get_config()
    cfg.storage.ensure_dirs()
    conn = sqlite3.connect(cfg.storage.sqlite_path.as_posix(), check_same_thread=False)
    return SqliteSaver(conn)


def _route_after_qa(state: GraphState) -> str:
    if state.get("did_gather"):
        return "memory"
    conf = state.get("confidence") or 0.0
    threshold = get_config().rag.low_confidence_threshold
    if conf < threshold and get_mcp_client() is not None:
        return "info_gatherer"
    return "memory"


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("orchestrator", orchestrator_node)
    g.add_node("qa", qa_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("ingestion", ingestion_node)
    g.add_node("curator", curator_node)
    g.add_node("info_gatherer", info_gatherer_node)
    g.add_node("criteria", criteria_node)
    g.add_node("memory", memory_node)
    g.add_node("monitor", monitor_tick_node)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges(
        "orchestrator",
        route,
        {
            "ask": "qa",
            "ingest": "ingestion",
            "criteria_update": "criteria",
            "monitor_tick": "monitor",
        },
    )

    g.add_conditional_edges(
        "qa",
        _route_after_qa,
        {"info_gatherer": "info_gatherer", "memory": "memory"},
    )
    g.add_edge("info_gatherer", "qa")
    g.add_edge("retrieval", "qa")

    g.add_edge("monitor", "curator")
    g.add_edge("curator", "ingestion")
    g.add_edge("ingestion", "memory")

    g.add_edge("criteria", "memory")
    g.add_edge("memory", END)

    return g.compile(checkpointer=_build_checkpointer())


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
