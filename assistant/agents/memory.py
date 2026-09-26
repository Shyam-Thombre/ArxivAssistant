"""Memory Manager.

Writes episodic records (every Q&A turn). The `consolidate()` hook is a stub
for the post-MVP job that promotes durable facts from episodic memory into
the user profile and biases chunk ranking based on feedback.
"""

from __future__ import annotations

from assistant.config import get_config
from assistant.memory.curation_store import base_arxiv_id, save_decisions
from assistant.memory.episodic import record_interaction
from assistant.state import GraphState


def memory_node(state: GraphState) -> GraphState:
    intent = state.get("intent")
    if intent == "ask" and state.get("question"):
        interaction_id = record_interaction(
            question=state.get("question", ""),
            answer=state.get("answer", "") or "",
            citations=list(state.get("citations") or []),
            rag_mode=get_config().rag.mode,
            confidence=state.get("confidence"),
            extra={
                "conversation_id": state.get("conversation_id"),
                "domain": state.get("domain"),
                "paper_ids": list(state.get("paper_ids") or []),
                "mcp_results": len(state.get("mcp_results") or []),
                "retrieved_chunks": len(state.get("retrieved_chunks") or []),
                "did_gather": bool(state.get("did_gather")),
            },
        )
        scratch = {**(state.get("scratch") or {}), "interaction_id": interaction_id}
        return {"scratch": scratch}
    if intent == "monitor_tick":
        ingested = {base_arxiv_id(value) for value in state.get("ingested") or []}
        errors = {
            base_arxiv_id(str(item.get("source") or "")): str(item.get("error") or "")
            for item in (state.get("scratch") or {}).get("ingest_errors", [])
        }
        threshold = get_config().curation.accept_threshold
        decisions: list[dict] = []
        for item in state.get("judged") or []:
            arxiv_id = base_arxiv_id(str(item.get("arxiv_id") or item.get("id") or ""))
            score = float(item.get("score") or 0.0)
            status = "rejected"
            if arxiv_id in ingested:
                status = "ingested"
            elif score >= threshold:
                status = "failed"
            decisions.append({**item, "status": status, "error": errors.get(arxiv_id)})
        save_decisions(decisions)
    return {}


def consolidate() -> None:
    """Stub: extract durable user-profile facts from episodic memory and bias
    chunk ranking from feedback signals. Post-MVP."""
    return None
