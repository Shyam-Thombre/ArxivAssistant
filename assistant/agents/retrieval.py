"""Retrieval Agent — vanilla (Phase 4) and agentic (Phase 7).

Vanilla path: hybrid retrieve -> rerank -> populate state.retrieved_chunks.
Agentic path: see Phase 7; selected by config.rag.mode.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select

from assistant.config import get_config
from assistant.llm import get_router
from assistant.rag.agentic import agentic_retrieve
from assistant.rag.reranker import rerank
from assistant.rag.retriever import hybrid_retrieve
from assistant.state import GraphState
from assistant.storage import Domain, Paper, session_scope

_REWRITE_SYSTEM = """Rewrite the current follow-up question as a standalone search query.
Preserve technical terms and named papers from the conversation. Return only the query."""


def _send_progress(stage: str) -> None:
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"type": "progress", "stage": stage})
    except (LookupError, RuntimeError):
        pass


def _topic_paper_ids(domain_name: str) -> list[str]:
    with session_scope() as session:
        return list(
            session.execute(
                select(Paper.id)
                .join(Domain, Paper.domain_id == Domain.id)
                .where(Domain.name == domain_name)
            ).scalars()
        )


def _standalone_question(question: str, history: list[dict]) -> str:
    if not history:
        return question
    lines: list[str] = []
    for turn in history:
        lines.append(f"User: {turn.get('question', '')}")
        lines.append(f"Assistant: {turn.get('answer', '')}")
    conversation = "\n".join(lines)
    llm = get_router().chat("retrieval")
    message = llm.invoke(
        [
            SystemMessage(content=_REWRITE_SYSTEM),
            HumanMessage(content=f"Conversation:\n{conversation}\n\nCurrent question:\n{question}"),
        ]
    )
    rewritten = str(getattr(message, "content", message)).strip()
    return rewritten or question


def retrieval_node(state: GraphState) -> GraphState:
    cfg = get_config()
    question = state.get("question", "").strip()
    if not question:
        return {"retrieved_chunks": []}

    _send_progress("retrieving")
    paper_ids = state.get("paper_ids")
    if paper_ids is None and state.get("domain"):
        paper_ids = _topic_paper_ids(str(state["domain"]))
    if paper_ids == []:
        return {"retrieved_chunks": []}

    history = list((state.get("scratch") or {}).get("history") or [])
    search_question = _standalone_question(question, history)

    if cfg.rag.mode == "agentic":
        top = agentic_retrieve(search_question, top_k=cfg.rag.top_k, paper_ids=paper_ids)
    else:
        candidates = hybrid_retrieve(
            search_question,
            top_k=max(cfg.rag.top_k * 3, 20),
            paper_ids=paper_ids,
        )
        top = rerank(search_question, candidates, top_k=cfg.rag.top_k)
    return {
        "retrieved_chunks": [
            {
                "id": c.id,
                "paper_id": c.paper_id,
                "text": c.text,
                "section_type": c.section_type,
                "section_title": c.section_title,
                "score": c.score,
            }
            for c in top
        ]
    }
