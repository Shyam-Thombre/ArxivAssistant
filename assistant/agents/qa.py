"""QA Agent — synthesizes an answer from retrieved chunks with citations.

Flow: ensure chunks exist (calling the retrieval agent inline so the qa node
can run without an explicit edge from retrieval back to itself in MVP), then
prompt the QA LLM with a context block, parse a JSON-ish response with the
chunk IDs it used.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from assistant.agents.retrieval import retrieval_node
from assistant.config import get_config
from assistant.llm import get_router
from assistant.state import GraphState

_SYSTEM = """You are a research assistant answering questions about academic papers.
You will be given retrieved chunks labeled [1], [2], and so on. Answer the user's
question using only those chunks. If the chunks are insufficient, say so explicitly.
Use inline [n] markers for every factual claim supported by a chunk.

Format the answer as readable Markdown. Use short paragraphs, descriptive headings
for multi-part answers, and bullet or numbered lists when presenting multiple items.
Keep brief answers simple and do not add a heading unnecessarily.

Respond in JSON with this exact shape:
{
  "answer": "<your prose answer>",
    "citations": [<source number>, ...],
  "confidence": <float 0..1>
}
"""


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for index, c in enumerate(chunks, start=1):
        title = c.get("section_title") or c.get("section_type", "body")
        parts.append(f"[{index}] (paper={c['paper_id']}, section={title})\n{c['text']}")
    return "\n\n".join(parts) if parts else "(no chunks retrieved)"


def _format_history(history: list[dict], limit: int) -> str:
    turns = history[-limit:] if limit > 0 else []
    return "\n\n".join(
        f"User: {turn.get('question', '')}\nAssistant: {turn.get('answer', '')}"
        for turn in turns
    )


def _format_mcp(results: list[dict]) -> str:
    if not results:
        return ""
    parts = []
    for r in results:
        if "summary" in r:
            parts.append(f"[external] {r['summary']}")
        else:
            parts.append(f"[tool={r.get('tool')}] {r.get('args')}")
    return "\n\n".join(parts)


def _parse(raw: str) -> dict:
    # Try strict JSON first, then fall back to the first {...} block.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"answer": raw.strip(), "citations": [], "confidence": 0.0}


def _renumber_citations(
    answer: str,
    chunks: list[dict],
    raw_citations: list[object] | None = None,
) -> tuple[str, list[str]]:
    chunk_ids = [str(chunk["id"]) for chunk in chunks]
    source_numbers: list[int] = []

    for match in re.finditer(r"\[(\d+)\]", answer):
        number = int(match.group(1))
        if 1 <= number <= len(chunk_ids) and number not in source_numbers:
            source_numbers.append(number)

    for raw in raw_citations or []:
        raw_text = str(raw)
        number = int(raw_text) if raw_text.isdigit() else None
        if number is None and raw_text in chunk_ids:
            number = chunk_ids.index(raw_text) + 1
        if number is not None and 1 <= number <= len(chunk_ids) and number not in source_numbers:
            source_numbers.append(number)

    renumber = {
        source_number: index for index, source_number in enumerate(source_numbers, start=1)
    }

    def replace_marker(match: re.Match[str]) -> str:
        replacement = renumber.get(int(match.group(1)))
        return f"[{replacement}]" if replacement is not None else ""

    normalized_answer = re.sub(r"\[(\d+)\]", replace_marker, answer)
    citations = [chunk_ids[source_number - 1] for source_number in source_numbers]
    return normalized_answer, citations


def _send_progress(stage: str) -> None:
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"type": "progress", "stage": stage})
    except (LookupError, RuntimeError):
        pass


def qa_node(state: GraphState) -> GraphState:
    question = state.get("question", "").strip()
    if not question:
        return {"answer": "(empty question)", "citations": [], "confidence": 0.0}

    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        chunks = retrieval_node(state).get("retrieved_chunks", []) or []

    if not chunks:
        return {
            "answer": (
                "No relevant chunks were retrieved. Ingest some papers first "
                "(`assistant ingest <arxiv-id>`) or refine your question."
            ),
            "citations": [],
            "confidence": 0.0,
            "retrieved_chunks": chunks,
        }

    _send_progress("answering")
    llm = get_router().chat("qa")
    mcp_block = _format_mcp(state.get("mcp_results") or [])
    user_content = f"Question:\n{question}\n\nRetrieved chunks:\n{_format_context(chunks)}"
    history = list((state.get("scratch") or {}).get("history") or [])
    history_block = _format_history(history, get_config().rag.history_turns)
    if history_block:
        user_content = f"Conversation so far:\n{history_block}\n\n{user_content}"
    if mcp_block:
        user_content += f"\n\nAdditional context from external tools:\n{mcp_block}"
    msg = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=user_content)])
    parsed = _parse(getattr(msg, "content", str(msg)))
    answer, citations = _renumber_citations(
        str(parsed.get("answer", "")).strip(),
        chunks,
        list(parsed.get("citations") or []),
    )
    return {
        "answer": answer,
        "citations": citations,
        "confidence": float(parsed.get("confidence", 0.5)),
        "retrieved_chunks": chunks,
    }
