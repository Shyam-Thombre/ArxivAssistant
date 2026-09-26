"""Cross-encoder reranker.

Optional layer between hybrid retrieval and the QA prompt. Loads lazily — if
sentence-transformers isn't installed, returns the input order unchanged with
a soft warning.
"""

from __future__ import annotations

import logging

from assistant.llm import get_router
from assistant.rag.retriever import RetrievedChunk

log = logging.getLogger(__name__)


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not chunks:
        return chunks
    try:
        model = get_router().reranker()
    except (ImportError, KeyError, ValueError) as e:
        log.warning("Reranker unavailable (%s); returning fused order.", e)
        return chunks[:top_k]
    pairs = [(query, c.text) for c in chunks]
    try:
        scores = model.predict(pairs)
    except Exception as e:  # noqa: BLE001
        log.warning("Reranking failed (%s); returning fused order.", e)
        return chunks[:top_k]
    scored = sorted(zip(chunks, scores), key=lambda t: float(t[1]), reverse=True)
    out: list[RetrievedChunk] = []
    for c, s in scored[:top_k]:
        c.score = float(s)
        c.source = "reranked"
        out.append(c)
    return out
