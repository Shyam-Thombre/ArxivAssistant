"""Hybrid retriever: dense (Qdrant) + sparse (BM25) with RRF fusion.

BM25 is built lazily over all chunks in SQLite. For a single-user library this
is fast enough; we can swap to Qdrant's native sparse vectors later without
changing callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from sqlalchemy import select

from assistant.config import get_config
from assistant.rag.embedder import Embedder
from assistant.storage import CHUNKS_COLLECTION, Chunk as ChunkRow, search as qdrant_search
from assistant.storage import session_scope

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class RetrievedChunk:
    id: str
    paper_id: str
    text: str
    section_type: str
    section_title: str
    score: float
    source: str  # "dense" | "sparse" | "fused"


@dataclass
class _BM25Index:
    bm25: Any
    chunk_ids: list[str]
    chunk_meta: dict[str, dict[str, Any]]  # id -> {paper_id, text, section_type, section_title}


@lru_cache(maxsize=1)
def _bm25() -> _BM25Index | None:
    from rank_bm25 import BM25Okapi

    with session_scope() as s:
        rows = s.execute(select(ChunkRow)).scalars().all()
        ids: list[str] = []
        corpus: list[list[str]] = []
        meta: dict[str, dict[str, Any]] = {}
        for r in rows:
            ids.append(r.id)
            corpus.append(_tokenize(r.text))
            meta[r.id] = {
                "paper_id": r.paper_id,
                "text": r.text,
                "section_type": r.section_type,
                "section_title": r.section_title or "",
            }
    if not corpus:
        return None
    return _BM25Index(bm25=BM25Okapi(corpus), chunk_ids=ids, chunk_meta=meta)


def invalidate_bm25_cache() -> None:
    """Call after ingestion so the BM25 index is rebuilt on next retrieval."""
    _bm25.cache_clear()


def _dense(query: str, top_k: int, filt: dict[str, Any] | None) -> list[RetrievedChunk]:
    vec = Embedder().embed_query(query)
    raw = qdrant_search(CHUNKS_COLLECTION, vec, top_k=top_k, metadata_filter=filt)
    out: list[RetrievedChunk] = []
    for r in raw:
        p = r["payload"] or {}
        out.append(
            RetrievedChunk(
                id=str(r["id"]),
                paper_id=str(p.get("paper_id", "")),
                text=str(p.get("text", "")),
                section_type=str(p.get("section_type", "body")),
                section_title=str(p.get("section_title", "")),
                score=float(r["score"]),
                source="dense",
            )
        )
    return out


def _sparse(
    query: str,
    top_k: int,
    paper_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    idx = _bm25()
    if idx is None:
        return []
    scores = idx.bm25.get_scores(_tokenize(query))
    allowed = set(paper_ids) if paper_ids is not None else None
    ranked = sorted(
        (
            (chunk_id, score)
            for chunk_id, score in zip(idx.chunk_ids, scores)
            if allowed is None or idx.chunk_meta[chunk_id]["paper_id"] in allowed
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]
    out: list[RetrievedChunk] = []
    for cid, sc in ranked:
        m = idx.chunk_meta[cid]
        out.append(
            RetrievedChunk(
                id=cid,
                paper_id=m["paper_id"],
                text=m["text"],
                section_type=m["section_type"],
                section_title=m["section_title"],
                score=float(sc),
                source="sparse",
            )
        )
    return out


def _rrf_fuse(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    *,
    k: int = 60,
) -> list[RetrievedChunk]:
    """Reciprocal rank fusion. k=60 is the standard."""
    by_id: dict[str, RetrievedChunk] = {}
    rrf_score: dict[str, float] = {}
    for rank, c in enumerate(dense):
        rrf_score[c.id] = rrf_score.get(c.id, 0.0) + 1.0 / (k + rank + 1)
        by_id.setdefault(c.id, c)
    for rank, c in enumerate(sparse):
        rrf_score[c.id] = rrf_score.get(c.id, 0.0) + 1.0 / (k + rank + 1)
        by_id.setdefault(c.id, c)
    fused = sorted(by_id.values(), key=lambda c: rrf_score[c.id], reverse=True)
    for c in fused:
        c.score = rrf_score[c.id]
        c.source = "fused"
    return fused


def hybrid_retrieve(
    query: str,
    *,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
    paper_ids: list[str] | None = None,
    candidate_k: int = 30,
) -> list[RetrievedChunk]:
    """Return top_k chunks fused from dense + sparse retrieval."""
    cfg = get_config()
    top_k = top_k or cfg.rag.top_k
    dense_filter = dict(metadata_filter or {})
    if paper_ids is not None:
        dense_filter["paper_id"] = paper_ids
    sparse = _sparse(query, candidate_k, paper_ids)
    if not sparse:
        return []
    dense = _dense(query, candidate_k, dense_filter or None)
    fused = _rrf_fuse(dense, sparse)
    return fused[:top_k]
