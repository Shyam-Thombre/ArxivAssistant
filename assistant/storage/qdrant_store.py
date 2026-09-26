"""Qdrant client + collection management.

Two collections:
- `papers`: one point per paper (summary embedding) — used for "find related"
  and dedup.
- `chunks`: one point per chunk — primary retrieval index.

Collection creation is deferred to first insert so we don't lock in a vector
dimension before the embedder is chosen.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Any, Iterable
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from assistant.config import get_config

PAPERS_COLLECTION = "papers"
CHUNKS_COLLECTION = "chunks"
_CLIENT_LOCK = RLock()
_LOGICAL_ID_KEY = "_logical_id"


@dataclass
class VectorRecord:
    id: str
    vector: list[float]
    payload: dict[str, Any]


def _point_id(collection: str, logical_id: str) -> str:
    try:
        return str(UUID(logical_id))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, f"assistant:{collection}:{logical_id}"))


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    cfg = get_config()
    cfg.storage.ensure_dirs()
    # Local on-disk mode — single-binary, no server needed.
    client = QdrantClient(path=str(cfg.storage.qdrant_path))
    atexit.register(client.close)
    return client


def ensure_collection(name: str, vector_size: int) -> None:
    """Idempotent: creates the collection if it doesn't exist."""
    with _CLIENT_LOCK:
        client = get_client()
        existing = {c.name for c in client.get_collections().collections}
        if name in existing:
            return
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )


def upsert(collection: str, records: Iterable[VectorRecord]) -> int:
    """Insert/update records. Creates the collection on first call (using the
    first record's vector size). Returns the count upserted.
    """
    records = list(records)
    if not records:
        return 0
    ensure_collection(collection, vector_size=len(records[0].vector))
    points = [
        qm.PointStruct(
            id=_point_id(collection, r.id),
            vector=r.vector,
            payload={**r.payload, _LOGICAL_ID_KEY: r.id},
        )
        for r in records
    ]
    with _CLIENT_LOCK:
        get_client().upsert(collection_name=collection, points=points)
    return len(points)


def search(
    collection: str,
    vector: list[float],
    top_k: int = 8,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Dense search. Returns a list of {id, score, payload} dicts."""
    flt = _build_filter(metadata_filter) if metadata_filter else None
    with _CLIENT_LOCK:
        client = get_client()
        response = client.query_points(
            collection_name=collection,
            query=vector,
            limit=top_k,
            query_filter=flt,
        )
    results: list[dict[str, Any]] = []
    for point in response.points:
        payload = dict(point.payload or {})
        logical_id = payload.pop(_LOGICAL_ID_KEY, point.id)
        results.append({"id": logical_id, "score": point.score, "payload": payload})
    return results


def _build_filter(spec: dict[str, Any]) -> qm.Filter:
    """Build equality filters, with list values matching any listed value."""
    return qm.Filter(
        must=[
            qm.FieldCondition(
                key=key,
                match=qm.MatchAny(any=value) if isinstance(value, list) else qm.MatchValue(value=value),
            )
            for key, value in spec.items()
        ]
    )


def delete_by_paper(paper_id: str) -> None:
    """Remove all chunk vectors belonging to a paper."""
    with _CLIENT_LOCK:
        client = get_client()
        existing = {collection.name for collection in client.get_collections().collections}
        if CHUNKS_COLLECTION not in existing:
            return
        client.delete(
            collection_name=CHUNKS_COLLECTION,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="paper_id", match=qm.MatchValue(value=paper_id))]
                )
            ),
        )
