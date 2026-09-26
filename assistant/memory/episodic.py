"""Episodic memory — one row per assistant interaction.

The Memory Manager writes here on every turn. Phase-9 stub does not yet bias
retrieval ranking from feedback, but the data shape supports it (see
Interaction.cited_chunk_ids + user_feedback).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from assistant.storage import Interaction, session_scope


def record_interaction(
    *,
    question: str,
    answer: str,
    citations: list[str],
    rag_mode: str | None,
    confidence: float | None,
    extra: dict[str, Any] | None = None,
) -> int:
    with session_scope() as s:
        row = Interaction(
            question=question,
            answer=answer,
            cited_chunk_ids=list(citations or []),
            rag_mode=rag_mode,
            confidence=confidence,
            extra=extra or {},
        )
        s.add(row)
        s.flush()
        return row.id


def set_feedback(interaction_id: int, score: int) -> None:
    """Score: -1 (bad), 0 (neutral), +1 (good)."""
    if score not in (-1, 0, 1):
        raise ValueError("feedback score must be -1, 0, or 1")
    with session_scope() as s:
        row = s.get(Interaction, interaction_id)
        if row is None:
            raise ValueError(f"interaction {interaction_id} not found")
        row.user_feedback = score


def recent(limit: int = 20) -> list[Interaction]:
    with session_scope() as s:
        return list(
            s.execute(select(Interaction).order_by(Interaction.id.desc()).limit(limit))
            .scalars()
        )
