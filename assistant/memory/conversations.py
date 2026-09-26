"""Persisted conversations and their interaction history."""

from __future__ import annotations

import uuid

from sqlalchemy import select

from assistant.storage import Conversation, Interaction, session_scope


def _conversation_dict(row: Conversation) -> dict:
    return {
        "id": row.id,
        "title": row.title,
        "scope_type": row.scope_type,
        "scope_target": row.scope_target,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def create_conversation(title: str, scope_type: str, scope_target: str | None) -> dict:
    if scope_type not in {"library", "topic", "paper"}:
        raise ValueError("scope_type must be library, topic, or paper")
    if scope_type != "library" and not scope_target:
        raise ValueError("topic and paper scopes require a target")
    with session_scope() as session:
        row = Conversation(
            id=str(uuid.uuid4()),
            title=title.strip() or "New conversation",
            scope_type=scope_type,
            scope_target=scope_target,
        )
        session.add(row)
        session.flush()
        return _conversation_dict(row)


def get_conversation(conversation_id: str) -> dict | None:
    with session_scope() as session:
        row = session.get(Conversation, conversation_id)
        return _conversation_dict(row) if row is not None else None


def list_conversations() -> list[dict]:
    with session_scope() as session:
        rows = session.execute(
            select(Conversation).order_by(Conversation.updated_at.desc())
        ).scalars()
        return [_conversation_dict(row) for row in rows]


def conversation_history(conversation_id: str, limit: int | None = None) -> list[dict]:
    with session_scope() as session:
        rows = session.execute(select(Interaction).order_by(Interaction.id.desc())).scalars()
        matches = [
            row for row in rows if (row.extra or {}).get("conversation_id") == conversation_id
        ]
    if limit is not None:
        matches = matches[:limit]
    return [
        {
            "id": row.id,
            "question": row.question,
            "answer": row.answer,
            "citations": list(row.cited_chunk_ids or []),
            "confidence": row.confidence,
            "feedback": row.user_feedback,
            "created_at": row.created_at.isoformat(),
        }
        for row in reversed(matches)
    ]


def delete_conversation(conversation_id: str) -> bool:
    with session_scope() as session:
        row = session.get(Conversation, conversation_id)
        if row is None:
            return False
        interactions = session.execute(select(Interaction)).scalars()
        for interaction in interactions:
            if (interaction.extra or {}).get("conversation_id") == conversation_id:
                session.delete(interaction)
        session.delete(row)
        return True