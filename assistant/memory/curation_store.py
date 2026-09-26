"""Persistence helpers for curator decisions shown in the inbox."""

from __future__ import annotations

import re

from sqlalchemy import select

from assistant.storage import CurationDecision, session_scope

_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)


def base_arxiv_id(arxiv_id: str) -> str:
    return _VERSION_SUFFIX.sub("", arxiv_id.strip())


def _decision_dict(row: CurationDecision) -> dict:
    return {
        "id": row.id,
        "arxiv_id": row.arxiv_id,
        "title": row.title,
        "authors": list(row.authors or []),
        "abstract": row.abstract,
        "categories": list(row.categories or []),
        "domain": row.domain,
        "score": row.score,
        "judge_reason": row.judge_reason,
        "status": row.status,
        "error": row.error,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def save_decisions(decisions: list[dict]) -> None:
    with session_scope() as session:
        for item in decisions:
            arxiv_id = base_arxiv_id(str(item.get("arxiv_id") or item.get("id") or ""))
            domain = str(item.get("domain") or "")
            if not arxiv_id or not domain:
                continue
            row = session.execute(
                select(CurationDecision).where(
                    CurationDecision.domain == domain,
                    CurationDecision.arxiv_id == arxiv_id,
                )
            ).scalar_one_or_none()
            if row is None:
                row = CurationDecision(arxiv_id=arxiv_id, domain=domain, title="")
                session.add(row)
            row.title = str(item.get("title") or arxiv_id)
            row.authors = list(item.get("authors") or [])
            row.abstract = str(item.get("abstract") or "")
            row.categories = list(item.get("categories") or [])
            row.score = float(item.get("score") or 0.0)
            row.judge_reason = str(item.get("judge_reason") or "")
            row.status = str(item.get("status") or "rejected")
            row.error = str(item["error"]) if item.get("error") else None


def list_decisions(domain: str | None = None, status: str | None = None) -> list[dict]:
    with session_scope() as session:
        query = select(CurationDecision)
        if domain:
            query = query.where(CurationDecision.domain == domain)
        if status:
            query = query.where(CurationDecision.status == status)
        rows = session.execute(query.order_by(CurationDecision.updated_at.desc())).scalars()
        return [_decision_dict(row) for row in rows]


def get_decision(decision_id: int) -> dict | None:
    with session_scope() as session:
        row = session.get(CurationDecision, decision_id)
        return _decision_dict(row) if row is not None else None


def set_decision_status(decision_id: int, status: str, error: str | None = None) -> dict:
    if status not in {"accepted", "rejected", "ingested", "failed", "dismissed"}:
        raise ValueError("invalid curation decision status")
    with session_scope() as session:
        row = session.get(CurationDecision, decision_id)
        if row is None:
            raise ValueError(f"curation decision {decision_id} not found")
        row.status = status
        row.error = error
        session.flush()
        return _decision_dict(row)