"""Versioned curation criteria per domain.

Each domain owns a chain of CriteriaVersion rows. The latest `is_active=True`
row is what the Curator uses. Updates append a new version rather than mutating
the previous one — full audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, update

from assistant.storage import CriteriaVersion, Domain, session_scope


@dataclass
class CriteriaSnapshot:
    domain: str
    version: int
    structured_rules: dict[str, Any]
    nl_addendum: str
    source_comment: str | None


_EMPTY_RULES: dict[str, Any] = {
    "include_keywords": [],
    "exclude_keywords": [],
    "must_have": [],
    "min_year": None,
}


def get_active(domain_name: str) -> CriteriaSnapshot | None:
    with session_scope() as s:
        dom = s.execute(select(Domain).where(Domain.name == domain_name)).scalar_one_or_none()
        if dom is None:
            return None
        row = s.execute(
            select(CriteriaVersion)
            .where(CriteriaVersion.domain_id == dom.id, CriteriaVersion.is_active.is_(True))
            .order_by(CriteriaVersion.version.desc())
        ).scalar_one_or_none()
        if row is None:
            return CriteriaSnapshot(
                domain=domain_name,
                version=0,
                structured_rules=dict(_EMPTY_RULES),
                nl_addendum="",
                source_comment=None,
            )
        return CriteriaSnapshot(
            domain=domain_name,
            version=row.version,
            structured_rules=dict(row.structured_rules or {}),
            nl_addendum=row.nl_addendum or "",
            source_comment=row.source_comment,
        )


def append_version(
    domain_name: str,
    structured_rules: dict[str, Any],
    nl_addendum: str,
    source_comment: str | None,
) -> int:
    """Append a new version; deactivate previous active versions. Returns the new version number."""
    with session_scope() as s:
        dom = s.execute(select(Domain).where(Domain.name == domain_name)).scalar_one_or_none()
        if dom is None:
            raise ValueError(
                f"Domain {domain_name!r} not found. Add it under domains: in config.yaml "
                f"and run `assistant domain sync`."
            )
        last = s.execute(
            select(CriteriaVersion)
            .where(CriteriaVersion.domain_id == dom.id)
            .order_by(CriteriaVersion.version.desc())
        ).scalar_one_or_none()
        next_v = (last.version + 1) if last is not None else 1
        s.execute(
            update(CriteriaVersion)
            .where(CriteriaVersion.domain_id == dom.id)
            .values(is_active=False)
        )
        s.add(
            CriteriaVersion(
                domain_id=dom.id,
                version=next_v,
                structured_rules=structured_rules,
                nl_addendum=nl_addendum,
                source_comment=source_comment,
                is_active=True,
            )
        )
    return next_v
