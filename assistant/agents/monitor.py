"""Scheduler / Monitor.

Polls arxiv for recent papers in each tracked domain and emits them as
candidates for the Curator. MVP uses the direct arxiv API (no MCP required
to be running). When the arxiv MCP is configured, it can be used instead;
that's a Phase 8 refinement.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import select

from assistant.config import DomainConfig, get_config
from assistant.state import GraphState
from assistant.storage import CurationDecision, Paper, session_scope

log = logging.getLogger(__name__)

_MAX_PER_DOMAIN = 20
_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)


def _base_arxiv_id(value: str) -> str:
    return _VERSION_SUFFIX.sub("", value.strip())


def _fetch_recent_for_domain(d: DomainConfig) -> list[dict[str, Any]]:
    if not d.arxiv_categories:
        return []
    import arxiv

    query = " OR ".join(f"cat:{c}" for c in d.arxiv_categories)
    search = arxiv.Search(
        query=query,
        max_results=_MAX_PER_DOMAIN,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    out: list[dict[str, Any]] = []
    client = arxiv.Client()
    for r in client.results(search):
        out.append(
            {
                "arxiv_id": r.get_short_id(),
                "title": r.title.strip(),
                "authors": [a.name for a in r.authors],
                "abstract": r.summary.strip(),
                "year": r.published.year if r.published else None,
                "categories": list(r.categories or []),
                "domain": d.name,
            }
        )
    return out


def monitor_tick_node(state: GraphState) -> GraphState:
    cfg = get_config()
    requested = state.get("domain")
    domains = [d for d in cfg.domains if not requested or d.name == requested]
    if not domains:
        return {"candidates": []}

    with session_scope() as session:
        known_ids = {
            _base_arxiv_id(value)
            for value in session.execute(select(Paper.arxiv_id)).scalars()
            if value
        }
        known_ids.update(
            _base_arxiv_id(value)
            for value in session.execute(select(CurationDecision.arxiv_id)).scalars()
            if value
        )

    candidates: list[dict[str, Any]] = []
    seen = set(known_ids)
    for d in domains:
        try:
            for candidate in _fetch_recent_for_domain(d):
                base_id = _base_arxiv_id(str(candidate.get("arxiv_id") or ""))
                if not base_id or base_id in seen:
                    continue
                seen.add(base_id)
                candidates.append(candidate)
        except Exception as e:  # noqa: BLE001
            log.warning("monitor: arxiv fetch failed for %s: %s", d.name, e)

    out: GraphState = {"candidates": candidates}
    return out
