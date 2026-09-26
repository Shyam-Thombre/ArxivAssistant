"""Ingestion Agent — PDF -> parsed sections -> chunks -> Qdrant + SQLite.

Reads source(s) from state and delegates to assistant.rag.ingest. Sources can
arrive two ways:
- `scratch["source"]`: a single arxiv ID or PDF path (from `assistant ingest`)
- `accepted`: a list of dicts from the Curator (from the monitor flow)
"""

from __future__ import annotations

from typing import Any

from assistant.rag.ingest import ingest_source
from assistant.state import GraphState


def _requests_from_state(state: GraphState) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    scratch = state.get("scratch") or {}
    single = scratch.get("source")
    if single:
        requests.append(
            {
                "source": single,
                "domain": scratch.get("domain") or state.get("domain"),
                "score": scratch.get("accepted_score"),
            }
        )
    for cand in state.get("accepted", []) or []:
        if isinstance(cand, dict):
            sid = cand.get("arxiv_id") or cand.get("id") or cand.get("source")
            if sid:
                requests.append(
                    {"source": sid, "domain": cand.get("domain"), "score": cand.get("score")}
                )
    return requests


def ingestion_node(state: GraphState) -> GraphState:
    requests = _requests_from_state(state)
    ingested: list[str] = []
    errors: list[dict[str, Any]] = []
    for request in requests:
        source = str(request["source"])
        try:
            paper_id = ingest_source(
                source,
                domain=request.get("domain"),
                accepted_score=request.get("score"),
            )
            ingested.append(paper_id)
        except Exception as e:  # noqa: BLE001
            errors.append({"source": source, "error": repr(e)})
    out: GraphState = {"ingested": ingested}
    if errors:
        out["scratch"] = {**(state.get("scratch") or {}), "ingest_errors": errors}
    return out
