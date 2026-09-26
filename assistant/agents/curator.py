"""Curator Agent — LLM-as-judge that scores candidate papers against criteria.

Input: state.candidates = list of dicts (abstract, title, authors, year, ...).
Output: state.accepted = candidates whose score >= config.curation.accept_threshold,
each annotated with its score and the judge's reasoning.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from assistant.config import get_config
from assistant.llm import get_router
from assistant.memory.criteria_store import CriteriaSnapshot, get_active
from assistant.state import GraphState

_SYSTEM = """You are a strict research-paper curator for a single user. Given a paper's
metadata + abstract and a set of curation criteria, return a JSON object:
{
  "score": <float 0..1>,            // how well the paper matches the criteria
  "reason": "<one or two sentences>"
}
0 = clearly irrelevant. 0.5 = borderline. 0.9+ = strong match. Be honest, not generous.
"""


def _parse(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"score": 0.0, "reason": "unparseable judge output"}


def _format_criteria(snap: CriteriaSnapshot | None) -> str:
    if snap is None:
        return "(no criteria configured for this domain — judge generously by relevance to the domain name only)"
    return (
        f"structured_rules:\n{json.dumps(snap.structured_rules, indent=2)}\n\n"
        f"nl_addendum:\n{snap.nl_addendum or '(none)'}"
    )


def _format_candidate(c: dict[str, Any]) -> str:
    return (
        f"Title: {c.get('title', '')}\n"
        f"Authors: {', '.join(c.get('authors', []))}\n"
        f"Year: {c.get('year', '')}\n"
        f"Venue: {c.get('venue', '')}\n"
        f"Categories: {', '.join(c.get('categories', []))}\n"
        f"Abstract:\n{c.get('abstract', '')}"
    )


def curator_node(state: GraphState) -> GraphState:
    cfg = get_config()
    candidates = state.get("candidates") or []
    if not candidates:
        return {"accepted": [], "judged": []}

    judge = get_router().chat("curator_judge")

    accepted: list[dict[str, Any]] = []
    judged: list[dict[str, Any]] = []
    for c in candidates:
        domain = str(c.get("domain") or state.get("domain") or "")
        snap = get_active(domain) if domain else None
        criteria_block = _format_criteria(snap)
        msg = judge.invoke(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(
                    content=(
                        f"Domain: {domain or '(unset)'}\n\n"
                        f"Criteria:\n{criteria_block}\n\n"
                        f"Paper:\n{_format_candidate(c)}"
                    )
                ),
            ]
        )
        parsed = _parse(getattr(msg, "content", str(msg)))
        score = float(parsed.get("score", 0.0))
        c2 = {
            **c,
            "domain": domain,
            "score": score,
            "judge_reason": parsed.get("reason", ""),
        }
        judged.append(c2)
        if score >= cfg.curation.accept_threshold:
            accepted.append(c2)
    return {"accepted": accepted, "judged": judged}
