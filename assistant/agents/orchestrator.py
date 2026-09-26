"""Orchestrator / Router.

Owns the top-level dispatch: looks at GraphState.intent and routes to the
right sub-flow. In MVP the intent is set by the CLI command; this node is a
thin pass-through that LangGraph's conditional edges read from.
"""

from __future__ import annotations

from assistant.state import GraphState


def orchestrator_node(state: GraphState) -> GraphState:
    intent = state.get("intent", "unknown")
    if intent == "unknown":
        raise ValueError("Orchestrator received state with no intent set.")
    return {}


def route(state: GraphState) -> str:
    """Conditional-edge function — returns the next node name."""
    return state.get("intent", "unknown")
