"""Multi-agent layer.

Every agent is a LangGraph node — a callable taking GraphState and returning a
partial state update. In MVP some agents are stubs (clearly marked) but every
node listed in the plan exists from day 1 so the graph topology is real.
"""
