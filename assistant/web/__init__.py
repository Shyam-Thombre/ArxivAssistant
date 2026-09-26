"""Local FastAPI application for the research assistant."""

from assistant.web.app import create_app

__all__ = ["create_app"]