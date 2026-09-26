"""Thin embedder wrapper.

Lets the rest of the code embed without touching LangChain or knowing which
provider is configured for the 'embedder' role.
"""

from __future__ import annotations

from assistant.llm import get_router


class Embedder:
    def __init__(self) -> None:
        self._client = get_router().embeddings("embedder")

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        # LangChain's `embed_documents` handles batching internally.
        return self._client.embed_documents(texts)
