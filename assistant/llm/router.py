"""Maps a role name (e.g. 'qa', 'embedder') to a live model client.

Roles are declared in config.yaml under `llm.roles`. The router is the only
place agents touch the model layer — swap a provider in YAML and nothing
else needs to change.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from assistant.config import RoleSpec, get_config, get_settings
from assistant.llm import providers

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


class LLMRouter:
    def __init__(self) -> None:
        self._cfg = get_config().llm
        self._settings = get_settings()
        self._chat_cache: dict[str, BaseChatModel] = {}
        self._emb_cache: dict[str, Embeddings] = {}
        self._reranker_cache: dict[str, object] = {}

    def _spec(self, role: str) -> RoleSpec:
        spec = self._cfg.roles.get(role)
        if spec is None:
            raise KeyError(
                f"No LLM role '{role}' configured. Add it under llm.roles in config.yaml."
            )
        return spec

    def chat(self, role: str) -> BaseChatModel:
        if role in self._chat_cache:
            return self._chat_cache[role]
        spec = self._spec(role)
        match spec.provider:
            case "anthropic":
                client = providers.chat_anthropic(spec.model, self._settings)
            case "openai":
                client = providers.chat_openai(spec.model, self._settings)
            case "openrouter":
                client = providers.chat_openrouter(spec.model, self._settings)
            case "ollama":
                client = providers.chat_ollama(spec.model, self._settings)
            case _:
                raise ValueError(
                    f"Role '{role}' has provider '{spec.provider}' which is not a chat provider."
                )
        self._chat_cache[role] = client
        return client

    def embeddings(self, role: str = "embedder") -> Embeddings:
        if role in self._emb_cache:
            return self._emb_cache[role]
        spec = self._spec(role)
        match spec.provider:
            case "ollama":
                client = providers.embeddings_ollama(spec.model, self._settings)
            case "openai":
                client = providers.embeddings_openai(spec.model, self._settings)
            case "openrouter":
                client = providers.embeddings_openrouter(spec.model, self._settings)
            case _:
                raise ValueError(
                    f"Embedder role '{role}' uses unsupported provider '{spec.provider}'. "
                    f"Use ollama, openai, or openrouter."
                )
        self._emb_cache[role] = client
        return client

    def reranker(self, role: str = "reranker"):
        """Return a configured local or remote reranker."""
        if role in self._reranker_cache:
            return self._reranker_cache[role]
        spec = self._spec(role)
        if spec.provider == "openrouter":
            client = providers.reranker_openrouter(spec.model, self._settings)
            self._reranker_cache[role] = client
            return client
        if spec.provider != "local":
            raise ValueError(
                f"Reranker role '{role}' uses unsupported provider '{spec.provider}'. "
                f"Use local or openrouter."
            )
        # Lazy import — keeps startup fast and avoids forcing the heavy dep until Phase 4.
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "Install sentence-transformers to use the local reranker: "
                "pip install sentence-transformers"
            ) from e
        client = CrossEncoder(spec.model)
        self._reranker_cache[role] = client
        return client


@lru_cache(maxsize=1)
def get_router() -> LLMRouter:
    return LLMRouter()
