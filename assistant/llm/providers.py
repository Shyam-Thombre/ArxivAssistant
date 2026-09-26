"""Provider factories.

Each factory takes a model name + Settings and returns a ready-to-use client.
Adding a new provider = one function here + a branch in the router.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

    from assistant.config import Settings


def chat_anthropic(model: str, settings: Settings) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    return ChatAnthropic(model=model, api_key=settings.anthropic_api_key)


def chat_openai(model: str, settings: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return ChatOpenAI(model=model, api_key=settings.openai_api_key)


def _openrouter_headers(settings: Settings) -> dict[str, str]:
    headers = {"X-OpenRouter-Title": settings.openrouter_app_name}
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    return headers


def chat_openrouter(model: str, settings: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return ChatOpenAI(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers=_openrouter_headers(settings),
    )


def chat_ollama(model: str, settings: Settings) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, base_url=settings.ollama_base_url)


def embeddings_ollama(model: str, settings: Settings) -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=model, base_url=settings.ollama_base_url)


def embeddings_openai(model: str, settings: Settings) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAIEmbeddings(model=model, api_key=settings.openai_api_key)


def embeddings_openrouter(model: str, settings: Settings) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return OpenAIEmbeddings(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers=_openrouter_headers(settings),
        check_embedding_ctx_length=False,
    )


class _OpenRouterReranker:
    def __init__(self, model: str, settings: Settings) -> None:
        self._model = model
        self._settings = settings

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        if any(pair_query != query for pair_query, _ in pairs):
            raise ValueError("OpenRouter reranking requires one shared query per request.")

        import httpx

        response = httpx.post(
            f"{self._settings.openrouter_base_url.rstrip('/')}/rerank",
            headers={
                "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                **_openrouter_headers(self._settings),
            },
            json={
                "model": self._model,
                "query": query,
                "documents": [document for _, document in pairs],
                "top_n": len(pairs),
            },
            timeout=60.0,
        )
        response.raise_for_status()
        scores = [float("-inf")] * len(pairs)
        for result in response.json().get("results", []):
            scores[int(result["index"])] = float(result["relevance_score"])
        return scores


def reranker_openrouter(model: str, settings: Settings) -> _OpenRouterReranker:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return _OpenRouterReranker(model, settings)
