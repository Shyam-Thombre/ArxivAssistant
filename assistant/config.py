"""Typed configuration loaded from YAML + environment.

Single source of truth for runtime settings. Loaded once via `get_settings()` and
passed to agents/services. API keys live in the environment, never in YAML.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai", "openrouter", "ollama", "local"]


class RoleSpec(BaseModel):
    provider: Provider
    model: str


class LLMConfig(BaseModel):
    roles: dict[str, RoleSpec]


class HybridConfig(BaseModel):
    dense_weight: float = 0.7
    sparse_weight: float = 0.3


class AgenticRAGConfig(BaseModel):
    max_iters: int = 3


class RAGConfig(BaseModel):
    mode: Literal["vanilla", "agentic"] = "vanilla"
    top_k: int = 8
    history_turns: int = 4
    low_confidence_threshold: float = 0.5  # below this, ask the Info Gatherer
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    agentic: AgenticRAGConfig = Field(default_factory=AgenticRAGConfig)


class DomainConfig(BaseModel):
    name: str
    arxiv_categories: list[str] = Field(default_factory=list)
    venues: list[str] = Field(default_factory=list)
    seed_papers: list[str] = Field(default_factory=list)


class MCPServerSpec(BaseModel):
    name: str
    transport: Literal["stdio", "http", "sse"] = "stdio"
    command: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class MCPConfig(BaseModel):
    servers: list[MCPServerSpec] = Field(default_factory=list)


class StorageConfig(BaseModel):
    qdrant_path: Path = Path("./data/qdrant")
    sqlite_path: Path = Path("./data/assistant.db")
    pdf_dir: Path = Path("./data/pdfs")

    def ensure_dirs(self) -> None:
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)


class CurationConfig(BaseModel):
    accept_threshold: float = 0.65


class MonitorConfig(BaseModel):
    enabled: bool = False
    interval_minutes: int = 360


class AppConfig(BaseModel):
    """Structure of config.yaml."""

    llm: LLMConfig
    rag: RAGConfig = Field(default_factory=RAGConfig)
    domains: list[DomainConfig] = Field(default_factory=list)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    curation: CurationConfig = Field(default_factory=CurationConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)


class Settings(BaseSettings):
    """Env-derived settings. Layered over AppConfig (which comes from YAML)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "AI Research Assistant"
    exa_api_key: str | None = None
    tavily_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    assistant_config: Path = Path("./config.yaml")


def load_app_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Copy config.yaml to your working directory "
            f"or set ASSISTANT_CONFIG."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(override=False)
    return Settings()


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    settings = get_settings()
    cfg_path = Path(os.environ.get("ASSISTANT_CONFIG", str(settings.assistant_config)))
    cfg = load_app_config(cfg_path)
    cfg.storage.ensure_dirs()
    return cfg
