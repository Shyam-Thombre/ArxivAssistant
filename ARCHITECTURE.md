# Architecture

This document describes how the AI Research Assistant is structured. For a user-facing overview, see [README.md](README.md). For open questions and pending work, see [PLANNER.md](PLANNER.md).

## High-level view

```
┌───────────────────────────────────────────────────────────────┐
│        CLI (Typer) / FastAPI + React local web app            │
└───────────────────────────────┬───────────────────────────────┘
                                │
                ┌───────────────▼────────────────┐
                │     LangGraph Orchestrator     │
                │  (state, checkpointer, router) │
                └──┬──────┬──────┬──────┬────────┘
                   │      │      │      │
        ┌──────────▼┐ ┌───▼──┐ ┌─▼────┐ ┌▼──────────┐
        │ QA / RAG  │ │Curator│ │Info │ │ Criteria  │
        │  Agent    │ │Agent  │ │Gath.│ │ Manager   │
        └─────┬─────┘ └───┬───┘ └──┬──┘ └─────┬─────┘
              │           │        │           │
   ┌──────────▼───────────▼────────▼───────────▼─────────┐
   │            Shared Services Layer                     │
   │  LLMRouter | MCPClient | Embedder | Reranker        │
   └──┬───────────┬────────────┬──────────────┬──────────┘
      │           │            │              │
  ┌───▼───┐   ┌───▼───┐    ┌───▼────┐    ┌────▼─────┐
  │Qdrant │   │SQLite │    │ Files  │    │  MCP     │
  │(vecDB)│   │(meta+ │    │(PDFs)  │    │ Servers  │
  │       │   │memory)│    │        │    │(arxiv,…) │
  └───────┘   └───────┘    └────────┘    └──────────┘
```

## Design principles

- **Config-driven.** Provider/model per agent role, RAG mode, MCP servers, storage paths — all in `config.yaml`. No code change should be required to swap any LLM, switch RAG mode, or add an MCP tool.
- **Single shared state.** A LangGraph `GraphState` TypedDict flows through every node. Each agent reads what it needs and writes the fields it owns — no other agent reaches into private state.
- **Citations always.** The QA agent writes numbered inline markers while state and storage retain stable `chunk_id` citations. Every interaction is logged with those IDs.
- **Versioned, never overwritten.** Curation criteria are append-only — every user comment becomes a new version, and the prior version remains queryable. Same idea will extend to profile facts later.
- **Stubs are real nodes.** Every agent in the design lives in the LangGraph from day 1, even if it returns trivially. This keeps the topology stable and makes "deepening" any agent a localized change.

## The multi-agent layer

All agents are LangGraph nodes — callables taking `GraphState` and returning a partial state update.

| Node | File | Role |
|---|---|---|
| `orchestrator` | `agents/orchestrator.py` | Reads `state.intent`; routes to the right branch via a conditional edge. |
| `qa` | `agents/qa.py` | Top-level QA. Ensures chunks are retrieved, prompts the QA LLM, parses JSON answer + citations + confidence. Detours through `info_gatherer` once when confidence is low. |
| `retrieval` | `agents/retrieval.py` | Picks vanilla vs agentic RAG, resolves paper/topic scope, and rewrites follow-up queries. |
| `curator` | `agents/curator.py` | LLM-as-judge over candidate abstracts. Loads criteria per candidate topic and preserves every decision. |
| `ingestion` | `agents/ingestion.py` | Calls `rag.ingest.ingest_source()` for every source in state (single from CLI, list from curator output). |
| `info_gatherer` | `agents/info_gatherer.py` | Binds MCP tools to the `info_gatherer` LLM and lets it call them. Triggered when QA confidence < `rag.low_confidence_threshold` AND at least one MCP server is configured. |
| `criteria` | `agents/criteria.py` | Takes a user comment + active criteria, asks the `criteria` LLM for an updated structured_rules + nl_addendum, appends a new version. |
| `memory` | `agents/memory.py` | Writes Q&A interactions and final curation decision statuses; consolidation is stubbed. |
| `monitor` | `agents/monitor.py` | Polls arxiv per domain and skips papers already ingested or judged. |

### Graph topology

```
START -> orchestrator -> {
    ask              -> qa
                          ├─if chunks missing─> retrieval_node inline
                          │                     └─vanilla hybrid or agentic loop
                          ├─if low confidence─> info_gatherer -> qa (once)
                          └─otherwise─────────> memory -> END
    ingest           -> ingestion -> memory -> END
    criteria_update  -> criteria  -> memory -> END
    monitor_tick     -> monitor -> curator -> ingestion -> memory -> END
}
```

Wired in `graph.py`. Conditional edges live in `orchestrator.route` (intent dispatch) and `_route_after_qa` (low-confidence detour). The `retrieval` callable is also registered as a public graph node, but the current ask path enters `qa`, which invokes retrieval when state has no chunks.

## Shared services

### `LLMRouter` (`llm/router.py`)

The only place agents touch the model layer. Three methods:

- `chat(role)` — returns a `BaseChatModel` for a chat-style role.
- `embeddings(role="embedder")` — returns an `Embeddings` object.
- `reranker(role="reranker")` — returns a local CrossEncoder (lazy import) or OpenRouter rerank client.

Adding a provider = a factory in `llm/providers.py` + a branch in the router. Adding a role = a YAML entry under `llm.roles`.

### `MCPClient` (`mcp/client.py`)

Builds a `MultiServerMCPClient` from `config.mcp.servers`. `get_tools_sync()` returns LangChain `BaseTool` objects bindable to any LLM. Degrades gracefully when the package or servers are missing — agents must check for empty.

### `Embedder` (`rag/embedder.py`)

Thin wrapper over `LLMRouter.embeddings("embedder")`. Used by ingestion (chunk + paper-summary vectors) and retrieval (query vector).

### Reranker (`rag/reranker.py`)

Optional layer between hybrid retrieval and the QA prompt. Supports a local cross-encoder via `sentence-transformers` or OpenRouter's rerank API. If unavailable, returns input order with a logged warning.

## RAG flow

### Vanilla (default)

1. `Embedder.embed_query(question)` → dense Qdrant search → top-`candidate_k` chunks with text in payload.
2. BM25 over all chunks in SQLite (`rank-bm25`, in-memory index, cached and invalidated on ingest) → top-`candidate_k`.
3. Reciprocal Rank Fusion (`k=60`) merges the two lists.
4. Reranker scores `(question, chunk_text)` pairs, returns top-`config.rag.top_k`.

### Agentic (`config.rag.mode: agentic`)

1. Decompose: `retrieval` LLM splits the question into 2–4 sub-queries.
2. For each sub-query, run vanilla hybrid retrieval and accumulate results.
3. Critique: LLM inspects accumulated chunks vs the *original* question; returns "gaps" (additional sub-queries) or empty.
4. If gaps and `iter < max_iters`, loop with the new sub-queries.
5. Final reranker pass against the original question → top-`top_k`.

Both paths accept the same paper-ID scope. Topic scope resolves to paper IDs in SQLite, and sparse results are filtered as strictly as dense results. `qa_node` labels the final chunks `[1]..[n]`, includes recent saved turns, and maps answer markers back to chunk IDs.

## Memory layers

Seven distinct layers, each with a clear store:

| Layer | Store | What it holds |
|---|---|---|
| **Working** | LangGraph state + SqliteSaver checkpointer | One run's intermediate state; resumable per `thread_id`. |
| **Episodic** | `interactions` table | Every Q&A turn: question, answer, cited chunk IDs, RAG mode, confidence, feedback. |
| **Semantic (paper KB)** | Qdrant `chunks` + `papers` + SQLite `papers`/`chunks` | Chunks for retrieval, paper-summary vectors for "related/dedup". |
| **Procedural (criteria)** | `criteria_versions` | Versioned per domain. Active = `is_active=True` + max version. |
| **Conversation** | `conversations` + scoped `interactions` | Fixed-scope saved chats; interaction `extra` links each turn. |
| **Curation inbox** | `curation_decisions` | Every judge result and its ingest/dismiss status. |
| **User profile** | `profile_facts` | Manual or consolidated; consumed by future ranking biases. |

Episodic writes happen in `memory_node` for `ask` intent. Feedback (`+1/0/-1`) is captured per interaction; bias on retrieval scoring is a future hook.

## Data model

SQLAlchemy declarative models in `storage/schema.py`:

- **Domain**: `name`, `arxiv_categories`, `venues`, `seed_papers`.
- **Paper**: `id` (arxiv ID or hash), title, authors, abstract, venue, year, pdf_path, summary, `domain_id`, `accepted_score`, status, extra JSON.
- **Chunk**: `id` (UUID matching Qdrant point), `paper_id`, `section_type` (abstract/intro/method/...), `section_title`, `order`, `text`, `char_start/end`.
- **CriteriaVersion**: `domain_id`, `version` (per-domain monotonic), `structured_rules` JSON, `nl_addendum`, `source_comment`, `is_active`.
- **Interaction**: `question`, `answer`, `cited_chunk_ids`, `rag_mode`, `confidence`, `user_feedback`, extra JSON.
- **Conversation**: UUID, title, fixed scope type/target, timestamps.
- **CurationDecision**: topic + base arXiv ID, paper metadata, score, reason, status, error.
- **ProfileFact**: `key`, `value`, `source` (manual/consolidated).

Qdrant collections:

- `papers` — one point per paper; payload = `{title, authors, year, arxiv_id}`; vector = embedding of abstract or generated summary.
- `chunks` — one point per chunk; payload = `{paper_id, section_type, section_title, order, text}`; vector = embedding of chunk text.

Vector dimensions are inferred from the first upsert (so the embedder can change without migrations as long as collections are empty or recreated).

## Configuration

`config.yaml` is loaded into typed Pydantic models in `config.py`:

- `AppConfig.llm.roles: dict[str, RoleSpec]` — per-role provider/model.
- `AppConfig.rag: RAGConfig` — mode, top_k, history_turns, low_confidence_threshold, hybrid weights, agentic max_iters.
- `AppConfig.domains: list[DomainConfig]` — name + arxiv categories + venues + seed papers.
- `AppConfig.mcp.servers: list[MCPServerSpec]` — name, transport, command/url.
- `AppConfig.storage: StorageConfig` — paths for Qdrant, SQLite, PDFs.
- `AppConfig.curation: CurationConfig` — accept threshold.
- `AppConfig.monitor: MonitorConfig` — enabled flag + interval.

`Settings` (Pydantic-settings) pulls API keys, the Ollama base URL, and optional OpenRouter endpoint/attribution settings from `.env`.

## Ingestion pipeline

`rag/ingest.py:ingest_source(source, domain=None, accepted_score=None)`:

1. **Resolve source.** arxiv ID → `arxiv_fetch.fetch()` downloads PDF + metadata. PDF path → use directly.
2. **Parse.** `parser.parse_pdf()` uses PyMuPDF to extract text, then heuristically detects section headings to produce a `ParsedPaper` of `Section` objects with `section_type` (abstract / intro / method / experiments / results / conclusion / etc.).
3. **Chunk.** `chunker.chunk_paper()` splits each section into ~1200-char chunks with 180-char overlap. Skips `references` and `acknowledgments`.
4. **Embed.** Chunk vectors in a batch; paper-level vector from abstract (or title fallback).
5. **Persist.** SQLite: upsert `Paper` + clear/rewrite `Chunk` rows for idempotency. Qdrant: upsert into `chunks` and `papers` collections (collections auto-created on first insert with the observed vector size).
6. **Invalidate BM25 cache** so newly-ingested chunks are searchable in the same process.

## MCP integration

`MultiServerMCPClient` from `langchain-mcp-adapters` is built from the config's server list. The Info Gatherer node binds its tools to the `info_gatherer` LLM via `bind_tools()` and lets the model decide which to call.

The QA detour: when `qa_node` returns `confidence < rag.low_confidence_threshold` AND `get_mcp_client() is not None`, the conditional edge routes through `info_gatherer`, which writes `mcp_results` + sets `did_gather=True`. The edge `info_gatherer → qa` then re-enters QA, which now augments its context with the MCP findings. `did_gather` prevents infinite loops.

## Scheduling

`scheduler.py:MonitorScheduler` wraps APScheduler's `BackgroundScheduler`. Each job invokes the monitor graph with a fresh checkpoint `thread_id`. Disabled by default — set `monitor.enabled: true` and `monitor.interval_minutes: N` to use it. `assistant monitor run` blocks and ticks on the configured interval.

## Local web layer

`assistant/web/app.py` exposes the library, topics, inbox, jobs, conversations, feedback, PDF, and streaming chat routes. `runner.py` gives each graph request a fresh checkpointer thread ID, streams LangGraph custom progress events as NDJSON, and serializes ingest/monitor jobs through one worker. FastAPI serves the Vite build from `assistant/web/static`; frontend source lives in `web/`.

The React app has four routed work surfaces:

- **Chat** creates, loads, and deletes fixed-scope conversations; renders Markdown answers and clickable numbered citations; and records feedback.
- **Library** searches and filters papers, queues arXiv/PDF ingestion, changes topic assignment, shows parsed sections, and starts paper-scoped chats.
- **Topics** merges config and database topic state, syncs config, starts topic-scoped chats, filters the library, runs monitors, and refines criteria.
- **Inbox** filters curator decisions and supports ingest, dismiss, and disagree/criteria-update actions.

Chat uses `application/x-ndjson`: an initial conversation event, custom progress events (`retrieving`, `answering`, `gathering`), then a final result enriched with citation details and the persisted interaction ID. Conversations do not reuse LangGraph checkpoint state; earlier turns are loaded from SQLite into `scratch.history` for each fresh invocation.

Write routes require `X-Requested-With: assistant-ui`, and `TrustedHostMiddleware` accepts localhost only. Embedded Qdrant calls are guarded by a process lock, and `assistant serve` always runs one Uvicorn worker.

## Checkpointer

LangGraph is compiled with `SqliteSaver` (from `langgraph-checkpoint-sqlite`, falling back to `MemorySaver`). CLI and web requests generate a fresh graph `thread_id`; web multi-turn context is loaded from persisted interactions and passed explicitly, preventing stale retrieved chunks from leaking between turns.

## Extension points (where to plug things in)

- **New LLM provider** → factory in `llm/providers.py`, branch in `llm/router.py`.
- **New agent** → file in `agents/`, node + edges in `graph.py`, state fields in `state.py`.
- **New MCP server** → entry under `mcp.servers` in `config.yaml`. No code.
- **New ingestion source** → `is_arxiv_id`-style branch in `rag/ingest.py:ingest_source()`.
- **Better PDF parser** → drop-in replacement for `rag/parser.py:parse_pdf()` returning `ParsedPaper`.
- **New retrieval strategy** → function in `rag/`, branch in `agents/retrieval.py`.
- **New CLI command** → `@app.command` in `cli.py` that builds a state dict and calls `_invoke()`.
- **New API route** → endpoint in `assistant/web/app.py`; write routes inherit the required request-header middleware.
- **New web workflow** → API types/client in `web/src/api.ts`, routed UI in `web/src/App.tsx`, then `npm run lint && npm run build`.
