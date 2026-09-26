# CLAUDE.md

Guidance for Claude Code (and other AI coding agents) working in this repository. Read this first before making non-trivial changes.

## TL;DR

- Single-user, personal AI research assistant. Tracks research domains, curates papers from arxiv, answers questions with citations.
- **Config-driven** is the core principle: provider/model is chosen per-agent-role in `config.yaml`, RAG mode (vanilla/agentic) is a config toggle, MCP servers are added by YAML edit. Don't hard-code these.
- **All nine agents are real LangGraph nodes** from day 1. Some are deeper than others. Do not delete or merge nodes — they're the public topology of the system.
- The approved design and build order live at `~/.claude/plans/i-want-to-develop-enchanted-wozniak.md`. The narrative ("why this shape?") is there.

## Repo layout

```
assistant/
  config.py          # Pydantic config (AppConfig + Settings) loaded from config.yaml + .env
  state.py           # LangGraph GraphState TypedDict — the shared state schema
  graph.py           # Wires all 9 agents into a LangGraph; SqliteSaver checkpointer
  cli.py             # Typer CLI — graph-backed commands use _invoke(); serve hosts the web app
  scheduler.py       # APScheduler wrapper around `monitor_tick` (disabled by default)

  web/
    app.py           # localhost FastAPI routes + SPA serving
    runner.py        # graph streaming + serialized in-memory jobs
    static/          # generated Vite build; gitignored

  agents/            # One file per node — all are real, some stubs are intentional
    orchestrator.py  # intent dispatch
    qa.py            # answers w/ citations + JSON parse; detours through info_gatherer on low conf
    retrieval.py     # vanilla vs agentic, paper/topic scope, follow-up rewrite
    curator.py       # per-topic LLM judge; preserves accepted and rejected results
    ingestion.py     # delegates to rag.ingest.ingest_source
    info_gatherer.py # binds MCP tools to the info_gatherer LLM
    criteria.py      # comment + active criteria -> new versioned criteria
    memory.py        # writes interactions + curation decisions; consolidate() is a stub
    monitor.py       # arxiv polling with library/inbox deduplication

  llm/
    router.py        # LLMRouter.chat/embeddings/reranker — only place agents touch the model layer
    providers.py     # one factory per (provider, modality)

  rag/
    arxiv_fetch.py   # arxiv ID -> PDF + metadata
    parser.py        # PyMuPDF + heuristic section detection -> ParsedPaper
    chunker.py       # section-aware chunking, 15% overlap
    embedder.py      # thin wrapper over LLMRouter.embeddings
    retriever.py     # hybrid dense+BM25 with RRF fusion; BM25 cached & invalidated on ingest
    reranker.py      # optional cross-encoder via sentence-transformers (lazy import)
    agentic.py       # decompose -> retrieve -> critique loop -> rerank
    ingest.py        # end-to-end: source -> chunks -> embeddings -> stores

  storage/
    schema.py        # SQLAlchemy models incl. papers, chats, decisions, criteria, memory
    domains.py       # shared config.yaml -> database topic synchronization
    sqlite_store.py  # engine + session_scope() context manager
    qdrant_store.py  # client + upsert + search + ensure_collection (lazy on first insert)

  memory/
    episodic.py      # record_interaction / set_feedback / recent
    criteria_store.py# get_active / append_version
    conversations.py # saved fixed-scope chats + history
    curation_store.py# curator inbox decisions and statuses

  mcp/
    client.py        # MultiServerMCPClient from config; aget_tools / get_tools_sync

  web/                 # React/Vite/TypeScript source; builds into assistant/web/static
```

`data/` is gitignored and holds the Qdrant store, SQLite DB, and downloaded PDFs.

## Conventions

### State management

- One `GraphState` TypedDict, declared in `state.py`. Add fields here when an agent needs to share something.
- Each agent **only writes the fields it owns** (see `ARCHITECTURE.md` for the field-by-agent map). Returning extra junk pollutes state.
- For multi-turn data that's not part of the public contract, use `state.scratch: dict[str, Any]`.

### LLM access

- **Never instantiate a model client directly.** Always go through `get_router().chat(role)` (or `.embeddings(role)` / `.reranker(role)`).
- **Never hardcode a role name** outside of `config.yaml` and the call site. If you find yourself adding a new role, add it to the YAML's example block and to the per-role default in `config.yaml`.
- Roles are arbitrary strings — `qa`, `retrieval`, `curator_judge`, `summarizer`, `criteria`, `info_gatherer`, `embedder`, `reranker`, `orchestrator` exist today. Add more by editing config; the router lazily resolves on first use.

### Storage

- All DB access goes through `with session_scope() as s:` — never create sessions ad hoc.
- Qdrant collections are created lazily on first insert (see `ensure_collection`). Don't pre-create them or assume a vector dimension. The first upsert sets it.
- Embedded Qdrant calls must remain under the store's process lock because web chat and background jobs use different threads.
- The BM25 index is in-memory and cached via `@lru_cache`. **Call `invalidate_bm25_cache()` after any operation that writes chunks** (the ingest path already does this — copy the pattern if you add another writer).

### Config

- Pydantic models in `config.py` are the source of truth for shape. If you add a field to `config.yaml`, add it to the corresponding model.
- API keys, base URLs → `.env` (Settings). Everything else → `config.yaml` (AppConfig).
- `get_config()` and `get_settings()` are `lru_cache`'d. They're fine to call from anywhere.

### CLI

- Every command should build a state dict and call `_invoke(state)` rather than calling `get_graph().invoke()` directly — `_invoke` attaches a fresh `thread_id` for the checkpointer.
- Use Rich for output (`console.print`, `Table`). Don't `print()`.

### Web

- Run one server process because embedded Qdrant is process-local. Qdrant calls are protected by the store lock.
- All `/api` writes require `X-Requested-With: assistant-ui`; keep localhost TrustedHost restrictions intact.
- Conversations have a fixed scope. Load prior turns from the conversation store and pass them via `scratch.history`; do not reuse LangGraph state across turns.
- Frontend source lives in `web/`; `npm run build` writes generated assets to `assistant/web/static/`.
- The API streams chat events as NDJSON. Keep event shapes compatible with `web/src/api.ts`.
- Background ingest and monitor jobs are intentionally serialized and in memory.

### Frontend

- Run `npm run lint` and `npm run build` from `web/` after changes; the build includes TypeScript checking.
- Run `npx prettier --check src/App.tsx src/api.ts` for the main frontend source files.
- Preserve fixed conversation scope (`library`, `topic`, or `paper`) and resolve citation numbers through the API-provided `sources` array.

## Common tasks

### Add a new agent

1. Create `agents/<name>.py` with a `<name>_node(state) -> partial_state` callable.
2. Register it in `graph.py`: `g.add_node("<name>", <name>_node)` + the edges that connect it.
3. If the agent needs new state fields, add them to `GraphState` in `state.py`.
4. If the agent calls an LLM, pick a role name and add it to `config.yaml`'s `llm.roles` block (with a sensible default provider/model).
5. Update `ARCHITECTURE.md`'s agent table.

### Add a new MCP server

Just edit `config.yaml`:

```yaml
mcp:
  servers:
    - name: my_server
      transport: stdio
      command: ["uvx", "some-mcp-server"]
      env: { SOME_KEY: "$SOME_ENV_VAR" }
```

No code changes. The Info Gatherer will pick it up.

### Swap a model

Edit one line in `config.yaml`:

```yaml
llm:
  roles:
    qa: { provider: ollama, model: qwen2.5:14b }  # was claude-sonnet-4-6
```

Done. No code changes. Test with `assistant config roles` then `assistant ask "..."`.

### Replace the PDF parser

`rag/parser.py:parse_pdf(pdf_path) -> ParsedPaper` is the swap point. Drop in Marker, GROBID, or anything else that returns the same shape (`full_text`, `sections: list[Section]`, `abstract`). Nothing else needs to change.

### Run a smoke test by hand

```powershell
assistant init
assistant ingest 2401.05566
assistant status                 # confirm papers/chunks rows > 0
assistant ask "summarize the method"
assistant serve --build
```

If embeddings fail, check Ollama is running (`ollama list`) or switch the `embedder` role to OpenAI.

## Gotchas

- **`langgraph-checkpoint-sqlite` is required for state persistence.** If it's not installed, `graph.py` silently falls back to `MemorySaver`. That's fine for dev but loses state across runs.
- **`sentence-transformers` is intentionally not in `pyproject.toml`.** Reranking degrades gracefully without it. Add it (`pip install sentence-transformers`) when you want better top-k quality.
- **Marker, the recommended better parser, is also not in deps** — it's heavy. PyMuPDF gets us to working end-to-end; swap when needed (see "Replace the PDF parser").
- **Manual CLI ingest has no topic option.** The shared ingest function and web UI support topic attribution; a future CLI `--domain` flag can thread it through `scratch`.
- **Web background jobs are in memory.** They are serialized but disappear on server restart.
- **Schema changes use `create_all`, not migrations.** New tables are automatic; changing existing columns needs an explicit migration strategy.
- **The BM25 index doesn't persist** — it's rebuilt per process from the SQLite chunks table. For very large libraries this becomes slow; the planned upgrade is Qdrant's native sparse vectors.
- **`config.yaml` keys with no default in `AppConfig` will error on startup.** If you add an optional field, give it a `Field(default_factory=...)`.
- **The QA -> info_gatherer detour can only fire once per invocation** (guarded by `state.did_gather`). Don't remove that guard or you'll loop.

## What NOT to do

- Don't create new agent files outside `agents/` and expect them to be wired automatically — the graph is built explicitly in `graph.py`.
- Don't add provider-specific code paths inside agents. If you need provider-specific behavior, push it into `llm/providers.py`.
- Don't import from `langchain` directly in agents when LangChain abstractions already wrap it (`get_router().chat(...)` returns a `BaseChatModel`; use `.invoke()`).
- Don't write user-facing docs into the code (long docstrings explaining what the project does). Code comments should explain *why*, not *what*. README/ARCHITECTURE/PLANNER are the user-facing docs.
- Don't commit anything in `data/` — it's gitignored.

## References

- **README.md** — user-facing intro + quickstart.
- **ARCHITECTURE.md** — full system architecture.
- **PLANNER.md** — assumptions, pending work, future requirements.
- **`~/.claude/plans/i-want-to-develop-enchanted-wozniak.md`** — the approved design plan that produced this codebase.
