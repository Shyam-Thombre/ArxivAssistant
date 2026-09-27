# AI Research Assistant

A personal, single-user research assistant that **tracks named research domains** (e.g. speech synthesis, RAG, attention mechanisms), continuously curates "good" papers from arxiv + selected conferences into a local knowledge base, and answers questions over that knowledge base with citations.

Built as a config-driven multi-agent system on top of **LangGraph + Qdrant + SQLite + MCP**. Any LLM role can be backed by Anthropic, OpenAI, OpenRouter, or local Ollama — the choice is one YAML edit per role.

## What it does

- **Track domains** you care about. Each domain has its own arxiv categories, target venues, and (importantly) its own versioned curation criteria.
- **Curate automatically.** A manual or scheduled monitor polls arxiv, an LLM-judge scores each abstract against the active topic criteria, and accepted papers are ingested into the local KB while every decision remains reviewable.
- **Guide the curator with plain English.** Type `assistant criteria add rag "skip survey papers"` and a Criteria Manager turns that comment into a new versioned rule set the judge will use going forward.
- **Answer questions** with hybrid retrieval (BM25 + dense), optional cross-encoder reranking, and an agentic loop (decompose → iterate → critique → synthesize) toggleable in config.
- **Extend with MCP.** Any MCP server listed in `config.yaml` becomes a tool the Info Gatherer can call — arxiv, Exa/Tavily web search, Semantic Scholar, Zotero, etc.
- **Remember.** Every Q&A turn is logged to episodic memory with citations; criteria are versioned; the user profile is a structured store that the consolidation job (stub) will populate over time.
- **Work in a local web UI.** Chat with paper/topic scope, browse the library, manage topics, review curator decisions, upload PDFs, and leave answer feedback from one localhost-only app.

## Quickstart

```powershell
# 1. Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env          # fill in the API keys for the providers you use

# 2. (Optional) local embeddings via Ollama — recommended
ollama pull bge-m3

# 3. Initialize storage
assistant init                  # creates ./data, SQLite schema, etc.

# 4. Configure a domain in config.yaml under `domains:`, then:
assistant domain sync           # write config-yaml domains into the DB

# 5. Try it out
assistant ingest 2401.05566     # ingest a paper directly
assistant ask "what method does the paper propose?"
assistant monitor tick          # one curation cycle: fetch -> judge -> ingest
assistant criteria add rag "prioritize papers with reproducible code"
assistant criteria show rag     # see the new versioned criteria

# 6. Start the local web UI (Node.js is required for the first build)
assistant serve --build         # opens http://127.0.0.1:8000
```

## CLI

| Command | What it does |
|---|---|
| `assistant init` | Create data directories + SQLite schema |
| `assistant status` | Row counts for the core domain, paper, chunk, criteria, and interaction tables |
| `assistant ask "<question>"` | Run the QA pipeline |
| `assistant ingest <arxiv-id\|pdf-path>` | Manually ingest a paper (bypasses curator) |
| `assistant domain add <name> --categories cs.CL,cs.IR --venues ACL,EMNLP` | Track a new domain (DB direct) |
| `assistant domain list` | List tracked domains |
| `assistant domain sync` | Pull domains from `config.yaml` into the DB |
| `assistant criteria add <domain> "<comment>"` | Append a new criteria version from a comment |
| `assistant criteria show <domain>` | Print active criteria for a domain |
| `assistant monitor tick [--domain <name>]` | One curation cycle |
| `assistant monitor run` | Start the scheduler (blocks; requires `monitor.enabled: true`) |
| `assistant config show` | Dump active config |
| `assistant config roles` | Per-role LLM table |
| `assistant serve [--build] [--no-open]` | Build and/or run the local FastAPI + React app |

## Web UI

`assistant serve` runs one Uvicorn process on `127.0.0.1:8000`. Use `--build` after frontend changes; it installs `web/` dependencies when needed and builds into `assistant/web/static/`. The app provides:

- fixed-scope, persisted conversations with numbered inline citations, source drawers, deletion, and feedback;
- a searchable paper library with arXiv/PDF ingestion, topic assignment, manual New/Reviewing/Read status, status filtering, starred Favorites, removal, and paper-scoped chat;
- topic status, topic-scoped chat, paper filtering, criteria refinement, and manual monitor runs;
- a curation inbox for accepted, rejected, failed, ingested, and dismissed decisions.

Chats created as "New conversation" receive a short LLM-generated name after their first successful answer. The name is based on the first user message, saved to SQLite, and streamed to the sidebar and chat header. Existing descriptive names are preserved; older unnamed chats are named on their next successful reply. Naming failures leave the answer intact and can retry on a later turn. The `conversation_title` role selects the naming model, with `qa` as a fallback for older configs.

New chat and conversation history live in the main sidebar below the page tabs. Select Whole library, One topic, or One paper inside the message composer before sending the first message, which creates the conversation. Existing conversations display their fixed scope in the same controls; start a new chat to choose a different scope.

Library, Topics, and Inbox open beside the always-visible conversation. On desktop, the workspace gets roughly 60% of the available content width and chat gets the rest, with independent scrolling. Narrow screens stack the workspace above chat while keeping the composer visible. Close the workspace pane or select Chat for a full-width conversation. Starting a chat, selecting history, or chatting about a paper/topic keeps the workspace open. Drafts and pending responses remain attached to their original conversation; a full page reload still does not resume an in-flight response, though completed turns remain saved.

The API accepts write requests only with the UI's `X-Requested-With` header and rejects foreign Host headers. It is intentionally local and has no login.

New papers start as New and unstarred. Reading status changes only when you select it; re-ingestion preserves status and favorites. Existing databases are upgraded automatically on startup, with existing papers defaulting to New and unstarred. Restart the API after upgrading.

Removing a paper requires confirmation and removes its library entry, chunks, and search vectors. The original PDF and saved chats are retained; citations to removed chunks become unavailable. The paper detail pane shows metadata and actions, not raw retrieval chunks.

### Answer attribution

Paper citations are selective: general background does not need a paper reference. For each proposed citation, the answer model must provide an exact supporting quote from a retrieved chunk. The application checks that the quote occurs in that chunk (ignoring whitespace differences), then a separate verification call checks whether the evidence supports the entire claim. Only accepted references become numbered citations.

Claims without an accepted paper citation are marked **LLM knowledge**, including general explanations and claims whose proposed evidence could not be verified. This label means the claim is not verified against a paper; it is not a guarantee of correctness. Answers can use general knowledge even when no paper chunks are retrieved. Unsupported paper-specific findings must not be invented.

Web chat displays this attribution as a compact **LK** chip styled like numbered citations, with the full meaning in its tooltip. Confidence remains available internally but is not shown as a badge below answers.

The verifier uses the `citation_verifier` role in `config.yaml`, falling back to `qa` when that role is absent. Verification adds one model call when candidate evidence exists. Invalid quotes, rejected evidence, and verification failures do not produce paper citations. Verification is model-based and can still make mistakes; stronger verifier models may improve accuracy. Existing saved answers are not re-verified automatically.

For frontend development, run the API with `assistant serve --no-open` and then:

```powershell
cd web
npm install
npm run dev       # Vite proxies /api to 127.0.0.1:8000
npm run lint
npm run build     # type-checks and writes assistant/web/static/
```

## Configuration

The single source of truth is `config.yaml`. The key idea is **per-role LLM mapping** — each agent role picks its own provider/model:

```yaml
llm:
  roles:
    orchestrator:  { provider: anthropic, model: claude-sonnet-4-6 }
    qa:            { provider: anthropic, model: claude-sonnet-4-6 }
    citation_verifier: { provider: anthropic, model: claude-sonnet-4-6 }
    conversation_title: { provider: anthropic, model: claude-haiku-4-5 }
    retrieval:     { provider: anthropic, model: claude-haiku-4-5 }
    curator_judge: { provider: openai,    model: gpt-4o-mini }
    summarizer:    { provider: ollama,    model: qwen2.5:7b }
    criteria:      { provider: anthropic, model: claude-sonnet-4-6 }
    info_gatherer: { provider: anthropic, model: claude-haiku-4-5 }
    embedder:      { provider: ollama,    model: bge-m3 }
    reranker:      { provider: local,     model: BAAI/bge-reranker-v2-m3 }

rag:
  mode: vanilla        # or "agentic"
  top_k: 8
  history_turns: 4
```

Switch any role to a different provider/model by editing one line — no code changes.

To use OpenRouter, set `OPENROUTER_API_KEY` in `.env` and assign any role a model from the [OpenRouter catalog](https://openrouter.ai/models):

```yaml
llm:
  roles:
    qa:       { provider: openrouter, model: anthropic/claude-sonnet-4.6 }
    embedder: { provider: openrouter, model: openai/text-embedding-3-small }
```

Model IDs are passed through unchanged, so provider-prefixed slugs, variants, and aliases supported by OpenRouter do not require code changes. Use a chat-capable model for agent roles and an embedding model for `embedder`. Optional `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME`, and `OPENROUTER_BASE_URL` settings are shown in `.env.example`.

API keys live in `.env`. Ollama base URL defaults to `http://localhost:11434`.

## Project layout

```
assistant/
  agents/        # 9 LangGraph nodes (orchestrator, qa, retrieval, curator, ...)
  llm/           # Per-role LLM router + provider factories
  rag/           # arxiv fetch, parser, chunker, embedder, retriever, reranker, agentic loop
  storage/       # SQLAlchemy schema, domain sync + locked Qdrant client
  memory/        # episodic, criteria, conversation + curation stores
  mcp/           # MCP client (loads servers from config)
  web/           # FastAPI routes, job runner, built React assets
  config.py, state.py, graph.py, cli.py, scheduler.py
web/              # Vite + React + TypeScript source
```

## Further reading

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — layers, agents, data model, RAG flow
- **[PLANNER.md](PLANNER.md)** — assumptions, pending work, future requirements
- **[CLAUDE.md](CLAUDE.md)** — guidance for AI coding agents working in this repo
