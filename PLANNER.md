# PLANNER.md

Living document of what we assumed, what is built, what's intentionally pending, and what's known to be a future improvement. Update this whenever the answer to "wait, why isn't X done?" or "what about Y later?" comes up.

---

## Assumptions

These are the premises the design rests on. If any of them changes, parts of the architecture should be revisited.

| # | Assumption | Where it bites if violated |
|---|---|---|
| A1 | **Single user.** No multi-tenancy, no auth, no per-user data isolation. | Storage layer would need user scoping; checkpointer thread IDs would need namespacing. |
| A2 | **Scalability is not a concern yet.** Library size ~ low thousands of papers, query throughput ~ tens/day. | BM25 in-memory cache, single-process scheduler, embedded Qdrant all break at scale. |
| A3 | **The user is comfortable editing `config.yaml`.** Configuration is the primary UX for changing behavior; CLI surfaces a subset for convenience. | Need a config-management UI / `config set` CLI for less technical use. |
| A4 | **Python 3.10+** on Windows. The package declares `requires-python = ">=3.10"` and uses modern union syntax with postponed annotations. | Lower versions need conditional type-hint imports; some deps may raise the practical floor over time. |
| A5 | **Network access at runtime.** arxiv fetch, Anthropic/OpenAI APIs, Ollama (localhost), MCP servers all assume up connectivity. | Offline-first mode would require local caching + degraded paths. |
| A6 | **Ollama is the default local model host** when a role is set to `provider: ollama`. Expected at `http://localhost:11434` unless `OLLAMA_BASE_URL` overrides. | Other local hosts (LM Studio, vLLM) need a new provider factory. |
| A7 | **arxiv is the authoritative paper source for now.** Conferences are tracked by venue metadata but not directly scraped — we rely on arxiv preprints carrying venue info. | True conference-only papers (no arxiv version) are missed. |
| A8 | **English-only.** Embeddings (bge-m3 is multilingual but prompts aren't), section heuristics, and curation criteria are all English-language. | Non-English papers parse poorly. |
| A9 | **Per-role LLM mapping is what the user wants** — not a single "smart default" or auto-routing. Cost/quality tuning is manual. | If the user prefers "just pick a model," we'd want a `provider_profile: cheap|balanced|premium` shortcut. |
| A10 | **The development machine has enough RAM for the embedder.** bge-m3 via Ollama is ~600MB. Reranker (`bge-reranker-v2-m3`) adds another ~600MB if installed. | RPi target (8GB) is tight when running both + a curator/judge call simultaneously. |
| A11 | **Curation criteria are domain-scoped, not global.** Each domain has its own version chain. | If the user wants global rules ("always exclude surveys"), we'd need a `global` pseudo-domain or a criteria-merge step in the curator. |

---

## Intentionally pending (clear implementation boundaries)

These have named entry points or established extension points so future work is localized.

| Item | Where | Why deferred | Acceptance criteria |
|---|---|---|---|
| **Memory consolidation** | `agents/memory.py:consolidate()` | Nightly job that promotes durable facts from `interactions` → `profile_facts` and biases chunk-ranking from feedback. Not needed until enough interaction data accumulates. | Has an extractor that turns N recent interactions into ≤ K profile-fact rows; retrieval ranker reads `user_feedback` to up/down-rank chunks. |
| **Auto-scheduler in monitor** | `scheduler.py` + `config.monitor.enabled` | Wired but `enabled: false` by default. Manual `assistant monitor tick` works. Keep off until the curation criteria are tuned enough to trust unattended ingestion. | User flips the flag, `assistant monitor run` runs for a week without producing low-quality ingests. |
| **Marker PDF parser** | `rag/parser.py:parse_pdf` | PyMuPDF is "good enough" and avoids a heavy install. Marker handles equations and multi-column layouts much better. | Drop-in replacement returning `ParsedPaper`; same shape, higher fidelity, no other code changes. |
| **Reranker dep in pyproject** | `pyproject.toml` | `sentence-transformers` is heavy; reranker degrades gracefully without it. Install on demand. | Either add as optional extra (`pip install -e .[rerank]`) or just document the manual install — TBD. |
| **CLI topic selection on manual ingest** | `cli.py:ingest` | The shared ingest path supports a topic and the web UI exposes it, but the CLI command has no `--domain` flag. | Add the flag and thread it through `scratch`. |
| **CLI multi-turn chat** | `cli.py:_invoke` | Saved, scoped multi-turn conversations exist in the web UI; CLI calls remain one-shot. | Add `assistant chat` over the conversation store. |
| **Feedback-biased retrieval** | `rag/retriever.py` + `memory/episodic.py` | Episodic memory captures `user_feedback` per interaction with cited chunks; retrieval doesn't yet read it. | Hybrid score blends a feedback prior per chunk; an A/B comparison shows uplift on questions whose chunks have prior +/-1 ratings. |
| **Better citation rendering in CLI** | `cli.py:ask` | Citations print as raw chunk IDs. Better: paper title + section. | `assistant ask` resolves chunk IDs to "[<paper title>, §<section>]" via SQLite lookup. |

---

## Tracked requirements

Items explicitly discussed in the design brainstorm. Their status is recorded here so implemented work is not mistaken for backlog.

1. **Local application: built.** FastAPI + React/Vite provides saved fixed-scope chat, citation drawers, library/topic/inbox workflows, feedback, PDF/arXiv ingestion, and serialized background jobs. `assistant serve --build` hosts the API and static app.
2. **Deployment on a Raspberry Pi** (Pi 5 / 8GB, single-user). Architecture is already RPi-friendly (embedded Qdrant, SQLite, Ollama for local roles). Outstanding work:
   - **Heavy roles to API**: orchestrator / qa / curator_judge stay on Anthropic/OpenAI; only `embedder` (and optionally `summarizer`) stay local.
   - **Disable reranker** on RPi (CPU inference of bge-reranker-v2-m3 is too slow there).
   - **Packaging**: Dockerfile + `docker-compose.yml` with Ollama and the assistant; or systemd unit for the scheduler.
   - **Confirm Marker is feasible** on ARM; otherwise stick with PyMuPDF.
3. **Modular MCP extensibility: built at the platform level.** Any MCP server is a YAML edit in `mcp/client.py`. Likely additions in order:
   - Semantic Scholar MCP — citation graph, influential-citation counts.
   - Zotero MCP — pull user's existing library / push curated papers.
   - Obsidian MCP — write daily notes / paper summaries.
4. **Memory-first behavior: partially built.** The implemented seven-layer model covers working state, episodic interactions, semantic paper knowledge, procedural criteria, saved conversations, curation decisions, and profile facts. Versioned criteria and feedback capture are built; feedback-biased retrieval and consolidation remain pending.
5. **Local-vs-API choice via config.** Built. Per-role mapping is in place; switching any role is a single YAML line.
6. **Vanilla vs agentic RAG via config.** Built. Toggle is `rag.mode`.

---

## Good-to-haves (not yet committed)

The list of "yes, eventually" items that didn't make MVP. Treat as a backlog, not a roadmap.

### RAG / quality

- **Qdrant native sparse vectors** to replace in-memory BM25 (scales beyond a few thousand chunks).
- **HyDE-style query expansion** for very short questions.
- **Citation graph traversal** during retrieval (jump from a cited chunk to chunks of cited papers).
- **CLI domain-filtered retrieval** via `assistant ask --domain`; paper/topic scope already works in the web UI and both retrieval modes.
- **Per-section weighting** (e.g. prefer Method/Results over Related Work for technique questions).
- **Better summary generation**: today the paper-level vector is the abstract; a generated summary (via `summarizer` LLM) would be richer.

### Ingestion

- **Conference scrapers** beyond arxiv: OpenReview, ACL Anthology, ICML/NeurIPS proceedings pages.
- **Newsletter/RSS sources** (Sebastian Raschka, Lilian Weng, AK on X) — same `candidates` shape, different fetcher.
- **Durable background jobs** that survive a server restart; current web jobs are serialized and in memory.

### UX

- **CLI sessions** (`assistant chat`); saved web conversations and inline feedback are built.
- **`config set` CLI** so config tweaks don't require a YAML editor.
- **Better PDF browsing**: open the source PDF at the cited section.

### Ops / deployment

- **Dockerfile + compose** including Ollama.
- **Systemd unit** for the monitor scheduler.
- **Backup/export**: a single command that bundles SQLite + Qdrant + PDFs.
- **Migrations**: SQLAlchemy + Alembic for schema evolution (currently `create_all` on every startup).
- **Tracing**: LangSmith or local OTel for agent step-level visibility.

### Memory

- **Auto-promotion to profile facts** from repeated user behavior (consolidation job).
- **Feedback-biased ranking** (see pending list).
- **Conflict detection across paper claims** — flag chunks that contradict each other for the same question.
- **Forget interface**: `assistant forget <chunk-id-or-paper-id>` for things the user wants out of memory.

### Multi-agent

- **Replan loop**: an explicit re-plan node when QA confidence stays low even after MCP gathering.
- **Tool-use logging** to a separate `mcp_invocations` table.
- **Per-domain agent customization** (e.g. a domain can override the `qa` system prompt with a domain-specific preamble).

---

## Notes on resolved decisions (don't relitigate)

These came up in the brainstorm and were settled. Recording them here so future sessions don't reopen them without a reason.

- **LangGraph over LangChain agents / CrewAI / AutoGen.** Chosen for stateful graphs, checkpointers, and conditional edges that fit a multi-intent assistant.
- **Qdrant over Chroma / Weaviate.** Embedded mode, hybrid search, metadata filters, RPi-friendly.
- **PyMuPDF first, Marker later.** Lean install for MVP; clear upgrade path.
- **SQLAlchemy + SQLite over plain sqlite3.** Typed models pay off as schema grows; Alembic when migrations matter.
- **Typer + Rich over Click / argparse.** Better DX, less boilerplate.
- **MCP via `langchain-mcp-adapters`.** Lets MCP tools be bound to any LangChain LLM with `bind_tools`.
- **Per-role LLM mapping over single-global or two-tier.** User picked it explicitly.
- **LLM-judge curation with versioned criteria over embedding-similarity-to-seed-set.** User picked it explicitly.
- **CLI plus a localhost FastAPI/React app.** The web layer is deliberately single-user, one-process, and without authentication.
- **Single-user, no auth, no scalability concerns** for now.

---

## How to use this file

- When a question comes up like "should we add X now?" — check Good-to-haves first.
- When something looks half-done or stubbed — check Intentionally pending; it's probably deliberate.
- When you're about to assume something — check Assumptions and either confirm or update.
- When a design decision is made — add a row to "Resolved decisions" so it's not relitigated.
