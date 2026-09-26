import { useEffect, useState } from "react";
import {
  NavLink,
  Route,
  Routes,
  useNavigate,
  useSearchParams,
} from "react-router-dom";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import {
  Archive,
  BookOpen,
  Bot,
  Check,
  ChevronRight,
  CirclePlus,
  ExternalLink,
  FileText,
  Inbox,
  Library,
  LoaderCircle,
  MessageSquare,
  PanelLeft,
  Play,
  Search,
  Send,
  Settings2,
  ThumbsDown,
  ThumbsUp,
  Upload,
  X,
} from "lucide-react";
import {
  api,
  type Decision,
  type Conversation,
  type Paper,
  type Source,
  type Status,
  streamMessage,
  type Topic,
  type Turn,
  uploadPdf,
  write,
} from "./api";

const progressLabels: Record<string, string> = {
  retrieving: "Searching the library",
  answering: "Composing an answer",
  gathering: "Gathering external context",
};

function useLoad<T>(path: string, initial: T, refresh = 0, pollMs = 0) {
  const [data, setData] = useState(initial);
  const [error, setError] = useState("");
  useEffect(() => {
    const load = () =>
      api<T>(path)
        .then(setData)
        .catch((err: Error) => setError(err.message));
    load();
    const timer = pollMs ? window.setInterval(load, pollMs) : undefined;
    return () => {
      if (timer) window.clearInterval(timer);
    };
  }, [path, refresh, pollMs]);
  return { data, setData, error };
}

function SourceDrawer({
  source,
  close,
  chat,
}: {
  source?: Source;
  close: () => void;
  chat: (paperId: string) => void;
}) {
  if (!source) return null;
  return (
    <aside className="drawer source-drawer">
      <header>
        <div>
          <span className="eyebrow">Cited passage</span>
          <h2>{source.title}</h2>
        </div>
        <button className="icon-button" onClick={close} title="Close">
          <X />
        </button>
      </header>
      <p className="section-label">{source.section}</p>
      <blockquote>{source.text}</blockquote>
      <div className="drawer-actions">
        {source.arxiv_url && (
          <a
            className="button secondary"
            href={source.arxiv_url}
            target="_blank"
          >
            arXiv <ExternalLink />
          </a>
        )}
        {source.pdf_url && (
          <a className="button secondary" href={source.pdf_url} target="_blank">
            Open PDF <FileText />
          </a>
        )}
        <button className="button" onClick={() => chat(source.paper_id)}>
          Chat about paper <MessageSquare />
        </button>
      </div>
    </aside>
  );
}

function Answer({
  turn,
  onSource,
}: {
  turn: Turn;
  onSource: (source: Source) => void;
}) {
  const marked = turn.answer.replace(/\[(\d+)]/g, "[$1](citation:$1)");
  return (
    <article className="answer">
      <div className="answer-mark">
        <Bot />
      </div>
      <div className="answer-body">
        <ReactMarkdown
          urlTransform={(url) =>
            url.startsWith("citation:") ? url : defaultUrlTransform(url)
          }
          components={{
            a: ({ href, children }) =>
              href?.startsWith("citation:") ? (
                <button
                  className="citation-chip"
                  onClick={() => {
                    const source = turn.sources[Number(href.split(":")[1]) - 1];
                    if (source) onSource(source);
                  }}
                >
                  {children}
                </button>
              ) : (
                <a href={href} target="_blank">
                  {children}
                </a>
              ),
          }}
        >
          {marked}
        </ReactMarkdown>
        {turn.confidence !== undefined && turn.confidence < 0.5 && (
          <span className="low-confidence">Low confidence</span>
        )}
        {turn.id && (
          <div className="feedback">
            <button
              title="Helpful"
              onClick={() =>
                write(`/interactions/${turn.id}/feedback`, "POST", { score: 1 })
              }
            >
              <ThumbsUp />
            </button>
            <button
              title="Not helpful"
              onClick={() =>
                write(`/interactions/${turn.id}/feedback`, "POST", {
                  score: -1,
                })
              }
            >
              <ThumbsDown />
            </button>
          </div>
        )}
      </div>
    </article>
  );
}

function ChatPage() {
  const { data: conversations, setData: setConversations } = useLoad<
    Conversation[]
  >("/conversations", []);
  const { data: topics } = useLoad<Topic[]>("/topics", []);
  const { data: papers } = useLoad<Paper[]>("/papers", []);
  const [active, setActive] = useState<Conversation>();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [message, setMessage] = useState("");
  const [stage, setStage] = useState("");
  const [source, setSource] = useState<Source>();
  const [scope, setScope] = useState<"library" | "topic" | "paper">("library");
  const [target, setTarget] = useState("");
  const [searchParams] = useSearchParams();
  const requestedConversation = searchParams.get("conversation");
  useEffect(() => {
    if (requestedConversation)
      api<Conversation>(`/conversations/${requestedConversation}`).then(
        (loaded) => {
          setActive(loaded);
          setTurns(loaded.turns ?? []);
        },
      );
  }, [requestedConversation]);
  async function selectConversation(item: Conversation) {
    const loaded = await api<Conversation>(`/conversations/${item.id}`);
    setActive(loaded);
    setTurns(loaded.turns ?? []);
  }
  async function deleteChat(item: Conversation) {
    await write(`/conversations/${item.id}`, "DELETE");
    setConversations(
      conversations.filter((conversation) => conversation.id !== item.id),
    );
    if (active?.id === item.id) {
      setActive(undefined);
      setTurns([]);
    }
  }
  async function newChat(forPaper?: string) {
    const scopeType = forPaper ? "paper" : scope;
    const scopeTarget =
      forPaper || (scopeType === "library" ? undefined : target);
    const created = await write<Conversation>("/conversations", "POST", {
      title: "New conversation",
      scope_type: scopeType,
      scope_target: scopeTarget,
    });
    setConversations([created, ...conversations]);
    setActive(created);
    setTurns([]);
  }
  async function send() {
    if (!active || !message.trim() || stage) return;
    const question = message.trim();
    setMessage("");
    setTurns((old) => [
      ...old,
      { question, answer: "", citations: [], sources: [] },
    ]);
    setStage("retrieving");
    try {
      await streamMessage(active.id, question, (event) => {
        if (event.type === "progress") setStage(String(event.stage));
        if (event.type === "final") {
          const result = event.result as Record<string, unknown>;
          const turn: Turn = {
            id: result.interaction_id as number | undefined,
            question,
            answer: String(result.answer ?? ""),
            citations: (result.citations as string[]) ?? [],
            sources: (result.sources as Source[]) ?? [],
            confidence: Number(result.confidence ?? 0),
          };
          setTurns((old) => [...old.slice(0, -1), turn]);
          setStage("");
        }
        if (event.type === "error") {
          setTurns((old) => [
            ...old.slice(0, -1),
            {
              question,
              answer: String(event.message),
              citations: [],
              sources: [],
            },
          ]);
          setStage("");
        }
      });
    } catch (err) {
      setStage("");
      setTurns((old) => [
        ...old.slice(0, -1),
        {
          question,
          answer: (err as Error).message,
          citations: [],
          sources: [],
        },
      ]);
    }
  }
  return (
    <div className="chat-layout">
      <aside className="conversation-rail">
        <div className="new-chat-controls">
          <select
            value={scope}
            onChange={(e) => {
              setScope(e.target.value as typeof scope);
              setTarget("");
            }}
          >
            <option value="library">Whole library</option>
            <option value="topic">One topic</option>
            <option value="paper">One paper</option>
          </select>
          {scope === "topic" && (
            <select value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">Choose topic</option>
              {topics.map((t) => (
                <option key={t.name}>{t.name}</option>
              ))}
            </select>
          )}
          {scope === "paper" && (
            <select value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">Choose paper</option>
              {papers.map((p) => (
                <option value={p.id} key={p.id}>
                  {p.title}
                </option>
              ))}
            </select>
          )}
          <button
            className="button full"
            onClick={() => newChat()}
            disabled={scope !== "library" && !target}
          >
            <CirclePlus /> New chat
          </button>
        </div>
        <div className="conversation-list">
          {conversations.map((item) => (
            <div className="conversation-item" key={item.id}>
              <button
                className={active?.id === item.id ? "active" : ""}
                onClick={() => selectConversation(item)}
              >
                <MessageSquare />
                <span>{item.title}</span>
                <ChevronRight />
              </button>
              <button
                className="delete-chat"
                onClick={() => deleteChat(item)}
                title="Delete conversation"
              >
                <X />
              </button>
            </div>
          ))}
        </div>
      </aside>
      <section className="chat-main">
        {!active ? (
          <div className="empty-state">
            <div className="empty-glyph">
              <MessageSquare />
            </div>
            <h1>Ask across your research</h1>
            <p>
              Start a scoped conversation with your whole library, one topic, or
              a single paper.
            </p>
          </div>
        ) : (
          <>
            <header className="page-header compact">
              <div>
                <span className="eyebrow">
                  {active.scope_type}
                  {active.scope_target ? ` / ${active.scope_target}` : ""}
                </span>
                <h1>{active.title}</h1>
              </div>
            </header>
            <div className="transcript">
              {turns.map((turn, index) => (
                <div key={index}>
                  <div className="question">{turn.question}</div>
                  {turn.answer ? (
                    <Answer turn={turn} onSource={setSource} />
                  ) : (
                    <div className="progress">
                      <LoaderCircle className="spin" />{" "}
                      {progressLabels[stage] ?? stage}
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div className="composer">
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                placeholder="Ask a research question..."
              />
              <button
                className="send-button"
                onClick={send}
                disabled={!message.trim() || !!stage}
                title="Send"
              >
                <Send />
              </button>
            </div>
          </>
        )}
      </section>
      <SourceDrawer
        source={source}
        close={() => setSource(undefined)}
        chat={(paperId) => {
          newChat(paperId);
          setSource(undefined);
        }}
      />
    </div>
  );
}

function LibraryPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [query, setQuery] = useState("");
  const [topic, setTopic] = useState(searchParams.get("domain") ?? "");
  const [refresh, setRefresh] = useState(0);
  const { data: topics } = useLoad<Topic[]>("/topics", [], refresh);
  const { data: papers } = useLoad<Paper[]>(
    `/papers?q=${encodeURIComponent(query)}&domain=${encodeURIComponent(topic)}`,
    [],
    refresh,
  );
  const [selected, setSelected] = useState<Paper>();
  const [adding, setAdding] = useState(false);
  const [arxiv, setArxiv] = useState("");
  const [addTopic, setAddTopic] = useState("");
  const [file, setFile] = useState<File>();
  async function addPaper() {
    if (file) await uploadPdf(file, addTopic || undefined);
    else
      await write("/papers/arxiv", "POST", {
        arxiv_id: arxiv,
        domain: addTopic || undefined,
      });
    setAdding(false);
    setRefresh((x) => x + 1);
  }
  async function chatAboutPaper(paper: Paper) {
    const conversation = await write<Conversation>("/conversations", "POST", {
      title: paper.title,
      scope_type: "paper",
      scope_target: paper.id,
    });
    navigate(`/chat?conversation=${conversation.id}`);
  }
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Knowledge base</span>
          <h1>Library</h1>
          <p>{papers.length} papers ready for retrieval</p>
        </div>
        <button className="button" onClick={() => setAdding(true)}>
          <CirclePlus /> Add paper
        </button>
      </header>
      <div className="toolbar">
        <label className="search">
          <Search />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search title or arXiv ID"
          />
        </label>
        <select value={topic} onChange={(e) => setTopic(e.target.value)}>
          <option value="">All topics</option>
          {topics.map((item) => (
            <option key={item.name}>{item.name}</option>
          ))}
        </select>
      </div>
      <div className="paper-table">
        <div className="table-head">
          <span>Paper</span>
          <span>Topic</span>
          <span>Year</span>
          <span />
        </div>
        {papers.map((paper) => (
          <button
            className="paper-row"
            key={paper.id}
            onClick={async () =>
              setSelected(await api<Paper>(`/papers/${paper.id}`))
            }
          >
            <span>
              <strong>{paper.title}</strong>
              <small>
                {paper.authors.slice(0, 3).join(", ") || paper.arxiv_id}
              </small>
            </span>
            <span>
              <i className="topic-dot" />
              {paper.domain || "Unsorted"}
            </span>
            <span>{paper.year || "—"}</span>
            <ChevronRight />
          </button>
        ))}
        {papers.length === 0 && (
          <div className="data-empty">
            No papers match this view. Add an arXiv ID or upload a PDF.
          </div>
        )}
      </div>
      {selected && (
        <aside className="drawer">
          <header>
            <div>
              <span className="eyebrow">{selected.arxiv_id}</span>
              <h2>{selected.title}</h2>
            </div>
            <button
              className="icon-button"
              onClick={() => setSelected(undefined)}
            >
              <X />
            </button>
          </header>
          <p className="authors">{selected.authors.join(", ")}</p>
          <p>{selected.abstract}</p>
          <label>
            Topic
            <select
              value={selected.domain || ""}
              onChange={async (e) => {
                const updated = await write<Paper>(
                  `/papers/${selected.id}`,
                  "PATCH",
                  { domain: e.target.value || null },
                );
                setSelected(updated);
                setRefresh((x) => x + 1);
              }}
            >
              <option value="">Unsorted</option>
              {topics.map((item) => (
                <option key={item.name}>{item.name}</option>
              ))}
            </select>
          </label>
          <div className="drawer-actions">
            {selected.has_pdf && (
              <a
                className="button secondary"
                href={`/api/papers/${selected.id}/pdf`}
                target="_blank"
              >
                Open PDF <FileText />
              </a>
            )}
            {selected.arxiv_id && (
              <a
                className="button secondary"
                href={`https://arxiv.org/abs/${selected.arxiv_id}`}
                target="_blank"
              >
                arXiv <ExternalLink />
              </a>
            )}
            <button className="button" onClick={() => chatAboutPaper(selected)}>
              Chat about this paper <MessageSquare />
            </button>
          </div>
          <h3>Sections</h3>
          <div className="sections">
            {selected.sections?.map((section) => (
              <details key={section.id}>
                <summary>{section.title}</summary>
                <p>{section.text}</p>
              </details>
            ))}
          </div>
        </aside>
      )}
      {adding && (
        <div className="modal-backdrop">
          <form
            className="modal"
            onSubmit={(e) => {
              e.preventDefault();
              addPaper();
            }}
          >
            <header>
              <h2>Add to library</h2>
              <button
                type="button"
                className="icon-button"
                onClick={() => setAdding(false)}
              >
                <X />
              </button>
            </header>
            <label>
              arXiv ID
              <input
                value={arxiv}
                onChange={(e) => setArxiv(e.target.value)}
                placeholder="2401.05566"
                disabled={!!file}
              />
            </label>
            <div className="divider">or</div>
            <label className="upload-field">
              <Upload />
              <span>{file?.name || "Choose a PDF up to 50 MB"}</span>
              <input
                type="file"
                accept="application/pdf"
                onChange={(e) => setFile(e.target.files?.[0])}
              />
            </label>
            <label>
              Topic
              <select
                value={addTopic}
                onChange={(e) => setAddTopic(e.target.value)}
              >
                <option value="">Unsorted</option>
                {topics.map((item) => (
                  <option key={item.name}>{item.name}</option>
                ))}
              </select>
            </label>
            <button className="button full" disabled={!arxiv && !file}>
              Queue ingestion
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

function TopicsPage() {
  const navigate = useNavigate();
  const [refresh, setRefresh] = useState(0);
  const { data: topics } = useLoad<Topic[]>("/topics", [], refresh);
  const [commenting, setCommenting] = useState<Topic>();
  const [comment, setComment] = useState("");
  async function chatAboutTopic(topic: Topic) {
    const conversation = await write<Conversation>("/conversations", "POST", {
      title: topic.name,
      scope_type: "topic",
      scope_target: topic.name,
    });
    navigate(`/chat?conversation=${conversation.id}`);
  }
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Research map</span>
          <h1>Topics</h1>
          <p>Config-defined areas and their active curation criteria</p>
        </div>
        <button
          className="button secondary"
          onClick={async () => {
            await write("/topics/sync", "POST");
            setRefresh((x) => x + 1);
          }}
        >
          <Settings2 /> Sync from config
        </button>
      </header>
      <div className="topic-grid">
        {topics.map((topic) => (
          <article className="topic-card" key={topic.name}>
            <header>
              <div>
                <h2>{topic.name}</h2>
                <span className={topic.synced ? "status ready" : "status"}>
                  {topic.synced ? "Synced" : "Not synced"}
                </span>
              </div>
              <strong>
                {topic.paper_count}
                <small> papers</small>
              </strong>
            </header>
            <p>
              {topic.criteria?.nl_addendum ||
                "No natural-language curation guidance yet."}
            </p>
            <div className="tag-row">
              {topic.arxiv_categories.map((item) => (
                <span key={item}>{item}</span>
              ))}
              {topic.venues.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
            <footer>
              <span>Criteria v{topic.criteria?.version ?? 0}</span>
              <div>
                <button
                  title="Chat about topic"
                  onClick={() => chatAboutTopic(topic)}
                >
                  <MessageSquare />
                </button>
                <button
                  title="View papers"
                  onClick={() =>
                    navigate(
                      `/library?domain=${encodeURIComponent(topic.name)}`,
                    )
                  }
                >
                  <Library />
                </button>
                <button
                  title="Run monitor"
                  onClick={() => write(`/topics/${topic.name}/monitor`, "POST")}
                >
                  <Play />
                </button>
                <button
                  title="Refine criteria"
                  onClick={() => {
                    setCommenting(topic);
                    setComment("");
                  }}
                >
                  <Settings2 />
                </button>
              </div>
            </footer>
          </article>
        ))}
        {topics.length === 0 && (
          <div className="data-empty">
            No topics are configured. Add one under domains in config.yaml, then
            sync.
          </div>
        )}
      </div>
      {commenting && (
        <div className="modal-backdrop">
          <form
            className="modal"
            onSubmit={async (e) => {
              e.preventDefault();
              await write(`/topics/${commenting.name}/criteria`, "POST", {
                comment,
              });
              setCommenting(undefined);
              setRefresh((x) => x + 1);
            }}
          >
            <header>
              <div>
                <span className="eyebrow">Refine criteria</span>
                <h2>{commenting.name}</h2>
              </div>
              <button
                type="button"
                className="icon-button"
                onClick={() => setCommenting(undefined)}
              >
                <X />
              </button>
            </header>
            <label>
              What should the curator learn?
              <textarea
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                rows={6}
              />
            </label>
            <button className="button full" disabled={!comment.trim()}>
              Create criteria version
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

function InboxPage() {
  const [topic, setTopic] = useState("");
  const [status, setStatus] = useState("");
  const [refresh, setRefresh] = useState(0);
  const { data: topics } = useLoad<Topic[]>("/topics", []);
  const { data: decisions } = useLoad<Decision[]>(
    `/inbox?domain=${encodeURIComponent(topic)}&status=${encodeURIComponent(status)}`,
    [],
    refresh,
  );
  const [disagree, setDisagree] = useState<Decision>();
  const [comment, setComment] = useState("");
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Curation review</span>
          <h1>Inbox</h1>
          <p>Review what the curator accepted, rejected, or failed to ingest</p>
        </div>
      </header>
      <div className="toolbar">
        <select value={topic} onChange={(e) => setTopic(e.target.value)}>
          <option value="">All topics</option>
          {topics.map((item) => (
            <option key={item.name}>{item.name}</option>
          ))}
        </select>
        <select value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="">All statuses</option>
          {["accepted", "rejected", "ingested", "failed", "dismissed"].map(
            (item) => (
              <option key={item}>{item}</option>
            ),
          )}
        </select>
      </div>
      <div className="inbox-list">
        {decisions.map((item) => (
          <article className="decision" key={item.id}>
            <div className="decision-score">
              <strong>{Math.round(item.score * 100)}</strong>
              <span>%</span>
              <i
                style={
                  { "--score": `${item.score * 100}%` } as React.CSSProperties
                }
              />
            </div>
            <div className="decision-body">
              <header>
                <div>
                  <span className="eyebrow">
                    {item.domain} / {item.arxiv_id}
                  </span>
                  <h2>{item.title}</h2>
                </div>
                <span className={`status ${item.status}`}>{item.status}</span>
              </header>
              <p>{item.judge_reason}</p>
              {item.error && <p className="error">{item.error}</p>}
              <footer>
                <span>{item.authors.slice(0, 3).join(", ")}</span>
                <div>
                  {["rejected", "failed"].includes(item.status) && (
                    <button
                      className="button small"
                      onClick={async () => {
                        await write(`/inbox/${item.id}/ingest`, "POST");
                        setRefresh((x) => x + 1);
                      }}
                    >
                      <Archive /> Ingest
                    </button>
                  )}
                  <button
                    className="button secondary small"
                    onClick={async () => {
                      await write(`/inbox/${item.id}/dismiss`, "POST");
                      setRefresh((x) => x + 1);
                    }}
                  >
                    <Check /> Dismiss
                  </button>
                  <button
                    className="button ghost small"
                    onClick={() => {
                      setDisagree(item);
                      setComment(`For ${item.title}: `);
                    }}
                  >
                    Disagree
                  </button>
                </div>
              </footer>
            </div>
          </article>
        ))}
        {decisions.length === 0 && (
          <div className="data-empty">
            No curation decisions match these filters. Run a topic monitor to
            fill the inbox.
          </div>
        )}
      </div>
      {disagree && (
        <div className="modal-backdrop">
          <form
            className="modal"
            onSubmit={async (e) => {
              e.preventDefault();
              await write(`/inbox/${disagree.id}/disagree`, "POST", {
                comment,
              });
              setDisagree(undefined);
              setRefresh((x) => x + 1);
            }}
          >
            <header>
              <h2>Teach the curator</h2>
              <button
                type="button"
                className="icon-button"
                onClick={() => setDisagree(undefined)}
              >
                <X />
              </button>
            </header>
            <label>
              Criteria comment
              <textarea
                rows={6}
                value={comment}
                onChange={(e) => setComment(e.target.value)}
              />
            </label>
            <button className="button full">Update criteria</button>
          </form>
        </div>
      )}
    </div>
  );
}

function Shell() {
  const { data: status } = useLoad<Status>(
    "/status",
    { paper_count: 0, rag_mode: "—", roles: {} },
    0,
    5000,
  );
  const { data: jobs } = useLoad<{ status: string }[]>("/jobs", [], 0, 2000);
  const [open, setOpen] = useState(false);
  const activeJobs = jobs.filter((job) =>
    ["queued", "running"].includes(job.status),
  ).length;
  const nav = [
    { to: "/chat", label: "Chat", icon: MessageSquare },
    { to: "/library", label: "Library", icon: Library },
    { to: "/topics", label: "Topics", icon: BookOpen },
    { to: "/inbox", label: "Inbox", icon: Inbox },
  ];
  return (
    <div className="shell">
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">
            <Archive />
          </div>
          <div>
            <strong>Fieldnotes</strong>
            <span>research assistant</span>
          </div>
        </div>
        <nav>
          {nav.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} onClick={() => setOpen(false)}>
              <Icon />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <footer>
          <div className="system-line">
            <span className={activeJobs ? "pulse" : ""} />
            {activeJobs ? `${activeJobs} job running` : "System ready"}
          </div>
          <dl>
            <div>
              <dt>Papers</dt>
              <dd>{status.paper_count}</dd>
            </div>
            <div>
              <dt>RAG</dt>
              <dd>{status.rag_mode}</dd>
            </div>
            <div>
              <dt>QA</dt>
              <dd title={status.roles.qa?.model}>
                {status.roles.qa?.model?.split("/").pop() || "—"}
              </dd>
            </div>
          </dl>
        </footer>
      </aside>
      <button className="mobile-menu" onClick={() => setOpen(!open)}>
        <PanelLeft />
      </button>
      <main>
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/library" element={<LibraryPage />} />
          <Route path="/topics" element={<TopicsPage />} />
          <Route path="/inbox" element={<InboxPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default Shell;
