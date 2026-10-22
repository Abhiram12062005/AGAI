# Build Prompt — CodeFix AI

> Paste everything below the line into a fresh Claude Code session with
> `C:\Users\abhir\Documents\sem-7\AGAI` as the working directory.

---

Build **CodeFix AI**, an autonomous multi-agent bug-resolution application, in this
directory (`AGAI/`, currently empty). This is a graded Agentic AI course project, so the
architecture is not negotiable — it must visibly match the spec below, because every
design decision has to be defensible with run data.

**Core user story:** I paste a bug report (plus a path to a Git repo), and the system
locates the buggy code, explains the root cause, writes a patch, runs the project's test
suite to prove the patch works, and then offers to open a Pull Request. If the tests
fail, it retries with the failure log attached — it does not restart from scratch.

## 1. Architecture — hierarchical multi-agent (do not flatten this into one agent)

A **Manager/Orchestrator** agent delegates to three specialised sub-agents and
re-delegates on failure:

- **Diagnoser** — ReAct loop (reason ↔ tool call) over the bug report, stack trace and
  repo until it converges on a root cause. Emits a `Diagnosis`.
- **Fixer** — generates a unified-diff patch + rationale from the `Diagnosis`. Emits a `Patch`.
- **Tester/PR** — applies the patch in an isolated copy, runs lint + the test suite,
  emits a `TestReport`; on Manager approval, commits to a branch and opens the PR.

Manager control flow: on `TestReport.passed` → dispatch PR agent. On failure →
re-dispatch **Fixer only**, with the `TestReport` attached, max **3** attempts
(configurable), then **escalate to a human** with a written summary of what was tried.
Never loop unbounded.

Implement this with **LangGraph** — real nodes and *conditional edges*, so the
re-delegation loop is a visible cycle in the compiled graph, not an `if` inside a Python
`while`. Ship a `graph.get_graph().draw_mermaid_png()` export as `docs/graph.png`.

Also implement two **comparison modes** behind the same interface, because Phase 4
benchmarks all three on the same bug set:
- `sequential` — straight Diagnose → Fix → Test → PR pipeline; any failure restarts the whole run.
- `p2p` — agents hand state directly to each other, no blackboard.
- `hierarchical` — the real design, default.

Select with `--arch hierarchical|sequential|p2p`.

## 2. Messaging — MCP + shared Blackboard

**Blackboard**: single source of truth for pipeline state, SQLite-backed (`runs.db`),
keyed by `task_id`. Holds the bug report, current `Diagnosis`, active `Patch`, latest
`TestReport`, attempt counter, and the full append-only message log. Agents read/write
*only* through it in hierarchical mode. The Manager must be able to dump complete
pipeline state at any instant — that dump is the observability audit point.

Every inter-agent exchange is a typed envelope written to the Blackboard, never free
text passed directly:

```python
class Envelope(BaseModel):
    task_id: str
    sender: str
    target_agent: str
    status: Literal["dispatched", "in_progress", "success", "failure"]
    payload: Diagnosis | Patch | TestReport | DispatchRef | Escalation
    confidence: float          # 0.0-1.0, self-reported by the sending agent
    timestamp: datetime
    latency_ms: int | None     # filled on completion
```

Pydantic models, strictly typed per stage:
- `Diagnosis{root_cause, implicated_files[], implicated_symbols[], evidence[], confidence}`
- `Patch{diff, rationale, files_touched[], attempt_no}`
- `TestReport{passed, total, failed, failing_assertions[], stack_trace, lint_findings[], duration_ms}`

Manager → agent dispatch references prior artifacts **by `task_id` + artifact id**, not by
re-sending full content (message size must stay bounded — keep that as a code comment,
it is a marked design point in the synopsis).

**MCP**: expose all tool access through a real MCP server (`mcp` Python SDK / FastMCP,
stdio transport) in `mcp_server/`, so every agent calls tools uniformly regardless of
backing model. Tools:
- `code_search(query, k)` — hybrid: ripgrep literal + vector search
- `read_file(path, start, end)`, `list_tree(path)`
- `parse_stacktrace(text)` → structured frames mapped to repo files
- `static_analysis(paths)` — ruff / eslint depending on project type
- `run_tests(path, selector?)` — pytest / jest auto-detected, subprocess, hard timeout
- `apply_patch(diff)`, `git_branch`, `git_commit`, `open_pull_request` (GitHub via `gh` CLI or PyGithub)
- `rag_retrieve(query, k, collection)`

Write a thin client so the LangGraph nodes bind MCP tools as LangChain tools.

## 3. Model selection — task–model fit, configurable

One model per role, justified by role, in `config/models.yaml`:

| Agent | Model | Why |
|---|---|---|
| Manager | `claude-haiku-5-5` | routing + state evaluation only; cheapest/fastest, does not need frontier capability |
| Diagnoser | `claude-opus-5-5` | multi-step reasoning over traces and logs |
| Fixer | `claude-sonnet-5-5` | code generation / editing |
| Judge | `claude-fable-5-1` | **must differ from both generating agents** to avoid self-preference bias |

Use the Anthropic Python SDK (`anthropic` / `langchain-anthropic`), key from
`ANTHROPIC_API_KEY` in `.env`. Make every role swappable from the YAML with no code
change, and log model + token usage + cost per call — Phase 1 has to report the
comparison that justified each pick.

## 4. Prompting + RAG

- **Diagnoser: explicit ReAct.** Thought → Action → Observation transcript, persisted
  verbatim into the trace (it is a deliverable, not debug output). Cap iterations (~8) and
  degrade gracefully to a best-effort diagnosis.
- **RAG** for Diagnoser and Fixer, over three collections: (a) repo code, AST/function-chunked
  with file+line metadata, (b) past bugs and their accepted fixes from this project's own run
  history, (c) project docs/README. Use ChromaDB + `sentence-transformers`
  (`all-MiniLM-L6-v2`) so no second API key is needed; keep the embedder behind an interface.
  Incremental re-index on file mtime, via a `codefix index <repo>` CLI command.
- Every generation must cite the retrieved chunks it used, and those citations surface in the UI.

## 5. Safety (hard requirements — graded demo runs on real repos)

- Never mutate the target repo in place. Copy to a Git worktree / temp clone per attempt,
  patch and test there, discard on failure.
- `run_tests` in a subprocess with a timeout and captured output; never shell-interpolate
  model output.
- Never `git push` or open a PR without explicit human confirmation in the UI/CLI, and never
  to the default branch — always a new `codefix/<task_id>` branch.
- Refuse to touch paths outside the repo root; blocklist `.env`, `.git/config`, CI secrets.

## 6. Observability — a graded chapter, build it first-class

- Structured JSONL trace per run in `traces/<task_id>.jsonl`: every envelope, every tool
  call, every LLM call with model/tokens/cost, per-agent latency, and the latency delta
  between messages.
- `runs.db` tables: `runs`, `messages`, `llm_calls`, `tool_calls`, `evaluations`.
- Optional LangSmith export behind an env flag (`LANGCHAIN_TRACING_V2`).
- A **dashboard** page: run list, per-run timeline (Gantt-style per agent), per-agent mean
  latency, attempt counts, failure-stage breakdown, cost per run, judge accuracy. It must
  answer "which agent is the bottleneck / accuracy weak point" at a glance.

## 7. Evaluation harness

- `benchmarks/bugset/` — a seeded bug set: a small sample Python repo (and one JS repo) with
  N reproducible injected bugs, each with a bug report, the failing test, and the known-good
  fix. Enough bugs × architectures to produce **40+ logged runs**.
- **LLM-as-judge** (judge model, distinct from the generators) scores each run: does the patch
  actually resolve the reported bug? Output `{verdict: pass|fail|partial, reasoning, score}`.
  Also report the objective signal (did the previously-failing test go green) alongside the
  judge verdict, and show judge-vs-tests agreement.
- `codefix bench --arch all --runs N` → CSV + a markdown report comparing hierarchical vs
  sequential vs p2p on judge accuracy, per-agent latency, retries, and cost.

## 8. Application surface

1. **CLI** (Typer): `codefix fix "<bug report>" --repo <path>`, `codefix index`,
   `codefix bench`, `codefix trace <task_id>`, `codefix serve`.
2. **Web app** — FastAPI backend + a UI (React or plain HTML/HTMX, your call; prefer the
   simpler thing that still looks good). Pages:
   - **New fix**: bug report textarea, repo path/URL, arch + retry-limit selectors, Run.
   - **Live run view**: the LangGraph pipeline drawn as nodes lighting up by status, the
     ReAct transcript streaming (SSE/WebSocket), the diff with syntax highlighting, test
     output, and on success an "Open Pull Request" button (explicit confirm).
   - **Dashboard**: as in §6.
   - **Runs**: history, filter by outcome, drill into any trace.

Make the demo path obvious and fast — this gets presented live.

## 9. Repo layout and quality

```
AGAI/
  codefix/{agents,graph,blackboard,rag,observability,eval,cli}/
  mcp_server/
  web/{api,ui}/
  config/models.yaml
  benchmarks/bugset/
  tests/
  docs/{architecture.md,graph.png,DEMO.md}
  .env.example  README.md  pyproject.toml
```

- Python 3.11+, `uv` or plain `pip`, pinned deps, `ruff` + `pytest` clean.
- Unit tests for the blackboard, envelope validation, retry/escalation logic, patch
  application and the MCP tool layer. Mock the LLM in tests — the suite must pass with
  no API key.
- `README.md` with a 5-minute quickstart; `docs/architecture.md` recording *why*
  hierarchical over sequential/event-driven, MCP+blackboard over P2P, and the per-role
  model choice — mirroring the synopsis so it can be lifted into the final report.

## 10. How to proceed

Build in this order, and make each stage runnable before moving on:

1. Scaffold, config, blackboard + envelope schemas, structured logging, tests.
2. MCP server + tool implementations, exercised standalone.
3. Diagnoser (ReAct) → Fixer → Tester as individual agents, callable one at a time.
4. LangGraph hierarchical graph with the re-delegation cycle + escalation; then the
   sequential and p2p modes.
5. RAG indexing + retrieval wired into Diagnoser/Fixer.
6. CLI end-to-end on one seeded bug.
7. Web app + live run view + dashboard.
8. Git/PR automation behind confirmation.
9. Bug set, judge, benchmark harness; generate the 40+ runs and the comparison report.

Start by writing a short plan of the files you will create, then build. Ask me only about
decisions that would change the architecture — pick sensible defaults for everything else
and tell me what you picked. When a stage is done, show me the command to run it.
