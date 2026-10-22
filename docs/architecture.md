# CodeFix AI — Architecture & design rationale

> Written so sections can be lifted into the final report. Every claim below is backed by
> code (file references) or by run data (`runs.db`, `traces/*.jsonl`, `benchmarks/results/*`).

## 1. Problem and user story

A developer pastes a bug report (optionally a stack trace) and points at a Git repo. The
system **locates** the defect, **explains** the root cause, **writes** a patch, **proves** it
with the project's own test suite, and **offers** a Pull Request. A failing verification does
not restart the job: the failure log is sent back to the agent that wrote the patch.

## 2. Architecture: hierarchical multi-agent (default)

![compiled LangGraph](graph.png)

The graph above is exported straight from the compiled LangGraph (`codefix graph`), so the
diagram cannot drift from the code (`codefix/graph/hierarchical.py`).

| Node | Role | Model | Output |
|---|---|---|---|
| `manager` | Orchestrator: routing + state evaluation only | `claude-haiku-5-5` | `DispatchRef` envelopes, `Escalation` |
| `diagnoser` | Explicit ReAct over report, trace, repo | `claude-opus-5-5` | `Diagnosis` |
| `fixer` | Patch generation from the diagnosis | `claude-sonnet-5-5` | `Patch` (unified diff + rationale) |
| `tester` | Isolated apply → lint → test suite | deterministic (no LLM) | `TestReport` |
| `pr_prepare` / `open_pr` | Branch + commit; push + PR after human confirmation | deterministic | `PRResult` |
| `escalate` | Hand-off note to a human | manager model | `Escalation` |

**Control flow (Manager):** `diagnosis_ready → fixer` (or one re-diagnosis if the diagnosis is
weak) → `tester` → on `passed` → `pr_prepare` → **interrupt** (human gate) → `open_pr`. On
failure → `fixer` again **with the TestReport attached by reference**, at most `max_attempts`
(default 3, `config/models.yaml` / `--max-attempts` / UI selector), then `escalate` with a
written summary of everything tried.

**The loop is a real cycle in the compiled graph** (`manager ⇄ fixer`, `manager ⇄ tester`)
expressed with `add_conditional_edges`, not a Python `while`. It is bounded in code, not by
the model: the Manager node checks `attempts >= max_attempts` before consulting its LLM, and
the LLM may only choose among the *legal* transitions for that decision point
(`codefix/agents/manager.py`). An invalid or failed routing answer falls back to the
deterministic policy, so a bad model output cannot create an unbounded loop
(`tests/test_graph_retry.py::test_manager_llm_invalid_decision_falls_back`).

### Why hierarchical rather than sequential or event-driven?

| Concern | Sequential pipeline | Event-driven / P2P | **Hierarchical (chosen)** |
|---|---|---|---|
| Failure handling | restart from scratch; re-pays the diagnosis and loses the failure log | each peer must know the retry rules; no global view | one place decides; retries only the failed stage, **with** the failure evidence |
| Bounded execution | trivially bounded but wasteful | needs distributed counters/timeouts | single attempt counter on the Blackboard, checked by the Manager |
| Observability | linear log | causality spread across peers | every hop is Manager ⇄ worker: a star-shaped, auditable message log |
| Cost of coordination | none | none | one cheap Haiku call per decision point (measured in the dashboard as "Manager (LLM time)") |
| Extensibility | reorder code | rewire peers | add a node + a Manager transition |

Both alternatives are implemented behind the same interface so the claim is testable, not
rhetorical (`--arch sequential|p2p`, `codefix bench --arch all`):

- **sequential** (`codefix/graph/sequential.py`): Diagnose → Fix → Test → PR; any failure
  restarts the whole run from diagnosis, carrying nothing over.
- **p2p** (`codefix/graph/p2p.py`): Diagnoser → Fixer → Tester; on failure the Tester hands
  its full TestReport straight back to the Fixer. No Manager, no Blackboard reads; each peer
  routes with a fixed local rule (no re-diagnosis, no early escalation).

## 3. Messaging: MCP for tools, Blackboard for state

### Blackboard (`codefix/blackboard/`)

A SQLite-backed (`runs.db`) single source of truth keyed by `task_id`: bug report, current
`Diagnosis`, active `Patch`, latest `TestReport`, attempt counter, escalation/PR results and
the **append-only message log**. In hierarchical mode agents read and write *only* through
it. `Blackboard.snapshot(task_id)` dumps the complete pipeline state at any instant; it is
the observability audit point (`codefix trace <id> --json`, "state dump" in the UI).

Every inter-agent exchange is a typed, validated envelope (`codefix/blackboard/schemas.py`):

```python
class Envelope(BaseModel):
    task_id: str
    sender: str
    target_agent: str
    status: Literal["dispatched", "in_progress", "success", "failure"]
    payload: Diagnosis | Patch | TestReport | DispatchRef | Escalation | PRResult  # discriminated by `kind`
    confidence: float          # 0.0-1.0, self-reported by the sending agent
    timestamp: datetime
    latency_ms: int | None     # filled on completion: time since the hand-off to this agent
```

Payloads are strict (`extra="forbid"`, bounded floats) and re-validated at the Blackboard
boundary (`tests/test_envelope.py`).

**Design point: bounded message size.** A Manager → agent dispatch references prior
artifacts **by `task_id` + artifact id** (`DispatchRef.artifact_refs`), never by re-sending
their content. A third-attempt dispatch is the same size as the first, while the P2P
hand-off carries the full TestReport inline. The benchmark report measures this as "Mean
envelope size (bytes)" per architecture.

### Why MCP + Blackboard rather than direct P2P hand-offs?

- **Uniform tool access.** All tools sit behind one MCP server (`mcp_server/`, stdio
  transport, `MCPServer` = the mcp 2.x name for FastMCP). Any agent, backed by any model,
  calls `code_search`, `read_file`, `run_tests`, … the same way. The LangGraph side binds
  them as LangChain `StructuredTool`s through a thin client (`codefix/mcp_client.py`). Swapping
  a model never touches tool code.
- **Safety in one place.** Path sandboxing, the secrets blocklist, workspace isolation, the
  timeouts and the PR confirmation guard live in the server, so *no* agent can bypass them.
- **Decoupled state.** Peers never need each other's memory; anything an agent needs is on
  the Blackboard. That is what makes re-dispatching a single stage (Fixer only) possible, and
  what makes a run resumable after the human PR gate (LangGraph `SqliteSaver` checkpoint).
- **Auditability.** The Blackboard log is total and ordered; P2P spreads causality across
  peers.

### MCP tools

| Tool | Notes |
|---|---|
| `code_search(query, k)` | hybrid: ripgrep literal + vector search over AST chunks, fused with reciprocal-rank fusion |
| `read_file(path, start, end)`, `list_tree(path)` | sandboxed to the repo root, blocklist enforced |
| `parse_stacktrace(text)` | Python / pytest / JS frames, mapped to repo files (innermost repo frame) |
| `static_analysis(paths)` | ruff (`E9,F`) or eslint, falling back to `node --check`; only *new* error-class findings gate |
| `run_tests(workspace, selector?)` | pytest / jest / node:test auto-detected; subprocess, hard timeout, JUnit-parsed |
| `create_workspace`, `apply_patch(diff)`, `git_branch`, `git_commit` | per-attempt isolated clone |
| `open_pull_request` | refuses unless `confirmed=true` and the branch is `codefix/*`; never the default branch |
| `rag_retrieve(query, k, collection)`, `index_repo`, `record_fix` | RAG over code / bugs / docs |

## 4. Model selection (task–model fit)

| Agent | Model | Why |
|---|---|---|
| Manager | `claude-haiku-5-5` (effort low) | chooses among 2–3 legal transitions from a compact state summary. Frontier capability is wasted here, and it runs at every decision point, so cost and latency dominate |
| Diagnoser | `claude-opus-5-5` (effort high, thinking summaries kept) | multi-step reasoning over traces, logs and code, where localisation errors are the most expensive downstream |
| Fixer | `claude-sonnet-5-5` (effort medium) | code editing once the defect is localised; strong coding at half Opus's price |
| Judge | `claude-fable-5-1` | **must differ from both generators** to avoid self-preference bias (enforced in `codefix/config.py`) |

Every role is swappable in `config/models.yaml` with no code change. Every call logs the
requested model, the served model (refusal fallbacks are on for Opus/Sonnet/Fable),
input/output/cache tokens, USD cost and latency (`llm_calls` table, `llm_call` trace
events, dashboard "Spend by agent / model"). To produce the Phase 1 comparison table,
re-run `codefix bench` with a role swapped in the YAML and compare the two reports.

## 5. Prompting and RAG

**Diagnoser: explicit ReAct** (`codefix/agents/diagnoser.py`). Each turn is a structured
output `{thought, action, action_input, final_answer}`; the action is executed through MCP
and the observation appended. The transcript is persisted verbatim as `react_step` trace
events and as `traces/<task_id>.react.md`. Iterations are capped (8); at the cap the model is
forced to answer, and failing that a best-effort diagnosis (`degraded=true`, reduced
confidence) is assembled from the gathered evidence. The Manager treats a degraded or
low-confidence diagnosis as weak and may request one re-diagnosis.

**Fixer: anchored edits → deterministic diff.** The model emits exact `find/replace` edits
and the unified diff is derived with `difflib` (`codefix/agents/patching.py`). LLM-written
hunks with hand-computed line numbers are the most common cause of "patch does not apply";
deriving the diff removes that failure class while still storing a standard unified diff.
Patches touching tests are rejected (`allow_test_edits: false`).

**RAG** (`codefix/rag/`): ChromaDB + `sentence-transformers/all-MiniLM-L6-v2` (no second API
key; the embedder sits behind an interface, with a hashing fallback). Three collections per repo:
(a) **code**: AST/function-level chunks with file and line metadata (Python `ast`;
brace-matched JS/TS); (b) **bugs**: past bugs plus their accepted fixes from this project's
own run history (written when a fix passes); (c) **docs**: README and `docs/`. Indexing is
incremental on file mtime (`codefix index <repo>`). Retrieved chunk ids are shown to the
model, and **citations are validated** against the ids actually retrieved; unknown ids are
dropped and logged as `citation_dropped`. The UI renders citations as chips that open the
cited source lines.

## 6. Safety

- **Never mutates the target repo.** Each attempt runs in `git clone --no-hardlinks` under
  `.codefix/sandbox/<task>-a<n>` (uncommitted tracked changes are overlaid). Failed workspaces
  are discarded.
- **No shell interpolation.** Every subprocess is an argv list (`sandbox.run` refuses strings);
  tests run with a hard timeout, captured output, and our credentials stripped from the env.
- **Paths:** anything resolving outside the repo root is refused. `.env*`, `.git/`, CI
  workflows, key material and `*secrets*` are blocklisted for read *and* patch.
- **PRs:** only after explicit human confirmation (UI modal with a review checkbox, or the
  CLI prompt), only on `codefix/<task_id>`, never the default branch. That is enforced three
  times: in the LangGraph interrupt, in the PR agent, and in the MCP tool.

## 7. Observability

- `traces/<task_id>.jsonl`: every envelope, tool call, LLM call (model, tokens, cost),
  ReAct step, Manager decision, RAG retrieval, test output, and the `delta_ms` between events.
- `runs.db`: `runs`, `messages`, `llm_calls`, `tool_calls`, `evaluations` (+ `artifacts`).
- LangSmith export: set `LANGCHAIN_TRACING_V2=true` (LangGraph spans + wrapped Anthropic client).
- Dashboard: run list, per-run Gantt timeline per agent, per-agent mean latency
  (bottleneck flagged), attempt distribution, failure-stage breakdown, cost per run, judge
  accuracy, plus an **accuracy weak point** panel separating *localisation* (did the
  Diagnoser name the right file?) from *repair given correct localisation* (did the Fixer's
  patch pass?).

## 8. Evaluation

`benchmarks/bugset/`: two sample repos (Python `pyshop`, 46 tests; JS `textkit`, node:test)
and 16 reproducible injected bugs (`bugs.yaml`) across easy/medium/hard, each with a report,
its failing test and the known-good fix (`codefix bench --verify-bugset` proves each bug fails
when injected and passes when fixed). 16 bugs × 3 architectures = **48 runs** per benchmark.

- **Objective signal:** the final patch is re-applied to a *fresh* clone and the
  previously-failing test plus the full suite are run, independently of the Tester.
- **LLM-as-judge:** `{verdict: pass|fail|partial, reasoning, score}` from the judge model,
  which sees the report, original source and diff but **not** the test results, so
  judge-vs-tests agreement measures the judge, not an echo.
- Each run gets a freshly materialised repo, so no architecture can retrieve another
  architecture's accepted fix for the same bug from the `bugs` collection.
- Output: `benchmarks/results/<bench_id>/runs.csv` + `report.md` (resolve rate, judge pass
  rate, judge accuracy, attempts, retries, escalations, per-agent latency, bottleneck, cost,
  tokens, envelope size, failure stages, per-bug and per-difficulty matrices).
