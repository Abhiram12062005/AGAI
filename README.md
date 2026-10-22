# CodeFix AI

Autonomous **hierarchical multi-agent** bug resolution. Paste a bug report and point at a Git
repo: a **Manager** delegates to a **Diagnoser** (explicit ReAct), a **Fixer** and a
**Tester/PR** agent. It finds the defect, explains the root cause, writes a patch, proves it
with the project's own test suite, and offers a Pull Request. Failed tests send the Fixer
back *with the failure log* (max 3 attempts, then a written escalation). Nothing is pushed
without your explicit confirmation.

- **LangGraph** with a real re-delegation cycle and a human-gated PR node → [`docs/graph.png`](docs/graph.png)
- **MCP server** (stdio) for all tool access + a **SQLite Blackboard** of typed envelopes
- **RAG** over code / past fixes / docs (ChromaDB + all-MiniLM-L6-v2), with cited chunks
- **Observability**: JSONL traces, `runs.db`, live web view, dashboard
- **Evaluation**: 16 seeded bugs × 3 architectures, LLM-as-judge vs. objective tests

Design rationale: [`docs/architecture.md`](docs/architecture.md) · Presenting it: [`docs/DEMO.md`](docs/DEMO.md)

## 5-minute quickstart

Requires Python 3.11+, Git, and Node 18+ (only for the JS sample repo).

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

Create a free Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey), put it in
`.env` as `GEMINI_API_KEY`, then:

The free tier has rate limits, and Google may use submitted content to improve its products.
Do not send confidential repository code unless that use is acceptable. CodeFix cost telemetry
assumes free-tier access and does not report charges from a paid Google project. To use Anthropic
instead, set `provider: anthropic` in `config/models.yaml` and configure `ANTHROPIC_API_KEY`.

```bash
codefix demo py05-region-case
```

That copies the sample `pyshop` repo, injects a seeded bug, attaches the real failing test
output and fixes it live. You watch the Manager, the ReAct steps, the patch, the tests and
then the PR prompt. No key yet? The same path runs offline with a scripted oracle (real
tools, real tests; it shows a retry):

```bash
codefix demo py05-region-case --llm mock
```

Web app (New fix · live run view · Runs · Dashboard):

```bash
codefix serve
```

Open http://127.0.0.1:8000.

## Commands

| Command | What it does |
|---|---|
| `codefix fix "<bug report>" --repo <path> [--arch hierarchical\|sequential\|p2p] [--max-attempts N] [-t trace.txt]` | Fix a bug in your own repo |
| `codefix demo <bug-id> [--llm live\|mock]` | End-to-end on a seeded bug (fastest demo path) |
| `codefix index <repo> [--force]` | Build/refresh the RAG index (incremental on mtime) |
| `codefix bench --arch all --runs N [--bugs py0,js] [--llm live\|mock]` | Benchmark → `benchmarks/results/<id>/runs.csv` + `report.md` |
| `codefix bench --verify-bugset` | Prove every seeded bug fails when injected and passes when fixed |
| `codefix trace <task_id> [--json] [--react]` | Replay a run / dump full Blackboard state / print the ReAct transcript |
| `codefix runs`, `codefix pr <task_id>` | List runs; push + open the PR for a passing run (asks first) |
| `codefix serve`, `codefix graph` | Web app; re-export `docs/graph*.png` from the compiled graphs |

## Defaults I picked (all configurable)

| Decision | Default | Where |
|---|---|---|
| Models | Manager/Judge `gemini-3.1-flash-lite`, Diagnoser/Fixer `gemini-3.8-flash` (Google AI Studio free tier) | `config/models.yaml` |
| Retry limit / ReAct cap | 3 Fixer attempts / 8 ReAct steps | `config/models.yaml` → `pipeline` |
| UI | FastAPI + plain HTML/JS (no build step, no CDN, so it works offline on stage) | `web/` |
| Patch format | model emits anchored find/replace edits; the unified diff is derived deterministically | `codefix/agents/patching.py` |
| Isolation | `git clone --no-hardlinks` per attempt under `.codefix/sandbox/` | `mcp_server/sandbox.py` |
| PR backend | PyGithub if `GITHUB_TOKEN` is set, else `gh` if installed, else push + compare URL | `mcp_server/tools.py` |
| JS sample tests | `node --test` (zero install); jest is auto-detected for real repos | `mcp_server/testrunner.py` |
| Package manager | plain `pip` + venv, exact pins in `pyproject.toml` | |

## Layout

```
codefix/        agents/ (diagnoser, fixer, tester, pr_agent, manager) · graph/ (hierarchical, sequential, p2p, runner)
                blackboard/ (schemas, SQLite store) · rag/ · observability/ · eval/ (bugset, oracle, judge, bench, report) · cli/
mcp_server/     MCP server + tool implementations + sandbox + test runner
web/            api/ (FastAPI) · ui/ (static pages)
config/         models.yaml
benchmarks/     bugset/ (repos + bugs.yaml) · results/
docs/           architecture.md · DEMO.md · graph.png
tests/          runs with no API key (LLM mocked)
```

## Quality

```bash
pytest
ruff check .
```

The suite needs **no API key** and no model download: the LLM is mocked, and embeddings
use the hashing fallback.
