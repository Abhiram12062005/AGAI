"""Single entry point used by the CLI, the web API and the benchmark harness.

A run: input guardrail -> RAG index -> the chosen architecture as Google ADK agents
(codefix/adk_side/pipelines.py), which reach the CrewAI workers over A2A. The human PR gate
needs no framework checkpoint: the paused run's complete state is on the Blackboard, so
`resolve_pr` resumes from it directly.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from codefix import config, guardrails, prompts
from codefix.agents.base import RunContext
from codefix.blackboard import Blackboard, new_task_id
from codefix.graph import steps
from codefix.graph.state import register, unregister
from codefix.llm import MockHandler, make_llm
from codefix.mcp_client import close_toolbox, get_toolbox

ARCHS = ("hierarchical", "sequential", "p2p")

_BB: Blackboard | None = None
_LOCK = threading.Lock()


def blackboard() -> Blackboard:
    global _BB
    with _LOCK:
        if _BB is None:
            config.ensure_dirs()
            _BB = Blackboard(config.DB_PATH, config.TRACES_DIR)
        return _BB


def _ephemeral(repo: Path) -> bool:
    try:
        Path(repo).resolve().relative_to((config.DATA_DIR / ".codefix").resolve())
        return True
    except ValueError:
        return False


def _tool_sink(bb: Blackboard) -> Callable[[dict], None]:
    def sink(ev: dict) -> None:
        args = {k: (v[:300] + "..." if isinstance(v, str) and len(v) > 300 else v) for k, v in ev["args"].items()}
        preview = json.dumps(ev["result"], default=str)[:2000]
        bb.record_tool_call(ev["task_id"], agent=ev["agent"], tool=ev["tool"], args=json.dumps(args, default=str),
                            ok=int(ev["ok"]), latency_ms=ev["latency_ms"], result_preview=preview[:500])
        bb.trace.emit(ev["task_id"], "tool_call", agent=ev["agent"], tool=ev["tool"], args=args, ok=ev["ok"],
                      latency_ms=ev["latency_ms"], result=preview)
    return sink


def _llm_sink(bb: Blackboard) -> Callable[[dict], None]:
    def sink(row: dict) -> None:
        task_id = row.pop("task_id")
        bb.record_llm_call(task_id, **row)
        bb.trace.emit(task_id, "llm_call", **row)
    return sink


def make_context(task_id: str, repo: Path, bug_report: str, stack_trace: str, arch: str, llm_mode: str,
                 max_attempts: int, mock_handler: MockHandler | None = None, extra: dict | None = None) -> RunContext:
    bb = blackboard()
    settings = config.settings()
    toolbox = get_toolbox(repo)
    toolbox.listener = _tool_sink(bb)
    llm = make_llm(llm_mode, settings, telemetry=_llm_sink(bb), handler=mock_handler)
    return RunContext(task_id=task_id, repo=repo, bug_report=bug_report, stack_trace=stack_trace, arch=arch, bb=bb,
                      toolbox=toolbox, llm=llm, settings=settings, max_attempts=max_attempts, extra=extra or {})


def run_pipeline(bug_report: str, repo: str | Path, arch: str = "hierarchical", max_attempts: int | None = None,
                 llm_mode: str = "live", stack_trace: str = "", task_id: str | None = None,
                 mock_handler: MockHandler | None = None, bench_id: str | None = None, bug_id: str | None = None,
                 auto_index: bool = True, extra: dict | None = None, precreated: bool = False) -> dict:
    """Run one bug through the selected architecture. Returns the final Blackboard snapshot.

    `precreated=True`: the caller already registered the run with `create_run_record` (the web
    API does, so the run is visible before the worker thread starts).
    """
    repo = Path(repo).resolve()
    settings = config.settings()
    max_attempts = max_attempts or settings.pipeline.max_attempts
    task_id = task_id or new_task_id()
    bb = blackboard()
    if not precreated:
        create_run_record(task_id, bug_report, repo, arch, max_attempts, llm_mode, stack_trace, bench_id, bug_id,
                          extra)
    started = time.time()
    try:
        screen = guardrails.screen_user_input(f"{bug_report}\n{stack_trace}")
        bb.trace.emit(task_id, "guardrail", stage="user_input", **screen.model_dump())
        if not screen.allowed:  # never reaches a model or a tool
            bb.update_run(task_id, status="rejected", outcome="blocked", failure_stage="guardrail",
                          meta={**bb.get_run(task_id)["meta"], "guardrail": screen.model_dump()})
            return bb.snapshot(task_id)
        ctx = make_context(task_id, repo, bug_report, stack_trace, arch, llm_mode, max_attempts, mock_handler,
                           bb.get_run(task_id)["meta"])
        register(ctx)
        if auto_index:
            stats = ctx.toolbox.call("index_repo", {}, task_id=task_id, agent="system")
            bb.trace.emit(task_id, "index", **stats)
        from codefix.adk_side import pipelines

        asyncio.run(pipelines.run(ctx))
    except Exception as e:
        bb.trace.emit(task_id, "run_error", error=repr(e), tb=traceback.format_exc()[-4000:])
        bb.update_run(task_id, status="error", outcome="error", failure_stage="error",
                      meta={**(bb.get_run(task_id) or {}).get("meta", {}), "error": str(e)[:500]})
    finally:
        unregister(task_id)
        if _ephemeral(repo):  # demo/bench copies are never reused: free their MCP server
            close_toolbox(repo)
        run = bb.get_run(task_id)
        if run and run["status"] == "running":  # pipeline ended without a terminal step
            bb.update_run(task_id, status="failed", outcome="failed")
        bb.update_run(task_id, duration_ms=int((time.time() - started) * 1000))
        bb.trace.emit(task_id, "run_finished", status=bb.get_run(task_id)["status"],
                      outcome=bb.get_run(task_id)["outcome"])
    return bb.snapshot(task_id)


def create_run_record(task_id: str, bug_report: str, repo: str | Path, arch: str, max_attempts: int,
                      llm_mode: str = "live", stack_trace: str = "", bench_id: str | None = None,
                      bug_id: str | None = None, extra: dict | None = None) -> None:
    repo = Path(repo).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"repo not found: {repo}")
    if arch not in ARCHS:
        raise ValueError(f"unknown arch '{arch}'")
    settings = config.settings()
    extra = dict(extra or {})
    extra.setdefault("variant", settings.pipeline.prompt_variant)
    extra.setdefault("persona", settings.pipeline.persona)
    prompts.check(extra["variant"], extra["persona"])
    blackboard().create_run(task_id, bug_report, str(repo), arch, max_attempts, stack_trace=stack_trace,
                            llm_mode=llm_mode, bench_id=bench_id, bug_id=bug_id,
                            meta={"models": {r: c.model for r, c in settings.roles.items()},
                                  "frameworks": {"orchestration": "google-adk", "workers": "crewai",
                                                 "agent_link": "a2a"}, **extra})


def resolve_pr(task_id: str, confirmed: bool) -> dict:
    """Resume a run paused at the human gate: its complete state is on the Blackboard."""
    bb = blackboard()
    run = bb.get_run(task_id)
    if run is None:
        raise KeyError(task_id)
    if run["status"] != "awaiting_approval":
        raise ValueError(f"run {task_id} is '{run['status']}', not awaiting PR approval")
    ctx = make_context(task_id, Path(run["repo"]), run["bug_report"], run["stack_trace"], run["arch"],
                       "mock" if run["llm_mode"] == "mock" else "live", run["max_attempts"],
                       mock_handler=(lambda *a: {}) if run["llm_mode"] == "mock" else None, extra=run["meta"])
    register(ctx)
    try:
        steps.open_pr(ctx, bool(confirmed))
    finally:
        unregister(task_id)
        if _ephemeral(Path(run["repo"])):
            close_toolbox(run["repo"])
    return bb.snapshot(task_id)


def export_graph_images(out_dir: Path) -> dict[str, str]:
    """docs/graph.png (hierarchical) + graph_<arch>.png from the Mermaid source of each architecture."""
    import base64

    import httpx

    from codefix.adk_side.pipelines import MERMAID

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for arch, src in MERMAID.items():
        (out_dir / f"graph_{arch}.mmd").write_text(src, encoding="utf-8")
        name = "graph.png" if arch == "hierarchical" else f"graph_{arch}.png"
        try:
            url = "https://mermaid.ink/img/" + base64.urlsafe_b64encode(src.encode()).decode() + "?type=png&bgColor=white"
            r = httpx.get(url, timeout=30)
            r.raise_for_status()
            (out_dir / name).write_bytes(r.content)
            written[arch] = str(out_dir / name)
        except Exception as e:  # mermaid.ink unreachable: keep the .mmd source
            written[arch] = f"(png failed: {e}); mermaid at {out_dir / f'graph_{arch}.mmd'}"
    return written
