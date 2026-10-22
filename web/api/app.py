"""FastAPI backend for the CodeFix AI web app.

Pages (static, no build step):  /  (new fix)   /run.html?id=   /runs.html   /dashboard.html
API:
  GET  /api/config                     models, archs, defaults, seeded bugs, whether a key is set
  POST /api/runs                       start a run (background thread) -> {task_id}
  GET  /api/runs                       history (filters: outcome, arch, llm_mode, bench_id)
  GET  /api/runs/{id}                  full Blackboard snapshot + derived views
  GET  /api/runs/{id}/events           SSE stream of the JSONL trace (live run view)
  GET  /api/runs/{id}/source           read-only source excerpt for a citation
  POST /api/runs/{id}/pr               {confirm: true|false} — the ONLY way a PR is opened from the UI
  GET  /api/dashboard                  aggregates for the dashboard
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from codefix import config
from codefix.blackboard import new_task_id
from codefix.graph import runner

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
app = FastAPI(title="CodeFix AI", version="0.1.0")
config.settings()  # loads .env (ANTHROPIC_API_KEY, GITHUB_TOKEN) before any request
_threads: dict[str, threading.Thread] = {}


class RunRequest(BaseModel):
    bug_report: str = ""
    repo: str = ""
    arch: str = "hierarchical"
    max_attempts: int = Field(3, ge=1, le=10)
    stack_trace: str = ""
    llm: str = "live"
    bug_id: str | None = None  # seeded bug: materialise a fresh demo repo


class PRRequest(BaseModel):
    confirm: bool


@app.get("/api/config")
def get_config():
    from codefix.eval.bugset import load_bugs

    s = config.settings()
    return {
        "archs": list(runner.ARCHS),
        "models": {r: c.model_dump() for r, c in s.roles.items()},
        "pipeline": s.pipeline.model_dump(),
        "has_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "bugs": [{"id": b.id, "repo": b.repo, "difficulty": b.difficulty, "report": b.report} for b in load_bugs()],
    }


@app.post("/api/runs")
def start_run(req: RunRequest):
    if req.arch not in runner.ARCHS:
        raise HTTPException(400, f"arch must be one of {list(runner.ARCHS)}")
    handler, report, repo, stack = None, req.bug_report, req.repo, req.stack_trace
    if req.bug_id:
        from codefix.eval.bugset import failing_output, load_bugs, materialize, trace_text
        from codefix.eval.oracle import OracleMock

        bugs = load_bugs([req.bug_id])
        if not bugs:
            raise HTTPException(404, "unknown bug id")
        bug = bugs[0]
        task_id = new_task_id()
        repo = str(materialize(bug, config.DATA_DIR / ".codefix" / "demo" / f"{bug.id}-{task_id}"))
        report = report or bug.report
        if bug.with_trace and not stack:
            stack = trace_text(failing_output(bug, Path(repo)))
        handler = OracleMock(bug) if req.llm == "mock" else None
    else:
        task_id = new_task_id()
        if req.llm == "mock":
            raise HTTPException(400, "mock LLM mode is only available for seeded bugs (pick one)")
    if not report.strip() or not repo:
        raise HTTPException(400, "bug_report and repo are required")
    if not Path(repo).is_dir():
        raise HTTPException(400, f"repo path not found: {repo}")
    if req.llm == "live" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(400, "ANTHROPIC_API_KEY is not set — add it to .env or use the mock LLM with a seeded bug")

    # register synchronously so the run page can load the moment we return the id
    runner.create_run_record(task_id, report, repo, req.arch, req.max_attempts, req.llm, stack, bug_id=req.bug_id)

    def work():
        runner.run_pipeline(report, repo, arch=req.arch, max_attempts=req.max_attempts, llm_mode=req.llm,
                            stack_trace=stack, task_id=task_id, mock_handler=handler, bug_id=req.bug_id,
                            precreated=True)

    t = threading.Thread(target=work, name=f"run-{task_id}", daemon=True)
    _threads[task_id] = t
    t.start()
    return {"task_id": task_id}


@app.get("/api/runs")
def list_runs(outcome: str | None = None, arch: str | None = None, llm_mode: str | None = None,
              bench_id: str | None = None, limit: int = 300):
    bb = runner.blackboard()
    q = ("SELECT r.task_id, r.created_at, r.arch, r.llm_mode, r.status, r.outcome, r.failure_stage, r.attempts,"
         " r.max_attempts, r.duration_ms, r.cost_usd, r.bug_id, r.bench_id, substr(r.bug_report,1,140) AS bug_report,"
         " e.verdict AS judge_verdict, e.objective_pass FROM runs r LEFT JOIN evaluations e ON e.task_id=r.task_id"
         " WHERE 1=1")
    args: list = []
    for col, val in (("r.outcome", outcome), ("r.arch", arch), ("r.llm_mode", llm_mode), ("r.bench_id", bench_id)):
        if val:
            if col == "r.outcome" and val == "running":
                q += " AND r.status='running'"
                continue
            q += f" AND {col}=?"
            args.append(val)
    q += " ORDER BY r.created_at DESC LIMIT ?"
    args.append(limit)
    return bb.query(q, tuple(args))


def _spans(messages: list[dict]) -> list[dict]:
    """Gantt spans per agent: from its in_progress message to its completion message."""
    open_: dict[str, str] = {}
    spans = []
    for m in messages:
        if m["status"] == "in_progress":
            open_[m["sender"]] = m["timestamp"]
        elif m["status"] in ("success", "failure") and m["sender"] in open_:
            spans.append({"agent": m["sender"], "start": open_.pop(m["sender"]), "end": m["timestamp"],
                          "status": m["status"], "kind": m["kind"]})
    return spans


@app.get("/api/runs/{task_id}")
def get_run(task_id: str):
    bb = runner.blackboard()
    try:
        snap = bb.snapshot(task_id)
    except KeyError:
        raise HTTPException(404, "unknown task") from None
    events = bb.trace.read(task_id)
    snap["react"] = [e for e in events if e["type"] == "react_step"]
    snap["test_outputs"] = [e for e in events if e["type"] == "test_output"]
    snap["manager_decisions"] = [e for e in events if e["type"] == "manager_decision"]
    snap["patches"] = [a["body"] | {"id": a["id"]} for a in bb.artifacts(task_id, "patch")]
    snap["test_reports"] = [a["body"] | {"id": a["id"]} for a in bb.artifacts(task_id, "test_report")]
    snap["spans"] = _spans(snap["messages"])
    snap["llm_calls"] = bb.query("SELECT agent, model, served_model, input_tokens, output_tokens, cache_read_tokens,"
                                 " cost_usd, latency_ms, timestamp, error FROM llm_calls WHERE task_id=? ORDER BY id",
                                 (task_id,))
    snap["tool_calls"] = bb.query("SELECT agent, tool, ok, latency_ms, timestamp FROM tool_calls WHERE task_id=?"
                                  " ORDER BY id", (task_id,))
    return snap


@app.get("/api/runs/{task_id}/events")
async def run_events(task_id: str):
    bb = runner.blackboard()
    if bb.get_run(task_id) is None:
        raise HTTPException(404, "unknown task")

    async def gen():
        offset, idle = 0, 0
        while True:
            evs = bb.trace.read(task_id, offset)
            for ev in evs:
                offset += 1
                yield f"data: {json.dumps(ev, default=str)}\n\n"
            run = bb.get_run(task_id)
            finished = run["status"] not in ("running",) and any(e["type"] == "run_finished" for e in evs)
            if finished:
                yield "event: done\ndata: {}\n\n"
                return
            idle = 0 if evs else idle + 1
            if idle and idle % 30 == 0:
                yield ": keepalive\n\n"
            if idle > 3600:
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs/{task_id}/source")
def source(task_id: str, path: str, start: int = 1, end: int | None = None):
    from mcp_server.sandbox import SandboxError, safe_path

    run = runner.blackboard().get_run(task_id)
    if not run:
        raise HTTPException(404, "unknown task")
    try:
        full = safe_path(Path(run["repo"]), path, must_exist=True)
    except (SandboxError, FileNotFoundError) as e:
        raise HTTPException(400, str(e)) from None
    lines = full.read_text(encoding="utf-8", errors="replace").splitlines()
    end = min(len(lines), end or start + 40)
    return {"path": path, "start": start, "end": end, "lines": lines[max(0, start - 1):end]}


@app.post("/api/runs/{task_id}/pr")
def open_pr(task_id: str, req: PRRequest):
    try:
        snap = runner.resolve_pr(task_id, confirmed=req.confirm)
    except KeyError:
        raise HTTPException(404, "unknown task") from None
    except ValueError as e:
        raise HTTPException(409, str(e)) from None
    return {"pr": snap["pr"], "status": snap["run"]["status"]}


@app.get("/api/dashboard")
def dashboard(llm_mode: str = Query("live"), bench_id: str | None = None):
    bb = runner.blackboard()
    where, args = "WHERE 1=1", []
    if llm_mode != "all":
        where += " AND r.llm_mode=?"
        args.append(llm_mode)
    if bench_id:
        where += " AND r.bench_id=?"
        args.append(bench_id)
    runs = bb.query(f"SELECT r.task_id, r.arch, r.outcome, r.status, r.failure_stage, r.attempts, r.duration_ms,"
                    f" r.cost_usd, r.created_at, r.bug_id, r.bench_id, e.verdict, e.objective_pass, e.agreement"
                    f" FROM runs r LEFT JOIN evaluations e ON e.task_id=r.task_id {where} ORDER BY r.created_at",
                    tuple(args))
    ids = tuple(r["task_id"] for r in runs) or ("",)
    ph = ",".join("?" * len(ids))
    agent_lat = bb.query(
        f"SELECT sender AS agent, COUNT(*) n, AVG(latency_ms) mean_ms, MAX(latency_ms) max_ms,"
        f" SUM(CASE WHEN status='failure' THEN 1 ELSE 0 END) failures FROM messages"
        f" WHERE task_id IN ({ph}) AND status IN ('success','failure') AND latency_ms IS NOT NULL"
        f" AND sender IN ('diagnoser','fixer','tester','pr') GROUP BY sender", ids)
    mgr = bb.query(f"SELECT 'manager' AS agent, COUNT(*) n, AVG(latency_ms) mean_ms, MAX(latency_ms) max_ms, 0 failures"
                   f" FROM llm_calls WHERE task_id IN ({ph}) AND agent='manager'", ids)
    agent_lat += [m for m in mgr if m["n"]]
    agent_cost = bb.query(f"SELECT agent, model, COUNT(*) calls, SUM(cost_usd) cost, SUM(input_tokens) inp,"
                          f" SUM(output_tokens) outp FROM llm_calls WHERE task_id IN ({ph}) GROUP BY agent, model", ids)
    # accuracy weak point: localisation (diagnoser) vs repair (fixer) on seeded bugs
    from codefix.eval.bugset import load_bugs

    truth = {b.id: set(b.files) for b in load_bugs()}
    loc = defaultdict(lambda: [0, 0])
    rep = defaultdict(lambda: [0, 0])
    for r in runs:
        if r["bug_id"] not in truth or r["status"] == "running":
            continue
        diag = bb.current(r["task_id"], "diagnosis")
        located = bool(diag and truth[r["bug_id"]] & set(diag.implicated_files))
        loc[r["arch"]][0] += located
        loc[r["arch"]][1] += 1
        if located:
            rep[r["arch"]][0] += int(r["outcome"] == "passed")
            rep[r["arch"]][1] += 1
    benches = bb.query("SELECT bench_id, COUNT(*) n, MIN(created_at) started, MAX(llm_mode) llm_mode FROM runs"
                       " WHERE bench_id IS NOT NULL GROUP BY bench_id ORDER BY started DESC")
    return {"runs": runs, "agent_latency": agent_lat, "agent_cost": agent_cost,
            "localisation": {a: {"hit": v[0], "n": v[1]} for a, v in loc.items()},
            "repair_given_located": {a: {"hit": v[0], "n": v[1]} for a, v in rep.items()},
            "benches": benches}


@app.get("/graph.png")
def graph_png():
    return FileResponse(config.ROOT / "docs" / "graph.png")


@app.get("/")
def index():
    return FileResponse(UI_DIR / "index.html")


app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
