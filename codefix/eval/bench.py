"""Benchmark harness: bugs x architectures x prompt variants x personas x repetitions
-> runs.db + CSV + markdown report (resolve rate, judge accuracy, cost, time per configuration).

Every run gets a freshly materialised repo (its own path => its own RAG index and an empty
'bugs' history collection), so no architecture can retrieve another architecture's accepted
fix for the same bug. Mock runs (`--llm mock`) are tagged and reported separately.
"""

from __future__ import annotations

import csv
import datetime as dt
import time
from collections.abc import Callable
from pathlib import Path

from codefix import config
from codefix.eval.bugset import Bug, failing_output, load_bugs, materialize, trace_text
from codefix.eval.judge import judge_run
from codefix.eval.oracle import OracleMock
from codefix.eval.report import write_report
from codefix.graph import runner
from codefix.llm import make_llm
from codefix.mcp_client import close_toolbox
from mcp_server.sandbox import rmtree

BENCH_DIR = config.DATA_DIR / ".codefix" / "bench"
RESULTS_DIR = config.ROOT / "benchmarks" / "results"


def run_bench(archs: list[str], runs: int = 1, bug_ids: list[str] | None = None, llm_mode: str = "live",
              judge: bool = True, max_attempts: int | None = None, variants: list[str] | None = None,
              personas: list[str] | None = None, progress: Callable[[str], None] = print) -> dict:
    bench_id = f"bench-{dt.datetime.now():%Y%m%d-%H%M%S}" + ("-mock" if llm_mode == "mock" else "")
    bugs = load_bugs(bug_ids)
    bb = runner.blackboard()
    judge_llm = None
    if judge:
        def mock_judge(*_):  # offline runs: the judge verdict is a placeholder, flagged as mock
            return {"verdict": "pass", "reasoning": "(mock judge)", "score": 1.0}

        judge_llm = make_llm(llm_mode, config.settings(), telemetry=runner._llm_sink(bb),
                             handler=mock_judge if llm_mode == "mock" else None)
    variants, personas = variants or ["react"], personas or ["default"]
    combos = [(a, v, p) for a in archs for v in variants for p in personas]
    total = len(bugs) * len(combos) * runs
    progress(f"{bench_id}: {len(bugs)} bugs x {len(archs)} archs x {len(variants)} variants x {len(personas)} "
             f"personas x {runs} = {total} runs ({llm_mode})")
    task_ids: list[str] = []
    i = 0
    for rep in range(runs):
        for bug in bugs:
            for arch, variant, persona in combos:
                i += 1
                tid = _one(bench_id, bug, arch, rep, llm_mode, judge_llm, max_attempts, progress, i, total,
                           variant, persona)
                if tid:
                    task_ids.append(tid)
    out_dir = RESULTS_DIR / bench_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = export_csv(bench_id, out_dir / "runs.csv")
    md_path = write_report(bench_id, out_dir / "report.md")
    progress(f"wrote {csv_path} and {md_path}")
    return {"bench_id": bench_id, "task_ids": task_ids, "csv": str(csv_path), "report": str(md_path)}


def _one(bench_id: str, bug: Bug, arch: str, rep: int, llm_mode: str, judge_llm, max_attempts, progress,
         i: int, total: int, variant: str = "react", persona: str = "default") -> str | None:
    # short suffix: Windows' 260-char path limit is close once git object paths are appended
    tag = "" if (variant, persona) == ("react", "default") else \
        f"-{ {'react': 'r', 'fewshot': 'f', 'react-reflect': 'rr'}.get(variant, variant)}{persona[:3]}"
    work = BENCH_DIR / bench_id / f"{bug.id}-{arch}{tag}-r{rep}"
    repo = materialize(bug, work / "repo")
    stack = trace_text(failing_output(bug, repo)) if bug.with_trace else ""
    started = time.time()
    tid = None
    try:
        snap = runner.run_pipeline(
            bug.report, repo, arch=arch, llm_mode=llm_mode, stack_trace=stack, max_attempts=max_attempts,
            mock_handler=OracleMock(bug) if llm_mode == "mock" else None,
            bench_id=bench_id, bug_id=bug.id,
            extra={"difficulty": bug.difficulty, "rep": rep, "variant": variant, "persona": persona})
        tid = snap["run"]["task_id"]
        ev = judge_run(judge_llm, runner.blackboard(), tid, bug, sandbox=work / "judge") if judge_llm else None
        r = runner.blackboard().get_run(tid)
        progress(f"[{i}/{total}] {bug.id:<32} {arch:<12} {variant:<13} {persona:<8} {r['outcome']:<9} attempts={r['attempts']} "
                 f"{int(time.time() - started)}s ${r['cost_usd']:.3f}"
                 + (f" judge={ev['verdict']} tests={'green' if ev['objective_pass'] else 'red'}" if ev else ""))
        return tid
    except Exception as e:  # one broken run must not kill a 48-run benchmark
        progress(f"[{i}/{total}] {bug.id} {arch} {variant} {persona} ERROR {e!r}")
        return tid
    finally:
        close_toolbox(repo)
        if tid:  # benchmarks never open PRs: drop the passing workspace kept for the PR step
            for ws in config.SANDBOX_DIR.glob(f"{tid}-*"):
                rmtree(ws)


CSV_SQL = """
SELECT r.bench_id, r.task_id, r.bug_id, r.arch, r.llm_mode, r.outcome, r.failure_stage, r.attempts, r.max_attempts,
       r.duration_ms, r.created_at,
       json_extract(r.meta, '$.difficulty') AS difficulty,
       COALESCE(json_extract(r.meta, '$.variant'), 'react') AS variant,
       COALESCE(json_extract(r.meta, '$.persona'), 'default') AS persona,
       (SELECT COALESCE(SUM(cost_usd),0) FROM llm_calls l WHERE l.task_id=r.task_id AND l.agent<>'judge') AS pipeline_cost_usd,
       (SELECT COALESCE(SUM(cost_usd),0) FROM llm_calls l WHERE l.task_id=r.task_id AND l.agent='judge') AS judge_cost_usd,
       (SELECT COUNT(*) FROM llm_calls l WHERE l.task_id=r.task_id AND l.agent<>'judge') AS llm_calls,
       (SELECT COALESCE(SUM(input_tokens+output_tokens),0) FROM llm_calls l WHERE l.task_id=r.task_id AND l.agent<>'judge') AS tokens,
       (SELECT COALESCE(SUM(input_tokens),0) FROM llm_calls l WHERE l.task_id=r.task_id AND l.agent<>'judge') AS input_tokens,
       (SELECT COALESCE(SUM(output_tokens),0) FROM llm_calls l WHERE l.task_id=r.task_id AND l.agent<>'judge') AS output_tokens,
       (SELECT COUNT(*) FROM messages m WHERE m.task_id=r.task_id) AS messages,
       (SELECT COUNT(*) FROM tool_calls t WHERE t.task_id=r.task_id) AS tool_calls,
       (SELECT COALESCE(SUM(latency_ms),0) FROM messages m WHERE m.task_id=r.task_id AND m.sender='diagnoser' AND m.status IN ('success','failure')) AS diagnoser_ms,
       (SELECT COALESCE(SUM(latency_ms),0) FROM messages m WHERE m.task_id=r.task_id AND m.sender='fixer' AND m.status IN ('success','failure')) AS fixer_ms,
       (SELECT COALESCE(SUM(latency_ms),0) FROM messages m WHERE m.task_id=r.task_id AND m.sender='tester' AND m.status IN ('success','failure')) AS tester_ms,
       (SELECT COALESCE(SUM(latency_ms),0) FROM llm_calls l WHERE l.task_id=r.task_id AND l.agent='manager') AS manager_ms,
       (SELECT AVG(LENGTH(payload)) FROM messages m WHERE m.task_id=r.task_id) AS avg_msg_bytes,
       e.verdict AS judge_verdict, e.score AS judge_score, e.objective_pass, e.target_test_pass, e.agreement
FROM runs r LEFT JOIN evaluations e ON e.task_id = r.task_id
WHERE r.bench_id = ? ORDER BY r.created_at
"""


def bench_rows(bench_id: str) -> list[dict]:
    return runner.blackboard().query(CSV_SQL, (bench_id,))


def export_csv(bench_id: str, path: Path) -> Path:
    rows = bench_rows(bench_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return path
