"""Markdown comparison report: hierarchical vs sequential vs p2p."""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict
from pathlib import Path


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else 0.0


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "n/a"


def summarize(rows: list[dict]) -> dict[str, dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["arch"]].append(r)
    out = {}
    for arch, rs in sorted(by.items()):
        n = len(rs)
        judged = [r for r in rs if r["judge_verdict"] not in (None, "error")]
        out[arch] = {
            "n": n,
            "resolved_tests": sum(1 for r in rs if r["objective_pass"]),
            "target_green": sum(1 for r in rs if r["target_test_pass"]),
            "pipeline_passed": sum(1 for r in rs if r["outcome"] == "passed"),
            "judge_pass": sum(1 for r in judged if r["judge_verdict"] == "pass"),
            "judge_partial": sum(1 for r in judged if r["judge_verdict"] == "partial"),
            "judged": len(judged),
            "agree": sum(1 for r in judged if r["agreement"]),
            "attempts": _mean([r["attempts"] for r in rs]),
            "retried": sum(1 for r in rs if (r["attempts"] or 0) > 1),
            "escalated": sum(1 for r in rs if r["outcome"] == "escalated"),
            "errors": sum(1 for r in rs if r["outcome"] == "error"),
            "duration_s": _mean([r["duration_ms"] / 1000 for r in rs if r["duration_ms"]]),
            "cost": _mean([r["pipeline_cost_usd"] for r in rs]),
            "cost_total": sum(r["pipeline_cost_usd"] or 0 for r in rs),
            "judge_cost": sum(r["judge_cost_usd"] or 0 for r in rs),
            "diagnoser_s": _mean([r["diagnoser_ms"] / 1000 for r in rs]),
            "fixer_s": _mean([r["fixer_ms"] / 1000 for r in rs]),
            "tester_s": _mean([r["tester_ms"] / 1000 for r in rs]),
            "manager_s": _mean([r["manager_ms"] / 1000 for r in rs]),
            "llm_calls": _mean([r["llm_calls"] for r in rs]),
            "tokens": _mean([r["tokens"] for r in rs]),
            "msg_bytes": _mean([r["avg_msg_bytes"] for r in rs]),
            "stages": Counter(r["failure_stage"] for r in rs if r["outcome"] != "passed"),
        }
    return out


def write_report(bench_id: str, path: Path, rows: list[dict] | None = None) -> Path:
    from codefix.eval.bench import bench_rows

    rows = rows if rows is not None else bench_rows(bench_id)
    s = summarize(rows)
    mode = {r["llm_mode"] for r in rows}
    lines = [f"# CodeFix AI benchmark — `{bench_id}`\n"]
    if "mock" in mode:
        lines.append("> **MOCK RUN** — agents were played by the offline oracle mock. These numbers validate the "
                     "harness and pipeline mechanics only; they are NOT model results.\n")
    bugs = sorted({r["bug_id"] for r in rows})
    lines.append(f"{len(rows)} runs · {len(bugs)} bugs · architectures: {', '.join(s)}\n")

    lines.append("## Headline comparison\n")
    lines.append("| Metric | " + " | ".join(s) + " |")
    lines.append("|---|" + "---|" * len(s))

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(v) for v in s.values()) + " |")

    row("Resolved (fresh-clone full suite green)", lambda v: f"{v['resolved_tests']}/{v['n']} ({_pct(v['resolved_tests'], v['n'])})")
    row("Previously-failing test green", lambda v: f"{v['target_green']}/{v['n']}")
    row("Judge verdict = pass", lambda v: f"{v['judge_pass']}/{v['judged']} ({_pct(v['judge_pass'], v['judged'])})")
    row("Judge verdict = partial", lambda v: str(v["judge_partial"]))
    row("**Judge accuracy** (agrees with tests)", lambda v: f"{_pct(v['agree'], v['judged'])}")
    row("Mean Fixer attempts", lambda v: f"{v['attempts']:.2f}")
    row("Runs needing a retry", lambda v: str(v["retried"]))
    row("Escalated to human", lambda v: str(v["escalated"]))
    row("Errors", lambda v: str(v["errors"]))
    row("Mean wall-clock / run", lambda v: f"{v['duration_s']:.1f}s")
    row("Mean pipeline cost / run", lambda v: f"${v['cost']:.4f}")
    row("Total pipeline cost", lambda v: f"${v['cost_total']:.3f}")
    row("Mean LLM calls / run", lambda v: f"{v['llm_calls']:.1f}")
    row("Mean tokens / run", lambda v: f"{v['tokens']:.0f}")
    row("Mean envelope size (bytes)", lambda v: f"{v['msg_bytes']:.0f}")

    lines.append("\n## Per-agent mean latency (seconds per run)\n")
    lines.append("| Agent | " + " | ".join(s) + " |")
    lines.append("|---|" + "---|" * len(s))
    for agent in ("diagnoser", "fixer", "tester", "manager"):
        row(agent.capitalize() if agent != "manager" else "Manager (LLM time)", lambda v, a=agent: f"{v[a + '_s']:.1f}")
    bottleneck = {a: max(("diagnoser", "fixer", "tester", "manager"), key=lambda k: v[k + "_s"]) for a, v in s.items()}
    lines.append("\nBottleneck agent per architecture: " + ", ".join(f"**{a}** → {b}" for a, b in bottleneck.items()))

    lines.append("\n## Failure-stage breakdown (non-passing runs)\n")
    stages = sorted({k for v in s.values() for k in v["stages"]}, key=str)
    if stages:
        lines.append("| Stage | " + " | ".join(s) + " |")
        lines.append("|---|" + "---|" * len(s))
        for stg in stages:
            row(str(stg), lambda v, g=stg: str(v["stages"].get(g, 0)))
    else:
        lines.append("_No failures._")

    lines.append("\n## Judge vs objective tests (all architectures)\n")
    cm = Counter((r["judge_verdict"], bool(r["objective_pass"])) for r in rows if r["judge_verdict"])
    lines.append("| Judge \\ Tests | green | red |\n|---|---|---|")
    for v in ("pass", "partial", "fail", "error"):
        if cm[(v, True)] or cm[(v, False)]:
            lines.append(f"| {v} | {cm[(v, True)]} | {cm[(v, False)]} |")

    lines.append("\n## Per-bug outcomes (✅ resolved · ❌ not resolved · attempts)\n")
    lines.append("| Bug | difficulty | " + " | ".join(s) + " |")
    lines.append("|---|---|" + "---|" * len(s))
    cell = defaultdict(list)
    diff = {}
    for r in rows:
        cell[(r["bug_id"], r["arch"])].append(f"{'✅' if r['objective_pass'] else '❌'}{r['attempts']}")
        diff[r["bug_id"]] = r["difficulty"]
    for b in bugs:
        lines.append(f"| {b} | {diff.get(b) or ''} | " + " | ".join(" ".join(cell[(b, a)]) or "-" for a in s) + " |")

    lines.append("\n## By difficulty (resolved / runs)\n")
    lines.append("| Difficulty | " + " | ".join(s) + " |")
    lines.append("|---|" + "---|" * len(s))
    for d in ("easy", "medium", "hard"):
        rs = [r for r in rows if r["difficulty"] == d]
        if rs:
            lines.append(f"| {d} | " + " | ".join(
                f"{sum(1 for r in rs if r['arch'] == a and r['objective_pass'])}/{sum(1 for r in rs if r['arch'] == a)}"
                for a in s) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
