"""Agent steps shared by all three architectures.

Each step runs one agent and records the exchange as typed envelopes on the Blackboard
(in_progress on start, success/failure on completion). Architectures differ only in *who
the envelopes are addressed to* and *where inputs come from* (Blackboard refs vs. inline).
"""

from __future__ import annotations

import traceback
from collections.abc import Callable

from codefix.agents import diagnoser, fixer, manager, pr_agent, tester
from codefix.agents.base import RunContext
from codefix.blackboard import Diagnosis, DispatchRef, Escalation, Patch, PRResult, TestReport


def _started(ctx: RunContext, agent: str, target: str, note: str) -> None:
    ctx.post(agent, target, "in_progress", DispatchRef(instruction=note), 1.0)


def diagnose(ctx: RunContext, reply_to: str) -> tuple[Diagnosis, str | None, bool]:
    _started(ctx, "diagnoser", reply_to, "diagnosing")
    try:
        diag, _ = diagnoser.run(ctx)
        ok = diag.confidence > 0 and bool(diag.root_cause.strip())
    except Exception as e:
        ctx.event("agent_error", agent="diagnoser", error=repr(e), tb=traceback.format_exc()[-3000:])
        diag, ok = Diagnosis(root_cause=f"Diagnoser failed: {e}", confidence=0.0, degraded=True), False
        if _fatal(e):
            raise
    env = ctx.post("diagnoser", reply_to, "success" if ok else "failure", diag, diag.confidence)
    return diag, env.artifact_id, ok


def fix(ctx: RunContext, reply_to: str, diag: Diagnosis, feedback: TestReport | None,
        previous: Patch | None) -> tuple[Patch, str | None]:
    attempt = ctx.bb.increment_attempts(ctx.task_id)
    _started(ctx, "fixer", reply_to, f"writing patch (attempt {attempt})")
    try:
        patch = fixer.run(ctx, diag, attempt, feedback=feedback, previous=previous)
    except Exception as e:
        ctx.event("agent_error", agent="fixer", error=repr(e), tb=traceback.format_exc()[-3000:])
        if _fatal(e):
            raise
        patch = Patch(diff="", rationale="", attempt_no=attempt, generation_error=str(e))
    conf = 0.0 if patch.generation_error else 0.8
    env = ctx.post("fixer", reply_to, "failure" if patch.generation_error else "success", patch, conf)
    return patch, env.artifact_id


def test(ctx: RunContext, reply_to: str | Callable[[TestReport], str], patch: Patch,
         patch_id: str | None) -> tuple[TestReport, str | None]:
    """`reply_to` may depend on the verdict (peer hand-off in sequential/p2p)."""
    _started(ctx, "tester", reply_to if isinstance(reply_to, str) else "system", f"testing attempt {patch.attempt_no}")
    try:
        report = tester.run(ctx, patch, patch_id)
    except Exception as e:
        ctx.event("agent_error", agent="tester", error=repr(e), tb=traceback.format_exc()[-3000:])
        report = TestReport(passed=False, failure_stage="infra", stack_trace=f"Tester error: {e}",
                            patch_artifact_id=patch_id, attempt_no=patch.attempt_no)
    target = reply_to if isinstance(reply_to, str) else reply_to(report)
    env = ctx.post("tester", target, "success" if report.passed else "failure", report, 1.0)
    return report, env.artifact_id


def pr_prepare(ctx: RunContext, reply_to: str, patch: Patch, report: TestReport,
               diag: Diagnosis | None) -> PRResult:
    _started(ctx, "pr", reply_to, "committing verified patch to codefix branch")
    try:
        res = pr_agent.prepare(ctx, patch, report, diag)
        ctx.post("pr", reply_to, "success", res, 1.0)
    except Exception as e:
        ctx.event("agent_error", agent="pr", error=repr(e))
        res = PRResult(branch=pr_agent.branch_name(ctx.task_id), note=f"branch preparation failed: {e}")
        ctx.post("pr", reply_to, "failure", res, 0.0)
    ctx.bb.update_run(ctx.task_id, status="awaiting_approval", outcome="passed", failure_stage=None)
    return res


def open_pr(ctx: RunContext, confirmed: bool) -> PRResult:
    run = ctx.bb.get_run(ctx.task_id)
    report: TestReport = ctx.bb.current(ctx.task_id, "test_report")  # type: ignore[assignment]
    patch: Patch = ctx.bb.get_artifact(report.patch_artifact_id)  # type: ignore[assignment]
    diag = ctx.bb.current(ctx.task_id, "diagnosis")
    if not confirmed:
        res = PRResult(branch=pr_agent.branch_name(ctx.task_id), note="PR declined by human")
        ctx.post("human", "pr", "failure", res, 1.0)
        ctx.bb.update_run(ctx.task_id, status="done")
        return res
    ctx.post("human", "pr", "dispatched", DispatchRef(instruction="open pull request (human confirmed)"), 1.0)
    res = pr_agent.open_pr(ctx, patch, report, diag, confirmed=True)  # type: ignore[arg-type]
    # no remote = branch intentionally kept local (success); a rejected push is a failure
    ok = res.pushed or "no 'origin' remote" in res.note
    ctx.post("pr", "human", "success" if ok else "failure", res, 1.0)
    ctx.bb.update_run(ctx.task_id, status="done", meta={**run["meta"], "pr_url": res.url})
    return res


def history(ctx: RunContext) -> list[dict]:
    patches = {a["id"]: a["body"] for a in ctx.bb.artifacts(ctx.task_id, "patch")}
    out = []
    for tr in ctx.bb.artifacts(ctx.task_id, "test_report"):
        b = tr["body"]
        p = patches.get(b.get("patch_artifact_id"), {})
        out.append({"attempt": b.get("attempt_no"), "rationale": p.get("rationale") or p.get("generation_error", ""),
                    "files": p.get("files_touched"), "result": f"{b.get('failure_stage')}: "
                    + ("; ".join(b.get("failing_assertions", [])[:3]) or b.get("stack_trace", "")[:200])})
    return out


def escalate(ctx: RunContext, reason: str, use_llm: bool) -> Escalation:
    hist = history(ctx)
    # no Manager in sequential/p2p: those get the templated hand-off note
    s = manager.escalation_summary(ctx, reason, hist, use_llm=use_llm)
    run = ctx.bb.get_run(ctx.task_id)
    esc = Escalation(reason=reason, summary=s.summary, attempts=run["attempts"], tried=s.tried)
    ctx.post("manager" if use_llm else "system", "human", "failure", esc, 1.0)
    last = ctx.bb.current(ctx.task_id, "test_report")
    stage = getattr(last, "failure_stage", None) or ("diagnosis" if run["attempts"] == 0 else "tests")
    ctx.bb.update_run(ctx.task_id, status="escalated", outcome="escalated", failure_stage=stage)
    return esc


def _fatal(e: Exception) -> bool:
    """Errors no retry can fix (bad credentials, missing key) abort the run instead of looping."""
    msg = str(e)
    return any(s in msg for s in ("API_KEY", "401", "authentication", "403", "credit balance"))
