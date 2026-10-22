"""Tester — applies a Patch in an isolated workspace, runs lint + the test suite, emits a TestReport.

Deterministic (no LLM): the verdict comes from the project's own tests and static analysis.
A failing attempt's workspace is discarded; a passing one is kept for the PR step.
"""

from __future__ import annotations

from codefix.agents.base import RunContext
from codefix.agents.patching import is_test_path
from codefix.blackboard import LintFinding, Patch, TestReport


def _gating(findings: list[dict]) -> set[tuple]:
    return {(f["path"], f["code"], f["message"]) for f in findings if f.get("gating")}


def run(ctx: RunContext, patch: Patch, patch_artifact_id: str | None = None) -> TestReport:
    base = {"patch_artifact_id": patch_artifact_id, "attempt_no": patch.attempt_no}
    if patch.generation_error or not patch.diff.strip():
        return TestReport(passed=False, failure_stage="patch",
                          stack_trace=f"Fixer could not produce an applicable patch: {patch.generation_error}", **base)
    if not ctx.settings.pipeline.allow_test_edits and any(is_test_path(f) for f in patch.files_touched):
        return TestReport(passed=False, failure_stage="patch",
                          stack_trace="Patch modifies test files, which is not allowed.", **base)
    ws = f"{ctx.task_id}-a{patch.attempt_no}"[:100]
    ctx.tool("tester", "create_workspace", name=ws)
    keep = False
    try:
        applied = ctx.tool("tester", "apply_patch", diff=patch.diff, workspace=ws)
        if not applied.get("ok"):
            return TestReport(passed=False, failure_stage="apply", workspace=ws,
                              stack_trace=f"git apply failed:\n{applied.get('error')}", **base)
        files = [f for f in patch.files_touched]
        after = ctx.tool("tester", "static_analysis", paths=files, workspace=ws)["findings"]
        try:
            before = ctx.tool("tester", "static_analysis", paths=files)["findings"]
        except Exception:
            before = []
        new_errors = _gating(after) - _gating(before)  # only findings the patch introduced gate
        lint = [LintFinding(path=f["path"], line=f["line"], code=f["code"], message=f["message"])
                for f in after][:30]
        if new_errors:
            return TestReport(passed=False, failure_stage="lint", lint_findings=lint, workspace=ws,
                              stack_trace="Patch introduced static-analysis errors:\n"
                              + "\n".join(f"{p}: {c} {m}" for p, c, m in sorted(new_errors)), **base)
        r = ctx.tool("tester", "run_tests", workspace=ws, timeout_s=ctx.settings.pipeline.test_timeout_s)
        report = TestReport(passed=bool(r["passed"]), total=r["total"], failed=r["failed"],
                            failing_assertions=r["failing_assertions"], stack_trace=r["stack_trace"],
                            lint_findings=lint, duration_ms=r["duration_ms"], failure_stage=r.get("failure_stage"),
                            workspace=ws, **base)
        ctx.event("test_output", agent="tester", workspace=ws, framework=r.get("framework"),
                  output=r.get("output", "")[-4000:])
        keep = report.passed
        return report
    finally:
        if not keep:
            try:
                ctx.tool("tester", "discard_workspace", workspace=ws)
            except Exception:
                pass
