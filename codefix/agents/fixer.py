"""Fixer — turns a Diagnosis into a Patch (unified diff + rationale).

The model sees the implicated files (numbered), retrieved code chunks and past accepted fixes,
and — on a retry — the previous attempt's diff plus the TestReport that rejected it. It emits
anchored find/replace edits; the unified diff is derived deterministically (see patching.py).
One internal repair turn is allowed when an edit anchor does not match.
"""

from __future__ import annotations

from codefix.agents.base import FixerOutput, RunContext, chunk_ids, clip, format_chunks
from codefix.agents.patching import EditError, build_diff, is_test_path
from codefix.blackboard import Diagnosis, Patch, TestReport
from codefix.llm import LLMError
from mcp_server.sandbox import norm_rel

SYSTEM = """You are the Fixer in CodeFix AI, a multi-agent bug-resolution system.
Given a diagnosed root cause, write the smallest correct code change that fixes the bug.

Rules:
- Return JSON with: rationale, edits[{path, find, replace}], citations, confidence.
- Each edit replaces `find` (exact text copied from the CURRENT file, occurring exactly once —
  include surrounding lines if needed) with `replace`. Preserve indentation exactly.
- Fix the root cause in the source code. Do NOT modify or delete tests to make them pass.
- Keep the change minimal: no refactors, renames or formatting changes beyond the fix.
- On a retry you are shown the previous attempt and the failing test output: explain in the
  rationale what was wrong with it and do something different.
- citations: ids of retrieved chunks (e.g. code:src/a.py:10-30, bugs:<id>) or file excerpts
  (file:<path>) you relied on.
"""


def _read_raw(ctx: RunContext, path: str) -> str | None:
    try:
        return ctx.tool("fixer", "read_file", path=path, raw=True)["content"].replace("\r\n", "\n")
    except Exception:
        return None


def run(ctx: RunContext, diagnosis: Diagnosis, attempt_no: int, feedback: TestReport | None = None,
        previous: Patch | None = None) -> Patch:
    allow_tests = ctx.settings.pipeline.allow_test_edits
    files = [f for f in diagnosis.implicated_files if allow_tests or not is_test_path(f)][:4]
    originals: dict[str, str] = {}
    for f in files:
        raw = _read_raw(ctx, f)
        if raw is not None:
            originals[f] = raw
    if not originals:  # diagnosis named nothing readable: fall back to hybrid search
        hits = ctx.tool("fixer", "code_search", query=diagnosis.root_cause[:300], k=5)["results"]
        for h in hits:
            if len(originals) >= 2:
                break
            if not is_test_path(h["file"]) and h["file"] not in originals:
                raw = _read_raw(ctx, h["file"])
                if raw is not None:
                    originals[h["file"]] = raw

    known = {f"file:{p}" for p in originals}
    chunks, past = [], []
    try:
        q = f"{diagnosis.root_cause}\n{' '.join(diagnosis.implicated_symbols)}"
        chunks = ctx.tool("fixer", "rag_retrieve", query=q, k=4, collection="code")["chunks"]
        past = ctx.tool("fixer", "rag_retrieve", query=ctx.bug_report, k=2, collection="bugs")["chunks"]
    except Exception as e:
        ctx.event("rag_error", agent="fixer", error=str(e))
    known |= chunk_ids(chunks) | chunk_ids(past)
    ctx.event("rag_retrieval", agent="fixer", ids=sorted(chunk_ids(chunks) | chunk_ids(past)))

    numbered = "\n\n".join(
        f"[file:{p}]\n" + "\n".join(f"{i:>5}| {ln}" for i, ln in enumerate(src.splitlines(), 1))[:12000]
        for p, src in originals.items())
    prompt = (f"BUG REPORT:\n{ctx.bug_report}\n\nDIAGNOSIS:\n{diagnosis.model_dump_json(indent=1)}\n\n"
              f"CURRENT FILES (line numbers are for reference only — do not include them in find/replace):\n"
              f"{numbered or '(none found)'}\n\nRELATED CODE CHUNKS:\n{format_chunks(chunks, 800)}\n\n"
              f"PAST ACCEPTED FIXES FOR SIMILAR BUGS:\n{format_chunks(past, 800)}\n")
    if feedback is not None:
        prompt += (f"\nTHIS IS ATTEMPT {attempt_no}. The previous attempt FAILED.\n"
                   f"PREVIOUS DIFF:\n{clip(previous.diff if previous else '(none)', 4000)}\n"
                   f"PREVIOUS RATIONALE: {previous.rationale if previous else ''}\n"
                   f"FAILURE STAGE: {feedback.failure_stage}\n"
                   f"FAILING ASSERTIONS:\n{clip(feedback.failing_assertions, 2000)}\n"
                   f"LINT FINDINGS:\n{clip([f.model_dump() for f in feedback.lint_findings], 1000)}\n"
                   f"TEST OUTPUT / STACK TRACE:\n{clip(feedback.stack_trace, 4000)}\n"
                   "Note: the previous diff was NOT applied — edit the CURRENT files shown above.\n")
    messages = [{"role": "user", "content": prompt}]

    last_error = ""
    for turn in range(2):
        try:
            res = ctx.llm.structured(role="fixer", system=SYSTEM, messages=messages, schema=FixerOutput,
                                     task_id=ctx.task_id, agent="fixer")
        except LLMError as e:
            return Patch(diff="", rationale="", attempt_no=attempt_no, generation_error=str(e))
        out: FixerOutput = res.parsed
        edits = [e.model_dump() for e in out.edits]
        try:
            for e in edits:
                e["path"] = norm_rel(e["path"])
                if not allow_tests and is_test_path(e["path"]):
                    raise EditError(f"editing test file '{e['path']}' is not allowed; fix the source code")
                if e["path"] not in originals:
                    raw = _read_raw(ctx, e["path"])
                    if raw is None:
                        raise EditError(f"file '{e['path']}' does not exist or is not readable")
                    originals[e["path"]] = raw
            built = build_diff(originals, edits)
            cites = [c for c in out.citations if c in known]
            return Patch(diff=built.diff, rationale=out.rationale, files_touched=built.files,
                         attempt_no=attempt_no, citations=cites)
        except EditError as e:
            last_error = str(e)
            ctx.event("fixer_edit_error", agent="fixer", turn=turn, error=last_error)
            messages += [{"role": "assistant", "content": res.assistant_content},
                         {"role": "user", "content": f"Your edits could not be applied: {last_error}. "
                          "Copy `find` text exactly from the CURRENT files and try again."}]
    return Patch(diff="", rationale="", attempt_no=attempt_no, generation_error=last_error)
