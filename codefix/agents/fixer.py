"""Fixer — turns a Diagnosis into a Patch (unified diff + rationale). A CrewAI crew.

Context gathering is deterministic: the implicated files (numbered), retrieved code chunks and
past accepted fixes, and — on a retry — the previous attempt's diff plus the TestReport that
rejected it. The model call is a CrewAI Agent + Task with `output_pydantic=FixerOutput`. The
Task's guardrail applies the anchored find/replace edits (the unified diff is derived
deterministically, see patching.py), rejects test edits and unsafe code (codefix/guardrails.py)
and, on failure, sends the reason back to the Fixer for one repair turn.

Prompt variant `react-reflect` adds a second agent, the Reviewer, whose Task critiques the patch
(Reflection pattern) and may return a revised one.
"""

from __future__ import annotations

from codefix import guardrails, prompts
from codefix.agents.base import FixerOutput, ReviewOutput, RunContext, chunk_ids, clip, format_chunks
from codefix.agents.patching import EditError, build_diff, is_test_path
from codefix.blackboard import Diagnosis, Patch, TestReport
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


def _gather(ctx: RunContext, diagnosis: Diagnosis) -> tuple[dict[str, str], list[dict], list[dict], set[str]]:
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
    return originals, chunks, past, known


def _prompt(ctx: RunContext, diagnosis: Diagnosis, originals: dict[str, str], chunks, past, attempt_no: int,
            feedback: TestReport | None, previous: Patch | None) -> str:
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
    return prompt


class _PatchBuilder:
    """Turns FixerOutput edits into a verified unified diff (the Task guardrail's work)."""

    def __init__(self, ctx: RunContext, originals: dict[str, str], known: set[str]):
        self.ctx, self.originals, self.known = ctx, originals, known
        self.built = None
        self.output: FixerOutput | None = None
        self.last_error = ""

    def build(self, out: FixerOutput):
        allow_tests = self.ctx.settings.pipeline.allow_test_edits
        edits = [e.model_dump() for e in out.edits]
        for e in edits:
            e["path"] = norm_rel(e["path"])
            if not allow_tests and is_test_path(e["path"]):
                raise EditError(f"editing test file '{e['path']}' is not allowed; fix the source code")
            if e["path"] not in self.originals:
                raw = _read_raw(self.ctx, e["path"])
                if raw is None:
                    raise EditError(f"file '{e['path']}' does not exist or is not readable")
                self.originals[e["path"]] = raw
        built = build_diff(self.originals, edits)
        safety = guardrails.patch_safety(built.diff)
        self.ctx.event("guardrail", stage="patch_safety", **safety.model_dump())
        if not safety.allowed:
            raise EditError("patch rejected by the safety guardrail: " + "; ".join(safety.reasons))
        return built

    def guardrail(self, task_output):  # no annotation: CrewAI checks it literally and ours is a string
        """CrewAI task guardrail: (True, output) to accept, (False, feedback) to send it back."""
        try:
            out = task_output.pydantic or FixerOutput.model_validate_json(task_output.raw)
            self.built, self.output = self.build(out), out
            return True, task_output.raw
        except Exception as e:  # EditError, validation error: feed the reason back to the Fixer
            self.last_error = str(e)
            self.ctx.event("fixer_edit_error", agent="fixer", error=self.last_error)
            return False, (f"Your edits could not be applied: {self.last_error}. "
                           "Copy `find` text exactly from the CURRENT files and try again.")


def run(ctx: RunContext, diagnosis: Diagnosis, attempt_no: int, feedback: TestReport | None = None,
        previous: Patch | None = None) -> Patch:
    from crewai import Agent, Crew, Process, Task

    from codefix.crew_side.llm import CodefixCrewLLM
    from codefix.observability.callbacks import crew_callbacks, install_crew_listener

    install_crew_listener()
    persona = ctx.extra.get("persona", "default")
    reflect = ctx.extra.get("variant") == "react-reflect"
    originals, chunks, past, known = _gather(ctx, diagnosis)
    prompt = _prompt(ctx, diagnosis, originals, chunks, past, attempt_no, feedback, previous)
    builder = _PatchBuilder(ctx, originals, known)

    fixer_llm = CodefixCrewLLM(model=ctx.settings.role("fixer").model, task_id=ctx.task_id, role="fixer",
                               agent="fixer", default_schema=FixerOutput)
    fixer = Agent(role="Fixer", goal="Write the smallest correct patch for the diagnosed root cause",
                  backstory=prompts.with_persona(SYSTEM, persona), llm=fixer_llm,
                  allow_delegation=False, verbose=False, max_iter=2)
    fix_task = Task(name="write_patch", description=prompt,  # kickoff() has no inputs: braces in code stay literal
                    expected_output="JSON: rationale, edits[{path, find, replace}], citations, confidence",
                    agent=fixer, output_pydantic=FixerOutput, guardrail=builder.guardrail,
                    guardrail_max_retries=1)
    agents, tasks = [fixer], [fix_task]
    review_task = None
    if reflect:
        reviewer = Agent(role="Reviewer", goal="Critique the proposed patch before it is tested",
                         backstory=prompts.with_persona(prompts.REVIEWER_SYSTEM, persona),
                         llm=CodefixCrewLLM(model=ctx.settings.role("reviewer").model, task_id=ctx.task_id,
                                            role="reviewer", agent="reviewer", default_schema=ReviewOutput),
                         allow_delegation=False, verbose=False, max_iter=2)
        review_task = Task(name="review_patch",
                           description=f"{prompt}\n\nReview the Fixer's proposed patch (in your context).",
                           expected_output="JSON: approved, issues, revised", agent=reviewer,
                           output_pydantic=ReviewOutput, context=[fix_task])
        agents.append(reviewer)
        tasks.append(review_task)

    crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False, **crew_callbacks(ctx.task_id))
    try:
        result = crew.kickoff()
    except Exception as e:
        if builder.built is None:  # guardrail retries exhausted or model error
            return Patch(diff="", rationale="", attempt_no=attempt_no,
                         generation_error=builder.last_error or str(e))
        result = None
    if builder.built is None:
        return Patch(diff="", rationale="", attempt_no=attempt_no, generation_error=builder.last_error or "no patch")

    out, built = builder.output, builder.built
    if review_task is not None and result is not None:
        review = review_task.output.pydantic if review_task.output else None
        ctx.event("reflection", agent="reviewer", approved=getattr(review, "approved", None),
                  issues=getattr(review, "issues", []))
        if review is not None and not review.approved and review.revised is not None:
            try:  # a revision must pass the same guardrails as the original
                built, out = builder.build(review.revised), review.revised
            except Exception as e:
                ctx.event("reflection_rejected", agent="reviewer", error=str(e))
    cites = [c for c in out.citations if c in known]
    return Patch(diff=built.diff, rationale=out.rationale, files_touched=built.files, attempt_no=attempt_no,
                 citations=cites)
