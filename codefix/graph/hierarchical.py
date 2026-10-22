"""HIERARCHICAL (default): Manager/Orchestrator delegates to Diagnoser, Fixer, Tester/PR.

    START -> manager --(conditional: state.next)--> diagnoser | fixer | tester | pr_prepare | escalate
    diagnoser -> manager      fixer -> manager      tester -> manager
    pr_prepare -> [interrupt: human confirmation] -> open_pr -> END
    escalate -> END

The re-delegation loop (manager -> fixer -> manager -> tester -> manager -> fixer ...) is a
real cycle in the compiled graph. Bounds are enforced by the Manager node in code:
attempts <= max_attempts, re-diagnoses <= max_rediagnoses; then escalate to a human.

Agents read inputs and write results ONLY through the Blackboard: the Manager's dispatch
carries artifact ids (DispatchRef), never artifact content.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from codefix.agents import manager
from codefix.blackboard import DispatchRef, TestReport
from codefix.graph import steps
from codefix.graph.state import PipelineState, ctx_for

WORKERS = ("diagnoser", "fixer", "tester")


def _dispatch(ctx, target: str, instruction: str, refs: dict, attempt: int = 0) -> None:
    # DESIGN POINT: refs by (task_id, artifact_id) only — message size stays bounded.
    ctx.post("manager", target, "dispatched",
             DispatchRef(instruction=instruction, artifact_refs={k: v for k, v in refs.items() if v},
                         attempt_no=attempt), 1.0)


def _latest_dispatch(ctx, agent: str) -> DispatchRef:
    for m in reversed(ctx.bb.messages(ctx.task_id)):
        if m["target_agent"] == agent and m["status"] == "dispatched" and m["sender"] == "manager":
            return DispatchRef.model_validate(m["payload"])
    raise RuntimeError(f"no dispatch found for {agent}")


def manager_node(state: PipelineState) -> dict:
    ctx = ctx_for(state)
    cfg = ctx.settings.pipeline
    run = ctx.bb.get_run(ctx.task_id)
    ev = state.get("last_event", "start")
    refs = {"diagnosis": run["diagnosis_id"], "patch": run["patch_id"], "test_report": run["test_report_id"]}
    attempts, max_attempts = run["attempts"], run["max_attempts"]
    reason = ""

    if ev == "start":
        nxt, why = "diagnoser", "new task"
    elif ev == "diagnosis_failed":
        nxt, why, reason = "escalate", "diagnoser failed", "no usable diagnosis"
    elif ev == "diagnosis_ready":
        d = ctx.bb.current(ctx.task_id, "diagnosis")
        weak = d.confidence < cfg.rediagnose_below_confidence or d.degraded or not d.implicated_files
        allowed = ["fixer"] + (["diagnoser"] if weak and state.get("rediagnoses", 0) < cfg.max_rediagnoses else [])
        nxt, why = manager.decide(ctx, "diagnosis received", allowed, "diagnoser" if len(allowed) > 1 else "fixer",
                                  {"diagnosis": d.model_dump(), "rediagnoses_used": state.get("rediagnoses", 0)})
    elif ev == "patch_ready":
        nxt, why = "tester", "verify patch"
    elif ev == "tests_passed":
        nxt, why = "pr_prepare", "tests green"
    elif ev == "tests_failed":
        tr: TestReport = ctx.bb.current(ctx.task_id, "test_report")
        if attempts >= max_attempts:
            nxt, why, reason = "escalate", "attempt cap reached", f"{attempts}/{max_attempts} attempts failed"
        else:
            nxt, why = manager.decide(
                ctx, "test report: FAILED", ["fixer", "escalate"], "fixer",
                {"attempts": f"{attempts}/{max_attempts}", "failure_stage": tr.failure_stage,
                 "failing_assertions": tr.failing_assertions[:8], "stack_trace_tail": tr.stack_trace[-1500:],
                 "previous_attempts": steps.history(ctx)})
            if nxt == "escalate":
                reason = f"manager escalated early: {why}"
    else:
        nxt, why, reason = "escalate", f"unknown event {ev}", f"unexpected state {ev}"

    ctx.event("manager_decision", event=ev, decision=nxt, reason=why, attempts=attempts)
    if nxt in WORKERS:
        instruction = {"diagnoser": "Diagnose the root cause" + (" (re-diagnosis requested)" if ev == "diagnosis_ready" else ""),
                       "fixer": f"Write a patch (attempt {attempts + 1}/{max_attempts})"
                                + (" — previous attempt failed, TestReport attached by ref" if ev == "tests_failed" else ""),
                       "tester": "Apply the patch in an isolated workspace, lint, run the test suite"}[nxt]
        _dispatch(ctx, nxt, instruction, refs, attempts)
    elif nxt == "pr_prepare":
        _dispatch(ctx, "pr", "Tests green: commit the verified patch to a codefix/<task_id> branch", refs, attempts)
    out: dict = {"next": nxt}
    if reason:
        out["escalation_reason"] = reason
    return out


def diagnoser_node(state: PipelineState) -> dict:
    ctx = ctx_for(state)
    _latest_dispatch(ctx, "diagnoser")
    _, _, ok = steps.diagnose(ctx, reply_to="manager")
    redo = state.get("rediagnoses", 0) + (1 if state.get("last_event") == "diagnosis_ready" else 0)
    return {"last_event": "diagnosis_ready" if ok else "diagnosis_failed", "rediagnoses": redo}


def fixer_node(state: PipelineState) -> dict:
    ctx = ctx_for(state)
    refs = _latest_dispatch(ctx, "fixer").artifact_refs
    diag = ctx.bb.get_artifact(refs.get("diagnosis"))
    feedback = ctx.bb.get_artifact(refs.get("test_report")) if state.get("last_event") == "tests_failed" else None
    previous = ctx.bb.get_artifact(feedback.patch_artifact_id) if feedback else None
    steps.fix(ctx, "manager", diag, feedback, previous)
    return {"last_event": "patch_ready"}


def tester_node(state: PipelineState) -> dict:
    ctx = ctx_for(state)
    refs = _latest_dispatch(ctx, "tester").artifact_refs
    patch = ctx.bb.get_artifact(refs["patch"])
    report, _ = steps.test(ctx, "manager", patch, refs["patch"])
    return {"last_event": "tests_passed" if report.passed else "tests_failed"}


def pr_prepare_node(state: PipelineState) -> dict:
    ctx = ctx_for(state)
    report = ctx.bb.current(ctx.task_id, "test_report")
    steps.pr_prepare(ctx, "manager", ctx.bb.get_artifact(report.patch_artifact_id), report,
                     ctx.bb.current(ctx.task_id, "diagnosis"))
    return {"final_status": "awaiting_approval"}


def open_pr_node(state: PipelineState) -> dict:
    res = steps.open_pr(ctx_for(state), bool(state.get("pr_confirmed")))
    return {"final_status": "pr_opened" if res.pushed else "pr_not_opened"}


def escalate_node(state: PipelineState) -> dict:
    steps.escalate(ctx_for(state), state.get("escalation_reason") or "unresolved", use_llm=True)
    return {"final_status": "escalated"}


def build(checkpointer=None):
    g = StateGraph(PipelineState)
    g.add_node("manager", manager_node)
    g.add_node("diagnoser", diagnoser_node)
    g.add_node("fixer", fixer_node)
    g.add_node("tester", tester_node)
    g.add_node("pr_prepare", pr_prepare_node)
    g.add_node("open_pr", open_pr_node)
    g.add_node("escalate", escalate_node)
    g.add_edge(START, "manager")
    g.add_conditional_edges("manager", lambda s: s["next"],
                            {n: n for n in ("diagnoser", "fixer", "tester", "pr_prepare", "escalate")})
    for w in WORKERS:
        g.add_edge(w, "manager")
    g.add_edge("pr_prepare", "open_pr")
    g.add_edge("open_pr", END)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=["open_pr"])
