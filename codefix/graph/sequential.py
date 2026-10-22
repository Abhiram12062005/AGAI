"""SEQUENTIAL baseline: straight Diagnose -> Fix -> Test -> PR pipeline.

No Manager and no feedback: any failure restarts the WHOLE run from diagnosis, carrying
nothing over (the next Fixer never sees why the last patch failed). Bounded by max_attempts
total Fixer attempts, then escalates with a templated note.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from codefix.blackboard import Diagnosis, DispatchRef, Patch, TestReport
from codefix.graph import steps
from codefix.graph.state import PipelineState, ctx_for


def diagnoser_node(state: PipelineState) -> dict:
    ctx = ctx_for(state)
    ctx.post("system", "diagnoser", "dispatched", DispatchRef(instruction="run pipeline from the start"), 1.0)
    diag, _, ok = steps.diagnose(ctx, reply_to="fixer")
    return {"diagnosis": diag.model_dump(), "last_event": "diagnosis_ready" if ok else "diagnosis_failed",
            "patch": None, "test_report": None}


def fixer_node(state: PipelineState) -> dict:
    ctx = ctx_for(state)
    patch, pid = steps.fix(ctx, "tester", Diagnosis.model_validate(state["diagnosis"]), None, None)
    return {"patch": patch.model_dump(), "patch_id": pid}


def tester_node(state: PipelineState) -> dict:
    ctx = ctx_for(state)
    run = ctx.bb.get_run(ctx.task_id)
    restart = "diagnoser" if run["attempts"] < run["max_attempts"] else "human"
    report, _ = steps.test(ctx, lambda r: "pr" if r.passed else restart, Patch.model_validate(state["patch"]),
                           state.get("patch_id"))
    return {"test_report": report.model_dump(), "last_event": "tests_passed" if report.passed else "tests_failed"}


def route_after_test(state: PipelineState) -> str:
    if state["last_event"] == "tests_passed":
        return "pr_prepare"
    ctx = ctx_for(state)
    run = ctx.bb.get_run(ctx.task_id)
    return "diagnoser" if run["attempts"] < run["max_attempts"] else "escalate"


def pr_prepare_node(state: PipelineState) -> dict:
    ctx = ctx_for(state)
    steps.pr_prepare(ctx, "human", Patch.model_validate(state["patch"]), TestReport.model_validate(state["test_report"]),
                     Diagnosis.model_validate(state["diagnosis"]))
    return {"final_status": "awaiting_approval"}


def open_pr_node(state: PipelineState) -> dict:
    res = steps.open_pr(ctx_for(state), bool(state.get("pr_confirmed")))
    return {"final_status": "pr_opened" if res.pushed else "pr_not_opened"}


def escalate_node(state: PipelineState) -> dict:
    ctx = ctx_for(state)
    reason = "diagnosis failed" if state.get("last_event") == "diagnosis_failed" else "attempt cap reached"
    steps.escalate(ctx, reason, use_llm=False)
    return {"final_status": "escalated"}


def build(checkpointer=None):
    g = StateGraph(PipelineState)
    for name, fn in (("diagnoser", diagnoser_node), ("fixer", fixer_node), ("tester", tester_node),
                     ("pr_prepare", pr_prepare_node), ("open_pr", open_pr_node), ("escalate", escalate_node)):
        g.add_node(name, fn)
    g.add_edge(START, "diagnoser")
    g.add_conditional_edges("diagnoser", lambda s: "fixer" if s["last_event"] == "diagnosis_ready" else "escalate",
                            {"fixer": "fixer", "escalate": "escalate"})
    g.add_edge("fixer", "tester")
    g.add_conditional_edges("tester", route_after_test,
                            {"pr_prepare": "pr_prepare", "diagnoser": "diagnoser", "escalate": "escalate"})
    g.add_edge("pr_prepare", "open_pr")
    g.add_edge("open_pr", END)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=["open_pr"])
