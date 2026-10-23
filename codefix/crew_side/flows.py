"""CrewAI Flows that run the worker side of the pipeline.

`WorkerFlow` serves one A2A skill request (fix | test | pr_prepare). A Flow router picks the step.
`PeerRepairFlow` is the P2P architecture's worker side: Fixer and Tester hand the patch and the
full TestReport straight to each other in flow state (no Manager, no Blackboard reads), looping
with a fixed local rule until the tests pass or the attempt cap is hit.

Inputs arrive either as Blackboard references (hierarchical: the A2A message carries artifact
ids, never content) or inline (sequential / p2p hand-offs).
"""

from __future__ import annotations

from typing import Any

from crewai.flow.flow import Flow, listen, or_, router, start

from codefix.agents.base import RunContext
from codefix.blackboard import Diagnosis, Patch, TestReport
from codefix.graph import steps
from codefix.graph.state import get_ctx


def _ctx(flow: Flow) -> RunContext:
    ctx = get_ctx(flow.state["task_id"])
    if ctx is None:
        raise RuntimeError(f"no run context for {flow.state['task_id']}")
    return ctx


def _resolve(ctx: RunContext, payload: dict, name: str, model):
    """An artifact either inline (`name`) or by Blackboard reference (`refs[name]`)."""
    if payload.get(name) is not None:
        return model.model_validate(payload[name])
    ref = (payload.get("refs") or {}).get(name)
    return ctx.bb.get_artifact(ref) if ref else None


def run_fix(ctx: RunContext, p: dict) -> dict:
    diag = _resolve(ctx, p, "diagnosis", Diagnosis)
    feedback = _resolve(ctx, p, "test_report", TestReport) if p.get("with_feedback") else None
    previous = None
    if feedback is not None:
        previous = (Patch.model_validate(p["previous_patch"]) if p.get("previous_patch")
                    else ctx.bb.get_artifact(feedback.patch_artifact_id))
    patch, pid = steps.fix(ctx, p.get("reply_to", "manager"), diag, feedback, previous)
    return {"patch_id": pid, "patch": patch.model_dump(mode="json")}


def run_test(ctx: RunContext, p: dict) -> dict:
    patch = _resolve(ctx, p, "patch", Patch)
    pid = p.get("patch_id") or (p.get("refs") or {}).get("patch")
    if p.get("reply_fail"):  # peer hand-off: the target depends on the verdict
        report, rid = steps.test(ctx, lambda r: p.get("reply_pass", "pr") if r.passed else p["reply_fail"], patch, pid)
    else:
        report, rid = steps.test(ctx, p.get("reply_to", "manager"), patch, pid)
    return {"report_id": rid, "report": report.model_dump(mode="json"), "passed": report.passed}


def run_pr_prepare(ctx: RunContext, p: dict) -> dict:
    report = _resolve(ctx, p, "test_report", TestReport)
    patch = _resolve(ctx, p, "patch", Patch) or ctx.bb.get_artifact(report.patch_artifact_id)
    diag = _resolve(ctx, p, "diagnosis", Diagnosis)
    res = steps.pr_prepare(ctx, p.get("reply_to", "manager"), patch, report, diag)
    return {"pr": res.model_dump(mode="json")}


class WorkerFlow(Flow):
    """One skill request: start -> router(skill) -> fix | test | pr_prepare."""

    suppress_flow_events = True  # console quiet; our listener still traces the run

    @start()
    def receive(self) -> str:
        _ctx(self).event("flow_step", framework="crewai", flow="WorkerFlow", step="receive",
                         skill=self.state["skill"])
        return self.state["skill"]

    @router(receive)
    def route(self) -> str:
        return f"do_{self.state['skill']}"  # labels must differ from the handler names

    @listen("do_fix")
    def fix(self) -> dict:
        self.state["result"] = run_fix(_ctx(self), self.state["payload"])
        return self.state["result"]

    @listen("do_test")
    def test(self) -> dict:
        self.state["result"] = run_test(_ctx(self), self.state["payload"])
        return self.state["result"]

    @listen("do_pr_prepare")
    def pr_prepare(self) -> dict:
        self.state["result"] = run_pr_prepare(_ctx(self), self.state["payload"])
        return self.state["result"]


class PeerRepairFlow(Flow):
    """P2P worker loop: fix -> test -> (passed | retry -> fix | exhausted)."""

    suppress_flow_events = True

    @start()
    def begin(self) -> None:
        self.state.setdefault("patch", None)
        self.state.setdefault("report", None)

    @listen(or_(begin, "retry"))
    def fix(self) -> None:
        ctx = _ctx(self)
        diag = Diagnosis.model_validate(self.state["payload"]["diagnosis"])
        report = TestReport.model_validate(self.state["report"]) if self.state["report"] else None
        previous = Patch.model_validate(self.state["patch"]) if self.state["patch"] and report else None
        patch, pid = steps.fix(ctx, "tester", diag, report, previous)
        self.state["patch"], self.state["patch_id"] = patch.model_dump(mode="json"), pid

    @listen(fix)
    def test(self) -> None:
        ctx = _ctx(self)
        run = ctx.bb.get_run(ctx.task_id)
        retry = "fixer" if run["attempts"] < run["max_attempts"] else "human"
        report, _ = steps.test(ctx, lambda r: "pr" if r.passed else retry, Patch.model_validate(self.state["patch"]),
                               self.state.get("patch_id"))
        self.state["report"] = report.model_dump(mode="json")

    @router(test)
    def decide(self) -> str:
        ctx = _ctx(self)
        if self.state["report"]["passed"]:
            return "passed"
        run = ctx.bb.get_run(ctx.task_id)
        return "retry" if run["attempts"] < run["max_attempts"] else "exhausted"

    @listen("passed")
    def pr(self) -> None:
        ctx = _ctx(self)
        steps.pr_prepare(ctx, "human", Patch.model_validate(self.state["patch"]),
                         TestReport.model_validate(self.state["report"]),
                         Diagnosis.model_validate(self.state["payload"]["diagnosis"]))
        self.state["outcome"] = "awaiting_approval"

    @listen("exhausted")
    def escalate(self) -> None:
        steps.escalate(_ctx(self), "attempt cap reached", use_llm=False)
        self.state["outcome"] = "escalated"


def run_skill(skill: str, task_id: str, payload: dict[str, Any]) -> dict:
    """Entry point used by the A2A executor (runs in a worker thread)."""
    if skill == "p2p_repair":
        flow = PeerRepairFlow()
        flow.kickoff(inputs={"task_id": task_id, "payload": payload})
        return {"outcome": flow.state.get("outcome"), "patch_id": flow.state.get("patch_id"),
                "passed": bool((flow.state.get("report") or {}).get("passed"))}
    if skill not in ("fix", "test", "pr_prepare"):
        raise ValueError(f"unknown skill '{skill}'")
    flow = WorkerFlow()
    flow.kickoff(inputs={"task_id": task_id, "skill": skill, "payload": payload})
    return flow.state["result"]
