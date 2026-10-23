"""The three orchestration architectures, built with Google ADK agents.

HIERARCHICAL (default) - `ManagerAgent`, a custom ADK agent:
    manager -> diagnoser (ADK LlmAgent, in-process)
    manager -> fixer / tester / pr  (CrewAI agents, over A2A)
  Every hop is Manager <-> worker. Dispatches carry Blackboard artifact ids only (DispatchRef),
  never content. On a failed TestReport the Manager re-dispatches ONLY the Fixer, with the
  report attached by reference; bounds (attempts, re-diagnoses) are enforced in code, the
  Manager LLM only chooses among the legal transitions. Then a PR waits for a human, or the
  Manager writes an escalation note.

SEQUENTIAL baseline - ADK workflow agents:
    LoopAgent(max_attempts)[ SequentialAgent[ diagnose -> fix (A2A) -> test (A2A) -> gate ] ] -> finish
  No Manager and no feedback: any failure restarts the whole attempt from diagnosis, carrying
  nothing over. Hand-offs are inline (ADK session state).

P2P - `PeerAgent`: the ADK Diagnoser hands its Diagnosis inline to the CrewAI peers
  (A2A `p2p_repair`), where Fixer and Tester pass the patch and full TestReport straight to each
  other with a fixed local rule. No Manager, no Blackboard reads.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.genai import types

from codefix.a2a_link import call_crew
from codefix.agents import manager
from codefix.agents.base import RunContext
from codefix.blackboard import DispatchRef, TestReport
from codefix.graph import steps
from codefix.graph.state import get_ctx

APP = "codefix"
WORKERS = ("diagnoser", "fixer", "tester")


def _rc(ictx: InvocationContext) -> RunContext:
    ctx = get_ctx(ictx.session.state["task_id"])
    if ctx is None:
        raise RuntimeError("run context missing")
    return ctx


def _say(agent: BaseAgent, ictx: InvocationContext, text: str, escalate: bool = False, **state) -> Event:
    return Event(author=agent.name, invocation_id=ictx.invocation_id, branch=ictx.branch,
                 content=types.Content(role="model", parts=[types.Part(text=text)]),
                 actions=EventActions(state_delta=state or {}, escalate=escalate or None))


# ------------------------------------------------------------------ hierarchical
class ManagerAgent(BaseAgent):
    """Orchestrator: routing + state evaluation; workers are reached by sub-agent call or A2A."""

    async def _run_async_impl(self, ictx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx = _rc(ictx)
        cfg = ctx.settings.pipeline
        ev, rediagnoses = "start", 0
        while True:
            run = ctx.bb.get_run(ctx.task_id)
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
                allowed = ["fixer"] + (["diagnoser"] if weak and rediagnoses < cfg.max_rediagnoses else [])
                nxt, why = await manager.decide(ctx, "diagnosis received", allowed,
                                                "diagnoser" if len(allowed) > 1 else "fixer",
                                                {"diagnosis": d.model_dump(), "rediagnoses_used": rediagnoses})
            elif ev == "patch_ready":
                nxt, why = "tester", "verify patch"
            elif ev == "tests_passed":
                nxt, why = "pr_prepare", "tests green"
            elif ev == "tests_failed":
                tr: TestReport = ctx.bb.current(ctx.task_id, "test_report")
                if attempts >= max_attempts:
                    nxt, why, reason = "escalate", "attempt cap reached", f"{attempts}/{max_attempts} attempts failed"
                else:
                    nxt, why = await manager.decide(
                        ctx, "test report: FAILED", ["fixer", "escalate"], "fixer",
                        {"attempts": f"{attempts}/{max_attempts}", "failure_stage": tr.failure_stage,
                         "failing_assertions": tr.failing_assertions[:8], "stack_trace_tail": tr.stack_trace[-1500:],
                         "previous_attempts": steps.history(ctx)})
                    if nxt == "escalate":
                        reason = f"manager escalated early: {why}"
            else:
                nxt, why, reason = "escalate", f"unknown event {ev}", f"unexpected state {ev}"

            ctx.event("manager_decision", event=ev, decision=nxt, reason=why, attempts=attempts)
            yield _say(self, ictx, f"{ev} -> {nxt}: {why}", manager_decision=nxt)
            dispatch = None
            if nxt in WORKERS:
                instruction = {"diagnoser": "Diagnose the root cause" + (" (re-diagnosis requested)" if ev == "diagnosis_ready" else ""),
                               "fixer": f"Write a patch (attempt {attempts + 1}/{max_attempts})"
                                        + (" — previous attempt failed, TestReport attached by ref" if ev == "tests_failed" else ""),
                               "tester": "Apply the patch in an isolated workspace, lint, run the test suite"}[nxt]
                dispatch = DispatchRef(instruction=instruction, artifact_refs={k: v for k, v in refs.items() if v},
                                       attempt_no=attempts)
                # DESIGN POINT: refs by (task_id, artifact_id) only — message size stays bounded.
                ctx.post("manager", nxt, "dispatched", dispatch, 1.0)
            elif nxt == "pr_prepare":
                dispatch = DispatchRef(instruction="Tests green: commit the verified patch to a codefix/<task_id> branch",
                                       artifact_refs={k: v for k, v in refs.items() if v}, attempt_no=attempts)
                ctx.post("manager", "pr", "dispatched", dispatch, 1.0)

            if nxt == "diagnoser":
                if ev == "diagnosis_ready":
                    rediagnoses += 1
                _, _, ok = await steps.diagnose(ctx, reply_to="manager")
                ev = "diagnosis_ready" if ok else "diagnosis_failed"
            elif nxt == "fixer":
                await call_crew(ctx, "fix", {"refs": dispatch.artifact_refs, "reply_to": "manager",
                                             "with_feedback": ev == "tests_failed"})
                ev = "patch_ready"
            elif nxt == "tester":
                res = await call_crew(ctx, "test", {"refs": dispatch.artifact_refs, "reply_to": "manager"})
                ev = "tests_passed" if res["passed"] else "tests_failed"
            elif nxt == "pr_prepare":
                await call_crew(ctx, "pr_prepare", {"refs": dispatch.artifact_refs, "reply_to": "manager"})
                yield _say(self, ictx, "awaiting human approval for the PR", final_status="awaiting_approval")
                return
            else:  # escalate
                summary = await manager.escalation_summary(ctx, reason or "unresolved", steps.history(ctx))
                steps.escalate(ctx, reason or "unresolved", use_llm=True, summary=summary)
                yield _say(self, ictx, f"escalated: {reason}", final_status="escalated")
                return


# ------------------------------------------------------------------ sequential
class DiagnoseStep(BaseAgent):
    async def _run_async_impl(self, ictx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx = _rc(ictx)
        ctx.post("system", "diagnoser", "dispatched", DispatchRef(instruction="run pipeline from the start"), 1.0)
        diag, _, ok = await steps.diagnose(ctx, reply_to="fixer")
        yield _say(self, ictx, f"diagnosis {'ready' if ok else 'failed'}", diagnosis=diag.model_dump(mode="json"),
                   diag_ok=ok, passed=False)


class FixStep(BaseAgent):
    async def _run_async_impl(self, ictx: InvocationContext) -> AsyncGenerator[Event, None]:
        if not ictx.session.state.get("diag_ok"):
            return
        res = await call_crew(_rc(ictx), "fix", {"diagnosis": ictx.session.state["diagnosis"], "reply_to": "tester"})
        yield _say(self, ictx, "patch written", patch=res["patch"], patch_id=res["patch_id"])


class TestStep(BaseAgent):
    async def _run_async_impl(self, ictx: InvocationContext) -> AsyncGenerator[Event, None]:
        if not ictx.session.state.get("diag_ok"):
            return
        ctx = _rc(ictx)
        run = ctx.bb.get_run(ctx.task_id)
        restart = "diagnoser" if run["attempts"] < run["max_attempts"] else "human"
        res = await call_crew(ctx, "test", {"patch": ictx.session.state["patch"],
                                            "patch_id": ictx.session.state.get("patch_id"),
                                            "reply_pass": "pr", "reply_fail": restart})
        yield _say(self, ictx, f"tests {'passed' if res['passed'] else 'failed'}", passed=res["passed"],
                   test_report=res["report"])


class GateStep(BaseAgent):
    """Leaves the loop when tests pass or the diagnosis failed; otherwise the next iteration restarts."""

    async def _run_async_impl(self, ictx: InvocationContext) -> AsyncGenerator[Event, None]:
        st = ictx.session.state
        stop = bool(st.get("passed")) or not st.get("diag_ok")
        yield _say(self, ictx, "stop" if stop else "restart from diagnosis", escalate=stop)


class FinishStep(BaseAgent):
    async def _run_async_impl(self, ictx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx, st = _rc(ictx), ictx.session.state
        if st.get("passed"):
            await call_crew(ctx, "pr_prepare", {"patch": st["patch"], "test_report": st["test_report"],
                                                "diagnosis": st["diagnosis"], "reply_to": "human"})
            yield _say(self, ictx, "awaiting human approval for the PR", final_status="awaiting_approval")
        else:
            reason = "diagnosis failed" if not st.get("diag_ok") else "attempt cap reached"
            steps.escalate(ctx, reason, use_llm=False)
            yield _say(self, ictx, f"escalated: {reason}", final_status="escalated")


# ------------------------------------------------------------------ p2p
class PeerAgent(BaseAgent):
    async def _run_async_impl(self, ictx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx = _rc(ictx)
        ctx.post("system", "diagnoser", "dispatched", DispatchRef(instruction="new bug report"), 1.0)
        diag, _, ok = await steps.diagnose(ctx, reply_to="fixer")
        if not ok:
            steps.escalate(ctx, "diagnosis failed", use_llm=False)
            yield _say(self, ictx, "escalated: diagnosis failed", final_status="escalated")
            return
        # the Diagnosis travels inline to the peers; they loop fix <-> test among themselves
        res = await call_crew(ctx, "p2p_repair", {"diagnosis": diag.model_dump(mode="json")})
        yield _say(self, ictx, f"peers finished: {res.get('outcome')}", final_status=res.get("outcome"))


# ------------------------------------------------------------------ entry
def build(arch: str, max_attempts: int) -> BaseAgent:
    if arch == "hierarchical":
        return ManagerAgent(name="manager", description="CodeFix Manager/Orchestrator (hierarchical)")
    if arch == "sequential":
        attempt = SequentialAgent(name="attempt", sub_agents=[
            DiagnoseStep(name="diagnose"), FixStep(name="fix_via_a2a"), TestStep(name="test_via_a2a"),
            GateStep(name="gate")])
        loop = LoopAgent(name="restart_loop", max_iterations=max_attempts, sub_agents=[attempt])
        return SequentialAgent(name="sequential_pipeline", sub_agents=[loop, FinishStep(name="finish")])
    if arch == "p2p":
        return PeerAgent(name="p2p_pipeline", description="Diagnoser hands off to CrewAI peers")
    raise ValueError(f"unknown arch '{arch}'")


async def run(ctx: RunContext) -> None:
    root = build(ctx.arch, ctx.max_attempts)
    runner = InMemoryRunner(agent=root, app_name=APP)
    session = await runner.session_service.create_session(app_name=APP, user_id="codefix",
                                                          state={"task_id": ctx.task_id})
    msg = types.Content(role="user", parts=[types.Part(text=ctx.bug_report)])
    async for _ in runner.run_async(user_id="codefix", session_id=session.id, new_message=msg):
        pass


MERMAID = {
    "hierarchical": """flowchart LR
    U([bug report]) --> G{{input guardrail}} --> M[Manager<br/>ADK custom agent]
    M <-->|in-process| D[Diagnoser<br/>ADK LlmAgent + MCP tools]
    M <-->|A2A fix| F[Fixer crew<br/>CrewAI]
    M <-->|A2A test| T[Tester<br/>CrewAI Flow]
    M -->|A2A pr_prepare| P[PR agent<br/>CrewAI Flow] --> H([human approval])
    M -->|attempt cap / early stop| E([escalation note])
    T -.->|tests failed| M
    M -.->|retry Fixer, TestReport by ref| F""",
    "sequential": """flowchart LR
    U([bug report]) --> L
    subgraph L[ADK LoopAgent: max_attempts]
      D[Diagnose<br/>ADK] --> F[Fix<br/>CrewAI via A2A] --> T[Test<br/>CrewAI via A2A] --> G{gate}
    end
    G -->|failed: restart from diagnosis| D
    G -->|passed| P[PR prepare] --> H([human approval])
    L -->|loop exhausted| E([escalation])""",
    "p2p": """flowchart LR
    U([bug report]) --> D[Diagnoser<br/>ADK]
    D -->|A2A p2p_repair, Diagnosis inline| F
    subgraph C[CrewAI PeerRepairFlow]
      F[Fixer] --> T[Tester]
      T -->|full TestReport inline| F
    end
    T -->|passed| P[PR prepare] --> H([human approval])
    T -->|attempts exhausted| E([escalation])""",
}
