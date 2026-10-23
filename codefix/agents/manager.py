"""Manager/Orchestrator — routing + state evaluation only (cheap model).

The control loop is the ADK `ManagerAgent` (codefix/adk_side/pipelines.py); each routing
decision and the escalation note are one-shot ADK LlmAgents run through `adk_side.runtime`.

Hard limits live in code (the graph): the attempt cap, the re-diagnosis cap and the set of
legal transitions at each decision point. The Manager LLM only chooses *among the legal
options* (e.g. "retry the Fixer with the failure attached" vs. "escalate now because this
failure is infrastructure, not code"). An out-of-set or failed answer falls back to the
deterministic default, so a bad routing answer can never create an unbounded loop.
"""

from __future__ import annotations

import json

from codefix.agents.base import EscalationSummary, ManagerDecision, RunContext, clip

SYSTEM = """You are the Manager of CodeFix AI, a hierarchical multi-agent bug-fixing pipeline
(Diagnoser -> Fixer -> Tester/PR). You do not write code. At each decision point you receive
the current pipeline state and a list of ALLOWED decisions. Choose exactly one allowed decision
and justify it in one or two sentences. Reply as JSON {decision, reason, confidence}.

Guidance:
- A weak diagnosis (low confidence, no implicated files, degraded) is worth one re-diagnosis.
- After a test failure, retrying the Fixer with the failure log is usually right while attempts
  remain. Escalate early only if the failure is clearly not fixable by changing source code
  (test infrastructure missing, timeout of the whole suite, identical failure repeated with the
  same diff).
"""


async def decide(ctx: RunContext, point: str, allowed: list[str], default: str, state: dict) -> tuple[str, str]:
    from codefix.adk_side.runtime import structured

    if len(allowed) == 1:
        return allowed[0], "only legal transition"
    if getattr(ctx.llm, "mode", "") == "mock" and not ctx.extra.get("mock_manager_llm"):
        return default, "deterministic policy (mock LLM mode)"
    prompt = (f"DECISION POINT: {point}\nALLOWED: {json.dumps(allowed)}\nDEFAULT POLICY: {default}\n\n"
              f"STATE:\n{clip(state, 5000)}")
    try:
        d = await structured(ctx, name="manager", role="manager", schema=ManagerDecision, system=SYSTEM, prompt=prompt)
        if d.decision in allowed:
            return d.decision, d.reason
        ctx.event("manager_invalid_decision", decision=d.decision, allowed=allowed)
        return default, f"invalid LLM decision '{d.decision}' -> default policy"
    except Exception as e:  # model unavailable or invalid output: the deterministic policy decides
        return default, f"manager LLM unavailable ({e}) -> default policy"


def fallback_summary(ctx: RunContext, reason: str, history: list[dict]) -> EscalationSummary:
    return EscalationSummary(
        summary=(f"CodeFix could not produce a verified fix for: {ctx.bug_report[:300]}\n"
                 f"Reason for escalation: {reason}. {len(history)} attempt(s) were made; see 'tried'."),
        tried=[f"attempt {h.get('attempt')}: {h.get('rationale', '')[:160]} -> {h.get('result', '')[:160]}"
               for h in history])


async def escalation_summary(ctx: RunContext, reason: str, history: list[dict]) -> EscalationSummary:
    from codefix.adk_side.runtime import structured

    if getattr(ctx.llm, "mode", "") == "mock":
        return fallback_summary(ctx, reason, history)
    prompt = (f"Write the escalation note for a human engineer.\nBUG: {ctx.bug_report}\nREASON: {reason}\n"
              f"ATTEMPTS:\n{clip(history, 6000)}")
    try:
        return await structured(ctx, name="escalation_writer", role="manager", schema=EscalationSummary,
                                system=SYSTEM, prompt=prompt)
    except Exception:
        return fallback_summary(ctx, reason, history)
