"""Manager/Orchestrator — routing + state evaluation only (cheap model).

Hard limits live in code (the graph): the attempt cap, the re-diagnosis cap and the set of
legal transitions at each decision point. The Manager LLM only chooses *among the legal
options* (e.g. "retry the Fixer with the failure attached" vs. "escalate now because this
failure is infrastructure, not code"). An out-of-set or failed answer falls back to the
deterministic default, so a bad routing answer can never create an unbounded loop.
"""

from __future__ import annotations

import json

from codefix.agents.base import EscalationSummary, ManagerDecision, RunContext, clip
from codefix.llm import LLMError

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


def decide(ctx: RunContext, point: str, allowed: list[str], default: str, state: dict) -> tuple[str, str]:
    if len(allowed) == 1:
        return allowed[0], "only legal transition"
    if getattr(ctx.llm, "mode", "") == "mock" and not ctx.extra.get("mock_manager_llm"):
        return default, "deterministic policy (mock LLM mode)"
    prompt = (f"DECISION POINT: {point}\nALLOWED: {json.dumps(allowed)}\nDEFAULT POLICY: {default}\n\n"
              f"STATE:\n{clip(state, 5000)}")
    try:
        res = ctx.llm.structured(role="manager", system=SYSTEM, messages=[{"role": "user", "content": prompt}],
                                 schema=ManagerDecision, task_id=ctx.task_id, agent="manager")
        d: ManagerDecision = res.parsed
        if d.decision in allowed:
            return d.decision, d.reason
        ctx.event("manager_invalid_decision", decision=d.decision, allowed=allowed)
        return default, f"invalid LLM decision '{d.decision}' -> default policy"
    except LLMError as e:
        return default, f"manager LLM unavailable ({e}) -> default policy"


def escalation_summary(ctx: RunContext, reason: str, history: list[dict], use_llm: bool = True) -> EscalationSummary:
    fallback = EscalationSummary(
        summary=(f"CodeFix could not produce a verified fix for: {ctx.bug_report[:300]}\n"
                 f"Reason for escalation: {reason}. {len(history)} attempt(s) were made; see 'tried'."),
        tried=[f"attempt {h.get('attempt')}: {h.get('rationale', '')[:160]} -> {h.get('result', '')[:160]}"
               for h in history])
    if not use_llm or getattr(ctx.llm, "mode", "") == "mock":
        return fallback
    prompt = (f"Write the escalation note for a human engineer.\nBUG: {ctx.bug_report}\nREASON: {reason}\n"
              f"ATTEMPTS:\n{clip(history, 6000)}")
    try:
        return ctx.llm.structured(role="manager", system=SYSTEM, messages=[{"role": "user", "content": prompt}],
                                  schema=EscalationSummary, task_id=ctx.task_id, agent="manager").parsed
    except LLMError:
        return fallback
