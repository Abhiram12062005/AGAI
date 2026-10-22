"""Offline ORACLE mock for the seeded bug set (`--llm mock`).

It plays the LLM roles using the bug's ground truth so the full pipeline (MCP tools,
workspaces, real test runs, Blackboard, traces, UI) can be demoed without an API key.
With fail_first=True the first Fixer attempt is a plausible-but-wrong patch, which exercises
the Manager's re-delegation loop. Runs made this way are tagged llm_mode='mock' everywhere and
are never mixed into live benchmark statistics.
"""

from __future__ import annotations

import json

from codefix.agents.base import (
    EscalationSummary,
    FixerOutput,
    JudgeVerdict,
    ManagerDecision,
    ReActStep,
)
from codefix.eval.bugset import Bug


class OracleMock:
    def __init__(self, bug: Bug, fail_first: bool = True):
        self.bug = bug
        self.fail_first = fail_first
        self.fixer_calls = 0

    def __call__(self, role, system, messages, schema, n):
        b = self.bug
        if schema is ReActStep:
            if len(messages) == 1:
                return {"thought": f"The report points at {b.files[0]}; read it before concluding.",
                        "action": "read_file", "action_input": json.dumps({"path": b.files[0]}), "final_answer": None}
            return {"thought": "The code confirms the defect.", "action": "final_answer", "action_input": "{}",
                    "final_answer": {"root_cause": b.root_cause, "implicated_files": b.files,
                                     "implicated_symbols": [], "evidence": [f"{b.files[0]}: {b.inject[0]['replace']}"],
                                     "confidence": 0.85, "citations": []}}
        if schema is FixerOutput:
            self.fixer_calls += 1
            first = b.inject[0]
            if self.fail_first and self.fixer_calls == 1:  # wrong: only adds a comment
                marker = "//" if first["file"].endswith(".js") else "#"
                return {"rationale": "(oracle mock) cosmetic change that does not fix the bug",
                        "edits": [{"path": first["file"], "find": first["replace"],
                                   "replace": f"{first['replace']}  {marker} reviewed"}],
                        "citations": [], "confidence": 0.4}
            return {"rationale": f"(oracle mock) revert the defect: {b.root_cause}",
                    "edits": [{"path": e["path"], "find": e["find"], "replace": e["replace"]} for e in b.fix_edits],
                    "citations": [f"file:{b.files[0]}"], "confidence": 0.9}
        if schema is ManagerDecision:
            return {"decision": "fixer", "reason": "(oracle mock)", "confidence": 1.0}
        if schema is EscalationSummary:
            return {"summary": "(oracle mock) escalation", "tried": []}
        if schema is JudgeVerdict:
            return {"verdict": "pass", "reasoning": "(oracle mock judge)", "score": 1.0}
        raise ValueError(f"oracle mock has no answer for {schema.__name__}")
