"""Shared run context + the structured-output schemas each LLM-backed agent must emit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from codefix.blackboard import Blackboard, Envelope
from codefix.config import Settings
from codefix.llm import BaseLLM
from codefix.mcp_client import Toolbox

OBS_LIMIT = 6000


@dataclass
class RunContext:
    task_id: str
    repo: Path
    bug_report: str
    stack_trace: str
    arch: str
    bb: Blackboard
    toolbox: Toolbox
    llm: BaseLLM
    settings: Settings
    max_attempts: int = 3
    extra: dict = field(default_factory=dict)

    def tool(self, agent: str, tool_name: str, /, **args: Any) -> Any:
        return self.toolbox.call(tool_name, args, task_id=self.task_id, agent=agent)

    def event(self, type_: str, **data: Any) -> None:
        self.bb.trace.emit(self.task_id, type_, **data)

    def post(self, sender: str, target: str, status: str, payload: BaseModel, confidence: float) -> Envelope:
        return self.bb.post(Envelope(task_id=self.task_id, sender=sender, target_agent=target, status=status,
                                     payload=payload, confidence=max(0.0, min(1.0, confidence))))


def clip(obj: Any, limit: int = OBS_LIMIT) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, indent=1, default=str)
    return s if len(s) <= limit else s[:limit] + f"\n... [truncated {len(s) - limit} chars]"


def chunk_ids(obj: Any) -> set[str]:
    """Collect every citable id (RAG chunk ids) that appears in a tool observation."""
    ids: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("id", "chunk_id") and isinstance(v, str) and ":" in v:
                ids.add(v)
            else:
                ids |= chunk_ids(v)
    elif isinstance(obj, list):
        for x in obj:
            ids |= chunk_ids(x)
    return ids


def format_chunks(chunks: list[dict], limit: int = 1200) -> str:
    out = []
    for c in chunks:
        loc = f"{c.get('file')}:{c.get('start_line')}-{c.get('end_line')}" if c.get("start_line") else c.get("file")
        out.append(f"[{c['id']}] ({loc})\n{c['text'][:limit]}")
    return "\n\n".join(out) or "(none)"


# ------------------------------------------------------------------ LLM output schemas
class FinalDiagnosis(BaseModel):
    root_cause: str = Field(description="Precise explanation of the defect and why it causes the bug")
    implicated_files: list[str] = Field(description="Repo-relative paths that must change")
    implicated_symbols: list[str] = Field(description="Functions/classes/methods involved")
    evidence: list[str] = Field(description="Concrete observations (file:line + what you saw)")
    confidence: float = Field(description="0.0-1.0 self-assessed confidence")
    citations: list[str] = Field(description="Ids of retrieved chunks you relied on, e.g. code:src/a.py:10-30")


class ReActStep(BaseModel):
    thought: str = Field(description="Your reasoning for this step")
    action: str = Field(description="Tool name to call, or 'final_answer'")
    action_input: str = Field(description="JSON object (as a string) with the tool arguments; '{}' for final_answer")
    final_answer: FinalDiagnosis | None = Field(description="Required when action == 'final_answer', else null")


class Edit(BaseModel):
    path: str = Field(description="Repo-relative file path")
    find: str = Field(description="Exact existing text to replace (must occur exactly once; include enough context)")
    replace: str = Field(description="Replacement text")


class FixerOutput(BaseModel):
    rationale: str = Field(description="Why this change fixes the root cause (and, on retries, what changed vs. the failed attempt)")
    edits: list[Edit]
    citations: list[str] = Field(description="Ids of the retrieved chunks / file excerpts you relied on")
    confidence: float


class ManagerDecision(BaseModel):
    decision: str = Field(description="One of the allowed decisions")
    reason: str
    confidence: float


class EscalationSummary(BaseModel):
    summary: str = Field(description="Written summary for a human engineer: the bug, what was tried, why it failed, suggested next step")
    tried: list[str] = Field(description="One line per attempt")


class JudgeVerdict(BaseModel):
    verdict: Literal["pass", "fail", "partial"]
    reasoning: str
    score: float = Field(description="0.0-1.0")
