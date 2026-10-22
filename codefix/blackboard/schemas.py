"""Typed message + artifact schemas.

Every inter-agent exchange is an `Envelope` written to the Blackboard. Payloads are
strictly typed per pipeline stage and discriminated by `kind` so an envelope can be
round-tripped through SQLite/JSONL and re-validated losslessly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

AgentName = Literal["manager", "diagnoser", "fixer", "tester", "pr", "human", "system"]
Status = Literal["dispatched", "in_progress", "success", "failure"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- payloads


class Diagnosis(_Strict):
    kind: Literal["diagnosis"] = "diagnosis"
    root_cause: str
    implicated_files: list[str] = Field(default_factory=list)
    implicated_symbols: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    # RAG chunk ids the diagnosis relied on (surfaced in the UI as citations).
    citations: list[str] = Field(default_factory=list)
    # True when the ReAct loop hit its iteration cap and this is a best-effort answer.
    degraded: bool = False
    react_steps: int = 0


class FileEdit(_Strict):
    path: str
    find: str
    replace: str


class Patch(_Strict):
    kind: Literal["patch"] = "patch"
    diff: str
    rationale: str
    files_touched: list[str] = Field(default_factory=list)
    attempt_no: int = Field(ge=1)
    citations: list[str] = Field(default_factory=list)
    # Non-empty when the Fixer's edits could not be turned into a valid diff.
    generation_error: str | None = None


class LintFinding(_Strict):
    path: str
    line: int
    code: str
    message: str


class TestReport(_Strict):
    __test__ = False  # keep pytest from collecting this class

    kind: Literal["test_report"] = "test_report"
    passed: bool
    total: int = 0
    failed: int = 0
    failing_assertions: list[str] = Field(default_factory=list)
    stack_trace: str = ""
    lint_findings: list[LintFinding] = Field(default_factory=list)
    duration_ms: int = 0
    # Which stage of the test harness produced a failure: apply | lint | tests | timeout | infra
    failure_stage: str | None = None
    patch_artifact_id: str | None = None
    workspace: str | None = None
    attempt_no: int = 0


class DispatchRef(_Strict):
    """Manager -> agent dispatch.

    DESIGN POINT (bounded message size): a dispatch never re-sends artifact content.
    It references prior artifacts by (task_id, artifact_id); the receiving agent pulls
    them from the Blackboard. Message size therefore stays O(1) in the number of retries
    instead of growing with every diff/test log that has been produced so far.
    """

    kind: Literal["dispatch"] = "dispatch"
    instruction: str
    artifact_refs: dict[str, str] = Field(default_factory=dict)  # role -> artifact_id
    attempt_no: int = 0


class Escalation(_Strict):
    kind: Literal["escalation"] = "escalation"
    reason: str
    summary: str
    attempts: int
    tried: list[str] = Field(default_factory=list)


class PRResult(_Strict):
    kind: Literal["pr_result"] = "pr_result"
    branch: str
    commit_sha: str | None = None
    url: str | None = None
    pushed: bool = False
    note: str = ""


Payload = Annotated[
    Union[Diagnosis, Patch, TestReport, DispatchRef, Escalation, PRResult],
    Field(discriminator="kind"),
]

PAYLOAD_TYPES: dict[str, type[BaseModel]] = {
    "diagnosis": Diagnosis,
    "patch": Patch,
    "test_report": TestReport,
    "dispatch": DispatchRef,
    "escalation": Escalation,
    "pr_result": PRResult,
}


class Envelope(_Strict):
    task_id: str
    sender: AgentName
    target_agent: AgentName
    status: Status
    payload: Payload
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=utcnow)
    latency_ms: int | None = None  # filled on completion (success/failure)
    artifact_id: str | None = None  # set by the Blackboard when the payload is stored

    @field_validator("task_id")
    @classmethod
    def _task_id_safe(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError("task_id must be alphanumeric/-/_ only")
        return v


class BugReport(_Strict):
    text: str
    repo: str
    stack_trace: str = ""
