"""LangGraph state + per-run context registry."""

from __future__ import annotations

import threading
from typing import TypedDict

from codefix.agents.base import RunContext


class PipelineState(TypedDict, total=False):
    task_id: str
    arch: str
    next: str               # Manager's routing decision (hierarchical)
    last_event: str         # what the last worker reported back
    rediagnoses: int
    escalation_reason: str
    pr_confirmed: bool
    final_status: str
    # Inline hand-off payloads — used ONLY by sequential / p2p (no blackboard reads there).
    diagnosis: dict | None
    patch: dict | None
    patch_id: str | None
    test_report: dict | None


_CTX: dict[str, RunContext] = {}
_LOCK = threading.Lock()


def register(ctx: RunContext) -> None:
    with _LOCK:
        _CTX[ctx.task_id] = ctx


def ctx_for(state: PipelineState) -> RunContext:
    return _CTX[state["task_id"]]


def unregister(task_id: str) -> None:
    with _LOCK:
        _CTX.pop(task_id, None)
