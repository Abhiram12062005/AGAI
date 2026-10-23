"""Per-run context registry.

Agents on both sides of the A2A boundary look up the live RunContext (Blackboard, MCP toolbox,
LLM client) by task_id. When the CrewAI service runs in another process, it rebuilds the context
from the Blackboard run record instead (see codefix/crew_side/a2a_server.py).
"""

from __future__ import annotations

import threading

from codefix.agents.base import RunContext

_CTX: dict[str, RunContext] = {}
_LOCK = threading.Lock()


def register(ctx: RunContext) -> None:
    with _LOCK:
        _CTX[ctx.task_id] = ctx


def get_ctx(task_id: str) -> RunContext | None:
    with _LOCK:
        return _CTX.get(task_id)


def unregister(task_id: str) -> None:
    with _LOCK:
        _CTX.pop(task_id, None)
