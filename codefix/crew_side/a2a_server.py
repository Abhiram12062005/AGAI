"""The CrewAI worker service as an A2A agent (official a2a-sdk, JSON-RPC binding).

The ADK side (Manager / pipelines) never imports the CrewAI code: it sends an A2A message
whose single data part is `{skill, task_id, payload}` and receives a data part back. Skills:

  fix         Fixer crew writes a Patch           -> {patch_id, patch}
  test        Tester applies + lints + runs tests -> {report_id, report, passed}
  pr_prepare  PR agent commits to codefix/<task>  -> {pr}
  p2p_repair  P2P peers loop fix <-> test         -> {outcome, patch_id, passed}

The same Starlette app is used in-process (httpx ASGI transport: real JSON-RPC, no socket),
mounted under /a2a/crew on the web server, or run standalone (`codefix crew-serve`).
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache

from a2a.helpers.proto_helpers import get_data_parts, new_data_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentProvider, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT
from starlette.applications import Starlette

import codefix.crew_side  # noqa: F401  (CrewAI env defaults)
from codefix.observability.callbacks import CURRENT_TASK

LOCAL_URL = "http://codefix-crew.local/"

SKILLS = [
    AgentSkill(id="fix", name="Fixer", tags=["crewai", "patch"],
               description="CrewAI Fixer agent: writes a minimal patch for a Diagnosis (guardrailed task)."),
    AgentSkill(id="test", name="Tester", tags=["crewai", "tests"],
               description="Applies a Patch in an isolated workspace, lints, runs the project's test suite."),
    AgentSkill(id="pr_prepare", name="PR agent", tags=["crewai", "git"],
               description="Commits a verified patch to a codefix/<task_id> branch (push needs a human)."),
    AgentSkill(id="p2p_repair", name="Peer repair loop", tags=["crewai", "p2p"],
               description="P2P architecture: Fixer and Tester hand off directly until green or out of attempts."),
]


def public_url() -> str:
    base = os.environ.get("CODEFIX_PUBLIC_URL")
    return f"{base.rstrip('/')}/a2a/crew/" if base else LOCAL_URL


def agent_card(url: str | None = None) -> AgentCard:
    return AgentCard(
        name="codefix-crew", version="0.2.0",
        description="CodeFix AI worker agents built with CrewAI (Fixer, Reviewer, Tester, PR).",
        provider=AgentProvider(organization="CodeFix AI (course project)", url="https://github.com/Abhiram12062005/AGAI"),
        supported_interfaces=[AgentInterface(url=url or public_url(), protocol_binding="JSONRPC",
                                             protocol_version=PROTOCOL_VERSION_CURRENT)],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["application/json"], default_output_modes=["application/json"], skills=SKILLS)


def _context_for(task_id: str):
    """In-process the run is registered; a standalone service rebuilds it from the Blackboard."""
    from codefix.graph.state import get_ctx, register

    ctx = get_ctx(task_id)
    if ctx is not None:
        return ctx
    from pathlib import Path

    from codefix.graph import runner

    run = runner.blackboard().get_run(task_id)
    if run is None:
        raise KeyError(f"unknown task {task_id}")
    if run["llm_mode"] == "mock":
        raise RuntimeError("mock runs cannot be served by a remote CrewAI service")
    ctx = runner.make_context(task_id, Path(run["repo"]), run["bug_report"], run["stack_trace"], run["arch"],
                              "live", run["max_attempts"], extra=run["meta"])
    register(ctx)
    return ctx


class CrewExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue) -> None:
        from codefix.crew_side.flows import run_skill

        data = (get_data_parts(context.message.parts) or [{}])[0]
        skill, task_id, payload = data.get("skill"), data.get("task_id"), data.get("payload") or {}
        try:
            _context_for(task_id)
            token = CURRENT_TASK.set(task_id)  # copied into the worker thread
            try:
                result = await asyncio.to_thread(run_skill, skill, task_id, payload)
            finally:
                CURRENT_TASK.reset(token)
            body = {"ok": True, "skill": skill, "result": result}
        except Exception as e:
            body = {"ok": False, "skill": skill, "error": f"{type(e).__name__}: {e}"}
        await event_queue.enqueue_event(new_data_message(body, context_id=context.context_id, task_id=context.task_id))

    async def cancel(self, context: RequestContext, event_queue) -> None:  # pragma: no cover
        raise NotImplementedError("CodeFix worker requests are not cancelable")


def build_app(url: str | None = None) -> Starlette:
    card = agent_card(url)
    handler = DefaultRequestHandler(agent_executor=CrewExecutor(), task_store=InMemoryTaskStore(), agent_card=card)
    return Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")])


@lru_cache(maxsize=1)
def local_app() -> Starlette:
    return build_app(LOCAL_URL)
