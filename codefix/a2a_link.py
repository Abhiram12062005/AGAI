"""A2A client used by the ADK side to reach the CrewAI worker agents.

Transport:
  * default: the CrewAI A2A app in-process via httpx's ASGI transport (real A2A JSON-RPC
    messages, no socket), so the CLI, tests and benchmarks need no second server;
  * `CODEFIX_CREW_A2A_URL=https://host/a2a/crew/`: a separately running CrewAI service
    (its agent card is fetched from /.well-known/agent-card.json).

Every exchange is traced as `a2a_request` / `a2a_response` (skill, message id, latency, ok),
which is the provenance record for work done on the other side of the framework boundary.
"""

from __future__ import annotations

import os
import time
import uuid

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers.proto_helpers import get_data_parts, new_data_part, new_message
from a2a.types import Role, SendMessageRequest

from codefix.agents.base import RunContext

TIMEOUT_S = 1800  # a Fixer turn on a slow model, or a full test suite, can take minutes


class A2AError(RuntimeError):
    pass


async def call_crew(ctx: RunContext, skill: str, payload: dict) -> dict:
    url = os.environ.get("CODEFIX_CREW_A2A_URL")
    msg = new_message([new_data_part({"skill": skill, "task_id": ctx.task_id, "payload": payload})],
                      role=Role.ROLE_USER, context_id=ctx.task_id)
    msg.message_id = f"{ctx.task_id}-{skill}-{uuid.uuid4().hex[:8]}"
    ctx.event("a2a_request", sender="adk", target="codefix-crew", skill=skill, message_id=msg.message_id,
              transport="http" if url else "in-process", bytes=len(msg.SerializeToString()))
    started = time.time()
    if url:
        http = httpx.AsyncClient(timeout=TIMEOUT_S)
        factory = ClientFactory(ClientConfig(httpx_client=http, streaming=False))
        client = await factory.create_from_url(url)
    else:
        from codefix.crew_side.a2a_server import LOCAL_URL, agent_card, local_app

        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=local_app()), base_url=LOCAL_URL,
                                 timeout=TIMEOUT_S)
        client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(agent_card(LOCAL_URL))
    body: dict = {}
    try:
        async for resp in client.send_message(SendMessageRequest(message=msg)):
            if resp.HasField("message"):
                parts = get_data_parts(resp.message.parts)
                body = parts[0] if parts else {}
    finally:
        await http.aclose()
    ok = bool(body.get("ok"))
    ctx.event("a2a_response", sender="codefix-crew", target="adk", skill=skill, message_id=msg.message_id,
              ok=ok, latency_ms=int((time.time() - started) * 1000), error=body.get("error"))
    if not ok:
        raise A2AError(body.get("error") or f"A2A call '{skill}' returned no result")
    return body["result"]
