"""Running ADK LLM agents.

Each LLM agent turn (a diagnosis, a Manager routing decision, an escalation note) runs in its
own ADK session through an `InMemoryRunner`, so the agent sees exactly the context it is given
and nothing another agent wrote. Pipeline state is shared through the Blackboard (and, for the
sequential baseline, ADK session state), not through chat history.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel

from codefix.adk_side.model import CodefixModel
from codefix.agents.base import RunContext
from codefix.observability.callbacks import adk_callbacks

APP = "codefix"


async def run_llm_agent(agent: LlmAgent, text: str, task_id: str, max_llm_calls: int = 30) -> str:
    """Run one agent on one user message; return its final text answer."""
    runner = InMemoryRunner(agent=agent, app_name=APP)
    session = await runner.session_service.create_session(app_name=APP, user_id="codefix",
                                                          state={"task_id": task_id})
    final = ""
    async for ev in runner.run_async(user_id="codefix", session_id=session.id,
                                     new_message=types.Content(role="user", parts=[types.Part(text=text)]),
                                     run_config=RunConfig(max_llm_calls=max_llm_calls)):
        if ev.content and ev.content.parts and not ev.get_function_calls() and not ev.get_function_responses():
            texts = [p.text for p in ev.content.parts if p.text and not p.thought]
            if texts:
                final = "\n".join(texts)
    return final


async def structured(ctx: RunContext, *, name: str, role: str, schema: type[BaseModel], system: str,
                     prompt: str) -> BaseModel:
    """A one-shot ADK LlmAgent whose answer is validated against `schema`."""
    model = CodefixModel(model=ctx.settings.role(role).model, task_id=ctx.task_id, role=role, agent=name,
                         schema_=schema)
    agent = LlmAgent(name=name, model=model, instruction=lambda _ctx, s=system: s,
                     description=f"CodeFix {name}", **adk_callbacks(ctx.task_id, allowed_tools=[]))
    await run_llm_agent(agent, prompt, ctx.task_id, max_llm_calls=3)
    if model.last is None:
        raise RuntimeError(f"{name} produced no answer")
    return model.last
