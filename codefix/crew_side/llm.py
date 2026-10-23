"""`CodefixCrewLLM`: the CrewAI LLM adapter (a CrewAI `BaseLLM`).

CrewAI builds the prompt from the Agent (role/goal/backstory) and the Task (description,
expected output, output_pydantic schema) and calls `llm.call(messages, response_model=...)`.
This adapter sends that prompt to CodeFix's own LLM layer, which returns schema-validated JSON
with per-call token/cost telemetry (or the offline mock answer).
"""

from __future__ import annotations

from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel

from codefix.graph.state import get_ctx


class CodefixCrewLLM(BaseLLM):
    task_id: str
    role: str
    agent: str
    default_schema: Any = None
    last: Any = None

    def call(self, messages: str | list[dict], tools: list[dict] | None = None, callbacks: list[Any] | None = None,
             available_functions: dict[str, Any] | None = None, from_task: Any = None, from_agent: Any = None,
             response_model: type[BaseModel] | None = None, **kwargs: Any) -> str | Any:
        ctx = get_ctx(self.task_id)
        if ctx is None:
            raise RuntimeError(f"no run context registered for {self.task_id}")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        system = "\n\n".join(str(m["content"]) for m in messages if m["role"] == "system")
        convo = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        schema = response_model or self.default_schema
        res = ctx.llm.structured(role=self.role, system=system, messages=convo or [{"role": "user", "content": ""}],
                                 schema=schema, task_id=self.task_id, agent=self.agent)
        self.last = res.parsed
        body = res.parsed.model_dump_json()
        # with a response_model CrewAI expects the JSON itself; otherwise its ReAct-style executor
        # expects a "Final Answer:" line
        return body if response_model else f"Thought: I have the answer.\nFinal Answer: {body}"

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 1_000_000

    def to_config_dict(self) -> dict[str, Any]:
        return {"model": self.model, "role": self.role}

