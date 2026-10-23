"""`CodefixModel`: the Google ADK model adapter (an ADK `BaseLlm`).

ADK agents talk to models through `BaseLlm.generate_content_async(LlmRequest) -> LlmResponse`.
This adapter forwards each request to CodeFix's own LLM layer (`codefix.llm`: Gemini or Claude
with per-call token/cost telemetry, or the offline mock), so the framework owns the agent loop,
tool execution and callbacks while every model call stays measured the same way.

ReAct mode (Diagnoser): the model answers each turn with an explicit structured
`{thought, action, action_input, final_answer}`. The adapter turns a tool action into an ADK
`function_call`, so ADK runs the tool (through its tool callbacks and guardrails) and feeds the
`function_response` back on the next turn, where the adapter renders it as the ReAct
"Observation". The step cap, forced final answer and invalid-action handling of the ReAct
protocol live here; the verbatim transcript is collected in a `ReactRecorder`.

Structured mode (Manager router, few-shot Diagnoser, escalation writer): one call, one
schema-validated JSON answer.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel, ConfigDict, PrivateAttr

from codefix.agents.base import ReActStep, chunk_ids, clip
from codefix.graph.state import get_ctx


@dataclass
class ReactRecorder:
    max_iter: int
    steps: int = 0
    transcript: list[dict] = field(default_factory=list)
    final: Any = None
    known_ids: set[str] = field(default_factory=set)
    files_seen: list[str] = field(default_factory=list)
    assistant_text: dict[str, str] = field(default_factory=dict)
    pending: dict[str, dict] = field(default_factory=dict)
    # turns that never reached ADK (an invalid action and its error observation), re-inserted
    # after the function response they followed so the model keeps seeing them
    extra_after: dict[str | None, list[dict]] = field(default_factory=dict)
    last_call: str | None = None


def _system_text(req: LlmRequest) -> str:
    si = req.config.system_instruction if req.config else None
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    parts = getattr(si, "parts", None) or []
    return "\n".join(p.text for p in parts if getattr(p, "text", None))


def _usage(res) -> types.GenerateContentResponseUsageMetadata:
    u = res.usage
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=u.input_tokens, candidates_token_count=u.output_tokens,
        total_token_count=u.input_tokens + u.output_tokens,
        cached_content_token_count=u.cache_read_input_tokens or None)


def _text_response(text: str, res=None) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]),
                       usage_metadata=_usage(res) if res is not None else None)


class CodefixModel(BaseLlm):
    """One instance per (run, agent): knows its task, role, output schema and mode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    task_id: str
    role: str
    agent: str
    schema_: Any
    react: bool = False
    _last: BaseModel | None = PrivateAttr(default=None)

    @property
    def last(self) -> BaseModel | None:
        return self._last

    def _legacy_output_schema_and_tools(self) -> bool:  # we never pair output_schema with tools
        return False

    async def generate_content_async(self, llm_request: LlmRequest,
                                     stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        ctx = get_ctx(self.task_id)
        if ctx is None:
            raise RuntimeError(f"no run context registered for {self.task_id}")
        system = _system_text(llm_request)
        if self.react:
            async for r in self._react_turn(ctx, system, llm_request):
                yield r
            return
        msgs = []
        for c in llm_request.contents:
            text = "\n".join(p.text for p in (c.parts or []) if p.text and not p.thought)
            if text:
                msgs.append({"role": "assistant" if c.role == "model" else "user", "content": text})
        res = await asyncio.to_thread(ctx.llm.structured, role=self.role, system=system,
                                      messages=msgs or [{"role": "user", "content": "Respond now."}],
                                      schema=self.schema_, task_id=self.task_id, agent=self.agent)
        self._last = res.parsed
        yield _text_response(res.parsed.model_dump_json(), res)

    # ------------------------------------------------------------------ ReAct
    def _history(self, ctx, req: LlmRequest, rec: ReactRecorder) -> list[dict]:
        msgs: list[dict] = []
        first_user_seen = False
        for c in req.contents:
            for p in c.parts or []:
                if p.function_call:
                    fc = p.function_call
                    msgs.append({"role": "assistant",
                                 "content": rec.assistant_text.get(fc.id or "") or json.dumps(
                                     {"action": fc.name, "action_input": json.dumps(fc.args or {})})})
                elif p.function_response:
                    fr = p.function_response
                    obs = clip(fr.response)
                    entry = rec.pending.pop(fr.id or "", None)
                    if entry is not None:
                        self._observe(ctx, rec, entry, fr.name, fr.response, obs)
                    step = (entry or {}).get("step", rec.steps)
                    msgs.append({"role": "user",
                                 "content": f"Observation:\n{obs}\n\nStep {step + 1} of {rec.max_iter}."})
                    msgs.extend(rec.extra_after.get(fr.id, []))
                elif p.text and not p.thought:
                    msgs.append({"role": "assistant" if c.role == "model" else "user", "content": p.text})
                    if c.role == "user" and not first_user_seen:
                        first_user_seen = True
                        msgs.extend(rec.extra_after.get(None, []))
        return msgs

    @staticmethod
    def _observe(ctx, rec: ReactRecorder, entry: dict, tool: str, response: Any, obs: str) -> None:
        entry["observation"] = obs
        rec.known_ids |= chunk_ids(response)
        if tool == "read_file" and isinstance(response, dict):
            inner = response.get("result", response)
            if isinstance(inner, dict) and inner.get("path"):
                rec.files_seen.append(inner["path"])
        ctx.event("react_step", agent="diagnoser", **entry)

    async def _react_turn(self, ctx, system: str, req: LlmRequest) -> AsyncGenerator[LlmResponse, None]:
        rec: ReactRecorder = ctx.extra["react_recorder"]
        messages = self._history(ctx, req, rec)
        while True:
            rec.steps += 1
            forced = rec.steps > rec.max_iter
            if forced:
                messages.append({"role": "user", "content": "Step limit reached. You MUST reply now with "
                                 "action='final_answer' and your best-effort final_answer."})
            res = await asyncio.to_thread(ctx.llm.structured, role=self.role, system=system, messages=messages,
                                          schema=ReActStep, task_id=self.task_id, agent=self.agent)
            st: ReActStep = res.parsed
            entry = {"step": rec.steps, "thought": st.thought, "action": st.action,
                     "action_input": st.action_input, "thinking_summary": res.thinking, "observation": None,
                     "forced": forced}
            assistant = {"role": "assistant", "content": res.text}
            if st.action == "final_answer" or st.final_answer is not None:
                entry["observation"] = "(final answer)"
                rec.transcript.append(entry)
                ctx.event("react_step", agent="diagnoser", **entry)
                if st.final_answer is not None:
                    rec.final = st.final_answer
                    yield _text_response(st.final_answer.model_dump_json(), res)
                    return
                if forced:
                    yield _text_response("null", res)
                    return
                nudge = [assistant, {"role": "user", "content": "final_answer was null. Provide it."}]
                messages += nudge
                rec.extra_after.setdefault(rec.last_call, []).extend(nudge)
                continue
            if forced:  # still asking for a tool after the cap: stop, the caller degrades
                rec.transcript.append(entry)
                ctx.event("react_step", agent="diagnoser", **entry)
                yield _text_response("null", res)
                return
            try:
                args = json.loads(st.action_input or "{}")
                if not isinstance(args, dict):
                    raise ValueError("action_input must be a JSON object")
                if st.action not in req.tools_dict:
                    raise ValueError(f"unknown tool '{st.action}'. Available: {', '.join(req.tools_dict)}")
            except Exception as e:  # invalid action: an error observation, like any failed tool call
                entry["observation"] = f"ERROR: {e}"
                rec.transcript.append(entry)
                ctx.event("react_step", agent="diagnoser", **entry)
                err = [assistant, {"role": "user", "content": f"Observation:\nERROR: {e}\n\n"
                                                              f"Step {rec.steps + 1} of {rec.max_iter}."}]
                messages += err
                rec.extra_after.setdefault(rec.last_call, []).extend(err)
                continue
            call_id = f"react-{self.task_id}-{rec.steps}"
            rec.assistant_text[call_id] = res.text
            rec.pending[call_id] = entry
            rec.transcript.append(entry)
            rec.last_call = call_id
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(
                function_call=types.FunctionCall(id=call_id, name=st.action, args=args))]), usage_metadata=_usage(res))
            return
