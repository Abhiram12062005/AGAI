"""Task monitoring through the frameworks' own callback hooks.

ADK   - before/after agent, before/after model, before/after tool and tool-error callbacks
        (`adk_callbacks`). The tool callbacks also enforce the tool-output guardrails.
CrewAI - an event-bus listener (`CrewTraceListener`) for crew/task/agent/LLM/flow/guardrail
        events, plus per-crew `step_callback` / `task_callback` (`crew_callbacks`).

Every hook writes a `callback` event (framework, hook, agent, details, latency) to the run's
JSONL trace, which feeds the live view and the dashboard. Hooks never raise into the
framework: a monitoring failure must not break a run.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from typing import Any

from codefix import guardrails

# The CrewAI event bus runs sync handlers in a thread pool with a copy of the caller's context,
# so the active task id travels with the event.
CURRENT_TASK: contextvars.ContextVar[str | None] = contextvars.ContextVar("codefix_task", default=None)


def _ctx(task_id: str | None):
    from codefix.graph.state import get_ctx

    return get_ctx(task_id) if task_id else None


def _emit(task_id: str | None, framework: str, hook: str, **data: Any) -> None:
    try:
        ctx = _ctx(task_id)
        if ctx is not None:
            ctx.event("callback", framework=framework, hook=hook, **data)
    except Exception:  # pragma: no cover - monitoring must never break the run
        pass


def _short(obj: Any, n: int = 300) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + "..."


# ------------------------------------------------------------------ ADK
def adk_callbacks(task_id: str, allowed_tools: list[str] | None = None) -> dict[str, Any]:
    """Keyword arguments for an ADK LlmAgent: monitoring + tool guardrails."""
    starts: dict[str, float] = {}
    lock = threading.Lock()

    def _mark(key: str) -> None:
        with lock:
            starts[key] = time.time()

    def _since(key: str) -> int | None:
        with lock:
            t = starts.pop(key, None)
        return int((time.time() - t) * 1000) if t else None

    def before_agent(callback_context):
        _mark(f"agent:{callback_context.agent_name}")
        _emit(task_id, "adk", "before_agent", agent=callback_context.agent_name,
              invocation=callback_context.invocation_id)
        return None

    def after_agent(callback_context):
        _emit(task_id, "adk", "after_agent", agent=callback_context.agent_name,
              latency_ms=_since(f"agent:{callback_context.agent_name}"))
        return None

    def before_model(callback_context, llm_request):
        _mark(f"model:{callback_context.agent_name}")
        _emit(task_id, "adk", "before_model", agent=callback_context.agent_name,
              contents=len(llm_request.contents), tools=sorted(llm_request.tools_dict))
        return None

    def after_model(callback_context, llm_response):
        parts = llm_response.content.parts if llm_response.content else []
        calls = [p.function_call.name for p in parts if p.function_call]
        _emit(task_id, "adk", "after_model", agent=callback_context.agent_name, function_calls=calls,
              text_chars=sum(len(p.text or "") for p in parts), error=llm_response.error_message,
              latency_ms=_since(f"model:{callback_context.agent_name}"))
        return None

    def before_tool(tool, args, tool_context):
        _mark(f"tool:{tool.name}")
        if allowed_tools is not None and tool.name not in allowed_tools:
            _emit(task_id, "adk", "before_tool", agent=tool_context.agent_name, tool=tool.name, blocked=True)
            return {"error": f"tool '{tool.name}' is not allowed for this agent"}
        if not isinstance(args, dict):
            return {"error": "tool arguments must be a JSON object"}
        _emit(task_id, "adk", "before_tool", agent=tool_context.agent_name, tool=tool.name, args=_short(args))
        return None

    def after_tool(tool, args, tool_context, tool_response):
        latency = _since(f"tool:{tool.name}")
        check = guardrails.validate_tool_output(tool.name, tool_response)
        if not check.allowed:
            _emit(task_id, "adk", "after_tool", agent=tool_context.agent_name, tool=tool.name,
                  latency_ms=latency, guardrail=check.model_dump())
            _guardrail_event(task_id, "tool_output", check)
            return {"error": "; ".join(check.reasons)}
        text = json.dumps(tool_response, default=str) if not isinstance(tool_response, str) else tool_response
        untrusted = guardrails.screen_untrusted(text)
        _emit(task_id, "adk", "after_tool", agent=tool_context.agent_name, tool=tool.name, latency_ms=latency,
              result_chars=len(text), flagged=untrusted.category == "untrusted_content")
        if untrusted.category == "untrusted_content":
            _guardrail_event(task_id, "untrusted_content", untrusted)
            return {"guardrail_warning": guardrails.UNTRUSTED_BANNER, "result": tool_response}
        return None

    def on_tool_error(tool, args, tool_context, error):
        _emit(task_id, "adk", "on_tool_error", agent=tool_context.agent_name, tool=tool.name,
              error=_short(str(error)), latency_ms=_since(f"tool:{tool.name}"))
        return {"error": f"{type(error).__name__}: {error}"}

    return {"before_agent_callback": before_agent, "after_agent_callback": after_agent,
            "before_model_callback": before_model, "after_model_callback": after_model,
            "before_tool_callback": before_tool, "after_tool_callback": after_tool,
            "on_tool_error_callback": on_tool_error}


def _guardrail_event(task_id: str | None, stage: str, result: guardrails.GuardrailResult) -> None:
    ctx = _ctx(task_id)
    if ctx is not None:
        ctx.event("guardrail", stage=stage, **result.model_dump())


# ------------------------------------------------------------------ CrewAI
def crew_callbacks(task_id: str) -> dict[str, Any]:
    """step_callback / task_callback for a CrewAI Crew."""

    def step(step_output):
        _emit(task_id, "crewai", "step_callback", step=type(step_output).__name__,
              thought=_short(getattr(step_output, "thought", "") or "", 200),
              output=_short(getattr(step_output, "output", None) or getattr(step_output, "text", "") or "", 300))

    def task_done(task_output):
        _emit(task_id, "crewai", "task_callback", task=getattr(task_output, "name", None) or "",
              agent=getattr(task_output, "agent", ""), output=_short(task_output.raw or "", 300))

    return {"step_callback": step, "task_callback": task_done}


_LISTENER = None
_LISTENER_LOCK = threading.Lock()


def install_crew_listener() -> None:
    """Register (once per process) a CrewAI event listener that traces framework events."""
    global _LISTENER
    with _LISTENER_LOCK:
        if _LISTENER is not None:
            return
        from crewai.events import BaseEventListener
        from crewai.events.types import crew_events, flow_events, llm_events, task_events

        try:
            from crewai.events.types import llm_guardrail_events as g
            guard_events = [getattr(g, n) for n in dir(g) if n.endswith("Event") and n.startswith("LLMGuardrail")]
        except Exception:  # pragma: no cover
            guard_events = []
        watched = [crew_events.CrewKickoffStartedEvent, crew_events.CrewKickoffCompletedEvent,
                   crew_events.CrewKickoffFailedEvent, task_events.TaskStartedEvent,
                   task_events.TaskCompletedEvent, task_events.TaskFailedEvent,
                   llm_events.LLMCallStartedEvent, llm_events.LLMCallCompletedEvent, llm_events.LLMCallFailedEvent,
                   flow_events.FlowStartedEvent, flow_events.FlowFinishedEvent,
                   flow_events.MethodExecutionStartedEvent, flow_events.MethodExecutionFinishedEvent,
                   flow_events.MethodExecutionFailedEvent, *guard_events]

        class CrewTraceListener(BaseEventListener):
            def setup_listeners(self, bus):
                for cls in watched:
                    bus.on(cls)(self._handler(cls.__name__))

            @staticmethod
            def _handler(name: str):
                def handle(source, event):
                    details = {}
                    for attr in ("task_name", "agent_role", "flow_name", "method_name", "model", "error",
                                 "retry_count", "success"):
                        v = getattr(event, attr, None)
                        if v not in (None, ""):
                            details[attr] = _short(v, 200) if not isinstance(v, (int, float, bool)) else v
                    _emit(CURRENT_TASK.get(), "crewai", name, **details)
                return handle

        _LISTENER = CrewTraceListener()
