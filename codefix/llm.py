"""LLM access layer with per-call model/token/cost telemetry.

* One model per role, from config/models.yaml (swappable with no code change).
* Structured outputs (`output_config.format` JSON schema) for every agent response, validated
  again with Pydantic on our side.
* Anthropic and Google AI Studio providers.
* Every call -> llm_calls row + `llm_call` JSONL trace event (model, served model, tokens,
  cache tokens, USD cost, latency, stop reason).
* `MockLLM` implements the same interface for tests/offline demos (no API key needed).
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from codefix.config import RoleConfig, Settings
from codefix.observability.pricing import Usage, cost_usd

FALLBACK_BETA = "server-side-fallback-2026-07-01"
_UNSUPPORTED_SCHEMA_KEYS = {"title", "default", "minimum", "maximum", "exclusiveMinimum",
                            "exclusiveMaximum", "minLength", "maxLength", "minItems", "maxItems",
                            "pattern", "format", "examples"}


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


@dataclass
class LLMResult:
    parsed: BaseModel
    text: str
    thinking: str = ""
    model: str = ""
    served_model: str = ""
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: int = 0
    stop_reason: str | None = None
    assistant_content: Any = None  # blocks to append verbatim for multi-turn (preserves thinking)


def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic -> JSON schema accepted by structured outputs (inline refs, closed objects)."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            out = {k: walk(v) for k, v in node.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
            if out.get("type") == "object" and "properties" in out:
                out["additionalProperties"] = False
                out["required"] = list(out["properties"])
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(raw)


def extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
        if not m:
            raise
        return json.loads(m.group(1))


Telemetry = Callable[[dict], None]


class BaseLLM:
    mode = "base"

    def __init__(self, settings: Settings, telemetry: Telemetry | None = None):
        self.settings = settings
        self.telemetry = telemetry

    def role(self, role: str) -> RoleConfig:
        return self.settings.role(role)

    def _emit(self, task_id: str | None, agent: str, result: LLMResult | None, model: str,
              latency_ms: int, error: str | None = None) -> None:
        if not (self.telemetry and task_id):
            return
        u = result.usage if result else Usage()
        self.telemetry({
            "task_id": task_id, "agent": agent, "model": model,
            "served_model": result.served_model if result else None,
            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cache_read_tokens": u.cache_read_input_tokens, "cache_write_tokens": u.cache_creation_input_tokens,
            "cost_usd": result.cost_usd if result else 0.0, "latency_ms": latency_ms,
            "stop_reason": result.stop_reason if result else None, "error": error,
        })

    def structured(self, *, role: str, system: str, messages: list[dict], schema: type[BaseModel],
                   task_id: str | None = None, agent: str | None = None) -> LLMResult:
        raise NotImplementedError


class AnthropicLLM(BaseLLM):
    mode = "live"

    def __init__(self, settings: Settings, telemetry: Telemetry | None = None):
        super().__init__(settings, telemetry)
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4, timeout=600)

    def _request(self, rc: RoleConfig, system: str, messages: list[dict], schema: dict) -> Any:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if rc.effort:
            output_config["effort"] = rc.effort
        kwargs: dict[str, Any] = {
            "model": rc.model,
            "max_tokens": rc.max_tokens,
            # stable system prompt first; auto-caching extends the cached prefix every ReAct step
            "system": [{"type": "text", "text": system}],
            "messages": messages,
            "output_config": output_config,
            "cache_control": {"type": "ephemeral"},
        }
        if rc.thinking_display:
            kwargs["thinking"] = {"type": "adaptive", "display": rc.thinking_display}
        if rc.fallbacks:
            return self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks=rc.fallbacks, **kwargs)
        return self.client.messages.create(**kwargs)

    def structured(self, *, role: str, system: str, messages: list[dict], schema: type[BaseModel],
                   task_id: str | None = None, agent: str | None = None) -> LLMResult:
        rc = self.role(role)
        agent = agent or role
        js = strict_schema(schema)
        started = time.time()
        msgs = list(messages)
        last_err: Exception | None = None
        for _ in range(2):  # one repair retry on invalid output
            try:
                resp = self._request(rc, system, msgs, js)
            except self._anthropic.APIStatusError as e:
                self._emit(task_id, agent, None, rc.model, int((time.time() - started) * 1000),
                           f"{e.status_code}: {e.message}")
                raise LLMError(f"{rc.model} API error {e.status_code}: {e.message}") from e
            except self._anthropic.APIConnectionError as e:
                self._emit(task_id, agent, None, rc.model, int((time.time() - started) * 1000), "connection")
                raise LLMError(f"connection error calling {rc.model}") from e
            latency = int((time.time() - started) * 1000)
            u = resp.usage
            usage = Usage(u.input_tokens or 0, u.output_tokens or 0,
                          getattr(u, "cache_read_input_tokens", 0) or 0,
                          getattr(u, "cache_creation_input_tokens", 0) or 0)
            served = getattr(resp, "model", rc.model) or rc.model
            text = "".join(b.text for b in resp.content if b.type == "text")
            thinking = "\n".join(getattr(b, "thinking", "") or "" for b in resp.content if b.type == "thinking")
            result = LLMResult(parsed=None, text=text, thinking=thinking.strip(), model=rc.model,  # type: ignore[arg-type]
                               served_model=served, usage=usage,
                               cost_usd=cost_usd(served, usage, self.settings.pricing),
                               latency_ms=latency, stop_reason=resp.stop_reason,
                               assistant_content=[b.model_dump(exclude_none=True) for b in resp.content])
            if resp.stop_reason == "refusal":
                self._emit(task_id, agent, result, rc.model, latency, "refusal")
                details = getattr(resp, "stop_details", None)
                raise LLMRefusal(f"{served} refused: {getattr(details, 'category', None)}")
            try:
                result.parsed = schema.model_validate(extract_json(text))
                self._emit(task_id, agent, result, rc.model, latency)
                return result
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                self._emit(task_id, agent, result, rc.model, latency, f"invalid output: {str(e)[:200]}")
                msgs = msgs + [
                    {"role": "assistant", "content": result.assistant_content},
                    {"role": "user", "content": f"That output did not validate ({str(e)[:500]}). "
                                                "Reply again with only valid JSON for the schema."},
                ]
        raise LLMError(f"{rc.model} produced invalid structured output: {last_err}")


class GoogleLLM(BaseLLM):
    """Google Gemini via the OpenAI-compatible Gemini API."""

    mode = "live"
    endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    def __init__(self, settings: Settings, telemetry: Telemetry | None = None):
        super().__init__(settings, telemetry)
        import httpx

        self._httpx = httpx
        self.client = httpx.Client(
            timeout=600,
            headers={"Authorization": f"Bearer {os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY', '')}"},
        )

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(block["text"]) for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            )
        return json.dumps(content, ensure_ascii=False)

    def _request(self, rc: RoleConfig, system: str, messages: list[dict], schema: dict) -> Any:
        request_messages = [{"role": "system", "content": system}]
        request_messages.extend(
            {"role": message["role"], "content": self._content_text(message.get("content", ""))}
            for message in messages
        )
        schema_prompt = (
            "\n\nReturn only a JSON object conforming to this JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        request_messages[0]["content"] += schema_prompt
        payload: dict[str, Any] = {
            "model": rc.model,
            "messages": request_messages,
            "max_tokens": rc.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if rc.effort:
            payload["reasoning_effort"] = rc.effort
        return self.client.post(self.endpoint, json=payload)

    def structured(self, *, role: str, system: str, messages: list[dict], schema: type[BaseModel],
                   task_id: str | None = None, agent: str | None = None) -> LLMResult:
        rc = self.role(role)
        agent = agent or role
        js = strict_schema(schema)
        started = time.time()
        msgs = list(messages)
        last_err: Exception | None = None
        for _ in range(2):
            try:
                resp = self._request(rc, system, msgs, js)
                if resp.is_error:
                    try:
                        detail = resp.json().get("error", {}).get("message", resp.text)
                    except (ValueError, AttributeError):
                        detail = resp.text
                    self._emit(task_id, agent, None, rc.model, int((time.time() - started) * 1000),
                               f"{resp.status_code}: {detail}")
                    raise LLMError(f"{rc.model} API error {resp.status_code}: {detail}")
                data = resp.json()
            except LLMError:
                raise
            except self._httpx.RequestError as e:
                self._emit(task_id, agent, None, rc.model, int((time.time() - started) * 1000), "connection")
                raise LLMError(f"connection error calling {rc.model}: {e}") from e
            except (ValueError, KeyError, IndexError, TypeError) as e:
                self._emit(task_id, agent, None, rc.model, int((time.time() - started) * 1000),
                           f"invalid API response: {e}")
                raise LLMError(f"{rc.model} returned an invalid API response: {e}") from e

            latency = int((time.time() - started) * 1000)
            choice = data["choices"][0]
            answer = choice["message"]
            text = self._content_text(answer.get("content") or "")
            usage_data = data.get("usage") or {}
            usage = Usage(
                input_tokens=usage_data.get("prompt_tokens", 0) or 0,
                output_tokens=usage_data.get("completion_tokens", 0) or 0,
                cache_read_input_tokens=(usage_data.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
            )
            result = LLMResult(
                parsed=None,  # type: ignore[arg-type]
                text=text,
                thinking=self._content_text(answer.get("reasoning_content") or ""),
                model=rc.model,
                served_model=data.get("model") or rc.model,
                usage=usage,
                cost_usd=cost_usd(data.get("model") or rc.model, usage, self.settings.pricing),
                latency_ms=latency,
                stop_reason=choice.get("finish_reason"),
                assistant_content=[{"type": "text", "text": text}],
            )
            if answer.get("refusal") or result.stop_reason == "content_filter":
                self._emit(task_id, agent, result, rc.model, latency, "refusal")
                raise LLMRefusal(f"{result.served_model} refused the request")
            try:
                result.parsed = schema.model_validate(extract_json(text))
                self._emit(task_id, agent, result, rc.model, latency)
                return result
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                self._emit(task_id, agent, result, rc.model, latency, f"invalid output: {str(e)[:200]}")
                msgs = msgs + [
                    {"role": "assistant", "content": result.assistant_content},
                    {"role": "user", "content": f"That output did not validate ({str(e)[:500]}). "
                                                  "Reply again with only valid JSON for the schema."},
                ]
        raise LLMError(f"{rc.model} produced invalid structured output: {last_err}")


MockHandler = Callable[[str, str, list[dict], type[BaseModel], int], dict]


class MockLLM(BaseLLM):
    """Deterministic stand-in: `handler(role, system, messages, schema, call_no) -> dict`."""

    mode = "mock"

    def __init__(self, settings: Settings, handler: MockHandler, telemetry: Telemetry | None = None):
        super().__init__(settings, telemetry)
        self.handler = handler
        self.calls: list[dict] = []

    def structured(self, *, role: str, system: str, messages: list[dict], schema: type[BaseModel],
                   task_id: str | None = None, agent: str | None = None) -> LLMResult:
        started = time.time()
        data = self.handler(role, system, messages, schema, len(self.calls))
        self.calls.append({"role": role, "schema": schema.__name__, "messages": messages})
        text = json.dumps(data)
        usage = Usage(input_tokens=sum(len(str(m.get("content", ""))) for m in messages) // 4,
                      output_tokens=len(text) // 4)
        result = LLMResult(parsed=schema.model_validate(data), text=text, model="mock",
                           served_model=f"mock:{self.role(role).model}", usage=usage, cost_usd=0.0,
                           latency_ms=int((time.time() - started) * 1000), stop_reason="end_turn",
                           assistant_content=[{"type": "text", "text": text}])
        self._emit(task_id, agent or role, result, "mock", result.latency_ms)
        return result


def make_llm(mode: str, settings: Settings, telemetry: Telemetry | None = None,
             handler: MockHandler | None = None) -> BaseLLM:
    if mode == "mock":
        if handler is None:
            raise ValueError("mock mode needs a handler")
        return MockLLM(settings, handler, telemetry)
    if settings.provider == "google":
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise LLMError("GEMINI_API_KEY is not set (get a free key from Google AI Studio) — or run with --llm mock")
        return GoogleLLM(settings, telemetry)
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise LLMError("ANTHROPIC_API_KEY is not set (put it in .env) — or run with --llm mock")
    return AnthropicLLM(settings, telemetry)
