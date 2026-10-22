"""AnthropicLLM request construction / response handling against a fake SDK client (no network)."""

import json
from types import SimpleNamespace

import pytest

from codefix import config
from codefix.agents.base import JudgeVerdict, ReActStep
from codefix.llm import (
    FALLBACK_BETA,
    AnthropicLLM,
    GoogleLLM,
    LLMError,
    LLMRefusal,
    make_llm,
    strict_schema,
)


class Block(SimpleNamespace):
    def model_dump(self, exclude_none=False):
        return {k: v for k, v in vars(self).items() if not (exclude_none and v is None)}


def message(text, stop="end_turn", model="claude-opus-5-5", thinking=None):
    content = ([Block(type="thinking", thinking=thinking, signature="sig")] if thinking else []) + \
              [Block(type="text", text=text, citations=None)]
    return SimpleNamespace(content=content, stop_reason=stop, model=model, stop_details=None,
                           usage=SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=500,
                                                 cache_creation_input_tokens=0))


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = SimpleNamespace(create=self._create("plain"))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create("beta")))

    def _create(self, kind):
        def create(**kw):
            self.calls.append((kind, kw))
            return self.replies.pop(0)
        return create


@pytest.fixture()
def llm(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-used")
    rows = []
    settings = config.load_settings()
    settings.provider = "anthropic"
    settings.roles["manager"] = settings.roles["manager"].model_copy(
        update={"model": "claude-haiku-5-5", "effort": "low", "fallbacks": None}
    )
    settings.roles["diagnoser"] = settings.roles["diagnoser"].model_copy(
        update={"model": "claude-opus-5-5", "effort": "high", "thinking_display": "summarized",
                "fallbacks": "default"}
    )
    settings.roles["fixer"] = settings.roles["fixer"].model_copy(
        update={"model": "claude-sonnet-5-5", "effort": "medium", "fallbacks": "default"}
    )
    obj = AnthropicLLM(settings, telemetry=rows.append)
    obj.rows = rows
    return obj


GOOD_STEP = {"thought": "t", "action": "read_file", "action_input": "{}", "final_answer": None}


def test_strict_schema_is_closed_and_inlined():
    s = strict_schema(ReActStep)
    assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
    fa = s["properties"]["final_answer"]["anyOf"][0]
    assert fa["type"] == "object" and fa["additionalProperties"] is False and "$ref" not in json.dumps(s)
    assert strict_schema(JudgeVerdict)["properties"]["verdict"]["enum"] == ["pass", "fail", "partial"]


def test_diagnoser_request_uses_beta_fallback_effort_and_thinking(llm):
    llm.client = FakeClient([message(json.dumps(GOOD_STEP), thinking="summary")])
    res = llm.structured(role="diagnoser", system="sys", messages=[{"role": "user", "content": "hi"}],
                         schema=ReActStep, task_id="t1", agent="diagnoser")
    kind, kw = llm.client.calls[0]
    assert kind == "beta" and kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"]["effort"] == "high"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "temperature" not in kw and "tool_choice" not in kw  # rejected on these models
    assert res.parsed.action == "read_file" and res.thinking == "summary"
    assert res.assistant_content[0]["type"] == "thinking"  # echoed verbatim on the next turn
    # 1000 in * $4 + 200 out * $20 + 500 cache-read * $0.20 per MTok
    assert res.cost_usd == pytest.approx((1000 * 4 + 200 * 20 + 500 * 0.2) / 1e6)
    assert llm.rows[0]["agent"] == "diagnoser" and llm.rows[0]["cost_usd"] == res.cost_usd


def test_manager_has_no_fallback_and_no_thinking(llm):
    llm.client = FakeClient([message('{"verdict":"pass","reasoning":"r","score":1}', model="claude-haiku-5-5")])
    llm.structured(role="manager", system="s", messages=[{"role": "user", "content": "x"}], schema=JudgeVerdict)
    kind, kw = llm.client.calls[0]
    assert kind == "plain" and "fallbacks" not in kw and "thinking" not in kw
    assert kw["output_config"]["effort"] == "low"


def test_invalid_output_gets_one_repair_turn(llm):
    llm.client = FakeClient([message("not json"), message(json.dumps(GOOD_STEP))])
    res = llm.structured(role="fixer", system="s", messages=[{"role": "user", "content": "x"}], schema=ReActStep,
                         task_id="t1")
    assert res.parsed.thought == "t" and len(llm.client.calls) == 2
    second = llm.client.calls[1][1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-1]["role"] == "user"
    assert llm.rows[0]["error"].startswith("invalid output")


def test_invalid_twice_raises(llm):
    llm.client = FakeClient([message("nope"), message("still nope")])
    with pytest.raises(LLMError):
        llm.structured(role="fixer", system="s", messages=[{"role": "user", "content": "x"}], schema=ReActStep)


def test_refusal_raises(llm):
    llm.client = FakeClient([message("", stop="refusal")])
    with pytest.raises(LLMRefusal):
        llm.structured(role="fixer", system="s", messages=[{"role": "user", "content": "x"}], schema=ReActStep)


def test_fallback_served_model_priced_by_served_model(llm):
    llm.client = FakeClient([message(json.dumps(GOOD_STEP), model="claude-opus-4-8")])
    res = llm.structured(role="fixer", system="s", messages=[{"role": "user", "content": "x"}], schema=ReActStep)
    assert res.model == "claude-sonnet-5-5" and res.served_model == "claude-opus-4-8"
    assert res.cost_usd == pytest.approx((1000 * 5 + 200 * 25 + 500 * 0.5) / 1e6)


class FakeGoogleResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.is_error = status_code >= 400
        self.text = json.dumps(data)

    def json(self):
        return self._data


class FakeGoogleClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.replies.pop(0)


def google_response(text, model="gemini-3.8-flash"):
    return FakeGoogleResponse({
        "model": model,
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 100},
    })


def test_default_config_selects_good_free_tier_models():
    settings = config.load_settings()

    assert settings.provider == "google"
    assert settings.role("diagnoser").model == "gemini-3.8-flash"
    assert settings.role("fixer").model == "gemini-3.8-flash"
    assert settings.role("judge").model == "gemini-3.1-flash-lite"


def test_google_structured_request_uses_free_provider_and_json_schema(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-used")
    rows = []
    llm = GoogleLLM(config.load_settings(), telemetry=rows.append)
    llm.client = FakeGoogleClient([google_response(json.dumps(GOOD_STEP))])
    res = llm.structured(
        role="diagnoser",
        system="system prompt",
        messages=[{"role": "user", "content": "inspect"}, {"role": "assistant", "content": [
            {"type": "text", "text": "previous response"},
        ]}],
        schema=ReActStep,
        task_id="google-1",
    )

    url, call = llm.client.calls[0]
    payload = call["json"]
    assert url == GoogleLLM.endpoint
    assert payload["model"] == "gemini-3.8-flash"
    assert payload["reasoning_effort"] == "high"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["role"] == "system"
    assert "JSON Schema" in payload["messages"][0]["content"]
    assert payload["messages"][2] == {"role": "assistant", "content": "previous response"}
    assert res.parsed.action == "read_file"
    assert res.cost_usd == 0
    assert rows[0]["model"] == "gemini-3.8-flash"


def test_google_structured_output_repairs_invalid_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-used")
    llm = GoogleLLM(config.load_settings())
    llm.client = FakeGoogleClient([google_response("not json"), google_response(json.dumps(GOOD_STEP))])

    result = llm.structured(role="fixer", system="s", messages=[{"role": "user", "content": "x"}],
                            schema=ReActStep)

    assert result.parsed.action == "read_file"
    assert len(llm.client.calls) == 2
    assert llm.client.calls[1][1]["json"]["messages"][-1]["role"] == "user"


def test_google_api_error_is_reported(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-used")
    llm = GoogleLLM(config.load_settings())
    llm.client = FakeGoogleClient([FakeGoogleResponse({"error": {"message": "bad API key"}}, 401)])

    with pytest.raises(LLMError, match="API error 401: bad API key"):
        llm.structured(role="diagnoser", system="s", messages=[], schema=ReActStep)


def test_make_llm_requires_google_key_for_default_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(LLMError, match="GEMINI_API_KEY is not set"):
        make_llm("live", config.load_settings())
