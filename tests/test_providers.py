"""Anthropic, OpenAI and Ollama providers, and provider selection.

Network calls are mocked (no live servers/keys needed), but the request
construction, response parsing, retry, and error paths are exercised for
real -- these are the parts most likely to break against an actual API.
"""

from __future__ import annotations

import json

import pytest

import orchestrator.llm as llm
from orchestrator.llm import (
    AnthropicProvider,
    LLMError,
    OllamaProvider,
    OpenAIProvider,
    build_provider,
    is_transient,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeRequests:
    """Stands in for the `requests` module: records calls, replays a script."""

    class RequestException(Exception):
        pass

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers or {}, "json": json,
                           "timeout": timeout})
        outcome = self.script.pop(0) if self.script else FakeResponse()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def fake_requests(monkeypatch):
    """Install a fake `requests` module and disable real retry sleeps."""
    installed = {}

    def install(script):
        fake = FakeRequests(script)
        monkeypatch.setitem(__import__("sys").modules, "requests", fake)
        monkeypatch.setattr(llm.time, "sleep", lambda s: None)
        installed["fake"] = fake
        return fake

    return install


ANTHROPIC_OK = FakeResponse(payload={
    "content": [{"type": "text", "text": "hello from claude"}],
    "usage": {"input_tokens": 12, "output_tokens": 5},
})

OPENAI_OK = FakeResponse(payload={
    "choices": [{"message": {"content": "hello from gpt"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
})

OLLAMA_OK = FakeResponse(payload={
    "message": {"content": "hello from llama"},
    "prompt_eval_count": 8, "eval_count": 3,
})


class TestAnthropicProvider:
    def test_generates_and_reports_usage(self, fake_requests):
        fake_requests([ANTHROPIC_OK])
        provider = AnthropicProvider(api_key="sk-ant-test")
        response = provider.generate("hi")
        assert response.text == "hello from claude"
        assert response.usage.prompt_tokens == 12
        assert response.usage.completion_tokens == 5
        assert response.usage.cost_usd > 0

    def test_sends_the_right_headers_and_body(self, fake_requests):
        fake = fake_requests([ANTHROPIC_OK])
        AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-5").generate(
            "hi", system="be terse")
        call = fake.calls[0]
        assert call["url"] == "https://api.anthropic.com/v1/messages"
        assert call["headers"]["x-api-key"] == "sk-ant-test"
        assert call["headers"]["anthropic-version"]
        assert call["json"]["model"] == "claude-sonnet-5"
        assert call["json"]["system"] == "be terse"
        assert call["json"]["messages"] == [{"role": "user", "content": "hi"}]

    def test_json_mode_instructs_via_the_system_prompt(self, fake_requests):
        fake = fake_requests([ANTHROPIC_OK])
        AnthropicProvider(api_key="sk-ant-test").generate("hi", json_mode=True)
        assert "JSON" in fake.calls[0]["json"]["system"]

    def test_missing_key_raises_immediately(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider()

    def test_retries_transient_errors_then_succeeds(self, fake_requests):
        fake = fake_requests([FakeResponse(status_code=529, text="overloaded"), ANTHROPIC_OK])
        response = AnthropicProvider(api_key="sk-ant-test", max_retries=2).generate("hi")
        assert response.text == "hello from claude"
        assert len(fake.calls) == 2

    def test_fatal_error_is_not_retried(self, fake_requests):
        fake = fake_requests([FakeResponse(status_code=401, text="invalid x-api-key")] * 3)
        with pytest.raises(LLMError, match="invalid x-api-key"):
            AnthropicProvider(api_key="sk-ant-test", max_retries=3).generate("hi")
        assert len(fake.calls) == 1


class TestOpenAIProvider:
    def test_generates_and_reports_usage(self, fake_requests):
        fake_requests([OPENAI_OK])
        response = OpenAIProvider(api_key="sk-test").generate("hi")
        assert response.text == "hello from gpt"
        assert response.usage.prompt_tokens == 10

    def test_sends_bearer_auth_and_system_message(self, fake_requests):
        fake = fake_requests([OPENAI_OK])
        OpenAIProvider(api_key="sk-test", model="gpt-4o-mini").generate("hi", system="be terse")
        call = fake.calls[0]
        assert call["url"] == "https://api.openai.com/v1/chat/completions"
        assert call["headers"]["Authorization"] == "Bearer sk-test"
        assert call["json"]["messages"][0] == {"role": "system", "content": "be terse"}
        assert call["json"]["messages"][1] == {"role": "user", "content": "hi"}

    def test_json_mode_sets_response_format(self, fake_requests):
        fake = fake_requests([OPENAI_OK])
        OpenAIProvider(api_key="sk-test").generate("hi", json_mode=True)
        assert fake.calls[0]["json"]["response_format"] == {"type": "json_object"}

    def test_custom_base_url_is_honoured(self, fake_requests):
        """Compatible endpoints (Azure, OpenRouter, vLLM) work via base_url."""
        fake = fake_requests([OPENAI_OK])
        OpenAIProvider(api_key="sk-test", base_url="https://openrouter.ai/api/v1").generate("hi")
        assert fake.calls[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"

    def test_missing_key_raises_immediately(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            OpenAIProvider()


class TestOllamaProvider:
    def test_needs_no_api_key(self, fake_requests):
        fake_requests([OLLAMA_OK])
        provider = OllamaProvider()  # must not raise
        response = provider.generate("hi")
        assert response.text == "hello from llama"

    def test_cost_is_always_zero(self, fake_requests):
        fake_requests([OLLAMA_OK])
        assert OllamaProvider().generate("hi").usage.cost_usd == 0.0

    def test_calls_the_local_chat_endpoint(self, fake_requests):
        fake = fake_requests([OLLAMA_OK])
        OllamaProvider(base_url="http://localhost:11434", model="llama3.1").generate("hi")
        call = fake.calls[0]
        assert call["url"] == "http://localhost:11434/api/chat"
        assert call["json"]["model"] == "llama3.1"
        assert call["json"]["stream"] is False

    def test_json_mode_sets_the_ollama_format_field(self, fake_requests):
        fake = fake_requests([OLLAMA_OK])
        OllamaProvider().generate("hi", json_mode=True)
        assert fake.calls[0]["json"]["format"] == "json"

    def test_connection_failure_suggests_starting_the_server(self, fake_requests):
        fake_requests([FakeRequests.RequestException("Connection refused")] * 10)
        with pytest.raises(LLMError, match="ollama serve"):
            OllamaProvider(max_retries=0).generate("hi")

    def test_usage_counts_come_from_ollama_fields(self, fake_requests):
        fake_requests([OLLAMA_OK])
        response = OllamaProvider().generate("hi")
        assert response.usage.prompt_tokens == 8
        assert response.usage.completion_tokens == 3


class TestProviderSelection:
    def test_build_provider_stub(self):
        from orchestrator.llm import StubProvider

        assert isinstance(build_provider("stub"), StubProvider)

    def test_build_provider_unknown_name_raises(self):
        with pytest.raises(LLMError, match="unknown provider"):
            build_provider("carrier-pigeon")

    def test_build_provider_ollama_needs_no_key(self):
        assert isinstance(build_provider("ollama"), OllamaProvider)

    def test_build_provider_anthropic_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(LLMError):
            build_provider("anthropic")

    def test_resolve_provider_precedence(self, monkeypatch):
        from orchestrator.config import Settings

        # Clear every provider key, or this asserts against whatever happens
        # to be in the developer's .env.local rather than against the rule.
        for key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "NVIDIA_API_KEY", "LLM_PROVIDER"):
            monkeypatch.delenv(key, raising=False)
        assert Settings(force_stub_llm=False).resolve_provider() == "stub"

        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
        assert Settings().resolve_provider() == "nvidia"

        monkeypatch.setenv("OPENAI_API_KEY", "x")
        assert Settings().resolve_provider() == "openai", "openai outranks nvidia"

        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        assert Settings().resolve_provider() == "anthropic", "anthropic outranks openai"

        monkeypatch.setenv("GEMINI_API_KEY", "x")
        assert Settings().resolve_provider() == "gemini", "gemini outranks the rest"

    def test_explicit_llm_provider_wins_over_key_presence(self, monkeypatch):
        from orchestrator.config import Settings

        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        assert Settings().resolve_provider() == "ollama"

    def test_ollama_is_never_auto_selected_by_key_presence(self, monkeypatch):
        """No key implies Ollama; that must never happen through auto-detection."""
        from orchestrator.config import Settings

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        assert Settings().resolve_provider() != "ollama"

    def test_force_stub_beats_everything(self, monkeypatch):
        from orchestrator.config import Settings

        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        assert Settings(force_stub_llm=True).resolve_provider() == "stub"

    def test_get_provider_degrades_to_stub_on_construction_failure(self, monkeypatch, capsys):
        from orchestrator import llm as llm_module
        from orchestrator.config import Settings

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        # get_provider() reads the module-level `settings`; swap it for one
        # that resolves to a provider whose key is genuinely absent, so the
        # try/except degrade-to-stub path actually runs.
        monkeypatch.setattr(llm_module, "settings", Settings())
        llm_module.set_provider(None)
        try:
            provider = llm_module.get_provider()
            assert provider.name == "stub"
            assert "unavailable" in capsys.readouterr().out
        finally:
            llm_module.set_provider(None)

    def test_describe_reports_the_resolved_model(self, monkeypatch):
        from orchestrator.config import Settings

        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_MODEL", "mixtral")
        info = Settings().describe()
        assert info["llm_mode"] == "ollama" and info["model"] == "mixtral"
