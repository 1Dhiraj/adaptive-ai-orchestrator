"""Agents, roles, memory and the LLM provider layer."""

from __future__ import annotations

import pytest

from orchestrator.agents import Agent, AgentManager, ROLE_PROMPTS, normalise_role
from orchestrator.llm import (
    LLMError,
    LLMResponse,
    StubProvider,
    estimate_tokens,
    get_provider,
    is_transient,
    price,
    set_provider,
)
from orchestrator.memory import MemoryManager, truncate
from orchestrator.models import LLMUsage, Step, content_hash
from orchestrator.tools import ToolManager, default_tool_manager
from orchestrator.tools.base import SimulatedTool, ToolInvocation


class TestRoleNormalisation:
    @pytest.mark.parametrize("declared,expected", [
        ("frontend engineer", "frontend"),
        ("Senior Backend Engineer", "backend"),
        ("dba", "database"),
        ("DBA", "database"),
        ("qa engineer", "testing"),
        ("security reviewer", "security"),
        ("technical writer", "writer"),
        ("sre", "devops"),
        ("product manager", "product"),
        ("underwater basket weaver", "generic"),
        ("", "generic"),
    ])
    def test_free_form_roles_map_to_known_keys(self, declared, expected):
        assert normalise_role(declared) == expected

    def test_every_known_role_has_a_prompt(self):
        for role in ROLE_PROMPTS:
            assert ROLE_PROMPTS[role].strip()


class TestAgent:
    def test_prompt_contains_task_and_context(self, stub_llm):
        agent = Agent("backend", llm=stub_llm)
        step = Step(id="api", description="Build the API", agent_role="backend",
                    depends_on=["ui"])
        prompt = agent.build_prompt(step, "### Output of `ui`\nUI is done")
        assert "Build the API" in prompt
        assert "UI is done" in prompt

    def test_tool_instructions_appear_only_when_a_tool_is_required(self, stub_llm):
        agent = Agent("backend", llm=stub_llm)
        without = agent.build_prompt(Step(id="a", description="x", agent_role="backend"), "")
        with_tool = agent.build_prompt(
            Step(id="a", description="x", agent_role="backend", requires_tool="github"), "")
        assert "TOOL_DIRECTIVE" not in without
        assert "TOOL_DIRECTIVE" in with_tool

    def test_execute_returns_output_and_usage(self, stub_llm):
        agent = Agent("backend", llm=stub_llm)
        outcome = agent.execute(Step(id="a", description="Build it", agent_role="backend"), "")
        assert outcome.output
        assert outcome.usage.calls == 1

    def test_tool_output_is_appended_not_substituted(self, stub_llm):
        tools = default_tool_manager()
        agent = Agent("backend", llm=stub_llm)
        step = Step(id="a", description="Build it", agent_role="backend", requires_tool="github")
        outcome = agent.execute(step, "", tools)
        assert outcome.tool_invocation is not None
        assert "Tool `github`" in outcome.output
        assert outcome.output.startswith("[backend]")  # the agent's own work survives
        assert "TOOL_DIRECTIVE" not in outcome.output

    def test_fallback_is_labelled_in_the_output(self, stub_llm):
        tools = default_tool_manager()
        tools.break_tool("github")
        agent = Agent("backend", llm=stub_llm)
        step = Step(id="a", description="x", agent_role="backend", requires_tool="github")
        assert "FALLBACK" in agent.execute(step, "", tools).output

    def test_computer_use_result_hides_model_tool_reasoning(self):
        invocation = ToolInvocation(
            requested="computer_use", tool_used="computer_use", ok=True,
            simulated=False, output="opened Notepad and typed 5 characters")

        merged = Agent.merge_tool_result(
            "We need to output a tool directive, but the response was cut off.",
            "computer_use", invocation)

        assert merged == (
            "Tool `computer_use`: opened Notepad and typed 5 characters")

    def test_role_specific_output_from_the_stub(self, stub_llm):
        step = Step(id="db", description="Design the schema", agent_role="database")
        assert Agent("database", llm=stub_llm).execute(step, "").output.startswith("[database]")


class TestAgentManager:
    def test_agents_are_pooled_by_role(self, stub_llm):
        manager = AgentManager(llm=stub_llm)
        assert manager.get("backend engineer") is manager.get("Senior Backend Dev")

    def test_custom_roles_can_be_registered(self, stub_llm):
        manager = AgentManager(llm=stub_llm)
        agent = manager.add("compliance_officer", "You are a compliance officer.")
        assert "compliance officer" in agent.system_prompt
        assert manager.get("compliance_officer") is agent

    def test_custom_role_keeps_its_own_identity(self, stub_llm):
        """A new role must not collapse into 'generic'."""
        manager = AgentManager(llm=stub_llm)
        agent = manager.add("compliance_officer", "You are a compliance officer.")
        assert agent.role == "compliance_officer"
        assert manager.get("compliance officer").system_prompt == agent.system_prompt

    def test_role_noise_words_are_stripped_on_registration(self, stub_llm):
        """'Cost Analyst' and 'cost engineer' are the same role."""
        manager = AgentManager(llm=stub_llm)
        agent = manager.add("Cost Analyst", "You are a cloud cost analyst.")
        assert agent.role == "cost"
        assert manager.get("cost engineer") is agent

    def test_registering_a_new_role_does_not_clobber_a_builtin(self):
        from orchestrator.agents import register_role

        before = ROLE_PROMPTS["data"]
        assert register_role("Data Steward", "You steward data.") == "data_steward"
        assert ROLE_PROMPTS["data"] == before

    def test_custom_role_is_used_when_a_step_names_it(self, stub_llm):
        manager = AgentManager(llm=stub_llm)
        manager.add("compliance_officer", "You are a compliance officer.")
        step = Step(id="c", description="List the controls", agent_role="compliance_officer")
        agent = manager.get(step.agent_role)
        assert "compliance officer" in agent.build_system_prompt()

    def test_register_role_returns_the_key(self):
        from orchestrator.agents import register_role

        # Noise words are stripped; everything else is kept verbatim.
        assert register_role("Senior Backend Engineer", "prompt") == "backend"
        assert register_role("Security Reviewer", "prompt") == "security_reviewer"
        assert register_role("data_steward", "prompt") == "data_steward"

    def test_custom_role_is_labelled_in_stub_output(self, stub_llm):
        manager = AgentManager(llm=stub_llm)
        manager.add("compliance_officer", "You are a compliance officer.")
        step = Step(id="c", description="List the controls", agent_role="compliance_officer")
        assert manager.get("compliance_officer").execute(step, "").output.startswith(
            "[compliance_officer]")

    def test_unknown_role_falls_back_to_generic(self, stub_llm):
        assert AgentManager(llm=stub_llm).get("wizard").role == "generic"


class TestMemory:
    def test_store_and_retrieve(self):
        memory = MemoryManager()
        memory.store("a", "output a")
        assert memory.get("a") == "output a"
        assert "a" in memory

    def test_context_lists_each_dependency(self):
        memory = MemoryManager()
        memory.store("a", "output a")
        memory.store("b", "output b")
        step = Step(id="c", description="x", agent_role="generic", depends_on=["a", "b"])
        context = memory.build_context(step)
        assert "output a" in context and "output b" in context

    def test_missing_dependency_is_marked(self):
        step = Step(id="c", description="x", agent_role="generic", depends_on=["a"])
        assert "not yet produced" in MemoryManager().build_context(step)

    def test_context_budget_can_be_expanded_for_exact_artifacts(self):
        memory = MemoryManager(char_budget=20)
        original = "complete report " * 40
        memory.store("writer", original)
        step = Step(id="store", description="archive", agent_role="writer",
                    depends_on=["writer"], requires_tool="artifact_store")
        ordinary = memory.build_context(step)
        expanded = memory.build_context(step, char_budget=2000)
        assert "chars elided" in ordinary
        assert original in expanded

    def test_input_hash_changes_with_upstream_output(self):
        memory = MemoryManager()
        step = Step(id="b", description="x", agent_role="generic", depends_on=["a"])
        memory.store("a", "first")
        first = memory.input_hash(step)
        memory.store("a", "second")
        assert memory.input_hash(step) != first

    def test_input_hash_changes_with_the_step_definition(self):
        memory = MemoryManager()
        step = Step(id="b", description="original", agent_role="generic")
        first = memory.input_hash(step)
        step.description = "changed"
        assert memory.input_hash(step) != first

    def test_input_hash_is_stable_when_nothing_changed(self):
        memory = MemoryManager()
        memory.store("a", "output")
        step = Step(id="b", description="x", agent_role="generic", depends_on=["a"])
        assert memory.input_hash(step) == memory.input_hash(step)

    def test_retry_knobs_do_not_affect_the_hash(self):
        memory = MemoryManager()
        step = Step(id="b", description="x", agent_role="generic")
        first = memory.input_hash(step)
        step.max_retries = 99
        assert memory.input_hash(step) == first

    def test_truncate_keeps_both_ends(self):
        result = truncate("A" * 100 + "B" * 100, 60)
        assert result.startswith("A") and result.endswith("B") and "elided" in result

    def test_short_text_is_untouched(self):
        assert truncate("short", 100) == "short"

    def test_clear_forgets_everything(self):
        memory = MemoryManager()
        memory.store("a", "x")
        memory.clear()
        assert memory.snapshot() == {}


class TestLLMLayer:
    def test_stub_is_deterministic(self):
        first = StubProvider().generate("same prompt", system="sys").text
        second = StubProvider().generate("same prompt", system="sys").text
        assert first == second

    def test_stub_varies_with_the_prompt(self):
        provider = StubProvider()
        assert provider.generate("prompt A").text != provider.generate("prompt B").text

    def test_json_mode_returns_parseable_json(self):
        import json

        response = StubProvider().generate("Build an app", json_mode=True)
        assert isinstance(json.loads(response.text), dict)

    def test_usage_accumulates_across_calls(self):
        provider = StubProvider()
        provider.generate("a")
        provider.generate("b")
        assert provider.total_usage.calls == 2
        assert provider.total_usage.total_tokens > 0

    def test_reset_usage(self):
        provider = StubProvider()
        provider.generate("a")
        provider.reset_usage()
        assert provider.total_usage.calls == 0

    def test_role_metadata_selects_the_template(self):
        provider = StubProvider()
        assert provider.generate("x", metadata={"role": "testing"}).text.startswith("[testing]")
        assert provider.generate("x", metadata={"role": "devops"}).text.startswith("[devops]")

    def test_usage_addition(self):
        total = LLMUsage(calls=1, prompt_tokens=10, completion_tokens=5, cost_usd=0.1) + \
                LLMUsage(calls=2, prompt_tokens=20, completion_tokens=5, cost_usd=0.2)
        assert total.calls == 3 and total.total_tokens == 40
        assert total.cost_usd == pytest.approx(0.3)

    def test_pricing_is_applied(self):
        assert price("gemini-2.5-flash", 1_000_000, 0) == pytest.approx(0.30)
        assert price("stub", 1_000_000, 1_000_000) == 0.0

    def test_unknown_model_uses_the_default_rate(self):
        assert price("some-new-model", 1_000_000, 0) > 0

    def test_token_estimate_is_positive(self):
        assert estimate_tokens("") >= 1
        assert estimate_tokens("a" * 400) == 100

    @pytest.mark.parametrize("message,expected", [
        ("429 rate limit exceeded", True),
        ("503 Service Unavailable", True),
        ("deadline exceeded", True),
        ("invalid api key", False),
        ("malformed request", False),
    ])
    def test_transient_classification(self, message, expected):
        assert is_transient(RuntimeError(message)) is expected

    def test_get_provider_returns_the_stub_without_a_key(self):
        set_provider(None)
        try:
            assert isinstance(get_provider(force_stub=True), StubProvider)
        finally:
            set_provider(None)

    def test_content_hash_is_stable_and_order_sensitive(self):
        assert content_hash("a", "b") == content_hash("a", "b")
        assert content_hash("a", "b") != content_hash("b", "a")


class FakeUsage:
    def __init__(self, prompt=17, candidates=23):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates


class FakeGeminiResponse:
    def __init__(self, text="generated", usage=None):
        self.text = text
        self.usage_metadata = usage


class FakeModels:
    """Stands in for ``genai.Client().models``."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.script.pop(0) if self.script else FakeGeminiResponse()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def fake_gemini(monkeypatch):
    """Install a fake ``google.genai`` client and return a provider factory."""
    from google import genai

    from orchestrator.llm import GeminiProvider

    def build(script, **kwargs):
        models = FakeModels(script)

        class FakeClient:
            def __init__(self, api_key=None):
                self.models = models

        monkeypatch.setattr(genai, "Client", FakeClient)
        monkeypatch.setattr("time.sleep", lambda seconds: None)  # no real backoff in tests
        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash", **kwargs)
        return provider, models

    return build


class TestGeminiProvider:
    def test_usage_is_read_from_the_response_metadata(self, fake_gemini):
        provider, _ = fake_gemini([FakeGeminiResponse("hello", FakeUsage(17, 23))])
        response = provider.generate("prompt")
        assert response.text == "hello"
        assert response.usage.prompt_tokens == 17
        assert response.usage.completion_tokens == 23
        assert response.usage.cost_usd > 0

    def test_usage_is_estimated_when_metadata_is_missing(self, fake_gemini):
        provider, _ = fake_gemini([FakeGeminiResponse("a" * 400, None)])
        assert provider.generate("prompt").usage.completion_tokens == 100

    def test_system_instruction_and_json_mode_are_passed_through(self, fake_gemini):
        provider, models = fake_gemini([FakeGeminiResponse()])
        provider.generate("prompt", system="be terse", json_mode=True)
        config = models.calls[0]["config"]
        assert config["system_instruction"] == "be terse"
        assert config["response_mime_type"] == "application/json"

    def test_transient_errors_are_retried_then_succeed(self, fake_gemini):
        provider, models = fake_gemini(
            [RuntimeError("503 unavailable"), FakeGeminiResponse("recovered")],
            max_retries=2)
        assert provider.generate("prompt").text == "recovered"
        assert len(models.calls) == 2

    def test_retries_are_exhausted_then_it_raises(self, fake_gemini):
        provider, models = fake_gemini([RuntimeError("429 rate limit")] * 4, max_retries=2)
        with pytest.raises(LLMError, match="3 attempt"):
            provider.generate("prompt")
        assert len(models.calls) == 3

    def test_fatal_errors_are_not_retried(self, fake_gemini):
        provider, models = fake_gemini([RuntimeError("invalid api key")] * 3, max_retries=3)
        with pytest.raises(LLMError, match="invalid api key"):
            provider.generate("prompt")
        assert len(models.calls) == 1

    def test_total_usage_accumulates(self, fake_gemini):
        provider, _ = fake_gemini([FakeGeminiResponse("a", FakeUsage()),
                                   FakeGeminiResponse("b", FakeUsage())])
        provider.generate("one")
        provider.generate("two")
        assert provider.total_usage.calls == 2
        assert provider.total_usage.prompt_tokens == 34
