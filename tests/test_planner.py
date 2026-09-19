"""TaskPlanner: JSON extraction, graph repair, planning end to end."""

from __future__ import annotations

import json

import pytest

from orchestrator.graph import DependencyGraph
from orchestrator.llm import LLMResponse, LLMProvider
from orchestrator.models import LLMUsage
from orchestrator.planner import PlanningError, TaskPlanner, extract_json, plan_from_json


class ScriptedProvider(LLMProvider):
    """Returns canned text, so planner robustness can be tested precisely."""

    name = "scripted"
    model = "scripted"

    def __init__(self, text: str):
        super().__init__()
        self.text = text
        self.prompts = []

    def _generate(self, prompt, system, json_mode, metadata=None):
        self.prompts.append(prompt)
        return LLMResponse(text=self.text, usage=LLMUsage(calls=1, prompt_tokens=10,
                                                          completion_tokens=20),
                           model=self.model, latency_s=0.0)


VALID_PLAN = json.dumps({"tasks": [
    {"id": "design", "role": "research", "description": "Design it", "depends_on": [], "tool": None},
    {"id": "build", "role": "backend", "description": "Build it", "depends_on": ["design"],
     "tool": "github"},
    {"id": "test", "role": "qa", "description": "Test it", "depends_on": ["build"], "tool": "ci"},
]})


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_unlabelled_fence(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_json_embedded_in_prose(self):
        assert extract_json('Sure! {"a": 1} Hope that helps.') == {"a": 1}

    def test_nested_braces_are_balanced_correctly(self):
        assert extract_json('x {"a": {"b": [1, 2]}} y') == {"a": {"b": [1, 2]}}

    def test_braces_inside_strings_do_not_confuse_it(self):
        assert extract_json('{"a": "a } brace"}') == {"a": "a } brace"}

    def test_garbage_returns_none(self):
        assert extract_json("no json at all") is None

    def test_empty_returns_none(self):
        assert extract_json("") is None

    def test_json_array_is_rejected(self):
        assert extract_json("[1, 2, 3]") is None


class TestPlanning:
    def test_valid_plan_becomes_a_graph(self):
        result = TaskPlanner(llm=ScriptedProvider(VALID_PLAN)).plan("Build something")
        assert result.graph.topological_order() == ["design", "build", "test"]
        assert not result.repairs
        assert not result.used_fallback_plan

    @pytest.mark.parametrize("description,expected_ids,expected_tool", [
        ("Scrape Madurai weather and email it to the team",
         {"collect", "validate", "summarise", "deliver"}, "gmail"),
        ("Write a literature review comparing multi-agent frameworks",
         {"scope", "research", "analyse", "write", "verify"}, None),
        ("Create a weekly report in Microsoft Word",
         {"prepare", "desktop_work", "verify"}, "hermes_desktop"),
        ("Build a web app for managing college events",
         {"requirements", "database", "backend", "frontend", "testing"}, "workspace"),
    ])
    def test_offline_plans_match_the_kind_of_task(self, description, expected_ids,
                                                   expected_tool):
        from orchestrator.llm import StubProvider
        from orchestrator.tools import default_tool_manager

        result = TaskPlanner(llm=StubProvider()).plan(
            description, tools=default_tool_manager())
        ids = set(result.graph.topological_order())
        tools = {result.graph.get(step_id).requires_tool for step_id in ids}
        assert ids == expected_ids
        if expected_tool:
            assert expected_tool in tools

    def test_unrelated_offline_task_does_not_become_a_software_project(self):
        from orchestrator.llm import StubProvider

        result = TaskPlanner(llm=StubProvider()).plan("Organize my digital study notes")
        assert set(result.graph.topological_order()) == {"understand", "execute", "verify"}
        assert "frontend" not in result.graph

    def test_software_workflow_receives_real_build_tools_before_planning(self):
        from orchestrator import Workflow
        from orchestrator.llm import StubProvider

        workflow = Workflow.from_description(
            "Build a web app for college events", llm=StubProvider(), persist=False,
            verbose=False)
        assert "workspace" in workflow.tools
        assert "terminal" in workflow.tools
        assert workflow.graph.get("backend").requires_tool == "workspace"
        assert workflow.graph.get("testing").requires_tool == "terminal"

    def test_non_software_workflow_does_not_receive_terminal(self):
        from orchestrator import Workflow
        from orchestrator.llm import StubProvider

        workflow = Workflow.from_description(
            "Summarize these meeting notes", llm=StubProvider(), persist=False,
            verbose=False)
        assert "terminal" not in workflow.tools

    def test_roles_are_normalised(self):
        result = TaskPlanner(llm=ScriptedProvider(VALID_PLAN)).plan("Build something")
        assert result.graph.get("test").agent_role == "testing"  # "qa" -> "testing"

    def test_invented_tool_names_resolve_to_real_ones(self):
        """Regression: the planner asked for 'email_service', which does not
        exist, so the step failed at runtime with nothing to call and the
        requirements report demanded credentials for a fictional tool."""
        from orchestrator.tools import default_tool_manager

        available = {t["name"] for t in default_tool_manager().describe()}
        for invented, expected in [("email_service", "email"), ("mailer", "email"),
                                   ("git", "github"), ("database_service", "postgres")]:
            resolved, note = TaskPlanner.resolve_tool(invented, available)
            assert resolved == expected, f"{invented} should resolve to {expected}"

    def test_unresolvable_tool_is_preserved_for_discovery(self):
        from orchestrator.tools import default_tool_manager

        available = {t["name"] for t in default_tool_manager().describe()}
        resolved, note = TaskPlanner.resolve_tool("quantum_teleporter", available)
        assert resolved == "quantum_teleporter"
        assert note and "does not exist" in note

    def test_build_graph_records_a_fuzzy_tool_repair(self):
        """A known alias resolves silently; an unlisted near-miss is reported,
        so a surprising substitution is always visible in the plan repairs."""
        from orchestrator.tools import default_tool_manager

        available = {t["name"] for t in default_tool_manager().describe()}
        graph, repairs = TaskPlanner.build_graph(
            [{"id": "send", "role": "devops", "description": "Send it",
              "depends_on": [], "tool": "email_dispatcher"}],
            available_tools=available)
        assert graph.get("send").requires_tool == "email"
        assert any("email_dispatcher" in r for r in repairs)

    def test_known_alias_resolves_without_noise(self):
        from orchestrator.tools import default_tool_manager

        available = {t["name"] for t in default_tool_manager().describe()}
        graph, repairs = TaskPlanner.build_graph(
            [{"id": "send", "role": "devops", "description": "Send it",
              "depends_on": [], "tool": "email_service"}],
            available_tools=available)
        assert graph.get("send").requires_tool == "email"
        assert not any("email_service" in r for r in repairs)

    def test_configured_gmail_outranks_simulated_sendgrid(self):
        """A live sender must be reached before falling through to stdout."""
        from orchestrator.tools import default_tool_manager

        chain = default_tool_manager().candidates_for("email")
        assert "gmail" in chain
        assert chain.index("gmail") < chain.index("console_notify")

    def test_tools_are_canonicalised(self):
        plan = json.dumps({"tasks": [
            {"id": "a", "role": "backend", "description": "x", "depends_on": [],
             "tool": "github_mcp"}]})
        result = TaskPlanner(llm=ScriptedProvider(plan)).plan("x")
        assert result.graph.get("a").requires_tool == "github"

    def test_null_like_tool_strings_become_none(self):
        plan = json.dumps({"tasks": [
            {"id": "a", "role": "backend", "description": "x", "depends_on": [], "tool": "none"}]})
        result = TaskPlanner(llm=ScriptedProvider(plan)).plan("x")
        assert result.graph.get("a").requires_tool is None

    def test_usage_is_reported(self):
        result = TaskPlanner(llm=ScriptedProvider(VALID_PLAN)).plan("x")
        assert result.usage.calls == 1

    def test_unusable_response_falls_back_to_a_generic_plan(self):
        result = TaskPlanner(llm=ScriptedProvider("I'd be happy to help!")).plan("Build an app")
        assert result.used_fallback_plan
        assert len(result.graph) == 5
        result.graph.validate()

    def test_empty_task_list_falls_back(self):
        result = TaskPlanner(llm=ScriptedProvider('{"tasks": []}')).plan("x")
        assert result.used_fallback_plan

    def test_stub_provider_produces_a_valid_plan(self, stub_llm):
        result = TaskPlanner(llm=stub_llm).plan("Build a task manager with auth")
        result.graph.validate()
        assert len(result.graph) >= 3

    def test_plan_is_deterministic_with_the_stub(self, stub_llm):
        planner = TaskPlanner(llm=stub_llm)
        first = planner.plan("Build a bookstore API").graph.to_dict()
        second = planner.plan("Build a bookstore API").graph.to_dict()
        assert first == second


class TestRepair:
    def test_ids_are_sanitised(self):
        graph, repairs = TaskPlanner.build_graph([
            {"id": "My Step!", "role": "backend", "description": "x", "depends_on": []}])
        assert "my_step" in graph

    def test_duplicate_ids_are_renamed(self):
        graph, repairs = TaskPlanner.build_graph([
            {"id": "a", "role": "backend", "description": "x", "depends_on": []},
            {"id": "a", "role": "qa", "description": "y", "depends_on": []},
        ])
        assert set(graph.topological_order()) == {"a", "a_1"}
        assert any("duplicate" in r for r in repairs)

    def test_dangling_dependencies_are_pruned(self):
        graph, repairs = TaskPlanner.build_graph([
            {"id": "a", "role": "backend", "description": "x", "depends_on": ["ghost"]}])
        assert graph.get("a").depends_on == []
        assert any("no such step" in r for r in repairs)

    def test_cycles_are_broken(self):
        graph, repairs = TaskPlanner.build_graph([
            {"id": "a", "role": "backend", "description": "x", "depends_on": ["b"]},
            {"id": "b", "role": "qa", "description": "y", "depends_on": ["a"]},
        ])
        graph.validate()
        assert any("cycle" in r for r in repairs)

    def test_string_dependency_is_accepted(self):
        graph, _ = TaskPlanner.build_graph([
            {"id": "a", "role": "backend", "description": "x", "depends_on": []},
            {"id": "b", "role": "qa", "description": "y", "depends_on": "a"},
        ])
        assert graph.get("b").depends_on == ["a"]

    def test_non_object_tasks_are_dropped(self):
        graph, repairs = TaskPlanner.build_graph([
            "nonsense",
            {"id": "a", "role": "backend", "description": "x", "depends_on": []},
        ])
        assert list(graph.topological_order()) == ["a"]
        assert any("dropped non-object" in r for r in repairs)

    def test_no_usable_tasks_raises(self):
        with pytest.raises(PlanningError):
            TaskPlanner.build_graph([])

    def test_missing_description_falls_back_to_the_id(self):
        graph, _ = TaskPlanner.build_graph([{"id": "a", "role": "backend"}])
        assert graph.get("a").description == "a"


class TestRevise:
    def test_revise_step_returns_new_text(self, stub_llm):
        planner = TaskPlanner(llm=stub_llm)
        graph = DependencyGraph.from_dict({"steps": [
            {"id": "db", "description": "Design a PostgreSQL schema", "agent_role": "database",
             "depends_on": [], "name": "DB", "requires_tool": None}]})
        revised = planner.revise_step(graph.get("db"), "use MongoDB instead")
        assert revised and revised != graph.get("db").description


class TestPlanFromFile:
    def test_hand_written_plan_loads(self, tmp_path):
        path = tmp_path / "plan.json"
        path.write_text(VALID_PLAN, encoding="utf-8")
        graph = plan_from_json(str(path))
        assert graph.topological_order() == ["design", "build", "test"]

    def test_bare_list_is_accepted(self, tmp_path):
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(json.loads(VALID_PLAN)["tasks"]), encoding="utf-8")
        assert len(plan_from_json(str(path))) == 3
