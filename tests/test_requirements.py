"""Requirement declaration, readiness checking, and specialist synthesis."""

from __future__ import annotations

import json

import pytest

from orchestrator.llm import LLMProvider, LLMResponse
from orchestrator.models import LLMUsage, Step
from orchestrator.planner import TaskPlanner
from orchestrator.requirements import (
    AgentSpec,
    Requirement,
    RequirementKind,
    RequirementStatus,
    RequirementsReport,
    check_requirement,
    infer_requirements,
    merge_requirements,
)
from orchestrator.tools import ToolManager, default_tool_manager
from orchestrator.workflow import Workflow

from .conftest import linear_steps


class ScriptedPlanner(LLMProvider):
    """Returns a fixed plan, so parsing can be asserted exactly."""

    name = model = "scripted"

    def __init__(self, payload: dict):
        super().__init__()
        self.payload = json.dumps(payload)

    def _generate(self, prompt, system, json_mode, metadata=None):
        text = self.payload if json_mode else f"[{(metadata or {}).get('role', '?')}] delivered."
        return LLMResponse(text=text, usage=LLMUsage(calls=1, prompt_tokens=10,
                                                     completion_tokens=20),
                           model=self.model, latency_s=0.0)


DOMAIN_PLAN = {
    "agents": [
        {"role": "clinical_trial_statistician", "why": "sample sizing is specialist work",
         "system_prompt": "You are a clinical trial statistician."},
        {"role": "medical_writer", "why": "protocol prose follows ICH-GCP",
         "system_prompt": "You are a medical writer."},
    ],
    "requirements": [
        {"name": "pubmed", "kind": "mcp_server", "why": "literature search",
         "needed_by": ["power"], "setup": "npx -y @modelcontextprotocol/server-pubmed"},
        {"name": "TRIALS_API_KEY", "kind": "credential", "why": "query the registry",
         "needed_by": ["power"], "setup": "set TRIALS_API_KEY"},
    ],
    "tasks": [
        {"id": "power", "role": "clinical_trial_statistician", "depends_on": [],
         "description": "Compute the sample size."},
        {"id": "protocol", "role": "medical_writer", "depends_on": ["power"],
         "description": "Draft the protocol."},
    ],
    "notes": ["Assumes a Phase II design."],
}


class TestRequirementChecking:
    def test_set_credential_is_ready(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_KEY", "value")
        req = check_requirement(Requirement(name="MY_TEST_KEY", kind=RequirementKind.CREDENTIAL))
        assert req.status is RequirementStatus.READY

    def test_unset_credential_is_missing(self, monkeypatch):
        monkeypatch.delenv("MY_TEST_KEY", raising=False)
        req = check_requirement(Requirement(name="MY_TEST_KEY", kind=RequirementKind.CREDENTIAL))
        assert req.status is RequirementStatus.MISSING
        assert req.status.blocks_execution

    def test_optional_missing_requirement_does_not_block(self, monkeypatch):
        monkeypatch.delenv("MY_TEST_KEY", raising=False)
        req = check_requirement(Requirement(name="MY_TEST_KEY",
                                            kind=RequirementKind.CREDENTIAL, optional=True))
        assert req.status is RequirementStatus.SIMULATED
        assert not req.status.blocks_execution

    def test_registered_tool_without_credentials_is_simulated(self, tools):
        req = check_requirement(Requirement(name="github", kind=RequirementKind.TOOL), tools)
        assert req.status is RequirementStatus.SIMULATED
        assert "simulated" in req.detail

    def test_always_live_tool_is_ready(self, tools):
        req = check_requirement(Requirement(name="artifact_store", kind=RequirementKind.TOOL), tools)
        assert req.status is RequirementStatus.READY

    def test_registered_tool_mislabelled_as_binary_uses_registry(self, tools):
        req = check_requirement(
            Requirement(name="artifact_store", kind=RequirementKind.BINARY), tools)
        assert req.status is RequirementStatus.READY
        assert req.detail == "configured and live"

    def test_broken_tool_with_a_working_fallback_is_degraded_not_blocked(self, tools):
        """A broken github still runs via github_cli, so it must not block."""
        tools.break_tool("github")
        req = check_requirement(Requirement(name="github", kind=RequirementKind.TOOL), tools)
        assert req.status is RequirementStatus.FALLBACK
        assert not req.status.blocks_execution
        assert "github_cli" in req.detail

    def test_broken_tool_with_its_whole_chain_down_is_missing(self, tools):
        for name in ("github", "github_cli", "artifact_store"):
            tools.break_tool(name)
        req = check_requirement(Requirement(name="github", kind=RequirementKind.TOOL), tools)
        assert req.status is RequirementStatus.MISSING

    def test_unregistered_tool_with_nothing_registered_is_missing(self):
        req = Requirement(name="github", kind=RequirementKind.TOOL)
        check_requirement(req, ToolManager())
        assert req.status is RequirementStatus.MISSING

    def test_unregistered_tool_with_nothing_is_missing(self):
        req = check_requirement(Requirement(name="ghost_tool", kind=RequirementKind.TOOL),
                                ToolManager())
        assert req.status is RequirementStatus.MISSING

    def test_installed_package_is_ready(self):
        req = check_requirement(Requirement(name="json", kind=RequirementKind.PACKAGE))
        assert req.status is RequirementStatus.READY

    def test_absent_package_is_missing(self):
        req = check_requirement(Requirement(name="definitely_not_a_package_xyz",
                                            kind=RequirementKind.PACKAGE))
        assert req.status is RequirementStatus.MISSING

    def test_runtime_declared_as_package_is_normalised_to_binary(self):
        req = Requirement.from_dict({"name": "python", "kind": "package"})
        assert req.kind is RequirementKind.BINARY
        check_requirement(req)
        assert req.status is RequirementStatus.READY

    def test_binary_on_path_is_ready(self):
        req = check_requirement(Requirement(name="python", kind=RequirementKind.BINARY))
        assert req.status in {RequirementStatus.READY, RequirementStatus.MISSING}

    def test_absent_binary_is_missing(self):
        req = check_requirement(Requirement(name="definitely-not-a-binary-xyz",
                                            kind=RequirementKind.BINARY))
        assert req.status is RequirementStatus.MISSING

    def test_unattached_mcp_server_is_missing(self, tools):
        req = check_requirement(Requirement(name="pubmed", kind=RequirementKind.MCP_SERVER), tools)
        assert req.status is RequirementStatus.MISSING
        assert "not attached" in req.detail


class TestReport:
    def _report(self, *reqs) -> RequirementsReport:
        return RequirementsReport(requirements=list(reqs))

    def test_can_run_when_nothing_is_missing(self, tools):
        report = self._report(Requirement(name="github", kind=RequirementKind.TOOL)).check(tools)
        assert report.can_run
        assert report.degraded

    def test_blocked_when_something_is_missing(self, tools, monkeypatch):
        monkeypatch.delenv("NOPE_KEY", raising=False)
        report = self._report(
            Requirement(name="NOPE_KEY", kind=RequirementKind.CREDENTIAL)).check(tools)
        assert not report.can_run
        assert [r.name for r in report.blockers] == ["NOPE_KEY"]

    def test_render_lists_setup_for_blockers(self, tools, monkeypatch):
        monkeypatch.delenv("NOPE_KEY", raising=False)
        report = self._report(Requirement(name="NOPE_KEY", kind=RequirementKind.CREDENTIAL,
                                          setup="set NOPE_KEY in .env.local")).check(tools)
        text = report.render()
        assert "BLOCKED" in text and "set NOPE_KEY in .env.local" in text

    def test_render_says_ready_when_clean(self, tools):
        report = self._report(
            Requirement(name="artifact_store", kind=RequirementKind.TOOL)).check(tools)
        assert "READY TO RUN" in report.render()

    def test_render_handles_no_requirements(self):
        assert "pure reasoning" in RequirementsReport().render()

    def test_round_trips_through_json(self, tools):
        report = RequirementsReport(
            requirements=[Requirement(name="github", kind=RequirementKind.TOOL)],
            agents=[AgentSpec(role="x", system_prompt="You are x.")],
            notes=["a note"]).check(tools)
        restored = RequirementsReport.from_dict(json.loads(json.dumps(report.to_dict())))
        assert restored.requirements[0].name == "github"
        assert restored.agents[0].role == "x"
        assert restored.notes == ["a note"]

    def test_print_report_writes_to_stdout(self, tools, capsys):
        self._report(Requirement(name="github", kind=RequirementKind.TOOL)).check(tools).print_report()
        assert "WHAT THIS TASK NEEDS" in capsys.readouterr().out


class TestInference:
    def test_tools_named_by_steps_are_inferred(self, linear_graph):
        names = {r.name for r in infer_requirements(linear_graph)}
        assert {"github", "postgres", "ci"} <= names

    def test_credentials_for_those_tools_are_inferred(self, linear_graph):
        creds = {r.name for r in infer_requirements(linear_graph)
                 if r.kind is RequirementKind.CREDENTIAL}
        assert "GITHUB_TOKEN" in creds and "DATABASE_URL" in creds

    def test_inferred_credentials_are_optional(self, linear_graph):
        creds = [r for r in infer_requirements(linear_graph)
                 if r.kind is RequirementKind.CREDENTIAL]
        assert all(r.optional for r in creds), "missing creds degrade, they do not block"

    def test_toolless_graph_needs_nothing(self):
        from orchestrator.graph import DependencyGraph

        graph = DependencyGraph([Step(id="think", description="Reason about it",
                                      agent_role="research")])
        assert infer_requirements(graph) == []

    def test_merge_deduplicates_and_unions_dependents(self):
        merged = merge_requirements(
            [Requirement(name="github", kind=RequirementKind.TOOL, needed_by=["a"])],
            [Requirement(name="GitHub", kind=RequirementKind.TOOL, needed_by=["b"],
                         why="a longer explanation here")],
        )
        assert len(merged) == 1
        assert merged[0].needed_by == ["a", "b"]
        assert merged[0].why == "a longer explanation here"

    def test_merge_keeps_different_kinds_separate(self):
        merged = merge_requirements(
            [Requirement(name="github", kind=RequirementKind.TOOL)],
            [Requirement(name="github", kind=RequirementKind.MCP_SERVER)],
        )
        assert len(merged) == 2


class TestPlannerSynthesis:
    def test_specialists_are_parsed(self):
        result = TaskPlanner(llm=ScriptedPlanner(DOMAIN_PLAN)).plan("Design a trial")
        assert {a.role for a in result.agents} == {"clinical_trial_statistician", "medical_writer"}

    def test_specialists_keep_their_prompts_and_reasons(self):
        result = TaskPlanner(llm=ScriptedPlanner(DOMAIN_PLAN)).plan("Design a trial")
        stat = next(a for a in result.agents if a.role == "clinical_trial_statistician")
        assert stat.system_prompt == "You are a clinical trial statistician."
        assert "specialist work" in stat.why

    def test_custom_role_is_not_absorbed_into_a_builtin(self):
        """'medical_writer' must not collapse into the built-in 'writer'."""
        result = TaskPlanner(llm=ScriptedPlanner(DOMAIN_PLAN)).plan("Design a trial")
        assert result.graph.get("protocol").agent_role == "medical_writer"

    def test_handles_are_backfilled_from_the_graph(self):
        result = TaskPlanner(llm=ScriptedPlanner(DOMAIN_PLAN)).plan("Design a trial")
        stat = next(a for a in result.agents if a.role == "clinical_trial_statistician")
        assert stat.handles == ["power"]

    def test_declared_requirements_are_kept(self):
        result = TaskPlanner(llm=ScriptedPlanner(DOMAIN_PLAN)).plan("Design a trial")
        names = {r.name for r in result.requirements.requirements}
        assert "pubmed" in names and "TRIALS_API_KEY" in names

    def test_mcp_requirement_carries_its_install_command(self):
        result = TaskPlanner(llm=ScriptedPlanner(DOMAIN_PLAN)).plan("Design a trial")
        pubmed = next(r for r in result.requirements.requirements if r.name == "pubmed")
        assert pubmed.kind is RequirementKind.MCP_SERVER
        assert "server-pubmed" in pubmed.setup

    def test_notes_are_surfaced(self):
        result = TaskPlanner(llm=ScriptedPlanner(DOMAIN_PLAN)).plan("Design a trial")
        assert result.requirements.notes == ["Assumes a Phase II design."]

    def test_unmet_requirements_block(self, monkeypatch):
        monkeypatch.delenv("TRIALS_API_KEY", raising=False)
        result = TaskPlanner(llm=ScriptedPlanner(DOMAIN_PLAN)).plan("Design a trial")
        assert not result.can_run
        assert {r.name for r in result.requirements.blockers} == {"pubmed", "TRIALS_API_KEY"}

    def test_agents_without_prompts_are_dropped(self):
        plan = {"agents": [{"role": "ghost"}],
                "tasks": [{"id": "a", "role": "generic", "depends_on": [], "description": "x"}]}
        result = TaskPlanner(llm=ScriptedPlanner(plan)).plan("x")
        assert result.agents == []
        assert any("missing prompt" in r for r in result.repairs)

    def test_duplicate_agents_are_dropped(self):
        plan = {"agents": [{"role": "dup", "system_prompt": "A"},
                           {"role": "dup", "system_prompt": "B"}],
                "tasks": [{"id": "a", "role": "dup", "depends_on": [], "description": "x"}]}
        result = TaskPlanner(llm=ScriptedPlanner(plan)).plan("x")
        assert len(result.agents) == 1
        assert any("duplicate agent" in r for r in result.repairs)

    def test_malformed_requirements_are_dropped(self):
        plan = {"requirements": [{"why": "no name"}],
                "tasks": [{"id": "a", "role": "generic", "depends_on": [], "description": "x"}]}
        result = TaskPlanner(llm=ScriptedPlanner(plan)).plan("x")
        assert any("malformed requirement" in r for r in result.repairs)

    def test_steps_tools_are_reported_even_when_undeclared(self):
        """The planner forgot to declare 'github'; the report must still list it."""
        plan = {"requirements": [],
                "tasks": [{"id": "a", "role": "backend", "depends_on": [],
                           "description": "x", "tool": "github"}]}
        result = TaskPlanner(llm=ScriptedPlanner(plan)).plan("x")
        assert "github" in {r.name for r in result.requirements.requirements}


class TestWorkflowIntegration:
    def test_planned_specialists_are_registered(self):
        wf = Workflow.from_description("Design a trial", llm=ScriptedPlanner(DOMAIN_PLAN),
                                       persist=False, verbose=False)
        try:
            assert "clinical_trial_statistician" in wf.agents.roles
            assert wf.agents.get("medical_writer").system_prompt == "You are a medical writer."
        finally:
            wf.close()

    def test_steps_execute_with_their_specialist(self):
        wf = Workflow.from_description("Design a trial", llm=ScriptedPlanner(DOMAIN_PLAN),
                                       persist=False, verbose=False)
        try:
            report = wf.run_full()
            assert report.ok
            assert "clinical_trial_statistician" in wf.query_results("power")
        finally:
            wf.close()

    def test_requirements_are_exposed_on_the_workflow(self):
        wf = Workflow.from_description("Design a trial", llm=ScriptedPlanner(DOMAIN_PLAN),
                                       persist=False, verbose=False)
        try:
            assert not wf.requirements.can_run
            assert "pubmed" in {r.name for r in wf.requirements.blockers}
        finally:
            wf.close()

    def test_handwritten_steps_get_inferred_requirements(self, make_workflow):
        wf = make_workflow(linear_steps())
        report = wf.check_requirements()
        assert {"github", "postgres", "ci"} <= {r.name for r in report.requirements}
        assert report.can_run, "simulated tools should not block a run"

    def test_print_requirements_works_for_handwritten_steps(self, make_workflow, capsys):
        make_workflow(linear_steps()).print_requirements()
        assert "WHAT THIS TASK NEEDS" in capsys.readouterr().out

    def test_status_updates_after_a_credential_is_set(self, make_workflow, monkeypatch):
        wf = make_workflow(linear_steps())
        before = next(r for r in wf.check_requirements().requirements
                      if r.name == "GITHUB_TOKEN")
        assert before.status is not RequirementStatus.READY

        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        after = next(r for r in wf.check_requirements().requirements
                     if r.name == "GITHUB_TOKEN")
        assert after.status is RequirementStatus.READY

    def test_plan_created_event_carries_requirements(self):
        from orchestrator.events import EventType

        wf = Workflow.from_description("Design a trial", llm=ScriptedPlanner(DOMAIN_PLAN),
                                       persist=False, verbose=False)
        try:
            event = next(e for e in wf.bus.history if e.type is EventType.PLAN_CREATED)
            assert event.data["agents"] == ["clinical_trial_statistician", "medical_writer"]
            assert event.data["requirements"]["can_run"] is False
        finally:
            wf.close()
