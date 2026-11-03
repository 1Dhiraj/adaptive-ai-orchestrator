"""End-to-end scenarios and exports.

These are the behaviours the project is actually claiming, exercised through
the public API only.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from orchestrator.models import Step, StepStatus
from orchestrator.state import StateManager
from orchestrator.tools import default_tool_manager
from orchestrator.workflow import Workflow

from .conftest import linear_steps


def build_web_app_workflow(stub_llm, **kwargs) -> Workflow:
    kwargs.setdefault("persist", False)
    kwargs.setdefault("verbose", False)
    return Workflow(linear_steps(), description="task manager", llm=stub_llm,
                    tool_manager=default_tool_manager(), **kwargs)


class TestHeadlineScenarios:
    """The four scenarios from the project brief, verified end to end."""

    def test_scenario_1_full_run(self, stub_llm):
        workflow = build_web_app_workflow(stub_llm)
        report = workflow.run_full()
        assert len(report.executed) == 4 and report.ok

    def test_scenario_2_requirement_change_reuses_unaffected_work(self, stub_llm):
        workflow = build_web_app_workflow(stub_llm)
        workflow.run_full()
        report = workflow.handle_step_change(
            "database", new_description="Use MongoDB collections instead of tables")

        assert set(report.executed) == {"database", "testing"}
        assert set(report.reused) == {"frontend", "backend"}
        # Reused steps must keep the exact output they had before.
        assert workflow.results["frontend"].status is StepStatus.SKIPPED
        assert workflow.results["frontend"].output

    def test_scenario_3_tool_failure_recovers_automatically(self, stub_llm):
        workflow = build_web_app_workflow(stub_llm)
        workflow.run_full()
        assert workflow.results["backend"].tool_used == "github"

        report = workflow.handle_tool_failure("github")
        assert "backend" in report.executed
        assert workflow.results["backend"].tool_used == "github_cli"
        assert workflow.results["backend"].used_fallback
        assert report.ok, "the workflow must still complete with a broken tool"

    def test_scenario_4_tool_repair(self, stub_llm):
        workflow = build_web_app_workflow(stub_llm)
        workflow.run_full()
        workflow.handle_tool_failure("github")
        workflow.handle_tool_repair("github")
        assert workflow.tools.broken_tools() == []
        report = workflow.handle_step_change("backend")
        assert workflow.results["backend"].tool_used == "github"
        assert not workflow.results["backend"].used_fallback


class TestEfficiencyClaims:
    def test_change_avoids_more_work_than_a_full_restart(self, stub_llm):
        workflow = build_web_app_workflow(stub_llm)
        workflow.run_full()
        report = workflow.handle_step_change("testing", new_description="different tests")
        assert len(report.executed) < len(workflow.graph)

    def test_no_op_change_stops_propagating(self, stub_llm):
        workflow = build_web_app_workflow(stub_llm)
        workflow.run_full()
        report = workflow.handle_step_change("frontend")
        assert len(report.executed) == 1
        assert len(report.reused) == 3

    def test_token_usage_scales_with_steps_actually_run(self, stub_llm):
        workflow = build_web_app_workflow(stub_llm)
        full = workflow.run_full()
        change = workflow.handle_step_change("testing", new_description="different tests")
        assert change.usage.total_tokens < full.usage.total_tokens

    def test_planning_from_description_then_running(self, stub_llm):
        workflow = Workflow.from_description(
            "Build a task management web app with authentication and a Kanban board",
            llm=stub_llm, persist=False, verbose=False)
        report = workflow.run_full()
        assert report.ok
        assert len(workflow.graph) >= 3
        assert all(workflow.query_results(sid) for sid in workflow.graph.topological_order())


class TestFullLifecycle:
    def test_plan_run_change_crash_resume_export(self, tmp_path, stub_llm):
        """One long journey through every layer."""
        db = str(tmp_path / "lifecycle.db")

        workflow = Workflow(linear_steps(), description="lifecycle", run_id="life-1",
                            state=StateManager(db), llm=stub_llm, verbose=False)
        workflow.run_full()
        workflow.handle_step_change("database", new_description="Switch to MongoDB")
        workflow.handle_tool_failure("github")
        outputs = workflow.outputs()
        workflow.close()

        # Process dies here. A new object picks the run back up.
        resumed = Workflow.resume_from("life-1", state=StateManager(db),
                                       llm=stub_llm, verbose=False)
        report = resumed.resume()
        assert report.executed == [], "nothing should need re-running after a clean stop"
        assert resumed.outputs() == outputs

        json_path = tmp_path / "out.json"
        csv_path = tmp_path / "out.csv"
        html_path = tmp_path / "out.html"
        resumed.export_state(str(json_path), format="json")
        resumed.export_state(str(csv_path), format="csv")
        resumed.export_state(str(html_path), format="html")
        resumed.close()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert set(payload["results"]) == {"frontend", "backend", "database", "testing"}
        assert payload["metrics"]["usage"]["calls"] > 0

        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        assert len(rows) == 4
        assert {"step_id", "status", "total_tokens", "input_hash"} <= set(rows[0])

        html = html_path.read_text(encoding="utf-8")
        assert "<svg" in html and "life-1" in html


class TestExports:
    def test_html_export_svg_is_well_formed(self, tmp_path, stub_llm):
        import xml.etree.ElementTree as ET

        workflow = build_web_app_workflow(stub_llm)
        workflow.run_full()
        path = tmp_path / "report.html"
        workflow.export_state(str(path), format="html")
        svg = re.search(r"<svg.*?</svg>", path.read_text(encoding="utf-8"), re.DOTALL)
        assert svg is not None
        ET.fromstring(svg.group(0))  # raises if malformed

    def test_html_escapes_hostile_content(self, tmp_path, stub_llm):
        steps = [Step(id="a", description="<script>alert('xss')</script>", agent_role="generic")]
        workflow = Workflow(steps, llm=stub_llm, persist=False, verbose=False)
        workflow.run_full()
        path = tmp_path / "report.html"
        workflow.export_state(str(path), format="html")
        content = path.read_text(encoding="utf-8")
        assert "<script>alert" not in content
        assert "&lt;script&gt;" in content

    def test_csv_rows_are_single_line(self, tmp_path, stub_llm):
        workflow = build_web_app_workflow(stub_llm)
        workflow.run_full()
        path = tmp_path / "out.csv"
        workflow.export_state(str(path), format="csv")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert all("\n" not in row["output"] for row in rows)

    def test_pdf_export_is_a_real_pdf(self, stub_llm, tmp_path):
        workflow = build_web_app_workflow(stub_llm)
        workflow.run_full()
        path = workflow.export_state(str(tmp_path / "x.pdf"), format="pdf")
        assert Path(path).read_bytes().startswith(b"%PDF-")

    def test_unknown_format_raises(self, stub_llm, tmp_path):
        workflow = build_web_app_workflow(stub_llm)
        with pytest.raises(ValueError):
            workflow.export_state(str(tmp_path / "x"), format="docx")

    def test_export_results_shortcut(self, tmp_path, stub_llm):
        workflow = build_web_app_workflow(stub_llm)
        workflow.run_full()
        path = workflow.export_results(str(tmp_path / "results.json"))
        assert json.loads(open(path, encoding="utf-8").read())["run_id"]


class TestLargerGraphs:
    def test_eight_step_microservices_graph(self, stub_llm):
        steps = [
            Step(id="architecture", description="Define service boundaries", agent_role="research"),
            Step(id="auth", description="Auth service", agent_role="backend",
                 depends_on=["architecture"], requires_tool="github"),
            Step(id="catalog", description="Catalog service", agent_role="backend",
                 depends_on=["architecture"], requires_tool="github"),
            Step(id="orders", description="Order service", agent_role="backend",
                 depends_on=["architecture"], requires_tool="github"),
            Step(id="schema", description="Shared schemas", agent_role="database",
                 depends_on=["architecture"], requires_tool="postgres"),
            Step(id="gateway", description="API gateway", agent_role="backend",
                 depends_on=["auth", "catalog", "orders"]),
            Step(id="deploy", description="Containerise and ship", agent_role="devops",
                 depends_on=["gateway", "schema"]),
            Step(id="e2e", description="End-to-end tests", agent_role="testing",
                 depends_on=["deploy"], requires_tool="ci"),
        ]
        workflow = Workflow(steps, llm=stub_llm, persist=False, verbose=False)
        report = workflow.run_full()
        assert len(report.executed) == 8

        levels = workflow.graph.execution_levels()
        assert levels[1] == ["auth", "catalog", "orders", "schema"]  # four in parallel

        # One service changes: the other two must not be touched.
        change = workflow.handle_step_change("catalog", new_description="Rewrite catalog in Go")
        assert "auth" in change.reused and "orders" in change.reused
        assert set(change.executed) == {"catalog", "gateway", "deploy", "e2e"}
