"""Workflow orchestration: adaptive re-execution, failures, approvals."""

from __future__ import annotations

import time

import pytest

from orchestrator.models import Step, StepStatus
from orchestrator.tools import ToolManager, default_tool_manager

from .conftest import AlwaysFailsTool, diamond_steps, linear_steps


class TestFullRun:
    def test_every_step_completes(self, workflow):
        report = workflow.run_full()
        assert len(report.executed) == 4
        assert report.reused == []
        assert report.ok
        assert all(workflow.status_of(sid) is StepStatus.DONE
                   for sid in workflow.graph.topological_order())

    def test_outputs_are_recorded_and_queryable(self, workflow):
        workflow.run_full()
        assert workflow.query_results("frontend")
        assert set(workflow.outputs()) == set(workflow.graph.topological_order())

    def test_usage_is_metered(self, workflow):
        workflow.run_full()
        usage = workflow.metrics()["usage"]
        assert usage["calls"] == 4
        assert usage["total_tokens"] > 0

    def test_execution_time_is_recorded(self, workflow):
        workflow.run_full()
        assert workflow.get_execution_time("frontend") >= 0

    def test_unknown_step_query_raises(self, workflow):
        with pytest.raises(KeyError):
            workflow.query_results("ghost")


class TestSharedContext:
    def test_downstream_prompt_contains_upstream_output(self, make_workflow):
        workflow = make_workflow()
        workflow.run_full()
        context = workflow.memory.build_context(workflow.graph.get("backend"), workflow.graph)
        assert workflow.query_results("frontend")[:40] in context

    def test_root_step_has_no_prerequisites(self, workflow):
        context = workflow.memory.build_context(workflow.graph.get("frontend"), workflow.graph)
        assert "no prerequisites" in context

    def test_context_is_truncated_to_the_budget(self, make_workflow):
        workflow = make_workflow()
        workflow.memory.char_budget = 100
        workflow.memory.store("frontend", "x" * 5000)
        context = workflow.memory.build_context(workflow.graph.get("backend"), workflow.graph)
        assert "elided" in context
        assert len(context) < 1000


class TestAdaptiveReExecution:
    def test_change_reruns_only_the_affected_cone(self, workflow):
        workflow.run_full()
        report = workflow.handle_step_change(
            "database", new_description="Use MongoDB collections instead of tables")
        assert set(report.executed) == {"database", "testing"}
        assert set(report.reused) == {"frontend", "backend"}

    def test_leaf_change_reruns_only_that_leaf(self, workflow):
        workflow.run_full()
        report = workflow.handle_step_change(
            "testing", new_description="Use property-based tests instead")
        assert report.executed == ["testing"]
        assert len(report.reused) == 3

    def test_identical_rerun_does_not_propagate(self, workflow):
        """The differentiator: a re-run that changes nothing stops there."""
        workflow.run_full()
        report = workflow.handle_step_change("frontend")  # no new description
        assert report.executed == ["frontend"]
        assert set(report.reused) == {"backend", "database", "testing"}

    def test_naive_invalidation_reruns_the_whole_cone(self, make_workflow):
        workflow = make_workflow(smart_invalidation=False)
        workflow.run_full()
        report = workflow.handle_step_change("frontend")
        assert len(report.executed) == 4  # no fingerprinting -> everything downstream

    def test_impact_analysis_runs_nothing(self, workflow):
        workflow.run_full()
        calls_before = workflow.metrics()["usage"]["calls"]
        impact = workflow.impact_of("backend")
        assert impact["affected"] == ["backend", "database", "testing"]
        assert impact["reusable"] == ["frontend"]
        assert workflow.metrics()["usage"]["calls"] == calls_before

    def test_sibling_branch_is_untouched(self, make_workflow):
        workflow = make_workflow(diamond_steps())
        workflow.run_full()
        report = workflow.handle_step_change("b", new_description="Rewrite the left branch in Rust")
        assert set(report.executed) == {"b", "d"}
        assert "c" in report.reused

    def test_change_updates_the_step_description(self, workflow):
        workflow.run_full()
        workflow.handle_step_change("database", new_description="Brand new requirement")
        assert workflow.graph.get("database").description == "Brand new requirement"

    def test_change_on_unknown_step_raises(self, workflow):
        workflow.run_full()
        with pytest.raises(KeyError):
            workflow.handle_step_change("ghost")

    def test_reuse_ratio_is_reported(self, workflow):
        workflow.run_full()
        report = workflow.handle_step_change("testing", new_description="different tests")
        assert report.reuse_ratio == 0.75

    def test_adding_a_step_runs_only_it(self, workflow):
        workflow.run_full()
        report = workflow.add_step(Step(id="deploy", description="Ship it",
                                        agent_role="devops", depends_on=["testing"]))
        assert report.executed == ["deploy"]


class TestToolFailure:
    def test_fallback_is_used_and_recorded(self, make_workflow):
        workflow = make_workflow(tool_manager=default_tool_manager())
        workflow.run_full()
        report = workflow.handle_tool_failure("github")
        assert report is not None
        assert "backend" in report.executed
        assert workflow.results["backend"].used_fallback
        assert workflow.results["backend"].tool_used == "github_cli"

    def test_unaffected_steps_are_reused(self, make_workflow):
        workflow = make_workflow(tool_manager=default_tool_manager())
        workflow.run_full()
        report = workflow.handle_tool_failure("github")
        assert "frontend" in report.reused

    def test_repair_clears_the_broken_flag(self, make_workflow):
        workflow = make_workflow(tool_manager=default_tool_manager())
        workflow.run_full()
        workflow.handle_tool_failure("github")
        assert workflow.tools.broken_tools() == ["github"]
        workflow.handle_tool_repair("github")
        assert workflow.tools.broken_tools() == []

    def test_repair_can_rerun_the_steps_that_fell_back(self, make_workflow):
        workflow = make_workflow(tool_manager=default_tool_manager())
        workflow.run_full()
        workflow.handle_tool_failure("github")
        report = workflow.handle_tool_repair("github", rerun=True)
        assert report is not None and "backend" in report.executed
        assert not workflow.results["backend"].used_fallback

    def test_breaking_an_unused_tool_runs_nothing(self, make_workflow):
        workflow = make_workflow(tool_manager=default_tool_manager())
        workflow.run_full()
        assert workflow.handle_tool_failure("slack") is None

    def test_step_fails_when_no_fallback_works(self, make_workflow):
        tools = ToolManager([AlwaysFailsTool()])
        steps = [Step(id="a", description="Do a thing", agent_role="backend",
                      requires_tool="always_fails", max_retries=0)]
        workflow = make_workflow(steps, tool_manager=tools)
        report = workflow.run_full()
        assert report.failed == ["a"]
        assert workflow.status_of("a") is StepStatus.FAILED


class TestFailureHandling:
    def _exploding_workflow(self, make_workflow, **kwargs):
        tools = ToolManager([AlwaysFailsTool()])
        steps = [
            Step(id="a", description="Root", agent_role="research"),
            Step(id="b", description="Explodes", agent_role="backend",
                 requires_tool="always_fails", depends_on=["a"],
                 max_retries=kwargs.pop("max_retries", 0), retry_backoff_s=0.001),
            Step(id="c", description="Downstream of the failure", agent_role="testing",
                 depends_on=["b"]),
            Step(id="d", description="Independent branch", agent_role="writer",
                 depends_on=["a"]),
        ]
        return make_workflow(steps, tool_manager=tools, **kwargs)

    def test_downstream_is_cancelled_not_run(self, make_workflow):
        workflow = self._exploding_workflow(make_workflow)
        report = workflow.run_full()
        assert report.failed == ["b"]
        assert report.cancelled == ["c"]
        assert workflow.status_of("c") is StepStatus.CANCELLED

    def test_independent_branches_still_complete(self, make_workflow):
        workflow = self._exploding_workflow(make_workflow)
        report = workflow.run_full()
        assert "d" in report.executed
        assert workflow.status_of("d") is StepStatus.DONE

    def test_retries_are_attempted_and_counted(self, make_workflow):
        workflow = self._exploding_workflow(make_workflow, max_retries=2)
        workflow.run_full()
        assert workflow.results["b"].attempts == 3

    def test_failed_run_is_not_ok(self, make_workflow):
        workflow = self._exploding_workflow(make_workflow)
        assert not workflow.run_full().ok

    def test_error_is_recorded_on_the_result(self, make_workflow):
        workflow = self._exploding_workflow(make_workflow)
        workflow.run_full()
        assert "always fails" in workflow.results["b"].error


class TestHumanInTheLoop:
    def _gated_workflow(self, make_workflow):
        steps = [
            Step(id="build", description="Build it", agent_role="backend"),
            Step(id="deploy", description="Deploy to production", agent_role="devops",
                 depends_on=["build"], requires_approval=True),
            Step(id="announce", description="Notify the team", agent_role="writer",
                 depends_on=["deploy"]),
        ]
        return make_workflow(steps)

    def test_run_pauses_at_the_gate(self, make_workflow):
        workflow = self._gated_workflow(make_workflow)
        report = workflow.run_full()
        assert report.awaiting_approval == ["deploy"]
        assert workflow.awaiting_approval() == ["deploy"]
        assert workflow.status_of("announce") is StepStatus.CANCELLED

    def test_approval_then_resume_completes_the_run(self, make_workflow):
        workflow = self._gated_workflow(make_workflow)
        workflow.run_full()
        workflow.approve("deploy")
        report = workflow.resume()
        assert set(report.executed) == {"deploy", "announce"}
        assert report.reused == ["build"]

    def test_pause_before_can_be_set_at_runtime(self, workflow):
        workflow.pause_before("database")
        report = workflow.run_full()
        assert "database" in report.awaiting_approval

    def test_handle_step_pause_is_an_alias(self, workflow):
        workflow.handle_step_pause("database")
        assert workflow.graph.get("database").requires_approval

    def test_approving_an_unknown_step_raises(self, workflow):
        with pytest.raises(KeyError):
            workflow.approve("ghost")


class TestParallelism:
    def test_concurrent_steps_overlap_in_time(self, make_workflow):
        """b and c are independent, so their execution windows must overlap."""
        from orchestrator.llm import StubProvider

        slow = StubProvider(latency_s=0.15)
        workflow = make_workflow(diamond_steps(), llm=slow, parallel=True, max_workers=4)
        workflow.agents.llm = slow
        for agent in ("backend", "frontend", "research", "testing"):
            workflow.agents.get(agent).llm = slow

        workflow.run_full()
        b, c = workflow.results["b"], workflow.results["c"]
        overlap = min(b.ended_at, c.ended_at) - max(b.started_at, c.started_at)
        assert overlap > 0, "independent steps did not run concurrently"

    def test_sequential_mode_does_not_overlap(self, make_workflow):
        from orchestrator.llm import StubProvider

        slow = StubProvider(latency_s=0.1)
        workflow = make_workflow(diamond_steps(), llm=slow, parallel=False)
        for role in ("backend", "frontend", "research", "testing"):
            workflow.agents.get(role).llm = slow

        workflow.run_full()
        b, c = workflow.results["b"], workflow.results["c"]
        assert b.ended_at <= c.started_at or c.ended_at <= b.started_at

    def test_results_are_identical_either_way(self, make_workflow):
        parallel = make_workflow(diamond_steps(), parallel=True)
        parallel.run_full()
        sequential = make_workflow(diamond_steps(), parallel=False)
        sequential.run_full()
        assert parallel.outputs() == sequential.outputs()


class TestCancellation:
    def test_cancel_stops_before_the_next_dependency_level(self, make_workflow):
        import threading
        import time
        from orchestrator.llm import StubProvider

        slow = StubProvider(latency_s=0.2)
        workflow = make_workflow([
            Step(id="a", description="First", agent_role="research"),
            Step(id="b", description="Second", agent_role="backend", depends_on=["a"]),
            Step(id="c", description="Third", agent_role="testing", depends_on=["b"]),
        ], llm=slow, parallel=False)
        for role in ("research", "backend", "testing"):
            workflow.agents.get(role).llm = slow

        holder = {}
        thread = threading.Thread(target=lambda: holder.setdefault("report", workflow.run_full()))
        thread.start()
        deadline = time.time() + 2
        while workflow.status_of("a") is not StepStatus.RUNNING and time.time() < deadline:
            time.sleep(0.01)

        assert workflow.request_cancel()
        assert not workflow.request_cancel(), "a repeated stop request should be idempotent"
        thread.join(timeout=3)

        report = holder["report"]
        assert report.cancel_requested
        assert workflow.status_of("a") is StepStatus.DONE
        assert workflow.status_of("b") is StepStatus.CANCELLED
        assert workflow.status_of("c") is StepStatus.CANCELLED
        assert not workflow.cancellation_requested

    def test_resume_after_cancel_runs_the_unfinished_steps(self, make_workflow):
        workflow = make_workflow([
            Step(id="a", description="First", agent_role="research"),
            Step(id="b", description="Second", agent_role="backend", depends_on=["a"]),
        ], parallel=False)
        workflow.request_cancel()
        first = workflow.run()
        assert first.cancel_requested
        second = workflow.resume()
        assert second.executed == ["a", "b"]
        assert second.ok


class TestReporting:
    def test_metrics_include_the_critical_path(self, make_workflow):
        workflow = make_workflow(diamond_steps())
        workflow.run_full()
        assert workflow.metrics()["parallelism"]["critical_path_length"] == 3

    def test_run_history_accumulates(self, workflow):
        workflow.run_full()
        workflow.handle_step_change("testing", new_description="new tests")
        assert len(workflow.metrics()["runs"]) == 2

    def test_print_summary_does_not_raise(self, workflow, capsys):
        workflow.run_full()
        workflow.print_summary()
        assert "WORKFLOW" in capsys.readouterr().out

    def test_to_dict_is_json_serialisable(self, workflow):
        import json

        workflow.run_full()
        assert json.loads(json.dumps(workflow.to_dict(), default=str))


class TestEditingTheWorkflow:
    """Editing must never leave the graph in an unrunnable state."""

    def _wf(self, make_workflow):
        from orchestrator.models import Step as S
        return make_workflow([
            S(id="a", description="First", agent_role="research"),
            S(id="b", description="Second", agent_role="backend", depends_on=["a"]),
            S(id="c", description="Third", agent_role="testing", depends_on=["b"]),
        ])

    def test_editing_a_description_clears_the_cached_result(self, make_workflow):
        workflow = self._wf(make_workflow)
        workflow.run_full()
        assert workflow.status_of("b") is StepStatus.DONE
        result = workflow.update_step("b", description="Completely different")
        assert result["changed"] == ["description"]
        assert workflow.status_of("b") is StepStatus.PENDING

    def test_editing_role_and_tool(self, make_workflow):
        workflow = self._wf(make_workflow)
        workflow.update_step("b", agent_role="devops", requires_tool="github")
        assert workflow.graph.get("b").agent_role == "devops"
        assert workflow.graph.get("b").requires_tool == "github"

    def test_rewiring_dependencies_changes_the_levels(self, make_workflow):
        workflow = self._wf(make_workflow)
        workflow.update_step("c", depends_on=["a"])
        assert workflow.graph.execution_levels() == [["a"], ["b", "c"]]

    def test_a_cycle_is_refused_and_nothing_changes(self, make_workflow):
        workflow = self._wf(make_workflow)
        before = workflow.graph.topological_order()
        with pytest.raises(ValueError, match="loop"):
            workflow.update_step("a", depends_on=["c"])
        assert workflow.graph.topological_order() == before
        assert workflow.graph.get("a").depends_on == []

    def test_an_unknown_dependency_is_refused(self, make_workflow):
        workflow = self._wf(make_workflow)
        with pytest.raises(ValueError, match="unknown dependency"):
            workflow.update_step("a", depends_on=["ghost"])

    def test_editing_an_unknown_step_raises(self, make_workflow):
        with pytest.raises(KeyError):
            self._wf(make_workflow).update_step("ghost", description="x")

    def test_no_change_reports_nothing_changed(self, make_workflow):
        workflow = self._wf(make_workflow)
        assert workflow.update_step("b", description="Second")["changed"] == []

    def test_edit_with_rerun_executes_immediately(self, make_workflow):
        workflow = self._wf(make_workflow)
        workflow.run_full()
        result = workflow.update_step("b", description="New thing", rerun=True)
        assert "b" in result["report"]["executed"]

    def test_removing_a_step_detaches_dependents(self, make_workflow):
        workflow = self._wf(make_workflow)
        workflow.run_full()
        result = workflow.remove_step("b")
        assert "b" not in workflow.graph
        assert workflow.graph.get("c").depends_on == []
        assert "c" in result["invalidated"]
        assert workflow.status_of("c") is StepStatus.PENDING

    def test_removing_an_unknown_step_raises(self, make_workflow):
        with pytest.raises(KeyError):
            self._wf(make_workflow).remove_step("ghost")

    def test_add_step_without_running_it(self, make_workflow):
        from orchestrator.models import Step as S
        workflow = self._wf(make_workflow)
        result = workflow.add_step(S(id="d", description="Fourth", agent_role="writer",
                                     depends_on=["c"]), rerun=False)
        assert result["added"] == "d"
        assert workflow.status_of("d") is StepStatus.PENDING

    def test_adding_an_invalid_step_leaves_the_graph_clean(self, make_workflow):
        from orchestrator.models import Step as S
        workflow = self._wf(make_workflow)
        with pytest.raises(Exception):
            workflow.add_step(S(id="d", description="x", agent_role="writer",
                                depends_on=["nonexistent"]), rerun=False)
        assert "d" not in workflow.graph, "a rejected step must not linger"
        workflow.graph.validate()

    def test_the_workflow_still_runs_after_editing(self, make_workflow):
        from orchestrator.models import Step as S
        workflow = self._wf(make_workflow)
        workflow.update_step("c", depends_on=["a"])
        workflow.add_step(S(id="d", description="Fourth", agent_role="writer",
                            depends_on=["c"]), rerun=False)
        workflow.remove_step("b")
        report = workflow.run_full()
        assert report.ok
        assert set(report.executed) == {"a", "c", "d"}


class TestConditionalBranches:
    def test_false_condition_skips_branch_but_allows_downstream(self, make_workflow):
        from orchestrator.models import Step as S

        workflow = make_workflow([
            S(id="review", description="Review it", agent_role="research"),
            S(id="repair", description="Repair failures", agent_role="backend",
              depends_on=["review"],
              condition={"source": "review", "operator": "contains",
                         "value": "definitely-not-in-output"}),
            S(id="report", description="Report result", agent_role="writer",
              depends_on=["repair"]),
        ])
        report = workflow.run_full()
        assert report.not_applicable == ["repair"]
        assert workflow.status_of("repair") is StepStatus.NOT_APPLICABLE
        assert workflow.status_of("report") is StepStatus.DONE

    def test_webhook_value_can_select_a_branch(self, make_workflow):
        from orchestrator.models import Step as S

        workflow = make_workflow([
            S(id="urgent", description="Handle urgent event", agent_role="writer",
              condition={"source": "trigger.priority", "operator": "equals",
                         "value": "high"})
        ])
        taken = workflow.run_full(runtime_inputs={"priority": "HIGH"})
        assert taken.executed == ["urgent"]
        skipped = workflow.run_full(runtime_inputs={"priority": "low"})
        assert skipped.not_applicable == ["urgent"]

    def test_condition_rejects_code_like_operators(self):
        from orchestrator.models import Step as S

        with pytest.raises(ValueError, match="invalid step condition"):
            S(id="unsafe", description="x", agent_role="writer",
              condition={"source": "a", "operator": "eval", "value": "x"})


class TestAgentHandoffs:
    def test_handoff_is_delivered_and_survives_restart(self, tmp_path, stub_llm):
        from orchestrator import StateManager
        from orchestrator.models import Step as S

        db = str(tmp_path / "handoffs.db")
        workflow = __import__("orchestrator").Workflow([
            S(id="research", description="Research", agent_role="research"),
            S(id="write", description="Write", agent_role="writer", depends_on=["research"]),
        ], run_id="handoff", state=StateManager(db), llm=stub_llm, verbose=False)
        workflow._capture_handoffs(
            workflow.graph.get("research"),
            'Done\nHANDOFF: {"to":"write","message":"Use finding 42"}')
        assert workflow.agent_messages[0]["message"] == "Use finding 42"
        workflow.close()

        resumed = __import__("orchestrator").Workflow.resume_from(
            "handoff", state=StateManager(db), llm=stub_llm, verbose=False)
        assert resumed.agent_messages == [
            {"from": "research", "to": "write", "message": "Use finding 42"}]
        resumed.close()
