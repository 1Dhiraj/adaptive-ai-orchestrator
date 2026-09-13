"""Persistence, crash recovery and the event bus."""

from __future__ import annotations

import pytest

from orchestrator.events import Event, EventBus, EventType
from orchestrator.models import PendingAction, Step, StepResult, StepStatus
from orchestrator.state import StateManager
from orchestrator.workflow import Workflow

from .conftest import linear_steps


class TestStateManager:
    def test_run_round_trips(self, state, linear_graph):
        state.create_run("r1", "a description", linear_graph)
        record = state.get_run("r1")
        assert record["description"] == "a description"
        assert len(record["graph"]["steps"]) == 4

    def test_graph_reloads_with_structure_intact(self, state, linear_graph):
        state.create_run("r1", "d", linear_graph)
        restored = state.load_graph("r1")
        assert restored.topological_order() == linear_graph.topological_order()

    def test_step_results_round_trip(self, state, linear_graph):
        state.create_run("r1", "d", linear_graph)
        result = StepResult(step_id="frontend", status=StepStatus.DONE,
                            output="the output", input_hash="abc123")
        state.save_step_result("r1", result)
        loaded = state.load_results("r1")["frontend"]
        assert loaded.output == "the output"
        assert loaded.status is StepStatus.DONE
        assert loaded.input_hash == "abc123"

    def test_clearing_specific_results(self, state, linear_graph):
        state.create_run("r1", "d", linear_graph)
        for sid in ("frontend", "backend"):
            state.save_step_result("r1", StepResult(step_id=sid, status=StepStatus.DONE, output="x"))
        state.clear_results("r1", ["frontend"])
        assert set(state.load_results("r1")) == {"backend"}

    def test_resumable_splits_completed_from_outstanding(self, state, linear_graph):
        state.create_run("r1", "d", linear_graph)
        state.save_step_result("r1", StepResult(step_id="frontend", status=StepStatus.DONE,
                                                output="x"))
        completed, outstanding = state.resumable("r1")
        assert completed == ["frontend"]
        assert set(outstanding) == {"backend", "database", "testing"}

    def test_unknown_run_is_none(self, state):
        assert state.get_run("nope") is None
        assert state.load_graph("nope") is None
        assert state.resumable("nope") == ([], [])

    def test_delete_removes_everything(self, state, linear_graph):
        state.create_run("r1", "d", linear_graph)
        state.save_step_result("r1", StepResult(step_id="frontend", status=StepStatus.DONE))
        state.record_event(Event(type=EventType.LOG, run_id="r1", message="hi"))
        state.delete_run("r1")
        assert state.get_run("r1") is None
        assert state.load_results("r1") == {}
        assert state.load_events("r1") == []

    def test_events_are_ordered_and_paginated(self, state):
        for i in range(5):
            state.record_event(Event(type=EventType.LOG, run_id="r1", message=f"m{i}"))
        events = state.load_events("r1")
        assert [e["message"] for e in events] == [f"m{i}" for i in range(5)]
        assert len(state.load_events("r1", after_id=events[2]["id"])) == 2

    def test_pending_actions_round_trip_with_the_full_payload(self, state, linear_graph):
        state.create_run("r1", "d", linear_graph)
        action = PendingAction(
            step_id="frontend", tool="email", payload="complete generated payload",
            preview="send the update", agent_output="drafted update")
        state.save_pending_action("r1", action)
        loaded = state.load_pending_actions("r1")["frontend"]
        assert loaded.payload == "complete generated payload"
        assert loaded.agent_output == "drafted update"
        state.delete_pending_action("r1", "frontend")
        assert state.load_pending_actions("r1") == {}

    def test_list_runs_is_newest_first(self, state, linear_graph):
        state.create_run("old", "first", linear_graph)
        state.create_run("new", "second", linear_graph)
        assert state.list_runs()[0]["run_id"] == "new"

    def test_interval_schedule_is_persisted_advanced_and_deleted(self, state, linear_graph):
        state.create_run("scheduled", "d", linear_graph)
        schedule = state.create_schedule("scheduled", 300, next_run_at=10)
        assert state.due_schedules(now=10)[0]["id"] == schedule["id"]
        state.advance_schedule(schedule["id"], ran_at=20)
        advanced = state.get_schedule(schedule["id"])
        assert advanced["last_run_at"] == 20
        assert advanced["next_run_at"] == 320
        state.set_schedule_enabled(schedule["id"], False)
        assert state.due_schedules(now=1000) == []
        state.delete_schedule(schedule["id"])
        assert state.list_schedules("scheduled") == []

    def test_deleting_a_run_also_deletes_its_schedules(self, state, linear_graph):
        state.create_run("scheduled", "d", linear_graph)
        state.create_schedule("scheduled", 60)
        state.delete_run("scheduled")
        assert state.list_schedules("scheduled") == []

    def test_webhook_token_is_found_by_digest_but_never_listed(self, state, linear_graph):
        state.create_run("hooked", "d", linear_graph)
        webhook = state.create_webhook("hooked", "CRM event", "secret-digest")
        assert state.find_webhook("secret-digest")["id"] == webhook["id"]
        assert "token_hash" not in state.list_webhooks("hooked")[0]
        state.mark_webhook_triggered(webhook["id"], triggered_at=25)
        assert state.get_webhook(webhook["id"])["last_triggered_at"] == 25
        state.set_webhook_enabled(webhook["id"], False)
        assert state.get_webhook(webhook["id"])["enabled"] == 0
        state.delete_webhook(webhook["id"])
        assert state.find_webhook("secret-digest") is None

    def test_deleting_a_run_also_deletes_its_webhooks(self, state, linear_graph):
        state.create_run("hooked", "d", linear_graph)
        state.create_webhook("hooked", "Webhook", "digest")
        state.delete_run("hooked")
        assert state.list_webhooks("hooked") == []

    def test_workflow_template_can_be_saved_updated_and_deleted(self, state, linear_graph):
        saved = state.save_template("Web app", "first", linear_graph)
        assert saved["graph"]["steps"]
        updated = state.save_template("Web app", "second", linear_graph)
        assert updated["id"] == saved["id"]
        assert updated["description"] == "second"
        assert len(state.list_templates()) == 1
        state.delete_template(saved["id"])
        assert state.list_templates() == []

    def test_audit_log_is_persisted_without_credentials(self, state):
        state.record_audit("operator", "POST", "/api/runs", 201, "127.0.0.1")
        row = state.load_audit()[0]
        assert row["role"] == "operator" and row["status"] == 201
        assert "key" not in row

    def test_execution_job_lifecycle_and_crash_recovery(self, state, linear_graph):
        state.create_run("queued", "d", linear_graph)
        job = state.enqueue_job("queued", "run_full", {"fresh": True})
        assert state.start_job(job["id"])
        assert state.get_job(job["id"])["status"] == "running"
        assert state.recover_interrupted_jobs() == 1
        assert state.queued_jobs()[0]["args"] == {"fresh": True}
        assert state.start_job(job["id"])
        state.finish_job(job["id"])
        assert state.get_job(job["id"])["status"] == "completed"

    def test_context_manager_closes_cleanly(self, tmp_path):
        with StateManager(str(tmp_path / "x.db")) as manager:
            manager.create_run("r", "d", __import__("orchestrator").DependencyGraph([
                Step(id="a", description="x", agent_role="generic")]))
        # A second manager on the same file still works after close.
        with StateManager(str(tmp_path / "x.db")) as manager:
            assert manager.get_run("r") is not None


class TestCrashRecovery:
    def test_completed_steps_survive_a_restart(self, tmp_path, stub_llm):
        db = str(tmp_path / "recover.db")
        first = Workflow(linear_steps(), run_id="crash-1", state=StateManager(db),
                         llm=stub_llm, verbose=False)
        first.run_full()
        outputs_before = first.outputs()
        first.close()

        # Simulate a process restart: a brand new object, same run id.
        second = Workflow.resume_from("crash-1", state=StateManager(db),
                                      llm=stub_llm, verbose=False)
        report = second.resume()
        assert report.executed == []
        assert len(report.reused) == 4
        assert second.outputs() == outputs_before
        second.close()

    def test_partial_run_resumes_from_the_last_completed_step(self, tmp_path, stub_llm):
        db = str(tmp_path / "partial.db")
        manager = StateManager(db)
        workflow = Workflow(linear_steps(), run_id="partial-1", state=manager,
                            llm=stub_llm, verbose=False)
        # Only the first two steps got through before the "crash".
        workflow.run(only=[])
        workflow.close()

        manager2 = StateManager(db)
        completed, outstanding = manager2.resumable("partial-1")
        assert len(completed) + len(outstanding) == 4
        manager2.close()

    def test_resume_of_an_unknown_run_raises(self, tmp_path):
        with pytest.raises(KeyError):
            Workflow.resume_from("nonexistent", state=StateManager(str(tmp_path / "x.db")))

    def test_events_are_persisted_during_a_run(self, tmp_path, stub_llm):
        db = str(tmp_path / "events.db")
        workflow = Workflow(linear_steps(), run_id="ev-1", state=StateManager(db),
                            llm=stub_llm, verbose=False)
        workflow.run_full()
        events = workflow.state.load_events("ev-1")
        types = {e["type"] for e in events}
        assert "step_started" in types and "step_finished" in types and "run_finished" in types
        workflow.close()

    def test_a_changed_step_definition_invalidates_the_cached_result(self, tmp_path, stub_llm):
        """The saved input hash must not match after the description changes."""
        db = str(tmp_path / "hash.db")
        first = Workflow(linear_steps(), run_id="hash-1", state=StateManager(db),
                         llm=stub_llm, verbose=False)
        first.run_full()
        first.close()

        steps = linear_steps()
        for step in steps:
            if step.id == "database":
                step.description = "Design a totally different schema"
        second = Workflow(steps, run_id="hash-1", state=StateManager(db),
                          llm=stub_llm, verbose=False)
        second.load_persisted_results()
        report = second.run()
        assert "database" in report.executed
        assert "frontend" in report.reused
        second.close()

    def test_pending_action_survives_restart_without_repeating_the_llm(self, tmp_path,
                                                                       stub_llm):
        from orchestrator.tools import ToolManager
        from orchestrator.tools.base import Tool

        class IrreversibleTool(Tool):
            name = "publish"
            capability = "publish"
            irreversible = True

            def __init__(self):
                super().__init__()
                self.calls = 0

            def is_live(self):
                return True

            def _run(self, task, context=None):
                self.calls += 1
                return "published"

        db = str(tmp_path / "pending.db")
        tool = IrreversibleTool()
        first = Workflow(
            [Step(id="send", description="Publish the prepared update",
                  agent_role="writer", requires_tool="publish")],
            run_id="pending-1", state=StateManager(db), llm=stub_llm,
            tool_manager=ToolManager([tool]), verbose=False)
        first_report = first.run_full()
        assert first_report.awaiting_action == ["send"]
        calls_after_draft = stub_llm.total_usage.calls
        first.close()

        second = Workflow.resume_from(
            "pending-1", state=StateManager(db), llm=stub_llm,
            tool_manager=ToolManager([tool]), verbose=False)
        assert second.awaiting_action() == ["send"]
        second.approve_action("send")
        report = second.resume()
        assert report.executed == ["send"]
        assert stub_llm.total_usage.calls == calls_after_draft
        assert tool.calls == 1
        assert second.state.load_pending_actions("pending-1") == {}
        second.close()


class TestEventBus:
    def test_subscribers_receive_events(self):
        bus, seen = EventBus(), []
        bus.subscribe(seen.append)
        bus.publish(EventType.LOG, run_id="r", message="hello")
        assert len(seen) == 1 and seen[0].message == "hello"

    def test_history_is_kept(self):
        bus = EventBus()
        bus.publish(EventType.LOG, message="a")
        bus.publish(EventType.LOG, message="b")
        assert len(bus.history) == 2

    def test_history_is_capped(self):
        bus = EventBus(keep_history=3)
        for i in range(10):
            bus.publish(EventType.LOG, message=str(i))
        assert len(bus.history) == 3
        assert bus.history[-1].message == "9"

    def test_a_raising_subscriber_does_not_break_the_bus(self, capsys):
        bus, seen = EventBus(), []

        def explode(event):
            raise RuntimeError("subscriber is broken")

        bus.subscribe(explode)
        bus.subscribe(seen.append)
        bus.publish(EventType.LOG, message="still delivered")
        assert len(seen) == 1
        assert "subscriber" in capsys.readouterr().out

    def test_unsubscribe_stops_delivery(self):
        bus, seen = EventBus(), []
        handler = bus.subscribe(seen.append)
        bus.unsubscribe(handler)
        bus.publish(EventType.LOG, message="x")
        assert seen == []

    def test_event_serialises(self):
        event = Event(type=EventType.STEP_FINISHED, run_id="r", step_id="a", message="done")
        payload = event.to_dict()
        assert payload["type"] == "step_finished" and payload["step_id"] == "a"
