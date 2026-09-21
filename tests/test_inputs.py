"""Human input collection and irreversible-action approval.

The n8n-shaped behaviour: the workflow pauses to ask you for what it cannot
invent, and stops before doing anything it cannot undo.
"""

from __future__ import annotations

import pytest

from orchestrator.inputs import (
    InputError,
    InputRequest,
    InputType,
    parse_declared_inputs,
    render_for_prompt,
    unsatisfied,
    values_of,
)
from orchestrator.models import Step, StepStatus
from orchestrator.tools import ToolManager
from orchestrator.tools.base import Tool


class LiveIrreversibleTool(Tool):
    """Stands in for a configured email/Slack tool: real and unundoable."""

    name = "send_it"
    capability = "notify"
    description = "Send something that cannot be recalled"
    fallbacks: list = []
    side_effect = True
    irreversible = True

    def __init__(self) -> None:
        super().__init__()
        self.sent: list = []

    def is_live(self) -> bool:
        return True

    def _run(self, task, context=None):
        self.sent.append({"payload": task, "context": context or {}})
        return f"[send_it] delivered ({len(task)} chars)"


class TestValidation:
    def test_email_is_validated(self):
        request = InputRequest(name="to", prompt="?", type=InputType.EMAIL)
        assert request.coerce("a@b.co") == "a@b.co"
        with pytest.raises(InputError, match="email"):
            request.coerce("nope")

    def test_url_is_validated(self):
        request = InputRequest(name="u", prompt="?", type=InputType.URL)
        assert request.coerce("https://x.dev/y")
        with pytest.raises(InputError, match="URL"):
            request.coerce("ftp://x")

    def test_number_coerces_int_and_float(self):
        request = InputRequest(name="n", prompt="?", type=InputType.NUMBER)
        assert request.coerce("42") == 42
        assert request.coerce("1.5") == 1.5
        with pytest.raises(InputError, match="number"):
            request.coerce("many")

    @pytest.mark.parametrize("raw,expected", [
        ("yes", True), ("y", True), ("true", True), ("1", True),
        ("no", False), ("n", False), ("false", False), ("0", False),
    ])
    def test_boolean_forms(self, raw, expected):
        request = InputRequest(name="b", prompt="?", type=InputType.BOOLEAN)
        assert request.coerce(raw) is expected

    def test_boolean_rejects_nonsense(self):
        request = InputRequest(name="b", prompt="?", type=InputType.BOOLEAN)
        with pytest.raises(InputError, match="yes/no"):
            request.coerce("maybe")

    def test_choice_restricts_to_options(self):
        request = InputRequest(name="c", prompt="?", type=InputType.CHOICE,
                               options=["a", "b"])
        assert request.coerce("a") == "a"
        with pytest.raises(InputError, match="one of"):
            request.coerce("z")

    def test_choice_without_options_degrades_to_text(self):
        assert InputRequest(name="c", prompt="?", type=InputType.CHOICE).type is InputType.TEXT

    def test_required_empty_is_rejected(self):
        with pytest.raises(InputError, match="required"):
            InputRequest(name="x", prompt="?").coerce("")

    def test_optional_empty_yields_the_default(self):
        request = InputRequest(name="x", prompt="?", required=False, default="fallback")
        assert request.coerce("") == "fallback"

    def test_unknown_type_falls_back_to_text(self):
        assert InputRequest(name="x", prompt="?", type="hologram").type is InputType.TEXT


class TestSatisfaction:
    def test_required_without_default_is_unsatisfied(self):
        assert not InputRequest(name="x", prompt="?").satisfied

    def test_default_satisfies(self):
        assert InputRequest(name="x", prompt="?", default="d").satisfied

    def test_optional_satisfies(self):
        assert InputRequest(name="x", prompt="?", required=False).satisfied

    def test_providing_satisfies(self):
        request = InputRequest(name="x", prompt="?")
        request.provide("value")
        assert request.satisfied and request.effective_value == "value"

    def test_values_of_skips_unanswered(self):
        requests = [InputRequest(name="a", prompt="?"),
                    InputRequest(name="b", prompt="?", default="d")]
        assert values_of(requests) == {"b": "d"}

    def test_unsatisfied_lists_only_blockers(self):
        requests = [InputRequest(name="a", prompt="?"),
                    InputRequest(name="b", prompt="?", default="d")]
        assert [r.name for r in unsatisfied(requests)] == ["a"]


class TestSecrets:
    def test_secret_value_is_redacted_in_serialisation(self):
        request = InputRequest(name="key", prompt="?", type=InputType.SECRET)
        request.provide("super-secret")
        assert request.to_dict()["value"] == "***"

    def test_secret_is_withheld_from_agent_prompts(self):
        secret = InputRequest(name="api_key", prompt="?", type=InputType.SECRET)
        secret.provide("sk-live-123")
        plain = InputRequest(name="channel", prompt="?")
        plain.provide("#general")

        rendered = render_for_prompt([secret, plain])
        assert "sk-live-123" not in rendered
        assert "withheld" in rendered
        assert "#general" in rendered

    def test_secret_default_is_not_serialised(self):
        request = InputRequest(name="k", prompt="?", type=InputType.SECRET, default="x")
        assert request.to_dict()["default"] is None


class TestParsing:
    def test_declared_inputs_are_parsed(self):
        parsed = parse_declared_inputs([
            {"name": "to", "prompt": "Who?", "type": "email"},
            {"name": "when", "prompt": "When?", "type": "choice", "options": ["now"]},
        ])
        assert [r.name for r in parsed] == ["to", "when"]
        assert parsed[0].type is InputType.EMAIL

    def test_nameless_and_duplicate_entries_are_dropped(self):
        parsed = parse_declared_inputs([
            {"prompt": "no name"}, {"name": "a", "prompt": "?"},
            {"name": "a", "prompt": "duplicate"}, "not a dict",
        ])
        assert [r.name for r in parsed] == ["a"]

    def test_step_accepts_plain_dicts(self):
        step = Step(id="s", description="d", agent_role="generic",
                    inputs=[{"name": "x", "prompt": "?"}])
        assert isinstance(step.inputs[0], InputRequest)


class TestStepIntegration:
    def test_input_values_change_the_definition_hash(self):
        step = Step(id="s", description="d", agent_role="generic",
                    inputs=[InputRequest(name="x", prompt="?")])
        before = step.definition_hash()
        step.provide_input("x", "value")
        assert step.definition_hash() != before

    def test_unknown_input_name_raises(self):
        step = Step(id="s", description="d", agent_role="generic")
        with pytest.raises(KeyError):
            step.provide_input("nope", "v")

    def test_step_serialises_with_inputs(self):
        import json

        step = Step(id="s", description="d", agent_role="generic",
                    inputs=[InputRequest(name="x", prompt="?", type=InputType.EMAIL)])
        payload = json.loads(json.dumps(step.to_dict()))
        assert payload["inputs"][0]["type"] == "email"
        assert Step.from_dict(payload).inputs[0].name == "x"


class TestWorkflowPausesForInput:
    def _workflow(self, make_workflow, **kwargs):
        steps = [
            Step(id="draft", description="Draft it.", agent_role="writer"),
            Step(id="send", description="Send it.", agent_role="devops",
                 depends_on=["draft"],
                 inputs=[
                     InputRequest(name="recipient", prompt="To whom?", type=InputType.EMAIL),
                     InputRequest(name="urgency", prompt="How urgent?", type=InputType.CHOICE,
                                  options=["normal", "high"], default="normal"),
                 ]),
        ]
        return make_workflow(steps, **kwargs)

    def test_run_stops_at_the_step_needing_input(self, make_workflow):
        workflow = self._workflow(make_workflow)
        report = workflow.run_full()
        assert report.awaiting_input == ["send"]
        assert workflow.status_of("send") is StepStatus.AWAITING_INPUT
        assert workflow.status_of("draft") is StepStatus.DONE

    def test_only_blocking_inputs_are_reported_as_pending(self, make_workflow):
        workflow = self._workflow(make_workflow)
        workflow.run_full()
        # 'urgency' has a default, so it does not block.
        assert [r.name for r in workflow.pending_inputs()["send"]] == ["recipient"]

    def test_the_form_offers_every_field(self, make_workflow):
        workflow = self._workflow(make_workflow)
        workflow.run_full()
        assert [r.name for r in workflow.input_form()["send"]] == ["recipient", "urgency"]

    def test_supplying_the_input_unblocks_the_run(self, make_workflow):
        workflow = self._workflow(make_workflow)
        workflow.run_full()
        workflow.provide_inputs({"recipient": "a@b.co"})
        report = workflow.resume()
        assert "send" in report.executed
        assert workflow.status_of("send") is StepStatus.DONE

    def test_invalid_input_is_rejected(self, make_workflow):
        workflow = self._workflow(make_workflow)
        workflow.run_full()
        with pytest.raises(InputError):
            workflow.provide_input("send", "recipient", "not-an-email")

    def test_a_rejected_form_applies_nothing(self, make_workflow):
        """One bad field must not half-apply the rest."""
        workflow = self._workflow(make_workflow)
        workflow.run_full()
        with pytest.raises(InputError):
            workflow.provide_inputs({"urgency": "high", "recipient": "bad"}, step_id="send")
        step = workflow.graph.get("send")
        assert not any(i.provided for i in step.inputs)

    def test_values_reach_the_agent_prompt(self, make_workflow):
        workflow = self._workflow(make_workflow)
        workflow.run_full()
        workflow.provide_inputs({"recipient": "a@b.co"})
        workflow.resume()
        prompt = workflow.agents.get("devops").build_prompt(
            workflow.graph.get("send"), "(none)", workflow.tools)
        assert "a@b.co" in prompt
        assert "authoritative" in prompt

    def test_changing_an_input_forces_a_rerun(self, make_workflow):
        workflow = self._workflow(make_workflow)
        workflow.run_full()
        workflow.provide_inputs({"recipient": "a@b.co"})
        workflow.resume()

        workflow.provide_input("send", "recipient", "different@b.co")
        report = workflow.run()
        assert "send" in report.executed, "a changed answer must invalidate the cached result"

    def test_downstream_of_a_blocked_step_is_cancelled(self, make_workflow):
        steps = [
            Step(id="ask", description="Needs input.", agent_role="writer",
                 inputs=[InputRequest(name="x", prompt="?")]),
            Step(id="after", description="Depends on it.", agent_role="writer",
                 depends_on=["ask"]),
        ]
        workflow = make_workflow(steps)
        report = workflow.run_full()
        assert report.awaiting_input == ["ask"]
        assert "after" in report.cancelled

    def test_blocked_on_human_summarises_everything(self, make_workflow):
        workflow = self._workflow(make_workflow)
        workflow.run_full()
        assert workflow.blocked_on_human()["inputs"] == ["send"]


class TestIrreversibleActionApproval:
    def _workflow(self, make_workflow, approval="live"):
        tool = LiveIrreversibleTool()
        tools = ToolManager([tool])
        steps = [Step(id="send", description="Send the notice.", agent_role="devops",
                      requires_tool="send_it")]
        workflow = make_workflow(steps, tool_manager=tools, action_approval=approval)
        return workflow, tool

    def test_real_irreversible_action_is_held(self, make_workflow):
        workflow, tool = self._workflow(make_workflow)
        report = workflow.run_full()
        assert report.awaiting_action == ["send"]
        assert workflow.status_of("send") is StepStatus.AWAITING_ACTION
        assert tool.sent == [], "nothing may be sent before approval"

    def test_the_preview_describes_the_action(self, make_workflow):
        workflow, _ = self._workflow(make_workflow)
        workflow.run_full()
        action = workflow.pending_actions()["send"]
        assert action.tool == "send_it"
        assert action.irreversible
        assert "REAL" in action.preview

    def test_approving_performs_it_without_a_new_llm_call(self, make_workflow):
        workflow, tool = self._workflow(make_workflow)
        workflow.run_full()
        workflow.approve_action("send")
        report = workflow.resume()

        assert tool.sent, "the action must actually happen once approved"
        assert "send" in report.executed
        assert report.usage.calls == 0, "approval must not regenerate the work"
        assert workflow.status_of("send") is StepStatus.DONE

    def test_rejecting_fails_the_step_without_acting(self, make_workflow):
        workflow, tool = self._workflow(make_workflow)
        workflow.run_full()
        workflow.reject_action("send", reason="wrong wording")
        assert tool.sent == []
        assert workflow.status_of("send") is StepStatus.FAILED
        assert "wrong wording" in workflow.results["send"].error

    def test_rejecting_an_unknown_action_raises(self, make_workflow):
        workflow, _ = self._workflow(make_workflow)
        with pytest.raises(KeyError):
            workflow.reject_action("send")

    def test_simulated_tools_are_not_gated_by_default(self, make_workflow):
        """A simulated email sends nothing, so approving it is pointless friction."""
        steps = [Step(id="send", description="Send it.", agent_role="devops",
                      requires_tool="email")]
        workflow = make_workflow(steps)  # email has no credentials -> simulated
        report = workflow.run_full()
        assert report.awaiting_action == []
        assert workflow.status_of("send") is StepStatus.DONE

    def test_all_mode_gates_simulated_actions_too(self, make_workflow):
        steps = [Step(id="send", description="Send it.", agent_role="devops",
                      requires_tool="email")]
        workflow = make_workflow(steps, action_approval="all")
        assert workflow.run_full().awaiting_action == ["send"]

    def test_never_mode_acts_immediately(self, make_workflow):
        workflow, tool = self._workflow(make_workflow, approval="never")
        report = workflow.run_full()
        assert report.awaiting_action == []
        assert tool.sent

    def test_reversible_tools_are_never_gated(self, make_workflow):
        steps = [Step(id="write", description="Write it.", agent_role="writer",
                      requires_tool="artifact_store")]
        workflow = make_workflow(steps, action_approval="all")
        assert workflow.run_full().awaiting_action == []

    def test_invalid_mode_is_rejected(self, make_workflow):
        with pytest.raises(ValueError, match="action_approval"):
            make_workflow([Step(id="a", description="d", agent_role="generic")],
                          action_approval="sometimes")

    def test_failed_action_marks_the_step_failed(self, make_workflow):
        from orchestrator.tools.base import ToolError

        class Exploding(LiveIrreversibleTool):
            name = "send_it"

            def _run(self, task, context=None):
                raise ToolError("the gateway rejected it")

        tools = ToolManager([Exploding()])
        steps = [Step(id="send", description="Send it.", agent_role="devops",
                      requires_tool="send_it", max_retries=0)]
        workflow = make_workflow(steps, tool_manager=tools)
        workflow.run_full()
        workflow.approve_action("send")
        report = workflow.resume()
        assert report.failed == ["send"]
        assert "gateway rejected" in workflow.results["send"].error

        # Approval is single-use even when the side effect fails. Resuming
        # must generate a new held action, never retry it automatically.
        second = workflow.resume()
        assert second.awaiting_action == ["send"]
        assert workflow.status_of("send") is StepStatus.AWAITING_ACTION

    def test_safe_preflight_failure_keeps_frozen_action_for_reapproval(self, make_workflow):
        from orchestrator.tools.base import ActionReviewRequiredError

        class NeedsReview(LiveIrreversibleTool):
            name = "send_it"

            def _run(self, task, context=None):
                raise ActionReviewRequiredError("draft no longer matches; nothing was sent")

        workflow = make_workflow(
            [Step(id="send", description="Send it.", agent_role="devops",
                  requires_tool="send_it", max_retries=0)],
            tool_manager=ToolManager([NeedsReview()]))
        workflow.run_full()
        original_payload = workflow.pending_actions()["send"].payload
        workflow.approve_action("send")

        report = workflow.resume()

        assert report.awaiting_action == ["send"]
        assert workflow.status_of("send") is StepStatus.AWAITING_ACTION
        assert workflow.pending_actions()["send"].payload == original_payload


class TestCombinedFlow:
    def test_input_then_action_then_done(self, make_workflow):
        """The whole n8n loop: ask, answer, hold, approve, act."""
        tool = LiveIrreversibleTool()
        steps = [
            Step(id="draft", description="Draft the notice.", agent_role="writer"),
            Step(id="send", description="Send the notice.", agent_role="devops",
                 requires_tool="send_it", depends_on=["draft"],
                 inputs=[InputRequest(name="to", prompt="To whom?", type=InputType.EMAIL)]),
        ]
        workflow = make_workflow(steps, tool_manager=ToolManager([tool]))

        first = workflow.run_full()
        assert first.awaiting_input == ["send"]

        workflow.provide_inputs({"to": "team@example.com"})
        second = workflow.resume()
        assert second.awaiting_action == ["send"]
        assert tool.sent == []

        workflow.approve_action("send")
        third = workflow.resume()
        assert "send" in third.executed
        assert workflow.status_of("send") is StepStatus.DONE

        # The answer must reach the tool itself, not merely the prompt --
        # otherwise routing would depend on the model echoing it back.
        assert tool.sent
        assert tool.sent[0]["context"]["inputs"]["to"] == "team@example.com"

    def test_input_values_are_passed_to_the_tool(self, make_workflow):
        tool = LiveIrreversibleTool()
        steps = [Step(id="send", description="Send it.", agent_role="devops",
                      requires_tool="send_it",
                      inputs=[InputRequest(name="channel", prompt="Where?")])]
        workflow = make_workflow(steps, tool_manager=ToolManager([tool]),
                                 action_approval="never")
        workflow.provide_inputs({"channel": "#ops"})
        workflow.run_full()
        assert tool.sent[0]["context"]["inputs"] == {"channel": "#ops"}

    def test_input_collected_on_one_step_reaches_a_later_step_s_tool(self, make_workflow):
        """Regression: planners collect a fact on one step and use it on another.
        Scoping answers to the asking step made those collections useless --
        the email went out with no recipient."""
        steps = [
            Step(id="collect", description="Get the details.", agent_role="research",
                 inputs=[InputRequest(name="recipient", prompt="To whom?",
                                      type=InputType.EMAIL)]),
            Step(id="send", description="Send it.", agent_role="writer",
                 requires_tool="email", depends_on=["collect"]),
        ]
        workflow = make_workflow(steps, action_approval="never")
        workflow.provide_inputs({"recipient": "ops@example.com"})
        workflow.run_full()
        assert "ops@example.com" in workflow.query_results("send")

    def test_a_step_s_own_answer_beats_a_workflow_wide_one(self, make_workflow):
        steps = [
            Step(id="collect", description="Get details.", agent_role="research",
                 inputs=[InputRequest(name="recipient", prompt="To whom?",
                                      type=InputType.EMAIL)]),
            Step(id="send", description="Send it.", agent_role="writer",
                 requires_tool="email", depends_on=["collect"],
                 inputs=[InputRequest(name="recipient", prompt="Override?",
                                      type=InputType.EMAIL)]),
        ]
        workflow = make_workflow(steps, action_approval="never")
        workflow.provide_input("collect", "recipient", "global@example.com")
        workflow.provide_input("send", "recipient", "local@example.com")
        workflow.run_full()
        assert "local@example.com" in workflow.query_results("send")

    def test_email_tool_routes_to_the_supplied_recipient(self, make_workflow):
        """The built-in email adapter must honour an operator-supplied address."""
        steps = [Step(id="send", description="Send it.", agent_role="devops",
                      requires_tool="email",
                      inputs=[InputRequest(name="recipient", prompt="To whom?",
                                           type=InputType.EMAIL)])]
        workflow = make_workflow(steps)
        workflow.provide_inputs({"recipient": "ops@example.com"})
        workflow.run_full()
        assert "ops@example.com" in workflow.query_results("send")
