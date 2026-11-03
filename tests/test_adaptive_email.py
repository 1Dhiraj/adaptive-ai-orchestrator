"""Browser-first email delivery without touching a real mailbox."""

from __future__ import annotations

import json

import pytest

from orchestrator.tools.adaptive_email import EmailDelivery, email_message, message_digest
from orchestrator.tools.base import ActionReviewRequiredError, Tool, ToolError


class FakeBrowserCodeTool(Tool):
    name = "web_browser_run_code_unsafe"

    def __init__(self, statuses: list[str]):
        super().__init__()
        self.statuses = iter(statuses)
        self.payloads: list[str] = []

    def is_live(self) -> bool:
        return True

    def _run(self, task: str, context: dict | None = None) -> str:
        assert '"code"' in task
        self.payloads.append(task)
        return "### Result\n" + json.dumps({"status": next(self.statuses)})


@pytest.fixture
def message() -> dict:
    return {"to": "friend@example.com", "subject": "Hello", "body": "Hi there"}


def test_browser_flow_opens_prepares_and_sends_exact_draft(tmp_path, monkeypatch, message):
    monkeypatch.setattr(EmailDelivery, "api_tool", staticmethod(lambda: None))
    browser = FakeBrowserCodeTool(["sign_in_needed", "draft_ready", "sent"])
    delivery = EmailDelivery(tmp_path, browser)

    assert delivery.open_browser("run", "mail", message, approved=True)["status"] == "sign_in_needed"
    assert delivery.prepare_browser("run", "mail", message)["status"] == "draft_ready"
    assert "via Gmail browser" in delivery.send("run", "mail", message)
    assert delivery.status("run", "mail", message)["status"] == "sent"
    assert browser.call_count == 3
    verifier = browser.payloads[-1]
    assert 'input[name=\\"cc\\"],input[name=\\"bcc\\"]' in verifier
    assert 'input[name=\\"bcc\\"],input[name=\\"to\\"]' not in verifier
    assert "invisible bidi characters" in verifier
    assert "expected one Send button" in verifier


def test_changed_browser_draft_returns_to_review_without_sending(tmp_path, monkeypatch, message):
    monkeypatch.setattr(EmailDelivery, "api_tool", staticmethod(lambda: None))
    browser = FakeBrowserCodeTool(["sign_in_needed", "draft_ready", "draft_changed"])
    delivery = EmailDelivery(tmp_path, browser)
    delivery.open_browser("run", "mail", message, approved=True)
    delivery.prepare_browser("run", "mail", message)

    with pytest.raises(ActionReviewRequiredError, match="Nothing was sent"):
        delivery.send("run", "mail", message)

    assert delivery.status("run", "mail", message)["status"] == "draft_ready"


def test_connected_browser_must_be_prepared_instead_of_silent_api_fallback(
        tmp_path, monkeypatch, message):
    class LiveApi(Tool):
        name = "api"

        def is_live(self) -> bool:
            return True

        def _run(self, task: str, context: dict | None = None) -> str:
            raise AssertionError("browser-first delivery must not call the API")

    monkeypatch.setattr(EmailDelivery, "api_tool", staticmethod(lambda: LiveApi()))
    delivery = EmailDelivery(tmp_path, FakeBrowserCodeTool([]))

    with pytest.raises(ToolError, match="Open Gmail"):
        delivery.send("run", "mail", message)


def test_message_uses_last_nested_directive_and_strips_internal_metadata():
    payload = '''ASSUMPTIONS: recipient, email_subject, email_body
Good afternoon da
TOOL_DIRECTIVE: {"to":"old@example.com","subject":"Old"}
TOOL_DIRECTIVE: {"arguments":{"to":"friend@example.com","subject":"Greetings","body":"Good afternoon da"}}'''

    assert email_message(payload) == {
        "to": "friend@example.com",
        "subject": "Greetings",
        "body": "Good afternoon da",
    }


def test_message_never_copies_thinking_trace_into_body():
    payload = '''We need to output email body, then TOOL_DIRECTIVE with arguments.
The whole output should be TOOL_DIRECTIVE: {"arguments":{"to":"friend@example.com"}}
</think>

Good afternoon da
TOOL_DIRECTIVE: {"arguments":{"to":"friend@example.com","subject":"Greetings"}}'''

    assert email_message(payload) == {
        "to": "friend@example.com",
        "subject": "Greetings",
        "body": "Good afternoon da",
    }


def test_message_prefers_body_inside_directive_over_surrounding_text():
    payload = '''This text must not become the email body.
TOOL_DIRECTIVE: {"arguments":{"to":"friend@example.com","subject":"Greetings","body":"Good afternoon da"}}'''

    assert email_message(payload) == {
        "to": "friend@example.com",
        "subject": "Greetings",
        "body": "Good afternoon da",
    }


def test_changed_message_invalidates_an_old_reviewed_draft(tmp_path, monkeypatch, message):
    monkeypatch.setattr(EmailDelivery, "api_tool", staticmethod(lambda: None))
    delivery = EmailDelivery(tmp_path, FakeBrowserCodeTool([]))
    delivery.write("run", "mail", {
        "status": "draft_ready", "method": "browser", "digest": message_digest(message),
    })

    changed = {**message, "subject": "Changed after review"}
    status = delivery.status("run", "mail", changed)

    assert status["stale_draft"] is True
    assert status["status"] == "choose_method"
    assert status["reviewed_digest"] != status["digest"]
