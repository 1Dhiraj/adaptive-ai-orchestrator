"""The computer_use capability: backend choice, and the rules that bound it.

These tests care much more about what the tool *refuses* than about what it
does. Driving a real screen is the one capability here that can act outside
the workspace, so every guard is worth a test that would fail loudly if
someone relaxed it.
"""

from __future__ import annotations

import json

import pytest

from orchestrator.tools import ToolManager, canonical_tool_name
from orchestrator.tools.base import Tool, ToolError
from orchestrator.tools.computer_use import (
    ComputerUseTool,
    allowed_targets,
    check_secrets,
)


class FakeBackend(Tool):
    """Stands in for a Playwright MCP tool or Hermes: records what it ran."""

    def __init__(self, name: str = "web_browser_navigate", live: bool = True) -> None:
        super().__init__()
        self.name = name
        self.capability = "browser"
        self._live = live
        self.calls: list = []

    def is_live(self) -> bool:
        return self._live

    def _run(self, task, context=None):
        self.calls.append(task)
        return f"ok: {task}"


def directive(**arguments) -> str:
    return "do the thing\nTOOL_DIRECTIVE: " + json.dumps({"arguments": arguments})


@pytest.fixture
def allow_everything(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
    monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "example.com, notepad")


class TestSecretRefusal:
    @pytest.mark.parametrize("text", [
        "log in with password: hunter2",
        "pay with card 4111 1111 1111 1111",
        "use api_key=sk-abcdefghijklmnop",
        "the otp: 447291",
    ])
    def test_credentials_are_detected(self, text):
        assert check_secrets(text) is not None

    def test_ordinary_instructions_are_not_flagged(self):
        assert check_secrets("open example.com and read the first heading") is None

    def test_execution_refuses_before_acting(self, allow_everything):
        backend = FakeBackend()
        tools = ToolManager([backend])
        tool = ComputerUseTool(run_id="secret-test", tools=tools)
        with pytest.raises(ToolError, match="password"):
            tool.execute("log in with password: hunter2\n" + directive(
                goal="log in", targets=["example.com"], actions=["type the password"]))
        # The point of refusing early: nothing reached the backend.
        assert backend.calls == []


class TestAllowList:
    def test_unset_allow_list_permits_nothing(self, monkeypatch):
        monkeypatch.delenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", raising=False)
        assert allowed_targets() == []

    def test_target_outside_the_list_is_refused(self, allow_everything):
        tools = ToolManager([FakeBackend()])
        tool = ComputerUseTool(run_id="allow-test", tools=tools)
        with pytest.raises(ToolError, match="allow-list"):
            tool.execute(directive(goal="g", targets=["evil.test"], actions=["open it"]))

    def test_subdomains_are_covered_but_lookalikes_are_not(self, allow_everything):
        tools = ToolManager([FakeBackend()])
        tool = ComputerUseTool(run_id="allow-test2", tools=tools)
        tool.execute(directive(goal="g", targets=["www.example.com"], actions=["open"]))
        with pytest.raises(ToolError, match="allow-list"):
            tool.execute(directive(goal="g", targets=["notexample.com"], actions=["open"]))

    def test_naming_no_target_is_refused(self, allow_everything):
        tools = ToolManager([FakeBackend()])
        tool = ComputerUseTool(run_id="allow-test3", tools=tools)
        with pytest.raises(ToolError, match="targets"):
            tool.execute(directive(goal="g", actions=["open something"]))


class TestBackendSelection:
    def test_browser_is_preferred_over_desktop(self):
        browser = FakeBackend("web_browser_navigate")
        desktop = FakeBackend("hermes_desktop")
        tool = ComputerUseTool(run_id="pick", tools=ToolManager([browser, desktop]))
        assert tool.backend() == "web_browser_navigate"

    def test_falls_back_to_desktop_when_no_browser(self):
        tool = ComputerUseTool(run_id="pick2",
                               tools=ToolManager([FakeBackend("hermes_desktop")]))
        assert tool.backend() == "hermes_desktop"

    def test_no_backend_reports_none(self):
        assert ComputerUseTool(run_id="pick3", tools=ToolManager([])).backend() == "none"

    def test_a_broken_backend_is_not_selected(self):
        browser = FakeBackend("web_browser_navigate")
        browser.break_it()
        tool = ComputerUseTool(run_id="pick4", tools=ToolManager([browser]))
        assert tool.backend() == "none"


class TestOptIn:
    def test_disabled_without_the_env_flag(self, monkeypatch):
        monkeypatch.delenv("ORCHESTRATOR_ALLOW_DESKTOP", raising=False)
        tool = ComputerUseTool(run_id="off", tools=ToolManager([FakeBackend()]))
        assert tool.is_live() is False

    def test_simulates_instead_of_acting_when_disabled(self, monkeypatch):
        monkeypatch.delenv("ORCHESTRATOR_ALLOW_DESKTOP", raising=False)
        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "example.com")
        backend = FakeBackend()
        tool = ComputerUseTool(run_id="off2", tools=ToolManager([backend]))
        out = tool.execute(directive(goal="g", targets=["example.com"], actions=["open"]))
        assert "[simulated:computer_use]" in out
        assert backend.calls == []

    def test_is_irreversible_so_approval_always_runs_first(self):
        assert ComputerUseTool().irreversible is True


class TestLimitsAndLogging:
    """These write a log, so they get a temp workspace root.

    Without it the log lands in the real workspace and survives between test
    sessions, and the next run appends to it instead of starting clean.
    """

    def test_action_limit_stops_the_run(self, allow_everything, tmp_path):
        backend = FakeBackend()
        tool = ComputerUseTool(run_id="limits", tools=ToolManager([backend]),
                               root=tmp_path)
        out = tool.execute(directive(goal="g", targets=["example.com"],
                                     actions=[f"step {i}" for i in range(10)],
                                     max_steps=3))
        assert len(backend.calls) == 3
        assert "3-action limit" in out

    def test_actions_are_written_to_a_log(self, allow_everything, tmp_path):
        tool = ComputerUseTool(run_id="logged", tools=ToolManager([FakeBackend()]),
                               root=tmp_path)
        tool.execute(directive(goal="g", targets=["example.com"], actions=["open"]))
        log = tool.workspace / "computer_use.log.jsonl"
        assert log.exists()
        events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert [e["event"] for e in events] == ["start", "action", "finish"]


class TestRegistration:
    def test_aliases_route_to_the_capability_not_the_desktop_backend(self):
        assert canonical_tool_name("computer-use") == "computer_use"
        assert canonical_tool_name("screen") == "computer_use"

    def test_it_is_in_the_default_catalogue(self):
        from orchestrator.tools import default_tool_manager

        assert default_tool_manager().get("computer_use") is not None
