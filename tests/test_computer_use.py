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

    def __init__(self, name: str = "web_browser_navigate", live: bool = True,
                 response: str | None = None) -> None:
        super().__init__()
        self.name = name
        self.capability = "browser"
        self._live = live
        self.response = response
        self.calls: list = []

    def is_live(self) -> bool:
        return self._live

    def _run(self, task, context=None):
        self.calls.append(task)
        return self.response if self.response is not None else f"ok: {task}"


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
        # Hermes accepts outcome-like natural-language actions. Browser and
        # native backends now compile a small typed action language instead.
        backend = FakeBackend("hermes_desktop")
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

    def test_descriptive_open_action_is_compiled_to_an_approved_url(
            self, allow_everything, tmp_path):
        backend = FakeBackend()
        tool = ComputerUseTool(run_id="browser-url", tools=ToolManager([backend]),
                               root=tmp_path)

        tool.execute(directive(goal="read it", targets=["example.com"],
                               actions=["open the page", "read the heading"]))

        assert len(backend.calls) == 1
        assert json.loads(backend.calls[0].removeprefix("TOOL_DIRECTIVE: ")) == {
            "arguments": {"url": "https://example.com"}}

    def test_descriptive_read_appends_a_browser_snapshot(
            self, allow_everything, tmp_path):
        navigate = FakeBackend()
        snapshot = FakeBackend("web_browser_snapshot", response="""
        - link "Skip to main content" [ref=a]
        - /url: https://support.google.com/websearch/answer/1
        - heading [level=3] [ref=b]: Faster feedback loops
        - text: Agents can automate repetitive engineering work.
        - /url: https://example.com/research
        """)
        tool = ComputerUseTool(
            run_id="browser-snapshot", tools=ToolManager([navigate, snapshot]),
            root=tmp_path)

        output = tool.execute(directive(
            goal="read it", targets=["example.com"],
            actions=["open the page", "read the heading"]))

        assert snapshot.calls == [""]
        assert "page highlights" in output
        assert "Faster feedback loops" in output
        assert "https://example.com/research" in output
        assert "Skip to main content" not in output
        assert "support.google.com" not in output

    def test_snapshot_prioritises_source_urls_before_long_page_text(
            self, allow_everything, tmp_path):
        navigate = FakeBackend()
        snapshot = FakeBackend("web_browser_snapshot", response=(
            "Page Title: Search results\n"
            + "\n".join(f"- text: overview filler {i} " + "x" * 90 for i in range(20))
            + "\n- /url: https://example.com/direct-source\n"
            + "- heading [level=3]: Direct source title\n"))
        tool = ComputerUseTool(
            run_id="browser-source-priority",
            tools=ToolManager([navigate, snapshot]), root=tmp_path)

        output = tool.execute(directive(
            goal="read it", targets=["example.com"],
            actions=["open the page", "read the results"]))

        assert "https://example.com/direct-source" in output

    def test_google_query_is_compiled_to_a_search_url(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "google.com")
        backend = FakeBackend()
        tool = ComputerUseTool(run_id="browser-search", tools=ToolManager([backend]),
                               root=tmp_path)

        tool.execute(directive(
            goal="research", targets=["google.com"],
            actions=["open the page", "enter query 'AI agents 2024', submit"]))

        arguments = json.loads(
            backend.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]
        assert arguments["url"] == "https://google.com/search?q=AI+agents+2024"

    def test_google_query_can_be_recovered_from_the_goal(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "google.com")
        backend = FakeBackend()
        tool = ComputerUseTool(run_id="browser-goal-search",
                               tools=ToolManager([backend]), root=tmp_path)

        tool.execute(directive(
            goal="Search Google for recent articles on AI coding agents; collect sources",
            targets=["google.com"], actions=["open the page"]))

        arguments = json.loads(
            backend.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]
        assert arguments["url"] == (
            "https://google.com/search?q=recent+articles+on+AI+coding+agents")

    def test_search_targets_do_not_replace_results_with_a_bare_site(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "google.com,github.com")
        navigate = FakeBackend()
        snapshot = FakeBackend("web_browser_snapshot", response="""
        Page Title: AI agents - Google Search
        - heading [level=3]: Survey of AI coding agents
        - /url: https://github.com/example/agent-survey
        - text: Benefits include automation; limitations include verification.
        """)
        tool = ComputerUseTool(
            run_id="browser-search-evidence",
            tools=ToolManager([navigate, snapshot]), root=tmp_path)

        output = tool.execute(directive(
            goal="research", targets=["google.com", "github.com"],
            actions=["open https://www.google.com", "enter query 'AI agents 2024'",
                     "extract github.com result URLs"]))

        assert len(navigate.calls) == 1
        assert "google.com/search?q=AI+agents+2024" in navigate.calls[0]
        assert snapshot.calls == [""]
        assert "https://github.com/example/agent-survey" in output

    def test_explicit_url_is_used_only_on_the_approved_target(
            self, allow_everything, tmp_path):
        backend = FakeBackend()
        tool = ComputerUseTool(run_id="browser-explicit", tools=ToolManager([backend]),
                               root=tmp_path)

        tool.execute(directive(
            goal="read it", targets=["example.com"],
            actions=["open https://www.example.com/report?q=ai", "read the heading"]))

        arguments = json.loads(
            backend.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]
        assert arguments["url"] == "https://www.example.com/report?q=ai"

        backend.calls.clear()
        tool.execute(directive(
            goal="stay bounded", targets=["example.com"],
            actions=["open https://evil.test/steal", "read the heading"]))
        arguments = json.loads(
            backend.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]
        assert arguments["url"] == "https://example.com"

    def test_named_browser_click_uses_snapshot_ref(self, allow_everything, tmp_path):
        navigate = FakeBackend()
        snapshot = FakeBackend("web_browser_snapshot", response='''
        Page Title: Example
        - button "Continue" [ref=e12]
        - text: Ready
        ''')
        click = FakeBackend("web_browser_click")
        tool = ComputerUseTool(
            run_id="browser-click",
            tools=ToolManager([navigate, snapshot, click]), root=tmp_path)

        output = tool.execute(directive(
            goal="continue", targets=["example.com"],
            actions=["open the page", "click Continue button", "read the result"]))

        arguments = json.loads(
            click.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]
        assert arguments == {"element": "Continue", "ref": "e12"}
        assert "click Continue" in output

    def test_browser_typing_uses_named_field_ref(self, allow_everything, tmp_path):
        navigate = FakeBackend()
        snapshot = FakeBackend("web_browser_snapshot", response='''
        - textbox "Search" [ref=q7]
        ''')
        type_tool = FakeBackend("web_browser_type")
        tool = ComputerUseTool(
            run_id="browser-type",
            tools=ToolManager([navigate, snapshot, type_tool]), root=tmp_path)

        tool.execute(directive(
            goal="fill search", targets=["example.com"],
            actions=["open the page", "type 'AI agents' into Search"]))

        arguments = json.loads(
            type_tool.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]
        assert arguments == {"element": "Search", "ref": "q7", "text": "AI agents"}

    def test_native_desktop_gets_typed_directives_not_natural_language(
            self, allow_everything, tmp_path):
        desktop = FakeBackend("desktop_native")
        tool = ComputerUseTool(
            run_id="native-actions", tools=ToolManager([desktop]), root=tmp_path)

        output = tool.execute(directive(
            goal="write a note", targets=["notepad"],
            actions=["launch the app", "type 'hello judges'", "save file"]))

        specs = [json.loads(call.removeprefix("TOOL_DIRECTIVE: "))["arguments"]
                 for call in desktop.calls]
        assert specs == [
            {"action": "launch", "app": "notepad"},
            {"action": "type", "text": "hello judges", "window": "notepad"},
            {"action": "key", "keys": "ctrl+s", "window": "notepad"},
        ]
        assert "3 action(s)" in output


class TestRegistration:
    def test_aliases_route_to_the_capability_not_the_desktop_backend(self):
        assert canonical_tool_name("computer-use") == "computer_use"
        assert canonical_tool_name("screen") == "computer_use"

    def test_it_is_in_the_default_catalogue(self):
        from orchestrator.tools import default_tool_manager

        assert default_tool_manager().get("computer_use") is not None
