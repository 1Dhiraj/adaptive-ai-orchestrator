"""The computer_use capability: backend choice, and the rules that bound it.

These tests care much more about what the tool *refuses* than about what it
does. Driving a real screen is the one capability here that can act outside
the workspace, so every guard is worth a test that would fail loudly if
someone relaxed it.
"""

from __future__ import annotations

import json
import os

import pytest

from orchestrator.tools import ToolManager, canonical_tool_name
from orchestrator.tools.base import Tool, ToolError
from orchestrator.tools.computer_use import (
    ComputerUseTool,
    allowed_targets,
    check_secrets,
)
from orchestrator.tools.desktop import DesktopTool


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


def test_localhost_goal_retains_server_port(monkeypatch):
    from orchestrator.tools.computer_use import _parse_plan, _target_allowed

    monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "127.0.0.1")
    task = directive(goal="Open http://127.0.0.1:4173/ and check Buy milk after reload",
                     targets=["127.0.0.1"], autonomous=True)
    plan = _parse_plan(task)
    assert plan.targets == ["http://127.0.0.1:4173/"]
    assert _target_allowed(plan.targets[0], allowed_targets())


def test_explicit_article_url_precedes_wikipedia_home(monkeypatch):
    from orchestrator.tools.computer_use import _parse_plan

    monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "wikipedia.org")
    plan = _parse_plan(directive(
        goal="Open https://en.wikipedia.org/wiki/Moon and read the article",
        targets=["wikipedia.org", "en.wikipedia.org"], autonomous=True))
    assert plan.targets[0] == "https://en.wikipedia.org/wiki/Moon"


def test_dead_localhost_server_uses_run_html(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
    monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "127.0.0.1")
    browser = FakeBackend()
    tool = ComputerUseTool(run_id="static-site", tools=ToolManager([browser]), root=tmp_path)
    (tool.workspace / "index.html").write_text("<h1>Demo</h1>", encoding="utf-8")
    tool.execute(directive(goal="Open http://127.0.0.1:1/ and read the site",
                           targets=["127.0.0.1"], actions=["read the page"]))
    url = json.loads(browser.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]["url"]
    assert url.startswith("http://127.0.0.1:")
    assert url.endswith("/index.html")


def test_todo_reload_requires_two_observed_assertions(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
    monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "127.0.0.1")
    navigate = FakeBackend()
    evaluate = FakeBackend(name="web_browser_evaluate", response="### Result\ntrue")
    tool = ComputerUseTool(run_id="todo-check", tools=ToolManager([navigate, evaluate]),
                           root=tmp_path)
    (tool.workspace / "index.html").write_text("<form></form>", encoding="utf-8")
    result = tool.execute(directive(
        goal="Open http://127.0.0.1:1/ and add task Buy milk, then reload and verify",
        targets=["127.0.0.1"], autonomous=True))
    assert "PASS" in result
    assert len(navigate.calls) == 2
    assert len(evaluate.calls) == 2


def test_calculator_requires_observed_twelve(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
    monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "127.0.0.1")
    navigate = FakeBackend()
    evaluate = FakeBackend(name="web_browser_evaluate", response="### Result\nfalse")
    tool = ComputerUseTool(run_id="calculator-check",
                           tools=ToolManager([navigate, evaluate]), root=tmp_path)
    (tool.workspace / "calculator.html").write_text("<output id='display'>0</output>",
                                                   encoding="utf-8")
    with pytest.raises(ToolError, match="did not show 12"):
        tool.execute(directive(goal="Open the calculator and check 7 + 5 = 12",
                               targets=["calculator.html"], autonomous=True))


def test_moon_facts_require_article_and_source_extract(monkeypatch, tmp_path):
    from orchestrator.tools.builtin import RestApiTool

    monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
    monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "wikipedia.org")
    source = ("only natural satellite of Earth; every 29.5 days; diameter is 3,474 km; "
              "main driver of Earth's tides; Apollo 11 on July 20, 1969")
    monkeypatch.setattr(RestApiTool, "execute", lambda self, *args, **kwargs: source)
    navigate = FakeBackend(response="### Page\n- Page URL: https://en.wikipedia.org/wiki/Moon")
    tool = ComputerUseTool(run_id="moon-check", tools=ToolManager([navigate]), root=tmp_path)
    result = tool.execute(directive(
        goal="Open https://en.wikipedia.org/wiki/Moon and give five facts",
        targets=["wikipedia.org"], autonomous=True))
    assert "5. Apollo 11" in result


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

    def test_own_native_desktop_is_preferred_when_both_are_available(self):
        tool = ComputerUseTool(
            run_id="pick-native",
            tools=ToolManager([FakeBackend("hermes_desktop"),
                               FakeBackend("desktop_native")]))
        assert tool.backend(["notepad"]) == "desktop_native"

    def test_native_app_target_uses_desktop_even_when_browser_is_connected(self):
        browser = FakeBackend("web_browser_navigate")
        desktop = FakeBackend("desktop_native")
        tool = ComputerUseTool(run_id="pick-app", tools=ToolManager([browser, desktop]))

        assert tool.backend(["notepad"]) == "desktop_native"

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

    def test_browser_output_keeps_page_paragraphs(
            self, allow_everything, tmp_path):
        navigate = FakeBackend()
        snapshot = FakeBackend("web_browser_snapshot", response='''
        Page Title: Example Domain
        - heading "Example Domain" [level=1]
        - paragraph: This domain is for use in illustrative examples.
        ''')
        tool = ComputerUseTool(
            run_id="browser-paragraph", tools=ToolManager([navigate, snapshot]),
            root=tmp_path)

        output = tool.execute(directive(
            goal="read", targets=["example.com"],
            actions=["open the page", "read the page"]))

        assert "This domain is for use in illustrative examples." in output

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

    def test_desktop_fallback_treats_exactly_as_an_instruction_not_text(
            self, allow_everything, tmp_path):
        desktop = FakeBackend("desktop_native")
        tool = ComputerUseTool(
            run_id="desktop-exact-text", tools=ToolManager([desktop]), root=tmp_path)

        tool.execute("model omitted directive", {
            "step_description": (
                "Open Windows Notepad and type exactly REVIEW DESKTOP PASSED. "
                "Verify the text is present."
            )
        })

        calls = [json.loads(call.removeprefix("TOOL_DIRECTIVE: "))["arguments"]
                 for call in desktop.calls]
        assert {"action": "type", "text": "REVIEW DESKTOP PASSED", "window": "notepad"} in calls

    def test_search_field_typing_automatically_submits_with_enter(
            self, allow_everything, tmp_path):
        navigate = FakeBackend()
        snapshot = FakeBackend("web_browser_snapshot", response='''
        - textbox "Search" [ref=q7]
        ''')
        type_tool = FakeBackend("web_browser_type")
        key_tool = FakeBackend("web_browser_press_key")
        tool = ComputerUseTool(
            run_id="browser-search-submit",
            tools=ToolManager([navigate, snapshot, type_tool, key_tool]), root=tmp_path)

        tool.execute(directive(
            goal="search", targets=["example.com"],
            actions=["open the page", "type 'AI agents' into Search"]))

        key_args = json.loads(
            key_tool.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]
        assert key_args == {"key": "Enter"}

    def test_browser_select_and_hover_use_accessible_refs(
            self, allow_everything, tmp_path):
        navigate = FakeBackend()
        snapshot = FakeBackend("web_browser_snapshot", response='''
        - combobox "Country" [ref=c4]
        - button "Help" [ref=h2]
        ''')
        select = FakeBackend("web_browser_select_option")
        hover = FakeBackend("web_browser_hover")
        tool = ComputerUseTool(
            run_id="browser-rich-actions",
            tools=ToolManager([navigate, snapshot, select, hover]), root=tmp_path)

        tool.execute(directive(
            goal="configure form", targets=["example.com"],
            actions=["open the page", "select 'India' from Country", "hover Help"]))

        select_args = json.loads(
            select.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]
        hover_args = json.loads(
            hover.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]
        assert select_args == {"element": "Country", "ref": "c4", "values": ["India"]}
        assert hover_args == {"element": "Help", "ref": "h2"}

    def test_browser_wait_screenshot_and_back_are_typed_calls(
            self, allow_everything, tmp_path):
        navigate = FakeBackend()
        snapshot = FakeBackend("web_browser_snapshot", response="Page Title: Example")
        wait = FakeBackend("web_browser_wait_for")
        screenshot = FakeBackend("web_browser_take_screenshot")
        back = FakeBackend("web_browser_navigate_back")
        tool = ComputerUseTool(
            run_id="browser-navigation-actions",
            tools=ToolManager([navigate, snapshot, wait, screenshot, back]), root=tmp_path)

        tool.execute(directive(
            goal="inspect", targets=["example.com"],
            actions=["open the page", "wait 2 seconds", "take screenshot", "go back"]))

        assert json.loads(wait.calls[0].removeprefix("TOOL_DIRECTIVE: ")) == {
            "arguments": {"time": 2.0}}
        assert json.loads(screenshot.calls[0].removeprefix("TOOL_DIRECTIVE: ")) == {
            "arguments": {"type": "png", "scale": "css"}}
        assert json.loads(back.calls[0].removeprefix("TOOL_DIRECTIVE: ")) == {
            "arguments": {}}

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

    def test_native_named_click_becomes_guarded_vision_click(
            self, allow_everything, tmp_path):
        desktop = FakeBackend("desktop_native")
        tool = ComputerUseTool(
            run_id="native-named-click", tools=ToolManager([desktop]), root=tmp_path)

        tool.execute(directive(
            goal="use the editor", targets=["notepad"],
            actions=["click Save button"]))

        spec = json.loads(
            desktop.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]
        assert spec == {"action": "click_target", "target": "Save button",
                        "button": "left", "window": "notepad"}

    def test_missing_model_directive_uses_bounded_step_description_plan(
            self, allow_everything, tmp_path):
        desktop = FakeBackend("desktop_native")
        tool = ComputerUseTool(
            run_id="native-fallback", tools=ToolManager([
                FakeBackend("web_browser_navigate"), desktop,
            ]), root=tmp_path)

        output = tool.execute(
            "The model discussed the task but omitted its directive.",
            {"step_description": "Open Notepad and type 'hello' into it."})

        specs = [json.loads(call.removeprefix("TOOL_DIRECTIVE: "))["arguments"]
                 for call in desktop.calls]
        assert specs == [
            {"action": "launch", "app": "notepad"},
            {"action": "type", "text": "hello", "window": "notepad"},
        ]
        assert "2 action(s)" in output

    def test_missing_directive_preview_shows_the_exact_recovery_plan(
            self, allow_everything, tmp_path):
        tool = ComputerUseTool(
            run_id="native-preview", tools=ToolManager([
                FakeBackend("web_browser_navigate"), FakeBackend("desktop_native"),
            ]), root=tmp_path)

        preview = tool.preview(
            "The model stopped before its directive.",
            {"step_description": "Open Notepad and type 'hello' into it."})

        assert "backend: desktop_native" in preview
        assert "touches: notepad" in preview
        assert "1. launch the app" in preview
        assert '2. type "hello"' in preview

    def test_invalid_model_target_is_recovered_from_step_description(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "google.com")
        navigate = FakeBackend()
        tool = ComputerUseTool(
            run_id="browser-recovery", tools=ToolManager([navigate]), root=tmp_path)

        tool.execute(
            directive(goal="research", targets=["open the page"], actions=["open the page"]),
            {"step_description": "Research AI agents on the web"})

        arguments = json.loads(
            navigate.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]
        assert arguments["url"] == "https://google.com"

    def test_url_targets_are_reduced_to_their_approved_hostname(
            self, allow_everything, tmp_path):
        navigate = FakeBackend()
        tool = ComputerUseTool(
            run_id="browser-target-normalise", tools=ToolManager([navigate]), root=tmp_path)

        tool.execute(directive(
            goal="read", targets=["https://example.com/private?q=1"],
            actions=["open the page"]))

        arguments = json.loads(
            navigate.calls[0].removeprefix("TOOL_DIRECTIVE: "))["arguments"]
        assert arguments["url"] == "https://example.com"

    def test_email_intent_cannot_succeed_by_only_opening_gmail(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
        monkeypatch.setenv(
            "ORCHESTRATOR_COMPUTER_USE_ALLOW", "mail.google.com,google.com")
        navigate = FakeBackend()
        tool = ComputerUseTool(
            run_id="email-no-false-success", tools=ToolManager([navigate]), root=tmp_path)

        with pytest.raises(ToolError, match="adaptive_email"):
            tool.execute(
                "The model omitted its directive.",
                {"step_description": "Open browser and send mail saying hello"})
        assert navigate.calls == []

    def test_raw_desktop_tool_rejects_a_missing_action(self, tmp_path):
        tool = DesktopTool(run_id="no-op", root=tmp_path)

        with pytest.raises(ToolError, match="nothing was done"):
            tool.execute("The model forgot the TOOL_DIRECTIVE")

    def test_desktop_focus_keeps_the_matching_foreground_instance(
            self, monkeypatch, tmp_path):
        import orchestrator.tools.desktop as desktop_module

        monkeypatch.setattr(desktop_module, "foreground_window",
                            lambda: "Untitled - Notepad")
        monkeypatch.setattr(desktop_module, "foreground_process",
                            lambda: "notepad.exe")
        monkeypatch.setattr(
            desktop_module, "list_windows",
            lambda: pytest.fail("must not select an older same-named window"))

        result = DesktopTool(run_id="focus-new-window", root=tmp_path)._focus(
            "notepad", ["notepad"])

        assert result == "focused 'Untitled - Notepad'"

    def test_desktop_focus_matches_executable_when_title_differs(
            self, monkeypatch, tmp_path):
        import orchestrator.tools.desktop as desktop_module

        front = ["Chrome", "chrome.exe"]
        monkeypatch.setattr(desktop_module, "foreground_window", lambda: front[0])
        monkeypatch.setattr(desktop_module, "foreground_process", lambda: front[1])
        monkeypatch.setattr(desktop_module, "list_windows", lambda: [
            {"handle": 7, "title": "Untitled - Paint"}])
        monkeypatch.setattr(desktop_module, "window_process", lambda handle: "mspaint.exe")
        class FakeGui:
            def IsIconic(self, handle): return False
            def SetForegroundWindow(self, handle): front[:] = ["Untitled - Paint", "mspaint.exe"]
        class FakeApi:
            def keybd_event(self, *args): pass
        class FakeCon:
            SW_RESTORE = 9
        monkeypatch.setattr(desktop_module, "_win32",
                            lambda: (FakeApi(), FakeCon(), FakeGui(), None))
        monkeypatch.setattr(desktop_module.time, "sleep", lambda seconds: None)

        tool = DesktopTool(run_id="focus-paint", root=tmp_path)
        assert tool._focus("mspaint", ["mspaint"]) == "focused 'Untitled - Paint'"
        assert "mspaint.exe" in tool._ensure_target("mspaint", ["mspaint"])

    def test_desktop_type_passes_exact_text_through_the_clipboard(
            self, monkeypatch, tmp_path):
        # Pasting, not synthesised keystrokes: see DesktopTool._type for why
        # injected typing corrupts text rather than merely dropping it.
        import orchestrator.tools.desktop as desktop_module

        monkeypatch.setattr(desktop_module, "_win32", lambda: (None, None, None, None))
        monkeypatch.setattr(desktop_module.time, "sleep", lambda _seconds: None)
        clipboard = ["whatever the person had copied"]
        monkeypatch.setattr(desktop_module, "_clipboard_get", lambda: clipboard[-1])
        monkeypatch.setattr(desktop_module, "_clipboard_set",
                            lambda value: clipboard.append(value))
        tool = DesktopTool(run_id="unicode-type", root=tmp_path)
        monkeypatch.setattr(tool, "_ensure_target",
                            lambda window, allow: "Untitled - Notepad [notepad.exe]")
        keys = []
        monkeypatch.setattr(tool, "_key",
                            lambda combo, allow, window="": keys.append(combo))
        monkeypatch.setattr(tool, "_send_unicode",
                            lambda _text: pytest.fail("should have pasted"))

        result = tool._type("MiXeD exact ✓", ["notepad"], "notepad")

        assert "MiXeD exact ✓" in clipboard
        assert keys == ["ctrl+v"]
        assert "13 character(s)" in result
        # Borrowed, not taken: the person gets their clipboard back.
        assert clipboard[-1] == "whatever the person had copied"

    def test_desktop_type_can_be_forced_to_use_keystrokes(
            self, monkeypatch, tmp_path):
        import orchestrator.tools.desktop as desktop_module

        monkeypatch.setattr(desktop_module, "_win32", lambda: (None, None, None, None))
        monkeypatch.setattr(desktop_module.time, "sleep", lambda _seconds: None)
        tool = DesktopTool(run_id="forced-keystrokes", root=tmp_path)
        monkeypatch.setattr(tool, "_ensure_target",
                            lambda window, allow: "Untitled - Notepad")
        captured = []
        monkeypatch.setattr(tool, "_send_unicode", captured.append)

        result = tool._type("exact", ["notepad"], "notepad", method="keystrokes")

        assert captured == ["exact"]
        # The caller is told this path is not verified, because it is not.
        assert "unverified" in result

    def test_desktop_type_falls_back_when_the_clipboard_cannot_be_reached(
            self, monkeypatch, tmp_path):
        import orchestrator.tools.desktop as desktop_module

        def no_clipboard(*_args):
            raise OSError("clipboard is held by another process")

        monkeypatch.setattr(desktop_module, "_win32", lambda: (None, None, None, None))
        monkeypatch.setattr(desktop_module.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(desktop_module, "_clipboard_get", no_clipboard)
        monkeypatch.setattr(desktop_module, "_clipboard_set", no_clipboard)
        tool = DesktopTool(run_id="clipboard-busy", root=tmp_path)
        monkeypatch.setattr(tool, "_ensure_target", lambda window, allow: "Notepad")
        captured = []
        monkeypatch.setattr(tool, "_send_unicode", captured.append)

        result = tool._type("still gets typed", ["notepad"], "notepad")

        assert captured == ["still gets typed"]
        assert "unverified" in result

    def test_desktop_type_rejects_an_unknown_method(self, monkeypatch, tmp_path):
        import orchestrator.tools.desktop as desktop_module

        monkeypatch.setattr(desktop_module, "_win32", lambda: (None, None, None, None))
        tool = DesktopTool(run_id="bad-method", root=tmp_path)
        monkeypatch.setattr(tool, "_ensure_target", lambda window, allow: "Notepad")

        with pytest.raises(ToolError, match="unknown typing method"):
            tool._type("text", ["notepad"], "notepad", method="telepathy")


class TestRegistration:
    def test_aliases_route_to_the_capability_not_the_desktop_backend(self):
        assert canonical_tool_name("computer-use") == "computer_use"
        assert canonical_tool_name("screen") == "computer_use"

    def test_it_is_in_the_default_catalogue(self):
        from orchestrator.tools import default_tool_manager

        assert default_tool_manager().get("computer_use") is not None

    def test_default_catalogue_binds_computer_use_to_its_backends(self):
        from orchestrator.tools import default_tool_manager

        manager = default_tool_manager()
        assert manager.get("computer_use").tools is manager


class TestAnySiteWildcard:
    """`*` opens the web, and deliberately not the desktop.

    A fixed list cannot answer an open question -- "find me the cheapest
    flight" names no site, so a list can only refuse it. But the two
    permissions are not comparable: opening an arbitrary URL is what a browser
    does all day inside a sandboxed tab, while driving an arbitrary desktop
    application puts synthetic clicks and keystrokes into whatever is in
    front, which may be a mail client or an open bank tab.
    """

    @pytest.fixture
    def anything(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "*, notepad")

    @pytest.mark.parametrize("site", [
        "amazon.in", "facebook.com", "hdfcbank.com", "bbc.co.uk",
        "some-site-nobody-listed.example", "localhost", "127.0.0.1",
    ])
    def test_any_website_is_allowed(self, anything, site):
        from orchestrator.tools.computer_use import _target_allowed, allowed_targets

        assert _target_allowed(site, allowed_targets())

    @pytest.mark.parametrize("app", ["calc", "cmd", "powershell", "chrome", "outlook"])
    def test_unnamed_desktop_apps_are_still_refused(self, anything, app):
        from orchestrator.tools.computer_use import _target_allowed, allowed_targets

        assert not _target_allowed(app, allowed_targets())

    def test_a_named_desktop_app_still_works(self, anything):
        from orchestrator.tools.computer_use import _target_allowed, allowed_targets

        assert _target_allowed("notepad", allowed_targets())

    def test_without_the_wildcard_nothing_extra_is_allowed(self, monkeypatch):
        from orchestrator.tools.computer_use import _target_allowed, allowed_targets

        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "example.com")
        assert not _target_allowed("amazon.in", allowed_targets())

    def test_credentials_are_still_refused_with_the_wildcard_on(self, anything, tmp_path):
        # The wildcard widens *where*, never *what*. A password must still not
        # be typed into a page, whichever page it is.
        backend = FakeBackend()
        tool = ComputerUseTool(run_id="wild-secret",
                               tools=ToolManager([backend]), root=tmp_path)
        with pytest.raises(ToolError, match="password"):
            tool.execute("log in with password: hunter2\n" + directive(
                goal="log in", targets=["anywhere.example"], actions=["open the page"]))
        assert backend.calls == []

    def test_the_preview_says_the_run_may_go_anywhere(self, anything, tmp_path):
        # Whoever approves this needs to see that it is not bounded by a list.
        tool = ComputerUseTool(run_id="wild-preview",
                               tools=ToolManager([FakeBackend()]), root=tmp_path)
        preview = tool.preview(directive(
            goal="shop around", targets=["amazon.in"], actions=["open the page"]))
        assert "any website" in preview


class TestOpeningItsOwnFiles:
    """Building something and looking at it is one task, not two.

    The second half used to be unreachable: a file has no hostname, so every
    allow-list check refused it and "open the page you just wrote" failed with
    a demand to name a site. The rule here is ownership rather than
    permission -- a file this run produced, in its own workspace or artifacts
    directory, reaches nothing the run did not already have.
    """

    def roots(self, tmp_path):
        work = tmp_path / "workspace"
        art = tmp_path / "artifacts"
        work.mkdir()
        art.mkdir()
        return [work, art]

    def test_a_file_it_produced_resolves_to_a_file_url(self, tmp_path):
        from orchestrator.tools.computer_use import local_file_url

        roots = self.roots(tmp_path)
        (roots[1] / "todo.html").write_text("<h1>todo</h1>", encoding="utf-8")

        url = local_file_url("todo.html", roots)
        # Served, not file:// -- browsers refuse the file: protocol under
        # automation, so a page the run built is published on loopback.
        assert url and url.startswith("http://127.0.0.1:")
        assert url.endswith("/todo.html")

    def test_a_file_url_is_accepted_too(self, tmp_path):
        from orchestrator.tools.computer_use import local_file_url

        roots = self.roots(tmp_path)
        target = roots[1] / "report.html"
        target.write_text("x", encoding="utf-8")
        served = local_file_url(target.as_uri(), roots)
        assert served and served.endswith("/report.html")

    def test_anything_outside_those_directories_is_refused(self, tmp_path):
        # Otherwise this becomes a way to read the rest of the disk.
        from orchestrator.tools.computer_use import local_file_url

        roots = self.roots(tmp_path)
        outside = tmp_path / "secrets.html"
        outside.write_text("private", encoding="utf-8")
        assert local_file_url(str(outside), roots) is None

    def test_traversal_out_of_the_workspace_is_refused(self, tmp_path):
        from orchestrator.tools.computer_use import local_file_url

        roots = self.roots(tmp_path)
        outside = tmp_path / "escape.html"
        outside.write_text("private", encoding="utf-8")
        assert local_file_url("../escape.html", roots) is None

    def test_a_missing_file_is_not_invented(self, tmp_path):
        from orchestrator.tools.computer_use import local_file_url

        assert local_file_url("nope.html", self.roots(tmp_path)) is None

    def test_a_website_is_left_to_the_allow_list(self, tmp_path):
        from orchestrator.tools.computer_use import local_file_url

        assert local_file_url("https://example.com/x.html",
                              self.roots(tmp_path)) is None

    def test_the_browser_opens_the_file_it_just_wrote(self, allow_everything, tmp_path):
        # End to end: no site named, and it still opens the artifact.
        navigate = FakeBackend("web_browser_navigate")
        tool = ComputerUseTool(run_id="own-file",
                               tools=ToolManager([navigate]), root=tmp_path)
        tool.open_roots = self.roots(tmp_path)
        (tool.open_roots[1] / "todo.html").write_text("<h1>todo</h1>", encoding="utf-8")

        tool.execute(directive(goal="open todo.html and confirm it renders",
                               actions=["open the page"]))

        opened = json.loads(navigate.calls[0].removeprefix("TOOL_DIRECTIVE: "))
        assert opened["arguments"]["url"].startswith("http://127.0.0.1:")
        assert opened["arguments"]["url"].endswith("/todo.html")

    def test_explicit_file_url_uses_browser_even_when_desktop_is_available(
            self, allow_everything, tmp_path):
        browser = FakeBackend("web_browser_navigate")
        desktop = FakeBackend("desktop_native")
        tool = ComputerUseTool(run_id="explicit-file",
                               tools=ToolManager([browser, desktop]), root=tmp_path)
        tool.open_roots = self.roots(tmp_path)
        page = tool.open_roots[0] / "calculator.html"
        page.write_text("<h1>Calculator</h1>", encoding="utf-8")

        tool.execute(directive(goal="open the calculator in a browser",
                               targets=[page.as_uri()], actions=["read the page"]))

        assert browser.calls
        assert not desktop.calls


    def test_it_opens_what_this_run_just_built_when_the_plan_names_a_fact(
            self, allow_everything, tmp_path):
        # Plans refer to their own outputs by fact name -- "the path given by
        # the todo_app_path fact" -- so the filename never reaches the tool.
        import time as _time

        navigate = FakeBackend("web_browser_navigate")
        tool = ComputerUseTool(run_id="built-it",
                               tools=ToolManager([navigate]), root=tmp_path)
        tool.open_roots = self.roots(tmp_path)
        _time.sleep(0.01)
        (tool.open_roots[1] / "todo.html").write_text("<h1>todo</h1>", encoding="utf-8")

        tool.execute(directive(
            goal="open the saved app and confirm the task appears",
            actions=["open the page", "read the page"]))

        opened = json.loads(navigate.calls[0].removeprefix("TOOL_DIRECTIVE: "))
        assert opened["arguments"]["url"].endswith("/todo.html")

    def test_it_does_not_reach_for_files_that_were_already_there(
            self, allow_everything, tmp_path):
        # Anything predating this run is not this run's output, and picking it
        # up would turn a missing target into a silently wrong one.
        import time as _time

        navigate = FakeBackend("web_browser_navigate")
        roots = self.roots(tmp_path)
        (roots[1] / "old.html").write_text("<h1>old</h1>", encoding="utf-8")
        _time.sleep(0.05)          # clear the filesystem's timestamp granularity

        tool = ComputerUseTool(run_id="not-mine",
                               tools=ToolManager([navigate]), root=tmp_path)
        tool.open_roots = roots

        with pytest.raises(ToolError, match="targets"):
            tool.execute(directive(goal="open the saved app and confirm it",
                                   actions=["open the page"]))
        assert navigate.calls == []


class TestReliableDesktopDemos:
    def test_notepad_requires_exact_readback(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "notepad")

        class NotepadBackend(FakeBackend):
            def _run(self, task, context=None):
                self.calls.append(task)
                spec = json.loads(task.removeprefix("TOOL_DIRECTIVE: "))["arguments"]
                if spec["action"] == "read_window":
                    return "[desktop_native] Notepad contains 5 character(s):\nwrong"
                return "ok"

        desktop = NotepadBackend("desktop_native")
        tool = ComputerUseTool(run_id="notepad-check",
                               tools=ToolManager([desktop]), root=tmp_path)
        with pytest.raises(ToolError, match="Notepad text mismatch"):
            tool.execute(directive(goal="Open a new Notepad document and type exactly 'Hello'. Leave it visible.",
                                   targets=["notepad"], actions=[]))

    def test_paint_saves_png_and_opens_it(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "paint")
        desktop = FakeBackend("desktop_native", response="Untitled - Paint")
        tool = ComputerUseTool(run_id="paint-check",
                               tools=ToolManager([desktop]), root=tmp_path)
        result = tool.execute(directive(
            goal="Open Paint, draw a house with roof, door, sun and grass, then save it",
            targets=["paint"], actions=[]))
        picture = tmp_path / "paint-check" / "artifacts" / "house-in-paint.png"
        assert picture.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert "Opened the PNG in Paint" in result
        specs = [json.loads(call.removeprefix("TOOL_DIRECTIVE: "))["arguments"]
                 for call in desktop.calls]
        assert any(spec.get("action") == "launch" and spec.get("path") == str(picture)
                   for spec in specs)


class TestLaunchingAnEditorOnItsWork:
    """An editor is only useful pointed at something.

    So `launch` takes one path -- and only one this run owns. The containment
    check is the whole point: "open the editor on the project you just wrote"
    must work, while "open it on anything else on this disk" must not.
    """

    def tool(self, tmp_path, monkeypatch):
        import orchestrator.tools.desktop as desktop_module

        monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "code, notepad")
        monkeypatch.setattr(desktop_module, "_win32", lambda: (None, None, None, None))
        monkeypatch.setattr(desktop_module, "foreground_window", lambda: "Code")
        monkeypatch.setattr(desktop_module.time, "sleep", lambda _s: None)
        monkeypatch.setattr(desktop_module.shutil, "which",
                            lambda name: f"C:\\bin\\{name if name.endswith('.exe') else name + '.exe'}")
        started = []
        monkeypatch.setattr(desktop_module.subprocess, "Popen",
                            lambda cmd, **kw: started.append((cmd, kw)))
        return DesktopTool(run_id="editor", root=tmp_path), started

    def test_it_opens_a_folder_this_run_owns(self, tmp_path, monkeypatch):
        tool, started = self.tool(tmp_path, monkeypatch)
        tool.workspace.mkdir(parents=True, exist_ok=True)
        (tool.workspace / "todo.html").write_text("<h1>x</h1>", encoding="utf-8")

        result = tool._launch("code", ["code"], ".")

        command, kwargs = started[0]
        assert command[0].endswith("code.exe")
        assert str(tool.workspace.resolve()) in command[-1]
        assert kwargs.get("shell") is False      # never through a shell
        assert "launched 'code'" in result

    def test_paint_allow_list_name_launches_mspaint(self, tmp_path, monkeypatch):
        tool, started = self.tool(tmp_path, monkeypatch)

        result = tool._launch("paint", ["paint"], "")

        assert started[0][0][0].endswith("mspaint.exe")
        assert "launched 'paint'" in result

    def test_a_folder_outside_the_run_is_refused(self, tmp_path, monkeypatch):
        tool, started = self.tool(tmp_path, monkeypatch)
        outside = tmp_path / "somewhere-else"
        outside.mkdir()

        with pytest.raises(ToolError, match="not inside this run"):
            tool._launch("code", ["code"], str(outside))
        assert started == []

    def test_traversal_is_refused(self, tmp_path, monkeypatch):
        tool, started = self.tool(tmp_path, monkeypatch)
        tool.workspace.mkdir(parents=True, exist_ok=True)

        with pytest.raises(ToolError, match="not inside this run"):
            tool._launch("code", ["code"], "../../Windows")
        assert started == []

    def test_an_app_outside_the_allow_list_is_still_refused(self, tmp_path, monkeypatch):
        tool, started = self.tool(tmp_path, monkeypatch)
        with pytest.raises(ToolError, match="allow-list"):
            tool._launch("powershell", ["code", "notepad"], "")
        assert started == []

    def test_a_missing_executable_says_so(self, tmp_path, monkeypatch):
        import orchestrator.tools.desktop as desktop_module

        tool, _ = self.tool(tmp_path, monkeypatch)
        monkeypatch.setattr(desktop_module.shutil, "which", lambda name: None)
        with pytest.raises(ToolError, match="not on PATH"):
            tool._launch("code", ["code"], "")


class TestDisplayScaling:
    """Measurement, capture and pointer must share one coordinate space.

    Windows lies about the screen to programs that have not declared DPI
    awareness: at 125% scaling GetSystemMetrics reports 1536x864 while a
    screenshot captures the real 1920x1080. The vision model is shown the
    capture, so it answers in physical pixels; the click is then issued in
    logical ones and lands 1.25x away. Nothing errors -- the pointer just
    arrives somewhere else, which is the hardest kind of bug to see, because
    the transcript says the click succeeded.
    """

    def test_declaring_awareness_is_idempotent_and_never_raises(self):
        import orchestrator.tools.desktop as desktop_module

        desktop_module._DPI_READY = False
        desktop_module._ensure_dpi_aware()
        desktop_module._ensure_dpi_aware()      # already set: must not raise
        assert desktop_module._DPI_READY

    @pytest.mark.skipif(os.name != "nt", reason="Windows display scaling only")
    def test_the_screen_measures_the_same_as_it_captures(self):
        import win32api
        from PIL import ImageGrab

        from orchestrator.tools.desktop import _win32

        _win32()
        measured = (win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1))
        captured = ImageGrab.grab().size
        assert measured == captured, (
            f"measured {measured} but captured {captured}: a click aimed at "
            "what the model saw would land somewhere else")
