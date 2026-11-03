"""Every directive parser accepts both spellings.

The project documents two shapes. `computer_use` and `desktop_native` show
models `{"arguments": {...}}`; the built-in adapters show the fields at the
top level. A model reading one prompt containing both will use either, and for
a while that choice silently decided whether a step worked -- writing a file
failed with "no path given" purely because the wrapped form had been picked,
which reads like a bad plan rather than a mismatched convention.

Five modules parse directives independently. This checks all of them together,
so adding a sixth that handles only one shape fails here rather than in front
of a user.
"""

from __future__ import annotations

import json

import pytest


def flat(**fields) -> str:
    return "do it\nTOOL_DIRECTIVE: " + json.dumps(fields)


def wrapped(**fields) -> str:
    return "do it\nTOOL_DIRECTIVE: " + json.dumps({"arguments": fields})


class TestEveryParserAcceptsBothForms:
    def test_builtin_adapters(self):
        from orchestrator.tools.builtin import _directive

        fields = {"action": "write", "path": "a.html", "content": "x"}
        assert _directive(flat(**fields)) == fields
        assert _directive(wrapped(**fields)) == fields

    def test_desktop(self):
        from orchestrator.tools.desktop import DesktopTool

        fields = {"action": "type", "text": "hello", "window": "notepad"}
        assert DesktopTool._directive(flat(**fields)) == fields
        assert DesktopTool._directive(wrapped(**fields)) == fields

    def test_computer_task(self):
        from orchestrator.tools.computer_task import ComputerTaskTool

        assert ComputerTaskTool._goal(flat(goal="count the files")) == "count the files"
        assert ComputerTaskTool._goal(wrapped(goal="count the files")) == "count the files"

    def test_computer_use(self, monkeypatch):
        from orchestrator.tools.computer_use import _parse_plan

        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "example.com")
        for payload in (flat(goal="read it", targets=["example.com"],
                             actions=["open the page"]),
                        wrapped(goal="read it", targets=["example.com"],
                                actions=["open the page"])):
            plan = _parse_plan(payload)
            assert plan.goal == "read it"
            assert plan.targets == ["example.com"]
            assert plan.actions == ["open the page"]

    def test_adaptive_email(self):
        from orchestrator.tools.adaptive_email import email_message

        fields = {"to": "a@b.c", "subject": "Hi", "body": "Text"}
        for payload in (flat(**fields), wrapped(**fields)):
            message = email_message(payload)
            assert message["to"] == "a@b.c"
            assert message["subject"] == "Hi"


class TestBothFormsReallyWrite:
    """Not just parsed -- the file lands either way."""

    @pytest.mark.parametrize("build, name", [(flat, "flat.html"),
                                             (wrapped, "wrapped.html")])
    def test_workspace_writes_the_file(self, tmp_path, build, name):
        from orchestrator.tools.workspace import WorkspaceFileTool

        tool = WorkspaceFileTool(run_id="forms", root=tmp_path)
        tool.execute(build(action="write", path=name, content="<h1>x</h1>"))
        assert (tool.workspace / name).read_text(encoding="utf-8") == "<h1>x</h1>"


class TestContentWithBracesSurvives:
    """The payload routinely contains braces -- that is the whole point.

    Every stylesheet, script and JSON sample an agent writes into a file has
    them. A regex like ``\{.*?\}`` stops at the first closing brace ending a
    line, which for a directive carrying an HTML document lands in the middle
    of its CSS. The truncated fragment fails to parse, the directive comes
    back empty, and the tool reports "no path given" for a plan that had said
    exactly where to write the file. Whether a step worked came down to
    whether the generated code happened to contain a brace.
    """

    HTML = ("<html><head><style>body{margin:0;padding:0}\n"
            "h1{color:red}</style></head><body>\n"
            "<script>function go(){ const s={a:1,b:[2,3]}; "
            "localStorage.setItem('k', JSON.stringify(s)); }</script>\n"
            "</body></html>")

    def test_a_whole_web_page_round_trips(self):
        from orchestrator.tools.base import extract_directive

        payload = ("Here is the file.\nTOOL_DIRECTIVE: " + json.dumps(
            {"action": "write", "path": "weather.html", "content": self.HTML}))
        parsed = extract_directive(payload)
        assert parsed["path"] == "weather.html"
        assert parsed["content"] == self.HTML

    def test_the_file_actually_lands_with_its_braces_intact(self, tmp_path):
        from orchestrator.tools.workspace import WorkspaceFileTool

        tool = WorkspaceFileTool(run_id="braces", root=tmp_path)
        tool.execute("TOOL_DIRECTIVE: " + json.dumps(
            {"action": "write", "path": "weather.html", "content": self.HTML}))
        written = (tool.workspace / "weather.html").read_text(encoding="utf-8")
        assert written == self.HTML
        assert "localStorage" in written

    def test_reasoning_that_mentions_the_marker_does_not_win(self):
        # A model may narrate "I will emit a TOOL_DIRECTIVE" before doing it.
        from orchestrator.tools.base import extract_directive

        payload = ("First I will send a TOOL_DIRECTIVE: with the file.\n"
                   "TOOL_DIRECTIVE: " + json.dumps(
                       {"action": "write", "path": "real.html", "content": "x"}))
        assert extract_directive(payload)["path"] == "real.html"

    def test_trailing_prose_after_the_directive_is_tolerated(self):
        from orchestrator.tools.base import extract_directive

        payload = ("TOOL_DIRECTIVE: " + json.dumps(
            {"action": "write", "path": "a.html", "content": "body{x}"})
            + "\nThat should do it.")
        assert extract_directive(payload)["path"] == "a.html"

    def test_no_directive_at_all_is_empty_not_an_error(self):
        from orchestrator.tools.base import extract_directive

        assert extract_directive("just some prose about {braces}") == {}


class TestFileHandoverWithoutADirective:
    """A fenced code block plus "save it as x.html" is a clear handover.

    Models reach for it constantly, and it used to be thrown away: the step
    failed with "no path given" while the finished file sat in the output.
    Packaging should not decide whether work survives.
    """

    PAYLOAD = ("Here is the finished app. Save it as weather.html.\n\n"
               "```html\n"
               "<html><style>body{margin:0}</style>\n"
               "<script>const s={a:1};</script></html>\n"
               "```\n")

    def test_the_file_is_written_from_the_fenced_block(self, tmp_path):
        from orchestrator.tools.workspace import WorkspaceFileTool

        tool = WorkspaceFileTool(run_id="fenced", root=tmp_path)
        tool.execute(self.PAYLOAD)
        written = (tool.workspace / "weather.html").read_text(encoding="utf-8")
        assert written.startswith("<html>")
        assert "const s={a:1}" in written
        assert "```" not in written

    def test_the_name_can_come_from_the_step_description(self, tmp_path):
        from orchestrator.tools.workspace import WorkspaceFileTool

        tool = WorkspaceFileTool(run_id="fenced2", root=tmp_path)
        tool.execute("```html\n<h1>hi</h1>\n```",
                     {"step_description": "Create the page and save it as index.html"})
        assert (tool.workspace / "index.html").exists()

    def test_a_directive_still_wins_over_a_code_block(self, tmp_path):
        # An explicit instruction is not a guess and must not be overridden.
        from orchestrator.tools.workspace import WorkspaceFileTool

        tool = WorkspaceFileTool(run_id="fenced3", root=tmp_path)
        tool.execute("```html\n<h1>ignored</h1>\n```\n"
                     "TOOL_DIRECTIVE: " + json.dumps(
                         {"action": "write", "path": "real.html",
                          "content": "<h1>kept</h1>"}))
        assert (tool.workspace / "real.html").read_text(encoding="utf-8") == "<h1>kept</h1>"

    def test_prose_with_no_file_at_all_is_still_refused(self, tmp_path):
        from orchestrator.tools.workspace import WorkspaceError, WorkspaceFileTool

        tool = WorkspaceFileTool(run_id="fenced4", root=tmp_path)
        with pytest.raises(WorkspaceError, match="no path given"):
            tool.execute("I thought about it but wrote nothing.")
