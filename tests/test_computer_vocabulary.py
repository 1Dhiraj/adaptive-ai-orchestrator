"""The browser and desktop action vocabularies, and the loops that use them.

Two things are being defended here.

The first is that an instruction compiles to the call it *says* it makes. A
browser action that compiles to plausible-but-wrong arguments does not fail
loudly; it returns something, and the run reports work it never did. So every
supported phrasing is checked against the exact payload it produces.

The second is that the vocabulary stays closed. An unrecognised action must
raise, never be quietly skipped -- a skipped action is the same failure mode
wearing a different hat.
"""

from __future__ import annotations

import json

import pytest

from orchestrator.llm import LLMResponse, LLMUsage
from orchestrator.tools import ToolManager
from orchestrator.tools.base import Tool, ToolError
from orchestrator.tools.computer_use import (
    BROWSER_VOCABULARY,
    DESKTOP_VOCABULARY,
    ComputerUseTool,
    _adapt_arguments,
    _browser_interaction,
    _native_action_spec,
    _snapshot_controls,
)

SNAPSHOT = """
- Page Title: Sign up
- heading "Create an account" [level=1] [ref=h1]
- textbox "Full name" [ref=n1]
- textbox "Email" [ref=e1]
- checkbox "Subscribe" [ref=c1]
- combobox "Country" [ref=k1]
- button "Continue" [ref=b1]
- button "Cancel" [ref=b2]
- link "Terms" [ref=l1]
- paragraph: By continuing you agree to the terms.
"""


class FakeTool(Tool):
    """A backend that records its payloads, optionally with a real schema."""

    def __init__(self, name: str, schema: dict | None = None,
                 response: str = "ok", live: bool = True) -> None:
        super().__init__()
        self.name = name
        self.capability = name
        self._schema = schema or {}
        self._live = live
        self.response = response
        self.calls: list[str] = []

    def is_live(self) -> bool:
        return self._live

    def argument_schema(self) -> dict:
        return self._schema

    def _run(self, task, context=None):
        self.calls.append(task)
        return self.response

    @property
    def arguments(self) -> list[dict]:
        return [json.loads(call.removeprefix("TOOL_DIRECTIVE: "))["arguments"]
                for call in self.calls if call.startswith("TOOL_DIRECTIVE: ")]


def compiled(action: str, snapshot: str = SNAPSHOT) -> tuple[str, dict]:
    """The tool name and arguments one instruction compiles to."""
    result = _browser_interaction(action, snapshot)
    assert result is not None, f"{action!r} compiled to nothing"
    names, arguments = result
    return names[0], arguments


# ---------------------------------------------------------------------------
# Browser vocabulary
# ---------------------------------------------------------------------------


class TestBrowserPointerActions:
    @pytest.mark.parametrize("action, expected", [
        ("click Continue", {"element": "Continue", "ref": "b1"}),
        ("click the Continue button",
         {"element": "Continue", "ref": "b1"}),
        ("double-click Continue",
         {"element": "Continue", "ref": "b1", "doubleClick": True}),
        ("right-click Continue",
         {"element": "Continue", "ref": "b1", "button": "right"}),
        ("middle-click Continue",
         {"element": "Continue", "ref": "b1", "button": "middle"}),
        ("ctrl-click Terms",
         {"element": "Terms", "ref": "l1", "modifiers": ["ControlOrMeta"]}),
        ("shift-click Terms",
         {"element": "Terms", "ref": "l1", "modifiers": ["Shift"]}),
    ])
    def test_click_variants(self, action, expected):
        name, arguments = compiled(action)
        assert name == "web_browser_click"
        assert arguments == expected

    def test_hover_and_drag_resolve_both_endpoints(self):
        assert compiled("hover Cancel") == (
            "web_browser_hover", {"element": "Cancel", "ref": "b2"})
        name, arguments = compiled("drag Continue to Cancel")
        assert name == "web_browser_drag"
        assert arguments == {"startElement": "Continue", "startRef": "b1",
                             "endElement": "Cancel", "endRef": "b2"}

    def test_an_ambiguous_label_is_refused_rather_than_guessed(self):
        # "button" appears on both Continue and Cancel; picking one silently
        # would click the wrong thing with nothing in the transcript to show it.
        with pytest.raises(ToolError, match="could not uniquely find"):
            _browser_interaction("click somethingelse", SNAPSHOT)


class TestBrowserTextEntry:
    def test_typing_targets_the_named_field(self):
        name, arguments = compiled("type 'Ada' into Full name")
        assert name == "web_browser_type"
        assert arguments == {"element": "Full name", "ref": "n1", "text": "Ada"}

    def test_typing_can_submit_in_the_same_call(self):
        _, arguments = compiled("type 'Ada' into Full name and submit")
        assert arguments["submit"] is True

    def test_clearing_a_field_types_nothing_into_it(self):
        assert compiled("clear Email") == (
            "web_browser_type", {"element": "Email", "ref": "e1", "text": ""})

    def test_select_uses_the_option_list(self):
        assert compiled("select 'India' from Country") == (
            "web_browser_select_option",
            {"element": "Country", "ref": "k1", "values": ["India"]})

    def test_checkbox_state_is_set_not_toggled(self):
        # Toggling would depend on the box's current state, which the plan
        # cannot see; setting it is idempotent and says what it means.
        name, arguments = compiled("check Subscribe")
        assert name == "web_browser_fill_form"
        assert arguments["fields"] == [{
            "name": "Subscribe", "element": "Subscribe", "ref": "c1",
            "type": "checkbox", "value": "true"}]
        _, unchecked = compiled("uncheck Subscribe")
        assert unchecked["fields"][0]["value"] == "false"

    def test_a_whole_form_becomes_one_call(self):
        name, arguments = compiled(
            "fill the form with Full name='Ada', Email='ada@example.com'")
        assert name == "web_browser_fill_form"
        assert [(f["name"], f["value"], f["type"]) for f in arguments["fields"]] == [
            ("Full name", "Ada", "textbox"),
            ("Email", "ada@example.com", "textbox")]

    def test_a_form_with_no_pairs_says_what_it_wanted(self):
        with pytest.raises(ToolError, match="field='value'"):
            _browser_interaction("fill the form with everything", SNAPSHOT)


class TestBrowserNavigationAndWaiting:
    @pytest.mark.parametrize("action, name, expected", [
        ("go back", "web_browser_navigate_back", {}),
        ("close the browser", "web_browser_close", {}),
        ("submit", "web_browser_press_key", {"key": "Enter"}),
        ("press Escape", "web_browser_press_key", {"key": "Escape"}),
        ("press Ctrl+A", "web_browser_press_key", {"key": "Control+A"}),
        ("scroll down", "web_browser_press_key", {"key": "PageDown"}),
        ("scroll up", "web_browser_press_key", {"key": "PageUp"}),
        ("scroll to the bottom", "web_browser_press_key", {"key": "End"}),
        ("scroll to the top", "web_browser_press_key", {"key": "Home"}),
        ("wait 3 seconds", "web_browser_wait_for", {"time": 3.0}),
        ("wait for 'Results'", "web_browser_wait_for", {"text": "Results"}),
        ("wait until 'Loading' is gone", "web_browser_wait_for",
         {"textGone": "Loading"}),
        ("find 'total cost'", "web_browser_find", {"text": "total cost"}),
    ])
    def test_action_compiles(self, action, name, expected):
        assert compiled(action) == (name, expected)

    def test_forward_and_reload_go_through_evaluate(self):
        # Playwright MCP has no forward or reload tool; saying so in the
        # transcript beats pretending the instruction was unsupported.
        name, arguments = compiled("go forward")
        assert name == "web_browser_evaluate"
        assert "history.forward" in arguments["function"]
        assert "location.reload" in compiled("reload")[1]["function"]

    def test_a_long_wait_is_capped(self):
        assert compiled("wait 9999 seconds")[1]["time"] == 30.0

    def test_navigation_and_reading_are_left_to_the_page_loop(self):
        for action in ("open the page", "read the page", "search for widgets",
                       "extract the headings"):
            assert _browser_interaction(action, SNAPSHOT) is None


class TestBrowserTabsWindowAndDialogs:
    @pytest.mark.parametrize("action, expected", [
        ("new tab", {"action": "new"}),
        ("new tab https://example.com", {"action": "new",
                                         "url": "https://example.com"}),
        ("switch to tab 2", {"action": "select", "index": 1}),
        ("close tab", {"action": "close"}),
        ("list tabs", {"action": "list"}),
    ])
    def test_tab_actions(self, action, expected):
        assert compiled(action) == ("web_browser_tabs", expected)

    def test_tab_numbers_are_one_based_for_people_zero_based_on_the_wire(self):
        assert compiled("switch to tab 1")[1]["index"] == 0

    @pytest.mark.parametrize("action, expected", [
        ("accept the dialog", {"accept": True}),
        ("dismiss the dialog", {"accept": False}),
        ("cancel the dialog", {"accept": False}),
        ("accept the dialog with 'Ada'", {"accept": True, "promptText": "Ada"}),
    ])
    def test_dialogs(self, action, expected):
        assert compiled(action) == ("web_browser_handle_dialog", expected)

    def test_resize_and_upload(self):
        assert compiled("resize to 1280x720") == (
            "web_browser_resize", {"width": 1280, "height": 720})
        assert compiled("upload 'report.pdf'") == (
            "web_browser_file_upload", {"paths": ["report.pdf"]})


class TestBrowserObservation:
    def test_screenshots_carry_the_fields_the_server_requires(self):
        name, arguments = compiled("screenshot")
        assert name == "web_browser_take_screenshot"
        assert arguments == {"type": "png", "scale": "css"}
        assert compiled("full-page screenshot")[1]["fullPage"] is True

    def test_console_level_follows_the_words_used(self):
        assert compiled("read the console")[1] == {"level": "info"}
        assert compiled("read the console errors")[1] == {"level": "error"}
        assert compiled("read the console warnings")[1] == {"level": "warning"}

    def test_network_requests_exclude_static_noise(self):
        assert compiled("read the network requests") == (
            "web_browser_network_requests", {"static": False})


class TestVocabularyIsClosed:
    @pytest.mark.parametrize("action", [
        "do the needful",
        "click",
        "hack the mainframe",
        "",
    ])
    def test_an_unknown_instruction_raises_with_the_menu(self, action):
        with pytest.raises(ToolError) as caught:
            _browser_interaction(action, SNAPSHOT)
        assert "unsupported browser action" in str(caught.value)
        assert "click <label>" in str(caught.value)

    def test_the_menu_is_the_one_the_model_is_shown(self):
        # The loop's system prompt quotes BROWSER_VOCABULARY. If the two ever
        # drift, the model is offered actions the compiler rejects.
        for phrase in ("click <label>", "scroll down", "accept the dialog"):
            assert phrase in BROWSER_VOCABULARY


# ---------------------------------------------------------------------------
# Schema adaptation
# ---------------------------------------------------------------------------


CURRENT_CLICK_SCHEMA = {
    "properties": {"target": {}, "element": {}, "button": {}, "doubleClick": {}},
    "required": ["target"],
}
LEGACY_CLICK_SCHEMA = {
    "properties": {"ref": {}, "element": {}},
    "required": ["ref"],
}


class TestSchemaAdaptation:
    def test_element_handles_are_renamed_to_what_the_server_declares(self):
        tool = FakeTool("web_browser_click", CURRENT_CLICK_SCHEMA)
        assert _adapt_arguments(tool, {"element": "Continue", "ref": "b1"}) == {
            "element": "Continue", "target": "b1"}

    def test_a_server_that_still_says_ref_keeps_getting_ref(self):
        tool = FakeTool("web_browser_click", LEGACY_CLICK_SCHEMA)
        assert _adapt_arguments(tool, {"element": "Continue", "ref": "b1"}) == {
            "element": "Continue", "ref": "b1"}

    def test_drag_endpoints_are_renamed_too(self):
        tool = FakeTool("web_browser_drag", {
            "properties": {"startTarget": {}, "endTarget": {},
                           "startElement": {}, "endElement": {}},
            "required": ["startTarget", "endTarget"]})
        adapted = _adapt_arguments(tool, {
            "startElement": "a", "startRef": "r1",
            "endElement": "b", "endRef": "r2"})
        assert adapted == {"startElement": "a", "startTarget": "r1",
                           "endElement": "b", "endTarget": "r2"}

    def test_fields_inside_a_form_are_renamed(self):
        tool = FakeTool("web_browser_fill_form", {
            "properties": {"fields": {"items": {"properties": {
                "target": {}, "name": {}, "type": {}, "value": {}}}}},
            "required": ["fields"]})
        adapted = _adapt_arguments(tool, {"fields": [
            {"name": "Email", "element": "Email", "ref": "e1",
             "type": "textbox", "value": "a@b.c"}]})
        assert adapted["fields"] == [
            {"name": "Email", "target": "e1", "type": "textbox", "value": "a@b.c"}]

    def test_required_fields_the_caller_does_not_care_about_are_filled(self):
        tool = FakeTool("web_browser_take_screenshot", {
            "properties": {"scale": {}, "type": {}}, "required": ["scale"]})
        assert _adapt_arguments(tool, {"type": "png"}) == {
            "type": "png", "scale": "css"}

    def test_fields_the_server_does_not_declare_are_dropped(self):
        # Strict servers reject unknown fields outright, which turns a
        # harmless extra into a failed action.
        tool = FakeTool("web_browser_navigate", {
            "properties": {"url": {}}, "required": ["url"]})
        assert _adapt_arguments(tool, {"url": "https://x.test", "ref": "b1"}) == {
            "url": "https://x.test"}

    def test_a_backend_with_no_schema_is_passed_through_untouched(self):
        assert _adapt_arguments(FakeTool("anything"), {"ref": "b1"}) == {"ref": "b1"}


# ---------------------------------------------------------------------------
# Desktop vocabulary
# ---------------------------------------------------------------------------


class TestDesktopVocabulary:
    @pytest.mark.parametrize("action, expected", [
        ("launch notepad", {"action": "launch", "app": "notepad"}),
        ("screenshot", {"action": "screenshot"}),
        ("list the windows", {"action": "windows"}),
        ("wait 2 seconds", {"action": "wait", "seconds": 2.0}),
        ("select all", {"action": "key", "keys": "ctrl+a", "window": "notepad"}),
        ("copy", {"action": "key", "keys": "ctrl+c", "window": "notepad"}),
        ("paste", {"action": "key", "keys": "ctrl+v", "window": "notepad"}),
        ("save file", {"action": "key", "keys": "ctrl+s", "window": "notepad"}),
        ("press ctrl+shift+n",
         {"action": "key", "keys": "ctrl+shift+n", "window": "notepad"}),
        ("read the window", {"action": "read_window", "window": "notepad"}),
        ("read the clipboard", {"action": "clipboard_get"}),
        ("maximise the window",
         {"action": "window_state", "state": "maximize", "window": "notepad"}),
        ("minimise the window",
         {"action": "window_state", "state": "minimize", "window": "notepad"}),
        ("close the window",
         {"action": "window_state", "state": "close", "window": "notepad"}),
    ])
    def test_action_compiles(self, action, expected):
        assert _native_action_spec(action, "notepad") == expected

    def test_typing_keeps_the_text_exactly(self):
        assert _native_action_spec("type 'Dear Ada,'", "notepad") == {
            "action": "type", "text": "Dear Ada,", "window": "notepad"}

    def test_clipboard_can_be_written(self):
        assert _native_action_spec("put 'hello' on the clipboard", "notepad") == {
            "action": "clipboard_set", "text": "hello"}

    @pytest.mark.parametrize("action, expected", [
        ("click at (400,300)",
         {"action": "click", "x": 400, "y": 300, "window": "notepad"}),
        ("double-click at (10,20)",
         {"action": "double_click", "x": 10, "y": 20, "window": "notepad"}),
        ("right-click at (10,20)",
         {"action": "right_click", "x": 10, "y": 20, "window": "notepad"}),
        ("move to (5,6)", {"action": "move", "x": 5, "y": 6, "window": "notepad"}),
    ])
    def test_pointer_coordinates(self, action, expected):
        assert _native_action_spec(action, "notepad") == expected

    def test_drag_carries_both_ends(self):
        assert _native_action_spec("drag (1,2) to (3,4)", "notepad") == {
            "action": "drag", "x": 1, "y": 2, "to_x": 3, "to_y": 4,
            "window": "notepad"}

    def test_scroll_direction_is_signed(self):
        assert _native_action_spec("scroll down", "notepad")["amount"] == -3
        assert _native_action_spec("scroll up 5", "notepad")["amount"] == 5

    def test_naming_a_thing_asks_the_screen_where_it_is(self):
        # No coordinates in the instruction, so the vision backend finds it.
        assert _native_action_spec("click the Save button", "notepad") == {
            "action": "click_target", "target": "Save button",
            "button": "left", "window": "notepad"}
        assert _native_action_spec("right-click the icon", "notepad")["button"] == "right"

    def test_unknown_desktop_action_raises_with_the_menu(self):
        with pytest.raises(ToolError) as caught:
            _native_action_spec("reformat the hard drive", "notepad")
        assert "unsupported desktop action" in str(caught.value)
        assert "launch <app>" in DESKTOP_VOCABULARY


# ---------------------------------------------------------------------------
# Reading a page well enough to choose a next action
# ---------------------------------------------------------------------------


class TestSnapshotControls:
    def test_controls_are_listed_by_the_label_a_person_would_read(self):
        listed = _snapshot_controls(SNAPSHOT)
        assert 'button "Continue"' in listed
        assert 'textbox "Email"' in listed
        assert 'heading "Create an account"' in listed

    def test_refs_are_not_shown(self):
        # The compiler resolves labels itself. Showing refs invites the model
        # to paste one into an instruction that does not take one.
        assert "ref=" not in _snapshot_controls(SNAPSHOT)

    def test_the_listing_is_bounded(self):
        crowded = "\n".join(
            f'- button "Button number {i}" [ref=b{i}]' for i in range(500))
        assert len(_snapshot_controls(crowded, limit=300)) <= 300


# ---------------------------------------------------------------------------
# Recovering a plan from a step's own words
# ---------------------------------------------------------------------------


class TestPlanFromDescription:
    """What a step types when the model omitted its directive.

    These are the words a person actually writes, and the text between the
    quotes is going into a real document. Capturing one word too many here is
    not a cosmetic bug: it writes the instruction into the file alongside the
    content, and the step still reports success.
    """

    @pytest.fixture(autouse=True)
    def allow_notepad(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "notepad")

    @pytest.mark.parametrize("description, expected", [
        ("Open Notepad and type the exact sentence: 'Hello there.'",
         'type "Hello there."'),
        ('Open Notepad and type "Hello, world." then save',
         'type "Hello, world."'),
        ("Open Notepad and write the following text: 'Line one'",
         'type "Line one"'),
        ("Open notepad and type Hello there", 'type "Hello there"'),
        # No quotes at all: the phrase naming the text must not become part
        # of it, or the document contains the instruction too.
        ("Launch Notepad and type the sentence: Adaptive orchestration works.",
         'type "Adaptive orchestration works"'),
        ("Open Notepad and type the exact sentence: Hello there",
         'type "Hello there"'),
        ("Open Notepad and write the following text: Line one",
         'type "Line one"'),
        ("Open notepad and type exactly Adaptive works",
         'type "Adaptive works"'),
    ])
    def test_the_text_typed_is_the_text_asked_for(self, description, expected):
        from orchestrator.tools.computer_use import _plan_from_description

        assert expected in _plan_from_description(description).actions

    def test_the_app_is_launched_before_anything_is_typed(self):
        from orchestrator.tools.computer_use import _plan_from_description

        actions = _plan_from_description(
            "Open Notepad and type 'hi'").actions
        assert actions[0] == "launch the app"

    @pytest.mark.parametrize("description", [
        "Read the Notepad document back and verify the text is there",
        "Confirm the Notepad window contains the sentence Adaptive works",
        "Check what the Notepad document currently says",
    ])
    def test_a_verification_step_actually_looks_at_the_document(self, description):
        # An empty plan is how a verification step comes back saying
        # everything is fine without having looked at anything.
        from orchestrator.tools.computer_use import _plan_from_description

        assert _plan_from_description(description).actions == ["read the window"]

    def test_reading_is_not_added_to_a_step_that_types(self):
        from orchestrator.tools.computer_use import _plan_from_description

        assert "read the window" not in _plan_from_description(
            "Open Notepad and type 'hi', then check it saved").actions


# ---------------------------------------------------------------------------
# Controls with no name of their own
# ---------------------------------------------------------------------------


UNNAMED = """
- heading "Checkboxes" [level=3] [ref=e8]
- form [ref=e9]:
  - checkbox [ref=e10]
  - checkbox [ref=e11] [checked]
- button "Go" [ref=e12]
- link "One" [ref=e13]
- link "Two" [ref=e14]
- heading "Dropdown List" [level=3] [ref=d1]
- combobox [ref=d2]:
  - option "Please select an option" [disabled] [selected]
  - option "Option 1"
  - option "Option 2"
"""


class TestUnnamedControls:
    """Plenty of real controls carry no accessible name at all.

    A bare row of checkboxes, an icon-only button, a <select> with a heading
    above it instead of a label. There is no text to match, so the page puts
    the only usable name on a *neighbour* -- and matching that neighbour is
    how "select from Dropdown" ends up failing with "Element is not a
    <select> element", an error about the page for a mistake about which
    element was meant.
    """

    def test_a_dropdown_is_found_by_the_option_asked_for(self):
        _, arguments = compiled("select 'Option 2' from Dropdown", UNNAMED)
        assert arguments["ref"] == "d2"          # the combobox, not the heading
        assert arguments["values"] == ["Option 2"]

    def test_a_named_dropdown_still_resolves_by_its_name(self):
        _, arguments = compiled("select 'India' from Country", """
- combobox "Country" [ref=c4]:
  - option "India"
""")
        assert arguments["ref"] == "c4"

    @pytest.mark.parametrize("action, expected_ref", [
        ("click the first link", "e13"),
        ("click the last link", "e14"),
        ("click the first button", "e12"),
        ("click the second checkbox", "e11"),
    ])
    def test_position_resolves_what_has_no_name(self, action, expected_ref):
        assert compiled(action, UNNAMED)[1]["ref"] == expected_ref

    def test_the_kind_of_control_is_honoured_not_ignored(self):
        # "the first link" must not become "the first control of any kind":
        # the checkboxes come earlier in this page than the links do.
        assert compiled("click the first link", UNNAMED)[1]["ref"] != "e10"

    @pytest.mark.parametrize("action, expected_ref", [
        ("check the first checkbox", "e10"),
        ("check checkbox 2", "e11"),
    ])
    def test_checkboxes_can_be_set_by_position(self, action, expected_ref):
        assert compiled(action, UNNAMED)[1]["fields"][0]["ref"] == expected_ref

    def test_a_position_past_the_end_is_refused(self):
        with pytest.raises(ToolError, match="could not uniquely find"):
            _browser_interaction("click the ninth link", UNNAMED)
