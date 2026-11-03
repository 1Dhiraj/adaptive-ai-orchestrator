"""The autonomous loops: acting on what the screen shows, not on a guess.

A scripted browser plan has to be right in one attempt -- it is written before
anyone has seen the page, and every later action depends on where the earlier
ones landed. These loops read the page after each action instead. That makes
them the part of the system most able to finish a real task, and the part most
able to go somewhere nobody approved, so the tests below are split evenly
between "does it get there" and "can it leave the fence".
"""

from __future__ import annotations

import json

import pytest

from orchestrator.llm import LLMResponse, LLMUsage
from orchestrator.tools import ToolManager
from orchestrator.tools.base import Tool, ToolError
from orchestrator.tools.computer_use import ComputerUseTool

PAGE = """
- Page Title: Example shop
- heading "Widgets" [level=1] [ref=h1]
- textbox "Search" [ref=s1]
- button "Add to basket" [ref=b1]
- link "Pricing" [ref=l1]
- paragraph: A widget costs 12 GBP.
"""


class ScriptedLLM:
    """Returns prepared decisions in order, and records what it was asked."""

    def __init__(self, *decisions: dict) -> None:
        self.decisions = list(decisions)
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def generate(self, prompt, system=None, json_mode=False, images=None,
                 metadata=None, **kwargs):
        self.prompts.append(prompt)
        self.systems.append(system or "")
        payload = self.decisions.pop(0) if self.decisions else {
            "done": True, "answer": "ran out of scripted turns"}
        return LLMResponse(text=json.dumps(payload),
                           usage=LLMUsage(10, 10, 20), model="scripted",
                           latency_s=0.0)


class FakeBrowserTool(Tool):
    """One Playwright-style MCP tool, with an optional failure to react to."""

    def __init__(self, name: str, response: str = "ok",
                 fail_first: str | None = None) -> None:
        super().__init__()
        self.name = name
        self.capability = name
        self.response = response
        self.fail_first = fail_first
        self.calls: list[str] = []

    def is_live(self) -> bool:
        return True

    def _run(self, task, context=None):
        self.calls.append(task)
        if self.fail_first and len(self.calls) == 1:
            raise ToolError(self.fail_first)
        return self.response

    @property
    def arguments(self) -> list[dict]:
        return [json.loads(call.removeprefix("TOOL_DIRECTIVE: "))["arguments"]
                for call in self.calls if call.startswith("TOOL_DIRECTIVE: ")]


@pytest.fixture
def allow_shop(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
    monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW",
                       "example.com, shop.example.com, notepad")


def browser_tools(snapshot: str = PAGE, **extra) -> dict:
    tools = {
        "navigate": FakeBrowserTool("web_browser_navigate", "navigated"),
        "snapshot": FakeBrowserTool("web_browser_snapshot", snapshot),
        "click": FakeBrowserTool("web_browser_click", "clicked"),
        "type": FakeBrowserTool("web_browser_type", "typed"),
        "press": FakeBrowserTool("web_browser_press_key", "pressed"),
    }
    tools.update(extra)
    return tools


def build(tmp_path, llm, tools: dict) -> ComputerUseTool:
    return ComputerUseTool(run_id="loop", tools=ToolManager(list(tools.values())),
                           root=tmp_path, llm=llm)


def directive(**arguments) -> str:
    return "do the thing\nTOOL_DIRECTIVE: " + json.dumps({"arguments": arguments})


# ---------------------------------------------------------------------------
# Getting there
# ---------------------------------------------------------------------------


class TestBrowserLoopReachesTheGoal:
    def test_main_heading_must_appear_as_a_heading_in_the_page(self, allow_shop, tmp_path):
        llm = ScriptedLLM({"done": True, "answer": "Example shop"})
        tools = browser_tools("Page Title: Example shop\n- paragraph: Welcome")
        with pytest.raises(ToolError, match="heading was not verified"):
            build(tmp_path, llm, tools).execute(directive(
                goal="tell me the exact main heading", targets=["example.com"]))

    def test_main_heading_can_be_verified(self, allow_shop, tmp_path):
        llm = ScriptedLLM({"done": True, "answer": "Example shop"})
        tools = browser_tools('Page Title: Example shop\n- heading "Example shop" [level=1] [ref=h1]')
        result = build(tmp_path, llm, tools).execute(directive(
            goal="tell me the exact main heading", targets=["example.com"]))
        assert "RESULT: Example shop" in result

    def test_a_goal_with_no_actions_is_driven_from_the_page(self, allow_shop, tmp_path):
        llm = ScriptedLLM(
            {"action": "click Pricing", "why": "the price is on that page"},
            {"done": True, "answer": "a widget costs 12 GBP"})
        tools = browser_tools()
        result = build(tmp_path, llm, tools).execute(
            directive(goal="find what a widget costs", targets=["example.com"]))

        assert tools["click"].arguments == [{"element": "Pricing", "ref": "l1"}]
        assert "RESULT: a widget costs 12 GBP" in result

    def test_the_page_is_re_read_after_every_action(self, allow_shop, tmp_path):
        llm = ScriptedLLM(
            {"action": "click Pricing"},
            {"action": "click Add to basket"},
            {"done": True, "answer": "done"})
        tools = browser_tools()
        build(tmp_path, llm, tools).execute(
            directive(goal="buy a widget", targets=["example.com"],
                      autonomous=True))
        # One snapshot after the initial navigation, one after each action.
        assert len(tools["snapshot"].calls) == 3

    def test_the_model_is_shown_labels_and_never_refs(self, allow_shop, tmp_path):
        llm = ScriptedLLM({"done": True, "answer": "nothing to do"})
        build(tmp_path, llm, browser_tools()).execute(
            directive(goal="look around", targets=["example.com"]))

        prompt = llm.prompts[0]
        assert 'button "Add to basket"' in prompt
        assert "ref=" not in prompt
        assert "example.com" in prompt  # it is told what it may touch

    def test_a_failed_action_becomes_an_observation_not_a_dead_run(
            self, allow_shop, tmp_path):
        # The whole point of a loop: a click that misses costs one turn, and
        # the model gets told why so it can try something else.
        llm = ScriptedLLM(
            {"action": "click Checkout"},          # not on the page
            {"action": "click Add to basket"},     # so try the real one
            {"done": True, "answer": "added"})
        tools = browser_tools()
        result = build(tmp_path, llm, tools).execute(
            directive(goal="add a widget", targets=["example.com"]))

        assert "could not uniquely find" in llm.prompts[1]
        assert tools["click"].arguments == [{"element": "Add to basket", "ref": "b1"}]
        assert "RESULT: added" in result

    def test_a_backend_error_is_reported_and_the_loop_continues(
            self, allow_shop, tmp_path):
        tools = browser_tools()
        tools["click"] = FakeBrowserTool("web_browser_click", "clicked",
                                         fail_first="element detached")
        llm = ScriptedLLM(
            {"action": "click Add to basket"},
            {"action": "click Add to basket"},
            {"done": True, "answer": "added on the second try"})
        result = build(tmp_path, llm, tools).execute(
            directive(goal="add a widget", targets=["example.com"]))

        assert "element detached" in llm.prompts[1]
        assert "RESULT: added on the second try" in result

    def test_an_action_this_browser_cannot_do_is_said_so_not_faked(
            self, allow_shop, tmp_path):
        tools = browser_tools()          # no upload tool registered
        llm = ScriptedLLM(
            {"action": "upload 'cv.pdf'"},
            {"done": True, "answer": "could not upload"})
        build(tmp_path, llm, tools).execute(
            directive(goal="apply for the job", targets=["example.com"]))
        assert "cannot do that" in llm.prompts[1]

    def test_evidence_from_the_final_page_is_returned(self, allow_shop, tmp_path):
        llm = ScriptedLLM({"done": True, "answer": "12 GBP"})
        result = build(tmp_path, llm, browser_tools()).execute(
            directive(goal="price a widget", targets=["example.com"]))
        assert "Example shop" in result


# ---------------------------------------------------------------------------
# Staying inside the fence
# ---------------------------------------------------------------------------


class TestBrowserLoopStaysBounded:
    def test_it_cannot_navigate_off_the_allow_list(self, allow_shop, tmp_path):
        llm = ScriptedLLM(
            {"action": "open https://evil.test/steal"},
            {"done": True, "answer": "gave up"})
        tools = browser_tools()
        build(tmp_path, llm, tools).execute(
            directive(goal="wander", targets=["example.com"]))

        opened = [a.get("url") for a in tools["navigate"].arguments]
        assert opened == ["https://example.com"]
        assert "not in the allow-list" in llm.prompts[1]

    def test_it_may_follow_a_link_to_another_allowed_host(self, allow_shop, tmp_path):
        llm = ScriptedLLM(
            {"action": "open https://shop.example.com/pricing"},
            {"done": True, "answer": "read the pricing"})
        tools = browser_tools()
        build(tmp_path, llm, tools).execute(
            directive(goal="read pricing", targets=["example.com"]))

        assert [a.get("url") for a in tools["navigate"].arguments] == [
            "https://example.com", "https://shop.example.com/pricing"]

    def test_a_credential_the_model_invents_is_refused(self, allow_shop, tmp_path):
        # The scripted path refuses secrets in the directive. The loop has to
        # refuse them in the model's own output too, or the guard is decorative.
        llm = ScriptedLLM({"action": "type 'password: hunter2' into Search"})
        tools = browser_tools()
        with pytest.raises(ToolError, match="password"):
            build(tmp_path, llm, tools).execute(
                directive(goal="sign in", targets=["example.com"]))
        assert tools["type"].calls == []

    def test_the_action_limit_stops_a_loop_that_will_not_finish(
            self, allow_shop, tmp_path):
        llm = ScriptedLLM(*([{"action": "scroll down"}] * 50))
        tools = browser_tools()
        with pytest.raises(ToolError, match="action or time limit"):
            build(tmp_path, llm, tools).execute(
                directive(goal="scroll forever", targets=["example.com"],
                          max_steps=4))
        assert len(tools["press"].calls) <= 4

    def test_an_unusable_reply_costs_a_turn_rather_than_the_run(
            self, allow_shop, tmp_path):
        llm = ScriptedLLM({"why": "thinking"},        # no action, not done
                          {"done": True, "answer": "recovered"})
        result = build(tmp_path, llm, browser_tools()).execute(
            directive(goal="do something", targets=["example.com"]))
        assert "RESULT: recovered" in result

    def test_without_a_model_it_falls_back_to_the_scripted_path(
            self, allow_shop, tmp_path):
        # No model attached: it must still run the plan it was given rather
        # than silently doing nothing.
        tools = browser_tools()
        tool = ComputerUseTool(run_id="no-llm", tools=ToolManager(list(tools.values())),
                               root=tmp_path, llm=None)
        result = tool.execute(directive(
            goal="open the shop", targets=["example.com"],
            actions=["open the page"]))
        assert "Example shop" in result


# ---------------------------------------------------------------------------
# Desktop
# ---------------------------------------------------------------------------


class TestDesktopLoop:
    def test_exact_text_is_read_back_before_success(self, allow_shop, tmp_path):
        class ReadableDesktop(FakeBrowserTool):
            def _run(self, task, context=None):
                self.calls.append(task)
                if json.loads(task.removeprefix("TOOL_DIRECTIVE: "))["arguments"]["action"] == "read_window":
                    return "[desktop_native] 'Notepad' contains 19 character(s):\nDesktop test passed"
                return "ok"

        desktop = ReadableDesktop("desktop_native")
        llm = ScriptedLLM({"action": "type 'Desktop test passed'"},
                          {"done": True, "answer": "done"})
        result = ComputerUseTool(run_id="readback", tools=ToolManager([desktop]),
                                 root=tmp_path, llm=llm).execute(directive(
            goal="type exactly Desktop test passed without extra text, then read it back",
            targets=["notepad"]))
        assert "verified document text" in result
        assert any(call.endswith('"action": "read_window", "window": "notepad"}}')
                   for call in desktop.calls)

    def test_exact_text_mismatch_fails(self, allow_shop, tmp_path):
        desktop = FakeBrowserTool("desktop_native", "[desktop_native] Notepad:\nwrong text")
        llm = ScriptedLLM({"action": "type 'Desktop test passed'"},
                          {"done": True, "answer": "done"})
        with pytest.raises(ToolError, match="desktop text mismatch"):
            ComputerUseTool(run_id="bad-readback", tools=ToolManager([desktop]),
                            root=tmp_path, llm=llm).execute(directive(
                goal="type exactly Desktop test passed without extra text",
                targets=["notepad"]))

    def test_drawing_cannot_finish_without_saving(self, allow_shop, tmp_path, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "paint")
        desktop = FakeBrowserTool("desktop_native", "Untitled - Paint")
        llm = ScriptedLLM(
            {"action": "launch paint"},
            {"action": "click canvas"},
            {"done": True, "answer": "saved the picture"})
        tool = ComputerUseTool(run_id="unsaved", tools=ToolManager([desktop]),
                               root=tmp_path, llm=llm)
        with pytest.raises(ToolError, match="drawing was not saved"):
            tool.execute(directive(goal="draw a house and save it in Paint",
                                   targets=["paint"]))

    def test_save_keystroke_is_not_proof_of_a_saved_picture(self, allow_shop, tmp_path, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW", "paint")
        desktop = FakeBrowserTool("desktop_native", "Untitled - Paint")
        llm = ScriptedLLM(
            {"action": "launch paint"},
            {"action": "click canvas"},
            {"action": "save file"},
            {"done": True, "answer": "saved the picture"})
        tool = ComputerUseTool(run_id="not-on-disk", tools=ToolManager([desktop]),
                               root=tmp_path, llm=llm)
        with pytest.raises(ToolError, match="no newly saved PNG"):
            tool.execute(directive(goal="draw a house and save it as My_House.png in Paint",
                                   targets=["paint"]))

    def test_vision_model_receives_fresh_screen_each_turn(
            self, allow_shop, tmp_path, monkeypatch):
        desktop = FakeBrowserTool("desktop_native", "Untitled - Notepad")

        class VisionLLM(ScriptedLLM):
            def __init__(self):
                super().__init__({"action": "launch notepad"},
                                 {"done": True, "answer": "open"})
                self.seen_images = []

            def supports_images(self):
                return True

            def generate(self, prompt, **kwargs):
                self.seen_images.append(kwargs.get("images"))
                return super().generate(prompt, **kwargs)

        llm = VisionLLM()
        monkeypatch.setattr(ComputerUseTool, "_desktop_screenshot",
                            staticmethod(lambda: b"screen-png"))
        ComputerUseTool(run_id="vision-desk", tools=ToolManager([desktop]),
                        root=tmp_path, llm=llm).execute(
            directive(goal="open notepad", targets=["notepad"]))

        assert llm.seen_images == [[b"screen-png"], [b"screen-png"]]
        assert "screenshot is attached" in llm.prompts[0]

    def test_a_desktop_goal_is_driven_one_observed_action_at_a_time(
            self, allow_shop, tmp_path):
        desktop = FakeBrowserTool("desktop_native", "ok")
        llm = ScriptedLLM(
            {"action": "launch notepad"},
            {"action": "type 'the quarterly summary'"},
            {"action": "save file"},
            {"done": True, "answer": "written and saved"})
        tool = ComputerUseTool(run_id="desk", tools=ToolManager([desktop]),
                               root=tmp_path, llm=llm)
        result = tool.execute(directive(goal="write a summary in Notepad",
                                        targets=["notepad"]))

        specs = [a for a in desktop.arguments if a.get("action") != "windows"]
        assert specs == [
            {"action": "launch", "app": "notepad"},
            {"action": "type", "text": "the quarterly summary", "window": "notepad"},
            {"action": "key", "keys": "ctrl+s", "window": "notepad"}]
        assert "RESULT: written and saved" in result

    def test_the_windows_are_re_listed_after_the_app_is_launched(
            self, allow_shop, tmp_path):
        desktop = FakeBrowserTool("desktop_native", "2 window(s): Untitled - Notepad")
        llm = ScriptedLLM({"action": "launch notepad"},
                          {"done": True, "answer": "open"})
        ComputerUseTool(run_id="desk2", tools=ToolManager([desktop]),
                        root=tmp_path, llm=llm).execute(
            directive(goal="open notepad", targets=["notepad"]))
        assert [a for a in desktop.arguments if a.get("action") == "windows"]
        assert "Untitled - Notepad" in llm.prompts[1]

    def test_an_unsupported_instruction_is_refused_and_explained(
            self, allow_shop, tmp_path):
        desktop = FakeBrowserTool("desktop_native", "ok")
        llm = ScriptedLLM({"action": "delete everything"},
                          {"done": True, "answer": "stopped"})
        ComputerUseTool(run_id="desk3", tools=ToolManager([desktop]),
                        root=tmp_path, llm=llm).execute(
            directive(goal="tidy up", targets=["notepad"]))

        assert "unsupported desktop action" in llm.prompts[1]
        assert not [a for a in desktop.arguments if a.get("action") != "windows"]

    def test_a_credential_is_refused_before_it_is_typed(self, allow_shop, tmp_path):
        desktop = FakeBrowserTool("desktop_native", "ok")
        llm = ScriptedLLM({"action": "type 'api_key=sk-abcdefghijklmnop'"})
        with pytest.raises(ToolError, match="API key"):
            ComputerUseTool(run_id="desk4", tools=ToolManager([desktop]),
                            root=tmp_path, llm=llm).execute(
                directive(goal="configure it", targets=["notepad"]))
        assert not [a for a in desktop.arguments if a.get("action") == "type"]


# ---------------------------------------------------------------------------
# How the mode is chosen
# ---------------------------------------------------------------------------


class TestModeSelection:
    def test_the_planned_steps_run_first_then_the_loop_carries_on(
            self, allow_shop, tmp_path):
        # The plan is the best guess an agent could make before seeing the
        # page, so it runs. It is not the last word, because a plan that stops
        # one action short of the goal used to end the step there.
        llm = ScriptedLLM({"done": True, "answer": "already on the pricing page"})
        tools = browser_tools()
        result = build(tmp_path, llm, tools).execute(directive(
            goal="open the shop", targets=["example.com"],
            actions=["open the page", "click Pricing"]))

        assert tools["click"].arguments == [{"element": "Pricing", "ref": "l1"}]
        assert len(llm.prompts) == 1
        assert "click Pricing -> clicked" in llm.prompts[0]
        assert "RESULT: already on the pricing page" in result

    def test_a_plan_that_stops_short_is_finished_by_the_loop(
            self, allow_shop, tmp_path):
        # This is the failure the seeding exists for: the plan opens the page
        # and says nothing about the link the task actually asked for.
        llm = ScriptedLLM(
            {"action": "click Pricing", "why": "the plan never got here"},
            {"done": True, "answer": "followed the link"})
        tools = browser_tools()
        build(tmp_path, llm, tools).execute(directive(
            goal="open the shop and follow the pricing link",
            targets=["example.com"], actions=["open the page"]))
        assert tools["click"].arguments == [{"element": "Pricing", "ref": "l1"}]

    def test_autonomous_skips_the_planned_steps_entirely(self, allow_shop, tmp_path):
        llm = ScriptedLLM({"done": True, "answer": "improvised"})
        tools = browser_tools()
        build(tmp_path, llm, tools).execute(directive(
            goal="open the shop", targets=["example.com"],
            actions=["click Pricing"], autonomous=True))
        assert tools["click"].calls == []
        assert llm.prompts != []

    def test_the_preview_says_which_mode_a_person_is_approving(
            self, allow_shop, tmp_path):
        tools = browser_tools()
        steered = build(tmp_path, ScriptedLLM(), tools)
        improvised = steered.preview(directive(
            goal="find the price", targets=["example.com"], autonomous=True))
        seeded = steered.preview(directive(
            goal="open it", targets=["example.com"], actions=["open the page"]))
        unsteered = ComputerUseTool(
            run_id="no-llm-preview", tools=ToolManager(list(tools.values())),
            root=tmp_path, llm=None).preview(directive(
                goal="open it", targets=["example.com"], actions=["open the page"]))

        assert "works the steps out from what is on screen" in improvised
        assert "then keeps going" in seeded
        assert "stops" in unsteered


class TestTheModelGoingAway:
    """A provider outage must not throw away work that already happened.

    The approved steps have usually run by the time steering begins. Letting a
    503 propagate loses them and reports nothing, which is strictly worse than
    stopping early and saying why.
    """

    class Unreachable:
        def generate(self, *args, **kwargs):
            raise RuntimeError("HTTP 503 Service temporarily overloaded")

    def test_seeded_work_survives_an_unreachable_model(self, allow_shop, tmp_path):
        tools = browser_tools()
        tool = ComputerUseTool(run_id="outage",
                               tools=ToolManager(list(tools.values())),
                               root=tmp_path, llm=self.Unreachable())

        with pytest.raises(ToolError, match="browser task stopped before completion.*503"):
            tool.execute(directive(
                goal="open the shop and follow pricing", targets=["example.com"],
                actions=["open the page", "click Pricing"]))

        assert tools["click"].arguments == [{"element": "Pricing", "ref": "l1"}]

    def test_the_desktop_loop_stops_the_same_way(self, allow_shop, tmp_path):
        desktop = FakeBrowserTool("desktop_native", "ok")
        with pytest.raises(ToolError, match="model could not be reached.*503"):
            ComputerUseTool(run_id="outage-desk",
                            tools=ToolManager([desktop]),
                            root=tmp_path, llm=self.Unreachable()).execute(
                directive(goal="write a note", targets=["notepad"],
                          actions=["launch notepad"]))
        assert desktop.calls  # the launch is logged, but not misreported as writing


class TestStaleElementRetry:
    """A stale handle means nothing happened, so it is safe to try again.

    The distinction matters in both directions. Retrying a stale ref recovers
    an action that never ran; retrying a timeout could repeat a click that
    already landed. So only errors that name the element handle are retried.
    """

    class FlakyOnce(Tool):
        """Fails the first call with the given error, then behaves."""

        def __init__(self, name, error):
            super().__init__()
            self.name = name
            self.capability = name
            self.error = error
            self.calls = []

        def is_live(self):
            return True

        def _run(self, task, context=None):
            self.calls.append(task)
            if len(self.calls) == 1:
                raise ToolError(self.error)
            return "clicked"

    def test_a_stale_ref_is_retried_against_a_fresh_page(self, allow_shop, tmp_path):
        tools = browser_tools()
        tools["click"] = self.FlakyOnce(
            "web_browser_click",
            "No element found for aria-ref=l1; waiting for locator")
        llm = ScriptedLLM({"done": True, "answer": "done"})

        result = build(tmp_path, llm, tools).execute(directive(
            goal="follow pricing", targets=["example.com"],
            actions=["click Pricing"]))

        assert len(tools["click"].calls) == 2
        assert "after re-reading the page" in result

    def test_an_ordinary_failure_is_not_retried(self, allow_shop, tmp_path):
        # A click that timed out may well have landed. Doing it again could
        # submit a form twice.
        tools = browser_tools()
        tools["click"] = self.FlakyOnce("web_browser_click", "navigation timeout")
        llm = ScriptedLLM({"done": True, "answer": "done"})

        result = build(tmp_path, llm, tools).execute(directive(
            goal="follow pricing", targets=["example.com"],
            actions=["click Pricing"]))

        assert len(tools["click"].calls) == 1
        assert "navigation timeout" in result

    def test_a_failure_that_repeats_is_reported_honestly(self, allow_shop, tmp_path):
        class AlwaysStale(Tool):
            name = "web_browser_click"
            capability = "web_browser_click"

            def __init__(self):
                super().__init__()
                self.calls = []

            def is_live(self):
                return True

            def _run(self, task, context=None):
                self.calls.append(task)
                raise ToolError("no element found for aria-ref")

        tools = browser_tools()
        tools["click"] = AlwaysStale()
        result = build(tmp_path, ScriptedLLM({"done": True, "answer": "x"}),
                       tools).execute(directive(
            goal="follow pricing", targets=["example.com"],
            actions=["click Pricing"]))

        assert len(tools["click"].calls) == 2
        assert "failed twice" in result
