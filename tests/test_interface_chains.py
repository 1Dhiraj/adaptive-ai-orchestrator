"""Merging split browser/desktop steps back into one.

The invariant being defended is not tidiness. Browser and desktop state does
not survive a step boundary: every ``computer_use`` step opens its own page or
raises its own window. So a plan split into "navigate to the dropdown page"
then "select Option 2" does not just cost an extra model call -- the second
step re-navigates, and whatever the first step did is gone before it starts.

Equally, merging the wrong pair is worse than not merging, so most of these
tests are about when it must leave the plan alone.
"""

from __future__ import annotations

import pytest

from orchestrator.models import Step
from orchestrator.planner import merge_interface_chains


def step(step_id, description, tool="computer_use", depends_on=(), **kwargs):
    return Step(id=step_id, name=step_id, description=description,
                agent_role="generic", depends_on=list(depends_on),
                requires_tool=tool, **kwargs)


@pytest.fixture(autouse=True)
def allow(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_COMPUTER_USE_ALLOW",
                       "example.com, wikipedia.org, notepad")


class TestItMerges:
    def test_a_two_step_browser_chain_becomes_one(self):
        repairs = []
        steps = merge_interface_chains([
            step("open_page", "Navigate to example.com and confirm it loaded"),
            step("read_it", "Read the heading on example.com",
                 depends_on=["open_page"]),
        ], repairs)

        assert [s.id for s in steps] == ["open_page"]
        assert "Then: Read the heading" in steps[0].description
        assert repairs and "does not survive a step boundary" in repairs[0]

    def test_a_three_step_chain_collapses_entirely(self):
        steps = merge_interface_chains([
            step("a", "Open notepad"),
            step("b", "Type the summary into notepad", depends_on=["a"]),
            step("c", "Save the notepad document", depends_on=["b"]),
        ], [])
        assert [s.id for s in steps] == ["a"]
        assert "Type the summary" in steps[0].description
        assert "Save the notepad" in steps[0].description

    def test_later_steps_are_rewired_to_the_surviving_step(self):
        steps = merge_interface_chains([
            step("browse", "Open example.com"),
            step("extract", "Read the price from example.com", depends_on=["browse"]),
            step("report", "Write the PDF", tool="artifact_store",
                 depends_on=["extract"]),
        ], [])
        by_id = {s.id: s for s in steps}
        assert set(by_id) == {"browse", "report"}
        assert by_id["report"].depends_on == ["browse"]


class TestItLeavesThingsAlone:
    def test_a_fan_out_is_not_merged(self):
        # Two steps depend on the first, so collapsing one of them into it
        # would reorder work the plan deliberately made parallel.
        steps = merge_interface_chains([
            step("open", "Open example.com"),
            step("left", "Read the heading on example.com", depends_on=["open"]),
            step("right", "Read the footer on example.com", depends_on=["open"]),
        ], [])
        assert len(steps) == 3

    def test_a_step_with_two_dependencies_is_not_merged(self):
        steps = merge_interface_chains([
            step("one", "Open example.com"),
            step("two", "Open wikipedia.org"),
            step("join", "Compare both pages", depends_on=["one", "two"]),
        ], [])
        assert len(steps) == 3

    def test_browser_and_desktop_are_never_merged(self):
        # One computer_use call picks one backend, so a merged step would
        # silently drop half of what it was asked to do.
        steps = merge_interface_chains([
            step("web", "Open example.com and read the price"),
            step("app", "Type the price into notepad", depends_on=["web"]),
        ], [])
        assert len(steps) == 2

    def test_other_tools_are_untouched(self):
        steps = merge_interface_chains([
            step("write", "Create the source files", tool="workspace"),
            step("test", "Run the tests", tool="terminal", depends_on=["write"]),
        ], [])
        assert len(steps) == 2

    def test_a_step_that_asks_the_operator_something_is_not_merged(self):
        from orchestrator.inputs import InputRequest

        steps = merge_interface_chains([
            step("open", "Open example.com"),
            step("act", "Read example.com", depends_on=["open"],
                 inputs=[InputRequest(name="which", prompt="Which section?")]),
        ], [])
        assert len(steps) == 2

    def test_a_conditional_step_is_not_merged(self):
        steps = merge_interface_chains([
            step("open", "Open example.com"),
            step("maybe", "Read example.com", depends_on=["open"],
                 condition={"source": "open", "operator": "contains",
                            "value": "ok"}),
        ], [])
        assert len(steps) == 2

    def test_two_very_long_descriptions_are_left_alone(self):
        steps = merge_interface_chains([
            step("open", "Open example.com. " + "detail " * 90),
            step("read", "Read example.com. " + "detail " * 90,
                 depends_on=["open"]),
        ], [])
        assert len(steps) == 2
