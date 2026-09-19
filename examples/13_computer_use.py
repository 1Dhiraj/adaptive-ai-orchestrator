#!/usr/bin/env python3
"""Driving a browser or the desktop, and every rule that bounds it.

Nothing here touches a real screen. computer_use is off unless
ORCHESTRATOR_ALLOW_DESKTOP is set, so this runs in simulation and prints
what *would* happen -- which is the point: you can see the guards work
before granting anything.

The guards, in the order they fire:

  1. off unless explicitly enabled
  2. the step must name the sites or apps it will touch
  3. those must be in the allow-list, which permits nothing when unset
  4. instructions containing a password, card number or key are refused
     outright, before the backend sees them
  5. action and time limits bound a run that goes wrong
  6. every action is appended to a log in the run's workspace

Because the tool is irreversible, the approval gate always runs first and a
person sees the plan before anything moves.
"""

# Run from anywhere without installing: put the project root on sys.path.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os

from orchestrator.tools import ToolManager
from orchestrator.tools.base import Tool, ToolError
from orchestrator.tools.computer_use import ComputerUseTool


class PretendBrowser(Tool):
    """Stands in for a Playwright MCP server so the example needs no setup."""

    name = "web_browser_navigate"
    capability = "browser"

    def is_live(self) -> bool:
        return True

    def _run(self, task, context=None):
        return f"(pretend browser) {task}"


def directive(**arguments) -> str:
    return "work on the page\nTOOL_DIRECTIVE: " + json.dumps({"arguments": arguments})


def show(label: str, fn) -> None:
    print(f"\n{label}")
    try:
        print("  " + str(fn()).replace("\n", "\n  "))
    except ToolError as exc:
        print(f"  REFUSED: {exc}")


def main() -> int:
    # Keep the example honest: simulation only, whatever the environment says.
    os.environ.pop("ORCHESTRATOR_ALLOW_DESKTOP", None)
    os.environ["ORCHESTRATOR_COMPUTER_USE_ALLOW"] = "example.com, notepad"

    tools = ToolManager([PretendBrowser()])
    tool = ComputerUseTool(run_id="example-13", tools=tools)

    print("=" * 70)
    print("Backend selection")
    print("=" * 70)
    print(f"  chosen backend : {tool.backend()}  (browser preferred over desktop)")
    print(f"  live           : {tool.is_live()}  (off until you opt in)")
    print(f"  irreversible   : {tool.irreversible}  (so approval runs first)")

    print()
    print("=" * 70)
    print("What a person approves before anything happens")
    print("=" * 70)
    print(tool.preview(directive(
        goal="Read the main heading",
        targets=["example.com"],
        actions=["open example.com", "read the first heading"])))

    print()
    print("=" * 70)
    print("The guards")
    print("=" * 70)

    show("1. A site that is not on the allow-list:",
         lambda: tool.execute(directive(
             goal="poke around", targets=["evil.test"], actions=["open it"])))

    show("2. No target named at all:",
         lambda: tool.execute(directive(goal="do something", actions=["click about"])))

    show("3. Instructions containing a password:",
         lambda: tool.execute(
             "sign in with password: hunter2\n" + directive(
                 goal="sign in", targets=["example.com"], actions=["type it"])))

    show("4. A card number:",
         lambda: tool.execute(
             "pay using 4111 1111 1111 1111\n" + directive(
                 goal="pay", targets=["example.com"], actions=["checkout"])))

    show("5. An allowed request, with the tool still disabled:",
         lambda: tool.execute(directive(
             goal="Read the main heading",
             targets=["example.com"],
             actions=["open example.com", "read the first heading"])))

    print()
    print("=" * 70)
    print("To use it for real")
    print("=" * 70)
    print("  ORCHESTRATOR_ALLOW_DESKTOP=1")
    print("  ORCHESTRATOR_COMPUTER_USE_ALLOW=example.com,notepad")
    print("  plus a Playwright MCP server in .mcp.json for browser work,")
    print("  or the Hermes agent installed for desktop work.")
    print("  Every action is logged to workspace/<run_id>/computer_use.log.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
