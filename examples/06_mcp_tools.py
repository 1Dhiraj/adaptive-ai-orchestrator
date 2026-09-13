"""Give agents real MCP (Model Context Protocol) tools.

Connects to an actual MCP server over stdio, discovers what it exposes, and
lets workflow steps call those tools — with the same fallback routing the
built-in adapters get.

    python examples/06_mcp_tools.py

Uses the small local server in tests/fixtures so it runs offline. Point it at
a real server (filesystem, GitHub, your own) via .mcp.json — see
.mcp.json.example.
"""

# Run from anywhere without installing: put the project root on sys.path.
# (If you `pip install -e .` instead, this block is harmless.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging

from orchestrator import Step, Workflow
from orchestrator.tools.mcp import McpServerSpec

# The MCP SDK's server logs to stderr at INFO; quiet it for a clean demo.
logging.getLogger("mcp").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
FAKE_SERVER = str(ROOT / "tests" / "fixtures" / "fake_mcp_server.py")

# In real use this comes from .mcp.json; declared inline here so the example
# is self-contained.
spec = McpServerSpec(
    name="local_tools",
    transport="stdio",
    command=sys.executable,
    args=[FAKE_SERVER],
    # Sharing a built-in capability makes these tools fallback candidates for
    # the whole 'vcs' chain -- and gives them one too.
    capability="vcs",
)

steps = [
    Step(id="pick", description="Choose two integers to add.", agent_role="research"),
    Step(id="compute", description="Add the two integers using the tool.",
         agent_role="backend", requires_tool="add", depends_on=["pick"]),
    Step(id="announce", description="Report the result.",
         agent_role="writer", requires_tool="echo", depends_on=["compute"]),
]

workflow = Workflow(steps, description="MCP tools demo", run_id="mcp-example",
                    persist=False, verbose=False)

registered = workflow.attach_mcp(specs=[spec])
print(f"Discovered from the MCP server: {registered}\n")

print("What each tool tells the agent about calling it:")
for name in registered:
    tool = workflow.tools.get(name)
    hint = tool.prompt_hint().replace("\n", "\n      ")
    print(f"  {name}:\n      {hint}")

print("\nThe agent's prompt now carries that schema, so it can emit a valid call:")
from orchestrator.agents import Agent  # noqa: E402

prompt = Agent("backend").build_prompt(workflow.graph.get("compute"), "(none)", workflow.tools)
print("  " + prompt.split("## Tool")[1].strip().replace("\n", "\n  "))

print("\n--- calling a real MCP tool directly ---")
directive = 'Adding them now.\nTOOL_DIRECTIVE: ' + json.dumps({"arguments": {"a": 5, "b": 3}})
invocation = workflow.tools.use("add", directive)
print(f"  tool_used={invocation.tool_used}  simulated={invocation.simulated}")
print(f"  server returned: {invocation.output}")

print("\n--- fallback still applies to MCP tools ---")
workflow.tools.break_tool("add")
invocation = workflow.tools.use("add", "some payload")
print(f"  'add' broken -> routed to '{invocation.tool_used}' (fallback={invocation.used_fallback})")
workflow.tools.repair_tool("add")

print("\n--- running the workflow ---")
report = workflow.run_full()
print(f"  {len(report.executed)} steps executed, ok={report.ok}")
print(f"  announce step used tool: {workflow.results['announce'].tool_used}")
print(f"  its output tail: ...{workflow.results['announce'].output[-80:].strip()}")

workflow.close()   # shuts the MCP server subprocess down cleanly
print("\nMCP connections closed.")
