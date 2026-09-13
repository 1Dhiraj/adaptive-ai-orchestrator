"""Tool fallback, custom tools and failure injection.

Every capability chain ends in something that always works, so a workflow can
finish even when every remote service is down.

    python examples/03_tools_and_fallback.py
"""

# Run from anywhere without installing: put the project root on sys.path.
# (If you `pip install -e .` instead, this block is harmless.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator import Step, Workflow, default_tool_manager
from orchestrator.tools import Tool, ToolError

# ---------------------------------------------------------------------------
# A custom tool. Give it the capability of an existing chain and it becomes a
# fallback candidate for every tool in that chain automatically.
# ---------------------------------------------------------------------------


class GitLabTool(Tool):
    name = "gitlab"
    capability = "vcs"          # same capability as github / github_cli
    description = "Push to GitLab instead of GitHub"
    fallbacks = ["artifact_store"]

    def is_live(self) -> bool:
        return False            # no credentials wired up in this example

    def _run(self, task, context=None):
        if self.is_broken:
            raise ToolError("gitlab is down")
        return f"[simulated:gitlab] would push {len(task)} chars"


tools = default_tool_manager()
tools.register(GitLabTool())

print("Fallback chains:")
for entry in tools.describe():
    chain = " -> ".join([entry["name"]] + entry["fallbacks"]) or entry["name"]
    print(f"  {entry['capability']:<9} {chain}")

steps = [
    Step(id="plan", description="Plan the release.", agent_role="research"),
    Step(id="publish", description="Publish the release notes and tag the repo.",
         agent_role="devops", requires_tool="github", depends_on=["plan"]),
    Step(id="announce", description="Tell the team the release is out.",
         agent_role="writer", requires_tool="slack", depends_on=["publish"]),
]

workflow = Workflow(steps, description="tool fallback demo", run_id="tools-demo",
                    tool_manager=tools, verbose=False)
workflow.run_full()
print(f"\nhealthy run    -> publish used '{workflow.results['publish'].tool_used}'")

# Break the primary. The orchestrator re-runs only the steps that touch it.
workflow.handle_tool_failure("github")
print(f"github down    -> publish used '{workflow.results['publish'].tool_used}' "
      f"(fallback={workflow.results['publish'].used_fallback})")

# Break the whole remote chain: the local artifact store still lands the work.
workflow.tools.break_tool("github_cli")
workflow.tools.break_tool("gitlab")
workflow.handle_step_change("publish")
print(f"whole chain down -> publish used '{workflow.results['publish'].tool_used}'")
print(f"                    status: {workflow.status_of('publish').value}")

workflow.handle_tool_repair("github", rerun=True)
print(f"github repaired -> publish used '{workflow.results['publish'].tool_used}'")

print("\nTool statistics:", workflow.tools.stats())
workflow.close()
