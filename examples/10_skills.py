"""Skills: reference material an agent follows, matched by role or keyword.

Different from a tool or an MCP server: a skill doesn't give an agent new
*capabilities*, it gives it a better *procedure* for what it can already do
-- your house style, your API conventions, your postmortem format.

    python examples/10_skills.py
"""

# Run from anywhere without installing: put the project root on sys.path.
# (If you `pip install -e .` instead, this block is harmless.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator import Step, Workflow
from orchestrator.skills import Skill, SkillLibrary

steps = [
    Step(id="build_api", description="Implement the REST API for order management.",
         agent_role="backend"),
    Step(id="write_copy", description="Write the changelog entry for this release.",
         agent_role="writer"),
    Step(id="postmortem", description="Write up yesterday's outage.",
         agent_role="devops"),
]

workflow = Workflow(steps, description="skills demo", run_id="skills-demo",
                    persist=False, verbose=False)

# Load the shipped examples in skills/ -- or build a library in code instead:
loaded = workflow.attach_skills("skills")
print(f"Loaded skills: {loaded}\n")

# --- see what would apply, before running anything --------------------------
print("What applies to each step:")
for step in workflow.graph:
    matched = [s.name for s in workflow.skills.match(step)]
    print(f"  {step.id:<12} ({step.agent_role:<9}) -> {matched or 'nothing'}")

# --- run it: the matched skill is folded into that step's prompt ------------
print("\nRunning...")
workflow.run_full()

prompt = workflow.agents.get("backend").build_prompt(
    workflow.graph.get("build_api"), "(none)", workflow.tools,
    workflow.skills.render_for(workflow.graph.get("build_api")))
print("\nWhat the backend agent's prompt actually contained:")
print("  " + prompt.split("## Reference material")[1][:220].replace("\n", "\n  "))

# --- skills can be built in code too, not just loaded from files -----------
print("\n--- an inline skill, no file needed ---")
inline = SkillLibrary([
    Skill(name="pci-scope", roles=["backend"],
          content="Never log full card numbers. Mask to the last 4 digits. "
                  "Any code touching card data must call redact_pan() first."),
])
step = workflow.graph.get("build_api")
print(f"  matches backend: {[s.name for s in inline.match(step)]}")
print(f"  matches devops : {[s.name for s in inline.match(workflow.graph.get('postmortem'))]}")

workflow.close()
