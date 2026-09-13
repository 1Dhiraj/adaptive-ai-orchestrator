"""The headline behaviour: change one thing, re-run only what that broke.

Shows three kinds of change and what each one costs:

  1. a mid-graph change   -> the changed step and its downstream cone
  2. a leaf change        -> just that leaf
  3. a re-run that reproduces the same output -> propagation stops immediately

    python examples/02_adaptive_reexecution.py
"""

# Run from anywhere without installing: put the project root on sys.path.
# (If you `pip install -e .` instead, this block is harmless.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator import Step, Workflow

steps = [
    Step(id="requirements", description="Clarify scope and acceptance criteria.",
         agent_role="research"),
    Step(id="database", description="Design the PostgreSQL schema for users, tasks and boards.",
         agent_role="database", requires_tool="postgres", depends_on=["requirements"]),
    Step(id="frontend", description="Build the React UI: login form and Kanban board.",
         agent_role="frontend", depends_on=["requirements"]),
    Step(id="backend", description="Implement the REST API for auth, tasks and boards.",
         agent_role="backend", requires_tool="github", depends_on=["database"]),
    Step(id="testing", description="Write and run integration tests across the stack.",
         agent_role="testing", requires_tool="ci", depends_on=["backend", "frontend"]),
]

workflow = Workflow(steps, description="adaptive re-execution demo", run_id="adaptive-demo",
                    verbose=False)

total = len(workflow.graph)
first = workflow.run_full()
print(f"full run: {len(first.executed)} steps executed\n")


def show(label: str, report) -> None:
    print(f"{label}")
    print(f"  re-ran : {sorted(report.executed)}")
    print(f"  reused : {sorted(report.reused)}")
    print(f"  avoided {len(report.reused)}/{total} steps "
          f"({report.reuse_ratio * 100:.0f}%), {report.usage.total_tokens} tokens spent\n")


# Impact analysis costs nothing — no agent runs, no tokens.
impact = workflow.impact_of("database")
print(f"if 'database' changes -> affected {impact['affected']}, reusable {impact['reusable']}\n")

show("1. mid-graph change (switch to MongoDB)", workflow.handle_step_change(
    "database",
    new_description="Design MongoDB collections for users, tasks and boards instead of tables."))

show("2. leaf change (different testing strategy)", workflow.handle_step_change(
    "testing",
    new_description="Write property-based tests with Hypothesis instead of example-based tests."))

# No new description: the step re-runs, produces exactly what it did before,
# and the signature match stops the change from propagating any further.
show("3. re-run that changes nothing", workflow.handle_step_change("requirements"))

naive = 3 * total
actual = sum(len(r.executed) for r in workflow.reports[1:])
print(f"A framework that restarts everything on each change would have run {naive} steps.")
print(f"This ran {actual} ({(1 - actual / naive) * 100:.0f}% fewer).")

workflow.close()
