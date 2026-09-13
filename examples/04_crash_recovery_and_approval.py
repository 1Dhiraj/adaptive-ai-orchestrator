"""Persistence, crash recovery, and a human-in-the-loop approval gate.

    python examples/04_crash_recovery_and_approval.py
"""

# Run from anywhere without installing: put the project root on sys.path.
# (If you `pip install -e .` instead, this block is harmless.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile

from orchestrator import StateManager, Step, Workflow

db = str(Path(tempfile.gettempdir()) / "orchestrator_example.db")
Path(db).unlink(missing_ok=True)

steps = [
    Step(id="build", description="Build the application bundle.", agent_role="devops"),
    Step(id="stage", description="Deploy to the staging environment.", agent_role="devops",
         depends_on=["build"]),
    # Nothing past this point runs without a human saying so.
    Step(id="prod", description="Deploy to production.", agent_role="devops",
         depends_on=["stage"], requires_approval=True),
    Step(id="announce", description="Announce the release.", agent_role="writer",
         requires_tool="slack", depends_on=["prod"]),
]

# --- run 1: stops at the approval gate -------------------------------------

workflow = Workflow(steps, description="release pipeline", run_id="release-1",
                    state=StateManager(db), verbose=False)
report = workflow.run_full()
print(f"run 1: executed {report.executed}")
print(f"       waiting for approval: {workflow.awaiting_approval()}")
print(f"       not started yet     : {report.cancelled}")
workflow.close()

# --- the process dies here -------------------------------------------------

print("\n...process restarts...\n")

recovered = Workflow.resume_from("release-1", state=StateManager(db), verbose=False)
completed, outstanding = recovered.state.resumable("release-1")
print(f"recovered from disk: {len(completed)} step(s) already done -> {completed}")
print(f"                     {len(outstanding)} outstanding        -> {outstanding}")

# An operator approves the production deploy.
recovered.approve("prod")
report = recovered.resume()
print(f"\nrun 2: executed {report.executed}")
print(f"       reused   {report.reused}  <- no work repeated across the restart")

recovered.print_summary()
recovered.close()
