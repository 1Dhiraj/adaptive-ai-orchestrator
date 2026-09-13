"""Smallest useful program: describe a project, run it, see the result.

    python examples/01_quickstart.py
"""

# Run from anywhere without installing: put the project root on sys.path.
# (If you `pip install -e .` instead, this block is harmless.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator import Workflow

workflow = Workflow.from_description(
    "Build a task management web app with user authentication and a Kanban board",
    run_id="quickstart",
)

workflow.run_full()
workflow.print_summary()

print("\nWhat could run in parallel:")
for depth, level in enumerate(workflow.graph.execution_levels()):
    print(f"  level {depth}: {', '.join(level)}")

print(f"\nHTML report: {workflow.export_state('artifacts/quickstart.html', format='html')}")
workflow.close()
