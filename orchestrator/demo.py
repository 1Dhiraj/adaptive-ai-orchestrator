"""The four headline scenarios, end to end.

Run with ``python -m orchestrator.cli demo`` (or ``python main.py``).
"""

from __future__ import annotations

from typing import Optional

from .config import settings
from .models import Step
from .workflow import Workflow

WEB_APP_STEPS = [
    Step(
        id="frontend",
        name="Build Frontend",
        description="Create the React UI: a login form, a Kanban board with drag-and-drop "
                    "columns, and a task detail panel.",
        agent_role="frontend",
    ),
    Step(
        id="backend",
        name="Build API",
        description="Design and implement the REST API for authentication, task CRUD and "
                    "board operations.",
        agent_role="backend",
        requires_tool="github",
        depends_on=["frontend"],
    ),
    Step(
        id="database",
        name="Design Schema",
        description="Design the PostgreSQL schema for users, tasks and boards, with indexes "
                    "and foreign keys, as executable SQL.",
        agent_role="database",
        requires_tool="postgres",
        depends_on=["backend"],
    ),
    Step(
        id="testing",
        name="Integration Tests",
        description="Write and run integration tests covering auth, task CRUD and board "
                    "operations end to end.",
        agent_role="testing",
        requires_tool="ci",
        depends_on=["database"],
    ),
]


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run_demo(export_html: Optional[str] = "dashboard.html",
             run_id: Optional[str] = None) -> int:
    config = settings.describe()
    print("Adaptive AI Task Orchestrator - scenario walkthrough")
    print(f"LLM: {config['llm_mode']}\n")

    workflow = Workflow(
        WEB_APP_STEPS,
        description="Task management web app with authentication and a Kanban board",
        run_id=run_id or "demo",
    )

    _rule("SCENARIO 1 - full run")
    first = workflow.run_full()

    _rule("SCENARIO 2 - requirement change: 'use MongoDB instead of PostgreSQL'")
    print("Impact analysis before running anything:")
    impact = workflow.impact_of("database")
    print(f"  changed  : {impact['changed']}")
    print(f"  affected : {impact['affected']}")
    print(f"  reusable : {impact['reusable']}")
    change = workflow.handle_step_change(
        "database",
        new_description="Design the MongoDB collections for users, tasks and boards, with "
                        "indexes and embedded document choices, replacing the PostgreSQL schema.",
    )

    _rule("SCENARIO 3 - tool failure: the GitHub API goes down")
    failure = workflow.handle_tool_failure("github")

    _rule("SCENARIO 4 - the GitHub API comes back")
    workflow.handle_tool_repair("github")
    print(f"  broken tools now: {workflow.tools.broken_tools() or 'none'}")

    workflow.print_summary()

    _rule("EFFICIENCY")
    total = len(workflow.graph)
    for label, report in (("full run", first), ("requirement change", change),
                          ("tool failure", failure)):
        if report is None:
            continue
        print(f"  {label:<20} executed {len(report.executed)}/{total}, "
              f"reused {len(report.reused)}/{total} "
              f"({report.reuse_ratio * 100:.0f}% of steps avoided)")
    naive = 3 * total
    actual = sum(len(r.executed) for r in (first, change, failure) if r)
    print(f"\n  A framework that restarts the whole workflow on every event would have run "
          f"{naive} steps.\n  This system ran {actual} "
          f"({(1 - actual / naive) * 100:.0f}% fewer step executions).")

    print(f"\njson  -> {workflow.export_state(format='json', path='artifacts/results.json')}")
    print(f"csv   -> {workflow.export_state(format='csv', path='artifacts/results.csv')}")
    if export_html:
        print(f"html  -> {workflow.export_state(format='html', path=export_html)}")
    print(f"\nresume this run with: python -m orchestrator.cli resume {workflow.run_id}")
    print("live dashboard      : python -m orchestrator.cli serve")

    workflow.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo())
