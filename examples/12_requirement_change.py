#!/usr/bin/env python3
"""A requirement changes halfway, and only the affected work re-runs.

This is the hidden-dependency case, the one plain dependency tracking gets
wrong. Four steps:

    requirements -> schema    (decides the database)
                 -> backend   (writes code *against* that database)
                 -> frontend  (never touches it)
                    tests     (depends on backend + frontend)

``backend`` has no edge to ``schema``. Nothing it depends on changes when the
database does, so a signature that only covers dependency outputs says it is
still valid -- and you are left with backend code full of MongoDB calls
pointing at a PostgreSQL database.

Assumption tracking fixes that: ``backend`` declared it relied on
``database.engine``, so changing that fact reaches it regardless of edges.

Runs on the offline model, so no API key is needed and the result is the
same every time.
"""

# Run from anywhere without installing: put the project root on sys.path.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator import DependencyGraph, Step, Workflow
from orchestrator.change_aware import Fact, FactStore


def build() -> Workflow:
    graph = DependencyGraph([
        Step(id="requirements", description="Agree the feature list",
             agent_role="analyst"),
        Step(id="schema", description="Design the database schema",
             agent_role="database", depends_on=["requirements"],
             output_type="code", assumes=["database.engine"]),
        # Deliberately no dependency on 'schema': this is the hidden link.
        Step(id="backend", description="Implement the API",
             agent_role="backend", depends_on=["requirements"],
             output_type="code", assumes=["database.engine"]),
        Step(id="frontend", description="Build the interface",
             agent_role="frontend", depends_on=["requirements"],
             output_type="code"),
        Step(id="tests", description="Integration tests",
             agent_role="testing", depends_on=["backend", "frontend"]),
    ])
    graph.facts = FactStore([
        Fact(key="database.engine", value="MongoDB", owner="schema",
             aliases=["mongo", "mongodb"]),
    ])
    return Workflow(graph=graph, description="Build a task management app",
                    run_id="example-12", change_aware=True,
                    persist=False, verbose=False)


def main() -> int:
    workflow = build()

    print("=" * 70)
    print("1. First run")
    print("=" * 70)
    workflow.run()
    for step in workflow.graph:
        print(f"  {step.id:<14}{workflow.results[step.id].status.value}")

    print()
    print("=" * 70)
    print('2. The client says: "use PostgreSQL instead of MongoDB"')
    print("=" * 70)
    proposal = workflow.prepare_change("use PostgreSQL instead of MongoDB")

    if proposal["kind"] != "facts":
        print("  The request did not match a known requirement, so the plan")
        print("  would be revised instead. Diff:", proposal["diff"])
        return 1

    for key, new in sorted(proposal["delta"].items()):
        print(f"  {key}: {proposal['old_values'][key]} -> {new}")

    print()
    print("  Affected, and why:")
    for step_id in proposal["affected"]:
        for reason in proposal["reasons"].get(step_id, []):
            print(f"    {step_id:<14}{reason}")

    reused = [s.id for s in workflow.graph if s.id not in set(proposal["affected"])]
    print(f"\n  Reused unchanged: {', '.join(reused) or '(none)'}")
    print("\n  Note 'backend' above: it has no edge to 'schema', so dependency")
    print("  tracking alone would have called it still valid and left stale")
    print("  MongoDB code behind.")

    print()
    print("=" * 70)
    print("3. Applying the change (a person confirms first)")
    print("=" * 70)
    workflow.apply_change(proposal, confirmed=True, rerun=True)
    for step in workflow.graph:
        print(f"  {step.id:<14}{workflow.results[step.id].status.value}")

    workflow.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
