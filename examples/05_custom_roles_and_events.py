"""Custom agent roles, and subscribing to the event stream.

    python examples/05_custom_roles_and_events.py
"""

# Run from anywhere without installing: put the project root on sys.path.
# (If you `pip install -e .` instead, this block is harmless.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter

from orchestrator import AgentManager, EventBus, EventType, Step, Workflow

# ---------------------------------------------------------------------------
# 1. Teach the system a role it does not ship with.
# ---------------------------------------------------------------------------

agents = AgentManager()
agents.add(
    "compliance_officer",
    "You are a compliance officer specialising in GDPR and SOC 2. You deliver a "
    "concrete control list, the evidence required for each control, and who owns it.",
)
agents.add(
    "cost_analyst",
    "You are a cloud cost analyst. You deliver a line-item cost estimate with "
    "assumptions stated, and name the single biggest lever for reducing spend.",
)

# ---------------------------------------------------------------------------
# 2. Subscribe to events. Everything observable flows through the bus, so this
#    is the same mechanism the dashboard and the SQLite recorder use.
# ---------------------------------------------------------------------------

bus = EventBus()
counts: Counter = Counter()
timeline = []


def observer(event) -> None:
    counts[event.type.value] += 1
    if event.type is EventType.STEP_FINISHED:
        timeline.append((event.step_id, event.data.get("duration_s"),
                         event.data.get("usage", {}).get("total_tokens")))
    elif event.type is EventType.TOOL_FALLBACK:
        print(f"  !! fallback: {event.message}")


bus.subscribe(observer)

steps = [
    Step(id="design", description="Design a multi-tenant SaaS architecture on AWS.",
         agent_role="research"),
    Step(id="compliance", description="Identify the compliance controls this design needs.",
         agent_role="compliance_officer", depends_on=["design"]),
    Step(id="cost", description="Estimate the monthly cloud cost of this design at 10k users.",
         agent_role="cost_analyst", depends_on=["design"]),
    Step(id="review", description="Review the design against the compliance and cost findings.",
         agent_role="reviewer", depends_on=["compliance", "cost"]),
]

workflow = Workflow(steps, description="architecture review", run_id="roles-demo",
                    agent_manager=agents, bus=bus, verbose=False)
workflow.run_full()

print("compliance + cost run concurrently:",
      workflow.graph.execution_levels()[1])

print("\nper-step results:")
for step_id, duration, tokens in timeline:
    print(f"  {step_id:<12} {duration:>6.2f}s  {tokens:>5} tokens")

print("\nevent counts:")
for event_type, count in sorted(counts.items()):
    print(f"  {event_type:<20} {count}")

print("\ncompliance officer said:")
print("  " + "\n  ".join(workflow.query_results("compliance").splitlines()[:6]))

workflow.close()
