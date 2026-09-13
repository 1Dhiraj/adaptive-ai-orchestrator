"""The n8n loop: it asks you for what it needs, then actually does the task.

  1. you give it a task
  2. it runs until it hits something only you can decide, and stops to ask
  3. you answer; it continues, using your answers in the prompt AND in the
     tool call itself
  4. before anything irreversible, it stops again and shows you exactly what
     it is about to do
  5. you approve; it acts -- without paying to regenerate the work

    python examples/08_ask_and_act.py
"""

# Run from anywhere without installing: put the project root on sys.path.
# (If you `pip install -e .` instead, this block is harmless.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator import Step, Workflow
from orchestrator.inputs import InputError, InputRequest, InputType

steps = [
    Step(id="analyse", description="Summarise this week's support tickets by theme.",
         agent_role="research"),
    Step(id="draft", description="Draft a customer-facing update on what we fixed.",
         agent_role="writer", depends_on=["analyse"]),
    Step(
        id="publish",
        description="Send the update to the customer mailing list.",
        agent_role="devops",
        requires_tool="email",
        depends_on=["draft"],
        inputs=[
            InputRequest(name="recipient", prompt="Which list should receive this?",
                         type=InputType.EMAIL,
                         why="the update is sent here; it cannot be recalled"),
            InputRequest(name="subject", prompt="Subject line?",
                         default="This week's product update"),
            InputRequest(name="tone", prompt="What tone?", type=InputType.CHOICE,
                         options=["formal", "friendly"], default="friendly"),
        ],
    ),
]

# action_approval="all" so the gate is visible without real credentials.
# The default, "live", only gates actions that would genuinely happen.
workflow = Workflow(steps, description="Weekly customer update", run_id="ask-and-act",
                    persist=False, verbose=False, action_approval="all")

print("TASK: send a weekly customer update\n")

# -- 1. run until it needs something ----------------------------------------
report = workflow.run_full()
print(f"Ran {len(report.executed)} step(s), then stopped.")
print(f"Waiting on you: {workflow.blocked_on_human()}\n")

print("It is asking for:")
for step_id, requests in workflow.input_form().items():
    for request in requests:
        flag = "required" if request.required and request.default is None else "optional"
        print(f"  {step_id}.{request.name:<10} [{flag}] {request.prompt}")
        if request.why:
            print(f"  {'':>22} why: {request.why}")

# -- 2. bad input is refused -------------------------------------------------
print("\nTrying an invalid answer:")
try:
    workflow.provide_input("publish", "recipient", "not-an-address")
except InputError as exc:
    print(f"  refused: {exc}")

# -- 3. answer properly ------------------------------------------------------
print("\nAnswering:")
workflow.provide_inputs(
    {"recipient": "customers@example.com", "tone": "formal"}, step_id="publish")
for name, value in workflow.graph.get("publish").input_values().items():
    print(f"  {name} = {value}")

report = workflow.resume()

# -- 4. it stops before doing the irreversible thing -------------------------
print(f"\nStopped again before acting: {report.awaiting_action}")
for step_id, action in workflow.pending_actions().items():
    print(f"\n  step        : {step_id}")
    print(f"  tool        : {action.tool}")
    print(f"  irreversible: {action.irreversible}")
    print(f"  what it does: {action.preview}")

# -- 5. approve, and it acts -------------------------------------------------
print("\nApproving...")
workflow.approve_action("publish")
report = workflow.resume()

print(f"  executed  : {report.executed}")
print(f"  LLM calls : {report.usage.calls}   <- 0: approving did not redo the work")
print(f"  status    : {workflow.status_of('publish').value}")
print(f"\nResult: {workflow.query_results('publish').splitlines()[-1]}")
print("Note the recipient reached the email tool itself, not just the prompt.")

# -- rejecting instead -------------------------------------------------------
print("\n--- what rejection looks like ---")
other = Workflow([Step(id="post", description="Post to Slack.", agent_role="devops",
                       requires_tool="slack")],
                 run_id="rejected", persist=False, verbose=False, action_approval="all")
other.run_full()
other.reject_action("post", reason="legal has not signed off")
print(f"  status: {other.status_of('post').value}")
print(f"  reason: {other.results['post'].error}")
print("  nothing was sent.")

workflow.close()
other.close()
