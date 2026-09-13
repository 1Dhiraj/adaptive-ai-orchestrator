"""High-level desktop workflow using the optional Hermes worker.

Run safely in simulation mode:
    python examples/11_hermes_desktop.py

For live use, install and configure Hermes first, then set
ORCHESTRATOR_ALLOW_DESKTOP=true. The workflow pauses for approval before the
desktop worker starts.
"""

from orchestrator import Step, Workflow


steps = [
    Step(
        id="prepare",
        name="Prepare report content",
        description=(
            "Prepare a short weekly project status report from the supplied workflow "
            "results. Include completed work, current problems and next actions."
        ),
        agent_role="project analyst",
    ),
    Step(
        id="format_in_word",
        name="Create the Word report",
        description=(
            "Open Microsoft Word, create a professional one-page status report using "
            "the prepared content, save it as weekly-status.docx in the workflow "
            "workspace, and confirm the saved file exists."
        ),
        agent_role="desktop operator",
        requires_tool="hermes_desktop",
        depends_on=["prepare"],
    ),
]

workflow = Workflow(
    steps,
    description="Prepare a weekly status report in Microsoft Word",
    action_approval="all",  # demonstrate the gate even when Hermes is simulated
)

first = workflow.run_full()
workflow.print_summary()

if first.awaiting_action:
    print("\nDesktop work is waiting for approval:")
    for action in workflow.pending_actions():
        print(f"- {action['preview']}")
    print(f"\nApprove in Python with: workflow.approve_action('{first.awaiting_action[0]}')")
    print("Then continue with: workflow.resume()")

workflow.close()
