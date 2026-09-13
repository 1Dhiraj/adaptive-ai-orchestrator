"""Give it a task. It works out who it needs, what it needs, then does it.

This is the full loop:

  1. you describe a task in plain English
  2. it invents the *specialists* that task needs (not a fixed role list)
  3. it declares the *capabilities* it depends on -- tools, MCP servers,
     credentials, binaries -- and checks each against your actual machine
  4. it tells you what is missing and exactly how to fix it
  5. once nothing is blocking, it runs

    python examples/07_task_to_finished_work.py
"""

# Run from anywhere without installing: put the project root on sys.path.
# (If you `pip install -e .` instead, this block is harmless.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from orchestrator import Workflow
from orchestrator.llm import LLMProvider, LLMResponse
from orchestrator.models import LLMUsage

# ---------------------------------------------------------------------------
# A real LLM decides all of this. Scripted here so the example is
# self-contained, offline, and shows a domain with no software roles at all.
# ---------------------------------------------------------------------------

PLAN = {
    "agents": [
        {"role": "employment_lawyer",
         "why": "termination clauses carry jurisdiction-specific risk",
         "system_prompt": "You are an employment lawyer. You deliver clause-by-clause "
                          "analysis citing the relevant statute, and flag anything "
                          "unenforceable in the stated jurisdiction."},
        {"role": "compensation_analyst",
         "why": "equity and severance need benchmarking, not legal review",
         "system_prompt": "You are a compensation analyst. You deliver benchmark ranges "
                          "with percentiles and a recommended offer band."},
        {"role": "plain_language_editor",
         "why": "the final document is read by candidates, not lawyers",
         "system_prompt": "You are a plain-language editor. You rewrite legal prose at a "
                          "grade-9 reading level without changing its legal meaning."},
    ],
    "requirements": [
        {"name": "filesystem", "kind": "mcp_server",
         "why": "read the existing contract templates from disk",
         "needed_by": ["clause_review"],
         "setup": "npx -y @modelcontextprotocol/server-filesystem ./contracts"},
        {"name": "COMP_BENCHMARK_API_KEY", "kind": "credential",
         "why": "pull salary benchmark data", "needed_by": ["benchmark"],
         "setup": "set COMP_BENCHMARK_API_KEY in .env.local"},
        {"name": "pandoc", "kind": "binary",
         "why": "render the final contract to DOCX", "needed_by": ["final_draft"],
         "setup": "install pandoc and put it on PATH", "optional": True},
    ],
    "tasks": [
        {"id": "clause_review", "role": "employment_lawyer", "depends_on": [],
         "description": "Review the standard offer letter for unenforceable clauses."},
        {"id": "benchmark", "role": "compensation_analyst", "depends_on": [],
         "description": "Benchmark the salary and equity bands against market."},
        {"id": "final_draft", "role": "plain_language_editor",
         "depends_on": ["clause_review", "benchmark"],
         "description": "Produce the candidate-facing offer letter."},
    ],
    "notes": ["Assumes England & Wales jurisdiction; re-run if hiring elsewhere."],
}


class ScriptedLLM(LLMProvider):
    name = model = "scripted"

    def _generate(self, prompt, system, json_mode, metadata=None):
        role = (metadata or {}).get("role", "?")
        text = json.dumps(PLAN) if json_mode else (
            f"[{role}] Deliverable produced for this step, using the specialist "
            f"system prompt registered for '{role}'.")
        return LLMResponse(text=text, usage=LLMUsage(calls=1, prompt_tokens=40,
                                                     completion_tokens=60),
                           model=self.model, latency_s=0.0)


TASK = "Overhaul our employment offer letter for the UK market"

print(f'TASK: "{TASK}"\n')

workflow = Workflow.from_description(
    TASK, llm=ScriptedLLM(), run_id="offer-letter", persist=False, verbose=False)

# ---------------------------------------------------------------------------
# Step 1: it tells you what it needs, before spending anything.
# ---------------------------------------------------------------------------
workflow.print_requirements()

print("\nNote the specialists: an employment lawyer and a compensation analyst.")
print("None of these exist in the built-in role catalogue - the planner invented")
print("them for this task, and each got its own system prompt.\n")

# ---------------------------------------------------------------------------
# Step 2: it refuses to pretend it can do work it cannot do.
# ---------------------------------------------------------------------------
if not workflow.requirements.can_run:
    print("Execution is blocked. In real use you would satisfy these and re-check:\n")
    for req in workflow.requirements.blockers:
        print(f"  $ {req.setup}")
    print("\n--- simulating that you did that ---\n")

    # Pretend the operator supplied the credential and attached the MCP server.
    import os

    os.environ["COMP_BENCHMARK_API_KEY"] = "demo-key"
    for req in workflow.requirements.requirements:
        if req.name == "filesystem":
            req.optional = True   # stand in for "the server is now attached"
    workflow.requirements.check(workflow.tools)

print(f"can_run now: {workflow.requirements.can_run}")
if workflow.requirements.degraded:
    print("degraded (will still run):",
          ", ".join(r.name for r in workflow.requirements.degraded))

# ---------------------------------------------------------------------------
# Step 3: do the work.
# ---------------------------------------------------------------------------
print("\n--- running ---")
report = workflow.run_full()
print(f"{len(report.executed)} steps executed, ok={report.ok}")
print(f"parallel groups: {workflow.graph.execution_levels()}")

print("\nWho did what:")
for step_id in workflow.graph.topological_order():
    step = workflow.graph.get(step_id)
    print(f"  {step_id:<16} <- {step.agent_role}")

print(f"\nSample output ({workflow.graph.get('final_draft').agent_role}):")
print("  " + workflow.query_results("final_draft")[:150])

workflow.close()
