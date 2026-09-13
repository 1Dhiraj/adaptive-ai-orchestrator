"""Connect agents to any HTTP API, the way n8n credentials work.

Declare a service once; every agent can then call it by name. Credentials
come from environment variables and never appear in a prompt.

Makes REAL calls to httpbin.org, so you can see the auth header arrive.

    python examples/09_api_connections.py
"""

# Run from anywhere without installing: put the project root on sys.path.
# (If you `pip install -e .` instead, this block is harmless.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import re

from orchestrator import Step, Workflow
from orchestrator.connections import ApiConnection, ApiConnectionTool, AuthSpec
from orchestrator.tools import ToolManager

# The token would normally live in .env.local; set here so the demo is live.
os.environ.setdefault("DEMO_API_TOKEN", "demo-token-abc123")

connection = ApiConnection(
    name="httpbin",
    base_url="https://httpbin.org",
    description="HTTP testing service",
    auth=AuthSpec(type="bearer", token_env="DEMO_API_TOKEN"),
    headers={"X-Client": "adaptive-orchestrator"},
)

tool = ApiConnectionTool(connection)
print(f"connection : {tool.name}")
print(f"base url   : {connection.base_url}")
print(f"auth       : {connection.auth.type} via {', '.join(connection.required_env())}")
print(f"live       : {tool.is_live()}")

print("\nWhat the agent is told about it:")
print("  " + tool.prompt_hint().replace("\n", "\n  "))

# --- safety is decided per call, not per service ---------------------------
print("\nApproval gate, decided per HTTP method:")
for method in ["GET", "POST", "DELETE"]:
    payload = 'x\nTOOL_DIRECTIVE: ' + json.dumps({"method": method, "path": "/anything"})
    gated = "needs approval" if tool.is_irreversible(payload) else "runs freely"
    print(f"  {method:<7} {gated:<15} {tool.preview(payload)}")

# --- a real authenticated request ------------------------------------------
print("\nREAL GET https://httpbin.org/bearer:")
result = tool.execute('fetch\nTOOL_DIRECTIVE: {"method": "GET", "path": "/bearer"}')
print("  " + re.sub(r"\s+", " ", result)[:200])
print("  ^ httpbin confirms the bearer token arrived; the model never saw it.")

# --- inside a workflow ------------------------------------------------------
print("\n--- inside a workflow ---")
tools = ToolManager()
workflow = Workflow(
    [Step(id="lookup", description="Fetch the current request headers and summarise them.",
          agent_role="backend", requires_tool="api_httpbin")],
    run_id="api-demo", persist=False, verbose=False,
)
workflow.tools = tools
registered = workflow.attach_connections(connections=[connection])
print(f"registered: {registered}")

report = workflow.run_full()
print(f"ran {len(report.executed)} step(s), ok={report.ok}")
print(f"tool used: {workflow.results['lookup'].tool_used}")

print("\nRequirements report picked up the credential automatically:")
for requirement in workflow.check_requirements().requirements:
    if requirement.name == "DEMO_API_TOKEN":
        print(f"  {requirement.name}: {requirement.status.value} — {requirement.detail}")

print("\nEach API gets its own capability, so one service never silently")
print("falls back to an unrelated one:")
print(f"  fallback chain for api_httpbin = {tools.candidates_for('api_httpbin')}")

workflow.close()
