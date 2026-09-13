"""Command-line interface.

    python -m orchestrator.cli plan   "Build a task manager with auth"
    python -m orchestrator.cli run    "Build a task manager with auth" --html report.html
    python -m orchestrator.cli demo
    python -m orchestrator.cli resume <run_id>
    python -m orchestrator.cli runs
    python -m orchestrator.cli mcp                      # list MCP server tools
    python -m orchestrator.cli run "..." --mcp          # give agents those tools
    python -m orchestrator.cli serve --port 8000
    python -m orchestrator.cli bench
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import settings
from .graph import DependencyGraph
from .llm import LLMError, StubProvider, build_provider, get_provider, set_provider
from .planner import TaskPlanner, plan_from_json
from .state import StateManager
from .workflow import Workflow


def _banner() -> None:
    config = settings.describe()
    print("Adaptive AI Task Orchestrator")
    print(f"  llm    : {config['llm_mode']} ({config['model']})")
    print(f"  workers: {config['max_workers']}")
    print(f"  db     : {config['db_path']}")
    if config["live_tools"]:
        print(f"  live tools: {', '.join(config['live_tools'])}")
    else:
        print("  live tools: none configured (all tools run simulated)")
    print()


def _maybe_force_stub(args: argparse.Namespace) -> None:
    """Resolve --stub / --provider into the process-wide provider.

    --stub always wins (useful to force reproducibility even if a key is
    set); otherwise --provider picks a specific one, bypassing
    auto-detection; otherwise the default precedence in
    Settings.resolve_provider() applies untouched.
    """
    if getattr(args, "stub", False):
        set_provider(StubProvider())
        return
    provider_name = getattr(args, "provider", None)
    if provider_name:
        try:
            set_provider(build_provider(provider_name))
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc


def _ask(request: Any) -> Any:
    """Prompt for one input on the terminal until a valid answer is given."""
    import getpass

    from .inputs import InputError, InputType

    print(f"\n  {request.prompt}")
    if request.why:
        print(f"    ({request.why})")
    hint = request.describe().split("(", 1)[-1].rstrip(")")
    while True:
        try:
            raw = (getpass.getpass(f"    {request.name} [{hint}]: ")
                   if request.type is InputType.SECRET
                   else input(f"    {request.name} [{hint}]: "))
        except (EOFError, KeyboardInterrupt):
            print()
            raise
        try:
            return request.provide(raw)
        except InputError as exc:
            print(f"    ! {exc}")


def _collect_inputs(workflow: Workflow) -> int:
    """Ask for every outstanding input. Returns how many were answered.

    Offers the whole form for a blocked step, not only the blocking fields,
    so optional values and defaults can be set too -- press Enter to accept.
    """
    answered = 0
    for step_id, requests in workflow.input_form().items():
        step = workflow.graph.get(step_id)
        print(f"\n  Step '{step_id}' ({step.name}) needs:")
        for request in requests:
            if request.provided:
                continue
            value = _ask(request)
            workflow.provide_input(step_id, request.name, value)
            answered += 1
    return answered


def _review_actions(workflow: Workflow, auto_approve: bool = False) -> int:
    """Show each held action and ask whether to perform it."""
    approved = 0
    for step_id, action in workflow.pending_actions().items():
        tool = workflow.tools.get(action.tool)
        is_real = tool is not None and tool.is_live()
        headline = ("About to perform a REAL, irreversible action:" if is_real else
                    "Approval rehearsal - this tool is SIMULATED, nothing will actually happen:")
        print(f"\n  {headline}")
        print(f"    step : {step_id}")
        print(f"    tool : {action.tool}")
        print(f"    what : {action.preview}")
        print(f"\n    payload preview:")
        for line in action.payload.strip().splitlines()[:8]:
            print(f"      {line[:96]}")

        if auto_approve:
            workflow.approve_action(step_id)
            approved += 1
            continue
        try:
            answer = input("\n    perform it? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            answer = "n"
        if answer in {"y", "yes"}:
            workflow.approve_action(step_id)
            approved += 1
        else:
            workflow.reject_action(step_id, reason="declined at the prompt")
            print("    skipped.")
    return approved


def _run_interactively(workflow: Workflow, auto_approve: bool = False,
                       max_rounds: int = 12) -> Any:
    """Run, stopping to ask for anything only a human can supply.

    Loops because answering one question can unblock a step whose output
    reveals the next question -- which is exactly how n8n-style workflows
    behave in practice.
    """
    report = workflow.run_full()
    for _ in range(max_rounds):
        blocked = workflow.blocked_on_human()
        if not any(blocked.values()):
            break

        if blocked["inputs"]:
            print("\n" + "-" * 70)
            print("  The workflow needs some information from you.")
            print("-" * 70)
            _collect_inputs(workflow)

        for step_id in blocked["approvals"]:
            step = workflow.graph.get(step_id)
            print(f"\n  Step '{step_id}' is gated: {step.description[:100]}")
            answer = input("    approve and continue? [y/N]: ").strip().lower()
            if answer in {"y", "yes"}:
                workflow.approve(step_id)
            else:
                print("    left paused.")

        if blocked["actions"]:
            print("\n" + "-" * 70)
            print("  Actions awaiting your sign-off.")
            print("-" * 70)
            _review_actions(workflow, auto_approve)

        report = workflow.resume()
    return report


def _exports(workflow: Workflow, args: argparse.Namespace) -> None:
    if getattr(args, "json", None):
        print(f"json  -> {workflow.export_state(args.json, format='json')}")
    if getattr(args, "csv", None):
        print(f"csv   -> {workflow.export_state(args.csv, format='csv')}")
    if getattr(args, "html", None):
        print(f"html  -> {workflow.export_state(args.html, format='html')}")


def _load_graph(args: argparse.Namespace) -> Optional[DependencyGraph]:
    if getattr(args, "plan_file", None):
        return plan_from_json(args.plan_file)
    return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    _maybe_force_stub(args)
    _banner()

    from .tools import default_tool_manager

    tools = default_tool_manager()
    mcp_registry = None
    if args.mcp:
        from .tools.mcp import McpRegistry

        mcp_registry = McpRegistry()
        attached = mcp_registry.attach(tools, config_path=args.mcp)
        print(f"MCP tools attached: {', '.join(attached) or 'none'}\n")

    try:
        result = TaskPlanner(llm=get_provider()).plan(args.description, tools=tools)

        # What it needs comes first -- that is the question being asked.
        result.print_requirements()

        print("\nPlanned steps:")
        for step_id in result.graph.topological_order():
            step = result.graph.get(step_id)
            deps = ", ".join(step.depends_on) or "-"
            tool = f", tool: {step.requires_tool}" if step.requires_tool else ""
            print(f"  {step_id:<26} [{step.agent_role}]  (deps: {deps}{tool})")
            print(f"      {step.description}")

        print(f"\nParallel groups: {result.graph.execution_levels()}")
        print(f"Planner tokens : {result.usage.total_tokens}")

        if result.repairs:
            print("\nRepairs applied to the plan:")
            for repair in result.repairs:
                print(f"  - {repair}")

        if args.json_out:
            payload = {"graph": result.graph.to_dict(),
                       "requirements": result.requirements.to_dict()}
            Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nplan written to {args.json_out}")

        if args.mermaid:
            print("\nmermaid:\n" + result.graph.to_mermaid())

        return 0 if result.can_run else 1
    finally:
        if mcp_registry is not None:
            mcp_registry.close_all()


def cmd_run(args: argparse.Namespace) -> int:
    _maybe_force_stub(args)
    _banner()
    graph = _load_graph(args)
    if graph is not None:
        workflow = Workflow(graph=graph, description=args.description or "",
                            run_id=args.run_id, parallel=not args.sequential,
                            smart_invalidation=not args.naive_invalidation,
                            action_approval=args.action_approval)
    else:
        if not args.description:
            print("error: give a project description or --plan-file", file=sys.stderr)
            return 2
        workflow = Workflow.from_description(
            args.description, run_id=args.run_id, parallel=not args.sequential,
            smart_invalidation=not args.naive_invalidation,
            action_approval=args.action_approval,
        )

    if args.mcp:
        registered = workflow.attach_mcp(args.mcp)
        print(f"MCP tools registered: {', '.join(registered) or 'none'}\n")

    if args.connections:
        registered = workflow.attach_connections(args.connections)
        print(f"API connections registered: {', '.join(registered) or 'none'}\n")

    # The convention "skills/" just works if present; no flag needed for the
    # common case, but --no-skills opts out and --skills points elsewhere.
    if not args.no_skills:
        loaded = workflow.attach_skills(args.skills)
        if loaded:
            print(f"Skills loaded: {', '.join(loaded)}\n")

    report = workflow.check_requirements()
    if not args.quiet_requirements:
        report.print_report()
        print()

    if report.blockers and not args.force:
        print("Refusing to run with unmet requirements. Satisfy them, or pass "
              "--force to run anyway (affected steps will fail or degrade).",
              file=sys.stderr)
        workflow.close()
        return 1

    if args.no_interaction:
        workflow.run_full()
        blocked = workflow.blocked_on_human()
        if any(blocked.values()):
            print(f"\nStopped, waiting on a human: {blocked}", file=sys.stderr)
    else:
        _run_interactively(workflow, auto_approve=args.yes)

    if args.change:
        workflow.handle_step_change(args.change, new_requirement=args.requirement)
    if args.break_tool:
        workflow.handle_tool_failure(args.break_tool)

    workflow.print_summary()
    _exports(workflow, args)
    print(f"\nrun id: {workflow.run_id}  (resume with: "
          f"python -m orchestrator.cli resume {workflow.run_id})")
    workflow.close()
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    _maybe_force_stub(args)
    _banner()
    workflow = Workflow.resume_from(args.run_id)
    workflow.resume()
    workflow.print_summary()
    _exports(workflow, args)
    workflow.close()
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    with StateManager() as state:
        runs = state.list_runs(limit=args.limit)
        if not runs:
            print("no persisted runs")
            return 0
        print(f"{'run id':<16}{'status':<10}{'steps done':<12}description")
        for record in runs:
            completed, outstanding = state.resumable(record["run_id"])
            total = len(completed) + len(outstanding)
            print(f"{record['run_id']:<16}{record['status']:<10}"
                  f"{f'{len(completed)}/{total}':<12}{record['description'][:50]}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    _maybe_force_stub(args)
    return run_demo(export_html=args.html, run_id=args.run_id)


def cmd_mcp(args: argparse.Namespace) -> int:
    """List the tools every configured MCP server exposes."""
    from .tools import default_tool_manager
    from .tools.mcp import McpRegistry, load_mcp_config

    _banner()
    specs = load_mcp_config(args.config)
    if not specs:
        print(f"no MCP servers configured in '{args.config}'")
        print("\nCreate one like this:\n")
        print(json.dumps({"mcpServers": {
            "filesystem": {"command": "npx",
                           "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
        }}, indent=2))
        return 0

    print(f"{len(specs)} server(s) configured in '{args.config}':\n")
    manager = default_tool_manager()
    registry = McpRegistry()
    try:
        registry.attach(manager, specs=specs)
        for spec in specs:
            target = spec.command if spec.transport == "stdio" else spec.url
            print(f"  {spec.name}  ({spec.transport}: {target})")
            tools = [
                manager.get(name) for name in manager.names
                if type(manager.get(name)).__name__ == "McpTool"
                and manager.get(name)._spec.name == spec.name
            ]
            if not tools:
                print("    (unreachable or exposes no tools)")
            for tool in tools:
                print(f"    - {tool.name:<28} {(tool.description or '')[:60]}")
            print()
    finally:
        registry.close_all()
    return 0


def cmd_connections(args: argparse.Namespace) -> int:
    """Show every configured API connection and whether it is usable."""
    from .connections import ApiConnectionTool, load_connections

    _banner()
    connections = load_connections(args.config)
    if not connections:
        print(f"no API connections configured in '{args.config}'")
        print("\nCreate one like this:\n")
        print(json.dumps({"connections": {
            "github": {
                "base_url": "https://api.github.com",
                "description": "GitHub REST API",
                "auth": {"type": "bearer", "token_env": "GITHUB_TOKEN"},
                "headers": {"Accept": "application/vnd.github+json"},
            }}}, indent=2))
        return 0

    print(f"{len(connections)} connection(s) in '{args.config}':\n")
    blocked = 0
    for connection in connections:
        tool = ApiConnectionTool(connection)
        missing = connection.auth.missing_env()
        status = "ready" if tool.is_live() else f"SIMULATED (missing {', '.join(missing)})"
        if missing:
            blocked += 1
        print(f"  {tool.name}")
        print(f"    url    : {connection.base_url}")
        print(f"    auth   : {connection.auth.type}"
              + (f" via {', '.join(connection.required_env())}"
                 if connection.required_env() else ""))
        print(f"    status : {status}")
        if connection.description:
            print(f"    about  : {connection.description}")
        print(f"    safe   : {', '.join(sorted(connection.safe_methods or SAFE_METHODS_DISPLAY))}"
              " run freely; other methods need approval")
        print()

    if blocked:
        print(f"{blocked} connection(s) will simulate until their variables are set.")
    return 0


SAFE_METHODS_DISPLAY = {"GET", "HEAD", "OPTIONS"}


def cmd_skills(args: argparse.Namespace) -> int:
    """List loaded skills, or show which ones would apply to a step."""
    from .skills import SkillLibrary

    _banner()
    library = SkillLibrary()
    loaded = library.load_dir(args.dir)
    if not loaded:
        print(f"no skills found in '{args.dir}'")
        print("\nCreate one like this (any .md file in that directory):\n")
        print("---\nname: incident-postmortem\ndescription: Our blameless postmortem format\n"
              "roles: devops, reviewer\nkeywords: incident, outage, postmortem\n---\n\n"
              "# Postmortem format\n1. Timeline (UTC timestamps)\n2. Impact\n...")
        return 0

    print(f"{len(loaded)} skill(s) in '{args.dir}':\n")
    for skill in library:
        print(f"  {skill.name}")
        if skill.description:
            print(f"    {skill.description}")
        if skill.roles:
            print(f"    roles   : {', '.join(skill.roles)}")
        if skill.keywords:
            print(f"    keywords: {', '.join(skill.keywords)}")
        if not skill.roles and not skill.keywords:
            print("    applies to every step (no roles/keywords set)")
        print(f"    source  : {skill.source}")
        print()

    if args.for_role:
        from .models import Step

        probe = Step(id="_probe", description=args.for_description or "",
                    agent_role=args.for_role)
        matched = library.match(probe)
        print(f"Would apply to role '{args.for_role}'"
              + (f" / description {args.for_description!r}" if args.for_description else "")
              + f": {[s.name for s in matched] or 'none'}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("error: uvicorn is not installed (pip install uvicorn fastapi)", file=sys.stderr)
        return 1
    _banner()
    print(f"dashboard: http://{args.host}:{args.port}/\n")
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    """Process durable jobs from the configured SQLite database."""
    from server.app import manager

    _banner()
    recovered = manager.state.recover_interrupted_jobs()
    if recovered:
        print(f"recovered {recovered} interrupted job(s)")
    print(f"worker: polling every {args.poll:.2f}s")
    while True:
        launched = manager.launch_queued_jobs()
        if args.once:
            deadline = time.time() + 60
            while any(manager.busy.values()) and time.time() < deadline:
                time.sleep(0.05)
            if not manager.state.queued_jobs():
                return 0
            continue
        if not launched:
            time.sleep(args.poll)


def cmd_bench(args: argparse.Namespace) -> int:
    from benchmarks.run_benchmarks import main as bench_main

    return bench_main(
        repeats=args.repeats,
        scenarios=args.scenarios,
        output=args.output,
        latency_s=args.latency,
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description="Adaptive AI Task Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--stub", action="store_true",
                        help="force the deterministic offline LLM even if a key is set")
    parser.add_argument("--provider", choices=["gemini", "anthropic", "openai", "nvidia", "ollama"],
                        help="use this LLM provider, bypassing auto-detection "
                             "(place before the subcommand, e.g. 'orchestrator --provider "
                             "ollama run ...')")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser(
        "plan", help="plan a task and report what it needs, without running it")
    p_plan.add_argument("description")
    p_plan.add_argument("--mcp", nargs="?", const=".mcp.json", metavar="CONFIG",
                        help="attach MCP servers first, so their tools count as available")
    p_plan.add_argument("--json-out", metavar="PATH", help="write the plan + requirements to JSON")
    p_plan.add_argument("--mermaid", action="store_true", help="also print a mermaid diagram")
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="plan and execute a workflow")
    p_run.add_argument("description", nargs="?", default="")
    p_run.add_argument("--plan-file", help="run a hand-written plan (JSON) instead of planning")
    p_run.add_argument("--run-id")
    p_run.add_argument("--sequential", action="store_true", help="disable parallel execution")
    p_run.add_argument("--naive-invalidation", action="store_true",
                       help="force the whole downstream cone to re-run on a change")
    p_run.add_argument("--change", help="after the full run, change this step")
    p_run.add_argument("--requirement", help="the new requirement for --change")
    p_run.add_argument("--break-tool", help="after the run, break this tool and recover")
    p_run.add_argument("--mcp", nargs="?", const=".mcp.json", metavar="CONFIG",
                       help="register tools from MCP servers (default config: .mcp.json)")
    p_run.add_argument("--connections", nargs="?", const="connections.json", metavar="CONFIG",
                       help="register API connections (default config: connections.json)")
    p_run.add_argument("--skills", default="skills", metavar="DIR",
                       help="directory of skill files to load (default: skills/)")
    p_run.add_argument("--no-skills", action="store_true",
                       help="do not load skills/ even if present")
    p_run.add_argument("--force", action="store_true",
                       help="run even when requirements are unmet")
    p_run.add_argument("--no-interaction", action="store_true",
                       help="never prompt; stop and report what a human would need to supply")
    p_run.add_argument("--yes", "-y", action="store_true",
                       help="auto-approve irreversible actions (they will really happen)")
    p_run.add_argument("--action-approval", choices=["live", "all", "never"], default="live",
                       help="when to pause before an irreversible action (default: live)")
    p_run.add_argument("--quiet-requirements", action="store_true",
                       help="skip the requirements report")
    p_run.add_argument("--json"), p_run.add_argument("--csv"), p_run.add_argument("--html")
    p_run.set_defaults(func=cmd_run)

    p_resume = sub.add_parser("resume", help="resume a persisted run after a crash")
    p_resume.add_argument("run_id")
    p_resume.add_argument("--json"), p_resume.add_argument("--csv"), p_resume.add_argument("--html")
    p_resume.set_defaults(func=cmd_resume)

    p_runs = sub.add_parser("runs", help="list persisted runs")
    p_runs.add_argument("--limit", type=int, default=25)
    p_runs.set_defaults(func=cmd_runs)

    p_demo = sub.add_parser("demo", help="run the four headline scenarios end to end")
    p_demo.add_argument("--html", default="dashboard.html")
    p_demo.add_argument("--run-id", default=None)
    p_demo.set_defaults(func=cmd_demo)

    p_mcp = sub.add_parser("mcp", help="list tools exposed by configured MCP servers")
    p_mcp.add_argument("--config", default=".mcp.json")
    p_mcp.set_defaults(func=cmd_mcp)

    p_conn = sub.add_parser("connections", help="list configured API connections")
    p_conn.add_argument("--config", default="connections.json")
    p_conn.set_defaults(func=cmd_connections)

    p_skills = sub.add_parser("skills", help="list skills and preview what applies to a step")
    p_skills.add_argument("--dir", default="skills")
    p_skills.add_argument("--for-role", help="show which skills would apply to this role")
    p_skills.add_argument("--for-description", help="paired with --for-role, for keyword matching")
    p_skills.set_defaults(func=cmd_skills)

    p_serve = sub.add_parser("serve", help="start the web dashboard")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_worker = sub.add_parser("worker", help="process jobs from the durable execution queue")
    p_worker.add_argument("--poll", type=float, default=1.0)
    p_worker.add_argument("--once", action="store_true",
                          help="process currently queued jobs, then exit")
    p_worker.set_defaults(func=cmd_worker)

    p_bench = sub.add_parser("bench", help="benchmark against baseline frameworks")
    p_bench.add_argument("--repeats", type=int, default=3)
    p_bench.add_argument("--scenarios", nargs="*", default=None)
    p_bench.add_argument("--output", default="benchmarks/results")
    p_bench.add_argument("--latency", type=float, default=0.05,
                         help="simulated per-LLM-call latency in seconds")
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    # Hosted models freely use Unicode punctuation. Windows may otherwise use
    # CP1252 and crash after a successful API call while printing the result.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
