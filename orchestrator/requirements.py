"""What a task needs before it can actually be done.

The planner does not just produce steps -- it declares the *capabilities* the
work depends on (tools, MCP servers, credentials, packages) and the
*specialists* it wants to do it. This module models those declarations and
checks each one against the real environment, so you get an honest answer to
"can this actually run, and what do I need to set up?" **before** any agent
burns a token.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .tools import ToolManager


class RequirementKind(str, Enum):
    TOOL = "tool"                # a built-in adapter, e.g. github, postgres
    MCP_SERVER = "mcp_server"    # an MCP server to connect to
    CREDENTIAL = "credential"    # an environment variable / API key
    PACKAGE = "package"          # an importable Python package
    BINARY = "binary"            # an executable on PATH, e.g. gh, psql


class RequirementStatus(str, Enum):
    READY = "ready"          # present and fully functional
    SIMULATED = "simulated"  # usable, but will produce labelled fake output
    FALLBACK = "fallback"    # unavailable, but a substitute covers it
    MISSING = "missing"      # unavailable, with nothing to cover it

    @property
    def blocks_execution(self) -> bool:
        return self is RequirementStatus.MISSING


#: Rendered in reports. ASCII so it survives any Windows console codepage.
_STATUS_MARK = {
    RequirementStatus.READY: "[ready]",
    RequirementStatus.SIMULATED: "[sim] ",
    RequirementStatus.FALLBACK: "[fall]",
    RequirementStatus.MISSING: "[MISS]",
}


@dataclass
class AgentSpec:
    """A specialist the planner decided this task needs.

    Unlike the built-in role catalogue, these are invented per task -- a
    "FHIR compliance auditor" or "Rust performance engineer" as easily as a
    "backend engineer".
    """

    role: str
    system_prompt: str
    why: str = ""
    #: Step ids this agent was created for.
    handles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "system_prompt": self.system_prompt,
                "why": self.why, "handles": self.handles}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSpec":
        return cls(
            role=str(data.get("role", "")).strip(),
            system_prompt=str(data.get("system_prompt", "")).strip(),
            why=str(data.get("why", "")).strip(),
            handles=list(data.get("handles") or []),
        )


@dataclass
class Requirement:
    """One prerequisite, plus how to satisfy it if it is missing."""

    name: str
    kind: RequirementKind
    why: str = ""
    #: Step ids that need this.
    needed_by: List[str] = field(default_factory=list)
    #: Concrete command or variable to set, shown when it is missing.
    setup: str = ""
    #: Whether the work can proceed (degraded) without it.
    optional: bool = False

    # Filled in by check_requirements().
    status: RequirementStatus = RequirementStatus.MISSING
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind.value, "why": self.why,
            "needed_by": self.needed_by, "setup": self.setup,
            "optional": self.optional, "status": self.status.value,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Requirement":
        name = str(data.get("name", "")).strip()
        kind = data.get("kind", "tool")
        try:
            kind_enum = RequirementKind(kind)
        except ValueError:
            kind_enum = RequirementKind.TOOL
        # Models often call language runtimes "packages" even though they are
        # executables. Checking ``import python`` then falsely reports that the
        # already-running interpreter is missing and asks the user to install
        # it. Normalise only unambiguous runtime names; real libraries remain
        # package requirements.
        if kind_enum is RequirementKind.PACKAGE and name.lower() in {
            "python", "python3", "node", "nodejs", "npm", "git",
        }:
            kind_enum = RequirementKind.BINARY
        return cls(
            name=name,
            kind=kind_enum,
            why=str(data.get("why", "")).strip(),
            needed_by=list(data.get("needed_by") or []),
            setup=str(data.get("setup", "")).strip(),
            optional=bool(data.get("optional", False)),
        )


# ---------------------------------------------------------------------------
# Readiness checking
# ---------------------------------------------------------------------------


def _check_tool(req: Requirement, tools: Optional["ToolManager"]) -> None:
    if tools is None:
        req.status = RequirementStatus.MISSING
        req.detail = "no tool registry available"
        return

    from .tools import canonical_tool_name

    name = canonical_tool_name(req.name) or req.name
    tool = tools.get(name)

    if tool is not None and not tool.is_broken:
        if tool.is_live():
            req.status = RequirementStatus.READY
            req.detail = "configured and live"
        else:
            req.status = RequirementStatus.SIMULATED
            req.detail = "registered without credentials; output will be labelled [simulated:...]"
        return

    # Either unregistered, or registered but broken. Either way the question
    # is the same: is there a healthy substitute the router would reach for?
    # A broken tool with a working fallback is a degraded run, not a blocked
    # one -- reporting it as MISSING would be wrong.
    substitutes = [
        candidate for candidate in tools.candidates_for(name)[1:]
        if tools.get(candidate) is not None and not tools.get(candidate).is_broken
    ]
    if substitutes:
        req.status = RequirementStatus.FALLBACK
        reason = "marked broken" if tool is not None else "not registered"
        req.detail = f"{reason}; would fall back to '{substitutes[0]}'"
    elif tool is not None:
        req.status = RequirementStatus.MISSING
        req.detail = "registered but marked broken, with no working fallback"
    else:
        req.status = RequirementStatus.MISSING
        req.detail = "no tool registered and no fallback available"


def _check_mcp_server(req: Requirement, tools: Optional["ToolManager"]) -> None:
    if importlib.util.find_spec("mcp") is None:
        req.status = RequirementStatus.MISSING
        req.detail = "the 'mcp' SDK is not installed (pip install mcp)"
        return

    if tools is not None:
        # An attached server shows up as registered McpTool instances.
        for tool_name in tools.names:
            tool = tools.get(tool_name)
            spec = getattr(tool, "_spec", None)
            if spec is not None and getattr(spec, "name", None) == req.name:
                req.status = RequirementStatus.READY
                req.detail = "connected; its tools are registered"
                return

    req.status = RequirementStatus.MISSING
    req.detail = "not attached; add it to .mcp.json and call attach_mcp()"


def _check_credential(req: Requirement) -> None:
    if os.environ.get(req.name):
        req.status = RequirementStatus.READY
        req.detail = "set in the environment"
    else:
        req.status = RequirementStatus.MISSING
        req.detail = f"environment variable '{req.name}' is not set"


def _check_package(req: Requirement) -> None:
    module = req.name.replace("-", "_")
    if importlib.util.find_spec(module) is not None:
        req.status = RequirementStatus.READY
        req.detail = "importable"
    else:
        req.status = RequirementStatus.MISSING
        req.detail = f"'{module}' is not importable"


def _check_binary(req: Requirement) -> None:
    path = shutil.which(req.name)
    if path:
        req.status = RequirementStatus.READY
        req.detail = path
    else:
        req.status = RequirementStatus.MISSING
        req.detail = f"'{req.name}' is not on PATH"


def check_requirement(req: Requirement, tools: Optional["ToolManager"] = None) -> Requirement:
    """Resolve one requirement against the real environment (mutates in place)."""
    if req.kind is RequirementKind.TOOL:
        _check_tool(req, tools)
    elif req.kind is RequirementKind.MCP_SERVER:
        _check_mcp_server(req, tools)
    elif req.kind is RequirementKind.CREDENTIAL:
        _check_credential(req)
    elif req.kind is RequirementKind.PACKAGE:
        _check_package(req)
    elif req.kind is RequirementKind.BINARY:
        _check_binary(req)

    # An optional requirement never blocks; downgrade a hard miss.
    if req.optional and req.status is RequirementStatus.MISSING:
        req.status = RequirementStatus.SIMULATED
        req.detail = (req.detail + "; optional, so the run will proceed without it").lstrip("; ")
    return req


@dataclass
class RequirementsReport:
    """Everything the task needs, and whether the environment provides it."""

    requirements: List[Requirement] = field(default_factory=list)
    agents: List[AgentSpec] = field(default_factory=list)
    #: Free-text notes from the planner (assumptions, caveats).
    notes: List[str] = field(default_factory=list)
    plan_errors: List[str] = field(default_factory=list)
    estimated_tokens: int = 0

    # -- checking ---------------------------------------------------------

    def check(self, tools: Optional["ToolManager"] = None) -> "RequirementsReport":
        for req in self.requirements:
            check_requirement(req, tools)
        return self

    # -- views ------------------------------------------------------------

    def by_status(self, status: RequirementStatus) -> List[Requirement]:
        return [r for r in self.requirements if r.status is status]

    @property
    def blockers(self) -> List[Requirement]:
        """Requirements that genuinely prevent the task from being done."""
        return [r for r in self.requirements if r.status.blocks_execution]

    @property
    def can_run(self) -> bool:
        """True when nothing is hard-missing. Simulated/fallback still runs."""
        return not self.blockers and not self.plan_errors

    @property
    def degraded(self) -> List[Requirement]:
        """Present but not fully real -- output will be simulated or substituted."""
        return [r for r in self.requirements
                if r.status in {RequirementStatus.SIMULATED, RequirementStatus.FALLBACK}]

    # -- rendering --------------------------------------------------------

    def render(self, width: int = 78) -> str:
        lines: List[str] = []
        lines.append("=" * width)
        lines.append("  WHAT THIS TASK NEEDS")
        lines.append("=" * width)

        if self.agents:
            lines.append("\nSpecialists to be created:")
            for agent in self.agents:
                handles = f"  (steps: {', '.join(agent.handles)})" if agent.handles else ""
                lines.append(f"  - {agent.role}{handles}")
                if agent.why:
                    lines.append(f"      {agent.why}")

        if self.requirements:
            lines.append("\nCapabilities required:")
            grouped: Dict[RequirementKind, List[Requirement]] = {}
            for req in self.requirements:
                grouped.setdefault(req.kind, []).append(req)

            for kind in RequirementKind:
                items = grouped.get(kind)
                if not items:
                    continue
                lines.append(f"\n  {kind.value.replace('_', ' ')}:")
                for req in items:
                    mark = _STATUS_MARK[req.status]
                    lines.append(f"    {mark} {req.name}")
                    if req.why:
                        lines.append(f"           why: {req.why}")
                    if req.detail:
                        lines.append(f"           status: {req.detail}")
                    if req.status.blocks_execution and req.setup:
                        lines.append(f"           setup: {req.setup}")
        else:
            lines.append("\nCapabilities required: none - this task is pure reasoning.")

        if self.notes:
            lines.append("\nPlanner notes:")
            for note in self.notes:
                lines.append(f"  - {note}")

        if self.estimated_tokens:
            lines.append(f"\nEstimated plan tokens (not measured usage): {self.estimated_tokens}")
        if self.plan_errors:
            lines.append("\nPlan errors:")
            lines.extend(f"  - {error}" for error in self.plan_errors)
        lines.append("\n" + "-" * width)
        if self.can_run:
            degraded = self.degraded
            if degraded:
                lines.append(f"READY TO RUN, with {len(degraded)} capability(ies) degraded:")
                for req in degraded:
                    lines.append(f"  - {req.name}: {req.detail}")
                lines.append("Those steps still complete; their tool output is clearly labelled.")
            else:
                lines.append("READY TO RUN - everything required is configured.")
        else:
            lines.append(f"BLOCKED - {len(self.blockers)} missing requirement(s), "
                         f"{len(self.plan_errors)} plan error(s):")
            for req in self.blockers:
                lines.append(f"  - {req.name} ({req.kind.value}): {req.detail}")
                if req.setup:
                    lines.append(f"      fix: {req.setup}")
        lines.append("=" * width)
        return "\n".join(lines)

    def print_report(self, width: int = 78) -> None:
        print(self.render(width))

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirements": [r.to_dict() for r in self.requirements],
            "agents": [a.to_dict() for a in self.agents],
            "notes": self.notes,
            "can_run": self.can_run,
            "blockers": [r.name for r in self.blockers],
            "degraded": [r.name for r in self.degraded],
            "plan_errors": self.plan_errors,
            "estimated_tokens": self.estimated_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequirementsReport":
        return cls(
            requirements=[Requirement.from_dict(r) for r in data.get("requirements") or []],
            agents=[AgentSpec.from_dict(a) for a in data.get("agents") or []],
            notes=list(data.get("notes") or []),
            plan_errors=list(data.get("plan_errors") or []),
            estimated_tokens=int(data.get("estimated_tokens") or 0),
        )


# ---------------------------------------------------------------------------
# Inference from a graph (used when the planner under-declares)
# ---------------------------------------------------------------------------

#: Tools whose real use needs credentials, and what those credentials are.
TOOL_CREDENTIALS: Dict[str, List[tuple]] = {
    "github": [("GITHUB_TOKEN", "GitHub personal access token"),
               ("GITHUB_REPO", "target repo as owner/name")],
    "ci": [("GITHUB_TOKEN", "GitHub token to dispatch workflow runs"),
           ("GITHUB_REPO", "repo whose workflow should run")],
    "postgres": [("DATABASE_URL", "PostgreSQL connection string")],
    "slack": [("SLACK_WEBHOOK_URL", "Slack incoming webhook URL")],
    "email": [("SENDGRID_API_KEY", "SendGrid API key"), ("EMAIL_TO", "recipient address")],
    # Sending needs an account to send *from*, not just an address to send to.
    "gmail": [("GMAIL_ADDRESS", "the Gmail account to send from"),
              ("GMAIL_APP_PASSWORD", "Gmail app password for that account")],
    # Not a secret either: the loop runs real shell commands, so it stays off
    # until someone deliberately allows it, and the wizard is where they find
    # out it exists.
    "computer_task": [
        ("ORCHESTRATOR_ALLOW_TERMINAL", "set to 1 to let agents run shell commands"),
    ],
    # Not secrets, but the wizard is where a person finds out this capability
    # exists and that it is off until they deliberately turn it on and say
    # where it may act. An empty allow-list permits nothing.
    "computer_use": [
        ("ORCHESTRATOR_ALLOW_DESKTOP", "set to 1 to let agents drive a browser or the desktop"),
        ("ORCHESTRATOR_COMPUTER_USE_ALLOW", "comma-separated domains and apps it may touch"),
    ],
    "stripe": [("STRIPE_API_KEY", "Stripe restricted API key")],
    "s3": [("AWS_ACCESS_KEY_ID", "AWS access key or use AWS_PROFILE"),
           ("AWS_SECRET_ACCESS_KEY", "AWS secret access key or use AWS_PROFILE"),
           ("AWS_S3_BUCKET", "S3 bucket name")],
    "hermes_desktop": [
        ("ORCHESTRATOR_ALLOW_DESKTOP", "explicit permission to run desktop automation")
    ],
    "terminal": [
        ("ORCHESTRATOR_ALLOW_TERMINAL", "explicit permission to run restricted project commands")
    ],
}


def infer_requirements(graph: Any) -> List[Requirement]:
    """Derive tool + credential requirements directly from the planned steps.

    A safety net: whatever the planner declares, every tool a step actually
    names still shows up in the report, along with the credentials that tool
    needs to do real (rather than simulated) work.
    """
    from .tools import canonical_tool_name

    by_tool: Dict[str, List[str]] = {}
    for step_id in graph.topological_order():
        tool = canonical_tool_name(graph.get(step_id).requires_tool)
        if tool:
            by_tool.setdefault(tool, []).append(step_id)

    requirements: List[Requirement] = []
    for tool, step_ids in sorted(by_tool.items()):
        requirements.append(Requirement(
            name=tool,
            kind=RequirementKind.TOOL,
            why=f"used by {', '.join(step_ids)}",
            needed_by=step_ids,
            setup=f"see .env.example for how to configure '{tool}'",
        ))
        for env_var, description in TOOL_CREDENTIALS.get(tool, []):
            requirements.append(Requirement(
                name=env_var,
                kind=RequirementKind.CREDENTIAL,
                why=f"{description} - needed for '{tool}' to do real work",
                needed_by=step_ids,
                setup=f"set {env_var} in .env.local",
                # Without it the tool simulates rather than fails, so the run
                # is degraded, not blocked.
                optional=True,
            ))
    return requirements


def merge_requirements(*groups: List[Requirement]) -> List[Requirement]:
    """Combine requirement lists, de-duplicating on (kind, name)."""
    merged: Dict[tuple, Requirement] = {}
    for group in groups:
        for req in group:
            key = (req.kind, req.name.lower())
            existing = merged.get(key)
            if existing is None:
                merged[key] = req
                continue
            # Keep the richer description, union the dependents.
            if len(req.why) > len(existing.why):
                existing.why = req.why
            if req.setup and not existing.setup:
                existing.setup = req.setup
            existing.needed_by = sorted(set(existing.needed_by) | set(req.needed_by))
            existing.optional = existing.optional and req.optional
    return list(merged.values())


#: Requirement kinds that a person can satisfy by typing a value. Everything
#: else needs something installed, so the best we can do is ask them to
#: confirm once they have done it.
_TYPEABLE = {RequirementKind.CREDENTIAL}


def to_input_requests(
    report: "RequirementsReport",
    include_simulated: bool = True,
) -> Dict[str, List["InputRequest"]]:
    """Turn unsatisfied requirements into questions, keyed by step id.

    Without this, a missing credential is silently downgraded to simulated
    output: the step reports success and the work never happens. Feeding
    these into ``step.inputs`` instead makes the run stop and ask, which is
    the behaviour an operator expects.

    A credential becomes a secret field, since supplying the value is all it
    takes. Anything else -- a binary, a package, an MCP server -- cannot be
    typed in, so it becomes a yes/no confirmation to tick once installed.
    Requirements with no ``needed_by`` are skipped: there is no step to
    attach the question to.
    """
    from .inputs import InputRequest, InputType

    wanted = {RequirementStatus.MISSING}
    if include_simulated:
        wanted.add(RequirementStatus.SIMULATED)

    unmet = [r for r in report.requirements
             if not r.optional and r.status in wanted and r.needed_by]

    # A tool is only ever simulated *because* of something underneath it. When
    # that cause is already being asked about for the same step, also asking
    # "is the tool ready?" is two questions for one fix.
    covered: set = set()
    for req in unmet:
        if req.kind in _TYPEABLE:
            covered.update(req.needed_by)

    questions: Dict[str, List[InputRequest]] = {}
    for req in unmet:
        if req.kind is RequirementKind.TOOL and set(req.needed_by) <= covered:
            continue

        if req.kind in _TYPEABLE:
            request = InputRequest(
                name=req.name,
                prompt=f"Value for {req.name}",
                type=InputType.SECRET,
                required=True,
                why=req.why or f"{req.name} is needed before this step can do real work",
            )
        else:
            setup = req.setup or f"install or configure '{req.name}'"
            request = InputRequest(
                name=f"{req.name}_ready",
                prompt=f"Have you set up '{req.name}'? ({setup})",
                type=InputType.BOOLEAN,
                required=True,
                why=req.why or f"'{req.name}' is not available yet",
            )

        # Deliberately the same object for every dependent step: one
        # credential answered once should unblock everything waiting on it,
        # not be retyped per step.
        for step_id in req.needed_by:
            questions.setdefault(step_id, []).append(request)
    return questions
