"""Layer 5 -- the workflow orchestrator.

Responsibilities:

* run a DAG of steps, in parallel where the dependencies allow it
* decide, per step, whether a cached result is still valid (adaptive re-execution)
* retry transient failures, cancel the downstream of fatal ones
* route around broken tools
* pause for human approval and resume
* persist everything so a crash costs one step, not a run

The scheduling rule is deliberately simple, and it is the whole trick::

    a step runs  <=>  it was explicitly forced
                 or   it has no usable cached result
                 or   its input signature no longer matches the cached one

The third clause is what makes change propagation *automatic*: a step's input
signature covers its own definition plus the exact outputs of its
dependencies, so a change ripples downstream on its own -- and stops rippling
the moment a re-run reproduces an identical output.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .agents import AgentManager
from .config import settings
from .events import EventBus, EventType, console_printer
from .graph import DependencyGraph
from .llm import LLMError, LLMProvider, get_provider, is_transient
from .memory import MemoryManager
from .models import (
    USABLE,
    ErrorClass,
    LLMUsage,
    PendingAction,
    Step,
    StepResult,
    StepStatus,
    content_hash,
    new_run_id,
)
from .requirements import RequirementsReport, infer_requirements
from .skills import DEFAULT_SKILLS_DIR, SkillLibrary
from .state import StateManager
from .tools import ToolManager, ToolUnavailableError, default_tool_manager


class ApprovalRequired(RuntimeError):
    """Raised when a run cannot continue past a human-in-the-loop gate."""

    def __init__(self, step_ids: Sequence[str]):
        self.step_ids = list(step_ids)
        super().__init__(f"awaiting approval for: {', '.join(self.step_ids)}")


class RunReport:
    """Summary of one call to :meth:`Workflow.run` and friends."""

    def __init__(self, run_id: str, reason: str):
        self.run_id = run_id
        self.reason = reason
        self.started_at = time.time()
        self.ended_at: Optional[float] = None
        self.executed: List[str] = []
        self.reused: List[str] = []
        self.failed: List[str] = []
        self.cancelled: List[str] = []
        self.not_applicable: List[str] = []
        self.awaiting_approval: List[str] = []
        self.awaiting_input: List[str] = []
        self.awaiting_action: List[str] = []
        self.cancel_requested = False
        self.usage = LLMUsage()

    @property
    def duration_s(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    @property
    def ok(self) -> bool:
        return not self.failed and not self.cancelled and not self.cancel_requested

    @property
    def waiting_on_human(self) -> List[str]:
        """Steps that stopped for a person, not for an error."""
        return sorted(set(self.awaiting_approval) | set(self.awaiting_input)
                      | set(self.awaiting_action))

    @property
    def reuse_ratio(self) -> float:
        total = len(self.executed) + len(self.reused)
        return round(len(self.reused) / total, 4) if total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "reason": self.reason,
            "duration_s": round(self.duration_s, 4),
            "executed": self.executed,
            "reused": self.reused,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "not_applicable": self.not_applicable,
            "awaiting_approval": self.awaiting_approval,
            "awaiting_input": self.awaiting_input,
            "awaiting_action": self.awaiting_action,
            "cancel_requested": self.cancel_requested,
            "waiting_on_human": self.waiting_on_human,
            "steps_executed": len(self.executed),
            "steps_reused": len(self.reused),
            "reuse_ratio": self.reuse_ratio,
            "usage": self.usage.to_dict(),
            "ok": self.ok,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<RunReport {self.reason}: {len(self.executed)} executed, "
                f"{len(self.reused)} reused, {len(self.failed)} failed>")


class Workflow:
    """Executes a :class:`DependencyGraph` adaptively."""

    def __init__(
        self,
        steps: Optional[Iterable[Step]] = None,
        graph: Optional[DependencyGraph] = None,
        *,
        description: str = "",
        run_id: Optional[str] = None,
        llm: Optional[LLMProvider] = None,
        tool_manager: Optional[ToolManager] = None,
        agent_manager: Optional[AgentManager] = None,
        memory: Optional[MemoryManager] = None,
        state: Optional[StateManager] = None,
        bus: Optional[EventBus] = None,
        max_workers: Optional[int] = None,
        parallel: bool = True,
        smart_invalidation: bool = True,
        persist: bool = True,
        verbose: bool = True,
        mcp_config: Optional[str] = None,
        action_approval: str = "live",
        skills_dir: Optional[str] = None,
    ):
        if graph is None:
            graph = DependencyGraph(steps or [])
        elif steps:
            raise ValueError("pass either 'steps' or 'graph', not both")
        graph.validate()

        self.graph = graph
        self.description = description
        self.run_id = run_id or new_run_id()

        self.llm = llm or get_provider()
        self.tools = tool_manager or default_tool_manager()
        # The default catalogue is created before the workflow id is known.
        # Replace only our built-in Hermes adapter with a run-scoped instance;
        # custom adapters supplied by callers are left untouched.
        from .tools.hermes import HermesDesktopTool

        if isinstance(self.tools.get("hermes_desktop"), HermesDesktopTool):
            self.tools.register(HermesDesktopTool(self.run_id))
        from .planner import is_software_task
        if is_software_task(self.description):
            from .tools.workspace import coding_team_tools
            for coding_tool in coding_team_tools(self.run_id):
                self.tools.register(coding_tool)
        self.agents = agent_manager or AgentManager(llm=self.llm)
        self.memory = memory or MemoryManager()
        self.bus = bus or EventBus()
        #: Input name -> whether answering it should export an environment
        #: variable. Populated by ask_for_missing_requirements(); a supplied
        #: credential has to reach os.environ or the tool stays simulated.
        self._requirement_env: Dict[str, bool] = {}

        self.max_workers = max_workers or settings.max_workers
        self.parallel = parallel
        #: When True, a step is reused if its input signature is unchanged,
        #: even if it sits downstream of something that was re-run.
        self.smart_invalidation = smart_invalidation
        #: When an irreversible tool call (send email, post to Slack, run a
        #: migration) must pause for explicit sign-off:
        #:   "live"  -- only when it would really happen (default)
        #:   "all"   -- also for simulated calls, to rehearse the flow
        #:   "never" -- unattended; act immediately
        if action_approval not in {"live", "all", "never"}:
            raise ValueError(
                f"action_approval must be 'live', 'all' or 'never', got {action_approval!r}")
        self.action_approval = action_approval

        self.results: Dict[str, StepResult] = {
            sid: StepResult(step_id=sid) for sid in self.graph.topological_order()
        }
        self.reports: List[RunReport] = []
        self._approved: Set[str] = set()
        #: Steps whose irreversible action the operator has signed off.
        self._approved_actions: Set[str] = set()
        self._pending_actions: Dict[str, PendingAction] = {}
        # Values supplied by an external trigger for the current execution.
        # Keep them in memory so arbitrary webhook data is not copied into
        # the workflow definition or durable metadata.
        self.runtime_inputs: Dict[str, Any] = {}
        self.agent_messages: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        #: Cooperative stop signal. Running model/tool calls are allowed to
        #: finish; no later dependency level is started after this is set.
        self._cancel_requested = threading.Event()
        self._mcp: Optional[Any] = None
        #: Populated by from_description(); otherwise inferred on demand by
        #: check_requirements().
        self.requirements: RequirementsReport = RequirementsReport()
        self.plan_result: Optional[Any] = None
        # Explicit, like mcp_config: a Workflow never reads from disk unless
        # asked to. The CLI wires the "skills/ just works" convention itself.
        self.skills = SkillLibrary()
        if skills_dir:
            self.attach_skills(skills_dir)
        if mcp_config:
            self.attach_mcp(mcp_config)

        self.state: Optional[StateManager] = None
        if persist:
            self.state = state or StateManager()
            self.bus.subscribe(self.state.subscriber())
            self.state.create_run(self.run_id, description, self.graph,
                                  meta={"parallel": parallel, "max_workers": self.max_workers})

        if verbose:
            self.bus.subscribe(console_printer())

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_description(cls, description: str, mcp_config: Optional[str] = None,
                         clarifications: Optional[Dict[str, Any]] = None,
                         **kwargs: Any) -> "Workflow":
        """Plan a workflow straight from plain English.

        The planner also decides which specialists the task needs and what
        capabilities it depends on; the specialists are registered
        automatically and the capability report is available afterwards as
        ``workflow.requirements``.
        """
        from .planner import TaskPlanner, is_software_task

        llm = kwargs.get("llm") or get_provider()

        # Requirements are checked against the registry the run will actually
        # use, so attached MCP servers count as satisfied.
        tools = kwargs.get("tool_manager")
        if tools is None:
            tools = default_tool_manager()
            kwargs["tool_manager"] = tools

        # The planner can only assign tools it sees. Attach the run-scoped
        # workspace and restricted terminal before planning software work.
        if is_software_task(description):
            planned_run_id = kwargs.get("run_id") or new_run_id()
            kwargs["run_id"] = planned_run_id
            from .tools.workspace import coding_team_tools
            for coding_tool in coding_team_tools(planned_run_id):
                tools.register(coding_tool)

        pre_attached: List[str] = []
        if mcp_config:
            from .tools.mcp import McpRegistry

            registry = McpRegistry()
            pre_attached = registry.attach(tools, config_path=mcp_config)

        plan = TaskPlanner(llm=llm).plan(description, tools=tools,
                                         clarifications=clarifications)
        workflow = cls(graph=plan.graph, description=description, **kwargs)
        if mcp_config:
            workflow._mcp = registry  # so close() shuts the servers down

        workflow.requirements = plan.requirements
        workflow.plan_result = plan
        registered = workflow.register_planned_agents(plan.requirements.agents)

        workflow.bus.publish(
            EventType.PLAN_CREATED, run_id=workflow.run_id,
            message=f"planned {len(plan.graph)} steps, "
                    f"{len(registered)} specialist(s), "
                    f"{len(plan.requirements.requirements)} requirement(s)",
            steps=plan.graph.topological_order(),
            levels=plan.graph.execution_levels(),
            repairs=plan.repairs,
            used_fallback_plan=plan.used_fallback_plan,
            agents=registered,
            requirements=plan.requirements.to_dict(),
            mcp_tools=pre_attached,
        )
        for repair in plan.repairs:
            workflow.bus.publish(EventType.LOG, run_id=workflow.run_id,
                                 message=f"plan repair: {repair}")
        for blocker in plan.requirements.blockers:
            workflow.bus.publish(
                EventType.LOG, run_id=workflow.run_id,
                message=f"BLOCKER: {blocker.name} ({blocker.kind.value}) - {blocker.detail}")
        return workflow

    def register_planned_agents(self, specs: Iterable[Any]) -> List[str]:
        """Create the specialists the planner invented. Returns their roles."""
        created: List[str] = []
        for spec in specs:
            self.agents.add(spec.role, spec.system_prompt)
            created.append(spec.role)
        return created

    @classmethod
    def resume_from(cls, run_id: str, state: Optional[StateManager] = None,
                    **kwargs: Any) -> "Workflow":
        """Rebuild a workflow from persisted state after a crash or restart."""
        state = state or StateManager()
        graph = state.load_graph(run_id)
        if graph is None:
            raise KeyError(f"no persisted run with id '{run_id}'")
        record = state.get_run(run_id) or {}
        workflow = cls(graph=graph, description=record.get("description", ""),
                       run_id=run_id, state=state, **kwargs)
        workflow.load_persisted_results()
        workflow.load_persisted_actions()
        workflow._load_persisted_messages()
        return workflow

    def _load_persisted_messages(self) -> None:
        if self.state is None:
            return
        self.agent_messages = [
            {"from": event["step_id"], "to": event["data"].get("to"),
             "message": event["data"].get("message_text", "")}
            for event in self.state.load_events(self.run_id, limit=10000)
            if event["type"] == EventType.AGENT_MESSAGE.value
        ]

    def check_requirements(self) -> RequirementsReport:
        """Re-check what this workflow needs against the current environment.

        Works whether the plan came from the planner or the steps were written
        by hand -- in the hand-written case the requirements are inferred from
        the tools the steps name. Re-run it after attaching an MCP server or
        setting a credential to see the status change.
        """
        if not self.requirements.requirements:
            self.requirements = RequirementsReport(
                requirements=infer_requirements(self.graph),
                agents=self.requirements.agents,
                notes=self.requirements.notes,
            )
        self.requirements.check(self.tools)
        return self.requirements

    def print_requirements(self, width: int = 78) -> None:
        """Show what the task needs and whether it can actually run."""
        self.check_requirements().print_report(width)

    def ask_for_missing_requirements(self, include_simulated: bool = True) -> List[str]:
        """Turn anything still missing into questions attached to the steps.

        By default an unmet requirement only downgrades a tool to simulated
        output, so the step "succeeds" having done nothing real. Calling this
        first converts those into pending inputs, which stops the step at
        AWAITING_INPUT until a person supplies the value.

        Returns the names asked about. Safe to call repeatedly: a question
        already attached to a step is not added twice, and requirements that
        have since been satisfied are dropped.
        """
        from .requirements import to_input_requests

        report = self.check_requirements()
        questions = to_input_requests(report, include_simulated=include_simulated)

        asked: List[str] = []
        for step_id, requests in questions.items():
            if step_id not in self.graph:
                continue
            step = self.graph.get(step_id)
            existing = {i.name for i in step.inputs}
            for request in requests:
                if request.name in existing:
                    continue
                step.inputs.append(request)
                asked.append(request.name)
                if request.name not in self._requirement_env:
                    self._requirement_env[request.name] = request.type.value == "secret"

        if asked:
            self.bus.publish(
                EventType.LOG, run_id=self.run_id,
                message=f"waiting on {len(asked)} unmet requirement(s): "
                        f"{', '.join(sorted(set(asked)))}")
            if self.state is not None:
                self.state.update_run(self.run_id, graph=self.graph)
        return asked

    def enable_coding_team(self, root: Optional[Any] = None) -> List[str]:
        """Give the agents a shared workspace and a terminal.

        This is what turns "the backend agent describes an API" into "the
        backend agent writes api/routes.py and the test agent runs pytest on
        it". Both tools are confined to ``workspace/<run_id>/``; the terminal
        additionally refuses privilege-escalating commands and needs
        ``ORCHESTRATOR_ALLOW_TERMINAL=1`` before it will execute anything.
        """
        from .tools.workspace import coding_team_tools

        registered = []
        for tool in coding_team_tools(self.run_id, root):
            self.tools.register(tool)
            registered.append(tool.name)
        self.bus.publish(EventType.LOG, run_id=self.run_id,
                         message=f"coding tools enabled: {', '.join(registered)}")
        return registered

    def attach_skills(self, path: str = DEFAULT_SKILLS_DIR) -> List[str]:
        """Load reference material agents should follow where it applies.

        A missing directory is not an error -- skills are optional. Call
        again (or with a different path) to add more; existing entries with
        the same name are replaced.
        """
        loaded = self.skills.load_dir(path)
        if loaded:
            self.bus.publish(EventType.LOG, run_id=self.run_id,
                             message=f"skills loaded: {', '.join(loaded)}")
        return loaded

    def attach_connections(self, config_path: str = "connections.json",
                           connections: Optional[List[Any]] = None) -> List[str]:
        """Register named API connections so agents can call those services.

        Each becomes a tool named ``api_<name>``; agents address it with a
        ``TOOL_DIRECTIVE`` giving method, path, query and body. Credentials
        come from the environment and are never shown to the model.
        """
        from .connections import attach_connections, connection_requirements, load_connections

        specs = connections if connections is not None else load_connections(config_path)
        registered = attach_connections(self.tools, specs)
        if specs:
            # Fold their credentials into the readiness report.
            self.requirements.requirements.extend(connection_requirements(specs))
            self.requirements.check(self.tools)
        if registered:
            self.bus.publish(EventType.LOG, run_id=self.run_id,
                             message=f"API connections registered: {', '.join(registered)}")
        return registered

    def attach_mcp(self, config_path: str = ".mcp.json", specs: Optional[List[Any]] = None) -> List[str]:
        """Register tools from MCP servers so agents can call them.

        Discovered tools join the normal registry, so a step can name one via
        ``requires_tool=`` and it participates in the same fallback routing as
        the built-ins. Servers that cannot be reached are skipped, not fatal.

        Returns the names of the tools that were registered.
        """
        from .tools.mcp import McpRegistry

        if self._mcp is None:
            self._mcp = McpRegistry()
        registered = self._mcp.attach(self.tools, specs=specs, config_path=config_path)
        if registered:
            self.bus.publish(EventType.LOG, run_id=self.run_id,
                             message=f"MCP: registered {len(registered)} tool(s): "
                                     f"{', '.join(sorted(registered)[:8])}"
                                     + (" ..." if len(registered) > 8 else ""))
        return registered

    def load_persisted_results(self) -> int:
        """Rehydrate cached step outputs. Returns how many were recovered."""
        if self.state is None:
            return 0
        stored = self.state.load_results(self.run_id)
        recovered = 0
        for step_id, result in stored.items():
            if step_id not in self.graph:
                continue
            self.results[step_id] = result
            if result.status in USABLE and result.output:
                self.memory.store(step_id, result.output)
                recovered += 1
        if recovered:
            self.bus.publish(EventType.LOG, run_id=self.run_id,
                             message=f"recovered {recovered} completed step(s) from storage")
        return recovered

    def load_persisted_actions(self) -> int:
        """Restore irreversible calls that were waiting when the process died."""
        if self.state is None:
            return 0
        restored = {
            step_id: action
            for step_id, action in self.state.load_pending_actions(self.run_id).items()
            if step_id in self.graph
        }
        with self._lock:
            self._pending_actions.update(restored)
        if restored:
            self.bus.publish(
                EventType.LOG, run_id=self.run_id,
                message=f"recovered {len(restored)} action(s) awaiting approval")
        return len(restored)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _should_run(self, step: Step, force: Set[str]) -> tuple:
        """Return ``(should_run, reason)`` for one step."""
        if step.id in force:
            return True, "forced"

        cached = self.results.get(step.id)
        if cached is not None and cached.status is StepStatus.NOT_APPLICABLE:
            return True, "branch is now active"
        if cached is None or cached.status not in USABLE or not cached.output:
            return True, "no cached result"

        if not self.smart_invalidation:
            return False, "cached"

        current_hash = self.memory.input_hash(step)
        if cached.input_hash and cached.input_hash == current_hash:
            return False, "inputs unchanged"
        return True, "inputs changed"

    def _condition_met(self, step: Step) -> tuple[bool, str]:
        """Evaluate a small declarative branch rule without executing code."""
        rule = step.condition
        if not rule:
            return True, ""
        source = str(rule.get("source", "")).strip()
        operator = str(rule.get("operator", "truthy")).strip().lower()
        expected = rule.get("value")
        if source.startswith("trigger."):
            actual: Any = self.runtime_inputs
            for part in source[8:].split("."):
                actual = actual.get(part) if isinstance(actual, dict) else None
        elif source == "trigger":
            actual = self.runtime_inputs
        else:
            actual = self.memory.get(source)
        left, right = str(actual or ""), str(expected or "")
        checks = {
            "contains": lambda: right.casefold() in left.casefold(),
            "not_contains": lambda: right.casefold() not in left.casefold(),
            "equals": lambda: left.casefold() == right.casefold(),
            "not_equals": lambda: left.casefold() != right.casefold(),
            "exists": lambda: actual is not None and actual != "",
            "truthy": lambda: bool(actual),
        }
        if operator not in checks:
            return False, f"unsupported condition operator '{operator}'"
        return checks[operator](), f"{source} {operator} {expected!r}"

    def _blocked_by(self, step: Step) -> List[str]:
        """Dependencies that did not produce a usable result."""
        return [
            dep for dep in step.depends_on
            if self.results.get(dep) is None or self.results[dep].status not in USABLE
        ]

    def _run_plan(self, force: Optional[Set[str]] = None, reason: str = "run") -> RunReport:
        force = set(force or ())
        report = RunReport(self.run_id, reason)
        self.bus.publish(EventType.RUN_STARTED, run_id=self.run_id, message=reason,
                         forced=sorted(force))

        levels = self.graph.execution_levels() if self.parallel else [
            [sid] for sid in self.graph.topological_order()
        ]

        for level in levels:
            if self._cancel_requested.is_set():
                self._cancel_remaining(report)
                break
            runnable: List[Step] = []

            for step_id in level:
                if self._cancel_requested.is_set():
                    self._cancel_remaining(report)
                    break
                step = self.graph.get(step_id)

                # An action approved since the last run completes without
                # asking the LLM to redo the work that produced it.
                pending = self._pending_actions.get(step_id)
                if pending is not None:
                    if step_id in self._approved_actions:
                        self._execute_pending_action(pending, report)
                        continue
                    self._mark(step_id, StepStatus.AWAITING_ACTION)
                    report.awaiting_action.append(step_id)
                    self.bus.publish(
                        EventType.ACTION_AWAITING_APPROVAL, run_id=self.run_id,
                        step_id=step_id,
                        message=f"still awaiting approval for '{pending.tool}'",
                        **pending.to_dict(include_step_id=False))
                    continue

                blockers = self._blocked_by(step)
                if blockers:
                    self._mark(step_id, StepStatus.CANCELLED,
                               error=f"upstream unavailable: {', '.join(blockers)}")
                    report.cancelled.append(step_id)
                    self.bus.publish(EventType.STEP_CANCELLED, run_id=self.run_id,
                                     step_id=step_id,
                                     message=f"skipped, upstream unavailable ({', '.join(blockers)})")
                    continue

                condition_met, condition_text = self._condition_met(step)
                if not condition_met:
                    result = StepResult(
                        step_id=step_id, status=StepStatus.NOT_APPLICABLE,
                        output=f"Branch not taken: {condition_text}",
                        started_at=time.time(), ended_at=time.time(),
                        input_hash=self.memory.input_hash(step),
                    )
                    result.output_hash = content_hash(result.output)
                    self.results[step_id] = result
                    self.memory.store(step_id, result.output)
                    self._persist(result)
                    report.not_applicable.append(step_id)
                    self.bus.publish(
                        EventType.STEP_SKIPPED, run_id=self.run_id, step_id=step_id,
                        message=f"branch not taken ({condition_text})")
                    continue

                should_run, why = self._should_run(step, force)
                if not should_run:
                    self._mark(step_id, StepStatus.SKIPPED)
                    report.reused.append(step_id)
                    self.bus.publish(EventType.STEP_SKIPPED, run_id=self.run_id,
                                     step_id=step_id, message=f"reused cached output ({why})")
                    continue

                # Anything only a human can supply stops the step here. This
                # comes before approval: there is no point approving a step
                # whose parameters are still unknown.
                missing = step.pending_inputs
                if missing:
                    self._mark(step_id, StepStatus.AWAITING_INPUT)
                    report.awaiting_input.append(step_id)
                    self.bus.publish(
                        EventType.STEP_AWAITING_INPUT, run_id=self.run_id, step_id=step_id,
                        message=f"needs {len(missing)} input(s): "
                                f"{', '.join(i.name for i in missing)}",
                        inputs=[i.to_dict() for i in missing])
                    continue

                if step.requires_approval and step_id not in self._approved:
                    self._mark(step_id, StepStatus.PAUSED)
                    report.awaiting_approval.append(step_id)
                    self.bus.publish(EventType.STEP_AWAITING_APPROVAL, run_id=self.run_id,
                                     step_id=step_id,
                                     message="waiting for approval; call approve() then resume()")
                    continue

                runnable.append(step)

            if self._cancel_requested.is_set():
                self._cancel_remaining(report)
                break
            self._execute_level(runnable, report)
            if self._cancel_requested.is_set():
                self._cancel_remaining(report)
                break

        report.ended_at = time.time()
        self.reports.append(report)

        if report.failed:
            status = "failed"
        elif report.cancel_requested:
            status = "cancelled"
        elif report.waiting_on_human:
            status = "paused"
        else:
            status = "done"
        if self.state is not None:
            self.state.update_run(self.run_id, status=status, graph=self.graph)
        summary = report.to_dict()
        summary.pop("run_id", None)  # already carried by the event itself
        self.bus.publish(
            EventType.RUN_FINISHED, run_id=self.run_id,
            message=(f"{len(report.executed)} executed, {len(report.reused)} reused, "
                     f"{len(report.failed)} failed in {report.duration_s:.2f}s"),
            **summary,
        )
        # A cancellation applies to one execution attempt. A later Resume can
        # continue from the completed results without requiring another API.
        self._cancel_requested.clear()
        return report

    def _cancel_remaining(self, report: RunReport) -> None:
        """Mark work that has not started as cancelled by the operator."""
        report.cancel_requested = True
        for step_id in self.graph.topological_order():
            result = self.results.get(step_id)
            if result is None or result.status in {StepStatus.PENDING, StepStatus.STALE}:
                self._mark(step_id, StepStatus.CANCELLED, error="run cancelled by user")
                if step_id not in report.cancelled:
                    report.cancelled.append(step_id)
                self.bus.publish(
                    EventType.STEP_CANCELLED, run_id=self.run_id, step_id=step_id,
                    message="not started because the run was cancelled")

    def _execute_level(self, steps: List[Step], report: RunReport) -> None:
        if not steps:
            return
        if len(steps) == 1 or not self.parallel:
            for step in steps:
                self._execute_step(step, report)
            return

        workers = min(self.max_workers, len(steps))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="step") as pool:
            futures = {pool.submit(self._execute_step, step, report): step for step in steps}
            for future in futures:
                # _execute_step never raises: it records failures on the result.
                future.result()

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def _execute_step(self, step: Step, report: RunReport) -> StepResult:
        result = StepResult(step_id=step.id, status=StepStatus.RUNNING, started_at=time.time())
        result.tool_requested = step.requires_tool
        result.input_hash = self.memory.input_hash(step)
        with self._lock:
            self.results[step.id] = result

        self.bus.publish(EventType.STEP_STARTED, run_id=self.run_id, step_id=step.id,
                         message=f"{step.agent_role}: {step.description[:90]}",
                         role=step.agent_role, tool=step.requires_tool)

        agent = self.agents.get(step.agent_role)
        context = self.memory.build_context(step, self.graph)

        last_error: Optional[BaseException] = None
        for attempt in range(step.max_retries + 1):
            result.attempts = attempt + 1
            skills_block = self.skills.render_for(step)
            try:
                outcome = self._call_agent(agent, step, context, self._action_gate, skills_block)
            except Exception as exc:  # noqa: BLE001 - classify, then decide
                last_error = exc
                retriable = self._classify(exc) is ErrorClass.RETRIABLE
                if retriable and attempt < step.max_retries:
                    delay = step.retry_backoff_s * (2 ** attempt)
                    self.bus.publish(EventType.STEP_RETRY, run_id=self.run_id, step_id=step.id,
                                     message=f"attempt {attempt + 1} failed ({exc}); retrying in {delay:.1f}s",
                                     attempt=attempt + 1)
                    time.sleep(delay)
                    continue
                break

            # -- held for approval -------------------------------------
            if outcome.is_deferred:
                self._defer_action(step, outcome, result, report)
                return result

            # -- success ------------------------------------------------
            self._capture_handoffs(step, outcome.output)
            result.status = StepStatus.DONE
            result.output = outcome.output
            result.usage = outcome.usage
            result.ended_at = time.time()
            result.output_hash = content_hash(outcome.output)
            if outcome.tool_invocation is not None:
                invocation = outcome.tool_invocation
                result.tool_used = invocation.tool_used
                result.used_fallback = invocation.used_fallback
                self.bus.publish(EventType.TOOL_CALLED, run_id=self.run_id, step_id=step.id,
                                 message=f"{invocation.tool_used}: {invocation.output[:120]}",
                                 **invocation.to_dict())
                if invocation.used_fallback:
                    self.bus.publish(
                        EventType.TOOL_FALLBACK, run_id=self.run_id, step_id=step.id,
                        message=f"'{step.requires_tool}' unavailable; used '{invocation.tool_used}'",
                        **invocation.to_dict())

            self.memory.store(step.id, outcome.output)
            with self._lock:
                self.results[step.id] = result
                report.executed.append(step.id)
                report.usage = report.usage + outcome.usage
            self._persist(result)
            self.bus.publish(EventType.STEP_FINISHED, run_id=self.run_id, step_id=step.id,
                             message=f"done in {result.duration_s:.2f}s "
                                     f"({result.usage.total_tokens} tokens)",
                             duration_s=round(result.duration_s, 4),
                             preview=outcome.output[:400],
                             usage=result.usage.to_dict())
            return result

        # -- exhausted ---------------------------------------------------
        result.status = StepStatus.FAILED
        result.error = str(last_error) if last_error else "unknown error"
        result.error_class = self._classify(last_error) if last_error else ErrorClass.FATAL
        result.ended_at = time.time()
        with self._lock:
            self.results[step.id] = result
            report.failed.append(step.id)
        self._persist(result)
        self.bus.publish(EventType.STEP_FAILED, run_id=self.run_id, step_id=step.id,
                         message=f"failed after {result.attempts} attempt(s): {result.error}",
                         error=result.error, error_class=result.error_class.value)
        return result

    def _call_agent(self, agent: Any, step: Step, context: str,
                    gate: Optional[Any] = None, skills_block: str = "") -> Any:
        """Invoke the agent, honouring ``step.timeout_s`` if one is set.

        The timeout is enforced by abandoning the worker thread rather than
        killing it -- Python cannot safely interrupt arbitrary code -- so a
        timed-out call may keep running in the background until it returns.
        """
        if self.runtime_inputs:
            context += ("\n\n### Data supplied by the workflow trigger\n```json\n"
                        + json.dumps(self.runtime_inputs, indent=2, default=str)
                        + "\n```")
        messages = [m for m in self.agent_messages if m.get("to") in {step.id, step.agent_role}]
        if messages:
            context += "\n\n### Direct handoffs from other agents\n" + "\n".join(
                f"- {m.get('from')}: {m.get('message')}" for m in messages)
        if not step.timeout_s:
            return agent.execute(step, context, self.tools, gate, skills_block,
                                 self.all_input_values())

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="timeout") as pool:
            future = pool.submit(agent.execute, step, context, self.tools, gate,
                                 skills_block, self.all_input_values())
            try:
                return future.result(timeout=step.timeout_s)
            except FutureTimeout as exc:
                future.cancel()
            raise TimeoutError(f"step '{step.id}' exceeded {step.timeout_s}s") from exc

    def _capture_handoffs(self, step: Step, output: str) -> None:
        """Store valid HANDOFF JSON lines addressed to a step or role."""
        for raw in re.findall(r"(?m)^HANDOFF:\s*(\{.*\})\s*$", output):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            target, message = str(data.get("to", "")).strip(), str(data.get("message", "")).strip()
            valid = target in self.graph or any(
                self.graph.get(sid).agent_role == target for sid in self.graph.topological_order())
            if not valid or not message:
                continue
            record = {"from": step.id, "to": target, "message": message[:4000]}
            with self._lock:
                self.agent_messages.append(record)
            self.bus.publish(EventType.AGENT_MESSAGE, run_id=self.run_id, step_id=step.id,
                             message=f"handoff to {target}", to=target,
                             message_text=record["message"])

    def _execute_pending_action(self, action: PendingAction, report: RunReport) -> StepResult:
        """Perform an approved action, reusing the already-generated payload."""
        step = self.graph.get(action.step_id)
        agent = self.agents.get(step.agent_role)
        result = self.results.get(action.step_id) or StepResult(step_id=action.step_id)
        result.started_at = result.started_at or time.time()
        result.tool_requested = action.tool
        result.input_hash = self.memory.input_hash(step)

        try:
            invocation = agent.call_tool(step, action.tool, action.payload, self.tools,
                                         self.all_input_values())
        except Exception as exc:  # noqa: BLE001 - a failed action is a failed step
            result.status = StepStatus.FAILED
            result.error = str(exc)
            result.error_class = self._classify(exc)
            result.ended_at = time.time()
            with self._lock:
                self.results[action.step_id] = result
                report.failed.append(action.step_id)
                self._pending_actions.pop(action.step_id, None)
            if self.state is not None:
                self.state.delete_pending_action(self.run_id, action.step_id)
            self._persist(result)
            self.bus.publish(EventType.STEP_FAILED, run_id=self.run_id,
                             step_id=action.step_id,
                             message=f"approved action failed: {exc}")
            return result

        result.status = StepStatus.DONE
        result.output = agent.merge_tool_result(action.agent_output, action.tool, invocation)
        result.usage = action.usage
        result.tool_used = invocation.tool_used
        result.used_fallback = invocation.used_fallback
        result.ended_at = time.time()
        result.output_hash = content_hash(result.output)

        self.memory.store(action.step_id, result.output)
        with self._lock:
            self.results[action.step_id] = result
            report.executed.append(action.step_id)
            self._pending_actions.pop(action.step_id, None)
        if self.state is not None:
            self.state.delete_pending_action(self.run_id, action.step_id)
        self._persist(result)

        self.bus.publish(EventType.ACTION_EXECUTED, run_id=self.run_id, step_id=action.step_id,
                         message=f"performed via '{invocation.tool_used}': "
                                 f"{invocation.output[:120]}",
                         **invocation.to_dict())
        self.bus.publish(EventType.STEP_FINISHED, run_id=self.run_id, step_id=action.step_id,
                         message="completed after approval (no LLM call repeated)",
                         duration_s=round(result.duration_s, 4),
                         preview=result.output[:400], usage=result.usage.to_dict())
        return result

    # -- real-world action gating -----------------------------------------

    def _action_gate(self, step: Step, tool_name: str, payload: str) -> bool:
        """Decide whether a tool call may proceed right now.

        Returns False for an irreversible action that has not been approved,
        which parks it as a :class:`PendingAction` instead of performing it.

        The default mode gates only actions that would be *real*. A tool
        running in simulation sends no email and writes to no database, so
        stopping to approve it is pure friction -- and would mean nobody ever
        sees the gate until the day they add credentials. Use ``"all"`` to
        rehearse the approval flow without credentials configured.
        """
        if self.action_approval == "never":
            return True
        tool = self.tools.get(tool_name)
        # Asked per call, not per tool: the same HTTP connection is harmless
        # for a GET and unundoable for a DELETE.
        if tool is None or not tool.is_irreversible(payload, {"step_id": step.id}):
            return True
        if self.action_approval == "live" and not tool.is_live():
            return True
        return step.id in self._approved_actions

    def _defer_action(self, step: Step, outcome: Any, result: StepResult,
                      report: RunReport) -> None:
        """Park an unapproved irreversible action and pause the step."""
        tool_name = outcome.deferred_tool
        tool = self.tools.get(tool_name)
        preview = tool.preview(outcome.output, {"step_id": step.id, "step_name": step.name}) \
            if tool is not None else f"{tool_name}: (no preview available)"

        action = PendingAction(
            step_id=step.id, tool=tool_name, payload=outcome.output,
            preview=preview,
            irreversible=bool(tool and tool.is_irreversible(outcome.output,
                                                            {"step_id": step.id})),
            agent_output=outcome.output, usage=outcome.usage,
        )
        with self._lock:
            self._pending_actions[step.id] = action
            report.awaiting_action.append(step.id)
        if self.state is not None:
            self.state.save_pending_action(self.run_id, action)

        result.status = StepStatus.AWAITING_ACTION
        result.ended_at = time.time()
        result.usage = outcome.usage
        with self._lock:
            self.results[step.id] = result
        self._persist(result)

        self.bus.publish(
            EventType.ACTION_AWAITING_APPROVAL, run_id=self.run_id, step_id=step.id,
            message=f"about to do something irreversible via '{tool_name}' -- "
                    f"approve_action('{step.id}') to proceed",
            **action.to_dict(include_step_id=False))

    @staticmethod
    def _classify(exc: Optional[BaseException]) -> ErrorClass:
        if exc is None:
            return ErrorClass.FATAL
        if isinstance(exc, (TimeoutError, FutureTimeout, ToolUnavailableError)):
            return ErrorClass.RETRIABLE
        if isinstance(exc, LLMError) or is_transient(exc):
            return ErrorClass.RETRIABLE
        return ErrorClass.FATAL

    def _mark(self, step_id: str, status: StepStatus, error: Optional[str] = None) -> None:
        with self._lock:
            result = self.results.get(step_id) or StepResult(step_id=step_id)
            result.status = status
            if error:
                result.error = error
            if status is StepStatus.SKIPPED:
                # Reused output stays valid; refresh the signature so the
                # next comparison is against current inputs.
                result.input_hash = self.memory.input_hash(self.graph.get(step_id))
            self.results[step_id] = result
        self._persist(self.results[step_id])

    def _persist(self, result: StepResult) -> None:
        if self.state is not None:
            self.state.save_step_result(self.run_id, result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full(self, fresh: bool = True,
                 runtime_inputs: Optional[Dict[str, Any]] = None) -> RunReport:
        """Run every step. With ``fresh=False`` this resumes from cache."""
        if runtime_inputs is not None:
            self.runtime_inputs = dict(runtime_inputs)
        if fresh:
            self.memory.clear()
            self.results = {sid: StepResult(step_id=sid) for sid in self.graph.topological_order()}
            with self._lock:
                self._pending_actions.clear()
                self._approved_actions.clear()
            if self.state is not None:
                self.state.clear_results(self.run_id)
                self.state.clear_pending_actions(self.run_id)
        return self._run_plan(force=set(self.graph.topological_order()) if fresh else set(),
                              reason="full run" if fresh else "resume")

    def run(self, only: Optional[Iterable[str]] = None) -> RunReport:
        """Run whatever is not already valid, optionally forcing some steps."""
        return self._run_plan(force=set(only or ()), reason="run")

    def resume(self) -> RunReport:
        """Continue a partially completed run (after a crash, or an approval)."""
        self.load_persisted_results()
        return self._run_plan(reason="resume")

    def request_cancel(self) -> bool:
        """Request a cooperative stop; return False if already requested.

        Python cannot safely kill an arbitrary model or tool call. The active
        call finishes, then the scheduler cancels every step that has not yet
        started. Completed outputs remain available for a later Resume.
        """
        if self._cancel_requested.is_set():
            return False
        self._cancel_requested.set()
        self.bus.publish(
            EventType.RUN_CANCEL_REQUESTED, run_id=self.run_id,
            message="cancellation requested; finishing active work, then stopping")
        return True

    @property
    def cancellation_requested(self) -> bool:
        return self._cancel_requested.is_set()

    # -- scenario: requirement change ----------------------------------

    def impact_of(self, step_id: str) -> Dict[str, Any]:
        """What a change to ``step_id`` would touch -- without running anything."""
        affected = self.graph.downstream_of(step_id)
        all_ids = set(self.graph.topological_order())
        return {
            "changed": step_id,
            "affected": sorted(affected),
            "reusable": sorted(all_ids - affected),
            "affected_count": len(affected),
            "reusable_count": len(all_ids - affected),
            "reuse_ratio": round(len(all_ids - affected) / len(all_ids), 4) if all_ids else 0.0,
        }

    def handle_step_change(self, step_id: str, new_description: Optional[str] = None,
                           new_requirement: Optional[str] = None) -> RunReport:
        """A requirement changed. Re-run the minimum necessary.

        ``new_description`` replaces the step's description outright;
        ``new_requirement`` asks the planner to fold a change into it.
        Supplying neither just invalidates the step as-is.
        """
        if step_id not in self.graph:
            raise KeyError(f"no such step: '{step_id}'")
        step = self.graph.get(step_id)

        if new_description:
            step.description = new_description
        elif new_requirement:
            from .planner import TaskPlanner

            step.description = TaskPlanner(llm=self.llm).revise_step(step, new_requirement)

        impact = self.impact_of(step_id)
        self.bus.publish(EventType.CHANGE_REQUESTED, run_id=self.run_id, step_id=step_id,
                         message=new_description or new_requirement or "step invalidated",
                         description=step.description)
        self.bus.publish(EventType.IMPACT_ANALYSED, run_id=self.run_id, step_id=step_id,
                         message=(f"{impact['affected_count']} step(s) potentially affected, "
                                  f"{impact['reusable_count']} reusable"),
                         **impact)

        if self.state is not None:
            self.state.update_run(self.run_id, graph=self.graph)

        # Only the changed step is forced. Everything downstream re-runs only
        # if its inputs actually changed -- unless smart invalidation is off,
        # in which case the whole downstream cone is forced.
        force = {step_id} if self.smart_invalidation else set(impact["affected"])
        return self._run_plan(force=force, reason=f"change:{step_id}")

    def update_step(self, step_id: str, *, description: Optional[str] = None,
                    agent_role: Optional[str] = None,
                    requires_tool: Optional[str] = None,
                    depends_on: Optional[List[str]] = None,
                    name: Optional[str] = None,
                    requires_approval: Optional[bool] = None,
                    inputs: Optional[List[Any]] = None,
                    condition: Optional[Dict[str, Any]] = None,
                    rerun: bool = False) -> Dict[str, Any]:
        """Edit any part of a step, validating the graph before committing.

        Changes are applied to a copy first: an edit that would create a cycle
        or point at a missing step leaves the workflow exactly as it was,
        rather than half-applied and unrunnable.

        Anything that changes what the step *produces* clears its cached
        result, so the normal input-signature machinery re-runs it and whatever
        genuinely depends on it. ``rerun=True`` executes immediately;
        otherwise the change is staged for the next run.
        """
        if step_id not in self.graph:
            raise KeyError(f"no such step: '{step_id}'")
        step = self.graph.get(step_id)

        before = step.to_dict()
        if depends_on is not None:
            # Validate on a copy so a bad edit cannot corrupt the live graph.
            trial = DependencyGraph.from_dict(self.graph.to_dict())
            trial.get(step_id).depends_on = [d for d in depends_on if d != step_id]
            missing = [d for d in trial.get(step_id).depends_on if d not in trial]
            if missing:
                raise ValueError(f"unknown dependency: {', '.join(missing)}")
            cycle = trial.find_cycle()
            if cycle:
                raise ValueError("that would create a loop: " + " -> ".join(cycle))
            step.depends_on = trial.get(step_id).depends_on

        if description is not None:
            step.description = description
        if agent_role is not None:
            step.agent_role = agent_role
        if name is not None:
            step.name = name
        if requires_approval is not None:
            step.requires_approval = requires_approval
            if not requires_approval:
                self._approved.discard(step_id)
        if requires_tool is not None:
            from .tools import canonical_tool_name

            step.requires_tool = canonical_tool_name(requires_tool) or None
        if inputs is not None:
            from .inputs import parse_declared_inputs

            step.inputs = parse_declared_inputs(
                [i if isinstance(i, dict) else i.to_dict() for i in inputs])
        if condition is not None:
            if condition:
                # Reuse Step's validation rather than accepting executable text.
                Step(id="_condition_check", description="check", agent_role="generic",
                     condition=condition)
                step.condition = dict(condition)
            else:
                step.condition = None

        self.graph.validate()
        changed = {k: v for k, v in step.to_dict().items() if before.get(k) != v}

        if changed:
            # Drop the cached result so this step (and anything whose inputs
            # really change as a result) runs again.
            with self._lock:
                self.results[step_id] = StepResult(step_id=step_id)
                self._pending_actions.pop(step_id, None)
            self.memory.forget(step_id)
            if self.state is not None:
                self.state.clear_results(self.run_id, [step_id])
                self.state.delete_pending_action(self.run_id, step_id)
                self.state.update_run(self.run_id, graph=self.graph)
            self.bus.publish(EventType.CHANGE_REQUESTED, run_id=self.run_id, step_id=step_id,
                             message=f"step edited: {', '.join(sorted(changed))}",
                             changed=sorted(changed))

        result: Dict[str, Any] = {"step_id": step_id, "changed": sorted(changed),
                                  "impact": self.impact_of(step_id)}
        if rerun and changed:
            result["report"] = self._run_plan(force={step_id},
                                              reason=f"edit:{step_id}").to_dict()
        return result

    def remove_step(self, step_id: str) -> Dict[str, Any]:
        """Delete a step and detach it from anything that depended on it."""
        if step_id not in self.graph:
            raise KeyError(f"no such step: '{step_id}'")

        dependents = sorted(self.graph.downstream_of(step_id, inclusive=False))
        self.graph.remove(step_id)
        self.graph.validate()

        with self._lock:
            self.results.pop(step_id, None)
            self._pending_actions.pop(step_id, None)
            self._approved.discard(step_id)
            self._approved_actions.discard(step_id)
            # Former dependents lost an input, so their cached work is stale.
            for other in dependents:
                self.results[other] = StepResult(step_id=other)
                self.memory.forget(other)
                self._pending_actions.pop(other, None)
        self.memory.forget(step_id)

        if self.state is not None:
            self.state.clear_results(self.run_id, [step_id, *dependents])
            for removed_action in [step_id, *dependents]:
                self.state.delete_pending_action(self.run_id, removed_action)
            self.state.update_run(self.run_id, graph=self.graph)
        self.bus.publish(EventType.LOG, run_id=self.run_id,
                         message=f"step '{step_id}' removed; {len(dependents)} "
                                 "dependent step(s) invalidated")
        return {"removed": step_id, "invalidated": dependents,
                "steps": self.graph.topological_order()}

    def add_step(self, step: Step, rerun: bool = True) -> Any:
        """Insert a new step into a live workflow.

        ``rerun=True`` (the default, kept for compatibility) executes it
        straight away and returns a :class:`RunReport`. Pass ``rerun=False``
        while assembling a workflow by hand -- otherwise each added step runs
        before the next one exists, which is rarely what an editor wants.
        """
        self.graph.add(step)
        try:
            self.graph.validate()
        except Exception:
            self.graph.remove(step.id)  # never leave the graph unrunnable
            raise
        with self._lock:
            self.results.setdefault(step.id, StepResult(step_id=step.id))
        if self.state is not None:
            self.state.update_run(self.run_id, graph=self.graph)
        self.bus.publish(EventType.LOG, run_id=self.run_id, step_id=step.id,
                         message=f"step '{step.id}' added to the workflow")
        if not rerun:
            return {"added": step.id, "steps": self.graph.topological_order()}
        return self._run_plan(force={step.id}, reason=f"add:{step.id}")

    # -- scenario: tool failure ----------------------------------------

    def handle_tool_failure(self, tool: str, rerun: bool = True) -> Optional[RunReport]:
        """Break a tool and re-run the work that depended on it."""
        self.tools.break_tool(tool)
        seeds = self.graph.nodes_using_tool(tool)
        affected = self.graph.affected_by(seeds) if seeds else set()
        self.bus.publish(EventType.TOOL_BROKEN, run_id=self.run_id,
                         message=(f"'{tool}' marked broken; {len(seeds)} step(s) use it, "
                                  f"{len(affected)} affected"),
                         tool=tool, users=sorted(seeds), affected=sorted(affected))
        if not rerun or not seeds:
            return None
        # As with a requirement change: force only the steps that touch the
        # tool and let signature comparison carry the change downstream. Without
        # smart invalidation the whole affected cone has to be forced, or
        # downstream steps would keep outputs derived from the failed tool.
        force = set(seeds) if self.smart_invalidation else affected
        return self._run_plan(force=force, reason=f"tool-failure:{tool}")

    def handle_tool_repair(self, tool: str, rerun: bool = False) -> Optional[RunReport]:
        """Mark a tool healthy again, optionally re-running steps that fell back."""
        self.tools.repair_tool(tool)
        self.bus.publish(EventType.TOOL_REPAIRED, run_id=self.run_id,
                         message=f"'{tool}' repaired", tool=tool)
        if not rerun:
            return None
        fell_back = {
            sid for sid, r in self.results.items()
            if r.used_fallback and r.tool_requested == tool
        }
        if not fell_back:
            return None
        return self._run_plan(force=fell_back, reason=f"tool-repair:{tool}")

    # -- scenario: human in the loop -----------------------------------

    def pause_before(self, step_id: str) -> None:
        """Require explicit approval before this step runs."""
        self.graph.get(step_id).requires_approval = True
        self._approved.discard(step_id)
        if self.state is not None:
            self.state.update_run(self.run_id, graph=self.graph)

    #: The spec calls this ``handle_step_pause``; kept as an alias.
    handle_step_pause = pause_before

    def approve(self, step_id: str) -> None:
        if step_id not in self.graph:
            raise KeyError(f"no such step: '{step_id}'")
        self._approved.add(step_id)
        self.bus.publish(EventType.STEP_APPROVED, run_id=self.run_id, step_id=step_id,
                         message="approved by operator")

    def awaiting_approval(self) -> List[str]:
        return sorted(sid for sid, r in self.results.items() if r.status is StepStatus.PAUSED)

    # -- human input ----------------------------------------------------

    def pending_inputs(self) -> Dict[str, List[Any]]:
        """Everything the workflow needs from you, keyed by step id.

        Covers the whole graph, not just steps reached so far, so a UI can
        ask for everything predictable up front instead of interrupting
        repeatedly.
        """
        pending: Dict[str, List[Any]] = {}
        for step_id in self.graph.topological_order():
            missing = self.graph.get(step_id).pending_inputs
            if missing:
                pending[step_id] = missing
        return pending

    def all_input_values(self) -> Dict[str, Any]:
        """Every answered input across the whole workflow.

        Planners routinely collect a fact on one step ("get the customer's
        email") and use it on another ("send the email"). Scoping answers to
        the step that asked would make those collections useless, so tools see
        the workflow-wide set, with the running step's own answers taking
        precedence on any name clash.
        """
        merged: Dict[str, Any] = dict(self.runtime_inputs)
        if self.runtime_inputs:
            merged["trigger"] = dict(self.runtime_inputs)
        for step_id in self.graph.topological_order():
            merged.update(self.graph.get(step_id).input_values())
        return merged

    def input_form(self) -> Dict[str, List[Any]]:
        """The full question set for every step that is waiting on input.

        Differs from :meth:`pending_inputs` on purpose: that returns only what
        *blocks* a step, which is what the scheduler needs. A form should also
        offer the optional fields and the ones with defaults, or there is no
        way to override them.
        """
        form: Dict[str, List[Any]] = {}
        for step_id in self.graph.topological_order():
            step = self.graph.get(step_id)
            if step.pending_inputs:
                form[step_id] = list(step.inputs)
        return form

    def provide_input(self, step_id: str, name: str, value: Any) -> Any:
        """Answer one question. Raises InputError if the value is invalid."""
        if step_id not in self.graph:
            raise KeyError(f"no such step: '{step_id}'")
        step = self.graph.get(step_id)
        coerced = step.provide_input(name, value)

        # An input feeds the step's signature, so a cached result computed
        # with a different answer must not be reused.
        result = self.results.get(step_id)
        if result is not None and result.status in USABLE:
            result.input_hash = ""

        request = next(i for i in step.inputs if i.name == name)

        # A credential asked for by ask_for_missing_requirements() only
        # unblocks the tool once it is actually in the environment -- the
        # tools read os.environ directly in their is_live() checks.
        if self._requirement_env.get(name) and coerced not in (None, ""):
            os.environ[name] = str(coerced)
            # Some tools read os.environ directly, others go through the
            # settings object built at import; refresh it so both see this.
            from .config import reload_settings

            reload_settings()
            self.check_requirements()

        self.bus.publish(
            EventType.STEP_INPUT_PROVIDED, run_id=self.run_id, step_id=step_id,
            message=f"{name} = {'***' if request.is_secret else coerced}",
            name=name, secret=request.is_secret)
        if self.state is not None:
            self.state.update_run(self.run_id, graph=self.graph)
        return coerced

    def provide_inputs(self, values: Dict[str, Any],
                       step_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer several questions at once, atomically.

        With ``step_id`` the keys are input names for that step. Without it,
        keys may be plain names (applied to every step that declares them) or
        ``"step_id.input_name"`` for precision.

        Every value is validated **before** any is applied, so a form with one
        bad field leaves the workflow exactly as it was. Without that, a
        rejected submission would still have half-updated the run.
        """
        targets: List[tuple] = []
        for key, raw in values.items():
            if step_id is not None:
                targets.append((step_id, key, raw))
            elif "." in key:
                target, _, name = key.partition(".")
                targets.append((target, name, raw))
            else:
                for candidate in self.graph.topological_order():
                    if any(i.name == key for i in self.graph.get(candidate).inputs):
                        targets.append((candidate, key, raw))

        # Validate everything first; any failure aborts the whole submission.
        for target_step, name, raw in targets:
            if target_step not in self.graph:
                raise KeyError(f"no such step: '{target_step}'")
            step = self.graph.get(target_step)
            request = next((i for i in step.inputs if i.name == name), None)
            if request is None:
                raise KeyError(f"step '{target_step}' has no input named '{name}'")
            request.coerce(raw)  # raises InputError, mutating nothing

        applied: Dict[str, Any] = {}
        for target_step, name, raw in targets:
            applied[f"{target_step}.{name}"] = self.provide_input(target_step, name, raw)
        return applied

    def awaiting_input(self) -> List[str]:
        return sorted(sid for sid, r in self.results.items()
                      if r.status is StepStatus.AWAITING_INPUT)

    # -- irreversible actions --------------------------------------------

    def pending_actions(self) -> Dict[str, PendingAction]:
        """Real-world actions held back, waiting for sign-off."""
        with self._lock:
            return dict(self._pending_actions)

    def approve_action(self, step_id: str) -> None:
        """Authorise the held action for this step. Resume to perform it."""
        if step_id not in self.graph:
            raise KeyError(f"no such step: '{step_id}'")
        self._approved_actions.add(step_id)
        action = self._pending_actions.get(step_id)
        self.bus.publish(EventType.ACTION_APPROVED, run_id=self.run_id, step_id=step_id,
                         message=f"action approved: {action.preview if action else step_id}")

    def reject_action(self, step_id: str, reason: str = "") -> None:
        """Refuse the held action. The step fails rather than acting."""
        with self._lock:
            action = self._pending_actions.pop(step_id, None)
        self._approved_actions.discard(step_id)
        if action is None:
            raise KeyError(f"no pending action for step '{step_id}'")
        if self.state is not None:
            self.state.delete_pending_action(self.run_id, step_id)

        result = self.results.get(step_id) or StepResult(step_id=step_id)
        result.status = StepStatus.FAILED
        result.error = f"action rejected by operator{': ' + reason if reason else ''}"
        result.error_class = ErrorClass.FATAL
        result.ended_at = time.time()
        with self._lock:
            self.results[step_id] = result
        self._persist(result)
        self.bus.publish(EventType.ACTION_REJECTED, run_id=self.run_id, step_id=step_id,
                         message=result.error)

    def awaiting_action(self) -> List[str]:
        return sorted(self._pending_actions)

    def blocked_on_human(self) -> Dict[str, List[str]]:
        """Everything currently waiting on a person, grouped by what it needs."""
        return {
            "inputs": self.awaiting_input(),
            "approvals": self.awaiting_approval(),
            "actions": self.awaiting_action(),
        }

    # -- queries --------------------------------------------------------

    def query_results(self, step_id: str) -> str:
        if step_id not in self.results:
            raise KeyError(f"no such step: '{step_id}'")
        return self.results[step_id].output

    def get_execution_time(self, step_id: str) -> float:
        return self.results[step_id].duration_s

    def status_of(self, step_id: str) -> StepStatus:
        return self.results[step_id].status

    def outputs(self) -> Dict[str, str]:
        return {sid: r.output for sid, r in self.results.items() if r.output}

    def metrics(self) -> Dict[str, Any]:
        results = list(self.results.values())
        usage = LLMUsage()
        for r in results:
            usage = usage + r.usage
        wall = sum(r.duration_s for r in results)
        by_status: Dict[str, int] = {}
        for r in results:
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        return {
            "run_id": self.run_id,
            "steps": len(results),
            "by_status": by_status,
            "total_step_seconds": round(wall, 4),
            "slowest_step": max(
                ((r.step_id, round(r.duration_s, 4)) for r in results),
                key=lambda x: x[1], default=(None, 0.0),
            ),
            "usage": usage.to_dict(),
            "tools": self.tools.stats(),
            "runs": [r.to_dict() for r in self.reports],
            "parallelism": {
                "enabled": self.parallel,
                "max_workers": self.max_workers,
                "levels": self.graph.execution_levels(),
                "critical_path_length": len(self.graph.execution_levels()),
            },
        }

    # -- reporting ------------------------------------------------------

    def print_summary(self, width: int = 100) -> None:
        symbols = {
            StepStatus.DONE: "[done]", StepStatus.SKIPPED: "[reused]",
            StepStatus.FAILED: "[FAILED]", StepStatus.CANCELLED: "[cancelled]",
            StepStatus.PAUSED: "[paused]", StepStatus.PENDING: "[pending]",
            StepStatus.RUNNING: "[running]", StepStatus.STALE: "[stale]",
        }
        print("\n" + "=" * width)
        print(f"WORKFLOW {self.run_id}" + (f" -- {self.description[:60]}" if self.description else ""))
        print("=" * width)
        for step_id in self.graph.topological_order():
            step = self.graph.get(step_id)
            result = self.results[step_id]
            deps = ", ".join(step.depends_on) or "-"
            tool = f" via {result.tool_used or step.requires_tool}" if step.requires_tool else ""
            if result.used_fallback:
                tool += " (FALLBACK)"
            print(f"\n{symbols.get(result.status, '[?]'):>12} {step_id}  "
                  f"({step.agent_role}; deps: {deps}{tool}; {result.duration_s:.2f}s)")
            body = result.error or result.output or "(no output)"
            for line in body.strip().splitlines()[:6]:
                print(f"             {line[:width - 14]}")
        metrics = self.metrics()
        print("\n" + "-" * width)
        print(f"tokens: {metrics['usage']['total_tokens']}  "
              f"cost: ${metrics['usage']['cost_usd']:.6f}  "
              f"llm calls: {metrics['usage']['calls']}  "
              f"step time: {metrics['total_step_seconds']:.2f}s")
        print("=" * width)

    def export_results(self, path: str = "results.json") -> str:
        from .export import export_json

        return export_json(self, path)

    def export_state(self, path: Optional[str] = None, format: str = "json") -> str:
        from .export import export_csv, export_html, export_json

        exporters = {"json": export_json, "html": export_html, "csv": export_csv}
        if format not in exporters:
            raise ValueError(f"unknown format '{format}'; expected one of {sorted(exporters)}")
        default_names = {"json": "results.json", "html": "dashboard.html", "csv": "results.csv"}
        return exporters[format](self, path or default_names[format])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "description": self.description,
            "graph": self.graph.to_dict(),
            "results": {sid: r.to_dict() for sid, r in self.results.items()},
            "metrics": self.metrics(),
            "tools": self.tools.describe(),
            "events": [e.to_dict() for e in self.bus.history],
        }

    def close(self) -> None:
        if self._mcp is not None:
            self._mcp.close_all()
            self._mcp = None
        if self.state is not None:
            self.state.close()
            self.state = None

    def __enter__(self) -> "Workflow":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
