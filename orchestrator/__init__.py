"""Adaptive AI Task Orchestrator.

A multi-agent system that plans a project from plain English, executes it as a
dependency graph of role-specialised agents, routes around broken tools, and
re-runs only the work a change actually invalidated.

Quick start::

    from orchestrator import Workflow

    wf = Workflow.from_description("Build a task manager with auth and a Kanban board")
    wf.run_full()
    wf.handle_step_change("database", new_requirement="use MongoDB instead")
    wf.print_summary()
    wf.export_state("dashboard.html", format="html")
"""

from .agents import (
    Agent,
    AgentManager,
    ROLE_PROMPTS,
    normalise_role,
    register_role,
    resolve_role,
)
from .config import Settings, settings
from .events import Event, EventBus, EventType
from .export import export_csv, export_html, export_json
from .graph import (
    CircularDependencyError,
    DependencyGraph,
    GraphError,
    MissingDependencyError,
)
from .llm import (
    AnthropicProvider,
    GeminiProvider,
    LLMError,
    LLMProvider,
    NvidiaProvider,
    OllamaProvider,
    OpenAIProvider,
    StubProvider,
    build_provider,
    get_provider,
    set_provider,
)
from .memory import MemoryManager
from .models import ErrorClass, LLMUsage, Step, StepResult, StepStatus
from .planner import PlanResult, PlanningError, TaskPlanner, plan_from_json
from .requirements import (
    AgentSpec,
    Requirement,
    RequirementKind,
    RequirementStatus,
    RequirementsReport,
    infer_requirements,
)
from .state import StateManager
from .tools import Tool, ToolManager, ToolUnavailableError, default_tool_manager
from .workflow import ApprovalRequired, RunReport, Workflow

__version__ = "1.0.0"

__all__ = [
    "Workflow", "RunReport", "ApprovalRequired",
    "Step", "StepResult", "StepStatus", "ErrorClass", "LLMUsage",
    "DependencyGraph", "GraphError", "CircularDependencyError", "MissingDependencyError",
    "TaskPlanner", "PlanResult", "PlanningError", "plan_from_json",
    "Requirement", "RequirementKind", "RequirementStatus", "RequirementsReport",
    "AgentSpec", "infer_requirements",
    "Agent", "AgentManager", "ROLE_PROMPTS", "register_role", "normalise_role",
    "resolve_role",
    "Tool", "ToolManager", "ToolUnavailableError", "default_tool_manager",
    "MemoryManager", "StateManager",
    "EventBus", "Event", "EventType",
    "LLMProvider", "GeminiProvider", "AnthropicProvider", "OpenAIProvider", "OllamaProvider", "NvidiaProvider",
    "StubProvider", "LLMError", "get_provider", "set_provider", "build_provider",
    "Settings", "settings",
    "export_json", "export_html", "export_csv",
    "__version__",
]
