"""Shared fixtures.

Every test runs against the deterministic stub LLM and an isolated on-disk
SQLite file, so nothing touches the network or the developer's real state.
"""

from __future__ import annotations

import pytest

from orchestrator.agents import AgentManager
from orchestrator.graph import DependencyGraph
from orchestrator.llm import StubProvider, set_provider
from orchestrator.memory import MemoryManager
from orchestrator.models import Step
from orchestrator.state import StateManager
from orchestrator.tools import ToolManager, default_tool_manager
from orchestrator.tools.base import Tool, ToolError
from orchestrator.workflow import Workflow


@pytest.fixture(autouse=True)
def stub_llm():
    """Force the offline provider for every test, and reset it afterwards."""
    provider = StubProvider()
    set_provider(provider)
    yield provider
    set_provider(None)


@pytest.fixture(autouse=True)
def isolated_role_registry():
    """ROLE_PROMPTS is module-global; keep custom roles from leaking between tests."""
    from orchestrator.agents import ROLE_PROMPTS

    snapshot = dict(ROLE_PROMPTS)
    yield
    ROLE_PROMPTS.clear()
    ROLE_PROMPTS.update(snapshot)


@pytest.fixture
def tools() -> ToolManager:
    return default_tool_manager()


@pytest.fixture
def state(tmp_path) -> StateManager:
    manager = StateManager(str(tmp_path / "test_state.db"))
    yield manager
    manager.close()


def linear_steps() -> list:
    """frontend -> backend -> database -> testing."""
    return [
        Step(id="frontend", description="Build the UI", agent_role="frontend"),
        Step(id="backend", description="Build the API", agent_role="backend",
             requires_tool="github", depends_on=["frontend"]),
        Step(id="database", description="Design the schema", agent_role="database",
             requires_tool="postgres", depends_on=["backend"]),
        Step(id="testing", description="Run integration tests", agent_role="testing",
             requires_tool="ci", depends_on=["database"]),
    ]


def diamond_steps() -> list:
    """a -> (b, c) -> d, so b and c are concurrent."""
    return [
        Step(id="a", description="Root task", agent_role="research"),
        Step(id="b", description="Left branch", agent_role="backend", depends_on=["a"]),
        Step(id="c", description="Right branch", agent_role="frontend", depends_on=["a"]),
        Step(id="d", description="Join", agent_role="testing", depends_on=["b", "c"]),
    ]


@pytest.fixture
def linear_graph() -> DependencyGraph:
    return DependencyGraph(linear_steps())


@pytest.fixture
def diamond_graph() -> DependencyGraph:
    return DependencyGraph(diamond_steps())


@pytest.fixture
def workflow(stub_llm) -> Workflow:
    """A ready-to-run, non-persisting workflow over the linear graph."""
    return Workflow(
        linear_steps(),
        description="test workflow",
        run_id="test-run",
        llm=stub_llm,
        agent_manager=AgentManager(llm=stub_llm),
        memory=MemoryManager(),
        persist=False,
        verbose=False,
    )


@pytest.fixture
def make_workflow(stub_llm):
    """Factory for workflows with custom steps/options."""

    def factory(steps=None, **kwargs) -> Workflow:
        kwargs.setdefault("persist", False)
        kwargs.setdefault("verbose", False)
        kwargs.setdefault("llm", stub_llm)
        kwargs.setdefault("agent_manager", AgentManager(llm=stub_llm))
        kwargs.setdefault("memory", MemoryManager())
        return Workflow(steps if steps is not None else linear_steps(), **kwargs)

    return factory


class AlwaysFailsTool(Tool):
    """A tool with no working fallback, for failure-path tests."""

    name = "always_fails"
    capability = "explosive"
    description = "raises on every call"
    fallbacks: list = []

    def is_live(self) -> bool:
        return True

    def _run(self, task, context=None):
        raise ToolError("this tool always fails")


class CountingTool(Tool):
    """Records how many times it ran, and can be told to fail N times first."""

    capability = "counted"
    fallbacks: list = []

    def __init__(self, name: str = "counting", fail_times: int = 0):
        super().__init__()
        self.name = name
        self.fail_times = fail_times
        self.runs = 0

    def is_live(self) -> bool:
        return True

    def _run(self, task, context=None):
        self.runs += 1
        if self.runs <= self.fail_times:
            raise ToolError(f"transient failure {self.runs}")
        return f"[{self.name}] ok (run {self.runs})"


@pytest.fixture
def failing_tool() -> AlwaysFailsTool:
    return AlwaysFailsTool()
