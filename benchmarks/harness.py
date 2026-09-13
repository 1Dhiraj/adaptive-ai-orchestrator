"""Benchmark harness.

Every system under test runs the *same* graph, with the *same* deterministic
stub LLM (fixed per-call latency), so the numbers reflect orchestration
strategy and nothing else. No API keys are involved, and repeated runs give
identical step/token counts.

What is measured per trial:

* ``executions``  - how many agent steps actually ran
* ``llm_calls``   - LLM invocations
* ``tokens``      - prompt + completion tokens
* ``wall_s``      - wall-clock seconds (so parallelism shows up)
* ``cost_usd``    - priced against the configured model's rate card
"""

from __future__ import annotations

import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from orchestrator.graph import DependencyGraph
from orchestrator.llm import StubProvider
from orchestrator.models import Step

#: Simulated latency of one LLM call, in seconds. Real Gemini calls take
#: 1-4s; 0.05 keeps the suite fast while preserving the relative shape.
DEFAULT_LATENCY_S = 0.05


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A graph plus the event that happens partway through."""

    name: str
    description: str
    steps: List[Step]
    #: What happens after the initial full run. ``None`` = nothing (full run only).
    event: Optional[str] = None          # "change" | "rerun" | "tool_failure" | None
    target: Optional[str] = None         # step id, or tool name for tool_failure
    new_description: Optional[str] = None

    def graph(self) -> DependencyGraph:
        # Fresh Step objects per trial: adapters mutate descriptions.
        return DependencyGraph(Step.from_dict(s.to_dict()) for s in self.steps)


def _web_app_steps() -> List[Step]:
    return [
        Step(id="requirements", description="Clarify scope and acceptance criteria for the app.",
             agent_role="research"),
        Step(id="database", description="Design the PostgreSQL schema for users, tasks and boards.",
             agent_role="database", requires_tool="postgres", depends_on=["requirements"]),
        Step(id="frontend", description="Build the React UI: login form and Kanban board.",
             agent_role="frontend", depends_on=["requirements"]),
        Step(id="backend", description="Implement the REST API for auth, tasks and boards.",
             agent_role="backend", requires_tool="github", depends_on=["database"]),
        Step(id="testing", description="Write and run integration tests across the stack.",
             agent_role="testing", requires_tool="ci", depends_on=["backend", "frontend"]),
    ]


def _microservices_steps() -> List[Step]:
    return [
        Step(id="architecture", description="Define service boundaries and the event contract.",
             agent_role="research"),
        Step(id="auth_svc", description="Build the authentication service.",
             agent_role="backend", requires_tool="github", depends_on=["architecture"]),
        Step(id="catalog_svc", description="Build the product catalog service.",
             agent_role="backend", requires_tool="github", depends_on=["architecture"]),
        Step(id="order_svc", description="Build the order service.",
             agent_role="backend", requires_tool="github", depends_on=["architecture"]),
        Step(id="schema", description="Design the shared database schemas for all services.",
             agent_role="database", requires_tool="postgres", depends_on=["architecture"]),
        Step(id="gateway", description="Build the API gateway routing to all services.",
             agent_role="backend", depends_on=["auth_svc", "catalog_svc", "order_svc"]),
        Step(id="deploy", description="Containerise everything and write the CI pipeline.",
             agent_role="devops", depends_on=["gateway", "schema"]),
        Step(id="testing", description="Write and run end-to-end tests across all services.",
             agent_role="testing", requires_tool="ci", depends_on=["deploy"]),
    ]


def default_scenarios() -> List[Scenario]:
    return [
        Scenario(
            name="full_run",
            description="Cold start: plan already known, execute all 5 steps once.",
            steps=_web_app_steps(),
        ),
        Scenario(
            name="early_change",
            description="Requirement changes at the root, invalidating almost everything.",
            steps=_web_app_steps(), event="change", target="requirements",
            new_description="Clarify scope for a mobile-first app with offline sync support.",
        ),
        Scenario(
            name="mid_change",
            description="Client switches database mid-project (the classic case).",
            steps=_web_app_steps(), event="change", target="database",
            new_description="Design MongoDB collections for users, tasks and boards instead.",
        ),
        Scenario(
            name="leaf_change",
            description="A change at a leaf: nothing downstream depends on it.",
            steps=_web_app_steps(), event="change", target="testing",
            new_description="Write property-based tests with Hypothesis instead of example tests.",
        ),
        Scenario(
            name="noop_rerun",
            description="A mid-graph step is re-run but produces an identical result "
                        "(e.g. a retried flaky step, or an edit that changed nothing).",
            steps=_web_app_steps(), event="rerun", target="database",
        ),
        Scenario(
            name="tool_failure",
            description="The GitHub API goes down mid-workflow.",
            steps=_web_app_steps(), event="tool_failure", target="github",
        ),
        Scenario(
            name="microservices_change",
            description="8-step microservices build, one service's requirements change.",
            steps=_microservices_steps(), event="change", target="catalog_svc",
            new_description="Build the catalog service in Go with a GraphQL interface.",
        ),
    ]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """Numbers from one execution phase (initial run, or post-event run)."""

    executions: int = 0
    llm_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    wall_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executions": self.executions, "llm_calls": self.llm_calls,
            "tokens": self.tokens, "cost_usd": round(self.cost_usd, 8),
            "wall_s": round(self.wall_s, 4),
        }


@dataclass
class TrialResult:
    system: str
    scenario: str
    initial: Trial = field(default_factory=Trial)
    after_event: Optional[Trial] = None
    notes: str = ""

    @property
    def total(self) -> Trial:
        second = self.after_event or Trial()
        return Trial(
            executions=self.initial.executions + second.executions,
            llm_calls=self.initial.llm_calls + second.llm_calls,
            tokens=self.initial.tokens + second.tokens,
            cost_usd=self.initial.cost_usd + second.cost_usd,
            wall_s=self.initial.wall_s + second.wall_s,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system, "scenario": self.scenario,
            "initial": self.initial.to_dict(),
            "after_event": self.after_event.to_dict() if self.after_event else None,
            "total": self.total.to_dict(),
            "notes": self.notes,
        }


class SystemAdapter(ABC):
    """Wraps one orchestration framework so the harness can drive it."""

    name: str = "abstract"
    #: Shown in the report so nobody mistakes a baseline for the real thing.
    implementation_note: str = ""

    def __init__(self, latency_s: float = DEFAULT_LATENCY_S):
        self.latency_s = latency_s
        self.provider = StubProvider(latency_s=latency_s)

    def _measure(self, fn: Callable[[], int]) -> Trial:
        """Run ``fn`` (returns steps executed) and capture usage deltas."""
        before = self.provider.total_usage
        started = time.perf_counter()
        executions = fn()
        wall = time.perf_counter() - started
        after = self.provider.total_usage
        return Trial(
            executions=executions,
            llm_calls=after.calls - before.calls,
            tokens=after.total_tokens - before.total_tokens,
            cost_usd=after.cost_usd - before.cost_usd,
            wall_s=wall,
        )

    @abstractmethod
    def run_scenario(self, scenario: Scenario) -> TrialResult:
        ...


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------


def _median_trial(trials: List[Trial]) -> Trial:
    return Trial(
        executions=int(statistics.median(t.executions for t in trials)),
        llm_calls=int(statistics.median(t.llm_calls for t in trials)),
        tokens=int(statistics.median(t.tokens for t in trials)),
        cost_usd=statistics.median(t.cost_usd for t in trials),
        wall_s=statistics.median(t.wall_s for t in trials),
    )


def run_matrix(
    adapters: List[Callable[[], SystemAdapter]],
    scenarios: List[Scenario],
    repeats: int = 3,
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[TrialResult]:
    """Run every (system, scenario) pair ``repeats`` times, keeping medians.

    Deterministic counters (executions, calls, tokens) are identical across
    repeats; the median matters for ``wall_s``, which is noisy.
    """
    results: List[TrialResult] = []
    for scenario in scenarios:
        for make_adapter in adapters:
            repeated: List[TrialResult] = []
            for _ in range(max(1, repeats)):
                adapter = make_adapter()
                if on_progress:
                    on_progress(f"{adapter.name} / {scenario.name}")
                repeated.append(adapter.run_scenario(scenario))

            merged = TrialResult(
                system=repeated[0].system,
                scenario=scenario.name,
                initial=_median_trial([r.initial for r in repeated]),
                after_event=(
                    _median_trial([r.after_event for r in repeated if r.after_event])
                    if repeated[0].after_event else None
                ),
                notes=repeated[0].notes,
            )
            results.append(merged)
    return results
