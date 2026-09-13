"""Systems under test.

Four adapters, all executing identical agent work through the same stub LLM:

``adaptive``
    This project, with hash-based change propagation.

``adaptive_naive``
    This project with ``smart_invalidation=False``: the whole downstream cone
    of a changed step is re-run. Isolates how much signature comparison itself
    contributes, separately from the dependency analysis.

``restart_all``
    The "just re-run the workflow" baseline that any script without impact
    analysis falls back to. This is the honest representation of what a
    sequential agent pipeline does when a requirement changes.

``langgraph``
    Real LangGraph, with the true DAG edges (so it parallelises too) and a
    SQLite checkpointer. On a change it is re-invoked on a fresh thread, which
    re-executes every node -- LangGraph checkpoints let you *resume an
    interrupted run*, but the framework has no notion of "this node's inputs
    are unchanged, skip it". You can hand-roll that gating on top of it; the
    point of the comparison is that you have to.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict
import asyncio
import os

from orchestrator.agents import AgentManager
from orchestrator.graph import DependencyGraph
from orchestrator.memory import MemoryManager
from orchestrator.tools import default_tool_manager
from orchestrator.workflow import Workflow

from .harness import Scenario, SystemAdapter, Trial, TrialResult


def _merge_outputs(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    return {**a, **b}


class LangGraphState(TypedDict):
    """LangGraph channel schema.

    Declared at module level on purpose: LangGraph resolves these annotations
    with ``get_type_hints``, which only sees module globals -- a TypedDict
    defined inside a function fails under ``from __future__ import annotations``.
    """

    outputs: Annotated[Dict[str, str], _merge_outputs]


def _apply_event(scenario: Scenario, graph: DependencyGraph) -> None:
    """Mutate the graph the way the scenario's event would."""
    if scenario.event == "change" and scenario.target and scenario.new_description:
        graph.get(scenario.target).description = scenario.new_description


# ---------------------------------------------------------------------------
# This project
# ---------------------------------------------------------------------------


class AdaptiveAdapter(SystemAdapter):
    name = "adaptive (this project)"
    implementation_note = "hash-based invalidation; parallel level execution"
    smart = True

    def _workflow(self, scenario: Scenario) -> Workflow:
        return Workflow(
            graph=scenario.graph(),
            description=scenario.description,
            llm=self.provider,
            agent_manager=AgentManager(llm=self.provider),
            tool_manager=default_tool_manager(),
            memory=MemoryManager(),
            smart_invalidation=self.smart,
            parallel=True,
            persist=False,
            verbose=False,
        )

    def run_scenario(self, scenario: Scenario) -> TrialResult:
        workflow = self._workflow(scenario)
        result = TrialResult(system=self.name, scenario=scenario.name,
                             notes=self.implementation_note)

        result.initial = self._measure(lambda: len(workflow.run_full().executed))

        if scenario.event in {"change", "rerun"} and scenario.target:
            result.after_event = self._measure(
                lambda: len(workflow.handle_step_change(
                    scenario.target, new_description=scenario.new_description).executed)
            )
        elif scenario.event == "tool_failure" and scenario.target:
            def fail() -> int:
                report = workflow.handle_tool_failure(scenario.target)
                return len(report.executed) if report else 0

            result.after_event = self._measure(fail)

        return result


class AdaptiveNaiveAdapter(AdaptiveAdapter):
    name = "adaptive (downstream-cone invalidation)"
    implementation_note = "dependency analysis only, no output-signature comparison"
    smart = False


# ---------------------------------------------------------------------------
# Baseline: restart the whole workflow
# ---------------------------------------------------------------------------


class RestartAllAdapter(SystemAdapter):
    name = "restart-all baseline"
    implementation_note = "no impact analysis: any change re-runs every step, sequentially"

    def run_scenario(self, scenario: Scenario) -> TrialResult:
        graph = scenario.graph()
        agents = AgentManager(llm=self.provider)
        tools = default_tool_manager()
        result = TrialResult(system=self.name, scenario=scenario.name,
                             notes=self.implementation_note)

        def run_all() -> int:
            memory = MemoryManager()
            count = 0
            for step_id in graph.topological_order():
                step = graph.get(step_id)
                outcome = agents.get(step.agent_role).execute(
                    step, memory.build_context(step, graph), tools)
                memory.store(step_id, outcome.output)
                count += 1
            return count

        result.initial = self._measure(run_all)

        if scenario.event:
            if scenario.event == "tool_failure" and scenario.target:
                tools.break_tool(scenario.target)
            else:
                _apply_event(scenario, graph)
            result.after_event = self._measure(run_all)

        return result


# ---------------------------------------------------------------------------
# Baseline: LangGraph
# ---------------------------------------------------------------------------


class LangGraphAdapter(SystemAdapter):
    name = "langgraph"
    implementation_note = "real LangGraph StateGraph + SqliteSaver; re-invoked on change"

    def __init__(self, latency_s: float = 0.05):
        super().__init__(latency_s)
        self.available = True
        try:
            import langgraph  # noqa: F401
        except ImportError:
            self.available = False

    def _build(self, graph: DependencyGraph, agents: AgentManager, tools: Any,
               counter: Dict[str, int]) -> Any:
        from langgraph.graph import END, START, StateGraph

        def make_node(step_id: str):
            def node(state: LangGraphState) -> Dict[str, Any]:
                step = graph.get(step_id)
                outputs = state.get("outputs", {})
                # Fan-in guard. LangGraph triggers a node once per inbound
                # edge that fires, even with defer=True, when its parents
                # finish in different supersteps. Without this the node would
                # run on incomplete context -- so a real LangGraph user writes
                # the same guard, and it keeps the step count honest.
                if any(dep not in outputs for dep in step.depends_on):
                    return {}
                memory = MemoryManager()
                memory.load(outputs)
                outcome = agents.get(step.agent_role).execute(
                    step, memory.build_context(step, graph), tools)
                counter["executions"] += 1
                return {"outputs": {step_id: outcome.output}}

            return node

        builder = StateGraph(LangGraphState)
        for step_id in graph.topological_order():
            # A node with several inbound edges is triggered once per edge
            # unless it is deferred, which would inflate LangGraph's step
            # count for fan-in nodes. defer=True makes it a true join, which
            # is what a competent LangGraph user would write.
            builder.add_node(step_id, make_node(step_id),
                             defer=len(graph.get(step_id).depends_on) > 1)

        # Real DAG edges, so LangGraph fans out in supersteps exactly where
        # the dependencies allow -- it gets the same parallelism we do.
        for step_id in graph.topological_order():
            deps = graph.get(step_id).depends_on
            if deps:
                for dep in deps:
                    builder.add_edge(dep, step_id)
            else:
                builder.add_edge(START, step_id)
        for leaf in graph.leaves():
            builder.add_edge(leaf, END)

        return builder

    def run_scenario(self, scenario: Scenario) -> TrialResult:
        result = TrialResult(system=self.name, scenario=scenario.name,
                             notes=self.implementation_note)
        if not self.available:
            result.notes = "langgraph not installed - skipped"
            return result

        from langgraph.checkpoint.sqlite import SqliteSaver

        graph = scenario.graph()
        agents = AgentManager(llm=self.provider)
        tools = default_tool_manager()
        counter = {"executions": 0}
        builder = self._build(graph, agents, tools, counter)

        with SqliteSaver.from_conn_string(":memory:") as saver:
            app = builder.compile(checkpointer=saver)

            def invoke(thread: str) -> int:
                counter["executions"] = 0
                app.invoke({"outputs": {}}, config={"configurable": {"thread_id": thread}})
                return counter["executions"]

            result.initial = self._measure(lambda: invoke("t0"))

            if scenario.event:
                if scenario.event == "tool_failure" and scenario.target:
                    tools.break_tool(scenario.target)
                else:
                    _apply_event(scenario, graph)
                # A changed requirement invalidates the checkpoint, so the
                # graph is re-invoked on a fresh thread: every node re-runs.
                result.after_event = self._measure(lambda: invoke("t1"))

        return result


class CrewAIAdapter(SystemAdapter):
    """Real CrewAI Agent/Task/Crew execution with the shared benchmark LLM."""
    name = "crewai"
    implementation_note = "real CrewAI 1.x sequential Crew; restarted after a change"

    def __init__(self, latency_s: float = 0.05):
        super().__init__(latency_s)
        try:
            os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
            os.environ.setdefault("OTEL_SDK_DISABLED", "true")
            # CrewAI creates a user-data directory during import. Keep benchmark
            # artifacts inside this project so CI and restricted workers work.
            import appdirs
            from pathlib import Path
            appdirs.user_data_dir = lambda *a, **k: str(Path("artifacts/.crewai").resolve())
            import crewai  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def _run_crew(self, graph: DependencyGraph) -> int:
        from crewai import Agent, BaseLLM, Crew, Process, Task
        provider = self.provider

        class BenchmarkLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None, available_functions=None,
                     from_task=None, from_agent=None, response_model=None):
                if isinstance(messages, list):
                    prompt = "\n".join(str(m.get("content", m)) if isinstance(m, dict)
                                       else str(m) for m in messages)
                else:
                    prompt = str(messages)
                return provider.generate(prompt, metadata={
                    "role": getattr(from_agent, "role", "")}).text

        llm = BenchmarkLLM(model="benchmark-stub")
        agents, tasks = {}, {}
        for sid in graph.topological_order():
            step = graph.get(sid)
            agent = agents.setdefault(step.agent_role, Agent(
                role=step.agent_role, goal=f"Complete {step.agent_role} work accurately",
                backstory="Benchmark specialist", llm=llm, verbose=False,
                allow_delegation=False))
            tasks[sid] = Task(description=step.description, expected_output="A complete result",
                              agent=agent, context=[tasks[d] for d in step.depends_on])
        Crew(agents=list(agents.values()), tasks=[tasks[s] for s in graph.topological_order()],
             process=Process.sequential, verbose=False).kickoff()
        return len(tasks)

    def run_scenario(self, scenario: Scenario) -> TrialResult:
        result = TrialResult(system=self.name, scenario=scenario.name,
                             notes=self.implementation_note)
        if not self.available:
            result.notes = "crewai not installed - skipped"
            return result
        graph = scenario.graph()
        result.initial = self._measure(lambda: self._run_crew(graph))
        if scenario.event:
            _apply_event(scenario, graph)
            result.after_event = self._measure(lambda: self._run_crew(graph))
        return result


class AutoGenAdapter(SystemAdapter):
    """Real AutoGen BaseChatAgent execution with the shared benchmark LLM."""
    name = "autogen"
    implementation_note = "real Microsoft AutoGen AgentChat 0.7 agents; restarted after change"

    def __init__(self, latency_s: float = 0.05):
        super().__init__(latency_s)
        try:
            import autogen_agentchat  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def _run_agents(self, graph: DependencyGraph) -> int:
        from autogen_agentchat.agents import BaseChatAgent
        from autogen_agentchat.base import Response
        from autogen_agentchat.messages import TextMessage
        provider = self.provider

        class BenchmarkAgent(BaseChatAgent):
            def __init__(self, name, role):
                super().__init__(name, role)
                self.role = role
            @property
            def produced_message_types(self): return (TextMessage,)
            async def on_messages(self, messages, cancellation_token):
                prompt = "\n".join(str(m.content) for m in messages)
                text = provider.generate(prompt, metadata={"role": self.role}).text
                return Response(chat_message=TextMessage(content=text, source=self.name))
            async def on_reset(self, cancellation_token): return None

        async def execute():
            outputs = {}
            for sid in graph.topological_order():
                step = graph.get(sid)
                context = "\n".join(outputs[d] for d in step.depends_on)
                agent = BenchmarkAgent(f"agent_{sid.replace('-', '_')}", step.agent_role)
                result = await agent.run(task=f"{step.description}\nContext:\n{context}")
                outputs[sid] = str(result.messages[-1].content)
            return len(outputs)
        return asyncio.run(execute())

    def run_scenario(self, scenario: Scenario) -> TrialResult:
        result = TrialResult(system=self.name, scenario=scenario.name,
                             notes=self.implementation_note)
        if not self.available:
            result.notes = "autogen-agentchat not installed - skipped"
            return result
        graph = scenario.graph()
        result.initial = self._measure(lambda: self._run_agents(graph))
        if scenario.event:
            _apply_event(scenario, graph)
            result.after_event = self._measure(lambda: self._run_agents(graph))
        return result


def default_adapters(latency_s: float = 0.05) -> List[Any]:
    """Factories (not instances) so each repeat gets clean counters."""
    return [
        lambda: AdaptiveAdapter(latency_s),
        lambda: AdaptiveNaiveAdapter(latency_s),
        lambda: RestartAllAdapter(latency_s),
        lambda: LangGraphAdapter(latency_s),
        lambda: CrewAIAdapter(latency_s),
        lambda: AutoGenAdapter(latency_s),
    ]
