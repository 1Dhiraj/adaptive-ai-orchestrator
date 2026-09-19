"""Offline checks for fact invalidation, confirmation, cutoff and recovery."""
import json

import pytest

from orchestrator.change_aware import (
    Fact, FactStore, check_plan, detect_assumptions, equivalent,
    parse_assumptions, translate_change,
)
from orchestrator.graph import DependencyGraph
from orchestrator.llm import LLMProvider, LLMResponse, StubProvider
from orchestrator.models import LLMUsage, Step, StepResult
from orchestrator.planner import TaskPlanner
from orchestrator.workflow import Workflow


class Worker(LLMProvider):
    name = "test"

    def __init__(self, outputs=None):
        super().__init__()
        self.seen = []
        self.outputs = outputs or {}

    def _generate(self, prompt, system, json_mode, metadata):
        sid = (metadata or {}).get("step_id", "")
        self.seen.append(sid)
        value = "PostgreSQL" if '"database": "PostgreSQL"' in prompt else "MongoDB"
        defaults = {"db": "schema for " + value,
                    "hidden": "client using " + value,
                    "downstream": "tests for " + value,
                    "independent": "A blue button"}
        text = self.outputs.get(sid, defaults.get(sid, "constant result"))
        if isinstance(text, list):
            text = text.pop(0)
        return LLMResponse(text, LLMUsage(calls=1, prompt_tokens=10, completion_tokens=5), "test", 0)


def graph():
    return DependencyGraph([
        Step("db", "Define storage", "database"),
        Step("hidden", "Write client", "backend"),  # deliberately no db edge
        Step("downstream", "Test client", "testing", depends_on=["hidden"]),
        Step("independent", "Draw button", "frontend"),
    ], facts=[Fact("database", "MongoDB", "db", ["Mongo"])])


def workflow(**kwargs):
    kwargs.setdefault("llm", Worker())
    kwargs.setdefault("persist", False)
    return Workflow(graph=graph(), verbose=False, **kwargs)


@pytest.mark.parametrize("parallel", [False, True])
def test_hidden_dependency_reruns_without_graph_edge(parallel):
    wf = workflow(parallel=parallel)
    wf.run_full()
    assert wf.results["hidden"].detected_assumptions == ["database"]
    proposal = wf.preview_fact_change({"database": "PostgreSQL"})
    assert proposal["affected"] == ["db", "downstream", "hidden"]
    assert wf.graph.facts["database"].value == "MongoDB"
    report = wf.apply_change(proposal, confirmed=True)
    assert set(report.executed) == {"db", "hidden", "downstream"}
    assert report.reused == ["independent"]
    assert "PostgreSQL" in wf.results["hidden"].output
    assert wf.graph.facts["database"].aliases == []
    assert wf.resume().executed == []


def test_owner_is_forced_even_without_declared_or_detected_assumptions():
    wf = workflow(llm=Worker({"db": "constant"}))
    wf.run_full()
    p = wf.preview_fact_change({"database": "PostgreSQL"})
    report = wf.apply_change(p, confirmed=True)
    assert "db" in report.executed
    assert wf.results["db"].change_status == "cut_off"
    assert "owner; P4" in wf.results["db"].invalidation_reasons[0]


def test_declared_assumptions_without_lexical_mention():
    wf = workflow(llm=Worker({"hidden": "client\nASSUMPTIONS: database\nHANDOFF: retained"}))
    wf.run_full()
    assert wf.results["hidden"].declared_assumptions == ["database"]
    assert "ASSUMPTIONS:" not in wf.results["hidden"].output
    assert "HANDOFF: retained" in wf.results["hidden"].output
    assert "hidden" in wf.preview_fact_change({"database": "PostgreSQL"})["affected"]


def test_whole_word_alias_detection():
    facts = FactStore([Fact("db", "MongoDB", "a", ["Mongo"])])
    assert detect_assumptions("MONGODB, or Mongo.", facts) == {"db"}
    assert detect_assumptions("MongoDBClient mongo_database", facts) == set()
    assert parse_assumptions("result\nASSUMPTIONS: db, invented", facts) == ("result", {"db"})


def test_confirmation_tamper_and_stale_proposals():
    wf = workflow()
    wf.run_full()
    p = wf.preview_fact_change({"database": "PostgreSQL"})
    with pytest.raises(ValueError, match="confirm"):
        wf.apply_change(p)
    altered = dict(p, delta={"database": "SQLite"})
    with pytest.raises(ValueError, match="modified"):
        wf.apply_change(altered, confirmed=True)
    wf.graph.get("db").description = "new description"
    with pytest.raises(ValueError, match="stale"):
        wf.apply_change(p, confirmed=True)
    assert wf.graph.facts["database"].value == "MongoDB"


@pytest.mark.parametrize("delta", [{"unknown": "x"}, {"database": ""}, {"database": None}])
def test_invalid_delta_is_atomic(delta):
    wf = workflow()
    with pytest.raises(ValueError):
        wf.preview_fact_change(delta)
    assert wf.graph.facts["database"].value == "MongoDB"


def test_legacy_setting_keeps_structural_reuse():
    wf = workflow(change_aware=False)
    wf.run_full()
    wf.graph.facts = wf.graph.facts.changed({"database": "PostgreSQL"})
    assert wf.resume().executed == []
    with pytest.raises(ValueError, match="CHANGE_AWARE"):
        wf.preview_fact_change({"database": "SQLite"})


@pytest.mark.parametrize("enabled,expected", [(False, {"a", "b"}), (True, {"a"})])
def test_semantic_cutoff_is_opt_in(enabled, expected):
    g = DependencyGraph([
        Step("a", "config", "writer", output_type="json"),
        Step("b", "consume", "writer", depends_on=["a"]),
    ], facts=[Fact("style", "old", "a")])
    llm = Worker({"a": ['{"x": 1, "y": 2}', '{"y": 2, "x": 1}']})
    wf = Workflow(graph=g, llm=llm, persist=False, verbose=False, semantic_cutoff=enabled)
    wf.run_full()
    p = wf.preview_fact_change({"style": "new"})
    assert set(wf.apply_change(p, confirmed=True).executed) == expected
    if enabled:
        assert wf.results["a"].output == '{"x": 1, "y": 2}'
        assert wf.results["a"].raw_output == '{"y": 2, "x": 1}'
        assert wf.results["a"].change_status == "cut_off"


@pytest.mark.parametrize("kind,old,new,same", [
    ("code", "x=1", "x = 1 # comment", True),
    ("code", 'x="a b"', 'x="a  b"', False),
    ("code", 'const x="a b";', 'const x="a  b";', False),
    ("json", '{"x":1}', '{"x":true}', False),
    ("json", '{"x":1}', '{"x":2}', False),
    ("json", 'bad', 'different bad', False),
])
def test_typed_equivalence_preserves_meaning(kind, old, new, same):
    assert equivalent(old, new, kind)[0] is same


def test_text_judge_threshold_and_usage():
    class Judge(Worker):
        def _generate(self, *args, **kwargs):
            self.seen.append("judge")
            return LLMResponse("EQUIVALENT", LLMUsage(calls=1, prompt_tokens=20), "judge", 0)
    judge = Judge()
    assert not equivalent("old one", "new two", "text", judge=judge)[0]
    assert judge.seen == []
    same, usage, _ = equivalent("red blue", "blue red", "text", judge=judge)
    assert same and usage.prompt_tokens == 20 and judge.seen == ["judge"]


def test_typed_checker_and_budget_block_before_execution():
    wf = workflow()
    wf.graph.get("hidden").accepts = ["json"]
    wf.graph.get("hidden").depends_on = ["db"]
    wf.graph.get("db").assumes = ["nonexistent"]
    errors = wf.check_requirements().plan_errors
    assert any("cannot accept text" in e for e in errors)
    assert any("unknown assumed fact" in e for e in errors)
    assert not wf.requirements.can_run
    with pytest.raises(ValueError, match="invalid plan"):
        wf.run_full()
    assert wf.llm.seen == []
    assert check_plan(graph(), 1)["errors"]
    g = DependencyGraph([Step("a", "x", "writer")], facts=[Fact("db", "x", "missing")])
    assert "owner missing" in check_plan(g)["errors"][0]


def test_persistence_restores_facts_assumptions_and_pending_change(state):
    wf = workflow(state=state, persist=True)
    wf.run_full()
    p = wf.preview_fact_change({"database": "PostgreSQL"})
    wf.apply_change(p, confirmed=True, rerun=False)
    restored = Workflow.resume_from(wf.run_id, state=state, llm=Worker(), verbose=False)
    assert restored.graph.facts["database"].value == "PostgreSQL"
    assert restored.results["hidden"].assumed_facts == {"database": "MongoDB"}
    assert set(restored.resume().executed) == {"db", "hidden", "downstream"}
    again = Workflow.resume_from(wf.run_id, state=state, llm=Worker(), verbose=False)
    assert again.resume().executed == []


def test_planner_extracts_fact_and_type_fields():
    class Planner(Worker):
        def _generate(self, *args, **kwargs):
            return LLMResponse(json.dumps({"tasks": [
                {"id": "db", "role": "database", "description": "storage", "output_type": "json",
                 "assumes": ["database"], "accepts": ["text"]}],
                "facts": [{"key": "database", "value": "MongoDB", "aliases": [], "owner": "db"}]}),
                LLMUsage(calls=1), "test", 0)
    plan = TaskPlanner(llm=Planner()).plan("Use MongoDB")
    assert plan.graph.facts["database"].value == "MongoDB"
    assert plan.graph.get("db").output_type == "json"
    assert plan.requirements.can_run
    restored = DependencyGraph.from_dict(plan.graph.to_dict())
    assert restored.to_dict() == plan.graph.to_dict()


def test_offline_translation_and_replan_are_proposals_only():
    wf = workflow(change_llm=StubProvider())
    p = wf.prepare_change("use PostgreSQL instead of MongoDB")
    assert p["delta"] == {"database": "PostgreSQL"}
    assert wf.graph.facts["database"].value == "MongoDB"
    p = wf.prepare_change("Add a legal review")
    assert p["kind"] == "replan" and "diff" in p
    assert wf.llm.seen == []
    wf.apply_change(p, confirmed=True, rerun=False)
    assert wf.graph.to_dict()["steps"] == p["graph"]["steps"]


def test_invalid_translator_response_does_not_modify_facts():
    facts = graph().facts
    with pytest.raises(ValueError, match="invalid JSON"):
        translate_change("change things", facts, Worker())
    assert facts["database"].value == "MongoDB"


def test_change_revokes_old_action_approval():
    from orchestrator.tools import ToolManager
    from tests.conftest import CountingTool
    tool = CountingTool()
    tool.irreversible = True
    wf = workflow(tool_manager=ToolManager([tool]))
    wf.graph.get("db").requires_tool = "counting"
    assert wf.run_full().awaiting_action == ["db"]
    wf.approve_action("db")
    p = wf.preview_fact_change({"database": "PostgreSQL"})
    report = wf.apply_change(p, confirmed=True)
    assert tool.runs == 0
    assert report.awaiting_action == ["db"]
    assert "PostgreSQL" in wf.pending_actions()["db"].payload
    wf.approve_action("db")
    wf.resume()
    assert tool.runs == 1


def test_result_serialization_is_backward_compatible():
    result = StepResult.from_dict({"step_id": "a", "status": "done", "output": "old"})
    assert result.raw_output == "old" and result.assumed_facts == {}
    assert StepResult.from_dict(result.to_dict()).to_dict() == result.to_dict()


def test_owner_policy_survives_disabled_structural_invalidation():
    wf = workflow(smart_invalidation=False)
    wf.run_full()
    p = wf.preview_fact_change({"database": "PostgreSQL"})
    assert "db" in wf.apply_change(p, confirmed=True).executed


def test_noop_change_executes_nothing():
    wf = workflow()
    wf.run_full()
    before = len(wf.llm.seen)
    p = wf.preview_fact_change({"database": "MongoDB"})
    assert p["affected"] == []
    assert wf.apply_change(p, confirmed=True) is None
    assert len(wf.llm.seen) == before


def test_nvidia_role_overrides_do_not_make_calls(monkeypatch):
    from orchestrator.llm import NvidiaProvider, provider_for_role
    monkeypatch.setenv("NVIDIA_MODEL_PLANNER", "test/planner")
    monkeypatch.setenv("NVIDIA_MODEL_EQUIVALENCE_JUDGE", "test/judge")
    base = NvidiaProvider(api_key="test-only", model="test/main")
    planner = provider_for_role(base, "planner")
    assert planner.model == "test/planner"
    assert planner is provider_for_role(base, "planner")
    assert provider_for_role(base, "equivalence_judge").model == "test/judge"
    assert base.model == "test/main" and base.total_usage.calls == 0
    stub = StubProvider()
    assert provider_for_role(stub, "planner") is stub


def test_judge_failure_is_not_a_cutoff():
    class BrokenJudge(Worker):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("offline")
    assert not equivalent("red blue", "blue red", "text", judge=BrokenJudge())[0]


def test_unknown_translation_key_is_rejected():
    class Translator(Worker):
        def _generate(self, *args, **kwargs):
            return LLMResponse('{"delta":{"invented":"value"}}', LLMUsage(), "test", 0)
    with pytest.raises(ValueError, match="unknown fact"):
        workflow(change_llm=Translator()).prepare_change("a change")
