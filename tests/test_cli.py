"""CLI entry points, the demo walkthrough, and the benchmark harness."""

from __future__ import annotations

import json

import pytest

from orchestrator.cli import build_parser, main


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Keep CLI runs off the developer's real state file."""
    import orchestrator.config as config

    monkeypatch.setattr(config.settings, "db_path", str(tmp_path / "cli.db"))
    monkeypatch.chdir(tmp_path)


class TestParser:
    def test_every_subcommand_is_registered(self):
        parser = build_parser()
        actions = [a for a in parser._actions if a.dest == "command"]
        assert set(actions[0].choices) == {"plan", "run", "resume", "runs", "demo",
                                           "serve", "worker", "bench", "mcp", "connections", "skills"}

    def test_missing_subcommand_exits(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestPlan:
    def test_plan_reports_requirements_and_steps(self, capsys):
        assert main(["--stub", "plan", "Build a task manager with auth"]) == 0
        out = capsys.readouterr().out
        assert "WHAT THIS TASK NEEDS" in out
        assert "Specialists to be created:" in out
        assert "Capabilities required:" in out
        assert "Planned steps:" in out
        assert "Parallel groups:" in out

    def test_plan_can_emit_mermaid(self, capsys):
        assert main(["--stub", "plan", "Build a task manager", "--mermaid"]) == 0
        assert "flowchart TD" in capsys.readouterr().out

    def test_plan_writes_json_with_requirements(self, tmp_path, capsys):
        out_path = tmp_path / "plan.json"
        assert main(["--stub", "plan", "Build a task manager",
                     "--json-out", str(out_path)]) == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["graph"]["steps"]
        assert "requirements" in payload and "can_run" in payload["requirements"]

    def test_plan_exits_nonzero_when_blocked(self, capsys, monkeypatch):
        """A plan that cannot run must exit non-zero, so scripts can gate on it."""
        from orchestrator import planner as planner_module
        from orchestrator.requirements import Requirement, RequirementKind

        original = planner_module.TaskPlanner.plan

        def with_blocker(self, description, extra_guidance="", tools=None):
            result = original(self, description, extra_guidance, tools)
            result.requirements.requirements.append(Requirement(
                name="DEFINITELY_NOT_SET_XYZ",
                kind=RequirementKind.CREDENTIAL,
                why="a credential this machine certainly does not have",
                setup="set DEFINITELY_NOT_SET_XYZ"))
            result.requirements.check(tools)
            return result

        monkeypatch.setattr(planner_module.TaskPlanner, "plan", with_blocker)
        assert main(["--stub", "plan", "Build a thing"]) == 1
        assert "BLOCKED" in capsys.readouterr().out


class TestRun:
    def test_run_executes_and_summarises(self, capsys):
        assert main(["--stub", "run", "Build a bookstore API", "--run-id", "cli-1"]) == 0
        out = capsys.readouterr().out
        assert "WORKFLOW cli-1" in out
        assert "tokens:" in out

    def test_run_with_a_change_reuses_work(self, capsys):
        assert main(["--stub", "run", "Build a bookstore API", "--run-id", "cli-2",
                     "--change", "testing", "--requirement", "use property-based tests"]) == 0
        assert "reused" in capsys.readouterr().out

    def test_run_with_a_broken_tool_falls_back(self, capsys):
        assert main(["--stub", "run", "Build a bookstore API and publish it to GitHub",
                     "--run-id", "cli-3",
                     "--break-tool", "github"]) == 0
        assert "FALLBACK" in capsys.readouterr().out

    def test_run_writes_requested_exports(self, tmp_path, capsys):
        assert main(["--stub", "run", "Build an app", "--run-id", "cli-4",
                     "--json", "out.json", "--csv", "out.csv", "--html", "out.html"]) == 0
        for name in ("out.json", "out.csv", "out.html"):
            assert (tmp_path / name).exists()

    def test_run_from_a_plan_file(self, tmp_path, capsys):
        plan = tmp_path / "plan.json"
        plan.write_text(json.dumps({"tasks": [
            {"id": "a", "role": "backend", "description": "Do a", "depends_on": []},
            {"id": "b", "role": "testing", "description": "Do b", "depends_on": ["a"]},
        ]}), encoding="utf-8")
        assert main(["--stub", "run", "--plan-file", str(plan), "--run-id", "cli-5"]) == 0
        assert "WORKFLOW cli-5" in capsys.readouterr().out

    def test_run_without_description_or_plan_is_an_error(self, capsys):
        assert main(["--stub", "run"]) == 2

    def test_sequential_flag_is_accepted(self, capsys):
        assert main(["--stub", "run", "Build an app", "--run-id", "cli-6", "--sequential"]) == 0


class TestResumeAndList:
    def test_runs_listing_is_empty_then_populated(self, capsys):
        assert main(["--stub", "runs"]) == 0
        assert "no persisted runs" in capsys.readouterr().out

        main(["--stub", "run", "Build an app", "--run-id", "keep-1"])
        capsys.readouterr()
        assert main(["--stub", "runs"]) == 0
        assert "keep-1" in capsys.readouterr().out

    def test_resume_reuses_everything_after_a_clean_run(self, capsys):
        main(["--stub", "run", "Build an app", "--run-id", "resume-1"])
        capsys.readouterr()
        assert main(["--stub", "resume", "resume-1"]) == 0
        out = capsys.readouterr().out
        assert "0 executed" in out

    def test_resuming_an_unknown_run_raises(self):
        with pytest.raises(KeyError):
            main(["--stub", "resume", "does-not-exist"])


class TestDemo:
    def test_demo_walks_all_four_scenarios(self, capsys):
        assert main(["--stub", "demo", "--run-id", "demo-test", "--html", "demo.html"]) == 0
        out = capsys.readouterr().out
        for scenario in ("SCENARIO 1", "SCENARIO 2", "SCENARIO 3", "SCENARIO 4"):
            assert scenario in out
        assert "EFFICIENCY" in out
        assert "fewer step executions" in out


class TestBenchmarks:
    def test_harness_runs_a_single_scenario(self, tmp_path):
        from benchmarks.run_benchmarks import main as bench_main

        assert bench_main(repeats=1, scenarios=["full_run"], latency_s=0.0,
                          output=str(tmp_path / "bench")) == 0
        payload = json.loads((tmp_path / "bench" / "results.json").read_text(encoding="utf-8"))
        assert payload["raw"]
        assert (tmp_path / "bench" / "report.md").exists()
        assert (tmp_path / "bench" / "report.html").exists()

    def test_unknown_scenario_is_an_error(self, tmp_path):
        from benchmarks.run_benchmarks import main as bench_main

        assert bench_main(scenarios=["nope"], output=str(tmp_path / "b")) == 2

    def test_adaptive_beats_restart_all_on_a_leaf_change(self):
        from benchmarks.baselines import AdaptiveAdapter, RestartAllAdapter
        from benchmarks.harness import default_scenarios

        scenario = next(s for s in default_scenarios() if s.name == "leaf_change")
        ours = AdaptiveAdapter(latency_s=0.0).run_scenario(scenario)
        theirs = RestartAllAdapter(latency_s=0.0).run_scenario(scenario)
        assert ours.total.executions < theirs.total.executions
        assert ours.total.tokens < theirs.total.tokens

    def test_fingerprinting_beats_cone_invalidation_on_a_noop(self):
        from benchmarks.baselines import AdaptiveAdapter, AdaptiveNaiveAdapter
        from benchmarks.harness import default_scenarios

        scenario = next(s for s in default_scenarios() if s.name == "noop_rerun")
        smart = AdaptiveAdapter(latency_s=0.0).run_scenario(scenario)
        naive = AdaptiveNaiveAdapter(latency_s=0.0).run_scenario(scenario)
        assert smart.total.executions < naive.total.executions

    def test_counts_are_reproducible_across_repeats(self):
        from benchmarks.baselines import AdaptiveAdapter
        from benchmarks.harness import default_scenarios

        scenario = next(s for s in default_scenarios() if s.name == "mid_change")
        first = AdaptiveAdapter(latency_s=0.0).run_scenario(scenario).total
        second = AdaptiveAdapter(latency_s=0.0).run_scenario(scenario).total
        assert (first.executions, first.tokens) == (second.executions, second.tokens)

    def test_langgraph_baseline_executes_each_node_once_per_invocation(self):
        from benchmarks.baselines import LangGraphAdapter
        from benchmarks.harness import default_scenarios

        adapter = LangGraphAdapter(latency_s=0.0)
        if not adapter.available:
            pytest.skip("langgraph is not installed")
        scenario = next(s for s in default_scenarios() if s.name == "microservices_change")
        result = adapter.run_scenario(scenario)
        # 8 nodes, no fan-in double-firing.
        assert result.initial.executions == 8
