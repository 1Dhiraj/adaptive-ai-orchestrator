"""The agentic shell loop: that it loops, and that it stays inside its bounds.

The loop is what separates this from one-shot execution -- the agent reads
each result before choosing the next command -- so the tests that matter are
the ones proving it actually reacts to output, and the ones proving it stops
when it should.
"""

from __future__ import annotations

import json

import pytest

from orchestrator.llm import LLMResponse, LLMUsage
from orchestrator.tools.base import ToolError
from orchestrator.tools.computer_task import ComputerTaskTool


class ScriptedModel:
    """Replays a fixed list of decisions and records what it was shown."""

    name = "scripted"
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, system=None, json_mode=False, metadata=None, images=None):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if self.replies else {"done": True, "answer": "ran out"}
        return LLMResponse(text=json.dumps(reply), usage=LLMUsage(calls=1),
                           model="scripted", latency_s=0.0)


def goal(text="do the thing"):
    return "x\nTOOL_DIRECTIVE: " + json.dumps({"arguments": {"goal": text}})


@pytest.fixture
def allow_terminal(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_TERMINAL", "1")


class TestTheLoop:
    def test_runs_several_commands_then_reports(self, allow_terminal, tmp_path):
        model = ScriptedModel([
            {"command": "echo first"},
            {"command": "echo second"},
            {"done": True, "answer": "both ran"},
        ])
        tool = ComputerTaskTool(run_id="loop", root=tmp_path, llm=model)
        out = tool.execute(goal())
        assert "echo first" in out and "echo second" in out
        assert "RESULT: both ran" in out

    def test_the_model_sees_what_the_last_command_printed(self, allow_terminal, tmp_path):
        """The whole point of a loop: output feeds the next decision."""
        model = ScriptedModel([
            {"command": "echo needle-in-output"},
            {"done": True, "answer": "saw it"},
        ])
        tool = ComputerTaskTool(run_id="feedback", root=tmp_path, llm=model)
        tool.execute(goal())
        assert any("needle-in-output" in p for p in model.prompts[1:])

    def test_a_failing_command_does_not_end_the_run(self, allow_terminal, tmp_path):
        model = ScriptedModel([
            {"command": "exit 3"},
            {"command": "echo recovered"},
            {"done": True, "answer": "carried on"},
        ])
        tool = ComputerTaskTool(run_id="recover", root=tmp_path, llm=model)
        out = tool.execute(goal())
        assert "recovered" in out and "carried on" in out

    def test_an_unusable_reply_is_shown_and_retried(self, allow_terminal, tmp_path):
        """A dropped-JSON turn should be visible, not a silent dead end."""
        model = ScriptedModel([
            {"thinking": "no command here"},
            {"command": "echo back on track"},
            {"done": True, "answer": "fine"},
        ])
        tool = ComputerTaskTool(run_id="unusable", root=tmp_path, llm=model)
        out = tool.execute(goal())
        assert "unusable reply" in out
        assert "back on track" in out


class TestBounds:
    def test_stops_at_the_command_limit(self, allow_terminal, tmp_path):
        model = ScriptedModel([{"command": "echo x"}] * 20)
        tool = ComputerTaskTool(run_id="capped", root=tmp_path, llm=model, max_steps=4)
        out = tool.execute(goal())
        assert "4-command limit" in out

    def test_blocked_commands_are_refused_not_run(self, allow_terminal, tmp_path):
        model = ScriptedModel([
            {"command": "shutdown /s"},
            {"done": True, "answer": "stopped"},
        ])
        tool = ComputerTaskTool(run_id="blocked", root=tmp_path, llm=model)
        out = tool.execute(goal())
        assert "REFUSED" in out

    def test_commands_run_inside_the_workspace(self, allow_terminal, tmp_path):
        model = ScriptedModel([
            {"command": "echo marker > proof.txt"},
            {"done": True, "answer": "written"},
        ])
        tool = ComputerTaskTool(run_id="scoped", root=tmp_path, llm=model)
        tool.execute(goal())
        assert (tool.workspace / "proof.txt").exists()


class TestOptIn:
    def test_simulates_without_the_terminal_flag(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ORCHESTRATOR_ALLOW_TERMINAL", raising=False)
        model = ScriptedModel([{"command": "echo nope"}])
        tool = ComputerTaskTool(run_id="off", root=tmp_path, llm=model)
        out = tool.execute(goal())
        assert "[simulated:computer_task]" in out
        assert model.prompts == []          # nothing was even asked of the model

    def test_needs_a_model(self, allow_terminal, tmp_path):
        tool = ComputerTaskTool(run_id="nollm", root=tmp_path, llm=None)
        assert tool.is_live() is False
        with pytest.raises(ToolError, match="no model"):
            tool.execute(goal())

    def test_needs_a_goal(self, allow_terminal, tmp_path):
        tool = ComputerTaskTool(run_id="nogoal", root=tmp_path, llm=ScriptedModel([]))
        with pytest.raises(ToolError, match="no goal"):
            tool.execute("   ")

    def test_is_irreversible_so_approval_runs_first(self):
        assert ComputerTaskTool().irreversible is True


class TestLogging:
    def test_every_command_is_logged(self, allow_terminal, tmp_path):
        model = ScriptedModel([
            {"command": "echo logged"},
            {"done": True, "answer": "ok"},
        ])
        tool = ComputerTaskTool(run_id="logs", root=tmp_path, llm=model)
        tool.execute(goal())
        events = [json.loads(line) for line in
                  (tool.workspace / "computer_task.log.jsonl").read_text(
                      encoding="utf-8").splitlines()]
        assert [e["event"] for e in events] == ["start", "command", "finish"]
        assert events[1]["command"] == "echo logged"
