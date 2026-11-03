"""The full Codex harness must write real files before a software step passes."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator.tools.base import ToolError
from orchestrator.tools.codex_agent import CodexAgentTool
import orchestrator.tools.codex_agent as codex_agent


def fake_provider(monkeypatch):
    monkeypatch.setenv("CODEX_AGENT_BACKEND", "exec")
    monkeypatch.setenv("CODEX_AGENT_MODEL", "test-model")
    monkeypatch.setattr(codex_agent, "CodexSubscriptionProvider",
                        lambda: SimpleNamespace(executable="codex-test", model="test-model"))


def test_codex_agent_runs_in_the_run_workspace_and_verifies_file(tmp_path, monkeypatch):
    fake_provider(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    seen = {}

    def run(command, **kwargs):
        seen.update(command=command, kwargs=kwargs)
        workspace = Path(command[command.index("-C") + 1])
        (workspace / "todo.html").write_text("<h1>Todo</h1>", encoding="utf-8")
        Path(command[command.index("--output-last-message") + 1]).write_text(
            "Created todo.html and checked it.", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout='{"type":"turn.completed"}\n', stderr="")

    monkeypatch.setattr(codex_agent.subprocess, "run", run)
    tool = CodexAgentTool("test-run", tmp_path)
    output = tool.execute('TOOL_DIRECTIVE: {"arguments":{"goal":"Create a todo app"}}')

    assert "todo.html" in output
    assert (tmp_path / "test-run" / "todo.html").is_file()
    assert seen["command"][seen["command"].index("--sandbox") + 1] == "workspace-write"
    assert seen["kwargs"]["env"].get("OPENAI_API_KEY") is None


def test_codex_agent_cannot_claim_creation_without_a_file(tmp_path, monkeypatch):
    fake_provider(monkeypatch)

    def run(command, **kwargs):
        Path(command[command.index("--output-last-message") + 1]).write_text(
            "Done", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(codex_agent.subprocess, "run", run)
    with pytest.raises(ToolError, match="changed no workspace files"):
        CodexAgentTool("empty-run", tmp_path).execute(
            'TOOL_DIRECTIVE: {"arguments":{"goal":"Create a todo app"}}')


def test_codex_agent_reports_the_model_error_not_a_cli_warning(tmp_path, monkeypatch):
    fake_provider(monkeypatch)

    def run(command, **kwargs):
        return SimpleNamespace(returncode=1, stderr="WARN irrelevant icon",
                               stdout='{"type":"turn.failed","error":{"message":"Selected model is at capacity"}}\n')

    monkeypatch.setattr(codex_agent.subprocess, "run", run)
    with pytest.raises(ToolError, match="Selected model is at capacity"):
        CodexAgentTool("capacity-run", tmp_path).execute(
            'TOOL_DIRECTIVE: {"arguments":{"goal":"Create a todo app"}}')


def test_agent_receives_task_goal_without_orchestrator_tool_instructions():
    goal = CodexAgentTool._goal(
        'TOOL_DIRECTIVE: {"arguments":{"goal":"Create greeting.html"}}',
        {"step_description": "Use codex_agent with a TOOL_DIRECTIVE"})
    assert goal == "Create greeting.html"
