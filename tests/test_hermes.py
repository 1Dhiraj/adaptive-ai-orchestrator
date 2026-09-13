"""Hermes desktop worker integration."""

from __future__ import annotations

import subprocess

import pytest

from orchestrator.models import Step
from orchestrator.tools import default_tool_manager
from orchestrator.tools.hermes import HermesDesktopTool


def test_registered_in_default_catalogue():
    assert "hermes_desktop" in default_tool_manager()


def test_workflow_replaces_default_adapter_with_run_scoped_workspace(make_workflow):
    workflow = make_workflow(
        [Step(id="desktop", description="Open Calculator",
              agent_role="desktop operator", requires_tool="hermes_desktop")],
        run_id="desktop-run",
    )
    tool = workflow.tools.get("hermes_desktop")
    assert isinstance(tool, HermesDesktopTool)
    assert tool.workspace.name == "desktop-run"


@pytest.mark.parametrize("alias", ["desktop", "desktop_automation", "computer_use", "hermes"])
def test_common_names_resolve_to_hermes(alias):
    from orchestrator.tools.builtin import canonical_tool_name

    assert canonical_tool_name(alias) == "hermes_desktop"


def test_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("ORCHESTRATOR_ALLOW_DESKTOP", raising=False)
    monkeypatch.setattr("orchestrator.tools.hermes.shutil.which", lambda _: "hermes")
    tool = HermesDesktopTool("run", tmp_path)
    assert not tool.is_live()
    assert "not executed" in tool.execute("Open Calculator")


def test_missing_binary_is_an_honest_simulation(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
    monkeypatch.setattr("orchestrator.tools.hermes.shutil.which", lambda _: None)
    result = HermesDesktopTool("run", tmp_path).execute("Open Calculator")
    assert "not installed" in result


def test_live_call_uses_argv_without_shell_and_adds_guardrails(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
    monkeypatch.setattr("orchestrator.tools.hermes.shutil.which", lambda _: "C:/bin/hermes.exe")
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, "desktop work finished\n", "")

    monkeypatch.setattr("orchestrator.tools.hermes.subprocess.run", fake_run)
    result = HermesDesktopTool("run", tmp_path).execute("Open Calculator and calculate 2+2")

    assert seen["args"][:2] == ["C:/bin/hermes.exe", "-z"]
    assert "No administrator privileges" in seen["args"][2]
    assert "calculate 2+2" in seen["args"][2]
    assert seen["kwargs"]["cwd"].endswith("run")
    assert "shell" not in seen["kwargs"]
    assert "completed" in result


def test_desktop_action_uses_workflow_approval_gate(make_workflow, monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_DESKTOP", "1")
    monkeypatch.setattr("orchestrator.tools.hermes.shutil.which", lambda _: "hermes")
    tool = HermesDesktopTool("run", tmp_path)
    workflow = make_workflow(
        [Step(id="desktop", description="Prepare a report in Word",
              agent_role="desktop operator", requires_tool="hermes_desktop")],
        action_approval="all",
    )
    workflow.tools.register(tool)

    report = workflow.run_full()
    assert report.awaiting_action == ["desktop"]
    assert tool.call_count == 0
