"""Tool ecosystem: fallback routing, failure injection, adapters."""

from __future__ import annotations

import pytest

from orchestrator.tools import ToolManager, ToolUnavailableError, default_tool_manager
from orchestrator.tools.base import SimulatedTool, Tool, ToolError
from orchestrator.tools.builtin import (
    ArtifactStoreTool,
    ConsoleNotifyTool,
    SQLiteLocalTool,
    _directive,
    _extract_sql,
    canonical_tool_name,
)

from .conftest import AlwaysFailsTool, CountingTool


class TestFallbackRouting:
    def test_primary_is_used_when_healthy(self, tools):
        assert tools.use("github", "payload").tool_used == "github"

    def test_falls_back_when_primary_is_broken(self, tools):
        tools.break_tool("github")
        invocation = tools.use("github", "payload")
        assert invocation.ok and invocation.used_fallback
        assert invocation.tool_used == "github_cli"

    def test_walks_the_whole_chain(self, tools):
        tools.break_tool("github")
        tools.break_tool("github_cli")
        invocation = tools.use("github", "payload")
        assert invocation.tool_used == "artifact_store"
        assert invocation.attempts == ["github", "github_cli", "artifact_store"]

    def test_raises_when_everything_is_broken(self, tools):
        for name in ("github", "github_cli", "artifact_store"):
            tools.break_tool(name)
        with pytest.raises(ToolUnavailableError) as excinfo:
            tools.use("github", "payload")
        assert "github" in str(excinfo.value)

    def test_repair_restores_the_primary(self, tools):
        tools.break_tool("github")
        tools.repair_tool("github")
        assert tools.use("github", "x").tool_used == "github"

    def test_same_capability_peers_are_candidates(self, tools):
        assert "postgres_cli" in tools.candidates_for("postgres")
        assert "sqlite_local" in tools.candidates_for("postgres")

    def test_unknown_tool_raises(self, tools):
        with pytest.raises(ToolUnavailableError):
            tools.use("no_such_tool", "payload")

    def test_break_unknown_tool_raises(self, tools):
        with pytest.raises(KeyError):
            tools.break_tool("no_such_tool")

    def test_a_raising_tool_is_routed_around(self):
        manager = ToolManager([
            AlwaysFailsTool(),
            SimulatedTool("backup", capability="explosive"),
        ])
        invocation = manager.use("always_fails", "payload")
        assert invocation.tool_used == "backup" and invocation.used_fallback
        assert "always fails" in invocation.errors["always_fails"]


class TestReporting:
    def test_history_and_stats_track_fallbacks(self, tools):
        tools.use("github", "a")
        tools.break_tool("github")
        tools.use("github", "b")
        stats = tools.stats()
        assert stats["invocations"] == 2
        assert stats["succeeded"] == 2
        assert stats["fallbacks_used"] == 1
        assert stats["success_rate"] == 1.0
        assert stats["broken"] == ["github"]

    def test_describe_lists_every_tool(self, tools):
        names = {entry["name"] for entry in tools.describe()}
        assert {"github", "postgres", "ci", "slack", "email", "rest_api"} <= names

    def test_failed_invocations_are_recorded(self):
        manager = ToolManager([AlwaysFailsTool()])
        with pytest.raises(ToolUnavailableError):
            manager.use("always_fails", "x")
        assert manager.stats()["failed"] == 1


class TestTerminalFallbacks:
    """Each capability chain must end in something that always works."""

    def test_artifact_store_writes_a_file(self, tmp_path, monkeypatch):
        import orchestrator.tools.builtin as builtin

        monkeypatch.setattr(builtin, "ARTIFACT_DIR", tmp_path)
        monkeypatch.setattr(builtin, "PROJECT_ROOT", tmp_path.parent)
        result = ArtifactStoreTool().execute("some output", {"step_id": "backend"})
        assert "artifact_store" in result
        assert list(tmp_path.glob("backend-*.md"))

    def test_sqlite_local_applies_real_sql(self, tmp_path):
        tool = SQLiteLocalTool(db_path=str(tmp_path / "scratch.sqlite3"))
        payload = "```sql\nCREATE TABLE users (id SERIAL PRIMARY KEY, email TEXT);\n```"
        assert "applied schema" in tool.execute(payload)

        import sqlite3

        with sqlite3.connect(str(tmp_path / "scratch.sqlite3")) as conn:
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "users" in names

    def test_sqlite_local_reports_when_there_is_no_sql(self, tmp_path):
        tool = SQLiteLocalTool(db_path=str(tmp_path / "s.sqlite3"))
        assert "no SQL found" in tool.execute("just prose, no statements here")

    def test_console_notify_always_live(self, capsys):
        assert ConsoleNotifyTool().is_live()
        ConsoleNotifyTool().execute("deploy finished")
        assert "deploy finished" in capsys.readouterr().out


class TestPayloadParsing:
    def test_directive_is_extracted(self):
        payload = 'Here is the plan.\nTOOL_DIRECTIVE: {"action": "create_file", "path": "x.md"}'
        assert _directive(payload)["action"] == "create_file"

    def test_missing_directive_yields_empty_dict(self):
        assert _directive("no directive here") == {}

    def test_malformed_directive_does_not_raise(self):
        assert _directive("TOOL_DIRECTIVE: {not json}") == {}

    def test_sql_is_pulled_from_a_fenced_block(self):
        payload = "Schema below.\n```sql\nCREATE TABLE t (id INT);\n```\nDone."
        assert _extract_sql(payload).startswith("CREATE TABLE")

    def test_prose_without_sql_yields_empty(self):
        assert _extract_sql("We will design a schema later.") == ""

    def test_aliases_are_canonicalised(self):
        assert canonical_tool_name("github_mcp") == "github"
        assert canonical_tool_name("postgres_mcp") == "postgres"
        assert canonical_tool_name("ci_tool") == "ci"
        assert canonical_tool_name(None) is None
        assert canonical_tool_name("Custom") == "custom"


class TestRetryInteraction:
    def test_a_tool_that_recovers_succeeds_on_a_later_call(self):
        tool = CountingTool(fail_times=1)
        manager = ToolManager([tool])
        with pytest.raises(ToolUnavailableError):
            manager.use("counting", "first")
        assert manager.use("counting", "second").ok
        assert tool.runs == 2

    def test_default_registry_is_isolated_between_calls(self):
        first, second = default_tool_manager(), default_tool_manager()
        first.break_tool("github")
        assert not second.get("github").is_broken
