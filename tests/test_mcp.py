"""Real MCP client integration, exercised against a genuine local MCP server.

``tests/fixtures/fake_mcp_server.py`` is a real MCP server (built on the
official SDK), launched as a stdio subprocess — this proves the client speaks
the actual protocol without needing network access or a Node/npx install.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from orchestrator.tools import ToolManager, ToolUnavailableError, default_tool_manager
from orchestrator.tools.mcp import (
    McpConnection,
    McpConnectionError,
    McpRegistry,
    McpServerSpec,
    McpServerUnavailable,
    McpTool,
    attach_mcp_servers,
    discover_mcp_tools,
    load_mcp_config,
)

FAKE_SERVER = str(Path(__file__).parent / "fixtures" / "fake_mcp_server.py")

pytestmark = pytest.mark.slow  # spawns a real subprocess; still fast (~1-2s)


@pytest.fixture
def fake_spec() -> McpServerSpec:
    return McpServerSpec(name="fake", transport="stdio", command=sys.executable, args=[FAKE_SERVER])


@pytest.fixture
def fake_tools(fake_spec):
    """Discovered tools from the fake server; closes the connection afterwards."""
    tools = discover_mcp_tools(fake_spec)
    yield tools
    tools[0]._connection.close()


class TestDiscovery:
    def test_every_declared_tool_is_found(self, fake_tools):
        assert {t.name for t in fake_tools} == {"echo", "add", "explode"}

    def test_descriptions_come_from_the_server(self, fake_tools):
        echo = next(t for t in fake_tools if t.name == "echo")
        assert "Echo" in echo.description

    def test_tools_report_live(self, fake_tools):
        assert all(t.is_live() for t in fake_tools)

    def test_each_tool_gets_a_distinct_capability_by_default(self, fake_tools):
        """Tools on one server are not interchangeable, so they must not share
        a capability -- that would let the router substitute one for another."""
        caps = {t.capability for t in fake_tools}
        assert len(caps) == len(fake_tools)
        assert all(c.startswith("mcp:fake:") for c in caps)

    def test_unreachable_server_raises_a_typed_error(self):
        spec = McpServerSpec(name="ghost", transport="stdio", command="no-such-executable-xyz",
                             connect_timeout_s=3.0)
        with pytest.raises(McpServerUnavailable):
            discover_mcp_tools(spec)

    def test_tool_prefix_is_applied(self, fake_spec):
        fake_spec.tool_prefix = "fs_"
        tools = discover_mcp_tools(fake_spec)
        try:
            assert {t.name for t in tools} == {"fs_echo", "fs_add", "fs_explode"}
        finally:
            tools[0]._connection.close()

    def test_custom_capability_overrides_the_default(self, fake_spec):
        fake_spec.capability = "vcs"
        tools = discover_mcp_tools(fake_spec)
        try:
            assert all(t.capability == "vcs" for t in tools)
        finally:
            tools[0]._connection.close()


class TestCalling:
    def test_single_string_arg_uses_the_raw_payload(self, fake_tools):
        echo = next(t for t in fake_tools if t.name == "echo")
        assert echo.execute("hello world") == "echo: hello world"

    def test_directive_supplies_structured_arguments(self, fake_tools):
        add = next(t for t in fake_tools if t.name == "add")
        payload = "compute it\nTOOL_DIRECTIVE: " + json.dumps({"arguments": {"a": 3, "b": 4}})
        assert add.execute(payload) == "7"

    def test_server_side_error_becomes_a_tool_error(self, fake_tools):
        from orchestrator.tools import ToolError

        boom = next(t for t in fake_tools if t.name == "explode")
        with pytest.raises(ToolError, match="always explodes"):
            boom.execute("x")

    def test_connection_is_reused_across_calls(self, fake_tools):
        echo = next(t for t in fake_tools if t.name == "echo")
        connection = echo._connection
        echo.execute("first")
        echo.execute("second")
        assert connection._thread is not None and connection._thread.is_alive()


class TestToolManagerIntegration:
    def test_mcp_tools_are_usable_through_the_manager(self, fake_spec):
        manager = ToolManager()
        registry = McpRegistry()
        registered = registry.attach(manager, specs=[fake_spec])
        try:
            assert "echo" in registered
            invocation = manager.use("echo", "hi there")
            assert invocation.ok and invocation.output == "echo: hi there"
        finally:
            registry.close_all()

    def test_mcp_tool_can_fall_back_to_a_builtin(self, fake_spec):
        """An MCP tool sharing a built-in capability becomes a real fallback."""
        fake_spec.capability = "vcs"
        manager = default_tool_manager()
        registry = McpRegistry()
        registry.attach(manager, specs=[fake_spec])
        try:
            manager.break_tool("github")
            manager.break_tool("github_cli")
            invocation = manager.use("github", "some payload")
            assert invocation.tool_used in {"echo", "add", "explode", "artifact_store"}
            assert invocation.used_fallback
        finally:
            registry.close_all()

    def test_a_broken_mcp_server_does_not_block_other_servers(self):
        good = McpServerSpec(name="fake", transport="stdio", command=sys.executable, args=[FAKE_SERVER])
        bad = McpServerSpec(name="ghost", transport="stdio", command="no-such-executable-xyz",
                            connect_timeout_s=3.0)
        manager = ToolManager()
        registered = attach_mcp_servers(manager, specs=[bad, good])
        try:
            assert "echo" in registered
        finally:
            for name in manager.names:
                tool = manager.get(name)
                if isinstance(tool, McpTool):
                    tool._connection.close()


class TestWorkflowIntegration:
    """The point of all this: an agent in a real workflow calling a real MCP tool."""

    def test_agent_step_calls_an_mcp_tool(self, fake_spec, stub_llm):
        from orchestrator import Step, Workflow

        workflow = Workflow(
            [Step(id="summarise", description="Summarise the findings.",
                  agent_role="writer", requires_tool="echo")],
            run_id="mcp-wf", persist=False, verbose=False, llm=stub_llm,
        )
        try:
            registered = workflow.attach_mcp(specs=[fake_spec])
            assert "echo" in registered

            report = workflow.run_full()
            assert report.ok and report.executed == ["summarise"]

            result = workflow.results["summarise"]
            assert result.tool_used == "echo"
            assert not result.used_fallback
            # The MCP server genuinely saw the agent's output and echoed it.
            assert "echo: " in result.output
        finally:
            workflow.close()

    def test_mcp_tool_failure_routes_to_a_fallback(self, fake_spec, stub_llm):
        from orchestrator import Step, Workflow

        fake_spec.capability = "vcs"          # share the github chain
        fake_spec.tool_prefix = "mcp_"
        workflow = Workflow(
            [Step(id="publish", description="Publish the notes.",
                  agent_role="devops", requires_tool="mcp_explode")],
            run_id="mcp-fallback", persist=False, verbose=False, llm=stub_llm,
        )
        try:
            workflow.attach_mcp(specs=[fake_spec])
            report = workflow.run_full()
            # mcp_explode always fails, so the router must fall through to a
            # working same-capability tool rather than failing the step.
            assert report.ok
            assert workflow.results["publish"].used_fallback
            assert workflow.results["publish"].tool_used != "mcp_explode"
        finally:
            workflow.close()

    def test_workflow_close_shuts_down_mcp_connections(self, fake_spec, stub_llm):
        from orchestrator import Workflow

        workflow = Workflow([], run_id="mcp-close", persist=False, verbose=False, llm=stub_llm)
        workflow.attach_mcp(specs=[fake_spec])
        connection = workflow.tools.get("echo")._connection
        assert connection.connected
        workflow.close()
        assert not connection.connected


class TestConfigLoading:
    def test_missing_file_yields_no_servers(self, tmp_path):
        assert load_mcp_config(str(tmp_path / "nope.json")) == []

    def test_stdio_server_is_parsed(self, tmp_path):
        config = tmp_path / ".mcp.json"
        config.write_text(json.dumps({"mcpServers": {
            "fake": {"command": sys.executable, "args": [FAKE_SERVER]},
        }}), encoding="utf-8")
        specs = load_mcp_config(str(config))
        assert len(specs) == 1
        assert specs[0].transport == "stdio" and specs[0].command == sys.executable

    def test_url_server_defaults_to_streamable_http(self, tmp_path):
        config = tmp_path / ".mcp.json"
        config.write_text(json.dumps({"mcpServers": {
            "remote": {"url": "https://example.com/mcp"},
        }}), encoding="utf-8")
        specs = load_mcp_config(str(config))
        assert specs[0].transport == "streamable_http"

    def test_explicit_sse_transport_is_respected(self, tmp_path):
        config = tmp_path / ".mcp.json"
        config.write_text(json.dumps({"mcpServers": {
            "remote": {"url": "https://example.com/mcp", "transport": "sse"},
        }}), encoding="utf-8")
        assert load_mcp_config(str(config))[0].transport == "sse"

    def test_attach_mcp_servers_reads_the_default_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = tmp_path / ".mcp.json"
        config.write_text(json.dumps({"mcpServers": {
            "fake": {"command": sys.executable, "args": [FAKE_SERVER]},
        }}), encoding="utf-8")
        manager = ToolManager()
        registered = attach_mcp_servers(manager)  # default config_path=".mcp.json"
        try:
            assert set(registered) == {"echo", "add", "explode"}
        finally:
            for name in registered:
                manager.get(name)._connection.close()


class TestPromptHints:
    """The agent must be told a tool's real signature, not just its name."""

    def test_multi_arg_schema_is_described(self, fake_tools):
        add = next(t for t in fake_tools if t.name == "add")
        hint = add.prompt_hint()
        assert '"a": <integer>' in hint and '"b": <integer>' in hint

    def test_optional_fields_are_flagged(self, fake_tools):
        echo = next(t for t in fake_tools if t.name == "echo")
        # 'text' is required on the fake server, so it carries no '?' marker.
        assert '"text": <string>' in echo.prompt_hint()

    def test_zero_arg_tool_says_so(self, fake_tools):
        explode = next(t for t in fake_tools if t.name == "explode")
        assert "no arguments" in explode.prompt_hint().lower()

    def test_agent_prompt_includes_the_schema(self, fake_spec, stub_llm):
        from orchestrator import Step
        from orchestrator.agents import Agent
        from orchestrator.tools import ToolManager

        manager = ToolManager()
        tools = discover_mcp_tools(fake_spec)
        try:
            for tool in tools:
                manager.register(tool)
            step = Step(id="c", description="Add them.", agent_role="backend",
                        requires_tool="add")
            prompt = Agent("backend", llm=stub_llm).build_prompt(step, "(none)", manager)
            assert '"a": <integer>' in prompt
            assert "TOOL_DIRECTIVE" in prompt
        finally:
            tools[0]._connection.close()

    def test_builtin_tools_still_get_a_hint(self):
        manager = default_tool_manager()
        assert manager.get("github").prompt_hint()


class TestArgumentBuilding:
    def test_no_input_properties_calls_with_empty_arguments(self, fake_tools):
        explode = next(t for t in fake_tools if t.name == "explode")
        assert explode._build_arguments("anything") == {}

    def test_multiple_required_fields_fall_back_to_input_key(self, fake_tools):
        add = next(t for t in fake_tools if t.name == "add")
        assert add._build_arguments("some prose, not json") == {"input": "some prose, not json"}


class TestSpecValidation:
    def test_stdio_requires_a_command(self):
        with pytest.raises(ValueError, match="command"):
            McpServerSpec(name="x", transport="stdio")

    def test_http_requires_a_url(self):
        with pytest.raises(ValueError, match="url"):
            McpServerSpec(name="x", transport="streamable_http")

    def test_unknown_transport_is_rejected(self):
        with pytest.raises(ValueError, match="transport"):
            McpServerSpec(name="x", transport="carrier_pigeon", command="x")


class TestConnectionLifecycle:
    def test_double_close_is_a_no_op(self, fake_spec):
        connection = McpConnection(fake_spec)
        connection.connect()
        connection.close()
        connection.close()  # must not raise or hang

    def test_calling_after_close_reconnects(self, fake_spec):
        """A closed (or crashed) session restarts rather than using a dead loop."""
        connection = McpConnection(fake_spec)
        connection.connect()
        first_thread = connection._thread
        connection.close()
        assert not connection.connected

        assert connection.call_tool("echo", {"text": "x"}) == "echo: x"
        assert connection.connected
        assert connection._thread is not first_thread  # genuinely a new session
        connection.close()

    def test_unreachable_server_raises_on_every_attempt(self):
        """A permanently broken server must not look healthy after one failure."""
        spec = McpServerSpec(name="ghost", transport="stdio",
                             command="no-such-executable-xyz", connect_timeout_s=3.0)
        connection = McpConnection(spec)
        for _ in range(2):
            with pytest.raises(McpConnectionError):
                connection.connect()


class TestCapabilityIsolation:
    """MCP tools on one server are not interchangeable and must not
    substitute for each other -- 'evaluate' falling back to 'close' shut the
    page instead of reading it."""

    def test_each_tool_gets_its_own_capability(self, fake_tools):
        caps = {t.capability for t in fake_tools}
        assert len(caps) == len(fake_tools), "tools must not share a capability"

    def test_one_tool_never_falls_back_to_a_sibling(self, fake_spec):
        from orchestrator.tools import ToolManager

        manager = ToolManager()
        tools = discover_mcp_tools(fake_spec)
        try:
            for tool in tools:
                manager.register(tool)
            # 'explode' always fails; it must NOT silently become 'echo'.
            assert manager.candidates_for("explode") == ["explode"]
            with pytest.raises(ToolUnavailableError):
                manager.use("explode", "x")
        finally:
            tools[0]._connection.close()

    def test_an_explicit_shared_capability_is_still_honoured(self, fake_spec):
        fake_spec.capability = "notify"
        tools = discover_mcp_tools(fake_spec)
        try:
            assert all(t.capability == "notify" for t in tools)
        finally:
            tools[0]._connection.close()
