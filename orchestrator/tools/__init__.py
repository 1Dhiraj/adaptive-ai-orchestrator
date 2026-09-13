"""Tool ecosystem: adapters, capability-based fallback routing."""

from .base import (
    SimulatedTool,
    Tool,
    ToolError,
    ToolInvocation,
    ToolManager,
    ToolUnavailableError,
)
from .builtin import (
    ArtifactStoreTool,
    CITool,
    ConsoleNotifyTool,
    EmailTool,
    GitHubCLITool,
    GitHubTool,
    LocalTestRunnerTool,
    PostgresCLITool,
    PostgresTool,
    RestApiTool,
    StripeTool,
    S3Tool,
    SQLiteLocalTool,
    SlackTool,
    TOOL_ALIASES,
    canonical_tool_name,
    default_tool_manager,
)
from .hermes import HermesDesktopTool

__all__ = [
    "Tool", "SimulatedTool", "ToolManager", "ToolError", "ToolUnavailableError",
    "ToolInvocation", "default_tool_manager", "canonical_tool_name", "TOOL_ALIASES",
    "GitHubTool", "GitHubCLITool", "ArtifactStoreTool",
    "PostgresTool", "PostgresCLITool", "SQLiteLocalTool",
    "CITool", "LocalTestRunnerTool",
    "SlackTool", "EmailTool", "ConsoleNotifyTool", "RestApiTool", "StripeTool", "S3Tool",
    "HermesDesktopTool",
    # MCP names are re-exported lazily via __getattr__ below.
    "McpTool", "McpServerSpec", "McpRegistry",
    "attach_mcp_servers", "discover_mcp_tools", "load_mcp_config",
]

#: MCP support needs the optional ``mcp`` SDK, so it is imported on first use
#: rather than at package import -- the orchestrator must still work without it.
_MCP_EXPORTS = {
    "McpTool", "McpServerSpec", "McpConnection", "McpRegistry",
    "McpServerUnavailable", "McpConnectionError",
    "attach_mcp_servers", "discover_mcp_tools", "load_mcp_config",
}


def __getattr__(name: str):
    if name in _MCP_EXPORTS:
        try:
            from . import mcp as _mcp
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                f"'{name}' needs the Model Context Protocol SDK. Install it with: pip install mcp"
            ) from exc
        return getattr(_mcp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
