"""Real MCP (Model Context Protocol) client integration.

Lets agents call tools exposed by actual MCP servers — the same protocol
Claude Desktop and Claude Code use — alongside the built-in GitHub/Postgres/
Slack adapters, participating in the same capability/fallback routing.

Each configured server gets one background thread running a persistent
asyncio event loop that holds the MCP session open (spawning the server
subprocess or reconnecting the HTTP session per call would be far too slow).
Synchronous callers — ``Tool.execute()`` runs on ordinary threads — hand work
to that loop via ``asyncio.run_coroutine_threadsafe`` and block on the result,
so the rest of the orchestrator never has to know MCP is async under the hood.

Config follows the same ``{"mcpServers": {...}}`` shape Claude Desktop and
Claude Code use, so an existing ``.mcp.json`` can be pointed at directly::

    {
      "mcpServers": {
        "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
        "weather":    {"url": "https://example.com/mcp", "transport": "streamable_http"}
      }
    }

Usage::

    from orchestrator.tools.mcp import attach_mcp_servers
    from orchestrator.tools import default_tool_manager

    tools = default_tool_manager()
    attached = attach_mcp_servers(tools, config_path=".mcp.json")
    # attached -> ["read_file", "list_directory", ...] -- now usable as
    # Step(..., requires_tool="read_file")
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolError, ToolManager
from .builtin import _directive

_DEFAULT_CONNECT_TIMEOUT_S = 20.0
_DEFAULT_CALL_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------


@dataclass
class McpServerSpec:
    """How to reach one MCP server."""

    name: str
    transport: str = "stdio"          # "stdio" | "sse" | "streamable_http"

    # stdio
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None

    # sse / streamable_http
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    #: Capability bucket for fallback routing. Tools discovered on this
    #: server share this capability, so they can fall back to each other —
    #: and to a built-in tool, if you point this at one, e.g. "vcs".
    capability: Optional[str] = None
    #: Extra fallback tool names, tried after same-capability peers.
    fallbacks: List[str] = field(default_factory=list)
    #: Prefix applied to every discovered tool name, to dodge collisions
    #: when two servers happen to expose a tool with the same name.
    tool_prefix: str = ""

    connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S
    call_timeout_s: float = _DEFAULT_CALL_TIMEOUT_S

    def __post_init__(self) -> None:
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"MCP server '{self.name}': stdio transport needs 'command'")
        if self.transport in {"sse", "streamable_http"} and not self.url:
            raise ValueError(f"MCP server '{self.name}': {self.transport} transport needs 'url'")
        if self.transport not in {"stdio", "sse", "streamable_http"}:
            raise ValueError(f"MCP server '{self.name}': unknown transport '{self.transport}'")


def load_mcp_config(path: str = ".mcp.json") -> List[McpServerSpec]:
    """Load server specs from a Claude Desktop/Code-style ``.mcp.json``.

    Missing file -> empty list (MCP is opt-in; no config means no servers).
    """
    config_path = Path(path)
    if not config_path.exists():
        return []

    data = json.loads(config_path.read_text(encoding="utf-8"))
    specs = []
    for name, entry in (data.get("mcpServers") or {}).items():
        if "url" in entry:
            specs.append(McpServerSpec(
                name=name,
                transport=entry.get("transport", "streamable_http"),
                url=entry["url"],
                headers=entry.get("headers"),
                capability=entry.get("capability"),
                fallbacks=entry.get("fallbacks", []),
                tool_prefix=entry.get("tool_prefix", ""),
                connect_timeout_s=float(entry.get("connect_timeout_s", _DEFAULT_CONNECT_TIMEOUT_S)),
                call_timeout_s=float(entry.get("call_timeout_s", _DEFAULT_CALL_TIMEOUT_S)),
            ))
        else:
            specs.append(McpServerSpec(
                name=name,
                transport="stdio",
                command=entry["command"],
                args=entry.get("args", []),
                env=entry.get("env"),
                cwd=entry.get("cwd"),
                capability=entry.get("capability"),
                fallbacks=entry.get("fallbacks", []),
                tool_prefix=entry.get("tool_prefix", ""),
                connect_timeout_s=float(entry.get("connect_timeout_s", _DEFAULT_CONNECT_TIMEOUT_S)),
                call_timeout_s=float(entry.get("call_timeout_s", _DEFAULT_CALL_TIMEOUT_S)),
            ))
    return specs


# ---------------------------------------------------------------------------
# The sync-over-async bridge
# ---------------------------------------------------------------------------


class McpConnectionError(ToolError):
    """The server could not be reached or failed to initialize."""


class McpConnection:
    """One persistent session to one MCP server, driven from a background thread.

    Not a :class:`Tool` itself — a single connection backs every tool the
    server exposes, since opening a new subprocess/HTTP session per call
    would defeat the point of keeping agents responsive.
    """

    def __init__(self, spec: McpServerSpec):
        self.spec = spec
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._close_event: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._error: Optional[BaseException] = None
        self._session: Optional[Any] = None
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return (self._ready.is_set() and self._error is None
                and not self._closed.is_set()
                and self._thread is not None and self._thread.is_alive())

    # -- lifecycle ----------------------------------------------------------

    def _needs_start(self) -> bool:
        """True if there is no usable session -- never started, closed, or died."""
        return (self._thread is None
                or not self._thread.is_alive()
                or self._closed.is_set()
                or self._error is not None)

    def connect(self) -> None:
        """Ensure a live session, starting or restarting one as needed.

        Idempotent, thread-safe, and *reconnecting*: if the previous session
        was closed or its server process died, this starts a fresh one rather
        than handing back a dead event loop. That makes a crashed MCP server
        recoverable the same way a broken tool is.
        """
        with self._lock:
            if self._needs_start():
                # Let any half-finished shutdown settle before resetting, so
                # the old thread's teardown cannot clobber the new state.
                previous = self._thread
                if previous is not None and previous.is_alive():
                    previous.join(timeout=5.0)

                self._ready.clear()
                self._closed.clear()
                self._error = None
                self._session = None
                self._loop = None
                self._close_event = None
                self._thread = threading.Thread(
                    target=self._thread_main, name=f"mcp-{self.spec.name}", daemon=True)
                self._thread.start()

        if not self._ready.wait(self.spec.connect_timeout_s):
            raise McpConnectionError(
                f"MCP server '{self.spec.name}' did not become ready within "
                f"{self.spec.connect_timeout_s}s")
        if self._error is not None:
            raise McpConnectionError(f"MCP server '{self.spec.name}' failed to start: {self._error}")

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            if self._thread is None or self._closed.is_set():
                return
            if self._loop is not None and self._close_event is not None:
                self._loop.call_soon_threadsafe(self._close_event.set)
        self._thread.join(timeout)

    # -- calls (safe to call from any thread) --------------------------------

    def list_tools(self) -> List[Any]:
        result = self._call_session(lambda session: session.list_tools())
        return list(result.tools)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        result = self._call_session(lambda session: session.call_tool(name, arguments))
        text = "\n".join(
            block.text for block in result.content
            if getattr(block, "type", None) == "text"
        )
        if getattr(result, "isError", False):
            raise ToolError(f"MCP tool '{name}' returned an error: {text or '(no message)'}")
        return text or f"[mcp:{self.spec.name}] '{name}' returned no text content"

    def _call_session(self, coro_factory) -> Any:
        self.connect()  # no-op when healthy; restarts a dead session
        loop, session = self._loop, self._session
        if loop is None or session is None:  # pragma: no cover - connect() raises first
            raise McpConnectionError(f"MCP server '{self.spec.name}' has no active session")

        coro = coro_factory(session)
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError as exc:
            # The loop closed between connect() and here. Close the coroutine
            # explicitly -- it was never awaited, and Python warns otherwise.
            coro.close()
            raise ToolError(
                f"MCP session for '{self.spec.name}' is no longer running: {exc}") from exc

        try:
            return future.result(self.spec.call_timeout_s)
        except FutureTimeoutError as exc:
            future.cancel()
            raise ToolError(
                f"MCP call to '{self.spec.name}' timed out after {self.spec.call_timeout_s}s"
            ) from exc
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - protocol/transport errors are all routable
            raise ToolError(f"MCP call to '{self.spec.name}' failed: {exc}") from exc

    # -- background thread ---------------------------------------------------

    def _thread_main(self) -> None:
        # ProactorEventLoop is required for asyncio subprocess support on
        # Windows; something else in the process (e.g. uvicorn) may have
        # changed the global policy, so pin it explicitly for this thread.
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            asyncio.run(self._async_main())
        finally:
            self._closed.set()
            self._ready.set()  # unblock a connect() that raced the shutdown

    async def _async_main(self) -> None:
        from mcp import ClientSession

        self._loop = asyncio.get_running_loop()
        self._close_event = asyncio.Event()

        try:
            async with self._open_transport() as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._close_event.wait()
        except Exception as exc:  # noqa: BLE001 - surfaced to connect() callers
            self._error = exc
            self._ready.set()

    def _open_transport(self):
        spec = self.spec
        if spec.transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            return stdio_client(StdioServerParameters(
                command=spec.command, args=spec.args, env=spec.env, cwd=spec.cwd))
        if spec.transport == "sse":
            from mcp.client.sse import sse_client

            return sse_client(spec.url, headers=spec.headers)
        if spec.transport == "streamable_http":
            from mcp.client.streamable_http import streamablehttp_client

            return streamablehttp_client(spec.url, headers=spec.headers)
        raise McpConnectionError(f"unknown transport '{spec.transport}'")  # pragma: no cover


# ---------------------------------------------------------------------------
# The Tool adapter
# ---------------------------------------------------------------------------


class McpTool(Tool):
    """One tool exposed by one MCP server, wired into the fallback router.

    ``execute()``'s payload becomes the call arguments: a trailing
    ``TOOL_DIRECTIVE: {"arguments": {...}}`` block (the same convention the
    built-in adapters use) is preferred when present; otherwise the payload
    is matched against the tool's declared input schema on a best-effort
    basis — the single required string field if there is exactly one, else
    passed as ``{"input": task}``.
    """

    def __init__(self, connection: McpConnection, mcp_tool: Any, spec: McpServerSpec):
        super().__init__()
        self._connection = connection
        self._mcp_tool = mcp_tool
        self._spec = spec
        self.name = f"{spec.tool_prefix}{mcp_tool.name}"
        # Each MCP tool gets its OWN capability by default. Tools sharing a
        # capability are treated as interchangeable fallbacks, and the tools on
        # one server almost never are: falling back from browser_evaluate to
        # browser_close does not read the page, it shuts it. Set `capability`
        # explicitly in the config only when the tools really are equivalent.
        self.capability = spec.capability or f"mcp:{spec.name}:{mcp_tool.name}"
        self.description = mcp_tool.description or f"MCP tool '{mcp_tool.name}' on '{spec.name}'"
        self.fallbacks = list(spec.fallbacks)

    def is_live(self) -> bool:
        return True  # a real server call, not a simulation, whenever it runs

    def prompt_hint(self) -> str:
        """Describe the tool's real input schema, so the agent can call it.

        Without this the model is told a tool exists but not what arguments
        it takes, and any tool needing more than one field is uncallable.
        """
        schema = getattr(self._mcp_tool, "inputSchema", None) or {}
        properties = schema.get("properties") or {}
        if not properties:
            return f"{self.description} Takes no arguments."

        required = set(schema.get("required") or [])
        fields = []
        for field_name, field_schema in properties.items():
            field_type = field_schema.get("type", "any")
            flag = "" if field_name in required else "?"
            fields.append(f'"{field_name}{flag}": <{field_type}>')
        return (f"{self.description}\n"
                f"Arguments schema: {{{', '.join(fields)}}}   (? = optional)")

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        arguments = self._build_arguments(task)
        return self._connection.call_tool(self._mcp_tool.name, arguments)

    def _build_arguments(self, task: str) -> Dict[str, Any]:
        directive = _directive(task)
        if "arguments" in directive and isinstance(directive["arguments"], dict):
            return directive["arguments"]

        schema = getattr(self._mcp_tool, "inputSchema", None) or {}
        required = schema.get("required") or []
        properties = schema.get("properties") or {}
        string_required = [
            field for field in required
            if properties.get(field, {}).get("type") == "string"
        ]
        if len(string_required) == 1:
            return {string_required[0]: task}
        if not properties:
            return {}
        return {"input": task}


# ---------------------------------------------------------------------------
# Discovery and registration
# ---------------------------------------------------------------------------


class McpServerUnavailable(RuntimeError):
    """Raised (and normally caught) when a configured server can't be reached."""


def discover_mcp_tools(spec: McpServerSpec, connection: Optional[McpConnection] = None) -> List[McpTool]:
    """Connect to one server and wrap every tool it exposes.

    Raises :class:`McpServerUnavailable` if the server cannot be reached —
    callers that want to keep going despite one bad server (like
    :func:`attach_mcp_servers`) should catch it.
    """
    connection = connection or McpConnection(spec)
    try:
        connection.connect()
        mcp_tools = connection.list_tools()
    except ToolError as exc:
        raise McpServerUnavailable(str(exc)) from exc
    return [McpTool(connection, tool, spec) for tool in mcp_tools]


def attach_mcp_servers(
    tool_manager: ToolManager,
    specs: Optional[List[McpServerSpec]] = None,
    config_path: str = ".mcp.json",
    on_error: Optional[Any] = None,
) -> List[str]:
    """Discover and register tools from every configured MCP server.

    Servers that fail to connect are skipped rather than raised, so one
    misconfigured server does not prevent the others (or the rest of the
    orchestrator) from starting; ``on_error(spec, exc)`` is called for each
    one, if given, otherwise the failure is printed.

    Returns the names of every tool that was successfully registered.
    """
    specs = specs if specs is not None else load_mcp_config(config_path)
    registered: List[str] = []

    for spec in specs:
        try:
            for tool in discover_mcp_tools(spec):
                tool_manager.register(tool)
                registered.append(tool.name)
        except McpServerUnavailable as exc:
            if on_error:
                on_error(spec, exc)
            else:
                print(f"[mcp] skipping server '{spec.name}': {exc}")

    return registered


class McpRegistry:
    """Tracks live connections so they can be closed together on shutdown.

    ``attach_mcp_servers`` is stateless and fine for one-shot setup; use this
    when you need to close connections cleanly (e.g. in a long-running
    dashboard process).
    """

    def __init__(self) -> None:
        self._connections: Dict[str, McpConnection] = {}

    def attach(self, tool_manager: ToolManager, specs: Optional[List[McpServerSpec]] = None,
               config_path: str = ".mcp.json") -> List[str]:
        specs = specs if specs is not None else load_mcp_config(config_path)
        registered: List[str] = []
        for spec in specs:
            connection = McpConnection(spec)
            try:
                for tool in discover_mcp_tools(spec, connection):
                    tool_manager.register(tool)
                    registered.append(tool.name)
                self._connections[spec.name] = connection
            except McpServerUnavailable as exc:
                print(f"[mcp] skipping server '{spec.name}': {exc}")
        return registered

    def close_all(self) -> None:
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()
