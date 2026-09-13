"""A tiny real MCP server, launched as a subprocess over stdio in tests.

This is a genuine MCP server (built on the official SDK's FastMCP helper),
not a mock — the point is to prove the client speaks the real protocol
without depending on network access or a Node/npx install.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake-tools")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the given text back, prefixed."""
    return f"echo: {text}"


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def explode() -> str:
    """Always raises, to exercise the tool-error path."""
    raise RuntimeError("the fake tool always explodes")


if __name__ == "__main__":
    mcp.run(transport="stdio")
