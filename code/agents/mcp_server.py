"""A tiny MCP server exposing the same two tools, plus a resource, over stdio."""
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from tools import calculate, read_note

mcp = MCPServer("notes-and-math")
mcp.tool()(calculate)


@mcp.tool()
def read_note_tool(name: str) -> str:
    """Read one of the user's notes by name."""
    try:
        return read_note(name)
    except KeyError as exc:
        # ToolError messages reach the model; other exceptions are replaced by a generic one.
        raise ToolError(str(exc).strip("'\"")) from exc


@mcp.resource("notes://{name}")
def note_resource(name: str) -> str:
    """Notes as addressable resources."""
    return read_note(name)


if __name__ == "__main__":
    mcp.run()  # stdio transport
