---
title: "4. MCP: a standard plug for tools"
---

## The problem

Every AI app used to wire up tools its own way. Ten apps and ten services meant up to 100 custom integrations.

The **Model Context Protocol (MCP)**, an open protocol introduced by Anthropic in 2024, standardizes the connection. A tool provider writes one **MCP server**. Any **MCP client** (a chat app, an IDE, your own agent) can use it.

```text
 AI app (client)  ←── MCP (JSON-RPC) ──→  MCP server  →  your files, DB, APIs
```

## What a server offers

| Capability | Meaning | Example |
|---|---|---|
| **Tools** | Actions the model can call | `calculate`, `read_note_tool` |
| **Resources** | Data the client can read by address | `notes://chess` |
| **Prompts** | Reusable prompt templates | "Summarize this note" |

Transport can be **stdio** (the client launches the server as a subprocess, good for local tools) or HTTP (for remote servers).

## A server in 30 lines

This server reuses the two tools from lesson 2 and adds a resource.

```python title="code/agents/mcp_server.py"
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
```

The SDK builds each tool's JSON schema from the function's type hints and docstring, so you don't write schemas by hand.

## A client that talks to it

This is what an AI app does behind the scenes: start the server, list its tools, call them.

```python title="code/agents/mcp_client_demo.py"
"""Talk to the MCP server the way an AI client would: list tools, then call one."""
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])
            print("schema for calculate:", tools.tools[0].input_schema)

            result = await session.call_tool("calculate", {"expression": "17 * 23 + 4"})
            print("calculate ->", result.content[0].text)

            bad = await session.call_tool("read_note_tool", {"name": "nope"})
            print("read_note_tool('nope') is_error =", bad.is_error, "|", bad.content[0].text)

            resource = await session.read_resource("notes://chess")
            print("resource notes://chess ->", resource.contents[0].text)


asyncio.run(main())
```

```bash
cd code/agents
python mcp_client_demo.py 2>/dev/null
```

```text
tools: ['calculate', 'read_note_tool']
schema for calculate: {'properties': {'expression': {'title': 'Expression', 'type': 'string'}}, 'required': ['expression'], 'type': 'object', 'title': 'calculateArguments'}
calculate -> 395
read_note_tool('nope') is_error = True | Error executing tool read_note_tool: no note called 'nope'; available: ai, chess, agents
resource notes://chess -> Alpha-beta pruning skips branches that cannot change the result.
```

## A lesson from the first run

My first version raised a plain `KeyError` inside the tool. The client saw only `Error executing tool read_note` and nothing about *why*, so a model would have no way to recover. The SDK replaces unexpected exceptions with that generic message.

The fix was to catch the error and raise the SDK's **`ToolError`**, whose message *is* sent to the client. That's lesson 3's "errors that say what to do next" applied at the protocol level.

> [!note] SDK version
> This course uses version 2 of the Python `mcp` package, where the high-level server class is `MCPServer` (it was `FastMCP` in version 1) and result fields are snake_case (`is_error`, not `isError`). If you install an older version, expect small differences.

## Connecting it to a real agent

The Anthropic SDK has helpers that convert MCP tools into tool definitions for the tool runner, so an agent can use any MCP server's tools without you writing wrappers. Check the SDK docs for the current helper names, since this area is still changing. I haven't run that integration here.

## Experiments

1. Replace the `NOTES` dictionary with files from a folder. Does the resource URI still work for names with slashes?
2. Add a tool that **writes** a note. Which rows of the permission table in lesson 3 apply, and how would you gate it?
3. Write a second client that calls `list_tools` and prints each tool's description. Is each one clear enough for a model to choose between?
