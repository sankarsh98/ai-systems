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
