# AI Systems

A hands-on course in two tracks, built with Docusaurus:

- **A. Agents, tools and MCP:** the tool-use loop, tool design, and an MCP server and client.
- **B. Inference efficiency:** KV cache, quantization, and speculative decoding, all as runnable NumPy demos.

```bash
npm install && npm start                      # preview the site
uv venv && uv pip install numpy mcp anthropic
cd code/inference && python kv_cache.py       # run any demo
cd ../agents && python agent_loop.py
```

`python agent_loop.py --real` needs `ANTHROPIC_API_KEY` and is untested here.
