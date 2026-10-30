# AI Systems

A hands-on course in two tracks, built with Quartz:

- **A. Agents, tools and MCP:** the tool-use loop, tool design, and an MCP server and client.
- **B. Inference efficiency:** KV cache, quantization, and speculative decoding, all as runnable NumPy demos.

Site: https://sankarsh98.github.io/ai-systems/

```bash
npm ci && npx quartz plugin install
npx quartz build -d docs --serve              # preview the site (content lives in docs/)
uv venv && uv pip install numpy mcp anthropic
cd code/inference && python kv_cache.py       # run any demo
cd ../agents && python agent_loop.py
```

`python agent_loop.py --real` needs `ANTHROPIC_API_KEY` and is untested here.
