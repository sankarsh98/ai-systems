---
title: "2. The tool-use loop"
---

## The protocol

When you give the Claude API a list of tools, a response can contain **`tool_use` blocks**. Each one has an `id`, a tool `name`, and an `input` dictionary. Your job:

1. Append the model's response to the conversation.
2. Run each requested tool.
3. Send the outputs back in **one** user message of `tool_result` blocks. Each result carries the `tool_use_id` of the request it answers.
4. Repeat until the response has no `tool_use` blocks (`stop_reason` is not `"tool_use"`).

If a tool fails, return the error text with `is_error: true`. Don't crash the loop. The model can often fix its own mistake once it sees what went wrong.

## The tools

Two small tools: a safe calculator (it parses arithmetic with `ast`, so there's no `eval`) and a note reader that raises an error for unknown notes.

```python title="code/agents/tools.py"
"""Two small tools shared by the agent loop and the MCP server."""
import ast
import operator

NOTES = {
    "ai": "Transformers predict the next token. Attention lets each token look at earlier ones.",
    "chess": "Alpha-beta pruning skips branches that cannot change the result.",
    "agents": "An agent is a loop: the model picks a tool, we run it, and we show it the result.",
}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression such as '17 * 23 + 4'. No names, no calls."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    return str(ev(ast.parse(expression, mode="eval").body))


def read_note(name: str) -> str:
    """Return the text of a note, or an error message if it does not exist."""
    if name not in NOTES:
        raise KeyError(f"no note called {name!r}; available: {', '.join(NOTES)}")
    return NOTES[name]


TOOL_FUNCTIONS = {"calculate": calculate, "read_note": read_note}

TOOL_SCHEMAS = [
    {"name": "calculate", "description": "Evaluate an arithmetic expression.",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}},
                      "required": ["expression"]}},
    {"name": "read_note", "description": "Read one of the user's notes by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}},
                      "required": ["name"]}},
]
```

## The loop

`run_agent` is the whole agent. The model is a function you pass in, so the same loop works with a fake model offline and the real API.

```python title="code/agents/agent_loop.py"
"""The agent loop, with a swappable model.

`run_agent` is the same loop you would write against the real Claude API.
`ScriptedModel` plays the model's part offline, so the mechanics run anywhere.
`claude_model` calls the real API (needs ANTHROPIC_API_KEY).
"""
from dataclasses import dataclass, field

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS


# --- Block and response shapes that mirror the Messages API ----------------
@dataclass
class Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class Response:
    content: list
    stop_reason: str


# --- The loop --------------------------------------------------------------
def run_agent(call_model, user_input, max_turns=8, verbose=True):
    messages = [{"role": "user", "content": user_input}]
    for turn in range(1, max_turns + 1):
        response = call_model(messages)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":          # the model is done
            return next(b.text for b in response.content if b.type == "text")

        results = []
        for block in (b for b in response.content if b.type == "tool_use"):
            try:
                output, is_error = TOOL_FUNCTIONS[block.name](**block.input), False
            except Exception as exc:                    # tell the model, don't crash
                output, is_error = f"Error: {exc}", True
            if verbose:
                print(f"  turn {turn}: {block.name}({block.input}) -> {output}")
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": str(output), "is_error": is_error})
        messages.append({"role": "user", "content": results})
    return "(stopped: max_turns reached)"


# --- An offline stand-in for the model -------------------------------------
class ScriptedModel:
    """Decides what to do from the conversation so far. Not intelligent, just deterministic."""

    def __call__(self, messages):
        results = [c for m in messages if m["role"] == "user" and isinstance(m["content"], list)
                   for c in m["content"] if c["type"] == "tool_result"]
        if len(results) == 0:
            return Response([Block("tool_use", id="t1", name="read_note", input={"name": "agents"}),
                             Block("tool_use", id="t2", name="read_note", input={"name": "nope"})],
                            "tool_use")                  # two calls at once; one will fail
        if len(results) == 2:
            words = len(results[0]["content"].split())
            return Response([Block("tool_use", id="t3", name="calculate",
                                   input={"expression": f"{words} * 23"})], "tool_use")
        words = len(results[0]["content"].split())
        return Response([Block("text", text=f"The 'agents' note has {words} "
                                            f"words, and 23x that is {results[-1]['content']}. "
                                            "(The 'nope' note doesn't exist, as the error said.)")],
                        "end_turn")


# --- The real thing --------------------------------------------------------
def claude_model():
    import anthropic
    client = anthropic.Anthropic()

    def call(messages):
        return client.messages.create(model="claude-opus-5-5", max_tokens=16000,
                                      tools=TOOL_SCHEMAS, messages=messages)
    return call


if __name__ == "__main__":
    import sys
    question = "How many words are in my 'agents' note, times 23? Also read my 'nope' note."
    model = claude_model() if "--real" in sys.argv else ScriptedModel()
    print(run_agent(model, question))
```

## Run it offline

`ScriptedModel` plays the model's part with fixed behavior, so you can watch the mechanics without an API key. It's **not** an intelligent model.

```bash
cd code/agents
python agent_loop.py
```

```text
  turn 1: read_note({'name': 'agents'}) -> An agent is a loop: the model picks a tool, we run it, and we show it the result.
  turn 1: read_note({'name': 'nope'}) -> Error: "no note called 'nope'; available: ai, chess, agents"
  turn 2: calculate({'expression': '19 * 23'}) -> 437
The 'agents' note has 19 words, and 23x that is 437. (The 'nope' note doesn't exist, as the error said.)
```

Notice three things:

- **Turn 1 has two tool calls at once.** Models can request several tools in one response. You return all results together.
- **One call failed.** The error went back as a `tool_result` with `is_error: true`, and the loop carried on.
- **The loop, not the model, enforces `max_turns`.** An agent with no turn limit can run forever and spend money.

## Run it for real (optional)

This part needs an API key, and I haven't been able to run it myself. The code follows the Anthropic Python SDK documentation, so treat it as untested.

```bash
export ANTHROPIC_API_KEY=...
python agent_loop.py --real
```

`claude_model()` builds a client and calls `client.messages.create(model="claude-opus-5-5", max_tokens=16000, tools=..., messages=...)`. Expect the model to choose its own tool calls, so the output will differ from the scripted run.

> [!note] The SDK can write the loop for you
> The Python SDK has a beta **tool runner** (`client.beta.messages.tool_runner` with the `@beta_tool` decorator) that runs this loop for you. Writing it by hand once, as above, is worth it because you'll recognize what the runner is doing.

## Experiments

1. Set `max_turns=1`. What does the function return, and why?
2. Add a `word_count` tool. Update `ScriptedModel` to use it instead of splitting the string itself.
3. Make `calculate` raise on `"2 ** 9999999"` before it hangs. What limit would you put on the exponent?
