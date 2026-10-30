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
