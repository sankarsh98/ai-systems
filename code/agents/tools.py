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
