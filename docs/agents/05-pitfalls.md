---
title: "5. Pitfalls and good habits"
---

## Failure modes you'll meet

| Problem | Symptom | Fix |
|---|---|---|
| **Runaway loop** | Token bill climbs, no answer | Cap turns, budget, and wall-clock time |
| **Context bloat** | Quality drops over a long run | Return less from tools, summarize or trim old results |
| **Wrong tool chosen** | Model calls a similar-looking tool | Fewer, clearer, non-overlapping tools |
| **Silent failure** | Loop "succeeds" but the work is wrong | Check results in code (tests, schema validation), not by asking the model |
| **Unsafe action** | A side effect nobody approved | Permission gates, least privilege, audit logs |
| **Prompt injection** | Model obeys text found in tool output | Isolate untrusted content from powerful tools |

## Habits that pay off

1. **Start with the simplest thing that works.** One call, then a workflow, then an agent. Add loops only when needed.
2. **Log every turn.** Print the tool name, input and output as `run_agent` does. When something breaks, the log tells you where.
3. **Keep a small eval.** Ten tasks with known good outcomes, run after every change. Without it, you can't tell whether an edit helped.
4. **Verify with code, not vibes.** If the task has a checkable result (tests pass, JSON validates, number matches), check it.
5. **Prefer reversible actions.** Drafts instead of sends. Branches instead of main.

## Reading the scripted demo critically

The offline model in lesson 2 is a stand-in. Real models:

- may call tools in a different order, or call extra ones,
- may misread an error, and
- may need an instruction like "if a tool fails, explain what failed" in the system prompt.

None of that appears in the scripted run, which is why the optional real run is worth doing when you can.

## Checkpoint

You can now:

- explain the tool-use loop and its stop conditions,
- write tools whose errors help the model recover,
- expose tools over MCP, and
- name the main safety risk (prompt injection) and the architecture-level defenses.

Next is Track B, which asks how you run the model behind all this quickly and cheaply.
