---
title: "1. What is an agent?"
---

A plain chatbot takes text in and gives text out. It can *describe* how to check the weather, but it can't check it.

An **agent** is a model placed inside a **loop** with **tools**:

```text
        ┌──────────────────────────────────────────┐
        │                                          │
 user → │  model  →  "call tool X with args"  →  your code runs X
        │    ↑                                      │
        │    └────────  result of X  ───────────────┘
        └──────────────────────────────────────────┘
                 repeat until the model answers
```

The model never runs anything itself. It only **asks**. Your program decides whether to run the request, runs it, and shows the model what happened.

## Workflows versus agents

Anthropic's engineering post [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) draws a useful line:

- **Workflow:** your code fixes the steps. "Summarize, then translate, then email." The model fills in each step.
- **Agent:** the model chooses the steps. You give it tools and a goal, and it decides what to do next.

Workflows are cheaper, faster and more predictable. Use an agent when you *can't* write the steps in advance.

> [!tip] A good test
> Could you write the steps as a flowchart before seeing the input? If yes, build a workflow. If the steps depend on what the model discovers along the way, you may need an agent.

## Why this matters for AI's future

Models are getting better at long tasks, and the thing that turns a good model into a useful product is the *surrounding system*: which tools it can use, what it sees, when a human approves an action. That system is ordinary software, and you can learn it now.

## The four pieces

| Piece | Job | Where it lives |
|---|---|---|
| **Model** | Decides the next action | Remote API |
| **Tools** | Do things: calculate, search, read files, call APIs | Your code |
| **Loop** | Runs tools, feeds results back, stops when done | Your code |
| **Context** | Everything the model can see: messages, tool results | The message list |

The next lesson builds all four in about 40 lines.
