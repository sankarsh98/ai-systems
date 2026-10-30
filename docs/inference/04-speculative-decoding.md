---
title: "4. Speculative decoding"
---

## The idea

Decode is slow because each token costs a full pass over the weights. But **checking** several tokens costs about the same as generating one, because the pass is limited by bytes moved, not by math.

So use two models:

- a small, fast **draft** model that guesses the next few tokens, and
- the big **target** model that verifies all of them in a single pass.

```text
draft guesses:   the  cat  sat  on   a
target checks:   ✓    ✓    ✓    ✗ (it wanted "the")
keep:            the  cat  sat  on→"the"       ← 4 tokens for one target pass
```

When the draft is right, you get several tokens per pass over the big model. When it's wrong, you lose little.

## It doesn't change the answer

The surprising part: with the right accept/reject rule, the output follows **exactly the same distribution** as the target model alone. It's a speedup, not an approximation.

For a drafted token `t` at some position, with target probability `p(t)` and draft probability `q(t)`:

1. Accept it with probability `min(1, p(t) / q(t))`.
2. If it's rejected, sample a replacement from the **leftover** mass `max(p - q, 0)`, renormalized, and stop.
3. If every draft is accepted, take one bonus token from the target.

```python title="code/inference/speculative.py"
"""Speculative decoding on a toy vocabulary.

A cheap draft model q proposes tokens; the expensive target model p checks them.
The accept/reject rule below guarantees the output follows p exactly.
"""
import numpy as np

rng = np.random.default_rng(1)
V = 6


def normalize(a):
    return a / a.sum()


# Toy "models": next-token distributions that depend on the previous token.
P = np.array([normalize(rng.random(V) ** 3) for _ in range(V)])        # target
Q = np.array([normalize(0.6 * p + 0.4 * normalize(rng.random(V))) for p in P])  # draft: similar, not equal


def sample(dist):
    return int(rng.choice(V, p=dist))


def speculative_step(prev, gamma):
    """Draft up to gamma tokens, verify with the target. Returns (tokens, accepted_count)."""
    drafts, state = [], prev
    for _ in range(gamma):
        t = sample(Q[state]); drafts.append((state, t)); state = t
    out = []
    for state, t in drafts:
        if rng.random() < min(1.0, P[state][t] / Q[state][t]):
            out.append(t)                                        # accepted
        else:
            residual = np.maximum(P[state] - Q[state], 0)        # resample from the leftover mass
            out.append(sample(normalize(residual)))
            return out, len(out) - 1
    out.append(sample(P[out[-1]]))                               # bonus token from the target
    return out, gamma


if __name__ == "__main__":
    # 1. Correctness: first generated token's distribution must match the target.
    prev, N = 0, 200_000
    counts = np.zeros(V)
    for _ in range(N):
        counts[speculative_step(prev, gamma=4)[0][0]] += 1
    print("token        :", *range(V))
    print("target p     :", *[f"{x:.3f}" for x in P[prev]])
    print("speculative  :", *[f"{x:.3f}" for x in counts / N])
    print(f"max abs diff : {np.abs(counts / N - P[prev]).max():.4f}\n")

    # 2. Speed: tokens produced per target-model pass.
    print("gamma  tokens per target pass")
    for gamma in (1, 2, 4, 8):
        total_tokens, passes, prev = 0, 0, 0
        for _ in range(20_000):
            toks, _ = speculative_step(prev, gamma)
            total_tokens += len(toks); passes += 1; prev = toks[-1]
        print(f"{gamma:>5}  {total_tokens / passes:.2f}")
    print("\n(Plain decoding always gives 1.00 token per target pass.)")
```

```bash
python speculative.py
```

```text
token        : 0 1 2 3 4 5
target p     : 0.069 0.439 0.002 0.437 0.016 0.039
speculative  : 0.069 0.439 0.001 0.436 0.016 0.039
max abs diff : 0.0007

gamma  tokens per target pass
    1  1.81
    2  2.46
    4  3.40
    8  4.42

(Plain decoding always gives 1.00 token per target pass.)
```

## Reading the output

**Correctness check.** Over 200,000 trials, the first token produced by the speculative procedure matches the target's distribution to within 0.0007. That's what the sampling error looks like, not a bias.

**Speed.** `gamma` is how many tokens the draft proposes. More drafts mean more tokens per target pass, with diminishing returns: going from 4 to 8 drafts adds about one token, not four.

> [!warning] The table leaves out the draft's cost
> "Tokens per target pass" counts only the big model's passes. Running the draft isn't free, so real speedups are smaller. A draft that's cheap *and* agrees often with the target is what makes this work, and that's why draft models are usually small models from the same family or trained to imitate the target.

## Why this toy agrees so well

I built the draft as a blend of the target's distribution and noise (60% target, 40% noise), so it's *designed* to agree often. Real acceptance rates depend on the task. Predictable text such as code and boilerplate is accepted more often than creative writing.

## Experiments

1. Make the draft worse: change `0.6 * p + 0.4 * ...` to `0.1 * p + 0.9 * ...`. How do the tokens per pass change?
2. Add a draft cost `c` (the draft takes `c` times as long as a target pass per token). Find the best `gamma` for `c = 0.05` and `c = 0.3`.
3. Make the draft *identical* to the target. What should tokens per pass become for `gamma = 4`? Check it.

## Checkpoint for Track B

You can now explain:

- why decode is memory-bound,
- what the KV cache saves and what it costs,
- how quantization trades bits for error, and why outliers matter, and
- how speculative decoding gets several tokens per big-model pass without changing the output.
