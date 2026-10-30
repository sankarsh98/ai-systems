---
title: "2. The KV cache"
---

## What attention needs

In a transformer, each new token **attends** to every earlier token. To do that it needs a **key** and a **value** vector for each earlier token, at every layer.

The keys and values for token 5 don't change when token 6 is generated. Recomputing them for every new token is pure waste. The fix is to **cache** them.

|  | Work per generated token |
|---|---|
| No cache | Recompute K and V for the **whole sequence so far** |
| KV cache | Compute K and V for **one** new token, append to the cache |

Over a whole answer, no cache means work that grows with the *square* of the length. With a cache it grows linearly.

## A runnable demo

A one-layer, one-head attention decoder in NumPy. It generates with and without a cache, and asserts the outputs are **identical**.

```python title="code/inference/kv_cache.py"
"""KV cache demo: a one-layer, one-head attention decoder in NumPy.

Both versions produce identical outputs. Without a cache, every new token
recomputes keys and values for the whole prefix. With a cache, we compute them
once and reuse them.
"""
import time
import numpy as np

D = 64
rng = np.random.default_rng(0)
Wq, Wk, Wv, Wo = (rng.normal(0, D**-0.5, (D, D)) for _ in range(4))
embed = rng.normal(size=(1000, D))


def attend(q, K, V):
    scores = K @ q / np.sqrt(D)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    return weights @ V


def next_token_no_cache(tokens):
    x = embed[tokens]                      # (n, D): recompute everything
    K, V = x @ Wk, x @ Wv
    out = attend(x[-1] @ Wq, K, V) @ Wo
    return int((out @ embed.T).argmax()), len(tokens)  # tokens processed this step


def generate_no_cache(prompt, steps):
    tokens, processed = list(prompt), 0
    for _ in range(steps):
        tok, n = next_token_no_cache(tokens)
        processed += n
        tokens.append(tok)
    return tokens, processed


def generate_cache(prompt, steps):
    K = np.zeros((0, D)); V = np.zeros((0, D))
    processed = 0
    x = embed[prompt]                      # prefill: process the prompt once
    K, V = x @ Wk, x @ Wv
    processed += len(prompt)
    q, tokens = x[-1] @ Wq, list(prompt)
    for _ in range(steps):
        out = attend(q, K, V) @ Wo
        tok = int((out @ embed.T).argmax())
        tokens.append(tok)
        xn = embed[tok]                    # decode: process ONE new token
        K = np.vstack([K, xn @ Wk]); V = np.vstack([V, xn @ Wv])
        q = xn @ Wq
        processed += 1
    return tokens, processed


if __name__ == "__main__":
    prompt = list(rng.integers(0, 1000, 256))
    for steps in (64, 256, 1024):
        t0 = time.perf_counter(); a, pa = generate_no_cache(prompt, steps); t1 = time.perf_counter()
        b, pb = generate_cache(prompt, steps); t2 = time.perf_counter()
        assert a == b, "outputs differ"
        print(f"{steps:>5} new tokens | no cache: {t1-t0:6.3f}s, {pa:>9,} token-passes"
              f" | cache: {t2-t1:6.3f}s, {pb:>6,} token-passes | speedup {((t1-t0)/(t2-t1)):4.1f}x")
```

```bash
cd code/inference
python kv_cache.py
```

```text
   64 new tokens | no cache:  0.008s,    18,400 token-passes | cache:  0.002s,    320 token-passes | speedup  3.3x
  256 new tokens | no cache:  0.029s,    98,176 token-passes | cache:  0.011s,    512 token-passes | speedup  2.7x
 1024 new tokens | no cache:  0.386s,   785,920 token-passes | cache:  0.121s,  1,280 token-passes | speedup  3.2x
```

Look at the **token-passes** column, which counts how many tokens had K and V recomputed: 785,920 against 1,280, a ratio of about 600. The wall-clock speedup is only about 3x because this toy's time is dominated by the final projection onto a 1,000-word vocabulary, which the cache doesn't help. In a real model, the layers dominate, and the cache's benefit grows with sequence length.

Your timings will differ. The token-pass counts won't.

## The price: memory

The cache isn't free. It stores 2 vectors (K and V) per token, per layer, per KV head:

```text
bytes per token = 2 × layers × kv_heads × head_dim × bytes_per_value
```

```python title="code/inference/memory_math.py"
"""How big is the KV cache? Pure arithmetic, no model needed."""

def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value=2):
    # 2 = one key vector and one value vector per head, per layer
    return 2 * layers * kv_heads * head_dim * bytes_per_value


def gib(n):
    return n / 2**30


configs = {
    "7B-class, full multi-head (32 KV heads)": dict(layers=32, kv_heads=32, head_dim=128),
    "same, grouped-query attention (8 KV heads)": dict(layers=32, kv_heads=8, head_dim=128),
    "same, GQA + 8-bit cache": dict(layers=32, kv_heads=8, head_dim=128, bytes_per_value=1),
}

for name, cfg in configs.items():
    per_token = kv_bytes_per_token(**cfg)
    print(f"{name}")
    print(f"  {per_token / 1024:.0f} KiB per token")
    for ctx in (4_096, 32_768, 131_072):
        print(f"  context {ctx:>7,}: {gib(per_token * ctx):6.2f} GiB per sequence")
    print()
```

```text
7B-class, full multi-head (32 KV heads)
  512 KiB per token
  context 131,072:  64.00 GiB per sequence

same, grouped-query attention (8 KV heads)
  128 KiB per token
  context 131,072:  16.00 GiB per sequence

same, GQA + 8-bit cache
  64 KiB per token
  context 131,072:   8.00 GiB per sequence
```

(Only the 128K-token rows are shown here. Run the script to see 4K and 32K.)

At long context, the cache for **one** conversation can be larger than the model's weights, and a server holds many conversations at once. That's why long-context serving is expensive, and why three ideas show up everywhere:

- **Grouped-query attention (GQA):** several query heads share one KV head, so the cache shrinks 4x here with little quality loss in practice.
- **Cache quantization:** store K and V in 8-bit or lower.
- **Smarter memory management:** paged allocation, so cache memory isn't wasted on unused space.

The example configuration (32 layers, head dimension 128) is typical of 7B-class models, not a specific product.

## Experiments

1. Change `D` in `kv_cache.py` to 16. Does the speedup change? Why?
2. Add a vocabulary size of 100,000 and watch the wall-clock speedup shrink. What does that tell you about where the time goes?
3. In `memory_math.py`, compute how many 32K-token conversations fit in 80 GiB after the 14 GB of weights, for each configuration.
