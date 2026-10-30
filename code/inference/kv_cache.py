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
