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
