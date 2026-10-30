"""Weight quantization: shrink a matrix to int8 / int4 and measure the damage."""
import numpy as np

rng = np.random.default_rng(0)


def quantize(W, bits, per_channel):
    qmax = 2 ** (bits - 1) - 1
    axis = 1 if per_channel else None          # one scale per output row, or one for all
    scale = np.abs(W).max(axis=axis, keepdims=per_channel) / qmax
    q = np.clip(np.round(W / scale), -qmax, qmax)
    return q * scale                            # dequantized, to measure error


def report(name, W, x):
    exact = W @ x
    print(name)
    print(f"  fp16 size: {W.size * 2 / 1e6:.1f} MB")
    for bits in (8, 4):
        for per_channel in (False, True):
            err = np.linalg.norm(quantize(W, bits, per_channel) @ x - exact) / np.linalg.norm(exact)
            mode = "per-channel" if per_channel else "per-tensor "
            print(f"  int{bits} {mode}: {W.size * bits / 8 / 1e6:5.1f} MB  relative output error {err:.4f}")
    print()


n = 1024
x = rng.normal(size=n)

W = rng.normal(0, 0.02, (n, n))
report("Well-behaved weights (Gaussian)", W, x)

W_out = W.copy()
W_out[7] *= 40                                  # one row with much larger weights
report("Same matrix, but one row is 40x larger (an 'outlier')", W_out, x)
