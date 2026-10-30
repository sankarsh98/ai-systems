---
title: "3. Quantization"
---

## Fewer bits per number

Weights are usually stored as 16-bit floats. **Quantization** stores them as small integers plus a **scale** factor:

```text
q = round(w / scale)        (an integer, e.g. -127 to 127 for 8 bits)
w ≈ q × scale               (what the model uses when computing)
```

| Format | Bytes per weight | 7B model weights |
|---|---|---|
| 16-bit | 2 | 14 GB |
| 8-bit | 1 | 7 GB |
| 4-bit | 0.5 | 3.5 GB |

Remember lesson 1: decode speed is limited by bytes moved. Smaller weights mean faster decoding, and they let the model fit on smaller hardware. (Scale factors add a little overhead that this table ignores.)

## The catch: error

Rounding loses information. The questions are *how much* and *where*. This demo quantizes a random weight matrix and measures the error in the matrix-vector product, which is what a layer computes.

```python title="code/inference/quantize.py"
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
```

```bash
python quantize.py
```

```text
Well-behaved weights (Gaussian)
  fp16 size: 2.1 MB
  int8 per-tensor :   1.0 MB  relative output error 0.0109
  int8 per-channel:   1.0 MB  relative output error 0.0079
  int4 per-tensor :   0.5 MB  relative output error 0.2025
  int4 per-channel:   0.5 MB  relative output error 0.1350

Same matrix, but one row is 40x larger (an 'outlier')
  fp16 size: 2.1 MB
  int8 per-tensor :   1.0 MB  relative output error 0.1740
  int8 per-channel:   1.0 MB  relative output error 0.0063
  int4 per-tensor :   0.5 MB  relative output error 0.5605
  int4 per-channel:   0.5 MB  relative output error 0.0895
```

## What the numbers teach

1. **8-bit is cheap, 4-bit is expensive.** Going from 8 to 4 bits multiplied the error by roughly 17-19 on the well-behaved matrix (0.011 to 0.20, 0.008 to 0.135). Only 16 levels remain, versus 256.
2. **One outlier ruins a shared scale.** With a single scale for the whole matrix (**per-tensor**), the one huge row forces a large step size, and all the normal rows get rounded coarsely. Error jumped from 0.011 to 0.174 at 8 bits.
3. **Per-channel scales fix that.** Giving each row its own scale confines the damage: 0.0063 even with the outlier, at the same memory cost.

Real LLMs do have outlier channels in their activations and weights, which is a main reason practical 4-bit methods use fine-grained scales (per group of weights) and careful handling of outliers. Common ones include GPTQ and AWQ. This toy shows *why* those details matter. It doesn't reproduce those methods.

> [!warning] What this demo doesn't show
> The error here is on random numbers. Real quality loss is measured on model outputs, such as perplexity and task accuracy, and can differ a lot. Always evaluate a quantized model on *your* task before trusting it.

## Other targets for quantization

- **Weights** (this lesson), the usual starting point.
- **Activations**, harder because they vary per input.
- **KV cache**, which attacks the memory cost from lesson 2.

## Experiments

1. Add a group-wise scheme: one scale per 64 weights along each row. How does the error compare with per-channel at 4 bits?
2. Make 5 outlier rows instead of 1. Which method degrades fastest?
3. Try 3-bit and 2-bit. At what point does the output become useless? (Use the relative error as your guide.)
