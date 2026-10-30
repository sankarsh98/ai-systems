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
