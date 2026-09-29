**Sliding-window (joint) patch vs the sum of single-layer patches over the same 5-layer window (Zhang & Nanda Section 5: 1.40-1.75x in GPT-2 XL)**

| Model | Metric | Token group | Sliding window 5 peak | Sum of single layers over the window, peak | Sliding / adding |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | LD / drop | first subject token | +0.0962 (L0) | +0.0863 (L1) | 1.12 |
| Qwen3-30B-A3B-Base | LD / drop | middle subject tokens | +0.3415 (L2) | +0.3711 (L2) | 0.92 |
| Qwen3-30B-A3B-Base | LD / drop | last subject token | +0.5922 (L2) | +0.7167 (L2) | 0.83 |
| Qwen3-30B-A3B-Base | LD / drop | first subsequent token | +0.0108 (L26) | +0.0080 (L40) | 1.34 |
| Qwen3-30B-A3B-Base | LD / drop | further tokens | +0.0093 (L28) | +0.0100 (L17) | 0.94 |
| Qwen3-30B-A3B-Base | LD / drop | last token | +0.4749 (L42) | +0.4892 (L42) | 0.97 |
| Qwen3-30B-A3B-Base | Δp | first subject token | +0.0071 (L1) | +0.0063 (L1) | 1.14 |
| Qwen3-30B-A3B-Base | Δp | middle subject tokens | +0.0192 (L1) | +0.0062 (L2) | 3.10 |
| Qwen3-30B-A3B-Base | Δp | last subject token | +0.0715 (L2) | +0.0193 (L3) | 3.71 |
| Qwen3-30B-A3B-Base | Δp | first subsequent token | +0.0001 (L26) | +0.0001 (L26) | 0.80 |
| Qwen3-30B-A3B-Base | Δp | further tokens | +0.0002 (L32) | +0.0003 (L30) | 0.73 |
| Qwen3-30B-A3B-Base | Δp | last token | +0.0484 (L42) | +0.0137 (L42) | 3.53 |
| Mixtral-8x7B, BOS | LD / drop | first subject token | +0.2109 (L2) | +0.2302 (L2) | 0.92 |
| Mixtral-8x7B, BOS | LD / drop | middle subject tokens | +0.2387 (L2) | +0.2835 (L2) | 0.84 |
| Mixtral-8x7B, BOS | LD / drop | last subject token | +0.6635 (L4) | +1.0968 (L2) | 0.60 |
| Mixtral-8x7B, BOS | LD / drop | first subsequent token | +0.0253 (L20) | +0.0186 (L20) | 1.36 |
| Mixtral-8x7B, BOS | LD / drop | further tokens | +0.0262 (L19) | +0.0245 (L19) | 1.07 |
| Mixtral-8x7B, BOS | LD / drop | last token | +0.3831 (L20) | +0.3502 (L21) | 1.09 |
| Mixtral-8x7B, BOS | Δp | first subject token | +0.0138 (L2) | +0.0168 (L2) | 0.82 |
| Mixtral-8x7B, BOS | Δp | middle subject tokens | +0.0094 (L2) | +0.0057 (L2) | 1.64 |
| Mixtral-8x7B, BOS | Δp | last subject token | +0.1188 (L4) | +0.0430 (L2) | 2.76 |
| Mixtral-8x7B, BOS | Δp | first subsequent token | +0.0002 (L20) | +0.0001 (L20) | 2.13 |
| Mixtral-8x7B, BOS | Δp | further tokens | +0.0004 (L19) | +0.0003 (L19) | 1.40 |
| Mixtral-8x7B, BOS | Δp | last token | +0.0572 (L20) | +0.0094 (L20) | 6.08 |
