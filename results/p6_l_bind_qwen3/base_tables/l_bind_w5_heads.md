**l_bind W5: top six and bottom two heads (final position; layers = top-4 attention layers + one null layer)**

| Model | Head | Val rescue [95% CI] | Disc. mean | z (val / disc) | ≥ 2 SD both | Share of attn_layer |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L42H10 | +0.928 [+0.860, +0.994] | +0.958 | +6.4 / +6.6 | yes | 0.60 |
| Qwen3-30B-A3B-Base | L39H29 | +0.592 [+0.555, +0.630] | +0.554 | +4.0 / +3.7 | yes | 0.62 |
| Qwen3-30B-A3B-Base | L39H27 | +0.520 [+0.483, +0.555] | +0.501 | +3.5 / +3.4 | yes | 0.55 |
| Qwen3-30B-A3B-Base | L39H31 | +0.507 [+0.475, +0.541] | +0.493 | +3.4 / +3.3 | yes | 0.54 |
| Qwen3-30B-A3B-Base | L42H12 | +0.365 [+0.339, +0.391] | +0.369 | +2.4 / +2.4 | yes | 0.24 |
| Qwen3-30B-A3B-Base | L43H24 | +0.311 [+0.286, +0.337] | +0.340 | +2.0 / +2.2 | yes | 0.32 |
| Qwen3-30B-A3B-Base | L39H24 | -0.271 [-0.292, -0.251] | -0.301 | -2.1 / -2.3 | yes | -0.29 |
| Qwen3-30B-A3B-Base | L39H28 | -0.638 [-0.672, -0.605] | -0.668 | -4.8 / -4.9 | yes | -0.67 |
