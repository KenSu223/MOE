**e1_a2e W5: top six and bottom two heads (final position; layers = top-4 attention layers + one null layer)**

| Model | Head | Val rescue [95% CI] | Disc. mean | z (val / disc) | ≥ 2 SD both | Share of attn_layer |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L42H10 | +1.699 [+1.589, +1.816] | +1.635 | +7.5 / +7.2 | yes | 0.80 |
| Qwen3-30B-A3B-Base | L39H27 | +0.877 [+0.818, +0.939] | +0.906 | +3.8 / +3.9 | yes | 0.50 |
| Qwen3-30B-A3B-Base | L43H24 | +0.855 [+0.788, +0.925] | +0.841 | +3.7 / +3.6 | yes | 0.56 |
| Qwen3-30B-A3B-Base | L39H31 | +0.743 [+0.694, +0.793] | +0.796 | +3.2 / +3.4 | yes | 0.42 |
| Qwen3-30B-A3B-Base | L39H29 | +0.520 [+0.474, +0.569] | +0.553 | +2.2 / +2.3 | yes | 0.30 |
| Qwen3-30B-A3B-Base | L42H31 | +0.429 [+0.378, +0.481] | +0.428 | +1.8 / +1.7 | no | 0.20 |
| Qwen3-30B-A3B-Base | L42H14 | -0.656 [-0.708, -0.607] | -0.605 | -3.1 / -3.0 | yes | -0.31 |
| Qwen3-30B-A3B-Base | L39H28 | -0.759 [-0.818, -0.706] | -0.800 | -3.6 / -3.9 | yes | -0.43 |
