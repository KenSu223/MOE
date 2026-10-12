**vt_a2e_orig W5: top six and bottom two heads (final position; layers = top-4 attention layers + one null layer)**

| Model | Head | Val rescue [95% CI] | Disc. mean | z (val / disc) | ≥ 2 SD both | Share of attn_layer |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L42H10 | +1.127 [+1.030, +1.221] | +0.793 | +7.6 / +6.9 | yes | 0.90 |
| Qwen3-30B-A3B-Base | L43H24 | +0.539 [+0.478, +0.600] | +0.336 | +3.5 / +2.8 | yes | 0.43 |
| Qwen3-30B-A3B-Base | L39H27 | +0.510 [+0.451, +0.568] | +0.226 | +3.3 / +1.8 | no | 0.37 |
| Qwen3-30B-A3B-Base | L39H29 | +0.490 [+0.439, +0.541] | +0.602 | +3.2 / +5.2 | yes | 0.36 |
| Qwen3-30B-A3B-Base | L39H31 | +0.323 [+0.282, +0.363] | +0.222 | +2.0 / +1.7 | no | 0.24 |
| Qwen3-30B-A3B-Base | L47H26 | +0.307 [+0.275, +0.338] | +0.231 | +1.9 / +1.8 | no | 5.52 |
| Qwen3-30B-A3B-Base | L47H29 | -0.353 [-0.385, -0.323] | -0.326 | -2.7 / -3.2 | yes | -6.33 |
| Qwen3-30B-A3B-Base | L39H28 | -0.405 [-0.476, -0.324] | -0.389 | -3.0 / -3.7 | yes | -0.30 |
