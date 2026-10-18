**l_ant W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| qwen3_14b | main | layer | L33 | +0.91 | +0.804 [+0.699, +0.918] | 0.097 [0.084, 0.111] | L33 +0.91, L29 +0.85, L34 +0.74, L30 +0.58 |
| qwen3_14b | main | attn_layer | L29 | +4.11 | +3.596 [+3.337, +3.862] | 0.434 [0.402, 0.468] | L29 +4.11, L32 +1.78, L33 +0.68, L37 +0.63 |
| qwen3_14b | main | block | L29 | +4.72 | +4.213 [+3.982, +4.455] | 0.509 [0.477, 0.542] | L29 +4.72, L32 +2.33, L33 +1.55, L37 +0.79 |
| qwen3_14b | main_interior | layer | L33 | +0.91 | +0.804 [+0.699, +0.918] | 0.097 [0.084, 0.111] | L33 +0.91, L29 +0.85, L34 +0.74, L30 +0.58 |
| qwen3_14b | main_interior | attn_layer | L29 | +4.11 | +3.596 [+3.337, +3.862] | 0.434 [0.402, 0.468] | L29 +4.11, L32 +1.78, L33 +0.68, L25 +0.47 |
| qwen3_14b | main_interior | block | L29 | +4.72 | +4.213 [+3.982, +4.455] | 0.509 [0.477, 0.542] | L29 +4.72, L32 +2.33, L33 +1.55, L34 +0.77 |
