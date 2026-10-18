**e1p_a2e W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | layer | L37 | +0.55 | +0.477 [+0.405, +0.551] | 0.082 [0.072, 0.092] | L37 +0.55, L43 +0.31, L42 +0.30, L44 +0.29 |
| Qwen3-30B-A3B-Base | main | attn_layer | L42 | +1.91 | +1.797 [+1.686, +1.904] | 0.310 [0.299, 0.322] | L42 +1.92, L39 +1.62, L43 +1.41, L45 +1.17 |
| Qwen3-30B-A3B-Base | main | block | L42 | +2.11 | +1.937 [+1.831, +2.041] | 0.334 [0.324, 0.345] | L42 +2.11, L43 +1.69, L39 +1.68, L37 +1.03 |
| Qwen3-30B-A3B-Base | main_interior | layer | L37 | +0.55 | +0.477 [+0.405, +0.551] | 0.082 [0.072, 0.092] | L37 +0.55, L43 +0.31, L42 +0.30, L41 +0.24 |
| Qwen3-30B-A3B-Base | main_interior | attn_layer | L42 | +1.91 | +1.797 [+1.686, +1.904] | 0.310 [0.299, 0.322] | L42 +1.92, L39 +1.62, L43 +1.41, L35 +1.01 |
| Qwen3-30B-A3B-Base | main_interior | block | L42 | +2.11 | +1.937 [+1.831, +2.041] | 0.334 [0.324, 0.345] | L42 +2.11, L43 +1.69, L39 +1.68, L37 +1.03 |
