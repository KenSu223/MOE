**Copy prefix 'Options: t1, t2.' + newline (current; coordinator decision 2026-10-11): pass rates per attribute order (cluster bootstrap over seeds / twins)**

| Model | Build | Subset | Pairs / items | Margin ≥ 1 both ways | Correct in both twins | Mean drop | Items passing both orders |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | cp_vt | neutral, order 1 | 2640 | 0.19 [0.11, 0.27] | 0.41 [0.30, 0.52] | 3.06 [2.48, 3.63] |  |
| Qwen3-30B-A3B-Base | cp_vt | neutral, order 2 | 2640 | 0.25 [0.15, 0.35] | 0.52 [0.41, 0.63] | 3.17 [2.55, 3.77] |  |
| Qwen3-30B-A3B-Base | cp_vt | orig, order 1 | 2640 | 0.26 [0.16, 0.37] | 0.45 [0.34, 0.55] | 3.84 [3.26, 4.42] |  |
| Qwen3-30B-A3B-Base | cp_vt | orig, order 2 | 2640 | 0.35 [0.25, 0.46] | 0.62 [0.51, 0.72] | 4.05 [3.49, 4.58] |  |
| Qwen3-30B-A3B-Base | cp_vt | unrelated, order 1 | 2640 | 0.02 [0.00, 0.06] | 0.06 [0.02, 0.11] | 0.09 [-0.23, 0.41] |  |
| Qwen3-30B-A3B-Base | cp_vt | unrelated, order 2 | 2640 | 0.03 [0.00, 0.08] | 0.11 [0.05, 0.17] | 0.32 [-0.04, 0.70] |  |
| Qwen3-30B-A3B-Base | cp_vt | orig, both orders pass | 2640 | 0.233 of items |  |  | 615 |
| Qwen3-30B-A3B-Base | cp_vt | neutral, both orders pass | 2640 | 0.133 of items |  |  | 350 |
| Qwen3-30B-A3B-Base | cp_vt | unrelated, both orders pass | 2640 | 0.017 of items |  |  | 45 |
| Qwen3-30B-A3B-Base | cp_nat | order 1 | 776 | 0.73 [0.70, 0.77] | 0.92 [0.90, 0.94] | 5.40 [5.24, 5.56] |  |
| Qwen3-30B-A3B-Base | cp_nat | order 2 | 776 | 0.74 [0.71, 0.77] | 0.93 [0.91, 0.95] | 5.39 [5.24, 5.54] |  |
| Qwen3-30B-A3B-Base | cp_nat | both orders pass | 776 | 0.598 of twins |  |  | 464 |
| Mixtral-8x7B, BOS | cp_vt | neutral, order 1 | 2640 | 0.24 [0.14, 0.34] | 0.43 [0.32, 0.54] | 3.16 [2.67, 3.66] |  |
| Mixtral-8x7B, BOS | cp_vt | neutral, order 2 | 2640 | 0.14 [0.06, 0.22] | 0.43 [0.32, 0.54] | 2.69 [2.19, 3.19] |  |
| Mixtral-8x7B, BOS | cp_vt | orig, order 1 | 2640 | 0.33 [0.23, 0.45] | 0.52 [0.41, 0.63] | 3.75 [3.22, 4.30] |  |
| Mixtral-8x7B, BOS | cp_vt | orig, order 2 | 2640 | 0.25 [0.16, 0.34] | 0.45 [0.33, 0.57] | 3.23 [2.71, 3.78] |  |
| Mixtral-8x7B, BOS | cp_vt | unrelated, order 1 | 2640 | 0.02 [0.00, 0.05] | 0.10 [0.04, 0.18] | 0.57 [0.16, 0.94] |  |
| Mixtral-8x7B, BOS | cp_vt | unrelated, order 2 | 2640 | 0.02 [0.00, 0.06] | 0.05 [0.01, 0.10] | -0.11 [-0.63, 0.37] |  |
| Mixtral-8x7B, BOS | cp_vt | orig, both orders pass | 2640 | 0.165 of items |  |  | 436 |
| Mixtral-8x7B, BOS | cp_vt | neutral, both orders pass | 2640 | 0.086 of items |  |  | 228 |
| Mixtral-8x7B, BOS | cp_vt | unrelated, both orders pass | 2640 | 0.009 of items |  |  | 23 |
| Mixtral-8x7B, BOS | cp_nat | order 1 | 776 | 0.90 [0.88, 0.92] | 0.98 [0.97, 0.99] | 6.91 [6.75, 7.06] |  |
| Mixtral-8x7B, BOS | cp_nat | order 2 | 776 | 0.89 [0.87, 0.91] | 0.97 [0.96, 0.98] | 6.74 [6.58, 6.91] |  |
| Mixtral-8x7B, BOS | cp_nat | both orders pass | 776 | 0.841 of twins |  |  | 653 |
