**VOID copy prefix 'Of X and Y, one <frame> t1 and the other <frame> t2.' (behaviour record): twin A's answer is option 1 (X) in every item, so the order with t1 = A's attribute is 'respectively'-congruent (X ↔ t1 answers both twins without the sentence); cluster bootstrap over seeds / twins**

| Model | Build | Subset | Pairs | Margin ≥ 1 both ways | Correct in both twins | Mean drop |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | cp_vt | neutral, incongruent | 2256 | 0.07 [0.03, 0.11] | 0.30 [0.20, 0.41] | 1.04 [0.50, 1.55] |
| Qwen3-30B-A3B-Base | cp_vt | neutral, congruent (X ↔ t1) | 2256 | 0.37 [0.26, 0.48] | 0.68 [0.58, 0.78] | 3.48 [2.95, 4.00] |
| Qwen3-30B-A3B-Base | cp_vt | orig, incongruent | 2256 | 0.10 [0.05, 0.16] | 0.37 [0.26, 0.47] | 1.54 [0.96, 2.12] |
| Qwen3-30B-A3B-Base | cp_vt | orig, congruent (X ↔ t1) | 2256 | 0.43 [0.32, 0.54] | 0.73 [0.62, 0.82] | 3.78 [3.13, 4.45] |
| Qwen3-30B-A3B-Base | cp_vt | unrelated, incongruent | 2256 | 0.00 [0.00, 0.01] | 0.03 [0.01, 0.08] | -1.17 [-1.63, -0.75] |
| Qwen3-30B-A3B-Base | cp_vt | unrelated, congruent (X ↔ t1) | 2256 | 0.07 [0.03, 0.14] | 0.37 [0.27, 0.48] | 1.21 [0.80, 1.66] |
| Qwen3-30B-A3B-Base | cp_nat | incongruent | 693 | 0.63 [0.60, 0.67] | 0.83 [0.80, 0.86] | 4.82 [4.59, 5.05] |
| Qwen3-30B-A3B-Base | cp_nat | congruent (X ↔ t1) | 693 | 0.90 [0.88, 0.92] | 0.98 [0.97, 0.99] | 6.98 [6.77, 7.17] |
| Mixtral-8x7B, BOS | cp_vt | neutral, incongruent | 2256 | 0.04 [0.01, 0.09] | 0.15 [0.07, 0.24] | -0.18 [-0.85, 0.52] |
| Mixtral-8x7B, BOS | cp_vt | neutral, congruent (X ↔ t1) | 2256 | 0.73 [0.62, 0.82] | 0.91 [0.83, 0.97] | 6.11 [5.47, 6.76] |
| Mixtral-8x7B, BOS | cp_vt | orig, incongruent | 2256 | 0.12 [0.05, 0.19] | 0.25 [0.15, 0.37] | 0.50 [-0.25, 1.25] |
| Mixtral-8x7B, BOS | cp_vt | orig, congruent (X ↔ t1) | 2256 | 0.67 [0.55, 0.77] | 0.88 [0.80, 0.95] | 6.35 [5.61, 7.12] |
| Mixtral-8x7B, BOS | cp_vt | unrelated, incongruent | 2256 | 0.00 [0.00, 0.01] | 0.02 [0.00, 0.06] | -2.87 [-3.45, -2.28] |
| Mixtral-8x7B, BOS | cp_vt | unrelated, congruent (X ↔ t1) | 2256 | 0.43 [0.31, 0.55] | 0.69 [0.57, 0.80] | 3.57 [3.04, 4.08] |
| Mixtral-8x7B, BOS | cp_nat | incongruent | 693 | 0.57 [0.53, 0.61] | 0.73 [0.70, 0.76] | 4.74 [4.42, 5.05] |
| Mixtral-8x7B, BOS | cp_nat | congruent (X ↔ t1) | 693 | 0.98 [0.97, 0.99] | 1.00 [0.99, 1.00] | 9.60 [9.40, 9.81] |
