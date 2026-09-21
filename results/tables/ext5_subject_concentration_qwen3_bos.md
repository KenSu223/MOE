**Qwen3-30B-A3B-Base (tokenizer defaults): is the subject-site layer effect carried by one expert, and by the same one across prompts? (validation cases)**

| Layer | Coalition (clean top-k) [CI] | Σ single-expert rescues [CI] | r(Σ, coalition) | Best single expert per case [CI] | Best single / coalition [CI] | Mean single | Singles > 0 | Experts needed for 50% of the coalition (median) | Distinct per-case best experts |
|---|---|---|---|---|---|---|---|---|---|
| L2 | +0.585 [+0.377, +0.816] | +0.265 [-0.041, +0.576] | 0.51 | +0.549 [+0.427, +0.687] | 0.94 [0.72, 1.32] | +0.033 | 41% | 1 (49% of cases: 1) | 37 (most common E097 in 19/128) |
| L4 | +0.750 [+0.488, +1.050] | +0.387 [+0.119, +0.666] | 0.70 | +0.658 [+0.483, +0.865] | 0.88 [0.70, 1.13] | +0.048 | 39% | 1 (48% of cases: 1) | 38 (most common E046 in 16/128) |
| L5 | +0.723 [+0.505, +0.955] | +0.396 [+0.153, +0.649] | 0.67 | +0.654 [+0.489, +0.842] | 0.90 [0.76, 1.09] | +0.049 | 36% | 1 (55% of cases: 1) | 36 (most common E031 in 11/128) |
| L42 | +0.014 [+0.001, +0.027] | -0.039 [-0.111, +0.029] | 0.60 | +0.047 [+0.035, +0.059] | n/a (no layer effect) | -0.005 | 16% | 1 (23% of cases: 1) | 21 (most common E014 in 31/128) |
| L44 | -0.000 [-0.012, +0.011] | -0.032 [-0.103, +0.040] | 0.74 | +0.038 [+0.028, +0.048] | n/a (no layer effect) | -0.004 | 14% | 1 (17% of cases: 1) | 23 (most common E006 in 38/128) |
