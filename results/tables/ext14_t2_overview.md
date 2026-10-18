**Phase 6 tier 2: selected expert, add-back, mixed head + expert greedy, knockout, writer (validation; cluster bootstrap)**

| Cell | Model | Selected expert | Pattern | Spec [95% CI] | All-MoE ceiling | k80 greedy / oracle / random | random k for greedy 1·5·10·15 | Mixed r(20) | Heads in first 10 | KO F selected | KO top-10 vs random | Writer D/T selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 a→e | Qwen3-30B-A3B-Base | L37E032 | A | +0.21 [+0.16, +0.27] | -0.17 [-0.21, -0.14] | own-max k80: 4 / 192 / n/a | 1→never · 5→never · 10→never · 15→never | 0.93 [0.92, 0.95] | 9.6 | +0.065 [+0.054, +0.075] | +0.145 vs +0.008 | -0.01 [-0.03, 0.01] |
| E1 a→e | Mixtral-8x7B (BOS) | L17E007 | A | +0.21 [+0.13, +0.28] | 0.33 [0.28, 0.39] | 2 / 3 / 48 | 1→21 · 5→never · 10→never · 15→never | 1.17 [1.11, 1.25] | 7.1 | -0.008 [-0.028, +0.011] | +0.192 vs -0.004 | 0.04 [-0.00, 0.10] |
