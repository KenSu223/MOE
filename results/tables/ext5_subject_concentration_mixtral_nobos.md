**Mixtral-8x7B-v0.1 (no BOS, paper protocol): is the subject-site layer effect carried by one expert, and by the same one across prompts? (validation cases)**

| Layer | Coalition (clean top-k) [CI] | Σ single-expert rescues [CI] | r(Σ, coalition) | Best single expert per case [CI] | Best single / coalition [CI] | Mean single | Singles > 0 | Experts needed for 50% of the coalition (median) | Distinct per-case best experts |
|---|---|---|---|---|---|---|---|---|---|
| L1 | +0.643 [+0.423, +0.879] | +0.649 [+0.416, +0.898] | 0.90 | +0.634 [+0.448, +0.831] | 0.99 [0.86, 1.16] | +0.324 | 56% | 1 (59% of cases: 1) | 8 (most common E000 in 37/128) |
| L4 | +1.275 [+0.959, +1.607] | +1.259 [+0.894, +1.641] | 0.94 | +1.222 [+0.929, +1.528] | 0.96 [0.89, 1.03] | +0.630 | 60% | 1 (67% of cases: 1) | 8 (most common E000 in 26/128) |
| L6 | +1.102 [+0.854, +1.362] | +0.850 [+0.613, +1.104] | 0.90 | +0.784 [+0.590, +1.001] | 0.71 [0.63, 0.80] | +0.425 | 63% | 1 (53% of cases: 1) | 8 (most common E006 in 35/128) |
| L18 | +0.067 [+0.028, +0.113] | +0.048 [-0.001, +0.104] | 0.91 | +0.068 [+0.035, +0.109] | n/a (no layer effect) | +0.024 | 31% | 1 (30% of cases: 1) | 6 (most common E002 in 73/128) |
| L19 | +0.082 [+0.050, +0.116] | +0.069 [+0.030, +0.110] | 0.87 | +0.069 [+0.045, +0.095] | n/a (no layer effect) | +0.035 | 38% | 1 (41% of cases: 1) | 7 (most common E000 in 54/128) |
