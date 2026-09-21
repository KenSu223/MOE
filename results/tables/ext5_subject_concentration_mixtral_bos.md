**Mixtral-8x7B-v0.1 (BOS, tokenizer default): is the subject-site layer effect carried by one expert, and by the same one across prompts? (validation cases)**

| Layer | Coalition (clean top-k) [CI] | Σ single-expert rescues [CI] | r(Σ, coalition) | Best single expert per case [CI] | Best single / coalition [CI] | Mean single | Singles > 0 | Experts needed for 50% of the coalition (median) | Distinct per-case best experts |
|---|---|---|---|---|---|---|---|---|---|
| L3 | +2.276 [+1.853, +2.709] | +2.232 [+1.826, +2.649] | 0.94 | +2.132 [+1.753, +2.540] | 0.94 [0.87, 1.00] | +1.116 | 65% | 1 (75% of cases: 1) | 8 (most common E000 in 24/128) |
| L4 | +2.041 [+1.620, +2.478] | +2.045 [+1.613, +2.498] | 0.94 | +1.956 [+1.572, +2.352] | 0.96 [0.90, 1.02] | +1.022 | 65% | 1 (74% of cases: 1) | 8 (most common E000 in 20/128) |
| L6 | +1.105 [+0.886, +1.347] | +0.933 [+0.730, +1.162] | 0.91 | +0.804 [+0.640, +0.992] | 0.73 [0.66, 0.80] | +0.467 | 70% | 1 (62% of cases: 1) | 8 (most common E006 in 33/128) |
| L18 | +0.055 [+0.034, +0.078] | +0.056 [+0.028, +0.086] | 0.81 | +0.064 [+0.048, +0.083] | n/a (no layer effect) | +0.028 | 32% | 1 (34% of cases: 1) | 6 (most common E002 in 76/128) |
| L19 | +0.073 [+0.050, +0.099] | +0.077 [+0.046, +0.108] | 0.84 | +0.072 [+0.053, +0.091] | n/a (no layer effect) | +0.039 | 36% | 1 (40% of cases: 1) | 7 (most common E000 in 66/128) |
