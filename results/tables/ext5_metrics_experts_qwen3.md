**Qwen3-30B-A3B-Base (tokenizer defaults): recurrence-first expert selection under each metric (threshold half the discovery split; all quantities in the metric's own units)**

| Metric | Layer | Selected expert | Disc. active | Disc. all-case | Val. active | Val. rescue [95% CI] | Spec [95% CI] | Spec sign | vs paper (Δ) |
|---|---|---|---|---|---|---|---|---|---|
| Δ | L44 (= Δ L*) | E069 | 114/128 | +0.483 | 116/128 | +0.506 [+0.363, +0.670] | +0.458 [+0.315, +0.623] | positive | same |
| Δp | L44 (= Δ L*) | E069 | 114/128 | +0.003 | 116/128 | +0.001 [+0.000, +0.002] | +0.001 [-0.000, +0.002] | indeterminate | same |
| Δlog p | L44 (= Δ L*) | E069 | 114/128 | +0.421 | 116/128 | +0.430 [+0.301, +0.570] | +0.393 [+0.264, +0.537] | positive | same |
| rank | L44 (= Δ L*) | E069 | 114/128 | +0.577 | 116/128 | +0.616 [+0.435, +0.821] | +0.559 [+0.375, +0.765] | positive | same |
| KL | L44 (= Δ L*) | E069 | 114/128 | +0.124 | 116/128 | +0.097 [+0.071, +0.125] | +0.079 [+0.053, +0.108] | positive | same |
| Δ/drop | L44 (= Δ L*) | E069 | 109/123 | +0.084 | 109/118 | +0.093 [+0.069, +0.120] | +0.086 [+0.060, +0.114] | positive | same |
