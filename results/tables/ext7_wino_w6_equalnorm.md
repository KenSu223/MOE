**W6: equal-norm check at the selected layer (Qwen3: gate-matched control, Table 9; Mixtral: other active expert, Table 11)**

| Model | Expert | Control | n (anchor-active val, directed) | Raw Spec vs control | Selected, equal norm | Control, equal norm | Equal-norm Spec |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | gate_matched | 223 | +1.034 [+0.866, +1.212] | +0.534 [+0.458, +0.617] | +0.084 [+0.055, +0.116] | +0.450 [+0.373, +0.534] |
| Mixtral-8x7B, BOS | L20E000 | active_pair | 246 | +0.996 [+0.900, +1.088] | +0.313 [+0.276, +0.351] | +0.152 [+0.129, +0.177] | +0.160 [+0.132, +0.189] |
