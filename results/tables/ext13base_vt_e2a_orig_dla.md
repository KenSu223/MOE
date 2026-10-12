**vt_e2a_orig W4 DLA: direct logit attribution of the final-position sublayer outputs (main family)**

| Model | DLA attention / drop | DLA MoE / drop | Attention share of DLA | Top attention layers | Top MoE layers |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 0.107 [0.093, 0.120] | 0.893 [0.881, 0.907] | 0.11 [0.09, 0.12] | L42 +0.030, L43 +0.027, L41 +0.018 | L43 +0.170, L42 +0.162, L41 +0.135 |
| Mixtral-8x7B, BOS | 0.353 [0.336, 0.372] | 0.646 [0.628, 0.664] | 0.35 [0.34, 0.37] | L19 +0.047, L25 +0.045, L31 +0.044 | L20 +0.108, L22 +0.099, L21 +0.098 |
