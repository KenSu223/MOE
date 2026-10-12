**vt_e2a_neu W4 DLA: direct logit attribution of the final-position sublayer outputs (main family)**

| Model | DLA attention / drop | DLA MoE / drop | Attention share of DLA | Top attention layers | Top MoE layers |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 0.124 [0.109, 0.139] | 0.877 [0.862, 0.892] | 0.12 [0.11, 0.14] | L42 +0.034, L43 +0.027, L41 +0.018 | L43 +0.175, L42 +0.155, L41 +0.118 |
| Mixtral-8x7B, BOS | 0.392 [0.374, 0.412] | 0.608 [0.588, 0.627] | 0.39 [0.37, 0.41] | L25 +0.054, L19 +0.044, L31 +0.039 | L27 +0.115, L20 +0.111, L22 +0.106 |
