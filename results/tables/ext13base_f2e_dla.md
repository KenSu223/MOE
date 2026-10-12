**f2e W4 DLA: direct logit attribution of the final-position sublayer outputs (main family)**

| Model | DLA attention / drop | DLA MoE / drop | Attention share of DLA | Top attention layers | Top MoE layers |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 1.019 [0.933, 1.114] | -0.021 [-0.116, 0.065] | 1.02 [0.94, 1.12] | L45 +0.434, L42 +0.288, L43 +0.148 | L43 +0.138, L44 +0.129, L42 +0.064 |
| Mixtral-8x7B, BOS | 1.324 [1.219, 1.439] | -0.320 [-0.434, -0.216] | 1.32 [1.22, 1.43] | L29 +0.218, L21 +0.168, L19 +0.167 | L23 +0.113, L21 +0.083, L22 +0.081 |
