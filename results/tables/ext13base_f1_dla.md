**f1 W4 DLA: direct logit attribution of the final-position sublayer outputs (main family)**

| Model | DLA attention / drop | DLA MoE / drop | Attention share of DLA | Top attention layers | Top MoE layers |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 1.197 [1.158, 1.239] | -0.196 [-0.237, -0.156] | 1.20 [1.16, 1.24] | L45 +0.563, L42 +0.390, L43 +0.208 | L43 +0.125, L44 +0.115, L42 +0.031 |
| Mixtral-8x7B, BOS | 0.992 [0.921, 1.062] | 0.008 [-0.061, 0.079] | 0.99 [0.92, 1.06] | L21 +0.180, L19 +0.147, L29 +0.132 | L24 +0.133, L23 +0.115, L20 +0.105 |
