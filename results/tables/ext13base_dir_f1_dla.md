**dir_f1 W4 DLA: direct logit attribution of the final-position sublayer outputs (main family)**

| Model | DLA attention / drop | DLA MoE / drop | Attention share of DLA | Top attention layers | Top MoE layers |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 1.047 [0.989, 1.104] | -0.047 [-0.103, 0.011] | 1.05 [0.99, 1.10] | L45 +0.409, L43 +0.217, L42 +0.204 | L44 +0.202, L43 +0.202, L42 +0.067 |
| Mixtral-8x7B, BOS | 0.832 [0.770, 0.894] | 0.169 [0.108, 0.231] | 0.83 [0.77, 0.89] | L21 +0.201, L19 +0.119, L29 +0.110 | L23 +0.163, L24 +0.150, L20 +0.130 |
