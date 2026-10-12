**cp_nat W4 DLA: direct logit attribution of the final-position sublayer outputs (main family)**

| Model | DLA attention / drop | DLA MoE / drop | Attention share of DLA | Top attention layers | Top MoE layers |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | -0.007 [-0.037, 0.022] | 1.007 [0.978, 1.037] | -0.01 [-0.04, 0.02] | L43 +0.050, L42 +0.039, L41 +0.032 | L43 +0.257, L41 +0.178, L44 +0.153 |
| Mixtral-8x7B, BOS | 0.395 [0.374, 0.415] | 0.606 [0.585, 0.626] | 0.39 [0.37, 0.42] | L25 +0.114, L31 +0.058, L22 +0.035 | L26 +0.192, L22 +0.149, L21 +0.130 |
