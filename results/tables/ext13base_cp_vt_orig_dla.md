**cp_vt_orig W4 DLA: direct logit attribution of the final-position sublayer outputs (main family)**

| Model | DLA attention / drop | DLA MoE / drop | Attention share of DLA | Top attention layers | Top MoE layers |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 0.106 [0.067, 0.146] | 0.894 [0.853, 0.933] | 0.11 [0.07, 0.15] | L42 +0.062, L43 +0.056, L39 +0.037 | L42 +0.259, L43 +0.221, L41 +0.158 |
| Mixtral-8x7B, BOS | 0.516 [0.487, 0.546] | 0.483 [0.453, 0.512] | 0.52 [0.49, 0.55] | L19 +0.086, L31 +0.068, L29 +0.053 | L22 +0.161, L20 +0.130, L23 +0.120 |
