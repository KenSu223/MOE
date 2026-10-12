**Knockout loss vs the expert's own direct write on the same clean prompts (eval items whose final position routes the expert; all-position reroute). W = DLA of c_e (baseline), N = net change of the layer's MoE direct write (baseline - knockout; includes the replacement expert), K = knockout loss of Delta (logits). 1 - K/W = compensation in the ext9 synthesis convention; K/N > 1 means the downstream network amplifies the loss.**

| model | condition | expert | routed items | direct write W | net layer direct change N | knockout K | K / W [95% CI] | K / N [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre | L37E032 | 24 | -0.0013 | -0.0037 | 0.1823 | -141.07 [-651.02, 977.51] | -49.00 [-487.30, 583.46] |
| Qwen3-30B-A3B-Base | f1pre | L42E115 | 7 | 0.1349 | 0.1358 | 0.0893 | 0.66 [-3.31, 8.27] | 0.66 [-6.85, 4.00] |
