**4c W2: final-position single-layer patches, Mixtral without vs with BOS on the identical 776-pair case set (main 128/128 pairs; pair bootstrap 95% CI)**

| Run | MoE peak (disc. L, val. / drop) | Attention peak | Block peak | Attention share (main val.) | Share (rep val.) | Mean drop (main) |
|---|---|---|---|---|---|---|
| Mixtral-8x7B, no BOS (ext12) | L20: 0.179 [0.169, 0.190] | L13: 0.145 [0.130, 0.161] | L19: 0.253 [0.240, 0.267] | 0.349 [0.337, 0.362] | 0.346 | +7.64 |
| Mixtral-8x7B, BOS (Phase 3) | L20: 0.172 [0.162, 0.183] | L13: 0.141 [0.127, 0.157] | L19: 0.251 [0.238, 0.264] | 0.344 [0.331, 0.357] | 0.350 | +7.78 |
