**Position x layer grid by site (validation; single-position, single-layer patch rescue / drop; not additive across positions)**

| Model | Condition | Pairs | Swapped word: attn peak | Swapped word: MoE peak | Tokens in between: max peak | Final: attn peak | Final: MoE peak | Hand-off (word ≥0.5 until → final ≥0.5 from) |
|---|---|---|---|---|---|---|---|---|
| Qwen3 | Phase-3 option swap (objects mostly) | 128 | L34 0.01 | L0 0.14 | 0.02 | L42 0.03 | L41 0.22 | L28 → L29 |
| Qwen3 | same twins, ask attribute (dir_opt) | 90 | L34 0.01 | L0 0.18 | 0.02 | L42 0.04 | L41 0.19 | L28 → L30 |
| Qwen3 | same twins, ask entity (dir_f1) | 90 | L27 0.05 | L2 0.22 | 0.02 | L43 0.21 | L43 0.09 | L34 → L36 |
| Qwen3 | ask entity, large pool (f1) | 128 | L34 0.04 | L0 0.13 | 0.01 | L39 0.29 | L37 0.14 | L25 → L35 |
| Qwen3 | ask attribute + Options prefix (cp_nat) | 140 | L6 0.02 | L0 0.14 | 0.02 | L41 0.06 | L43 0.16 | L26 → L29 |
| Qwen3 | name templates, ask attribute (vt_e2a_orig) | 128 | L6 0.32 | L13 0.06 | 0.02 | L42 0.05 | L43 0.17 | L20 → L30 |
| Qwen3 | name templates, ask entity (vt_a2e_orig) | 37 | L36 0.06 | L2 0.53 | 0.02 | L39 0.37 | L37 0.07 | L30 → L32 |
| Qwen3 | natural sentence, ask entity (f2e) | 32 | L34 0.02 | L0 0.20 | 0.04 | L45 0.20 | L37 0.08 | L6 → L37 |
| Mixtral BOS | Phase-3 option swap (objects mostly) | 128 | L12 0.05 | L0 0.91 | 0.02 | L13 0.14 | L20 0.17 | L12 → L13 |
| Mixtral BOS | same twins, ask attribute (dir_opt) | 94 | L12 0.04 | L0 0.95 | 0.03 | L13 0.12 | L20 0.15 | L12 → L13 |
| Mixtral BOS | same twins, ask entity (dir_f1) | 94 | L15 0.06 | L0 0.55 | 0.01 | L16 0.27 | L20 0.15 | L14 → L16 |
| Mixtral BOS | ask entity, large pool (f1) | 101 | L15 0.05 | L0 0.51 | 0.01 | L16 0.29 | L20 0.12 | L13 → L16 |
| Mixtral BOS | ask attribute + Options prefix (cp_nat) | 204 | L12 0.06 | L0 0.90 | 0.03 | L13 0.15 | L20 0.17 | L12 → L13 |
| Mixtral BOS | name templates, ask attribute (vt_e2a_orig) | 128 | L7 0.38 | L0 0.99 | 0.04 | L19 0.15 | L19 0.18 | L9 → L13 |
| Mixtral BOS | natural sentence, ask entity (f2e) | 18 | L16 0.01 | L0 0.42 | 0.04 | L19 0.25 | L19 0.08 | L5 → L17 |
