**Attention at the swapped name (entity -> attribute; 'site' = the swapped name or noun; validation)**

| Model | Items | Pairs | Name site: attn peak | Name site: attn L4–L9 sum | Name site: MoE peak | Final: attn peak |
|---|---|---|---|---|---|---|
| Qwen3 | name templates (vt_e2a_orig) | 128 | L6 0.32 | 0.34 | L13 0.06 | 0.05 |
| Qwen3 | name templates (vt_e2a_neu) | 128 | L6 0.31 | 0.34 | L0 0.07 | 0.06 |
| Qwen3 | name templates (cp_vt_orig) | 128 | L6 0.19 | 0.20 | L37 0.06 | 0.08 |
| Qwen3 | name templates (cp_vt_neu) | 128 | L6 0.18 | 0.18 | L37 0.05 | 0.10 |
| Qwen3 | Phase-3 option swap, objects | 118 | L34 0.01 | 0.02 | L0 0.15 | 0.03 |
| Qwen3 | Phase-3 option swap, names | 10 | L3 0.04 | 0.03 | L36 0.03 | 0.07 |
| Mixtral BOS | name templates (vt_e2a_orig) | 128 | L7 0.38 | 0.65 | L0 0.99 | 0.15 |
| Mixtral BOS | name templates (vt_e2a_neu) | 128 | L7 0.41 | 0.68 | L0 0.99 | 0.14 |
| Mixtral BOS | name templates (cp_vt_orig) | 128 | L9 0.36 | 0.76 | L0 0.99 | 0.15 |
| Mixtral BOS | name templates (cp_vt_neu) | 128 | L9 0.46 | 0.84 | L0 0.99 | 0.13 |
| Mixtral BOS | Phase-3 option swap, objects | 118 | L12 0.05 | 0.03 | L0 0.90 | 0.14 |
| Mixtral BOS | Phase-3 option swap, names | 10 | L7 0.21 | 0.42 | L0 0.97 | 0.16 |
