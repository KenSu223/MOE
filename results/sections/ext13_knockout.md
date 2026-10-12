**Summary.** Phase 4's knockout (engine route mask at every position, reroute: the expert leaves the router's menu and the next-best expert takes the slot; F = fraction of the clean margin lost) applied to every Phase-5 condition on clean prompts of the validation and replication pairs (both twins). Targets: each condition's own selected expert, Phase 3's WinoGrande expert (Qwen3 L41E117, Mixtral L20E000) on every condition, the CounterFact experts as cross-task controls, the condition's population top-1/3/5/10 sets, frequency-matched random sets and same-layer controls. Qwen3-30B-A3B-Base, WinoGrande expert L41E117 knocked out — entity → attribute: dir_opt +0.025 [+0.011, +0.041]; vt_e2a_orig +0.001 [-0.009, +0.013]; vt_e2a_neu +0.001 [-0.009, +0.012] | copyable answer (prefix): cp_nat +0.027 [+0.019, +0.034]; cp_vt_orig -0.001 [-0.009, +0.006]; cp_vt_neu +0.000 [-0.009, +0.011] | attribute → entity: f1 -0.004 [-0.009, +0.002]; dir_f1 +0.001 [-0.008, +0.011]; f2e +0.004 [-0.009, +0.017]; vt_a2e_orig -0.009 [-0.031, -0.006]. Qwen3-30B-A3B-Base, own selections — f1: L37E032 +0.042 [+0.029, +0.055] (rank 1/17 in its layer), necessary; top-10 +0.189 (random 10-sets +0.007) | dir_f1: L44E116 +0.021 [+0.007, +0.036] (rank 1/17 in its layer), below the necessity criterion; top-10 +0.182 (random 10-sets -0.022) | dir_opt: L41E117 +0.025 [+0.011, +0.041] (rank 2/17 in its layer), below the necessity criterion; top-10 +0.185 (random 10-sets -0.006) | f2e: L43E036 +0.015 [-0.004, +0.032] (rank 1/17 in its layer), below the necessity criterion; top-10 +0.029 (random 10-sets -0.020) | vt_e2a_orig: L39E121 +0.005 [-0.013, +0.030] (rank 15/17 in its layer), below the necessity criterion; top-10 +0.056 (random 10-sets +0.001) | vt_e2a_neu: L42E123 +0.020 [-0.006, +0.046] (rank 1/17 in its layer), below the necessity criterion; top-10 +0.161 (random 10-sets +0.037) | vt_a2e_orig: L37E032 +0.034 [+0.024, +0.118] (rank 2/17 in its layer), necessary; top-10 +0.288 (random 10-sets +0.003) | cp_vt_orig: L39E001 +0.008 [-0.017, +0.035] (rank 6/17 in its layer), below the necessity criterion; top-10 +0.097 (random 10-sets +0.030) | cp_vt_neu: L42E123 -0.007 [-0.021, +0.010] (rank 8/17 in its layer), below the necessity criterion; top-10 +0.058 (random 10-sets +0.013) | cp_nat: L41E117 +0.027 [+0.019, +0.034] (rank 1/17 in its layer), necessary; top-10 +0.203 (random 10-sets +0.030). Mixtral-8x7B (BOS), WinoGrande expert L20E000 knocked out — entity → attribute: dir_opt +0.025 [+0.016, +0.034]; vt_e2a_orig +0.012 [-0.016, +0.040]; vt_e2a_neu +0.023 [-0.007, +0.056] | copyable answer (prefix): cp_nat +0.045 [+0.039, +0.050]; cp_vt_orig +0.020 [-0.012, +0.062]; cp_vt_neu -0.001 [-0.017, +0.034] | attribute → entity: f1 +0.005 [-0.002, +0.012]; dir_f1 +0.002 [-0.004, +0.008]; f2e +0.000 [-0.007, +0.008]. Mixtral-8x7B (BOS), own selections — f1: L20E001 +0.008 [-0.006, +0.022] (rank 1/7 in its layer), below the necessity criterion; top-10 +0.229 (random 10-sets -0.002) | dir_f1: L20E001 -0.005 [-0.021, +0.011] (rank 3/7 in its layer), below the necessity criterion; top-10 +0.243 (random 10-sets -0.027) | dir_opt: L20E000 +0.025 [+0.016, +0.034] (rank 1/8 in its layer), necessary; top-10 +0.171 (random 10-sets -0.000) | f2e: L19E004 +0.030 [+0.005, +0.056] (rank 1/7 in its layer), below the necessity criterion; top-10 +0.017 (random 10-sets +0.007) | vt_e2a_orig: L19E006 +0.042 [+0.024, +0.061] (rank 1/7 in its layer), necessary; top-10 +0.075 (random 10-sets +0.014) | vt_e2a_neu: L19E006 +0.037 [+0.013, +0.064] (rank 1/7 in its layer), necessary; top-10 +0.058 (random 10-sets +0.040) | cp_vt_orig: L19E006 +0.017 [-0.004, +0.036] (rank 1/7 in its layer), below the necessity criterion; top-10 +0.087 (random 10-sets -0.002) | cp_vt_neu: L19E006 +0.035 [+0.012, +0.071] (rank 1/7 in its layer), necessary; top-10 +0.045 (random 10-sets +0.013) | cp_nat: L21E006 +0.023 [+0.019, +0.027] (rank 1/7 in its layer), necessary; top-10 +0.185 (random 10-sets +0.009). **P5-8 (is the WinoGrande expert necessary when the answer can be copied?)** Qwen3-30B-A3B-Base L41E117: cp_vt_orig F +0.002 vs vt_e2a_orig -0.010, difference +0.012 [-0.005, +0.038] (paired, shared source items) -> unchanged within the CI; Qwen3-30B-A3B-Base L41E117: cp_vt_neu F -0.006 vs vt_e2a_neu -0.005, difference -0.001 [-0.030, +0.033] (paired, shared source items) -> unchanged within the CI; Qwen3-30B-A3B-Base L41E117: cp_nat F +0.023 vs dir_opt +0.033, difference -0.010 [-0.041, +0.020] (paired, shared source items) -> unchanged within the CI; Mixtral-8x7B (BOS) L20E000: cp_vt_orig F +0.025 vs vt_e2a_orig -0.004, difference +0.029 [-0.000, +0.057] (paired, shared source items) -> unchanged within the CI; Mixtral-8x7B (BOS) L20E000: cp_vt_neu F -0.000 vs vt_e2a_neu +0.004, difference -0.004 [-0.008, +0.015] (paired, shared source items) -> unchanged within the CI; Mixtral-8x7B (BOS) L20E000: cp_nat F +0.032 vs dir_opt +0.036, difference -0.003 [-0.014, +0.007] (paired, shared source items) -> unchanged within the CI. **Direction (same items, answer in context vs generated):** Qwen3-30B-A3B-Base L41E117: vt_a2e_orig F -0.009 vs cp_vt_orig -0.001, difference -0.008 [-0.030, +0.000] (unpaired) -> unchanged within the CI; Qwen3-30B-A3B-Base L41E117: dir_f1 F -0.001 vs cp_nat +0.023, difference -0.024 [-0.052, +0.004] (paired, shared source items) -> unchanged within the CI; Qwen3-30B-A3B-Base L41E117: dir_f1 F +0.001 vs dir_opt +0.025, difference -0.024 [-0.042, -0.007] (paired, shared source items) -> reduced to about zero; Qwen3-30B-A3B-Base L41E117: vt_a2e_orig F -0.009 vs vt_e2a_orig +0.001, difference -0.011 [-0.034, -0.000] (unpaired) -> unchanged within the CI; Mixtral-8x7B (BOS) L20E000: dir_f1 F +0.008 vs cp_nat +0.032, difference -0.024 [-0.047, -0.003] (paired, shared source items) -> reduced to about zero; Mixtral-8x7B (BOS) L20E000: dir_f1 F +0.002 vs dir_opt +0.025, difference -0.023 [-0.034, -0.012] (paired, shared source items) -> reduced to about zero.

### What was run

**Design.** Models Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1 with BOS (Phases 3-4 default), bf16, engine E4 route masks (`PrefillSpec.route_mask`, reroute; Phase 4 verification `results/verify_ext9_engine_olmoe.json`). Per condition (registry `data/wino_ref/p5_conditions.json`, agent p5-data): items = clean prompts of both twins of every pair except the main discovery pairs (validation + replication splits; the discovery pairs selected the experts), Δ toward each prompt's own answer. Full scope (baseline, targets, population sets, null) = these eval items; sub scope (same-layer controls, 5 frequency-matched random sets per set size, final-position-only masks, cross-condition knockouts) = the main validation pairs. Same-layer controls for the main targets (own selected expert, WinoGrande expert): Mixtral the 7 other experts of the layer, Qwen3 the 16 experts of the layer most often routed at the final position on the condition's eval prompts (targets excluded). Random sets: each member of the population top-k replaced by one of the 4 (Qwen3) / 3 (Mixtral) experts of its layer with the nearest final-position routing frequency (Phase 4 rule). Null = the unmasked baseline recomputed in passes of another composition (noise floor). F = (Σ Δ_base − Σ Δ_ko) / Σ Δ_base; bootstrap (5,000) over the condition's units: template seeds for the semi-synthetic template conditions (vt_*, cp_vt_*; 48 name pairs per seed are not independent), source pairs for cp_nat (both prefix orders together), pairs otherwise. Cross-condition differences resample the conditions independently; paired differences use the source items present in both conditions with a joint bootstrap over shared units. Not run: zero mode (Phase 4: zero ≈ reroute for every target, `results/tables/ext9_knockout_final_zero.md`) and the wikitext NLL control (generic damage of the same experts measured in Phase 4, ≤ 0.008 nats/token for single experts).

| Model | Condition | Eval pairs (units) | Baseline Δ | Pair acc | Selected expert | Population top-3 | GPU (min) |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1 (F1 attribute → entity (fill-in-the-blank)) | 384 (384 pair_id) | 2.36 | 1.000 | L37E032 | L37E032 L44E116 L39E103 | 5.0 |
| Qwen3-30B-A3B-Base | dir_f1 (776 set, attribute → entity (fill-in-the-blank)) | 90 (90 pair_id) | 2.61 | 1.000 | L44E116 | L44E116 L37E032 L43E070 | 4.0 |
| Qwen3-30B-A3B-Base | dir_opt (776 set, entity → attribute (Phase-3 option swap)) | 90 (90 pair_id) | 4.38 | 1.000 | L41E117 | L43E081 L41E117 L31E124 | 3.6 |
| Qwen3-30B-A3B-Base | f2e (attribute → entity, natural sentence) | 32 (32 pair_id) | 4.08 | 1.000 | L43E036 | L37E107 L44E116 L43E036 | 3.0 |
| Qwen3-30B-A3B-Base | vt_e2a_orig (templates entity → attribute, original verb) | 128 (17 seed) | 3.27 | 1.000 | L39E121 | L42E123 L39E121 L45E092 | 4.9 |
| Qwen3-30B-A3B-Base | vt_e2a_neu (templates entity → attribute, neutral verb) | 128 (12 seed) | 3.02 | 1.000 | L42E123 | L42E123 L39E121 L45E092 | 4.7 |
| Qwen3-30B-A3B-Base | vt_a2e_orig (templates attribute → entity, original verb) | 37 (5 seed) | 1.82 | 1.000 | L37E032 | L37E032 L32E061 L39E103 | 2.7 |
| Qwen3-30B-A3B-Base | cp_vt_orig (copy: templates e → a + prefix, original verb) | 128 (13 seed) | 2.80 | 1.000 | L39E001 | L42E123 L39E001 L39E121 | 4.5 |
| Qwen3-30B-A3B-Base | cp_vt_neu (copy: templates e → a + prefix, neutral verb) | 128 (8 seed) | 2.78 | 1.000 | L42E123 | L42E123 L39E121 L37E101 | 4.9 |
| Qwen3-30B-A3B-Base | cp_nat (copy: 776 set e → a + prefix) | 460 (230 base_pair_id) | 3.02 | 1.000 | L41E117 | L43E081 L41E117 L34E119 | 6.0 |
| Mixtral-8x7B (BOS) | f1 (F1 attribute → entity (fill-in-the-blank)) | 101 (101 pair_id) | 2.18 | 1.000 | L20E001 | L19E004 L20E001 L20E005 | 6.4 |
| Mixtral-8x7B (BOS) | dir_f1 (776 set, attribute → entity (fill-in-the-blank)) | 94 (94 pair_id) | 2.33 | 1.000 | L20E001 | L19E004 L20E001 L23E000 | 6.1 |
| Mixtral-8x7B (BOS) | dir_opt (776 set, entity → attribute (Phase-3 option swap)) | 94 (94 pair_id) | 4.83 | 1.000 | L20E000 | L20E000 L19E006 L21E006 | 6.1 |
| Mixtral-8x7B (BOS) | f2e (attribute → entity, natural sentence) | 18 (18 pair_id) | 3.47 | 1.000 | L19E004 | L19E004 L17E005 L20E005 | 4.4 |
| Mixtral-8x7B (BOS) | vt_e2a_orig (templates entity → attribute, original verb) | 128 (18 seed) | 3.25 | 1.000 | L19E006 | L19E006 L20E000 L21E003 | 6.3 |
| Mixtral-8x7B (BOS) | vt_e2a_neu (templates entity → attribute, neutral verb) | 128 (12 seed) | 2.92 | 1.000 | L19E006 | L19E006 L21E003 L20E000 | 6.2 |
| Mixtral-8x7B (BOS) | cp_vt_orig (copy: templates e → a + prefix, original verb) | 128 (10 seed) | 2.63 | 1.000 | L19E006 | L19E006 L21E003 L14E007 | 6.3 |
| Mixtral-8x7B (BOS) | cp_vt_neu (copy: templates e → a + prefix, neutral verb) | 128 (8 seed) | 2.70 | 1.000 | L19E006 | L21E003 L19E006 L20E005 | 4.6 |
| Mixtral-8x7B (BOS) | cp_nat (copy: 776 set e → a + prefix) | 640 (320 base_pair_id) | 3.67 | 1.000 | L21E006 | L20E000 L19E006 L21E006 | 12.3 |


### 1. Targets and population sets (eval items)

| model | condition | knockout | role | routed at final | F [95% CI] | pair acc base -> ko | F routed items | null F |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1 | L37E032 | selected | 0.9818 | +0.042 [+0.029, +0.055] | 1.000 -> 0.979 | 0.043 | -0.004 |
| Qwen3-30B-A3B-Base | f1 | L41E117 | wg | 0.0221 | -0.004 [-0.009, +0.002] | 1.000 -> 1.000 | 0.0127 | -0.004 |
| Qwen3-30B-A3B-Base | f1 | L44E069 | cf | 0.0924 | +0.000 [-0.007, +0.007] | 1.000 -> 1.000 | 0.0161 | -0.004 |
| Qwen3-30B-A3B-Base | f1 | L42E115 | cf | 0.3568 | -0.009 [-0.016, -0.002] | 1.000 -> 1.000 | -0.012 | -0.004 |
| Qwen3-30B-A3B-Base | f1 | top-3 | pop |  | +0.089 [+0.077, +0.101] | 1.000 -> 0.974 |  | -0.004 |
| Qwen3-30B-A3B-Base | f1 | top-5 | pop |  | +0.105 [+0.092, +0.119] | 1.000 -> 0.964 |  | -0.004 |
| Qwen3-30B-A3B-Base | f1 | top-10 | pop |  | +0.189 [+0.175, +0.204] | 1.000 -> 0.922 |  | -0.004 |
| Qwen3-30B-A3B-Base | dir_f1 | L44E116 | selected | 0.85 | +0.021 [+0.007, +0.036] | 1.000 -> 1.000 | 0.0233 | -0.012 |
| Qwen3-30B-A3B-Base | dir_f1 | L41E117 | wg | 0.0 | +0.001 [-0.008, +0.011] | 1.000 -> 1.000 |  | -0.012 |
| Qwen3-30B-A3B-Base | dir_f1 | L44E069 | cf | 0.0833 | -0.001 [-0.012, +0.010] | 1.000 -> 1.000 | 0.0069 | -0.012 |
| Qwen3-30B-A3B-Base | dir_f1 | L42E115 | cf | 0.1333 | -0.006 [-0.018, +0.005] | 1.000 -> 1.000 | -0.0243 | -0.012 |
| Qwen3-30B-A3B-Base | dir_f1 | top-3 | pop |  | +0.107 [+0.088, +0.128] | 1.000 -> 1.000 |  | -0.012 |
| Qwen3-30B-A3B-Base | dir_f1 | top-5 | pop |  | +0.128 [+0.105, +0.150] | 1.000 -> 0.989 |  | -0.012 |
| Qwen3-30B-A3B-Base | dir_f1 | top-10 | pop |  | +0.182 [+0.146, +0.218] | 1.000 -> 0.967 |  | -0.012 |
| Qwen3-30B-A3B-Base | dir_opt | L41E117 | selected+wg | 0.7722 | +0.025 [+0.011, +0.041] | 1.000 -> 1.000 | 0.0327 | +0.011 |
| Qwen3-30B-A3B-Base | dir_opt | L44E069 | cf | 0.0111 | -0.003 [-0.009, +0.004] | 1.000 -> 1.000 |  | +0.011 |
| Qwen3-30B-A3B-Base | dir_opt | L42E115 | cf | 0.0111 | +0.001 [-0.007, +0.008] | 1.000 -> 1.000 |  | +0.011 |
| Qwen3-30B-A3B-Base | dir_opt | top-3 | pop |  | +0.067 [+0.043, +0.090] | 1.000 -> 0.989 |  | +0.011 |
| Qwen3-30B-A3B-Base | dir_opt | top-5 | pop |  | +0.122 [+0.091, +0.154] | 1.000 -> 0.967 |  | +0.011 |
| Qwen3-30B-A3B-Base | dir_opt | top-10 | pop |  | +0.185 [+0.154, +0.215] | 1.000 -> 0.944 |  | +0.011 |
| Qwen3-30B-A3B-Base | f2e | L43E036 | selected | 0.9844 | +0.015 [-0.004, +0.032] | 1.000 -> 1.000 | 0.0149 | -0.018 |
| Qwen3-30B-A3B-Base | f2e | L41E117 | wg | 0.0625 | +0.004 [-0.009, +0.017] | 1.000 -> 1.000 |  | -0.018 |
| Qwen3-30B-A3B-Base | f2e | L44E069 | cf | 0.4375 | +0.001 [-0.011, +0.013] | 1.000 -> 1.000 | -0.0079 | -0.018 |
| Qwen3-30B-A3B-Base | f2e | L42E115 | cf | 0.625 | +0.008 [-0.011, +0.027] | 1.000 -> 1.000 | 0.009 | -0.018 |
| Qwen3-30B-A3B-Base | f2e | top-3 | pop |  | +0.013 [-0.029, +0.047] | 1.000 -> 1.000 |  | -0.018 |
| Qwen3-30B-A3B-Base | f2e | top-5 | pop |  | +0.018 [-0.032, +0.058] | 1.000 -> 1.000 |  | -0.018 |
| Qwen3-30B-A3B-Base | f2e | top-10 | pop |  | +0.029 [-0.047, +0.091] | 1.000 -> 1.000 |  | -0.018 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L39E121 | selected | 0.7188 | +0.005 [-0.013, +0.030] | 1.000 -> 1.000 | 0.0133 | +0.001 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L41E117 | wg | 0.4648 | +0.001 [-0.009, +0.013] | 1.000 -> 1.000 | -0.0037 | +0.001 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L44E069 | cf | 0.2383 | +0.010 [-0.009, +0.034] | 1.000 -> 1.000 | 0.0504 | +0.001 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L42E115 | cf | 0.0781 | -0.001 [-0.011, +0.008] | 1.000 -> 1.000 | -0.0047 | +0.001 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | top-3 | pop |  | +0.035 [+0.003, +0.066] | 1.000 -> 1.000 |  | +0.001 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | top-5 | pop |  | +0.025 [-0.016, +0.065] | 1.000 -> 1.000 |  | +0.001 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | top-10 | pop |  | +0.056 [-0.030, +0.141] | 1.000 -> 0.969 |  | +0.001 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L42E123 | selected | 1.0 | +0.020 [-0.006, +0.046] | 1.000 -> 0.992 | 0.0201 | +0.000 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L41E117 | wg | 0.3867 | +0.001 [-0.009, +0.012] | 1.000 -> 1.000 | 0.0019 | +0.000 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L44E069 | cf | 0.2031 | +0.009 [-0.009, +0.035] | 1.000 -> 1.000 | 0.0492 | +0.000 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L42E115 | cf | 0.0938 | -0.002 [-0.005, +0.000] | 1.000 -> 1.000 | 0.0026 | +0.000 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | top-3 | pop |  | +0.031 [-0.018, +0.074] | 1.000 -> 0.992 |  | +0.000 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | top-5 | pop |  | +0.037 [-0.041, +0.111] | 1.000 -> 1.000 |  | +0.000 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | top-10 | pop |  | +0.161 [+0.080, +0.243] | 1.000 -> 0.891 |  | +0.000 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L37E032 | selected | 1.0 | +0.034 [+0.024, +0.118] | 1.000 -> 1.000 | 0.0335 | +0.004 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L41E117 | wg | 0.0 | -0.009 [-0.031, -0.006] | 1.000 -> 1.000 |  | +0.004 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L44E069 | cf | 0.0 | -0.001 [-0.017, +0.013] | 1.000 -> 1.000 |  | +0.004 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L42E115 | cf | 0.6486 | +0.003 [+0.000, +0.017] | 1.000 -> 1.000 | -0.0029 | +0.004 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | top-3 | pop |  | +0.275 [+0.250, +0.377] | 1.000 -> 0.973 |  | +0.004 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | top-5 | pop |  | +0.288 [+0.265, +0.402] | 1.000 -> 0.973 |  | +0.004 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | top-10 | pop |  | +0.288 [+0.280, +0.426] | 1.000 -> 0.865 |  | +0.004 |
| Qwen3-30B-A3B-Base | cp_vt_orig | L39E001 | selected | 0.7109 | +0.008 [-0.017, +0.035] | 1.000 -> 1.000 | 0.0173 | +0.001 |
| Qwen3-30B-A3B-Base | cp_vt_orig | L41E117 | wg | 0.4258 | -0.001 [-0.009, +0.006] | 1.000 -> 1.000 | -0.0073 | +0.001 |
| Qwen3-30B-A3B-Base | cp_vt_orig | L44E069 | cf | 0.3828 | +0.005 [-0.003, +0.017] | 1.000 -> 1.000 | 0.0229 | +0.001 |
| Qwen3-30B-A3B-Base | cp_vt_orig | L42E115 | cf | 0.2148 | +0.002 [-0.006, +0.011] | 1.000 -> 1.000 | 0.0034 | +0.001 |
| Qwen3-30B-A3B-Base | cp_vt_orig | top-3 | pop |  | +0.035 [-0.022, +0.099] | 1.000 -> 1.000 |  | +0.001 |
| Qwen3-30B-A3B-Base | cp_vt_orig | top-5 | pop |  | +0.072 [-0.009, +0.165] | 1.000 -> 0.977 |  | +0.001 |
| Qwen3-30B-A3B-Base | cp_vt_orig | top-10 | pop |  | +0.097 [-0.012, +0.210] | 1.000 -> 0.914 |  | +0.001 |
| Qwen3-30B-A3B-Base | cp_vt_neu | L42E123 | selected | 1.0 | -0.007 [-0.021, +0.010] | 1.000 -> 1.000 | -0.0067 | -0.010 |
| Qwen3-30B-A3B-Base | cp_vt_neu | L41E117 | wg | 0.543 | +0.000 [-0.009, +0.011] | 1.000 -> 1.000 | 0.0093 | -0.010 |
| Qwen3-30B-A3B-Base | cp_vt_neu | L44E069 | cf | 0.2617 | -0.000 [-0.006, +0.010] | 1.000 -> 1.000 | -0.0032 | -0.010 |
| Qwen3-30B-A3B-Base | cp_vt_neu | L42E115 | cf | 0.1367 | -0.004 [-0.018, +0.013] | 1.000 -> 1.000 | 0.0095 | -0.010 |
| Qwen3-30B-A3B-Base | cp_vt_neu | top-3 | pop |  | +0.029 [-0.009, +0.089] | 1.000 -> 1.000 |  | -0.010 |
| Qwen3-30B-A3B-Base | cp_vt_neu | top-5 | pop |  | +0.043 [-0.001, +0.100] | 1.000 -> 1.000 |  | -0.010 |
| Qwen3-30B-A3B-Base | cp_vt_neu | top-10 | pop |  | +0.058 [+0.023, +0.086] | 1.000 -> 1.000 |  | -0.010 |
| Qwen3-30B-A3B-Base | cp_nat | L41E117 | selected+wg | 0.8924 | +0.027 [+0.019, +0.034] | 1.000 -> 0.998 | 0.0302 | -0.004 |
| Qwen3-30B-A3B-Base | cp_nat | L44E069 | cf | 0.012 | -0.005 [-0.012, +0.001] | 1.000 -> 1.000 | -0.0412 | -0.004 |
| Qwen3-30B-A3B-Base | cp_nat | L42E115 | cf | 0.0043 | -0.009 [-0.015, -0.003] | 1.000 -> 1.000 |  | -0.004 |
| Qwen3-30B-A3B-Base | cp_nat | top-3 | pop |  | +0.070 [+0.053, +0.086] | 1.000 -> 0.983 |  | -0.004 |
| Qwen3-30B-A3B-Base | cp_nat | top-5 | pop |  | +0.103 [+0.087, +0.120] | 1.000 -> 0.978 |  | -0.004 |
| Qwen3-30B-A3B-Base | cp_nat | top-10 | pop |  | +0.203 [+0.180, +0.226] | 1.000 -> 0.937 |  | -0.004 |
| Mixtral-8x7B (BOS) | f1 | L20E001 | selected | 0.7426 | +0.008 [-0.006, +0.022] | 1.000 -> 0.990 | 0.0071 | +0.001 |
| Mixtral-8x7B (BOS) | f1 | L20E000 | wg | 0.1881 | +0.005 [-0.002, +0.012] | 1.000 -> 1.000 | 0.0137 | +0.001 |
| Mixtral-8x7B (BOS) | f1 | L19E002 | cf | 0.0396 | +0.003 [-0.004, +0.010] | 1.000 -> 1.000 | 0.0246 | +0.001 |
| Mixtral-8x7B (BOS) | f1 | L21E001 | cf | 0.3515 | +0.008 [+0.002, +0.015] | 1.000 -> 1.000 | 0.006 | +0.001 |
| Mixtral-8x7B (BOS) | f1 | L18E001 | cf | 0.1188 | +0.003 [-0.005, +0.011] | 1.000 -> 1.000 | 0.0164 | +0.001 |
| Mixtral-8x7B (BOS) | f1 | top-3 | pop |  | +0.062 [+0.037, +0.088] | 1.000 -> 0.990 |  | +0.001 |
| Mixtral-8x7B (BOS) | f1 | top-5 | pop |  | +0.190 [+0.155, +0.228] | 1.000 -> 0.941 |  | +0.001 |
| Mixtral-8x7B (BOS) | f1 | top-10 | pop |  | +0.229 [+0.187, +0.272] | 1.000 -> 0.822 |  | +0.001 |
| Mixtral-8x7B (BOS) | dir_f1 | L20E001 | selected | 0.8777 | -0.005 [-0.021, +0.011] | 1.000 -> 1.000 | -0.0052 | -0.007 |
| Mixtral-8x7B (BOS) | dir_f1 | L20E000 | wg | 0.0745 | +0.002 [-0.004, +0.008] | 1.000 -> 1.000 | 0.0216 | -0.007 |
| Mixtral-8x7B (BOS) | dir_f1 | L19E002 | cf | 0.0106 | +0.001 [-0.005, +0.007] | 1.000 -> 1.000 |  | -0.007 |
| Mixtral-8x7B (BOS) | dir_f1 | L21E001 | cf | 0.3723 | -0.001 [-0.011, +0.008] | 1.000 -> 1.000 | -0.0049 | -0.007 |
| Mixtral-8x7B (BOS) | dir_f1 | L18E001 | cf | 0.0106 | +0.000 [-0.007, +0.007] | 1.000 -> 1.000 |  | -0.007 |
| Mixtral-8x7B (BOS) | dir_f1 | top-3 | pop |  | +0.056 [+0.029, +0.080] | 1.000 -> 1.000 |  | -0.007 |
| Mixtral-8x7B (BOS) | dir_f1 | top-5 | pop |  | +0.132 [+0.099, +0.166] | 1.000 -> 0.979 |  | -0.007 |
| Mixtral-8x7B (BOS) | dir_f1 | top-10 | pop |  | +0.243 [+0.199, +0.286] | 1.000 -> 0.894 |  | -0.007 |
| Mixtral-8x7B (BOS) | dir_opt | L20E000 | selected+wg | 0.984 | +0.025 [+0.016, +0.034] | 1.000 -> 1.000 | 0.0256 | -0.000 |
| Mixtral-8x7B (BOS) | dir_opt | L19E002 | cf | 0.0053 | -0.001 [-0.004, +0.002] | 1.000 -> 1.000 |  | -0.000 |
| Mixtral-8x7B (BOS) | dir_opt | L21E001 | cf | 0.0213 | +0.000 [-0.002, +0.003] | 1.000 -> 1.000 |  | -0.000 |
| Mixtral-8x7B (BOS) | dir_opt | L18E001 | cf | 0.0 | -0.000 [-0.003, +0.003] | 1.000 -> 1.000 |  | -0.000 |
| Mixtral-8x7B (BOS) | dir_opt | top-3 | pop |  | +0.073 [+0.057, +0.089] | 1.000 -> 1.000 |  | -0.000 |
| Mixtral-8x7B (BOS) | dir_opt | top-5 | pop |  | +0.127 [+0.108, +0.146] | 1.000 -> 1.000 |  | -0.000 |
| Mixtral-8x7B (BOS) | dir_opt | top-10 | pop |  | +0.171 [+0.149, +0.194] | 1.000 -> 1.000 |  | -0.000 |
| Mixtral-8x7B (BOS) | f2e | L19E004 | selected | 1.0 | +0.030 [+0.005, +0.056] | 1.000 -> 1.000 | 0.03 | +0.004 |
| Mixtral-8x7B (BOS) | f2e | L20E000 | wg | 0.2778 | +0.000 [-0.007, +0.008] | 1.000 -> 1.000 | 0.0 | +0.004 |
| Mixtral-8x7B (BOS) | f2e | L19E002 | cf | 0.0 | -0.001 [-0.007, +0.006] | 1.000 -> 1.000 |  | +0.004 |
| Mixtral-8x7B (BOS) | f2e | L21E001 | cf | 0.4444 | +0.010 [+0.002, +0.019] | 1.000 -> 1.000 | 0.0148 | +0.004 |
| Mixtral-8x7B (BOS) | f2e | L18E001 | cf | 0.1111 | -0.008 [-0.023, +0.006] | 1.000 -> 1.000 |  | +0.004 |
| Mixtral-8x7B (BOS) | f2e | top-3 | pop |  | +0.029 [-0.013, +0.065] | 1.000 -> 1.000 |  | +0.004 |
| Mixtral-8x7B (BOS) | f2e | top-5 | pop |  | +0.032 [-0.010, +0.069] | 1.000 -> 1.000 |  | +0.004 |
| Mixtral-8x7B (BOS) | f2e | top-10 | pop |  | +0.017 [-0.036, +0.062] | 1.000 -> 1.000 |  | +0.004 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L19E006 | selected | 0.9375 | +0.042 [+0.024, +0.061] | 1.000 -> 1.000 | 0.0437 | -0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L20E000 | wg | 0.9297 | +0.012 [-0.016, +0.040] | 1.000 -> 1.000 | 0.0123 | -0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L19E002 | cf | 0.1875 | -0.006 [-0.013, -0.001] | 1.000 -> 1.000 | -0.0166 | -0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L21E001 | cf | 0.2617 | +0.001 [-0.004, +0.006] | 1.000 -> 1.000 | 0.0094 | -0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L18E001 | cf | 0.0781 | +0.002 [-0.004, +0.006] | 1.000 -> 1.000 | -0.0205 | -0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | top-3 | pop |  | +0.052 [+0.016, +0.089] | 1.000 -> 1.000 |  | -0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | top-5 | pop |  | +0.060 [+0.016, +0.104] | 1.000 -> 0.992 |  | -0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | top-10 | pop |  | +0.075 [+0.020, +0.127] | 1.000 -> 0.945 |  | -0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L19E006 | selected | 0.9062 | +0.037 [+0.013, +0.064] | 1.000 -> 1.000 | 0.0373 | +0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L20E000 | wg | 0.8594 | +0.023 [-0.007, +0.056] | 1.000 -> 1.000 | 0.0262 | +0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L19E002 | cf | 0.0664 | +0.004 [-0.001, +0.008] | 1.000 -> 1.000 | -0.0026 | +0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L21E001 | cf | 0.3555 | +0.004 [+0.001, +0.007] | 1.000 -> 1.000 | 0.0123 | +0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L18E001 | cf | 0.1094 | +0.001 [-0.005, +0.006] | 1.000 -> 1.000 | -0.0362 | +0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | top-3 | pop |  | +0.053 [+0.010, +0.094] | 1.000 -> 1.000 |  | +0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | top-5 | pop |  | +0.060 [+0.014, +0.099] | 1.000 -> 1.000 |  | +0.003 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | top-10 | pop |  | +0.058 [+0.012, +0.105] | 1.000 -> 1.000 |  | +0.003 |
| Mixtral-8x7B (BOS) | cp_vt_orig | L19E006 | selected | 0.8906 | +0.017 [-0.004, +0.036] | 1.000 -> 1.000 | 0.0172 | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_orig | L20E000 | wg | 0.7812 | +0.020 [-0.012, +0.062] | 1.000 -> 1.000 | 0.0237 | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_orig | L19E002 | cf | 0.0547 | -0.002 [-0.009, +0.005] | 1.000 -> 1.000 | -0.0146 | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_orig | L21E001 | cf | 0.332 | +0.003 [-0.004, +0.011] | 1.000 -> 1.000 | 0.0113 | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_orig | L18E001 | cf | 0.0 | +0.005 [-0.002, +0.014] | 1.000 -> 1.000 |  | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_orig | top-3 | pop |  | +0.076 [+0.028, +0.130] | 1.000 -> 1.000 |  | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_orig | top-5 | pop |  | +0.083 [+0.004, +0.184] | 1.000 -> 0.977 |  | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_orig | top-10 | pop |  | +0.087 [-0.003, +0.198] | 1.000 -> 0.945 |  | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_neu | L19E006 | selected | 0.6953 | +0.035 [+0.012, +0.071] | 1.000 -> 1.000 | 0.0367 | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_neu | L20E000 | wg | 0.6797 | -0.001 [-0.017, +0.034] | 1.000 -> 1.000 | -0.0022 | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_neu | L19E002 | cf | 0.0156 | +0.002 [-0.002, +0.007] | 1.000 -> 1.000 |  | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_neu | L21E001 | cf | 0.3906 | +0.013 [-0.001, +0.029] | 1.000 -> 1.000 | 0.0388 | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_neu | L18E001 | cf | 0.0 | +0.003 [-0.002, +0.013] | 1.000 -> 1.000 |  | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_neu | top-3 | pop |  | +0.047 [+0.016, +0.105] | 1.000 -> 1.000 |  | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_neu | top-5 | pop |  | +0.070 [+0.022, +0.141] | 1.000 -> 1.000 |  | -0.001 |
| Mixtral-8x7B (BOS) | cp_vt_neu | top-10 | pop |  | +0.045 [-0.069, +0.138] | 1.000 -> 0.984 |  | -0.001 |
| Mixtral-8x7B (BOS) | cp_nat | L21E006 | selected | 0.9219 | +0.023 [+0.019, +0.027] | 1.000 -> 1.000 | 0.0248 | -0.001 |
| Mixtral-8x7B (BOS) | cp_nat | L20E000 | wg | 0.9734 | +0.045 [+0.039, +0.050] | 1.000 -> 1.000 | 0.0459 | -0.001 |
| Mixtral-8x7B (BOS) | cp_nat | L19E002 | cf | 0.0219 | -0.001 [-0.002, +0.001] | 1.000 -> 1.000 | 0.0012 | -0.001 |
| Mixtral-8x7B (BOS) | cp_nat | L21E001 | cf | 0.0219 | +0.001 [-0.001, +0.002] | 1.000 -> 1.000 | 0.0034 | -0.001 |
| Mixtral-8x7B (BOS) | cp_nat | L18E001 | cf | 0.0031 | +0.002 [+0.000, +0.003] | 1.000 -> 1.000 |  | -0.001 |
| Mixtral-8x7B (BOS) | cp_nat | top-3 | pop |  | +0.101 [+0.092, +0.111] | 1.000 -> 0.998 |  | -0.001 |
| Mixtral-8x7B (BOS) | cp_nat | top-5 | pop |  | +0.146 [+0.135, +0.157] | 1.000 -> 0.991 |  | -0.001 |
| Mixtral-8x7B (BOS) | cp_nat | top-10 | pop |  | +0.185 [+0.172, +0.198] | 1.000 -> 0.988 |  | -0.001 |


### 2. Main targets against same-layer controls (validation pairs)

| model | condition | target | target F | controls mean | controls max | max control | rank | z |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1 | L37E032 | 0.0484 | 0.0038 | 0.0444 | L37E026 | 1/17 | 3.1693 |
| Qwen3-30B-A3B-Base | f1 | L41E117 | -0.0055 | 0.0027 | 0.0146 | L41E018 | 16/17 | -1.3517 |
| Qwen3-30B-A3B-Base | dir_f1 | L44E116 | 0.0213 | -0.0065 | 0.0024 | L44E108 | 1/17 | 5.8369 |
| Qwen3-30B-A3B-Base | dir_f1 | L41E117 | 0.0011 | -0.0048 | 0.0133 | L41E062 | 5/17 | 0.5793 |
| Qwen3-30B-A3B-Base | dir_opt | L41E117 | 0.0254 | 0.0111 | 0.0297 | L41E053 | 2/17 | 1.6523 |
| Qwen3-30B-A3B-Base | f2e | L43E036 | 0.0146 | -0.029 | -0.0168 | L43E097 | 1/17 | 4.367 |
| Qwen3-30B-A3B-Base | f2e | L41E117 | 0.0041 | -0.0296 | -0.0122 | L41E121 | 1/17 | 3.9062 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L39E121 | 0.005 | 0.01 | 0.0179 | L39E098 | 15/17 | -1.095 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L41E117 | 0.0014 | 0.006 | 0.0292 | L41E062 | 11/17 | -0.4379 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L42E123 | 0.0201 | -0.0053 | 0.0122 | L42E074 | 1/17 | 3.3384 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L41E117 | 0.0011 | -0.0049 | 0.0099 | L41E062 | 3/17 | 0.9181 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L37E032 | 0.0335 | -0.0046 | 0.0651 | L37E026 | 2/17 | 1.6511 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L41E117 | -0.0093 | -0.0066 | 0.0297 | L41E042 | 9/17 | -0.1819 |
| Qwen3-30B-A3B-Base | cp_vt_orig | L39E001 | 0.0084 | 0.004 | 0.0176 | L39E007 | 6/17 | 0.4721 |
| Qwen3-30B-A3B-Base | cp_vt_orig | L41E117 | -0.0013 | 0.0001 | 0.0126 | L41E017 | 12/17 | -0.2078 |
| Qwen3-30B-A3B-Base | cp_vt_neu | L42E123 | -0.0067 | -0.0094 | 0.0072 | L42E074 | 8/17 | 0.2877 |
| Qwen3-30B-A3B-Base | cp_vt_neu | L41E117 | 0.0002 | -0.0062 | 0.0147 | L41E062 | 5/17 | 0.6556 |
| Qwen3-30B-A3B-Base | cp_nat | L41E117 | 0.0344 | -0.0052 | 0.0116 | L41E053 | 1/17 | 6.3972 |
| Mixtral-8x7B (BOS) | f1 | L20E001 | 0.0079 | 0.0015 | 0.0074 | L20E005 | 1/7 | 1.291 |
| Mixtral-8x7B (BOS) | f1 | L20E000 | 0.0051 | 0.0015 | 0.0074 | L20E005 | 3/7 | 0.7214 |
| Mixtral-8x7B (BOS) | dir_f1 | L20E001 | -0.0046 | -0.006 | 0.0049 | L20E005 | 3/7 | 0.191 |
| Mixtral-8x7B (BOS) | dir_f1 | L20E000 | 0.002 | -0.006 | 0.0049 | L20E005 | 2/7 | 1.0997 |
| Mixtral-8x7B (BOS) | dir_opt | L20E000 | 0.0251 | 0.0008 | 0.0073 | L20E001 | 1/8 | 8.0894 |
| Mixtral-8x7B (BOS) | f2e | L19E004 | 0.03 | 0.006 | 0.01 | L19E006 | 1/7 | 7.746 |
| Mixtral-8x7B (BOS) | f2e | L20E000 | 0.0 | 0.003 | 0.009 | L20E004 | 6/8 | -0.6167 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L19E006 | 0.0422 | -0.0022 | 0.0117 | L19E000 | 1/7 | 6.2181 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L20E000 | 0.0117 | -0.0016 | 0.0008 | L20E006 | 1/8 | 9.5941 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L19E006 | 0.0368 | 0.0066 | 0.0202 | L19E000 | 1/7 | 4.2924 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L20E000 | 0.0231 | 0.0026 | 0.006 | L20E003 | 1/8 | 5.1222 |
| Mixtral-8x7B (BOS) | cp_vt_orig | L19E006 | 0.0167 | -0.003 | 0.0019 | L19E000 | 1/7 | 5.5421 |
| Mixtral-8x7B (BOS) | cp_vt_orig | L20E000 | 0.0202 | -0.0006 | 0.0032 | L20E001 | 1/8 | 9.0252 |
| Mixtral-8x7B (BOS) | cp_vt_neu | L19E006 | 0.0345 | -0.0039 | 0.0042 | L19E001 | 1/7 | 6.1197 |
| Mixtral-8x7B (BOS) | cp_vt_neu | L20E000 | -0.0009 | 0.0016 | 0.0045 | L20E004 | 8/8 | -1.741 |
| Mixtral-8x7B (BOS) | cp_nat | L21E006 | 0.0258 | -0.0015 | 0.0014 | L21E000 | 1/7 | 9.2239 |
| Mixtral-8x7B (BOS) | cp_nat | L20E000 | 0.0542 | 0.0012 | 0.0052 | L20E001 | 1/8 | 24.6715 |


### 3. Population sets against frequency-matched random sets (validation pairs)

| model | condition | set | members | population set F | random mean | random max | rank |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1 | top-1 | L37E032 | 0.0484 | 0.0219 | 0.0258 | 1/6 |
| Qwen3-30B-A3B-Base | f1 | top-3 | L37E032 L44E116 L39E103 | 0.1066 | 0.0013 | 0.0072 | 1/6 |
| Qwen3-30B-A3B-Base | f1 | top-5 | L37E032 L44E116 L39E103 L32E061 L43E070 | 0.1152 | 0.01 | 0.0252 | 1/6 |
| Qwen3-30B-A3B-Base | f1 | top-10 | L37E032 L44E116 L39E103 L32E061 L43E070 L44E022 L36E009 L33E122 L42E102 L35E090 | 0.1948 | 0.0071 | 0.0347 | 1/6 |
| Qwen3-30B-A3B-Base | dir_f1 | top-1 | L44E116 | 0.0213 | -0.01 | -0.0093 | 1/6 |
| Qwen3-30B-A3B-Base | dir_f1 | top-3 | L44E116 L37E032 L43E070 | 0.1074 | -0.0114 | 0.0235 | 1/6 |
| Qwen3-30B-A3B-Base | dir_f1 | top-5 | L44E116 L37E032 L43E070 L40E037 L43E114 | 0.1279 | -0.0045 | 0.02 | 1/6 |
| Qwen3-30B-A3B-Base | dir_f1 | top-10 | L44E116 L37E032 L43E070 L40E037 L43E114 L41E062 L42E018 L42E102 L44E022 L31E124 | 0.1818 | -0.0223 | 0.0168 | 1/6 |
| Qwen3-30B-A3B-Base | dir_opt | top-1 | L43E081 | 0.0186 | 0.028 | 0.032 | 5/6 |
| Qwen3-30B-A3B-Base | dir_opt | top-3 | L43E081 L41E117 L31E124 | 0.0667 | 0.0064 | 0.0128 | 1/6 |
| Qwen3-30B-A3B-Base | dir_opt | top-5 | L43E081 L41E117 L31E124 L44E122 L41E053 | 0.1219 | 0.0216 | 0.0273 | 1/6 |
| Qwen3-30B-A3B-Base | dir_opt | top-10 | L43E081 L41E117 L31E124 L44E122 L41E053 L43E114 L40E040 L39E071 L42E123 L41E062 | 0.1849 | -0.0056 | 0.0194 | 1/6 |
| Qwen3-30B-A3B-Base | f2e | top-1 | L37E107 | -0.0165 | -0.0264 | -0.0256 | 1/6 |
| Qwen3-30B-A3B-Base | f2e | top-3 | L37E107 L44E116 L43E036 | 0.0129 | -0.03 | -0.0278 | 1/6 |
| Qwen3-30B-A3B-Base | f2e | top-5 | L37E107 L44E116 L43E036 L42E055 L36E120 | 0.018 | -0.0252 | -0.0081 | 1/6 |
| Qwen3-30B-A3B-Base | f2e | top-10 | L37E107 L44E116 L43E036 L42E055 L36E120 L42E102 L44E022 L37E055 L41E121 L44E098 | 0.0295 | -0.0202 | 0.017 | 1/6 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | top-1 | L42E123 | 0.0294 | 0.0084 | 0.0114 | 1/6 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | top-3 | L42E123 L39E121 L45E092 | 0.0354 | 0.0089 | 0.0188 | 1/6 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | top-5 | L42E123 L39E121 L45E092 L41E106 L41E017 | 0.0246 | 0.014 | 0.0293 | 3/6 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | top-10 | L42E123 L39E121 L45E092 L41E106 L41E017 L37E101 L22E035 L39E001 L36E087 L43E042 | 0.056 | 0.0011 | 0.0197 | 1/6 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | top-1 | L42E123 | 0.0201 | -0.0039 | -0.0023 | 1/6 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | top-3 | L42E123 L39E121 L45E092 | 0.031 | 0.0046 | 0.0079 | 1/6 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | top-5 | L42E123 L39E121 L45E092 L44E044 L37E101 | 0.0372 | 0.0023 | 0.0154 | 1/6 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | top-10 | L42E123 L39E121 L45E092 L44E044 L37E101 L22E035 L23E105 L39E001 L43E097 L42E074 | 0.1613 | 0.0373 | 0.0541 | 1/6 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | top-1 | L37E032 | 0.0335 | 0.047 | 0.0651 | 5/6 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | top-3 | L37E032 L32E061 L39E103 | 0.2751 | -0.0145 | 0.0056 | 1/6 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | top-5 | L37E032 L32E061 L39E103 L42E031 L34E085 | 0.2881 | 0.0387 | 0.0874 | 1/6 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | top-10 | L37E032 L32E061 L39E103 L42E031 L34E085 L38E115 L35E090 L40E115 L36E065 L41E112 | 0.2881 | 0.0033 | 0.0511 | 1/6 |
| Qwen3-30B-A3B-Base | cp_vt_orig | top-1 | L42E123 | 0.0225 | -0.0017 | 0.0188 | 1/6 |
| Qwen3-30B-A3B-Base | cp_vt_orig | top-3 | L42E123 L39E001 L39E121 | 0.035 | 0.0111 | 0.023 | 1/6 |
| Qwen3-30B-A3B-Base | cp_vt_orig | top-5 | L42E123 L39E001 L39E121 L37E101 L22E035 | 0.072 | -0.0014 | 0.0267 | 1/6 |
| Qwen3-30B-A3B-Base | cp_vt_orig | top-10 | L42E123 L39E001 L39E121 L37E101 L22E035 L41E106 L45E092 L40E075 L44E122 L37E002 | 0.0969 | 0.0299 | 0.0438 | 1/6 |
| Qwen3-30B-A3B-Base | cp_vt_neu | top-1 | L42E123 | -0.0067 | -0.0117 | -0.0002 | 2/6 |
| Qwen3-30B-A3B-Base | cp_vt_neu | top-3 | L42E123 L39E121 L37E101 | 0.0295 | 0.007 | 0.0228 | 1/6 |
| Qwen3-30B-A3B-Base | cp_vt_neu | top-5 | L42E123 L39E121 L37E101 L41E106 L39E001 | 0.0427 | -0.0079 | 0.0186 | 1/6 |
| Qwen3-30B-A3B-Base | cp_vt_neu | top-10 | L42E123 L39E121 L37E101 L41E106 L39E001 L45E092 L22E035 L41E112 L25E002 L42E016 | 0.0583 | 0.0135 | 0.0539 | 1/6 |
| Qwen3-30B-A3B-Base | cp_nat | top-1 | L43E081 | 0.0188 | -0.0017 | 0.0014 | 1/6 |
| Qwen3-30B-A3B-Base | cp_nat | top-3 | L43E081 L41E117 L34E119 | 0.0839 | -0.0099 | -0.0023 | 1/6 |
| Qwen3-30B-A3B-Base | cp_nat | top-5 | L43E081 L41E117 L34E119 L44E122 L41E053 | 0.1188 | 0.0119 | 0.0339 | 1/6 |
| Qwen3-30B-A3B-Base | cp_nat | top-10 | L43E081 L41E117 L34E119 L44E122 L41E053 L42E073 L38E021 L42E123 L39E007 L39E071 | 0.2393 | 0.0296 | 0.0579 | 1/6 |
| Mixtral-8x7B (BOS) | f1 | top-1 | L19E004 | 0.0422 | 0.0027 | 0.0079 | 1/6 |
| Mixtral-8x7B (BOS) | f1 | top-3 | L19E004 L20E001 L20E005 | 0.0624 | -0.002 | 0.0 | 1/6 |
| Mixtral-8x7B (BOS) | f1 | top-5 | L19E004 L20E001 L20E005 L15E006 L17E005 | 0.19 | 0.0011 | 0.0139 | 1/6 |
| Mixtral-8x7B (BOS) | f1 | top-10 | L19E004 L20E001 L20E005 L15E006 L17E005 L17E007 L21E000 L18E002 L24E002 L18E000 | 0.2288 | -0.0018 | 0.0269 | 1/6 |
| Mixtral-8x7B (BOS) | dir_f1 | top-1 | L19E004 | 0.062 | -0.0045 | -0.002 | 1/6 |
| Mixtral-8x7B (BOS) | dir_f1 | top-3 | L19E004 L20E001 L23E000 | 0.0557 | -0.0066 | -0.002 | 1/6 |
| Mixtral-8x7B (BOS) | dir_f1 | top-5 | L19E004 L20E001 L23E000 L20E005 L16E007 | 0.1324 | -0.0306 | -0.0109 | 1/6 |
| Mixtral-8x7B (BOS) | dir_f1 | top-10 | L19E004 L20E001 L23E000 L20E005 L16E007 L24E002 L21E000 L18E000 L17E005 L15E006 | 0.243 | -0.0268 | -0.0014 | 1/6 |
| Mixtral-8x7B (BOS) | dir_opt | top-1 | L20E000 | 0.0251 | 0.0012 | 0.0073 | 1/6 |
| Mixtral-8x7B (BOS) | dir_opt | top-3 | L20E000 L19E006 L21E006 | 0.073 | -0.0014 | 0.0021 | 1/6 |
| Mixtral-8x7B (BOS) | dir_opt | top-5 | L20E000 L19E006 L21E006 L26E002 L18E005 | 0.1273 | 0.0031 | 0.0101 | 1/6 |
| Mixtral-8x7B (BOS) | dir_opt | top-10 | L20E000 L19E006 L21E006 L26E002 L18E005 L22E005 L13E005 L24E002 L21E004 L16E007 | 0.1711 | -0.0001 | 0.0095 | 1/6 |
| Mixtral-8x7B (BOS) | f2e | top-1 | L19E004 | 0.03 | 0.0076 | 0.01 | 1/6 |
| Mixtral-8x7B (BOS) | f2e | top-3 | L19E004 L17E005 L20E005 | 0.029 | 0.0068 | 0.025 | 1/6 |
| Mixtral-8x7B (BOS) | f2e | top-5 | L19E004 L17E005 L20E005 L21E000 L20E001 | 0.032 | 0.0088 | 0.021 | 1/6 |
| Mixtral-8x7B (BOS) | f2e | top-10 | L19E004 L17E005 L20E005 L21E000 L20E001 L23E007 L18E002 L24E002 L22E000 L17E001 | 0.017 | 0.0074 | 0.028 | 2/6 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | top-1 | L19E006 | 0.0422 | -0.0036 | 0.0117 | 1/6 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | top-3 | L19E006 L20E000 L21E003 | 0.0522 | 0.0066 | 0.0131 | 1/6 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | top-5 | L19E006 L20E000 L21E003 L25E002 L17E007 | 0.0602 | 0.0013 | 0.014 | 1/6 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | top-10 | L19E006 L20E000 L21E003 L25E002 L17E007 L14E007 L18E006 L24E004 L22E000 L23E000 | 0.0753 | 0.0141 | 0.0233 | 1/6 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | top-1 | L19E006 | 0.0368 | 0.0089 | 0.0202 | 1/6 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | top-3 | L19E006 L21E003 L20E000 | 0.0535 | 0.0169 | 0.0212 | 1/6 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | top-5 | L19E006 L21E003 L20E000 L23E000 L25E002 | 0.0599 | 0.0062 | 0.0162 | 1/6 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | top-10 | L19E006 L21E003 L20E000 L23E000 L25E002 L14E007 L18E006 L17E007 L27E005 L20E005 | 0.0581 | 0.0404 | 0.0649 | 2/6 |
| Mixtral-8x7B (BOS) | cp_vt_orig | top-1 | L19E006 | 0.0167 | -0.0013 | 0.0019 | 1/6 |
| Mixtral-8x7B (BOS) | cp_vt_orig | top-3 | L19E006 L21E003 L14E007 | 0.0758 | -0.0128 | -0.0004 | 1/6 |
| Mixtral-8x7B (BOS) | cp_vt_orig | top-5 | L19E006 L21E003 L14E007 L20E000 L18E006 | 0.0827 | 0.0159 | 0.0276 | 1/6 |
| Mixtral-8x7B (BOS) | cp_vt_orig | top-10 | L19E006 L21E003 L14E007 L20E000 L18E006 L23E007 L19E004 L20E005 L17E007 L23E000 | 0.0871 | -0.0017 | 0.0196 | 1/6 |
| Mixtral-8x7B (BOS) | cp_vt_neu | top-1 | L21E003 | 0.0034 | -0.0006 | 0.0025 | 1/6 |
| Mixtral-8x7B (BOS) | cp_vt_neu | top-3 | L21E003 L19E006 L20E005 | 0.0466 | -0.0008 | 0.0061 | 1/6 |
| Mixtral-8x7B (BOS) | cp_vt_neu | top-5 | L21E003 L19E006 L20E005 L14E007 L23E000 | 0.07 | -0.0096 | 0.0141 | 1/6 |
| Mixtral-8x7B (BOS) | cp_vt_neu | top-10 | L21E003 L19E006 L20E005 L14E007 L23E000 L19E004 L13E001 L23E007 L20E000 L18E006 | 0.0448 | 0.0125 | 0.0468 | 2/6 |
| Mixtral-8x7B (BOS) | cp_nat | top-1 | L20E000 | 0.0542 | 0.0015 | 0.0052 | 1/6 |
| Mixtral-8x7B (BOS) | cp_nat | top-3 | L20E000 L19E006 L21E006 | 0.1213 | 0.0054 | 0.0089 | 1/6 |
| Mixtral-8x7B (BOS) | cp_nat | top-5 | L20E000 L19E006 L21E006 L26E002 L16E007 | 0.1683 | 0.0036 | 0.0115 | 1/6 |
| Mixtral-8x7B (BOS) | cp_nat | top-10 | L20E000 L19E006 L21E006 L26E002 L16E007 L18E005 L24E002 L21E004 L22E005 L22E000 | 0.2138 | 0.0087 | 0.0222 | 1/6 |


### 4. P5-8: copy vs generate, and direction

Contrasts of F between conditions (a − b). C1 = entity → attribute, answer generated (dir_opt, vt_e2a_*); C2 = the same items with a prefix that names both attributes (cp_*), answer copyable; C3 = attribute → entity (fill-in-the-blank; the entity is copyable from the sentence). C2 − C1 isolates copy availability at fixed direction and format; C3 − C2 isolates direction at fixed copy availability (the fill-in wrapper still differs).

| model | contrast | expert(s) | a | b | F a | F b | a - b [95% CI] | paired: source items (units) | paired F a / F b | paired a - b [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | copy availability (templates, orig verb): C2 - C1 | L41E117 | cp_vt_orig | vt_e2a_orig | -0.0013 | 0.0014 | -0.003 [-0.016, +0.010] | 28 (8) | +0.002 / -0.010 | +0.012 [-0.005, +0.038] |
| Qwen3-30B-A3B-Base | copy availability (templates, orig verb): C2 - C1 | L39E121 | cp_vt_orig | vt_e2a_orig | -0.0005 | 0.005 | -0.005 [-0.039, +0.032] | 28 (8) | -0.021 / -0.019 | -0.002 [-0.022, +0.043] |
| Qwen3-30B-A3B-Base | copy availability (templates, orig verb): C2 - C1 | own top-10 | cp_vt_orig | vt_e2a_orig | 0.0969 | 0.056 | +0.041 [-0.097, +0.177] | 28 (8) | +0.067 / +0.043 | +0.023 [-0.119, +0.207] |
| Qwen3-30B-A3B-Base | copy availability (templates, neutral verb): C2 - C1 | L41E117 | cp_vt_neu | vt_e2a_neu | 0.0002 | 0.0011 | -0.001 [-0.015, +0.013] | 25 (4) | -0.006 / -0.005 | -0.001 [-0.030, +0.033] |
| Qwen3-30B-A3B-Base | copy availability (templates, neutral verb): C2 - C1 | L42E123 | cp_vt_neu | vt_e2a_neu | -0.0067 | 0.0201 | -0.027 [-0.056, +0.004] | 25 (4) | -0.005 / +0.025 | -0.031 [-0.058, -0.008] |
| Qwen3-30B-A3B-Base | copy availability (templates, neutral verb): C2 - C1 | own top-10 | cp_vt_neu | vt_e2a_neu | 0.0583 | 0.1613 | -0.103 [-0.192, -0.017] | 25 (4) | +0.048 / +0.129 | -0.081 [-0.126, -0.029] |
| Qwen3-30B-A3B-Base | copy availability (natural 776 pairs): C2 - C1 | L41E117 | cp_nat | dir_opt | 0.0267 | 0.0254 | +0.001 [-0.016, +0.017] | 29 (29) | +0.023 / +0.033 | -0.010 [-0.041, +0.020] |
| Qwen3-30B-A3B-Base | copy availability (natural 776 pairs): C2 - C1 | own top-10 | cp_nat | dir_opt | 0.2035 | 0.1849 | +0.018 [-0.019, +0.056] | 29 (29) | +0.181 / +0.147 | +0.034 [-0.012, +0.084] |
| Qwen3-30B-A3B-Base | direction (templates, orig; copy fixed): C3 - C2 | L41E117 | vt_a2e_orig | cp_vt_orig | -0.0093 | -0.0013 | -0.008 [-0.030, +0.000] |  |  |  |
| Qwen3-30B-A3B-Base | direction (templates, orig; copy fixed): C3 - C2 | L39E001 | vt_a2e_orig | cp_vt_orig | -0.0177 | 0.0084 | -0.026 [-0.062, +0.024] |  |  |  |
| Qwen3-30B-A3B-Base | direction (templates, orig; copy fixed): C3 - C2 | own top-10 | vt_a2e_orig | cp_vt_orig | 0.2881 | 0.0969 | +0.191 [+0.084, +0.366] |  |  |  |
| Qwen3-30B-A3B-Base | direction (natural 776 pairs; copy fixed): C3 - C2 | L41E117 | dir_f1 | cp_nat | 0.0011 | 0.0267 | -0.026 [-0.038, -0.013] | 29 (29) | -0.001 / +0.023 | -0.024 [-0.052, +0.004] |
| Qwen3-30B-A3B-Base | direction (natural 776 pairs; copy fixed): C3 - C2 | own top-10 | dir_f1 | cp_nat | 0.1818 | 0.2035 | -0.022 [-0.064, +0.021] | 29 (29) | +0.097 / +0.181 | -0.084 [-0.149, -0.019] |
| Qwen3-30B-A3B-Base | direction on identical items (fill-in a→e - option swap e→a) | L41E117 | dir_f1 | dir_opt | 0.0011 | 0.0254 | -0.024 [-0.042, -0.008] | 90 (90) | +0.001 / +0.025 | -0.024 [-0.042, -0.007] |
| Qwen3-30B-A3B-Base | direction on identical items (fill-in a→e - option swap e→a) | own top-10 | dir_f1 | dir_opt | 0.1818 | 0.1849 | -0.003 [-0.040, +0.034] | 90 (90) | +0.182 / +0.185 | -0.003 [-0.040, +0.032] |
| Qwen3-30B-A3B-Base | direction (templates, orig): a→e - e→a | L41E117 | vt_a2e_orig | vt_e2a_orig | -0.0093 | 0.0014 | -0.011 [-0.034, -0.000] |  |  |  |
| Qwen3-30B-A3B-Base | direction (templates, orig): a→e - e→a | L39E121 | vt_a2e_orig | vt_e2a_orig | -0.0548 | 0.005 | -0.060 [-0.085, +0.053] |  |  |  |
| Qwen3-30B-A3B-Base | direction (templates, orig): a→e - e→a | own top-10 | vt_a2e_orig | vt_e2a_orig | 0.2881 | 0.056 | +0.232 [+0.153, +0.390] |  |  |  |
| Qwen3-30B-A3B-Base | verb: neutral - orig (e→a) | L41E117 | vt_e2a_neu | vt_e2a_orig | 0.0011 | 0.0014 | -0.000 [-0.015, +0.015] |  |  |  |
| Qwen3-30B-A3B-Base | verb: neutral - orig (e→a) | L39E121 | vt_e2a_neu | vt_e2a_orig | 0.0113 | 0.005 | +0.006 [-0.036, +0.057] |  |  |  |
| Qwen3-30B-A3B-Base | verb: neutral - orig (e→a) | own top-10 | vt_e2a_neu | vt_e2a_orig | 0.1613 | 0.056 | +0.105 [-0.011, +0.220] |  |  |  |
| Mixtral-8x7B (BOS) | copy availability (templates, orig verb): C2 - C1 | L20E000 | cp_vt_orig | vt_e2a_orig | 0.0202 | 0.0117 | +0.009 [-0.037, +0.059] | 35 (8) | +0.025 / -0.004 | +0.029 [-0.000, +0.057] |
| Mixtral-8x7B (BOS) | copy availability (templates, orig verb): C2 - C1 | L19E006 | cp_vt_orig | vt_e2a_orig | 0.0167 | 0.0422 | -0.025 [-0.053, +0.001] | 35 (8) | +0.020 / +0.043 | -0.022 [-0.055, +0.004] |
| Mixtral-8x7B (BOS) | copy availability (templates, orig verb): C2 - C1 | own top-10 | cp_vt_orig | vt_e2a_orig | 0.0871 | 0.0753 | +0.012 [-0.096, +0.136] | 35 (8) | +0.108 / +0.060 | +0.048 [-0.035, +0.140] |
| Mixtral-8x7B (BOS) | copy availability (templates, neutral verb): C2 - C1 | L20E000 | cp_vt_neu | vt_e2a_neu | -0.0009 | 0.0231 | -0.024 [-0.060, +0.023] | 31 (5) | -0.000 / +0.004 | -0.004 [-0.008, +0.015] |
| Mixtral-8x7B (BOS) | copy availability (templates, neutral verb): C2 - C1 | L19E006 | cp_vt_neu | vt_e2a_neu | 0.0345 | 0.0368 | -0.002 [-0.039, +0.042] | 31 (5) | +0.032 / +0.035 | -0.003 [-0.028, +0.011] |
| Mixtral-8x7B (BOS) | copy availability (templates, neutral verb): C2 - C1 | own top-10 | cp_vt_neu | vt_e2a_neu | 0.0448 | 0.0581 | -0.013 [-0.129, +0.099] | 31 (5) | +0.048 / +0.044 | +0.004 [-0.074, +0.066] |
| Mixtral-8x7B (BOS) | copy availability (natural 776 pairs): C2 - C1 | L20E000 | cp_nat | dir_opt | 0.0446 | 0.0251 | +0.019 [+0.009, +0.030] | 30 (30) | +0.032 / +0.036 | -0.003 [-0.014, +0.007] |
| Mixtral-8x7B (BOS) | copy availability (natural 776 pairs): C2 - C1 | own top-10 | cp_nat | dir_opt | 0.1846 | 0.1711 | +0.013 [-0.013, +0.040] | 30 (30) | +0.149 / +0.162 | -0.013 [-0.036, +0.009] |
| Mixtral-8x7B (BOS) | direction (natural 776 pairs; copy fixed): C3 - C2 | L20E000 | dir_f1 | cp_nat | 0.002 | 0.0446 | -0.043 [-0.051, -0.035] | 30 (30) | +0.008 / +0.032 | -0.024 [-0.047, -0.003] |
| Mixtral-8x7B (BOS) | direction (natural 776 pairs; copy fixed): C3 - C2 | L21E006 | dir_f1 | cp_nat | 0.0114 | 0.023 | -0.012 [-0.023, +0.000] | 30 (30) | -0.005 / +0.015 | -0.020 [-0.040, -0.001] |
| Mixtral-8x7B (BOS) | direction (natural 776 pairs; copy fixed): C3 - C2 | own top-10 | dir_f1 | cp_nat | 0.243 | 0.1846 | +0.058 [+0.012, +0.104] | 30 (30) | +0.255 / +0.149 | +0.106 [+0.036, +0.173] |
| Mixtral-8x7B (BOS) | direction on identical items (fill-in a→e - option swap e→a) | L20E000 | dir_f1 | dir_opt | 0.002 | 0.0251 | -0.023 [-0.035, -0.012] | 94 (94) | +0.002 / +0.025 | -0.023 [-0.034, -0.012] |
| Mixtral-8x7B (BOS) | direction on identical items (fill-in a→e - option swap e→a) | own top-10 | dir_f1 | dir_opt | 0.243 | 0.1711 | +0.072 [+0.028, +0.115] | 94 (94) | +0.243 / +0.171 | +0.072 [+0.029, +0.114] |
| Mixtral-8x7B (BOS) | verb: neutral - orig (e→a) | L20E000 | vt_e2a_neu | vt_e2a_orig | 0.0231 | 0.0117 | +0.011 [-0.031, +0.054] |  |  |  |
| Mixtral-8x7B (BOS) | verb: neutral - orig (e→a) | L19E006 | vt_e2a_neu | vt_e2a_orig | 0.0368 | 0.0422 | -0.005 [-0.036, +0.027] |  |  |  |
| Mixtral-8x7B (BOS) | verb: neutral - orig (e→a) | own top-10 | vt_e2a_neu | vt_e2a_orig | 0.0581 | 0.0753 | -0.017 [-0.088, +0.055] |  |  |  |


**Reading (P5-8, WinoGrande expert).** Qwen3-30B-A3B-Base L41E117: cp_vt_orig F +0.002 vs vt_e2a_orig -0.010, difference +0.012 [-0.005, +0.038] (paired, shared source items) -> unchanged within the CI; Qwen3-30B-A3B-Base L41E117: cp_vt_neu F -0.006 vs vt_e2a_neu -0.005, difference -0.001 [-0.030, +0.033] (paired, shared source items) -> unchanged within the CI; Qwen3-30B-A3B-Base L41E117: cp_nat F +0.023 vs dir_opt +0.033, difference -0.010 [-0.041, +0.020] (paired, shared source items) -> unchanged within the CI; Mixtral-8x7B (BOS) L20E000: cp_vt_orig F +0.025 vs vt_e2a_orig -0.004, difference +0.029 [-0.000, +0.057] (paired, shared source items) -> unchanged within the CI; Mixtral-8x7B (BOS) L20E000: cp_vt_neu F -0.000 vs vt_e2a_neu +0.004, difference -0.004 [-0.008, +0.015] (paired, shared source items) -> unchanged within the CI; Mixtral-8x7B (BOS) L20E000: cp_nat F +0.032 vs dir_opt +0.036, difference -0.003 [-0.014, +0.007] (paired, shared source items) -> unchanged within the CI. If F stays at the e → a level the expert is still necessary although the answer token is in the prefix (the attribute is still computed by the MoE); if it falls to about zero the copy route replaces it.


**Reading (direction).** Qwen3-30B-A3B-Base L41E117: vt_a2e_orig F -0.009 vs cp_vt_orig -0.001, difference -0.008 [-0.030, +0.000] (unpaired) -> unchanged within the CI; Qwen3-30B-A3B-Base L41E117: dir_f1 F -0.001 vs cp_nat +0.023, difference -0.024 [-0.052, +0.004] (paired, shared source items) -> unchanged within the CI; Qwen3-30B-A3B-Base L41E117: dir_f1 F +0.001 vs dir_opt +0.025, difference -0.024 [-0.042, -0.007] (paired, shared source items) -> reduced to about zero; Qwen3-30B-A3B-Base L41E117: vt_a2e_orig F -0.009 vs vt_e2a_orig +0.001, difference -0.011 [-0.034, -0.000] (unpaired) -> unchanged within the CI; Mixtral-8x7B (BOS) L20E000: dir_f1 F +0.008 vs cp_nat +0.032, difference -0.024 [-0.047, -0.003] (paired, shared source items) -> reduced to about zero; Mixtral-8x7B (BOS) L20E000: dir_f1 F +0.002 vs dir_opt +0.025, difference -0.023 [-0.034, -0.012] (paired, shared source items) -> reduced to about zero.


### 5. Final position only, and which expert takes the slot

| model | condition | knockout | all positions | final position only | final / all |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1 | L37E032 | 0.0484 | 0.0334 | 0.69 |
| Qwen3-30B-A3B-Base | f1 | L41E117 | -0.0055 | 0.0097 | -1.7692 |
| Qwen3-30B-A3B-Base | f1 | L44E069 | 0.0019 | -0.0002 | -0.1111 |
| Qwen3-30B-A3B-Base | f1 | L42E115 | -0.008 | -0.0013 | 0.1579 |
| Qwen3-30B-A3B-Base | f1 | pop:k10 | 0.1948 | 0.1643 | 0.8436 |
| Qwen3-30B-A3B-Base | dir_f1 | L44E116 | 0.0213 | 0.02 | 0.9375 |
| Qwen3-30B-A3B-Base | dir_f1 | L41E117 | 0.0011 | -0.004 | -3.75 |
| Qwen3-30B-A3B-Base | dir_f1 | L44E069 | -0.0013 | -0.0013 | 1.0 |
| Qwen3-30B-A3B-Base | dir_f1 | L42E115 | -0.0061 | -0.0048 | 0.7826 |
| Qwen3-30B-A3B-Base | dir_f1 | pop:k10 | 0.1818 | 0.1106 | 0.6085 |
| Qwen3-30B-A3B-Base | dir_opt | L41E117 | 0.0254 | 0.0353 | 1.3863 |
| Qwen3-30B-A3B-Base | dir_opt | L44E069 | -0.0025 | 0.0111 | -4.375 |
| Qwen3-30B-A3B-Base | dir_opt | L42E115 | 0.0005 | 0.0101 | 21.3333 |
| Qwen3-30B-A3B-Base | dir_opt | pop:k10 | 0.1849 | 0.1612 | 0.8718 |
| Qwen3-30B-A3B-Base | f2e | L43E036 | 0.0146 | -0.0187 | -1.2787 |
| Qwen3-30B-A3B-Base | f2e | L41E117 | 0.0041 | -0.0316 | -7.7647 |
| Qwen3-30B-A3B-Base | f2e | L44E069 | 0.0007 | -0.0319 | -44.3333 |
| Qwen3-30B-A3B-Base | f2e | L42E115 | 0.0084 | -0.0201 | -2.4 |
| Qwen3-30B-A3B-Base | f2e | pop:k10 | 0.0295 | -0.0323 | -1.0976 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L39E121 | 0.005 | 0.0132 | 2.6418 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L41E117 | 0.0014 | -0.0055 | -3.8421 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L44E069 | 0.0101 | 0.0089 | 0.8815 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L42E115 | -0.0008 | -0.0046 | 5.6364 |
| Qwen3-30B-A3B-Base | vt_e2a_orig | pop:k10 | 0.056 | 0.0587 | 1.0481 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L42E123 | 0.0201 | 0.0061 | 0.3012 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L41E117 | 0.0011 | 0.0013 | 1.1429 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L44E069 | 0.009 | 0.0054 | 0.6036 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L42E115 | -0.0023 | -0.0088 | 3.8929 |
| Qwen3-30B-A3B-Base | vt_e2a_neu | pop:k10 | 0.1613 | 0.0756 | 0.4686 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L37E032 | 0.0335 | 0.0446 | 1.3333 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L41E117 | -0.0093 | -0.0009 | 0.1 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L44E069 | -0.0009 | -0.0112 | 12.0 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L42E115 | 0.0028 | -0.0019 | -0.6667 |
| Qwen3-30B-A3B-Base | vt_a2e_orig | pop:k10 | 0.2881 | 0.0446 | 0.1548 |
| Qwen3-30B-A3B-Base | cp_vt_orig | L39E001 | 0.0084 | 0.0239 | 2.8542 |
| Qwen3-30B-A3B-Base | cp_vt_orig | L41E117 | -0.0013 | 0.0035 | -2.6667 |
| Qwen3-30B-A3B-Base | cp_vt_orig | L44E069 | 0.0047 | 0.011 | 2.3333 |
| Qwen3-30B-A3B-Base | cp_vt_orig | L42E115 | 0.0017 | 0.0077 | 4.4 |
| Qwen3-30B-A3B-Base | cp_vt_orig | pop:k10 | 0.0969 | 0.0875 | 0.9029 |
| Qwen3-30B-A3B-Base | cp_vt_neu | L42E123 | -0.0067 | -0.0237 | 3.5526 |
| Qwen3-30B-A3B-Base | cp_vt_neu | L41E117 | 0.0002 | -0.0061 | -35.0 |
| Qwen3-30B-A3B-Base | cp_vt_neu | L44E069 | -0.0002 | -0.0118 | 67.0 |
| Qwen3-30B-A3B-Base | cp_vt_neu | L42E115 | -0.0042 | -0.0137 | 3.25 |
| Qwen3-30B-A3B-Base | cp_vt_neu | pop:k10 | 0.0583 | 0.0409 | 0.7018 |
| Qwen3-30B-A3B-Base | cp_nat | L41E117 | 0.0344 | 0.0176 | 0.5109 |
| Qwen3-30B-A3B-Base | cp_nat | L44E069 | -0.0026 | -0.0069 | 2.6286 |
| Qwen3-30B-A3B-Base | cp_nat | L42E115 | -0.0074 | -0.0063 | 0.8571 |
| Qwen3-30B-A3B-Base | cp_nat | pop:k10 | 0.2393 | 0.1781 | 0.744 |
| Mixtral-8x7B (BOS) | f1 | L20E001 | 0.0079 | 0.0128 | 1.6071 |
| Mixtral-8x7B (BOS) | f1 | L20E000 | 0.0051 | 0.0006 | 0.1111 |
| Mixtral-8x7B (BOS) | f1 | L19E002 | 0.0026 | -0.0014 | -0.5556 |
| Mixtral-8x7B (BOS) | f1 | L21E001 | 0.0082 | 0.0006 | 0.069 |
| Mixtral-8x7B (BOS) | f1 | L18E001 | 0.0028 | -0.0003 | -0.1 |
| Mixtral-8x7B (BOS) | f1 | pop:k10 | 0.2288 | 0.2189 | 0.9566 |
| Mixtral-8x7B (BOS) | dir_f1 | L20E001 | -0.0046 | 0.006 | -1.3125 |
| Mixtral-8x7B (BOS) | dir_f1 | L20E000 | 0.002 | -0.0051 | -2.5714 |
| Mixtral-8x7B (BOS) | dir_f1 | L19E002 | 0.0011 | -0.0034 | -3.0 |
| Mixtral-8x7B (BOS) | dir_f1 | L21E001 | -0.0014 | -0.0037 | 2.6 |
| Mixtral-8x7B (BOS) | dir_f1 | L18E001 | 0.0 | -0.0049 |  |
| Mixtral-8x7B (BOS) | dir_f1 | pop:k10 | 0.243 | 0.2541 | 1.0459 |
| Mixtral-8x7B (BOS) | dir_opt | L20E000 | 0.0251 | 0.0231 | 0.9231 |
| Mixtral-8x7B (BOS) | dir_opt | L19E002 | -0.0011 | -0.0017 | 1.5 |
| Mixtral-8x7B (BOS) | dir_opt | L21E001 | 0.0004 | -0.0015 | -3.6667 |
| Mixtral-8x7B (BOS) | dir_opt | L18E001 | -0.0003 | -0.0011 | 4.0 |
| Mixtral-8x7B (BOS) | dir_opt | pop:k10 | 0.1711 | 0.1478 | 0.8639 |
| Mixtral-8x7B (BOS) | f2e | L19E004 | 0.03 | 0.018 | 0.6 |
| Mixtral-8x7B (BOS) | f2e | L20E000 | 0.0 | 0.009 |  |
| Mixtral-8x7B (BOS) | f2e | L19E002 | -0.001 | 0.006 | -6.0 |
| Mixtral-8x7B (BOS) | f2e | L21E001 | 0.01 | 0.009 | 0.9 |
| Mixtral-8x7B (BOS) | f2e | L18E001 | -0.008 | 0.004 | -0.5 |
| Mixtral-8x7B (BOS) | f2e | pop:k10 | 0.017 | 0.003 | 0.1765 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L19E006 | 0.0422 | 0.0297 | 0.7046 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L20E000 | 0.0117 | 0.0096 | 0.8205 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L19E002 | -0.0065 | -0.0084 | 1.3023 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L21E001 | 0.0009 | 0.0006 | 0.6667 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L18E001 | 0.0015 | -0.0033 | -2.2 |
| Mixtral-8x7B (BOS) | vt_e2a_orig | pop:k10 | 0.0753 | 0.0516 | 0.6853 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L19E006 | 0.0368 | 0.0248 | 0.6727 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L20E000 | 0.0231 | 0.0226 | 0.9783 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L19E002 | 0.0037 | 0.0038 | 1.0455 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L21E001 | 0.004 | 0.0055 | 1.375 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L18E001 | 0.001 | 0.0017 | 1.6667 |
| Mixtral-8x7B (BOS) | vt_e2a_neu | pop:k10 | 0.0581 | 0.0356 | 0.6138 |
| Mixtral-8x7B (BOS) | cp_vt_orig | L19E006 | 0.0167 | 0.0122 | 0.7333 |
| Mixtral-8x7B (BOS) | cp_vt_orig | L20E000 | 0.0202 | 0.0191 | 0.945 |
| Mixtral-8x7B (BOS) | cp_vt_orig | L19E002 | -0.0019 | -0.0048 | 2.6 |
| Mixtral-8x7B (BOS) | cp_vt_orig | L21E001 | 0.0032 | 0.0013 | 0.4118 |
| Mixtral-8x7B (BOS) | cp_vt_orig | L18E001 | 0.0052 | -0.0013 | -0.25 |
| Mixtral-8x7B (BOS) | cp_vt_orig | pop:k10 | 0.0871 | 0.0849 | 0.9745 |
| Mixtral-8x7B (BOS) | cp_vt_neu | L19E006 | 0.0345 | 0.0258 | 0.7487 |
| Mixtral-8x7B (BOS) | cp_vt_neu | L20E000 | -0.0009 | 0.0045 | -5.0 |
| Mixtral-8x7B (BOS) | cp_vt_neu | L19E002 | 0.0023 | -0.0011 | -0.4615 |
| Mixtral-8x7B (BOS) | cp_vt_neu | L21E001 | 0.0128 | 0.0083 | 0.6479 |
| Mixtral-8x7B (BOS) | cp_vt_neu | L18E001 | 0.0031 | 0.0004 | 0.1176 |
| Mixtral-8x7B (BOS) | cp_vt_neu | pop:k10 | 0.0448 | 0.0163 | 0.3629 |
| Mixtral-8x7B (BOS) | cp_nat | L21E006 | 0.0258 | 0.0182 | 0.7076 |
| Mixtral-8x7B (BOS) | cp_nat | L20E000 | 0.0542 | 0.0493 | 0.91 |
| Mixtral-8x7B (BOS) | cp_nat | L19E002 | -0.0001 | -0.0001 | 1.0 |
| Mixtral-8x7B (BOS) | cp_nat | L21E001 | 0.0006 | -0.0009 | -1.5714 |
| Mixtral-8x7B (BOS) | cp_nat | L18E001 | 0.0022 | -0.0009 | -0.3846 |
| Mixtral-8x7B (BOS) | cp_nat | pop:k10 | 0.2138 | 0.1786 | 0.8353 |


| model | condition | expert | routed at final | clean one-for-one swap | top replacements (share) |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1 | L37E032 | 0.9818 | 0.9721 | L37E104 (0.1061), L37E081 (0.0955), L37E015 (0.0955) |
| Qwen3-30B-A3B-Base | f1 | L41E117 | 0.0221 | 1.0 | L41E018 (0.2941), L41E112 (0.1765), L41E121 (0.1765) |
| Qwen3-30B-A3B-Base | f1 | L44E069 | 0.0924 | 0.9577 | L44E027 (0.1268), L44E081 (0.1127), L44E006 (0.0845) |
| Qwen3-30B-A3B-Base | f1 | L42E115 | 0.3568 | 0.9854 | L42E021 (0.1825), L42E030 (0.1204), L42E092 (0.1131) |
| Qwen3-30B-A3B-Base | dir_f1 | L44E116 | 0.85 | 0.9739 | L44E047 (0.2092), L44E068 (0.0784), L44E001 (0.0588) |
| Qwen3-30B-A3B-Base | dir_f1 | L41E117 | 0.0 | 0.0 |  |
| Qwen3-30B-A3B-Base | dir_f1 | L44E069 | 0.0833 | 1.0 | L44E044 (0.2), L44E047 (0.1333), L44E017 (0.1333) |
| Qwen3-30B-A3B-Base | dir_f1 | L42E115 | 0.1333 | 0.9583 | L42E016 (0.125), L42E030 (0.125), L42E123 (0.125) |
| Qwen3-30B-A3B-Base | dir_opt | L41E117 | 0.7722 | 0.9856 | L41E121 (0.1511), L41E011 (0.0863), L41E079 (0.0863) |
| Qwen3-30B-A3B-Base | dir_opt | L44E069 | 0.0111 | 1.0 | L44E020 (0.5), L44E022 (0.5) |
| Qwen3-30B-A3B-Base | dir_opt | L42E115 | 0.0111 | 1.0 | L42E093 (0.5), L42E080 (0.5) |
| Qwen3-30B-A3B-Base | f2e | L43E036 | 0.9844 | 0.9524 | L43E042 (0.0794), L43E081 (0.0794), L43E104 (0.0794) |
| Qwen3-30B-A3B-Base | f2e | L41E117 | 0.0625 | 1.0 | L41E112 (0.25), L41E106 (0.25), L41E045 (0.25) |
| Qwen3-30B-A3B-Base | f2e | L44E069 | 0.4375 | 1.0 | L44E027 (0.1429), L44E003 (0.1429), L44E056 (0.1429) |
| Qwen3-30B-A3B-Base | f2e | L42E115 | 0.625 | 1.0 | L42E016 (0.175), L42E123 (0.175), L42E071 (0.075) |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L39E121 | 0.7188 | 0.9891 | L39E086 (0.0815), L39E003 (0.0761), L39E055 (0.0652) |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L41E117 | 0.4648 | 1.0 | L41E018 (0.2269), L41E126 (0.1849), L41E112 (0.1345) |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L44E069 | 0.2383 | 0.9344 | L44E006 (0.2131), L44E022 (0.1803), L44E011 (0.0984) |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L42E115 | 0.0781 | 1.0 | L42E027 (0.35), L42E023 (0.3), L42E002 (0.15) |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L42E123 | 1.0 | 1.0 | L42E023 (0.1953), L42E055 (0.1289), L42E074 (0.0859) |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L41E117 | 0.3867 | 1.0 | L41E018 (0.202), L41E112 (0.1818), L41E079 (0.1717) |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L44E069 | 0.2031 | 0.8846 | L44E006 (0.4038), L44E118 (0.1346), L44E003 (0.0577) |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L42E115 | 0.0938 | 1.0 | L42E003 (0.4583), L42E023 (0.3333), L42E027 (0.0833) |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L37E032 | 1.0 | 1.0 | L37E073 (0.7027), L37E091 (0.1216), L37E110 (0.0541) |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L41E117 | 0.0 | 0.0 |  |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L44E069 | 0.0 | 0.0 |  |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L42E115 | 0.6486 | 0.9792 | L42E021 (0.6667), L42E006 (0.1042), L42E016 (0.0625) |
| Qwen3-30B-A3B-Base | cp_vt_orig | L39E001 | 0.7109 | 0.978 | L39E007 (0.2363), L39E095 (0.1099), L39E059 (0.0934) |
| Qwen3-30B-A3B-Base | cp_vt_orig | L41E117 | 0.4258 | 1.0 | L41E018 (0.3028), L41E112 (0.2844), L41E009 (0.1101) |
| Qwen3-30B-A3B-Base | cp_vt_orig | L44E069 | 0.3828 | 0.9694 | L44E022 (0.3265), L44E030 (0.1531), L44E027 (0.102) |
| Qwen3-30B-A3B-Base | cp_vt_orig | L42E115 | 0.2148 | 1.0 | L42E002 (0.2364), L42E023 (0.1818), L42E093 (0.1091) |
| Qwen3-30B-A3B-Base | cp_vt_neu | L42E123 | 1.0 | 0.9883 | L42E068 (0.2305), L42E002 (0.1289), L42E032 (0.125) |
| Qwen3-30B-A3B-Base | cp_vt_neu | L41E117 | 0.543 | 1.0 | L41E018 (0.2878), L41E112 (0.1942), L41E079 (0.1583) |
| Qwen3-30B-A3B-Base | cp_vt_neu | L44E069 | 0.2617 | 1.0 | L44E022 (0.2388), L44E027 (0.1642), L44E008 (0.1642) |
| Qwen3-30B-A3B-Base | cp_vt_neu | L42E115 | 0.1367 | 1.0 | L42E093 (0.3429), L42E023 (0.2571), L42E002 (0.1714) |
| Qwen3-30B-A3B-Base | cp_nat | L41E117 | 0.8924 | 0.9805 | L41E079 (0.1571), L41E011 (0.1255), L41E018 (0.1145) |
| Qwen3-30B-A3B-Base | cp_nat | L44E069 | 0.012 | 0.9091 | L44E006 (0.2727), L44E027 (0.1818), L44E080 (0.1818) |
| Qwen3-30B-A3B-Base | cp_nat | L42E115 | 0.0043 | 1.0 | L42E123 (0.75), L42E032 (0.25) |
| Mixtral-8x7B (BOS) | f1 | L20E001 | 0.7426 | 1.0 | L20E006 (0.4333), L20E000 (0.3933), L20E005 (0.1067) |
| Mixtral-8x7B (BOS) | f1 | L20E000 | 0.1881 | 1.0 | L20E005 (0.5263), L20E001 (0.2632), L20E003 (0.1316) |
| Mixtral-8x7B (BOS) | f1 | L19E002 | 0.0396 | 1.0 | L19E006 (0.5), L19E004 (0.25), L19E007 (0.25) |
| Mixtral-8x7B (BOS) | f1 | L21E001 | 0.3515 | 1.0 | L21E005 (0.338), L21E004 (0.2958), L21E007 (0.2113) |
| Mixtral-8x7B (BOS) | f1 | L18E001 | 0.1188 | 1.0 | L18E000 (0.5417), L18E005 (0.3333), L18E006 (0.125) |
| Mixtral-8x7B (BOS) | dir_f1 | L20E001 | 0.8777 | 1.0 | L20E006 (0.4424), L20E000 (0.3939), L20E003 (0.0727) |
| Mixtral-8x7B (BOS) | dir_f1 | L20E000 | 0.0745 | 1.0 | L20E005 (0.5), L20E001 (0.2857), L20E003 (0.1429) |
| Mixtral-8x7B (BOS) | dir_f1 | L19E002 | 0.0106 | 1.0 | L19E006 (1.0) |
| Mixtral-8x7B (BOS) | dir_f1 | L21E001 | 0.3723 | 1.0 | L21E005 (0.3143), L21E004 (0.3), L21E007 (0.2714) |
| Mixtral-8x7B (BOS) | dir_f1 | L18E001 | 0.0106 | 1.0 | L18E005 (0.5), L18E000 (0.5) |
| Mixtral-8x7B (BOS) | dir_opt | L20E000 | 0.984 | 1.0 | L20E007 (0.5676), L20E005 (0.1676), L20E001 (0.1405) |
| Mixtral-8x7B (BOS) | dir_opt | L19E002 | 0.0053 | 1.0 | L19E003 (1.0) |
| Mixtral-8x7B (BOS) | dir_opt | L21E001 | 0.0213 | 1.0 | L21E004 (0.5), L21E005 (0.5) |
| Mixtral-8x7B (BOS) | dir_opt | L18E001 | 0.0 | 0.0 |  |
| Mixtral-8x7B (BOS) | f2e | L19E004 | 1.0 | 1.0 | L19E001 (0.6389), L19E007 (0.1667), L19E000 (0.1111) |
| Mixtral-8x7B (BOS) | f2e | L20E000 | 0.2778 | 1.0 | L20E005 (0.7), L20E001 (0.2), L20E006 (0.1) |
| Mixtral-8x7B (BOS) | f2e | L19E002 | 0.0 | 0.0 |  |
| Mixtral-8x7B (BOS) | f2e | L21E001 | 0.4444 | 1.0 | L21E005 (0.25), L21E007 (0.25), L21E000 (0.1875) |
| Mixtral-8x7B (BOS) | f2e | L18E001 | 0.1111 | 1.0 | L18E006 (0.5), L18E003 (0.5) |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L19E006 | 0.9375 | 1.0 | L19E000 (0.375), L19E004 (0.1875), L19E003 (0.1625) |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L20E000 | 0.9297 | 1.0 | L20E001 (0.2689), L20E005 (0.1597), L20E004 (0.1387) |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L19E002 | 0.1875 | 1.0 | L19E000 (1.0) |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L21E001 | 0.2617 | 1.0 | L21E004 (0.5672), L21E005 (0.2687), L21E000 (0.0896) |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L18E001 | 0.0781 | 1.0 | L18E005 (0.95), L18E000 (0.05) |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L19E006 | 0.9062 | 1.0 | L19E002 (0.3103), L19E001 (0.2155), L19E004 (0.1897) |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L20E000 | 0.8594 | 1.0 | L20E005 (0.3545), L20E001 (0.2864), L20E004 (0.1409) |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L19E002 | 0.0664 | 1.0 | L19E000 (1.0) |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L21E001 | 0.3555 | 1.0 | L21E004 (0.6703), L21E005 (0.1868), L21E003 (0.0769) |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L18E001 | 0.1094 | 1.0 | L18E005 (0.8214), L18E000 (0.1429), L18E006 (0.0357) |
| Mixtral-8x7B (BOS) | cp_vt_orig | L19E006 | 0.8906 | 1.0 | L19E004 (0.2982), L19E000 (0.2675), L19E003 (0.2368) |
| Mixtral-8x7B (BOS) | cp_vt_orig | L20E000 | 0.7812 | 1.0 | L20E001 (0.375), L20E005 (0.165), L20E007 (0.155) |
| Mixtral-8x7B (BOS) | cp_vt_orig | L19E002 | 0.0547 | 1.0 | L19E000 (1.0) |
| Mixtral-8x7B (BOS) | cp_vt_orig | L21E001 | 0.332 | 1.0 | L21E004 (0.5765), L21E003 (0.2118), L21E005 (0.1176) |
| Mixtral-8x7B (BOS) | cp_vt_orig | L18E001 | 0.0 | 0.0 |  |
| Mixtral-8x7B (BOS) | cp_vt_neu | L19E006 | 0.6953 | 1.0 | L19E001 (0.4607), L19E004 (0.4551), L19E003 (0.0449) |
| Mixtral-8x7B (BOS) | cp_vt_neu | L20E000 | 0.6797 | 1.0 | L20E001 (0.3851), L20E005 (0.3046), L20E004 (0.1954) |
| Mixtral-8x7B (BOS) | cp_vt_neu | L19E002 | 0.0156 | 1.0 | L19E000 (1.0) |
| Mixtral-8x7B (BOS) | cp_vt_neu | L21E001 | 0.3906 | 1.0 | L21E004 (0.74), L21E007 (0.16), L21E000 (0.05) |
| Mixtral-8x7B (BOS) | cp_vt_neu | L18E001 | 0.0 | 0.0 |  |
| Mixtral-8x7B (BOS) | cp_nat | L21E006 | 0.9219 | 1.0 | L21E003 (0.3992), L21E004 (0.1636), L21E000 (0.1585) |
| Mixtral-8x7B (BOS) | cp_nat | L20E000 | 0.9734 | 1.0 | L20E007 (0.4374), L20E005 (0.2737), L20E001 (0.1421) |
| Mixtral-8x7B (BOS) | cp_nat | L19E002 | 0.0219 | 1.0 | L19E004 (0.6429), L19E000 (0.2857), L19E003 (0.0714) |
| Mixtral-8x7B (BOS) | cp_nat | L21E001 | 0.0219 | 1.0 | L21E004 (0.2857), L21E006 (0.25), L21E003 (0.1786) |
| Mixtral-8x7B (BOS) | cp_nat | L18E001 | 0.0031 | 1.0 | L18E004 (1.0) |


### 6. Self-repair: knockout loss vs the expert's own direct write

Only meaningful for experts with a sizeable direct write (W); for experts that act through later layers (W ≈ 0) the ratios are undefined (see the writer section).

| model | condition | expert | routed items | direct write W | net layer direct change N | knockout K | K / W [95% CI] | K / N [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1 | L37E032 | 754 | 0.0021 | 0.0022 | 0.1016 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | f1 | L41E117 | 17 | 0.019 | 0.0125 | 0.0368 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | f1 | L44E069 | 71 | 0.0245 | 0.0139 | 0.0458 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | f1 | L42E115 | 274 | 0.0046 | 0.0005 | -0.0287 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | dir_f1 | L44E116 | 153 | 0.2842 | 0.2486 | 0.0596 | 0.21 [0.07, 0.34] | 0.24 [0.08, 0.39] |
| Qwen3-30B-A3B-Base | dir_f1 | L44E069 | 15 | 0.0495 | -0.0024 | 0.0167 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | dir_f1 | L42E115 | 24 | 0.015 | 0.009 | -0.0729 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | dir_opt | L41E117 | 139 | 0.3776 | 0.3351 | 0.1425 | 0.38 [0.19, 0.55] | 0.43 [0.22, 0.62] |
| Qwen3-30B-A3B-Base | f2e | L43E036 | 63 | 0.1373 | 0.0975 | 0.0605 | 0.44 [-0.18, 1.18] | n/a |
| Qwen3-30B-A3B-Base | f2e | L44E069 | 28 | 0.0205 | 0.0101 | -0.029 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | f2e | L42E115 | 40 | 0.0225 | 0.0124 | 0.0359 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L39E121 | 184 | 0.1801 | 0.1505 | 0.0421 | 0.23 [-1.01, 0.47] | 0.28 [-1.61, 0.75] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L41E117 | 119 | 0.1313 | 0.1056 | -0.011 | -0.08 [-12.79, 6.94] | -0.10 [-5.04, 8.49] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L44E069 | 61 | 0.1769 | 0.1258 | 0.1762 | 1.00 [-3.23, 1.59] | 1.40 [-1.07, 3.07] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L42E115 | 20 | -0.0082 | 0.0735 | -0.0187 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L42E123 | 256 | 0.1896 | 0.1751 | 0.0608 | 0.32 [-0.10, 0.81] | 0.35 [-0.11, 0.94] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L41E117 | 99 | 0.2296 | 0.2044 | 0.0051 | 0.02 [-0.14, 2.99] | 0.02 [-0.15, 1.09] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L44E069 | 52 | 0.2133 | 0.155 | 0.1875 | 0.88 [-19.91, 1.22] | 1.21 [-0.10, 2.73] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L42E115 | 24 | -0.0063 | 0.0285 | 0.0104 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L37E032 | 74 | -0.0002 | -0.0017 | 0.0608 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L42E115 | 48 | -0.003 | 0.0027 | -0.0052 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_orig | L39E001 | 182 | 0.1589 | 0.1257 | 0.0515 | 0.32 [-0.87, 0.84] | 0.41 [-1.85, 1.36] |
| Qwen3-30B-A3B-Base | cp_vt_orig | L41E117 | 109 | 0.012 | 0.0042 | -0.0235 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_orig | L44E069 | 98 | 0.0112 | -0.0036 | 0.0561 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_orig | L42E115 | 55 | -0.0044 | -0.0339 | 0.0091 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_neu | L42E123 | 256 | 0.0784 | 0.0478 | -0.0186 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_neu | L41E117 | 139 | 0.0097 | 0.0289 | 0.0261 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_neu | L44E069 | 67 | 0.003 | -0.015 | -0.0075 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_neu | L42E115 | 35 | 0.0101 | -0.0213 | 0.025 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_nat | L41E117 | 821 | 0.327 | 0.3086 | 0.093 | 0.28 [0.22, 0.35] | 0.30 [0.23, 0.37] |
| Qwen3-30B-A3B-Base | cp_nat | L44E069 | 11 | 0.0384 | -0.0686 | -0.1364 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | f1 | L20E001 | 150 | 0.1337 | 0.0527 | 0.0158 | 0.12 [-0.19, 0.42] | n/a |
| Mixtral-8x7B (BOS) | f1 | L20E000 | 38 | 0.0542 | 0.0238 | 0.0296 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | f1 | L19E002 | 8 | 0.0132 | 0.0081 | 0.0469 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | f1 | L21E001 | 71 | 0.0745 | 0.0045 | 0.0123 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | f1 | L18E001 | 24 | -0.0113 | -0.0175 | 0.0312 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | dir_f1 | L20E001 | 165 | 0.1975 | 0.0838 | -0.0121 | -0.06 [-0.28, 0.15] | n/a |
| Mixtral-8x7B (BOS) | dir_f1 | L20E000 | 14 | 0.0608 | 0.0026 | 0.0446 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | dir_f1 | L21E001 | 70 | 0.1155 | 0.0013 | -0.0107 | -0.09 [-0.69, 0.38] | n/a |
| Mixtral-8x7B (BOS) | dir_opt | L20E000 | 185 | 0.4258 | 0.2481 | 0.1243 | 0.29 [0.19, 0.40] | 0.50 [0.34, 0.67] |
| Mixtral-8x7B (BOS) | f2e | L19E004 | 36 | 0.2181 | 0.0864 | 0.1042 | 0.48 [0.08, 0.86] | n/a |
| Mixtral-8x7B (BOS) | f2e | L20E000 | 10 | 0.0251 | -0.0381 | 0.0 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | f2e | L21E001 | 16 | 0.1489 | 0.0321 | 0.0625 | 0.42 [0.15, 2.20] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L19E006 | 240 | 0.2876 | 0.1198 | 0.1417 | 0.49 [0.29, 0.82] | 1.18 [0.78, 2.22] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L20E000 | 238 | 0.2722 | 0.088 | 0.0399 | 0.15 [-0.31, 0.40] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L19E002 | 48 | 0.1367 | -0.0533 | -0.0755 | -0.55 [-2.17, 0.09] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L21E001 | 67 | 0.1565 | 0.0287 | 0.0373 | 0.24 [0.00, 0.79] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L18E001 | 20 | 0.038 | -0.033 | -0.0375 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L19E006 | 232 | 0.1773 | 0.056 | 0.1056 | 0.60 [0.17, 1.20] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L20E000 | 220 | 0.2496 | 0.1018 | 0.0756 | 0.30 [-0.22, 0.55] | 0.74 [-1.19, 2.36] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L19E002 | 17 | 0.215 | 0.0174 | -0.0147 | -0.07 [-0.34, 0.62] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L21E001 | 91 | 0.1334 | 0.0359 | 0.0357 | 0.27 [0.11, 1.17] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L18E001 | 28 | 0.0352 | -0.0224 | -0.058 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | cp_vt_orig | L19E006 | 228 | 0.2751 | 0.092 | 0.0471 | 0.17 [-0.04, 0.44] | n/a |
| Mixtral-8x7B (BOS) | cp_vt_orig | L20E000 | 200 | 0.336 | 0.1366 | 0.0688 | 0.20 [-0.19, 0.37] | 0.50 [-2.09, 1.38] |
| Mixtral-8x7B (BOS) | cp_vt_orig | L19E002 | 14 | 0.2276 | -0.089 | -0.0804 | -0.35 [-0.35, -0.35] | n/a |
| Mixtral-8x7B (BOS) | cp_vt_orig | L21E001 | 85 | 0.1456 | 0.0216 | 0.0206 | 0.14 [-0.08, 0.65] | n/a |
| Mixtral-8x7B (BOS) | cp_vt_neu | L19E006 | 178 | 0.176 | 0.0857 | 0.1074 | 0.61 [0.06, 0.95] | n/a |
| Mixtral-8x7B (BOS) | cp_vt_neu | L20E000 | 174 | 0.1434 | -0.041 | -0.0065 | -0.05 [-1.52, 0.45] | n/a |
| Mixtral-8x7B (BOS) | cp_vt_neu | L21E001 | 100 | 0.1026 | 0.0325 | 0.0887 | 0.87 [0.36, 1.43] | n/a |
| Mixtral-8x7B (BOS) | cp_nat | L21E006 | 1180 | 0.2631 | 0.0675 | 0.0907 | 0.34 [0.28, 0.41] | n/a |
| Mixtral-8x7B (BOS) | cp_nat | L20E000 | 1246 | 0.3608 | 0.2145 | 0.1681 | 0.47 [0.41, 0.52] | 0.78 [0.70, 0.88] |
| Mixtral-8x7B (BOS) | cp_nat | L19E002 | 28 | 0.0515 | 0.0018 | 0.0045 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | cp_nat | L21E001 | 28 | 0.0775 | 0.015 | 0.0134 | n/a (W < 0.1: acts indirectly) | n/a |


### 7. Cross-condition knockout (other conditions' selected experts)

| model | evaluated on | knockout | selected on | F [95% CI] |
|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1 | L44E116 | dir_f1 | +0.021 [+0.004, +0.038] |
| Qwen3-30B-A3B-Base | f1 | L43E036 | f2e | +0.007 [-0.006, +0.020] |
| Qwen3-30B-A3B-Base | f1 | L39E121 | vt_e2a_orig | -0.012 [-0.029, +0.004] |
| Qwen3-30B-A3B-Base | f1 | L42E123 | vt_e2a_neu | -0.008 [-0.023, +0.007] |
| Qwen3-30B-A3B-Base | f1 | L39E001 | cp_vt_orig | +0.003 [-0.013, +0.018] |
| Qwen3-30B-A3B-Base | dir_f1 | L37E032 | f1 | +0.075 [+0.057, +0.094] |
| Qwen3-30B-A3B-Base | dir_f1 | L43E036 | f2e | -0.001 [-0.015, +0.013] |
| Qwen3-30B-A3B-Base | dir_f1 | L39E121 | vt_e2a_orig | -0.003 [-0.017, +0.011] |
| Qwen3-30B-A3B-Base | dir_f1 | L42E123 | vt_e2a_neu | -0.001 [-0.015, +0.013] |
| Qwen3-30B-A3B-Base | dir_f1 | L39E001 | cp_vt_orig | +0.004 [-0.010, +0.018] |
| Qwen3-30B-A3B-Base | dir_opt | L37E032 | f1 | +0.000 [-0.011, +0.012] |
| Qwen3-30B-A3B-Base | dir_opt | L44E116 | dir_f1 | +0.001 [-0.012, +0.013] |
| Qwen3-30B-A3B-Base | dir_opt | L43E036 | f2e | +0.007 [-0.005, +0.019] |
| Qwen3-30B-A3B-Base | dir_opt | L39E121 | vt_e2a_orig | +0.003 [-0.009, +0.014] |
| Qwen3-30B-A3B-Base | dir_opt | L42E123 | vt_e2a_neu | +0.008 [-0.004, +0.021] |
| Qwen3-30B-A3B-Base | dir_opt | L39E001 | cp_vt_orig | +0.009 [-0.004, +0.021] |
| Qwen3-30B-A3B-Base | f2e | L37E032 | f1 | -0.002 [-0.027, +0.025] |
| Qwen3-30B-A3B-Base | f2e | L44E116 | dir_f1 | +0.016 [-0.014, +0.048] |
| Qwen3-30B-A3B-Base | f2e | L39E121 | vt_e2a_orig | +0.000 [-0.022, +0.025] |
| Qwen3-30B-A3B-Base | f2e | L42E123 | vt_e2a_neu | -0.001 [-0.023, +0.025] |
| Qwen3-30B-A3B-Base | f2e | L39E001 | cp_vt_orig | -0.006 [-0.034, +0.022] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L37E032 | f1 | +0.000 [-0.008, +0.008] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L44E116 | dir_f1 | +0.010 [+0.002, +0.016] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L43E036 | f2e | +0.019 [-0.004, +0.051] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L42E123 | vt_e2a_neu | +0.029 [+0.009, +0.050] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L39E001 | cp_vt_orig | +0.017 [-0.005, +0.046] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L37E032 | f1 | -0.007 [-0.019, +0.003] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L44E116 | dir_f1 | -0.012 [-0.025, -0.002] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L43E036 | f2e | +0.003 [-0.027, +0.044] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L39E121 | vt_e2a_orig | +0.011 [-0.022, +0.058] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L39E001 | cp_vt_orig | +0.011 [-0.015, +0.051] |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L44E116 | dir_f1 | -0.010 [-0.013, +0.030] |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L43E036 | f2e | -0.019 [-0.024, +0.042] |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L39E121 | vt_e2a_orig | -0.055 [-0.064, +0.053] |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L42E123 | vt_e2a_neu | -0.023 [-0.027, +0.026] |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L39E001 | cp_vt_orig | -0.018 [-0.048, +0.025] |
| Qwen3-30B-A3B-Base | cp_vt_orig | L37E032 | f1 | +0.007 [-0.007, +0.023] |
| Qwen3-30B-A3B-Base | cp_vt_orig | L44E116 | dir_f1 | -0.001 [-0.013, +0.013] |
| Qwen3-30B-A3B-Base | cp_vt_orig | L43E036 | f2e | +0.001 [-0.012, +0.014] |
| Qwen3-30B-A3B-Base | cp_vt_orig | L39E121 | vt_e2a_orig | -0.001 [-0.022, +0.030] |
| Qwen3-30B-A3B-Base | cp_vt_orig | L42E123 | vt_e2a_neu | +0.022 [-0.012, +0.061] |
| Qwen3-30B-A3B-Base | cp_vt_neu | L37E032 | f1 | -0.002 [-0.014, +0.010] |
| Qwen3-30B-A3B-Base | cp_vt_neu | L44E116 | dir_f1 | -0.014 [-0.029, -0.004] |
| Qwen3-30B-A3B-Base | cp_vt_neu | L43E036 | f2e | -0.007 [-0.019, +0.003] |
| Qwen3-30B-A3B-Base | cp_vt_neu | L39E121 | vt_e2a_orig | +0.010 [-0.016, +0.060] |
| Qwen3-30B-A3B-Base | cp_vt_neu | L39E001 | cp_vt_orig | -0.001 [-0.050, +0.062] |
| Qwen3-30B-A3B-Base | cp_nat | L37E032 | f1 | -0.013 [-0.029, +0.001] |
| Qwen3-30B-A3B-Base | cp_nat | L44E116 | dir_f1 | -0.016 [-0.029, -0.004] |
| Qwen3-30B-A3B-Base | cp_nat | L43E036 | f2e | +0.007 [-0.007, +0.021] |
| Qwen3-30B-A3B-Base | cp_nat | L39E121 | vt_e2a_orig | -0.011 [-0.025, +0.001] |
| Qwen3-30B-A3B-Base | cp_nat | L42E123 | vt_e2a_neu | -0.007 [-0.023, +0.009] |
| Qwen3-30B-A3B-Base | cp_nat | L39E001 | cp_vt_orig | -0.013 [-0.028, -0.000] |
| Mixtral-8x7B (BOS) | f1 | L19E004 | f2e | +0.042 [+0.026, +0.059] |
| Mixtral-8x7B (BOS) | f1 | L19E006 | vt_e2a_orig | +0.008 [-0.007, +0.022] |
| Mixtral-8x7B (BOS) | f1 | L21E006 | cp_nat | +0.004 [-0.008, +0.016] |
| Mixtral-8x7B (BOS) | dir_f1 | L19E004 | f2e | +0.062 [+0.045, +0.079] |
| Mixtral-8x7B (BOS) | dir_f1 | L19E006 | vt_e2a_orig | -0.002 [-0.015, +0.011] |
| Mixtral-8x7B (BOS) | dir_f1 | L21E006 | cp_nat | +0.011 [+0.001, +0.022] |
| Mixtral-8x7B (BOS) | dir_opt | L20E001 | f1 | +0.007 [+0.003, +0.012] |
| Mixtral-8x7B (BOS) | dir_opt | L19E004 | f2e | -0.000 [-0.004, +0.004] |
| Mixtral-8x7B (BOS) | dir_opt | L19E006 | vt_e2a_orig | +0.042 [+0.033, +0.051] |
| Mixtral-8x7B (BOS) | dir_opt | L21E006 | cp_nat | +0.013 [+0.005, +0.021] |
| Mixtral-8x7B (BOS) | f2e | L20E001 | f1 | +0.004 [-0.008, +0.015] |
| Mixtral-8x7B (BOS) | f2e | L19E006 | vt_e2a_orig | +0.010 [-0.004, +0.024] |
| Mixtral-8x7B (BOS) | f2e | L21E006 | cp_nat | +0.005 [-0.006, +0.014] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L20E001 | f1 | -0.002 [-0.006, +0.003] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L19E004 | f2e | -0.008 [-0.020, +0.002] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L21E006 | cp_nat | -0.005 [-0.015, +0.002] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L20E001 | f1 | -0.005 [-0.018, +0.005] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L19E004 | f2e | +0.006 [-0.001, +0.016] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L21E006 | cp_nat | -0.002 [-0.011, +0.007] |
| Mixtral-8x7B (BOS) | cp_vt_orig | L20E001 | f1 | +0.003 [-0.001, +0.007] |
| Mixtral-8x7B (BOS) | cp_vt_orig | L19E004 | f2e | -0.008 [-0.018, +0.000] |
| Mixtral-8x7B (BOS) | cp_vt_orig | L21E006 | cp_nat | -0.003 [-0.012, +0.007] |
| Mixtral-8x7B (BOS) | cp_vt_neu | L20E001 | f1 | +0.000 [-0.003, +0.006] |
| Mixtral-8x7B (BOS) | cp_vt_neu | L19E004 | f2e | -0.009 [-0.025, +0.007] |
| Mixtral-8x7B (BOS) | cp_vt_neu | L21E006 | cp_nat | +0.000 [-0.008, +0.007] |
| Mixtral-8x7B (BOS) | cp_nat | L20E001 | f1 | +0.005 [+0.001, +0.009] |
| Mixtral-8x7B (BOS) | cp_nat | L19E004 | f2e | -0.001 [-0.005, +0.003] |
| Mixtral-8x7B (BOS) | cp_nat | L19E006 | vt_e2a_orig | +0.043 [+0.031, +0.054] |


### 8. Strata

| model | condition | stratum | items | knockout | F [95% CI] |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1 | names=False | 374 | L37E032 | +0.075 [+0.062, +0.087] |
| Qwen3-30B-A3B-Base | f1 | names=False | 374 | L41E117 | -0.002 [-0.010, +0.006] |
| Qwen3-30B-A3B-Base | f1 | names=False | 374 | L44E069 | +0.003 [-0.006, +0.011] |
| Qwen3-30B-A3B-Base | f1 | names=False | 374 | L42E115 | -0.004 [-0.012, +0.004] |
| Qwen3-30B-A3B-Base | f1 | names=False | 374 | pop:k10 | +0.171 [+0.154, +0.189] |
| Qwen3-30B-A3B-Base | f1 | names=True | 394 | L37E032 | +0.004 [-0.020, +0.027] |
| Qwen3-30B-A3B-Base | f1 | names=True | 394 | L41E117 | -0.005 [-0.014, +0.003] |
| Qwen3-30B-A3B-Base | f1 | names=True | 394 | L44E069 | -0.003 [-0.015, +0.009] |
| Qwen3-30B-A3B-Base | f1 | names=True | 394 | L42E115 | -0.016 [-0.027, -0.005] |
| Qwen3-30B-A3B-Base | f1 | names=True | 394 | pop:k10 | +0.211 [+0.188, +0.234] |
| Qwen3-30B-A3B-Base | dir_f1 | names=False | 168 | L44E116 | +0.022 [+0.007, +0.037] |
| Qwen3-30B-A3B-Base | dir_f1 | names=False | 168 | L41E117 | -0.000 [-0.010, +0.009] |
| Qwen3-30B-A3B-Base | dir_f1 | names=False | 168 | L44E069 | -0.003 [-0.013, +0.008] |
| Qwen3-30B-A3B-Base | dir_f1 | names=False | 168 | L42E115 | -0.006 [-0.017, +0.004] |
| Qwen3-30B-A3B-Base | dir_f1 | names=False | 168 | pop:k10 | +0.182 [+0.145, +0.220] |
| Qwen3-30B-A3B-Base | dir_f1 | names=True | 12 | L44E116 | +0.009 [-0.039, +0.075] |
| Qwen3-30B-A3B-Base | dir_f1 | names=True | 12 | L41E117 | +0.023 [-0.030, +0.104] |
| Qwen3-30B-A3B-Base | dir_f1 | names=True | 12 | L44E069 | +0.018 [-0.038, +0.090] |
| Qwen3-30B-A3B-Base | dir_f1 | names=True | 12 | L42E115 | -0.005 [-0.089, +0.071] |
| Qwen3-30B-A3B-Base | dir_f1 | names=True | 12 | pop:k10 | +0.176 [+0.085, +0.257] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=neutral | 100 | L44E116 | +0.033 [+0.013, +0.053] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=neutral | 100 | L41E117 | +0.002 [-0.009, +0.013] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=neutral | 100 | L44E069 | -0.001 [-0.015, +0.014] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=neutral | 100 | L42E115 | -0.005 [-0.020, +0.009] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=neutral | 100 | pop:k10 | +0.250 [+0.201, +0.297] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=one-sided | 32 | L44E116 | -0.003 [-0.024, +0.014] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=one-sided | 32 | L41E117 | -0.007 [-0.026, +0.014] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=one-sided | 32 | L44E069 | -0.011 [-0.027, +0.004] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=one-sided | 32 | L42E115 | +0.009 [-0.010, +0.029] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=one-sided | 32 | pop:k10 | +0.127 [+0.062, +0.202] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=reverse | 40 | L44E116 | +0.011 [-0.025, +0.046] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=reverse | 40 | L41E117 | +0.007 [-0.021, +0.037] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=reverse | 40 | L44E069 | +0.006 [-0.018, +0.032] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=reverse | 40 | L42E115 | -0.018 [-0.046, +0.009] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_assoc=reverse | 40 | pop:k10 | +0.099 [+0.043, +0.154] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_local_assoc=False | 166 | L44E116 | +0.020 [+0.005, +0.035] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_local_assoc=False | 166 | L41E117 | -0.002 [-0.012, +0.009] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_local_assoc=False | 166 | L44E069 | -0.001 [-0.011, +0.011] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_local_assoc=False | 166 | L42E115 | -0.004 [-0.016, +0.007] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_local_assoc=False | 166 | pop:k10 | +0.186 [+0.148, +0.225] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_local_assoc=True | 14 | L44E116 | +0.035 [+0.007, +0.086] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_local_assoc=True | 14 | L41E117 | +0.032 [+0.010, +0.073] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_local_assoc=True | 14 | L44E069 | -0.010 [-0.047, +0.035] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_local_assoc=True | 14 | L42E115 | -0.029 [-0.073, +0.008] |
| Qwen3-30B-A3B-Base | dir_f1 | stratum_local_assoc=True | 14 | pop:k10 | +0.140 [+0.051, +0.220] |
| Qwen3-30B-A3B-Base | dir_opt | names=False | 168 | L41E117 | +0.026 [+0.010, +0.041] |
| Qwen3-30B-A3B-Base | dir_opt | names=False | 168 | L44E069 | -0.004 [-0.010, +0.003] |
| Qwen3-30B-A3B-Base | dir_opt | names=False | 168 | L42E115 | -0.001 [-0.008, +0.006] |
| Qwen3-30B-A3B-Base | dir_opt | names=False | 168 | pop:k10 | +0.190 [+0.158, +0.222] |
| Qwen3-30B-A3B-Base | dir_opt | names=True | 12 | L41E117 | +0.022 [-0.025, +0.088] |
| Qwen3-30B-A3B-Base | dir_opt | names=True | 12 | L44E069 | +0.017 [-0.006, +0.044] |
| Qwen3-30B-A3B-Base | dir_opt | names=True | 12 | L42E115 | +0.028 [+0.003, +0.069] |
| Qwen3-30B-A3B-Base | dir_opt | names=True | 12 | pop:k10 | +0.104 [+0.009, +0.223] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_assoc=neutral | 100 | L41E117 | +0.015 [+0.001, +0.029] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_assoc=neutral | 100 | L44E069 | -0.002 [-0.007, +0.004] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_assoc=neutral | 100 | L42E115 | -0.004 [-0.011, +0.002] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_assoc=neutral | 100 | pop:k10 | +0.221 [+0.177, +0.263] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_assoc=one-sided | 32 | L41E117 | +0.011 [-0.025, +0.046] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_assoc=one-sided | 32 | L44E069 | +0.004 [-0.010, +0.017] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_assoc=one-sided | 32 | L42E115 | +0.002 [-0.017, +0.021] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_assoc=one-sided | 32 | pop:k10 | +0.160 [+0.093, +0.222] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_assoc=reverse | 40 | L41E117 | +0.065 [+0.031, +0.107] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_assoc=reverse | 40 | L44E069 | +0.003 [-0.009, +0.017] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_assoc=reverse | 40 | L42E115 | +0.016 [-0.001, +0.035] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_assoc=reverse | 40 | pop:k10 | +0.159 [+0.108, +0.209] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_local_assoc=False | 166 | L41E117 | +0.025 [+0.009, +0.041] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_local_assoc=False | 166 | L44E069 | -0.003 [-0.010, +0.004] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_local_assoc=False | 166 | L42E115 | +0.001 [-0.007, +0.009] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_local_assoc=False | 166 | pop:k10 | +0.191 [+0.158, +0.223] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_local_assoc=True | 14 | L41E117 | +0.035 [-0.010, +0.071] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_local_assoc=True | 14 | L44E069 | +0.002 [-0.007, +0.013] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_local_assoc=True | 14 | L42E115 | -0.004 [-0.028, +0.015] |
| Qwen3-30B-A3B-Base | dir_opt | stratum_local_assoc=True | 14 | pop:k10 | +0.125 [+0.004, +0.201] |
| Qwen3-30B-A3B-Base | f2e | names=False | 52 | L43E036 | +0.016 [-0.005, +0.035] |
| Qwen3-30B-A3B-Base | f2e | names=False | 52 | L41E117 | +0.007 [-0.003, +0.016] |
| Qwen3-30B-A3B-Base | f2e | names=False | 52 | L44E069 | -0.003 [-0.017, +0.009] |
| Qwen3-30B-A3B-Base | f2e | names=False | 52 | L42E115 | +0.011 [-0.002, +0.026] |
| Qwen3-30B-A3B-Base | f2e | names=False | 52 | pop:k10 | +0.042 [-0.027, +0.097] |
| Qwen3-30B-A3B-Base | f2e | names=True | 12 | L43E036 | +0.009 [-0.030, +0.063] |
| Qwen3-30B-A3B-Base | f2e | names=True | 12 | L41E117 | -0.009 [-0.060, +0.063] |
| Qwen3-30B-A3B-Base | f2e | names=True | 12 | L44E069 | +0.021 [-0.005, +0.062] |
| Qwen3-30B-A3B-Base | f2e | names=True | 12 | L42E115 | -0.006 [-0.094, +0.087] |
| Qwen3-30B-A3B-Base | f2e | names=True | 12 | pop:k10 | -0.032 [-0.298, +0.215] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=neutral | 218 | L39E121 | +0.005 [-0.013, +0.027] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=neutral | 218 | L41E117 | +0.002 [-0.007, +0.012] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=neutral | 218 | L44E069 | +0.009 [-0.011, +0.037] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=neutral | 218 | L42E115 | -0.002 [-0.009, +0.005] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=neutral | 218 | pop:k10 | +0.046 [-0.036, +0.129] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=one-sided | 14 | L39E121 | +0.010 [-0.045, +0.077] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=one-sided | 14 | L41E117 | -0.007 [-0.070, +0.056] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=one-sided | 14 | L44E069 | +0.024 [-0.011, +0.069] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=one-sided | 14 | L42E115 | +0.021 [-0.062, +0.107] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=one-sided | 14 | pop:k10 | +0.112 [-0.037, +0.267] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=reverse | 24 | L39E121 | -0.002 [-0.031, +0.052] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=reverse | 24 | L41E117 | +0.003 [-0.021, +0.030] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=reverse | 24 | L44E069 | +0.007 [-0.006, +0.023] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=reverse | 24 | L42E115 | -0.007 [-0.022, +0.006] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | stratum_assoc=reverse | 24 | pop:k10 | +0.111 [+0.025, +0.206] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=neutral | 200 | L42E123 | +0.022 [-0.004, +0.049] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=neutral | 200 | L41E117 | -0.002 [-0.014, +0.010] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=neutral | 200 | L44E069 | +0.007 [-0.008, +0.027] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=neutral | 200 | L42E115 | -0.004 [-0.007, +0.000] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=neutral | 200 | pop:k10 | +0.156 [+0.074, +0.242] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=one-sided | 20 | L42E123 | +0.025 [+0.002, +0.044] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=one-sided | 20 | L41E117 | +0.004 [-0.019, +0.031] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=one-sided | 20 | L44E069 | -0.002 [-0.021, +0.024] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=one-sided | 20 | L42E115 | -0.002 [-0.022, +0.015] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=one-sided | 20 | pop:k10 | +0.152 [+0.081, +0.238] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=reverse | 36 | L42E123 | +0.008 [-0.027, +0.036] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=reverse | 36 | L41E117 | +0.017 [-0.002, +0.034] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=reverse | 36 | L44E069 | +0.027 [-0.015, +0.091] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=reverse | 36 | L42E115 | +0.004 [-0.008, +0.019] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | stratum_assoc=reverse | 36 | pop:k10 | +0.195 [+0.107, +0.294] |
| Qwen3-30B-A3B-Base | vt_a2e_orig | stratum_assoc=neutral | 52 | L37E032 | +0.044 [+0.025, +0.118] |
| Qwen3-30B-A3B-Base | vt_a2e_orig | stratum_assoc=neutral | 52 | L41E117 | +0.000 [-0.031, +0.003] |
| Qwen3-30B-A3B-Base | vt_a2e_orig | stratum_assoc=neutral | 52 | L44E069 | +0.012 [-0.017, +0.015] |
| Qwen3-30B-A3B-Base | vt_a2e_orig | stratum_assoc=neutral | 52 | L42E115 | +0.016 [+0.000, +0.018] |
| Qwen3-30B-A3B-Base | vt_a2e_orig | stratum_assoc=neutral | 52 | pop:k10 | +0.292 [+0.280, +0.426] |
| Qwen3-30B-A3B-Base | cp_vt_orig | stratum_assoc=neutral | 202 | L39E001 | +0.013 [-0.015, +0.042] |
| Qwen3-30B-A3B-Base | cp_vt_orig | stratum_assoc=neutral | 202 | L41E117 | +0.000 [-0.009, +0.009] |
| Qwen3-30B-A3B-Base | cp_vt_orig | stratum_assoc=neutral | 202 | L44E069 | +0.002 [-0.005, +0.010] |
| Qwen3-30B-A3B-Base | cp_vt_orig | stratum_assoc=neutral | 202 | L42E115 | +0.002 [-0.008, +0.013] |
| Qwen3-30B-A3B-Base | cp_vt_orig | stratum_assoc=neutral | 202 | pop:k10 | +0.099 [+0.005, +0.211] |
| Qwen3-30B-A3B-Base | cp_vt_orig | stratum_assoc=reverse | 38 | L39E001 | -0.003 [-0.036, +0.031] |
| Qwen3-30B-A3B-Base | cp_vt_orig | stratum_assoc=reverse | 38 | L41E117 | -0.007 [-0.025, +0.005] |
| Qwen3-30B-A3B-Base | cp_vt_orig | stratum_assoc=reverse | 38 | L44E069 | -0.002 [-0.019, +0.010] |
| Qwen3-30B-A3B-Base | cp_vt_orig | stratum_assoc=reverse | 38 | L42E115 | +0.001 [-0.015, +0.015] |
| Qwen3-30B-A3B-Base | cp_vt_orig | stratum_assoc=reverse | 38 | pop:k10 | +0.095 [-0.102, +0.285] |
| Qwen3-30B-A3B-Base | cp_vt_orig | order=1 | 132 | L39E001 | -0.001 [-0.034, +0.025] |
| Qwen3-30B-A3B-Base | cp_vt_orig | order=1 | 132 | L41E117 | -0.008 [-0.032, +0.007] |
| Qwen3-30B-A3B-Base | cp_vt_orig | order=1 | 132 | L44E069 | +0.004 [-0.007, +0.017] |
| Qwen3-30B-A3B-Base | cp_vt_orig | order=1 | 132 | L42E115 | +0.000 [-0.013, +0.011] |
| Qwen3-30B-A3B-Base | cp_vt_orig | order=1 | 132 | pop:k10 | +0.068 [-0.050, +0.190] |
| Qwen3-30B-A3B-Base | cp_vt_orig | order=2 | 124 | L39E001 | +0.019 [-0.007, +0.049] |
| Qwen3-30B-A3B-Base | cp_vt_orig | order=2 | 124 | L41E117 | +0.006 [-0.005, +0.022] |
| Qwen3-30B-A3B-Base | cp_vt_orig | order=2 | 124 | L44E069 | +0.005 [-0.007, +0.022] |
| Qwen3-30B-A3B-Base | cp_vt_orig | order=2 | 124 | L42E115 | +0.004 [-0.010, +0.022] |
| Qwen3-30B-A3B-Base | cp_vt_orig | order=2 | 124 | pop:k10 | +0.129 [+0.025, +0.236] |
| Qwen3-30B-A3B-Base | cp_vt_neu | stratum_assoc=neutral | 184 | L42E123 | -0.004 [-0.018, +0.013] |
| Qwen3-30B-A3B-Base | cp_vt_neu | stratum_assoc=neutral | 184 | L41E117 | +0.001 [-0.011, +0.013] |
| Qwen3-30B-A3B-Base | cp_vt_neu | stratum_assoc=neutral | 184 | L44E069 | -0.001 [-0.011, +0.010] |
| Qwen3-30B-A3B-Base | cp_vt_neu | stratum_assoc=neutral | 184 | L42E115 | -0.002 [-0.017, +0.017] |
| Qwen3-30B-A3B-Base | cp_vt_neu | stratum_assoc=neutral | 184 | pop:k10 | +0.040 [+0.007, +0.069] |
| Qwen3-30B-A3B-Base | cp_vt_neu | stratum_assoc=reverse | 56 | L42E123 | -0.002 [-0.025, +0.015] |
| Qwen3-30B-A3B-Base | cp_vt_neu | stratum_assoc=reverse | 56 | L41E117 | -0.002 [-0.029, +0.020] |
| Qwen3-30B-A3B-Base | cp_vt_neu | stratum_assoc=reverse | 56 | L44E069 | +0.003 [-0.009, +0.019] |
| Qwen3-30B-A3B-Base | cp_vt_neu | stratum_assoc=reverse | 56 | L42E115 | -0.012 [-0.036, +0.007] |
| Qwen3-30B-A3B-Base | cp_vt_neu | stratum_assoc=reverse | 56 | pop:k10 | +0.113 [+0.051, +0.176] |
| Qwen3-30B-A3B-Base | cp_vt_neu | order=1 | 132 | L42E123 | -0.002 [-0.010, +0.011] |
| Qwen3-30B-A3B-Base | cp_vt_neu | order=1 | 132 | L41E117 | +0.007 [-0.007, +0.025] |
| Qwen3-30B-A3B-Base | cp_vt_neu | order=1 | 132 | L44E069 | +0.002 [-0.006, +0.011] |
| Qwen3-30B-A3B-Base | cp_vt_neu | order=1 | 132 | L42E115 | +0.005 [-0.013, +0.030] |
| Qwen3-30B-A3B-Base | cp_vt_neu | order=1 | 132 | pop:k10 | +0.057 [+0.024, +0.092] |
| Qwen3-30B-A3B-Base | cp_vt_neu | order=2 | 124 | L42E123 | -0.012 [-0.035, +0.011] |
| Qwen3-30B-A3B-Base | cp_vt_neu | order=2 | 124 | L41E117 | -0.007 [-0.017, +0.002] |
| Qwen3-30B-A3B-Base | cp_vt_neu | order=2 | 124 | L44E069 | -0.003 [-0.012, +0.013] |
| Qwen3-30B-A3B-Base | cp_vt_neu | order=2 | 124 | L42E115 | -0.014 [-0.024, -0.005] |
| Qwen3-30B-A3B-Base | cp_vt_neu | order=2 | 124 | pop:k10 | +0.060 [-0.014, +0.108] |
| Qwen3-30B-A3B-Base | cp_nat | names=False | 868 | L41E117 | +0.027 [+0.019, +0.035] |
| Qwen3-30B-A3B-Base | cp_nat | names=False | 868 | L44E069 | -0.005 [-0.011, +0.002] |
| Qwen3-30B-A3B-Base | cp_nat | names=False | 868 | L42E115 | -0.009 [-0.016, -0.003] |
| Qwen3-30B-A3B-Base | cp_nat | names=False | 868 | pop:k10 | +0.217 [+0.195, +0.240] |
| Qwen3-30B-A3B-Base | cp_nat | names=True | 52 | L41E117 | +0.021 [-0.009, +0.046] |
| Qwen3-30B-A3B-Base | cp_nat | names=True | 52 | L44E069 | -0.010 [-0.040, +0.012] |
| Qwen3-30B-A3B-Base | cp_nat | names=True | 52 | L42E115 | +0.000 [-0.016, +0.014] |
| Qwen3-30B-A3B-Base | cp_nat | names=True | 52 | pop:k10 | +0.006 [-0.034, +0.049] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=associative | 28 | L41E117 | +0.021 [-0.002, +0.049] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=associative | 28 | L44E069 | +0.018 [-0.015, +0.058] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=associative | 28 | L42E115 | -0.013 [-0.046, +0.011] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=associative | 28 | pop:k10 | +0.155 [+0.045, +0.239] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=neutral | 388 | L41E117 | +0.017 [+0.005, +0.029] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=neutral | 388 | L44E069 | -0.007 [-0.017, +0.001] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=neutral | 388 | L42E115 | -0.008 [-0.017, +0.001] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=neutral | 388 | pop:k10 | +0.207 [+0.172, +0.242] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=no_probe | 32 | L41E117 | -0.016 [-0.047, +0.012] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=no_probe | 32 | L44E069 | -0.004 [-0.025, +0.016] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=no_probe | 32 | L42E115 | -0.049 [-0.080, -0.025] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=no_probe | 32 | pop:k10 | +0.067 [+0.004, +0.123] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=one-sided | 144 | L41E117 | +0.029 [+0.010, +0.047] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=one-sided | 144 | L44E069 | -0.009 [-0.024, +0.005] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=one-sided | 144 | L42E115 | -0.006 [-0.024, +0.009] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=one-sided | 144 | pop:k10 | +0.131 [+0.075, +0.193] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=reverse | 328 | L41E117 | +0.041 [+0.027, +0.054] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=reverse | 328 | L44E069 | -0.003 [-0.015, +0.008] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=reverse | 328 | L42E115 | -0.006 [-0.016, +0.004] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_assoc=reverse | 328 | pop:k10 | +0.251 [+0.216, +0.285] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_local_assoc=False | 808 | L41E117 | +0.025 [+0.016, +0.033] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_local_assoc=False | 808 | L44E069 | -0.004 [-0.011, +0.002] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_local_assoc=False | 808 | L42E115 | -0.009 [-0.016, -0.003] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_local_assoc=False | 808 | pop:k10 | +0.202 [+0.178, +0.228] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_local_assoc=True | 112 | L41E117 | +0.039 [+0.019, +0.058] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_local_assoc=True | 112 | L44E069 | -0.009 [-0.025, +0.006] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_local_assoc=True | 112 | L42E115 | -0.004 [-0.021, +0.010] |
| Qwen3-30B-A3B-Base | cp_nat | stratum_local_assoc=True | 112 | pop:k10 | +0.211 [+0.151, +0.271] |
| Qwen3-30B-A3B-Base | cp_nat | order=1 | 460 | L41E117 | +0.025 [+0.015, +0.034] |
| Qwen3-30B-A3B-Base | cp_nat | order=1 | 460 | L44E069 | -0.009 [-0.018, -0.000] |
| Qwen3-30B-A3B-Base | cp_nat | order=1 | 460 | L42E115 | -0.009 [-0.017, -0.001] |
| Qwen3-30B-A3B-Base | cp_nat | order=1 | 460 | pop:k10 | +0.205 [+0.181, +0.230] |
| Qwen3-30B-A3B-Base | cp_nat | order=2 | 460 | L41E117 | +0.029 [+0.019, +0.038] |
| Qwen3-30B-A3B-Base | cp_nat | order=2 | 460 | L44E069 | -0.001 [-0.009, +0.006] |
| Qwen3-30B-A3B-Base | cp_nat | order=2 | 460 | L42E115 | -0.008 [-0.017, +0.001] |
| Qwen3-30B-A3B-Base | cp_nat | order=2 | 460 | pop:k10 | +0.202 [+0.177, +0.227] |
| Mixtral-8x7B (BOS) | f1 | names=False | 178 | L20E001 | +0.006 [-0.009, +0.021] |
| Mixtral-8x7B (BOS) | f1 | names=False | 178 | L20E000 | +0.007 [-0.000, +0.013] |
| Mixtral-8x7B (BOS) | f1 | names=False | 178 | L19E002 | +0.001 [-0.005, +0.006] |
| Mixtral-8x7B (BOS) | f1 | names=False | 178 | L21E001 | +0.009 [+0.002, +0.015] |
| Mixtral-8x7B (BOS) | f1 | names=False | 178 | L18E001 | +0.001 [-0.005, +0.008] |
| Mixtral-8x7B (BOS) | f1 | names=False | 178 | pop:k10 | +0.243 [+0.198, +0.290] |
| Mixtral-8x7B (BOS) | f1 | names=True | 24 | L20E001 | +0.025 [+0.009, +0.037] |
| Mixtral-8x7B (BOS) | f1 | names=True | 24 | L20E000 | -0.008 [-0.040, +0.023] |
| Mixtral-8x7B (BOS) | f1 | names=True | 24 | L19E002 | +0.019 [-0.024, +0.066] |
| Mixtral-8x7B (BOS) | f1 | names=True | 24 | L21E001 | +0.003 [-0.020, +0.029] |
| Mixtral-8x7B (BOS) | f1 | names=True | 24 | L18E001 | +0.016 [-0.037, +0.083] |
| Mixtral-8x7B (BOS) | f1 | names=True | 24 | pop:k10 | +0.101 [+0.012, +0.182] |
| Mixtral-8x7B (BOS) | dir_f1 | names=False | 186 | L20E001 | -0.005 [-0.021, +0.010] |
| Mixtral-8x7B (BOS) | dir_f1 | names=False | 186 | L20E000 | +0.002 [-0.004, +0.008] |
| Mixtral-8x7B (BOS) | dir_f1 | names=False | 186 | L19E002 | +0.000 [-0.006, +0.006] |
| Mixtral-8x7B (BOS) | dir_f1 | names=False | 186 | L21E001 | -0.001 [-0.011, +0.009] |
| Mixtral-8x7B (BOS) | dir_f1 | names=False | 186 | L18E001 | +0.000 [-0.007, +0.007] |
| Mixtral-8x7B (BOS) | dir_f1 | names=False | 186 | pop:k10 | +0.248 [+0.204, +0.291] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=neutral | 90 | L20E001 | -0.007 [-0.034, +0.018] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=neutral | 90 | L20E000 | +0.002 [-0.007, +0.011] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=neutral | 90 | L19E002 | +0.002 [-0.007, +0.011] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=neutral | 90 | L21E001 | +0.001 [-0.011, +0.011] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=neutral | 90 | L18E001 | -0.003 [-0.015, +0.009] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=neutral | 90 | pop:k10 | +0.277 [+0.211, +0.340] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=one-sided | 40 | L20E001 | -0.016 [-0.048, +0.013] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=one-sided | 40 | L20E000 | -0.003 [-0.013, +0.006] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=one-sided | 40 | L19E002 | -0.001 [-0.015, +0.011] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=one-sided | 40 | L21E001 | +0.007 [-0.010, +0.025] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=one-sided | 40 | L18E001 | +0.000 [-0.013, +0.012] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=one-sided | 40 | pop:k10 | +0.169 [+0.097, +0.240] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=reverse | 54 | L20E001 | +0.005 [-0.023, +0.031] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=reverse | 54 | L20E000 | +0.007 [-0.005, +0.021] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=reverse | 54 | L19E002 | +0.001 [-0.010, +0.011] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=reverse | 54 | L21E001 | -0.011 [-0.034, +0.013] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=reverse | 54 | L18E001 | +0.004 [-0.007, +0.014] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_assoc=reverse | 54 | pop:k10 | +0.237 [+0.148, +0.329] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_local_assoc=False | 148 | L20E001 | -0.004 [-0.024, +0.015] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_local_assoc=False | 148 | L20E000 | +0.003 [-0.004, +0.010] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_local_assoc=False | 148 | L19E002 | +0.002 [-0.005, +0.008] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_local_assoc=False | 148 | L21E001 | -0.000 [-0.010, +0.010] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_local_assoc=False | 148 | L18E001 | +0.001 [-0.007, +0.008] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_local_assoc=False | 148 | pop:k10 | +0.259 [+0.209, +0.307] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_local_assoc=True | 40 | L20E001 | -0.006 [-0.031, +0.017] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_local_assoc=True | 40 | L20E000 | -0.001 [-0.013, +0.011] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_local_assoc=True | 40 | L19E002 | -0.001 [-0.012, +0.011] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_local_assoc=True | 40 | L21E001 | -0.005 [-0.033, +0.015] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_local_assoc=True | 40 | L18E001 | -0.003 [-0.016, +0.010] |
| Mixtral-8x7B (BOS) | dir_f1 | stratum_local_assoc=True | 40 | pop:k10 | +0.189 [+0.088, +0.270] |
| Mixtral-8x7B (BOS) | dir_opt | names=False | 186 | L20E000 | +0.025 [+0.017, +0.035] |
| Mixtral-8x7B (BOS) | dir_opt | names=False | 186 | L19E002 | -0.001 [-0.004, +0.002] |
| Mixtral-8x7B (BOS) | dir_opt | names=False | 186 | L21E001 | +0.000 [-0.002, +0.002] |
| Mixtral-8x7B (BOS) | dir_opt | names=False | 186 | L18E001 | -0.000 [-0.003, +0.003] |
| Mixtral-8x7B (BOS) | dir_opt | names=False | 186 | pop:k10 | +0.173 [+0.150, +0.195] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=neutral | 90 | L20E000 | +0.030 [+0.018, +0.043] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=neutral | 90 | L19E002 | -0.002 [-0.007, +0.002] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=neutral | 90 | L21E001 | +0.001 [-0.002, +0.005] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=neutral | 90 | L18E001 | +0.001 [-0.004, +0.006] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=neutral | 90 | pop:k10 | +0.196 [+0.160, +0.232] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=one-sided | 40 | L20E000 | +0.025 [+0.006, +0.050] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=one-sided | 40 | L19E002 | +0.002 [-0.003, +0.007] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=one-sided | 40 | L21E001 | +0.003 [-0.001, +0.006] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=one-sided | 40 | L18E001 | -0.001 [-0.007, +0.004] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=one-sided | 40 | pop:k10 | +0.117 [+0.071, +0.160] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=reverse | 54 | L20E000 | +0.017 [+0.002, +0.032] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=reverse | 54 | L19E002 | -0.001 [-0.004, +0.003] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=reverse | 54 | L21E001 | -0.002 [-0.006, +0.002] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=reverse | 54 | L18E001 | -0.001 [-0.006, +0.004] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_assoc=reverse | 54 | pop:k10 | +0.174 [+0.142, +0.206] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_local_assoc=False | 148 | L20E000 | +0.024 [+0.016, +0.034] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_local_assoc=False | 148 | L19E002 | -0.001 [-0.004, +0.003] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_local_assoc=False | 148 | L21E001 | +0.002 [-0.001, +0.004] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_local_assoc=False | 148 | L18E001 | -0.001 [-0.004, +0.003] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_local_assoc=False | 148 | pop:k10 | +0.175 [+0.150, +0.198] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_local_assoc=True | 40 | L20E000 | +0.028 [+0.003, +0.054] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_local_assoc=True | 40 | L19E002 | -0.003 [-0.010, +0.003] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_local_assoc=True | 40 | L21E001 | -0.004 [-0.009, +0.001] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_local_assoc=True | 40 | L18E001 | +0.001 [-0.007, +0.009] |
| Mixtral-8x7B (BOS) | dir_opt | stratum_local_assoc=True | 40 | pop:k10 | +0.156 [+0.099, +0.215] |
| Mixtral-8x7B (BOS) | f2e | names=False | 32 | L19E004 | +0.032 [+0.005, +0.060] |
| Mixtral-8x7B (BOS) | f2e | names=False | 32 | L20E000 | -0.002 [-0.009, +0.005] |
| Mixtral-8x7B (BOS) | f2e | names=False | 32 | L19E002 | +0.001 [-0.005, +0.007] |
| Mixtral-8x7B (BOS) | f2e | names=False | 32 | L21E001 | +0.009 [+0.000, +0.019] |
| Mixtral-8x7B (BOS) | f2e | names=False | 32 | L18E001 | -0.004 [-0.016, +0.009] |
| Mixtral-8x7B (BOS) | f2e | names=False | 32 | pop:k10 | +0.018 [-0.041, +0.069] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | stratum_assoc=neutral | 234 | L19E006 | +0.043 [+0.026, +0.060] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | stratum_assoc=neutral | 234 | L20E000 | +0.010 [-0.018, +0.039] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | stratum_assoc=neutral | 234 | L19E002 | -0.007 [-0.014, -0.001] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | stratum_assoc=neutral | 234 | L21E001 | +0.001 [-0.004, +0.005] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | stratum_assoc=neutral | 234 | L18E001 | +0.001 [-0.004, +0.006] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | stratum_assoc=neutral | 234 | pop:k10 | +0.075 [+0.023, +0.123] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | stratum_assoc=reverse | 14 | L19E006 | +0.028 [-0.003, +0.055] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | stratum_assoc=reverse | 14 | L20E000 | +0.034 [-0.003, +0.083] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | stratum_assoc=reverse | 14 | L19E002 | -0.008 [-0.019, +0.004] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | stratum_assoc=reverse | 14 | L21E001 | +0.003 [-0.008, +0.012] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | stratum_assoc=reverse | 14 | L18E001 | -0.003 [-0.028, +0.009] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | stratum_assoc=reverse | 14 | pop:k10 | +0.067 [-0.045, +0.171] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=neutral | 222 | L19E006 | +0.036 [+0.011, +0.062] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=neutral | 222 | L20E000 | +0.019 [-0.009, +0.054] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=neutral | 222 | L19E002 | +0.003 [-0.002, +0.007] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=neutral | 222 | L21E001 | +0.004 [+0.000, +0.007] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=neutral | 222 | L18E001 | -0.001 [-0.007, +0.004] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=neutral | 222 | pop:k10 | +0.057 [+0.010, +0.105] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=one-sided | 18 | L19E006 | +0.054 [+0.029, +0.108] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=one-sided | 18 | L20E000 | +0.027 [-0.015, +0.062] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=one-sided | 18 | L19E002 | +0.004 [+0.000, +0.011] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=one-sided | 18 | L21E001 | +0.002 [-0.005, +0.018] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=one-sided | 18 | L18E001 | +0.011 [+0.000, +0.020] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=one-sided | 18 | pop:k10 | +0.060 [+0.042, +0.099] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=reverse | 16 | L19E006 | +0.029 [+0.007, +0.045] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=reverse | 16 | L20E000 | +0.062 [+0.036, +0.089] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=reverse | 16 | L19E002 | +0.009 [+0.005, +0.015] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=reverse | 16 | L21E001 | +0.007 [+0.000, +0.013] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=reverse | 16 | L18E001 | +0.009 [+0.000, +0.019] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | stratum_assoc=reverse | 16 | pop:k10 | +0.069 [+0.000, +0.149] |
| Mixtral-8x7B (BOS) | cp_vt_orig | stratum_assoc=neutral | 232 | L19E006 | +0.015 [-0.006, +0.033] |
| Mixtral-8x7B (BOS) | cp_vt_orig | stratum_assoc=neutral | 232 | L20E000 | +0.024 [-0.007, +0.066] |
| Mixtral-8x7B (BOS) | cp_vt_orig | stratum_assoc=neutral | 232 | L19E002 | -0.002 [-0.010, +0.006] |
| Mixtral-8x7B (BOS) | cp_vt_orig | stratum_assoc=neutral | 232 | L21E001 | +0.002 [-0.005, +0.011] |
| Mixtral-8x7B (BOS) | cp_vt_orig | stratum_assoc=neutral | 232 | L18E001 | +0.006 [-0.001, +0.015] |
| Mixtral-8x7B (BOS) | cp_vt_orig | stratum_assoc=neutral | 232 | pop:k10 | +0.088 [-0.006, +0.202] |
| Mixtral-8x7B (BOS) | cp_vt_orig | order=1 | 130 | L19E006 | +0.009 [-0.011, +0.029] |
| Mixtral-8x7B (BOS) | cp_vt_orig | order=1 | 130 | L20E000 | +0.015 [-0.016, +0.050] |
| Mixtral-8x7B (BOS) | cp_vt_orig | order=1 | 130 | L19E002 | -0.002 [-0.011, +0.006] |
| Mixtral-8x7B (BOS) | cp_vt_orig | order=1 | 130 | L21E001 | +0.002 [-0.006, +0.011] |
| Mixtral-8x7B (BOS) | cp_vt_orig | order=1 | 130 | L18E001 | +0.004 [-0.004, +0.012] |
| Mixtral-8x7B (BOS) | cp_vt_orig | order=1 | 130 | pop:k10 | +0.069 [-0.011, +0.157] |
| Mixtral-8x7B (BOS) | cp_vt_orig | order=2 | 126 | L19E006 | +0.026 [+0.002, +0.048] |
| Mixtral-8x7B (BOS) | cp_vt_orig | order=2 | 126 | L20E000 | +0.026 [-0.012, +0.074] |
| Mixtral-8x7B (BOS) | cp_vt_orig | order=2 | 126 | L19E002 | -0.002 [-0.009, +0.007] |
| Mixtral-8x7B (BOS) | cp_vt_orig | order=2 | 126 | L21E001 | +0.004 [-0.003, +0.014] |
| Mixtral-8x7B (BOS) | cp_vt_orig | order=2 | 126 | L18E001 | +0.007 [-0.002, +0.017] |
| Mixtral-8x7B (BOS) | cp_vt_orig | order=2 | 126 | pop:k10 | +0.108 [-0.002, +0.238] |
| Mixtral-8x7B (BOS) | cp_vt_neu | stratum_assoc=neutral | 228 | L19E006 | +0.035 [+0.011, +0.071] |
| Mixtral-8x7B (BOS) | cp_vt_neu | stratum_assoc=neutral | 228 | L20E000 | +0.000 [-0.017, +0.037] |
| Mixtral-8x7B (BOS) | cp_vt_neu | stratum_assoc=neutral | 228 | L19E002 | +0.003 [-0.001, +0.009] |
| Mixtral-8x7B (BOS) | cp_vt_neu | stratum_assoc=neutral | 228 | L21E001 | +0.012 [-0.002, +0.028] |
| Mixtral-8x7B (BOS) | cp_vt_neu | stratum_assoc=neutral | 228 | L18E001 | +0.003 [-0.003, +0.012] |
| Mixtral-8x7B (BOS) | cp_vt_neu | stratum_assoc=neutral | 228 | pop:k10 | +0.042 [-0.067, +0.141] |
| Mixtral-8x7B (BOS) | cp_vt_neu | order=1 | 130 | L19E006 | +0.035 [+0.012, +0.068] |
| Mixtral-8x7B (BOS) | cp_vt_neu | order=1 | 130 | L20E000 | -0.004 [-0.030, +0.043] |
| Mixtral-8x7B (BOS) | cp_vt_neu | order=1 | 130 | L19E002 | +0.001 [-0.003, +0.006] |
| Mixtral-8x7B (BOS) | cp_vt_neu | order=1 | 130 | L21E001 | +0.013 [+0.001, +0.025] |
| Mixtral-8x7B (BOS) | cp_vt_neu | order=1 | 130 | L18E001 | +0.001 [-0.005, +0.014] |
| Mixtral-8x7B (BOS) | cp_vt_neu | order=1 | 130 | pop:k10 | +0.051 [-0.061, +0.137] |
| Mixtral-8x7B (BOS) | cp_vt_neu | order=2 | 126 | L19E006 | +0.035 [+0.010, +0.074] |
| Mixtral-8x7B (BOS) | cp_vt_neu | order=2 | 126 | L20E000 | +0.003 [-0.008, +0.041] |
| Mixtral-8x7B (BOS) | cp_vt_neu | order=2 | 126 | L19E002 | +0.004 [-0.001, +0.010] |
| Mixtral-8x7B (BOS) | cp_vt_neu | order=2 | 126 | L21E001 | +0.013 [-0.003, +0.035] |
| Mixtral-8x7B (BOS) | cp_vt_neu | order=2 | 126 | L18E001 | +0.005 [+0.000, +0.017] |
| Mixtral-8x7B (BOS) | cp_vt_neu | order=2 | 126 | pop:k10 | +0.038 [-0.070, +0.147] |
| Mixtral-8x7B (BOS) | cp_nat | names=False | 1188 | L21E006 | +0.024 [+0.019, +0.029] |
| Mixtral-8x7B (BOS) | cp_nat | names=False | 1188 | L20E000 | +0.047 [+0.041, +0.052] |
| Mixtral-8x7B (BOS) | cp_nat | names=False | 1188 | L19E002 | -0.001 [-0.002, +0.001] |
| Mixtral-8x7B (BOS) | cp_nat | names=False | 1188 | L21E001 | +0.001 [-0.000, +0.002] |
| Mixtral-8x7B (BOS) | cp_nat | names=False | 1188 | L18E001 | +0.001 [-0.000, +0.003] |
| Mixtral-8x7B (BOS) | cp_nat | names=False | 1188 | pop:k10 | +0.192 [+0.178, +0.206] |
| Mixtral-8x7B (BOS) | cp_nat | names=True | 92 | L21E006 | +0.011 [+0.003, +0.021] |
| Mixtral-8x7B (BOS) | cp_nat | names=True | 92 | L20E000 | +0.018 [+0.003, +0.034] |
| Mixtral-8x7B (BOS) | cp_nat | names=True | 92 | L19E002 | +0.002 [-0.002, +0.007] |
| Mixtral-8x7B (BOS) | cp_nat | names=True | 92 | L21E001 | +0.002 [-0.006, +0.010] |
| Mixtral-8x7B (BOS) | cp_nat | names=True | 92 | L18E001 | +0.004 [-0.001, +0.008] |
| Mixtral-8x7B (BOS) | cp_nat | names=True | 92 | pop:k10 | +0.086 [+0.046, +0.129] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=associative | 20 | L21E006 | +0.057 [+0.007, +0.111] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=associative | 20 | L20E000 | +0.035 [-0.010, +0.076] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=associative | 20 | L19E002 | +0.000 [-0.010, +0.010] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=associative | 20 | L21E001 | +0.007 [+0.003, +0.013] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=associative | 20 | L18E001 | +0.012 [+0.004, +0.026] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=associative | 20 | pop:k10 | +0.258 [+0.180, +0.311] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=neutral | 592 | L21E006 | +0.021 [+0.015, +0.026] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=neutral | 592 | L20E000 | +0.041 [+0.034, +0.048] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=neutral | 592 | L19E002 | +0.000 [-0.002, +0.002] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=neutral | 592 | L21E001 | +0.001 [-0.001, +0.003] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=neutral | 592 | L18E001 | +0.002 [-0.000, +0.004] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=neutral | 592 | pop:k10 | +0.173 [+0.155, +0.192] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=no_probe | 88 | L21E006 | +0.020 [+0.002, +0.039] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=no_probe | 88 | L20E000 | +0.044 [+0.020, +0.068] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=no_probe | 88 | L19E002 | -0.003 [-0.007, +0.002] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=no_probe | 88 | L21E001 | -0.003 [-0.008, +0.001] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=no_probe | 88 | L18E001 | +0.002 [-0.003, +0.006] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=no_probe | 88 | pop:k10 | +0.126 [+0.079, +0.180] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=one-sided | 204 | L21E006 | +0.014 [+0.005, +0.024] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=one-sided | 204 | L20E000 | +0.053 [+0.041, +0.065] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=one-sided | 204 | L19E002 | -0.000 [-0.004, +0.003] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=one-sided | 204 | L21E001 | +0.000 [-0.003, +0.003] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=one-sided | 204 | L18E001 | +0.001 [-0.003, +0.004] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=one-sided | 204 | pop:k10 | +0.189 [+0.153, +0.224] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=reverse | 376 | L21E006 | +0.031 [+0.022, +0.040] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=reverse | 376 | L20E000 | +0.046 [+0.035, +0.057] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=reverse | 376 | L19E002 | -0.002 [-0.004, +0.001] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=reverse | 376 | L21E001 | +0.001 [-0.001, +0.004] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=reverse | 376 | L18E001 | +0.001 [-0.003, +0.004] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_assoc=reverse | 376 | pop:k10 | +0.208 [+0.183, +0.233] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_local_assoc=False | 1100 | L21E006 | +0.024 [+0.020, +0.029] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_local_assoc=False | 1100 | L20E000 | +0.043 [+0.038, +0.049] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_local_assoc=False | 1100 | L19E002 | -0.000 [-0.002, +0.001] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_local_assoc=False | 1100 | L21E001 | +0.001 [-0.000, +0.003] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_local_assoc=False | 1100 | L18E001 | +0.002 [+0.000, +0.004] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_local_assoc=False | 1100 | pop:k10 | +0.188 [+0.174, +0.202] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_local_assoc=True | 180 | L21E006 | +0.015 [+0.005, +0.024] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_local_assoc=True | 180 | L20E000 | +0.053 [+0.038, +0.070] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_local_assoc=True | 180 | L19E002 | -0.004 [-0.008, +0.000] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_local_assoc=True | 180 | L21E001 | -0.001 [-0.004, +0.003] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_local_assoc=True | 180 | L18E001 | -0.002 [-0.006, +0.003] |
| Mixtral-8x7B (BOS) | cp_nat | stratum_local_assoc=True | 180 | pop:k10 | +0.163 [+0.129, +0.196] |
| Mixtral-8x7B (BOS) | cp_nat | order=1 | 640 | L21E006 | +0.024 [+0.019, +0.028] |
| Mixtral-8x7B (BOS) | cp_nat | order=1 | 640 | L20E000 | +0.044 [+0.039, +0.050] |
| Mixtral-8x7B (BOS) | cp_nat | order=1 | 640 | L19E002 | -0.001 [-0.002, +0.001] |
| Mixtral-8x7B (BOS) | cp_nat | order=1 | 640 | L21E001 | +0.002 [-0.000, +0.003] |
| Mixtral-8x7B (BOS) | cp_nat | order=1 | 640 | L18E001 | +0.002 [-0.001, +0.004] |
| Mixtral-8x7B (BOS) | cp_nat | order=1 | 640 | pop:k10 | +0.182 [+0.169, +0.196] |
| Mixtral-8x7B (BOS) | cp_nat | order=2 | 640 | L21E006 | +0.022 [+0.018, +0.027] |
| Mixtral-8x7B (BOS) | cp_nat | order=2 | 640 | L20E000 | +0.045 [+0.039, +0.050] |
| Mixtral-8x7B (BOS) | cp_nat | order=2 | 640 | L19E002 | -0.001 [-0.003, +0.001] |
| Mixtral-8x7B (BOS) | cp_nat | order=2 | 640 | L21E001 | +0.000 [-0.002, +0.002] |
| Mixtral-8x7B (BOS) | cp_nat | order=2 | 640 | L18E001 | +0.002 [-0.000, +0.003] |
| Mixtral-8x7B (BOS) | cp_nat | order=2 | 640 | pop:k10 | +0.187 [+0.173, +0.201] |


### Caveats

- Item sets differ between conditions (each has its own margin filter and split); only the paired contrasts compare the same source items. Template conditions have about 27-28 validation seeds: their CIs are wide by design.
- F is a ratio of means over items; small baselines (low-margin conditions) inflate F's variance.
- bf16: rows of different pass compositions differ at the 0.1-logit level (null row of every table).
- The selected expert and population ranking come from p5-data's discovery runs (STR single-expert patches); knockout tests necessity on held-out pairs only.
- Figures: `results/figures/ext13ko_<proto>_<cond>.png` (targets vs same-layer controls; sets vs random), `results/figures/ext13ko_overview.png` (all conditions).
