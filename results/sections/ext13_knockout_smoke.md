**Summary.** Phase 4's knockout (engine route mask at every position, reroute: the expert leaves the router's menu and the next-best expert takes the slot; F = fraction of the clean margin lost) applied to every Phase-5 condition on clean prompts of the validation and replication pairs (both twins). Targets: each condition's own selected expert, Phase 3's WinoGrande expert (Qwen3 L41E117, Mixtral L20E000) on every condition, the CounterFact experts as cross-task controls, the condition's population top-1/3/5/10 sets, frequency-matched random sets and same-layer controls. Qwen3-30B-A3B-Base, own selections — f1pre: L37E032 +0.074 [+0.000, +0.158] (rank 1/17 in its layer), not distinguishable from 0; top-10 +0.127 (random 10-sets -0.041).

### What was run

**Design.** Models Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1 with BOS (Phases 3-4 default), bf16, engine E4 route masks (`PrefillSpec.route_mask`, reroute; Phase 4 verification `results/verify_ext9_engine_olmoe.json`). Per condition (registry `data/wino_ref/p5_conditions.json`, agent p5-data): items = clean prompts of both twins of every pair except the main discovery pairs (validation + replication splits; the discovery pairs selected the experts), Δ toward each prompt's own answer. Full scope (baseline, targets, population sets, null) = these eval items; sub scope (same-layer controls, 5 frequency-matched random sets per set size, final-position-only masks, cross-condition knockouts) = the main validation pairs. Same-layer controls for the main targets (own selected expert, WinoGrande expert): Mixtral the 7 other experts of the layer, Qwen3 the 16 experts of the layer most often routed at the final position on the condition's eval prompts (targets excluded). Random sets: each member of the population top-k replaced by one of the 4 (Qwen3) / 3 (Mixtral) experts of its layer with the nearest final-position routing frequency (Phase 4 rule). Null = the unmasked baseline recomputed in passes of another composition (noise floor). F = (Σ Δ_base − Σ Δ_ko) / Σ Δ_base; bootstrap (5,000) over the condition's units: template seeds for the semi-synthetic template conditions (vt_*, cp_vt_*; 48 name pairs per seed are not independent), source pairs for cp_nat (both prefix orders together), pairs otherwise. Cross-condition differences resample the conditions independently; paired differences use the source items present in both conditions with a joint bootstrap over shared units. Not run: zero mode (Phase 4: zero ≈ reroute for every target, `results/tables/ext9_knockout_final_zero.md`) and the wikitext NLL control (generic damage of the same experts measured in Phase 4, ≤ 0.008 nats/token for single experts).

| Model | Condition | Eval pairs (units) | Baseline Δ | Pair acc | Selected expert | Population top-3 | GPU (min) |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre (F1 pre-neutral (smoke)) | 12 (12 pair_id) | 2.47 | 1.000 | L37E032 | L37E032 L44E116 L43E070 | 1.8 |


### 1. Targets and population sets (eval items)

| model | condition | knockout | role | routed at final | F [95% CI] | pair acc base -> ko | F routed items | null F |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre | L37E032 | selected | 1.0 | +0.074 [+0.000, +0.158] | 1.000 -> 0.833 | 0.0738 | -0.019 |
| Qwen3-30B-A3B-Base | f1pre | L41E117 | wg | 0.0 | -0.004 [-0.038, +0.026] | 1.000 -> 1.000 |  | -0.019 |
| Qwen3-30B-A3B-Base | f1pre | L44E069 | cf | 0.1667 | -0.004 [-0.015, +0.009] | 1.000 -> 1.000 |  | -0.019 |
| Qwen3-30B-A3B-Base | f1pre | L42E115 | cf | 0.2917 | -0.006 [-0.035, +0.029] | 1.000 -> 1.000 | 0.0431 | -0.019 |
| Qwen3-30B-A3B-Base | f1pre | top-3 | pop |  | +0.091 [+0.019, +0.178] | 1.000 -> 0.833 |  | -0.019 |
| Qwen3-30B-A3B-Base | f1pre | top-5 | pop |  | +0.097 [+0.024, +0.188] | 1.000 -> 0.833 |  | -0.019 |
| Qwen3-30B-A3B-Base | f1pre | top-10 | pop |  | +0.127 [+0.057, +0.212] | 1.000 -> 0.833 |  | -0.019 |


### 2. Main targets against same-layer controls (validation pairs)

| model | condition | target | target F | controls mean | controls max | max control | rank | z |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre | L37E032 | 0.0055 | -0.0349 | -0.0055 | L37E104 | 1/17 | 2.229 |
| Qwen3-30B-A3B-Base | f1pre | L41E117 | 0.0221 | -0.0269 | 0.0055 | L41E055 | 1/17 | 3.2515 |


### 3. Population sets against frequency-matched random sets (validation pairs)

| model | condition | set | members | population set F | random mean | random max | rank |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre | top-1 | L37E032 | 0.0055 | -0.0155 | -0.011 | 1/6 |
| Qwen3-30B-A3B-Base | f1pre | top-3 | L37E032 L44E116 L43E070 | 0.011 | -0.0243 | -0.0166 | 1/6 |
| Qwen3-30B-A3B-Base | f1pre | top-5 | L37E032 L44E116 L43E070 L44E022 L37E107 | 0.0221 | -0.0243 | 0.0166 | 1/6 |
| Qwen3-30B-A3B-Base | f1pre | top-10 | L37E032 L44E116 L43E070 L44E022 L37E107 L39E103 L43E036 L39E004 L43E114 L43E087 | 0.0497 | -0.0409 | 0.0 | 1/6 |


### 4. P5-8: copy vs generate, and direction

Contrasts of F between conditions (a − b). C1 = entity → attribute, answer generated (dir_opt, vt_e2a_*); C2 = the same items with a prefix that names both attributes (cp_*), answer copyable; C3 = attribute → entity (fill-in-the-blank; the entity is copyable from the sentence). C2 − C1 isolates copy availability at fixed direction and format; C3 − C2 isolates direction at fixed copy availability (the fill-in wrapper still differs).

_(table not available)_


### 5. Final position only, and which expert takes the slot

| model | condition | knockout | all positions | final position only | final / all |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre | L37E032 | 0.0055 | 0.0 | 0.0 |
| Qwen3-30B-A3B-Base | f1pre | L41E117 | 0.0221 | 0.011 | 0.5 |
| Qwen3-30B-A3B-Base | f1pre | L44E069 | -0.011 | -0.0221 | 2.0 |
| Qwen3-30B-A3B-Base | f1pre | L42E115 | -0.0387 | -0.0331 | 0.8571 |
| Qwen3-30B-A3B-Base | f1pre | pop:k10 | 0.0497 | 0.0497 | 1.0 |


| model | condition | expert | routed at final | clean one-for-one swap | top replacements (share) |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre | L37E032 | 1.0 | 0.9583 | L37E078 (0.1667), L37E045 (0.125), L37E073 (0.0833) |
| Qwen3-30B-A3B-Base | f1pre | L41E117 | 0.0 | 0.0 |  |
| Qwen3-30B-A3B-Base | f1pre | L44E069 | 0.1667 | 1.0 | L44E081 (0.25), L44E072 (0.25), L44E121 (0.25) |
| Qwen3-30B-A3B-Base | f1pre | L42E115 | 0.2917 | 1.0 | L42E093 (0.2857), L42E030 (0.2857), L42E092 (0.1429) |


### 6. Self-repair: knockout loss vs the expert's own direct write

Only meaningful for experts with a sizeable direct write (W); for experts that act through later layers (W ≈ 0) the ratios are undefined (see the writer section).

| model | condition | expert | routed items | direct write W | net layer direct change N | knockout K | K / W [95% CI] | K / N [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre | L37E032 | 24 | -0.0013 | -0.0037 | 0.1823 | -141.07 [-651.02, 977.51] | -49.00 [-487.30, 583.46] |
| Qwen3-30B-A3B-Base | f1pre | L42E115 | 7 | 0.1349 | 0.1358 | 0.0893 | 0.66 [-3.31, 8.27] | 0.66 [-6.85, 4.00] |


### 8. Strata

| model | condition | stratum | items | knockout | F [95% CI] |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre | names=False | 16 | L37E032 | +0.036 [-0.037, +0.114] |
| Qwen3-30B-A3B-Base | f1pre | names=False | 16 | L41E117 | -0.011 [-0.053, +0.028] |
| Qwen3-30B-A3B-Base | f1pre | names=False | 16 | L44E069 | -0.011 [-0.021, +0.000] |
| Qwen3-30B-A3B-Base | f1pre | names=False | 16 | L42E115 | -0.005 [-0.036, +0.037] |
| Qwen3-30B-A3B-Base | f1pre | names=False | 16 | pop:k10 | +0.083 [+0.020, +0.161] |


### Caveats

- Item sets differ between conditions (each has its own margin filter and split); only the paired contrasts compare the same source items. Template conditions have about 27-28 validation seeds: their CIs are wide by design.
- F is a ratio of means over items; small baselines (low-margin conditions) inflate F's variance.
- bf16: rows of different pass compositions differ at the 0.1-logit level (null row of every table).
- The selected expert and population ranking come from p5-data's discovery runs (STR single-expert patches); knockout tests necessity on held-out pairs only.
- Figures: `results/figures/ext13ko_<proto>_<cond>_smoke.png` (targets vs same-layer controls; sets vs random), `results/figures/ext13ko_overview_smoke.png` (all conditions).
