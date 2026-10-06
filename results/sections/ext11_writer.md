**Summary.** Splitting each single-expert STR patch at the final position into its direct path (the expert's write δ_e projected on W_U[r] − W_U[r′], final norm frozen, as ext8's DLA) and the rest (total − direct = everything later layers do with it) shows that the Qwen3 experts are **writers** and the Mixtral experts **partly writers** (half to most of their effect is direct). Direct share (Σ direct / Σ total over the validation cases where the expert is routed): Qwen3 WinoGrande L41E117 **1.04 [0.99, 1.10]**, CounterFact L42E115 **0.96 [0.88, 1.06]** and L44E069 **1.32 [1.20, 1.43]** (its write is larger than its net effect: later layers undo 0.39 of 1.59 logits); Mixtral WinoGrande L20E000 **0.56 [0.52, 0.62]**, CounterFact L19E002 0.71 [0.55, 0.94], L21E001 0.87 [0.75, 1.03], L18E001 0.54 [0.43, 0.69]. Over all clean-active experts the direct share grows with depth (middle band / late band: Qwen3 CounterFact 0.54 / 0.99; Qwen3 WinoGrande 0.60 / 1.09; Mixtral CounterFact 0.80 / 0.80; Mixtral WinoGrande 0.61 / 1.23); experts in the first half of the network contribute ≤ 0.03 of the drop (Mixtral WinoGrande 0.11), almost only indirectly, and above a crossover layer (Qwen3 CounterFact L42, Qwen3 WinoGrande L41, Mixtral WinoGrande L22) experts write more than their net effect, so downstream layers partly cancel late writes. The patch-free DLA ranks experts well because the largest total effects belong to writers: the top-10 experts by direct and by total effect overlap in 10/10 (Qwen3 CounterFact), 9/10 (Qwen3 WinoGrande), 10/10 (Mixtral CounterFact), 7/10 (Mixtral WinoGrande). Routing is set by the local context and the write by the full context: L41E117 is routed at the final position of 84 % of the WinoGrande prompts and of 86 % of their context-free local prompts ('The bag was too'), but its DLA there is +0.08 vs +0.54 logits; in wikitext it fires on 7 % of tokens but on 66 % of copulas and 59 % of degree adverbs (a predicate-complement slot detector), never at IOI's final token. Mixtral's L20E000 is routed at 96 % of WinoGrande final positions and 95 % of the local prompts (DLA +0.32 vs +0.06), and at 22 % of wikitext tokens. The CounterFact experts are broad: L44E069 / L42E115 are routed at 45 / 40 % of wikitext tokens, rising to 94 / 95 % before CounterFact place names, and L44E069 writes most for place relations (DLA when routed +1.08 logits vs +0.01 for occupations). In vocabulary space the writes are answer-shaped: L41E117's δ_e puts the correct trigger at median rank 98 of 151,936 tokens (top-10 in 30 % of rows) and its 50 most promoted tokens are WinoGrande trigger words 12× more often than the vocabulary base rate (antonym axes such as small / smaller / larger); L44E069's δ_e puts the true object at median rank 5758 and its clean output c_e promotes CounterFact object tokens 28× over base (place-name pieces). In Mixtral the clean outputs are even more answer-like: L20E000's c_e ranks the correct trigger at median 10 of 32,000 (top-10 in 51 % of prompts; δ_e 521), L21E001's c_e the true object at 7. For single experts the exact-norm direct effect is 0.996–0.999 times the frozen-norm DLA (mean |difference| 0.004–0.013 logits). Freezing the final norm is harmless at these scales: for the summed MoE writes the exact-norm and frozen-norm direct effects differ by ≤ 0.006 of the drop on the mean (median per-row |error| 0.006–0.018), and at the last layer, where total − direct is the norm error alone, total ≈ direct (slope 1.00, 0.99, 1.07, 1.02).

### What was run

**Part A (CPU, existing rows).** For every STR row of the four ext8 add-back tasks (CounterFact STR: Direction-6 cases and donors; WinoGrande STR: the 512 directed cases of the main family) and every clean-active (layer, expert) at the final position: total T = single-expert patch rescue (ext6 / ext7 `str_expert_rows`, parent = corrupted run, c_e := c_e(clean)), direct D = ext8's DLA of δ_e = c_e(clean) − c_e(corrupt), (δ_e ⊙ γ)·(W_U[r] − W_U[r′]) / rms(h_final, corrupted run) (`addback_dla.parquet`, pass 0), indirect I = T − D. I collects everything that happens after layer l at the final position (later attention reads the changed residual through the final token's own query, key and value; later routers and experts see a different input) plus the final-norm nonlinearity, which D ignores. At the last layer there is no later computation, so T − D there measures the frozen-norm error plus pass noise. CounterFact values are donor means per case, bootstrap over cases; WinoGrande directed cases, bootstrap over pairs; 2,000 resamples. Population (band, layer) values use all cases (no selection involved); expert values use validation cases (the experts and the ext8 population ranking were chosen on discovery). Share = Σ D / Σ T over the same units; slope = OLS of T on D (T carries the pass noise, D is nearly noise-free).

**Part C (GPU, prefill only, existing DiagSpec).** `scripts/ext11_writer_routing.py`: prompt sets = WinoGrande main (the 512 clean prompts of the main family's directed cases), their context-free local prompts (`local_prompt()` of `scripts/ext7_wino_scan.py`, both answers; true = own trigger, foil = twin trigger), the remaining margin-pool pairs and their local prompts, CounterFact STR (Direction-6 clean prompts and donors), the 1,024 clean prompts of the base CounterFact filter scan, the 1,600 IOI clean prompts (true = IO, foil = S) and 1,100 consecutive wikitext-103 test windows of 127 tokens (Mixtral: BOS prepended). Recorded: final-position routing at every layer, the routed experts' own DLA at the final position (`contrib_dla`, c_e itself, norm frozen at the prompt's own scale) and the routing of every token at the layers of the targets and both tasks' population top-10 (`route_all_layers`). wikitext contexts use heuristic word lists (no POS tagger installed): copulas (was, is, were, be, became, seemed, ...), degree adverbs (too, very, so, quite, more, ...), determiners, prepositions; 'WinoGrande trigger vocabulary' = all sentence-final trigger words of the model's W1-W6 pairs; 'CF place name' = a target_true / target_new string of a CounterFact place relation beginning at the next word. Qwen3: 13,187 prompts / 275,016 tokens in 3 prefill passes (2.7 GPU min; wino_pool 3686, wino_pool_local 3686, ioi 1600, wiki 1100, cf_str 1067, cf_scan 1024, wino_main 512, wino_local 512); all-token routing at layers 31, 34, 39, 40, 41, 42, 43, 44; final-position routing at the WinoGrande main prompts agrees with the ext7 source run for 99.3 % of (prompt, expert) pairs; Mixtral (BOS): 9,399 prompts / 245,441 tokens in 3 prefill passes (4.8 GPU min; wino_pool 1790, wino_pool_local 1790, ioi 1600, wiki 1100, cf_str 1071, cf_scan 1024, wino_main 512, wino_local 512); all-token routing at layers 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28; final-position routing at the WinoGrande main prompts agrees with the ext7 source run for 99.7 % of (prompt, expert) pairs.

**Part B (GPU, one prefill pass per model, ext9 engine E4c `DiagSpec.contrib_final_vectors`).** `scripts/ext11_writer_vocab.py`: prompts = CounterFact STR clean prompts and every selected donor, the 512 WinoGrande main prompts (the corrupted prompt of a directed case is its twin's clean prompt), their local prompts and the first 400 IOI clean prompts; experts = targets + both tasks' population top-10. Vectors at the final position: δ_e = c_e(clean) − c_e(corrupt) per STR row where e is clean-active (c_e(corrupt) = 0 if not routed), projected as ((δ_e ⊙ γ) W_U^T) / rms(h_final, corrupted run), i.e. ext8's DLA over the whole vocabulary; c_e(clean) projected at the prompt's own scale. Per vector: rank of r among all tokens (1 = most promoted) and of r′ from the bottom, top-20 promoted / suppressed tokens, class means (z = (class mean − vocabulary mean) / SD of the projection) and top-50 class shares for the WinoGrande trigger vocabulary (every trigger of the model's W1-W6 pairs), the first tokens of CounterFact objects per relation group (target_true and target_new strings) and the IOI names, rank of r within its own class; exact-norm direct effect Δ(norm(h_corrupt + δ_e)) − Δ(norm(h_corrupt)) through the real final RMSNorm (fp32, offline from `resid_final`) vs the frozen DLA. Analysis `scripts/ext11_writer_vocab_analyze.py`. Qwen3: 2,491 prompts in one prefill pass (1.0 GPU min, peak 7.1 GB), 23,709 vectors; same-pass check frozen projection vs the engine's contrib_dla max |diff| 0.0e+00 logits; Mixtral (BOS): 2,495 prompts in one prefill pass (1.7 GPU min, peak 9.2 GB), 25,366 vectors; same-pass check frozen projection vs the engine's contrib_dla max |diff| 0.0e+00 logits.

### A1. Selected experts and each task's population top-10: total, direct, indirect

| model | task | expert | role | active cases | total T (logits) | direct D (logits) | indirect I (logits) | T / drop | D / drop | direct share D/T | r(D, T) | slope T on D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | CounterFact | L44E069 | own target, pop #1 | 96/108 | +1.21 [+0.92, +1.54] | +1.59 [+1.18, +2.09] | -0.39 [-0.58, -0.21] | +0.101 [+0.078, +0.126] | +0.133 [+0.100, +0.171] | 1.32 [1.20, 1.43] | 0.95 [0.89, 0.98] | 0.65 [0.60, 0.75] |
| Qwen3-30B-A3B-Base | CounterFact | L42E115 | own target, pop #2 | 102/108 | +0.89 [+0.72, +1.05] | +0.85 [+0.71, +1.00] | +0.03 [-0.05, +0.12] | +0.075 [+0.061, +0.089] | +0.072 [+0.060, +0.084] | 0.96 [0.88, 1.06] | 0.85 [0.78, 0.90] | 0.96 [0.80, 1.11] |
| Qwen3-30B-A3B-Base | CounterFact | L41E117 | other-task target | 16/108 | +0.01 [-0.04, +0.05] | +0.00 [-0.01, +0.02] | +0.00 [-0.04, +0.04] | +0.001 [-0.004, +0.005] | +0.000 [-0.001, +0.002] | 0.68 [-4.76, 5.39] | 0.37 [-0.21, 0.68] | 1.02 [-0.50, 2.51] |
| Qwen3-30B-A3B-Base | CounterFact | L43E046 | pop #3 | 72/108 | +0.36 [+0.18, +0.56] | +0.37 [+0.19, +0.58] | -0.01 [-0.20, +0.13] | +0.029 [+0.015, +0.044] | +0.030 [+0.016, +0.045] | 1.02 [0.67, 1.90] | 0.62 [-0.09, 0.96] | 0.62 [-0.08, 1.13] |
| Qwen3-30B-A3B-Base | CounterFact | L43E005 | pop #4 | 69/108 | +0.40 [+0.20, +0.62] | +0.43 [+0.23, +0.67] | -0.04 [-0.10, +0.03] | +0.031 [+0.016, +0.048] | +0.034 [+0.019, +0.053] | 1.09 [0.94, 1.30] | 0.96 [0.87, 0.98] | 0.92 [0.75, 1.08] |
| Qwen3-30B-A3B-Base | CounterFact | L40E127 | pop #5 | 78/108 | +0.30 [+0.20, +0.42] | +0.29 [+0.20, +0.40] | +0.01 [-0.04, +0.07] | +0.023 [+0.015, +0.033] | +0.022 [+0.015, +0.031] | 0.96 [0.80, 1.17] | 0.86 [0.69, 0.94] | 0.91 [0.63, 1.18] |
| Qwen3-30B-A3B-Base | CounterFact | L44E006 | pop #6 | 63/108 | +0.48 [+0.27, +0.71] | +0.64 [+0.36, +0.99] | -0.16 [-0.30, -0.04] | +0.044 [+0.025, +0.065] | +0.058 [+0.033, +0.089] | 1.33 [1.10, 1.60] | 0.94 [0.89, 0.97] | 0.67 [0.58, 0.84] |
| Qwen3-30B-A3B-Base | CounterFact | L41E001 | pop #7 | 90/108 | +0.24 [+0.19, +0.30] | +0.20 [+0.16, +0.24] | +0.05 [+0.01, +0.09] | +0.020 [+0.015, +0.024] | +0.016 [+0.013, +0.019] | 0.80 [0.67, 0.97] | 0.67 [0.55, 0.79] | 1.00 [0.70, 1.40] |
| Qwen3-30B-A3B-Base | CounterFact | L44E098 | pop #8 | 7/108 | +0.64 [+0.10, +1.40] | +1.10 [+0.24, +2.15] | -0.47 [-0.93, -0.07] | +0.052 [+0.009, +0.126] | +0.090 [+0.021, +0.181] | 1.74 [1.16, 3.02] | 0.91 [0.86, 1.00] | 0.64 [0.32, 0.86] |
| Qwen3-30B-A3B-Base | CounterFact | L43E104 | pop #9 | 77/108 | +0.26 [+0.15, +0.39] | +0.35 [+0.21, +0.51] | -0.09 [-0.18, -0.01] | +0.023 [+0.013, +0.035] | +0.031 [+0.018, +0.046] | 1.36 [1.05, 1.79] | 0.84 [0.65, 0.93] | 0.66 [0.43, 0.85] |
| Qwen3-30B-A3B-Base | CounterFact | L40E030 | pop #10 | 72/108 | +0.36 [+0.20, +0.58] | +0.31 [+0.20, +0.46] | +0.05 [-0.02, +0.16] | +0.031 [+0.018, +0.048] | +0.026 [+0.017, +0.036] | 0.86 [0.71, 1.10] | 0.93 [0.87, 0.97] | 1.43 [0.94, 1.74] |
| Mixtral-8x7B (BOS) | CounterFact | L19E002 | own target, pop #2 | 68/106 | +0.93 [+0.64, +1.32] | +0.66 [+0.54, +0.80] | +0.27 [+0.03, +0.58] | +0.063 [+0.044, +0.087] | +0.045 [+0.038, +0.052] | 0.71 [0.55, 0.94] | 0.62 [0.47, 0.84] | 1.54 [0.71, 2.78] |
| Mixtral-8x7B (BOS) | CounterFact | L21E001 | own target, pop #1 | 69/106 | +0.99 [+0.78, +1.18] | +0.86 [+0.73, +0.98] | +0.13 [-0.03, +0.28] | +0.069 [+0.055, +0.083] | +0.060 [+0.050, +0.070] | 0.87 [0.75, 1.03] | 0.68 [0.52, 0.83] | 1.07 [0.80, 1.34] |
| Mixtral-8x7B (BOS) | CounterFact | L18E001 | own target, pop #4 | 83/106 | +0.48 [+0.37, +0.59] | +0.26 [+0.21, +0.31] | +0.22 [+0.12, +0.32] | +0.034 [+0.026, +0.042] | +0.018 [+0.015, +0.021] | 0.54 [0.43, 0.69] | 0.45 [0.24, 0.64] | 1.07 [0.63, 1.49] |
| Mixtral-8x7B (BOS) | CounterFact | L20E000 | other-task target | 27/106 | +0.35 [+0.21, +0.54] | +0.27 [+0.19, +0.37] | +0.08 [-0.02, +0.21] | +0.028 [+0.016, +0.043] | +0.021 [+0.015, +0.028] | 0.77 [0.58, 1.10] | 0.74 [0.55, 0.91] | 1.35 [0.56, 2.12] |
| Mixtral-8x7B (BOS) | CounterFact | L20E005 | pop #3 | 58/106 | +0.58 [+0.44, +0.73] | +0.51 [+0.38, +0.66] | +0.07 [+0.01, +0.13] | +0.046 [+0.035, +0.059] | +0.040 [+0.030, +0.052] | 0.88 [0.78, 0.98] | 0.91 [0.80, 0.96] | 0.97 [0.87, 1.10] |
| Mixtral-8x7B (BOS) | CounterFact | L22E001 | pop #5 | 77/106 | +0.55 [+0.40, +0.72] | +0.50 [+0.40, +0.61] | +0.05 [-0.05, +0.16] | +0.040 [+0.029, +0.051] | +0.037 [+0.030, +0.043] | 0.91 [0.77, 1.12] | 0.75 [0.62, 0.86] | 1.08 [0.73, 1.44] |
| Mixtral-8x7B (BOS) | CounterFact | L22E005 | pop #6 | 65/106 | +0.41 [+0.27, +0.57] | +0.39 [+0.29, +0.50] | +0.03 [-0.05, +0.11] | +0.033 [+0.021, +0.045] | +0.031 [+0.022, +0.040] | 0.94 [0.79, 1.16] | 0.87 [0.73, 0.93] | 1.27 [0.80, 1.57] |
| Mixtral-8x7B (BOS) | CounterFact | L20E006 | pop #7 | 25/106 | +0.79 [+0.48, +1.11] | +0.72 [+0.44, +1.00] | +0.07 [-0.10, +0.23] | +0.051 [+0.032, +0.069] | +0.046 [+0.030, +0.062] | 0.91 [0.74, 1.14] | 0.86 [0.78, 0.95] | 0.95 [0.65, 1.28] |
| Mixtral-8x7B (BOS) | CounterFact | L28E002 | pop #8 | 62/106 | +0.36 [+0.23, +0.51] | +0.27 [+0.15, +0.41] | +0.09 [-0.00, +0.18] | +0.026 [+0.017, +0.037] | +0.020 [+0.011, +0.029] | 0.75 [0.54, 1.01] | 0.76 [0.59, 0.94] | 0.82 [0.50, 1.17] |
| Mixtral-8x7B (BOS) | CounterFact | L23E002 | pop #9 | 58/106 | +0.38 [+0.28, +0.50] | +0.31 [+0.24, +0.39] | +0.07 [+0.00, +0.14] | +0.028 [+0.020, +0.037] | +0.023 [+0.017, +0.029] | 0.82 [0.69, 0.99] | 0.81 [0.63, 0.91] | 1.23 [0.78, 1.51] |
| Mixtral-8x7B (BOS) | CounterFact | L19E006 | pop #10 | 59/106 | +0.29 [+0.21, +0.39] | +0.22 [+0.17, +0.27] | +0.07 [+0.02, +0.14] | +0.025 [+0.018, +0.034] | +0.019 [+0.014, +0.024] | 0.75 [0.62, 0.90] | 0.80 [0.66, 0.90] | 1.44 [1.06, 1.77] |
| Qwen3-30B-A3B-Base | WinoGrande | L41E117 | own target, pop #1 | 223/256 | +1.15 [+0.99, +1.32] | +1.20 [+1.02, +1.37] | -0.05 [-0.11, +0.02] | +0.140 [+0.122, +0.160] | +0.146 [+0.125, +0.167] | 1.04 [0.99, 1.10] | 0.87 [0.83, 0.90] | 0.88 [0.80, 0.99] |
| Qwen3-30B-A3B-Base | WinoGrande | L44E069 | other-task target | 7/256 | +0.34 [-0.04, +0.62] | +0.39 [+0.11, +0.62] | -0.05 [-0.22, +0.17] | +0.053 [-0.006, +0.126] | +0.061 [+0.015, +0.108] | 1.15 [-2.66, 2.52] | 0.68 [-0.76, 0.99] | 0.85 [-0.73, 2.90] |
| Qwen3-30B-A3B-Base | WinoGrande | L42E115 | other-task target | 7/256 | +1.21 [+0.47, +1.88] | +1.40 [+0.37, +2.49] | -0.19 [-0.62, +0.15] | +0.192 [+0.089, +0.237] | +0.221 [+0.075, +0.314] | 1.15 [0.82, 1.35] | 0.64 [-0.13, 0.98] | 0.57 [-0.32, 1.10] |
| Qwen3-30B-A3B-Base | WinoGrande | L43E081 | pop #2 | 253/256 | +0.71 [+0.60, +0.81] | +1.01 [+0.85, +1.16] | -0.30 [-0.37, -0.23] | +0.086 [+0.074, +0.099] | +0.123 [+0.104, +0.142] | 1.42 [1.33, 1.51] | 0.83 [0.75, 0.88] | 0.63 [0.57, 0.70] |
| Qwen3-30B-A3B-Base | WinoGrande | L39E071 | pop #3 | 205/256 | +0.56 [+0.47, +0.67] | +0.41 [+0.33, +0.48] | +0.16 [+0.10, +0.21] | +0.069 [+0.058, +0.081] | +0.050 [+0.041, +0.059] | 0.72 [0.66, 0.80] | 0.72 [0.60, 0.82] | 1.12 [0.99, 1.25] |
| Qwen3-30B-A3B-Base | WinoGrande | L44E122 | pop #4 | 223/256 | +0.34 [+0.26, +0.43] | +0.56 [+0.40, +0.73] | -0.22 [-0.32, -0.13] | +0.041 [+0.032, +0.051] | +0.067 [+0.049, +0.088] | 1.64 [1.44, 1.83] | 0.86 [0.80, 0.91] | 0.45 [0.41, 0.51] |
| Qwen3-30B-A3B-Base | WinoGrande | L41E053 | pop #5 | 246/256 | +0.28 [+0.22, +0.33] | +0.33 [+0.24, +0.42] | -0.05 [-0.10, -0.01] | +0.034 [+0.027, +0.040] | +0.040 [+0.030, +0.051] | 1.18 [1.02, 1.35] | 0.79 [0.73, 0.84] | 0.56 [0.49, 0.67] |
| Qwen3-30B-A3B-Base | WinoGrande | L34E119 | pop #6 | 196/256 | +0.49 [+0.36, +0.63] | +0.26 [+0.20, +0.32] | +0.24 [+0.13, +0.35] | +0.059 [+0.044, +0.075] | +0.031 [+0.025, +0.037] | 0.52 [0.43, 0.64] | 0.50 [0.40, 0.64] | 1.62 [1.24, 1.98] |
| Qwen3-30B-A3B-Base | WinoGrande | L31E124 | pop #7 | 150/256 | +0.34 [+0.16, +0.55] | +0.18 [+0.10, +0.26] | +0.17 [+0.05, +0.31] | +0.042 [+0.021, +0.066] | +0.022 [+0.013, +0.032] | 0.51 [0.39, 0.76] | 0.67 [0.57, 0.75] | 2.03 [1.46, 2.60] |
| Qwen3-30B-A3B-Base | WinoGrande | L43E114 | pop #8 | 208/256 | +0.31 [+0.23, +0.38] | +0.37 [+0.28, +0.47] | -0.06 [-0.11, -0.02] | +0.037 [+0.028, +0.046] | +0.045 [+0.035, +0.056] | 1.20 [1.06, 1.37] | 0.80 [0.64, 0.88] | 0.74 [0.62, 0.89] |
| Qwen3-30B-A3B-Base | WinoGrande | L40E040 | pop #9 | 247/256 | +0.18 [+0.13, +0.25] | +0.15 [+0.10, +0.20] | +0.04 [-0.00, +0.07] | +0.023 [+0.015, +0.030] | +0.018 [+0.013, +0.024] | 0.81 [0.67, 1.00] | 0.72 [0.59, 0.81] | 1.08 [0.87, 1.29] |
| Qwen3-30B-A3B-Base | WinoGrande | L41E062 | pop #10 | 210/256 | +0.25 [+0.19, +0.30] | +0.25 [+0.19, +0.31] | -0.00 [-0.05, +0.04] | +0.030 [+0.024, +0.037] | +0.030 [+0.024, +0.038] | 1.01 [0.84, 1.23] | 0.55 [0.36, 0.72] | 0.58 [0.32, 0.85] |
| Mixtral-8x7B (BOS) | WinoGrande | L20E000 | own target, pop #1 | 246/256 | +1.15 [+1.06, +1.25] | +0.65 [+0.58, +0.72] | +0.50 [+0.43, +0.57] | +0.150 [+0.140, +0.160] | +0.084 [+0.077, +0.092] | 0.56 [0.52, 0.62] | 0.55 [0.45, 0.64] | 0.84 [0.67, 1.03] |
| Mixtral-8x7B (BOS) | WinoGrande | L19E002 | other-task target | 6/256 | +0.19 [-0.10, +0.47] | +0.09 [-0.00, +0.18] | +0.10 [-0.11, +0.29] | +0.016 [-0.008, +0.043] | +0.007 [-0.000, +0.017] | 0.45 [-0.67, nan] | 0.92 [-0.76, 0.98] | 2.82 [-17.14, 3.25] |
| Mixtral-8x7B (BOS) | WinoGrande | L21E001 | other-task target | 12/256 | +0.14 [+0.05, +0.23] | +0.11 [+0.07, +0.14] | +0.03 [-0.02, +0.11] | +0.019 [+0.008, +0.029] | +0.015 [+0.010, +0.022] | 0.79 [0.53, 1.39] | 0.55 [0.30, 0.84] | 1.81 [0.56, 3.01] |
| Mixtral-8x7B (BOS) | WinoGrande | L18E001 | other-task target | 3/256 | +0.54 [+0.12, +0.75] | +0.15 [-0.06, +0.25] | +0.39 [+0.19, +0.50] | +0.118 [+0.039, +0.142] | +0.032 [-0.020, +0.048] | 0.27 [-0.51, 0.34] | 0.94 [-1.00, 0.94] | 1.95 [-31.70, 1.95] |
| Mixtral-8x7B (BOS) | WinoGrande | L19E006 | pop #2 | 256/256 | +0.97 [+0.89, +1.04] | +0.42 [+0.37, +0.46] | +0.55 [+0.48, +0.61] | +0.126 [+0.117, +0.134] | +0.054 [+0.049, +0.059] | 0.43 [0.39, 0.47] | 0.43 [0.28, 0.56] | 0.83 [0.57, 1.09] |
| Mixtral-8x7B (BOS) | WinoGrande | L21E006 | pop #3 | 231/256 | +0.70 [+0.62, +0.77] | +0.47 [+0.42, +0.53] | +0.22 [+0.18, +0.27] | +0.090 [+0.082, +0.099] | +0.061 [+0.055, +0.068] | 0.68 [0.63, 0.73] | 0.71 [0.60, 0.79] | 1.11 [0.93, 1.31] |
| Mixtral-8x7B (BOS) | WinoGrande | L26E002 | pop #4 | 237/256 | +0.63 [+0.54, +0.73] | +0.92 [+0.77, +1.08] | -0.29 [-0.35, -0.23] | +0.082 [+0.072, +0.093] | +0.120 [+0.103, +0.138] | 1.46 [1.40, 1.53] | 0.92 [0.89, 0.94] | 0.63 [0.59, 0.66] |
| Mixtral-8x7B (BOS) | WinoGrande | L18E005 | pop #5 | 240/256 | +0.38 [+0.30, +0.46] | +0.15 [+0.11, +0.18] | +0.23 [+0.18, +0.29] | +0.049 [+0.039, +0.059] | +0.019 [+0.015, +0.023] | 0.39 [0.35, 0.43] | 0.85 [0.79, 0.89] | 2.23 [1.97, 2.49] |
| Mixtral-8x7B (BOS) | WinoGrande | L16E007 | pop #6 | 248/256 | +0.46 [+0.40, +0.52] | +0.09 [+0.07, +0.11] | +0.37 [+0.31, +0.42] | +0.060 [+0.051, +0.069] | +0.012 [+0.010, +0.015] | 0.20 [0.17, 0.24] | 0.45 [0.31, 0.55] | 1.67 [1.22, 2.07] |
| Mixtral-8x7B (BOS) | WinoGrande | L24E002 | pop #7 | 220/256 | +0.64 [+0.55, +0.74] | +0.65 [+0.56, +0.74] | -0.01 [-0.05, +0.03] | +0.084 [+0.072, +0.095] | +0.086 [+0.073, +0.097] | 1.02 [0.96, 1.09] | 0.85 [0.81, 0.88] | 0.96 [0.81, 1.08] |
| Mixtral-8x7B (BOS) | WinoGrande | L22E005 | pop #8 | 94/256 | +0.74 [+0.50, +0.97] | +1.11 [+0.71, +1.51] | -0.37 [-0.55, -0.19] | +0.083 [+0.058, +0.105] | +0.125 [+0.083, +0.163] | 1.51 [1.35, 1.63] | 0.88 [0.85, 0.90] | 0.58 [0.53, 0.64] |
| Mixtral-8x7B (BOS) | WinoGrande | L21E004 | pop #9 | 179/256 | +0.60 [+0.50, +0.70] | +0.45 [+0.39, +0.51] | +0.15 [+0.10, +0.21] | +0.077 [+0.066, +0.088] | +0.058 [+0.050, +0.065] | 0.75 [0.70, 0.81] | 0.82 [0.76, 0.88] | 1.37 [1.19, 1.54] |
| Mixtral-8x7B (BOS) | WinoGrande | L25E002 | pop #10 | 201/256 | +0.42 [+0.35, +0.49] | +0.55 [+0.47, +0.64] | -0.13 [-0.17, -0.09] | +0.056 [+0.047, +0.065] | +0.073 [+0.063, +0.085] | 1.32 [1.22, 1.43] | 0.78 [0.65, 0.88] | 0.73 [0.57, 0.89] |

Validation cases where the expert is clean-active at the final position (CounterFact: donor mean per case; WinoGrande: directed cases, bootstrap over pairs). T = single-expert STR patch rescue (source run), D = DLA of delta_e with the final norm frozen at the corrupted run (ext8), I = T - D. Fractions = ratio of means over the active cases. pop #k = rank in the task's ext8 population ranking (discovery all-case single rescue).

![Total vs direct per (case, expert)](figures/ext11_A_scatter.png)

### A2. By depth

| model | task | band | layers | pairs | sum T / drop | sum D / drop | sum I / drop | share D/T | r(D, T) | slope T on D | mean T - D (logits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | CounterFact | all | 0-47 | 82560 | +0.566 [+0.482, +0.656] | +0.459 [+0.434, +0.484] | +0.107 [+0.032, +0.186] | 0.81 [0.71, 0.93] | 0.81 [0.78, 0.84] | 0.80 [0.76, 0.84] | +0.003 |
| Qwen3-30B-A3B-Base | CounterFact | early | 0-23 | 41280 | +0.023 [-0.028, +0.074] | +0.001 [-0.002, +0.004] | +0.022 [-0.029, +0.073] | 0.04 [-0.53, 0.53] | 0.10 [0.05, 0.15] | 0.62 [0.30, 1.04] | +0.001 |
| Qwen3-30B-A3B-Base | CounterFact | middle | 24-39 | 27520 | +0.134 [+0.096, +0.173] | +0.073 [+0.066, +0.080] | +0.061 [+0.025, +0.099] | 0.54 [0.43, 0.75] | 0.40 [0.35, 0.44] | 0.99 [0.88, 1.10] | +0.006 |
| Qwen3-30B-A3B-Base | CounterFact | late | 40-43 | 6880 | +0.294 [+0.271, +0.316] | +0.291 [+0.275, +0.308] | +0.003 [-0.015, +0.019] | 0.99 [0.94, 1.05] | 0.87 [0.82, 0.90] | 0.88 [0.81, 0.94] | +0.001 |
| Qwen3-30B-A3B-Base | CounterFact | last4 | 44-47 | 6880 | +0.115 [+0.090, +0.140] | +0.094 [+0.074, +0.113] | +0.021 [+0.005, +0.039] | 0.82 [0.70, 0.95] | 0.94 [0.93, 0.95] | 0.76 [0.72, 0.81] | +0.008 |
| Qwen3-30B-A3B-Base | CounterFact | last | 47-47 | 1720 | -0.011 [-0.024, +0.004] | -0.046 [-0.054, -0.037] | +0.035 [+0.025, +0.046] | 4.30 [-22.19, 37.06] | 0.89 [0.86, 0.92] | 1.00 [0.97, 1.02] | +0.053 |
| Mixtral-8x7B (BOS) | CounterFact | all | 0-31 | 13632 | +0.478 [+0.429, +0.531] | +0.322 [+0.299, +0.345] | +0.156 [+0.111, +0.203] | 0.67 [0.61, 0.75] | 0.80 [0.78, 0.82] | 1.02 [0.97, 1.07] | +0.031 |
| Mixtral-8x7B (BOS) | CounterFact | early | 0-15 | 6816 | +0.028 [+0.007, +0.050] | +0.009 [+0.007, +0.011] | +0.020 [-0.002, +0.041] | 0.30 [0.17, 1.08] | 0.12 [0.08, 0.16] | 0.61 [0.39, 0.82] | +0.008 |
| Mixtral-8x7B (BOS) | CounterFact | middle | 16-23 | 3408 | +0.381 [+0.351, +0.414] | +0.307 [+0.292, +0.322] | +0.075 [+0.048, +0.101] | 0.80 [0.75, 0.86] | 0.77 [0.73, 0.80] | 1.12 [1.03, 1.23] | +0.060 |
| Mixtral-8x7B (BOS) | CounterFact | late | 24-27 | 1704 | +0.111 [+0.093, +0.128] | +0.089 [+0.073, +0.104] | +0.022 [+0.013, +0.031] | 0.80 [0.73, 0.88] | 0.84 [0.80, 0.86] | 0.99 [0.91, 1.08] | +0.035 |
| Mixtral-8x7B (BOS) | CounterFact | last4 | 28-31 | 1704 | -0.043 [-0.067, -0.017] | -0.082 [-0.107, -0.058] | +0.040 [+0.028, +0.052] | 1.93 [1.49, 3.66] | 0.86 [0.83, 0.89] | 0.93 [0.88, 0.98] | +0.063 |
| Mixtral-8x7B (BOS) | CounterFact | last | 31-31 | 426 | -0.055 [-0.068, -0.041] | -0.055 [-0.066, -0.044] | +0.000 [-0.007, +0.007] | 1.01 [0.89, 1.16] | 0.82 [0.76, 0.87] | 1.07 [0.97, 1.16] | +0.002 |
| Qwen3-30B-A3B-Base | WinoGrande | all | 0-47 | 196608 | +1.030 [+0.820, +1.228] | +0.926 [+0.903, +0.947] | +0.105 [-0.104, +0.302] | 0.90 [0.75, 1.13] | 0.46 [0.43, 0.48] | 0.78 [0.76, 0.80] | +0.002 |
| Qwen3-30B-A3B-Base | WinoGrande | early | 0-23 | 98304 | -0.029 [-0.179, +0.119] | +0.027 [+0.022, +0.033] | -0.056 [-0.204, +0.093] | -0.95 [-4.08, 5.71] | 0.02 [0.00, 0.03] | 0.23 [0.07, 0.41] | -0.002 |
| Qwen3-30B-A3B-Base | WinoGrande | middle | 24-39 | 65536 | +0.547 [+0.445, +0.645] | +0.330 [+0.314, +0.347] | +0.217 [+0.120, +0.313] | 0.60 [0.51, 0.73] | 0.35 [0.33, 0.38] | 1.08 [1.02, 1.13] | +0.014 |
| Qwen3-30B-A3B-Base | WinoGrande | late | 40-43 | 16384 | +0.522 [+0.500, +0.544] | +0.568 [+0.546, +0.589] | -0.046 [-0.068, -0.022] | 1.09 [1.04, 1.13] | 0.83 [0.81, 0.84] | 0.75 [0.72, 0.78] | -0.012 |
| Qwen3-30B-A3B-Base | WinoGrande | last4 | 44-47 | 16384 | -0.010 [-0.038, +0.018] | +0.000 [-0.024, +0.023] | -0.010 [-0.028, +0.007] | -0.05 [-8.40, 9.23] | 0.85 [0.83, 0.86] | 0.70 [0.67, 0.73] | -0.003 |
| Qwen3-30B-A3B-Base | WinoGrande | last | 47-47 | 4096 | -0.061 [-0.078, -0.044] | -0.067 [-0.083, -0.051] | +0.006 [+0.001, +0.011] | 1.10 [1.02, 1.21] | 0.94 [0.92, 0.95] | 0.99 [0.97, 1.00] | +0.006 |
| Mixtral-8x7B (BOS) | WinoGrande | all | 0-31 | 32768 | +1.056 [+1.012, +1.100] | +0.663 [+0.644, +0.681] | +0.393 [+0.355, +0.432] | 0.63 [0.60, 0.65] | 0.79 [0.78, 0.79] | 0.82 [0.79, 0.84] | +0.048 |
| Mixtral-8x7B (BOS) | WinoGrande | early | 0-15 | 16384 | +0.111 [+0.086, +0.136] | +0.034 [+0.028, +0.040] | +0.077 [+0.054, +0.101] | 0.30 [0.24, 0.39] | 0.45 [0.39, 0.49] | 1.99 [1.82, 2.13] | +0.019 |
| Mixtral-8x7B (BOS) | WinoGrande | middle | 16-23 | 8192 | +0.831 [+0.807, +0.855] | +0.507 [+0.484, +0.530] | +0.324 [+0.299, +0.348] | 0.61 [0.59, 0.64] | 0.69 [0.67, 0.71] | 0.84 [0.79, 0.91] | +0.158 |
| Mixtral-8x7B (BOS) | WinoGrande | late | 24-27 | 4096 | +0.334 [+0.319, +0.349] | +0.412 [+0.389, +0.435] | -0.078 [-0.091, -0.064] | 1.23 [1.19, 1.27] | 0.85 [0.84, 0.86] | 0.64 [0.62, 0.67] | -0.076 |
| Mixtral-8x7B (BOS) | WinoGrande | last4 | 28-31 | 4096 | -0.220 [-0.247, -0.190] | -0.290 [-0.326, -0.253] | +0.070 [+0.057, +0.083] | 1.32 [1.27, 1.37] | 0.91 [0.90, 0.92] | 0.77 [0.75, 0.80] | +0.068 |
| Mixtral-8x7B (BOS) | WinoGrande | last | 31-31 | 1024 | -0.138 [-0.152, -0.124] | -0.140 [-0.154, -0.126] | +0.002 [-0.001, +0.004] | 1.01 [1.00, 1.03] | 0.95 [0.94, 0.97] | 1.02 [1.00, 1.03] | +0.006 |

All cases (discovery + validation; no selection is involved). Sums over every clean-active expert of the band per case, mean over cases / mean drop. r and slope over (case, expert) pairs. 'last' = the last layer only, where T - D is the frozen-norm error plus pass noise (no downstream computation).

![Per-layer sums of total and direct effects](figures/ext11_A_layers.png)

![Direct share by layer](figures/ext11_A_share_by_layer.png)

### A3. Sets, frozen-norm error, pass noise

| model | task | own targets | own: T / drop | own: share | pop top-10: T / drop | pop top-10: D / drop | pop top-10: share | all-MoE direct exact / frozen | frozen-norm err, median |err| / drop (MoE sum) | single-patch pass noise SD (logits) | r(D, T) / r(D, mean of 2 passes) / r(T, T') |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | CounterFact | L44E069, L42E115 | +0.164 [+0.137, +0.194] | 1.16 [1.08, 1.26] | +0.307 [+0.270, +0.343] | +0.345 [+0.306, +0.387] | 1.13 [1.05, 1.21] | 0.515 / 0.514 | 0.011 | 0.153 (n=1080) | 0.853 / 0.859 / 0.972 |
| Mixtral-8x7B (BOS) | CounterFact | L19E002, L21E001, L18E001 | +0.124 [+0.101, +0.149] | 0.74 [0.64, 0.86] | +0.258 [+0.226, +0.292] | +0.207 [+0.190, +0.225] | 0.80 [0.73, 0.88] | 0.352 / 0.358 | 0.009 | 0.120 (n=1060) | 0.657 / 0.668 / 0.963 |
| Qwen3-30B-A3B-Base | WinoGrande | L41E117 | +0.122 [+0.104, +0.142] | 1.04 [0.99, 1.10] | +0.480 [+0.433, +0.524] | +0.502 [+0.453, +0.550] | 1.05 [1.00, 1.10] | 0.946 / 0.952 | 0.018 | 0.305 (n=1280) | 0.423 / 0.507 / 0.830 |
| Mixtral-8x7B (BOS) | WinoGrande | L20E000 | +0.144 [+0.133, +0.155] | 0.56 [0.52, 0.61] | +0.735 [+0.694, +0.776] | +0.555 [+0.517, +0.594] | 0.75 [0.72, 0.79] | 0.700 / 0.703 | 0.006 | 0.082 (n=1280) | 0.576 / 0.581 / 0.974 |

Validation. Sets: per case sum over the listed experts that are clean-active. Frozen-norm error of the summed MoE writes (ext8 addback_direct: exact final RMSNorm vs frozen at the corrupted run, fp32, same path). Pass noise: ext8 in-pass singles (exact family, k = 1, top-10 of each first-donor row) vs the source run's single patches.

**Population rankings by total vs by direct effect** (validation, all-case means; experts routed in ≥ 5 % of the cases)

| model | task | experts | Spearman ρ | top-10 overlap | top-5 by T | top-5 by D | rank under D of the T top-3 |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | CounterFact | 1379 | 0.35 | 10/10 | L44E069, L42E115, L44E006, L43E005, L40E030 | L44E069, L42E115, L44E006, L43E005, L43E104 | L44E069 1, L42E115 2, L44E006 3 |
| Qwen3-30B-A3B-Base | WinoGrande | 1051 | 0.39 | 9/10 | L41E117, L43E081, L39E071, L34E119, L44E122 | L41E117, L43E081, L44E122, L39E071, L41E053 | L41E117 1, L43E081 2, L39E071 4 |
| Mixtral-8x7B (BOS) | CounterFact | 218 | 0.82 | 10/10 | L21E001, L19E002, L22E001, L18E001, L20E005 | L21E001, L19E002, L22E001, L20E005, L22E005 | L21E001 1, L19E002 2, L22E001 3 |
| Mixtral-8x7B (BOS) | WinoGrande | 149 | 0.87 | 7/10 | L20E000, L19E006, L21E006, L26E002, L24E002 | L26E002, L20E000, L24E002, L25E002, L21E006 | L20E000 2, L19E006 6, L21E006 5 |

### C1. Final-position routing and DLA of the target experts by prompt set

| model | expert | set | prompts | routed at final position (%) | DLA when routed (logits, true - foil) | DLA > 0 (%) | r(DLA, Delta) when routed | mean Delta of the set |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | WinoGrande main (512 prompts) | 512 | 84.4 [80.3, 88.3] | +0.54 [+0.48, +0.60] | 83 | 0.43 | +4.16 |
| Qwen3-30B-A3B-Base | L44E069 | WinoGrande main (512 prompts) | 512 | 2.7 [1.0, 4.9] | +0.08 [-0.01, +0.20] | 64 | 0.21 | +4.16 |
| Qwen3-30B-A3B-Base | L42E115 | WinoGrande main (512 prompts) | 512 | 2.0 [0.6, 3.7] | +0.51 [+0.10, +1.04] | 70 |  | +4.16 |
| Qwen3-30B-A3B-Base | L41E117 | WinoGrande local prompts (main pairs) | 512 | 86.3 [82.2, 90.0] | +0.08 [+0.05, +0.11] | 56 | 0.52 | +0.55 |
| Qwen3-30B-A3B-Base | L44E069 | WinoGrande local prompts (main pairs) | 512 | 3.1 [1.4, 5.5] | +0.01 [-0.00, +0.02] | 56 | 0.44 | +0.55 |
| Qwen3-30B-A3B-Base | L42E115 | WinoGrande local prompts (main pairs) | 512 | 2.3 [0.8, 4.1] | -0.04 [-0.11, +0.01] | 42 | -0.23 | +0.55 |
| Qwen3-30B-A3B-Base | L41E117 | WinoGrande margin pool (other pairs) | 3686 | 83.6 [81.8, 85.1] | +0.31 [+0.30, +0.33] | 75 | 0.33 | +3.78 |
| Qwen3-30B-A3B-Base | L44E069 | WinoGrande margin pool (other pairs) | 3686 | 3.4 [2.6, 4.2] | +0.15 [+0.04, +0.29] | 58 | 0.14 | +3.78 |
| Qwen3-30B-A3B-Base | L42E115 | WinoGrande margin pool (other pairs) | 3686 | 2.9 [2.3, 3.7] | +0.08 [+0.02, +0.14] | 66 | 0.17 | +3.78 |
| Qwen3-30B-A3B-Base | L41E117 | local prompts (pool pairs) | 3686 | 73.0 [71.0, 74.9] | +0.05 [+0.04, +0.06] | 54 | 0.43 | +0.43 |
| Qwen3-30B-A3B-Base | L44E069 | local prompts (pool pairs) | 3686 | 5.6 [4.7, 6.7] | -0.00 [-0.01, +0.01] | 46 | -0.02 | +0.43 |
| Qwen3-30B-A3B-Base | L42E115 | local prompts (pool pairs) | 3686 | 4.0 [3.2, 4.9] | -0.00 [-0.01, +0.01] | 48 | 0.11 | +0.43 |
| Qwen3-30B-A3B-Base | L41E117 | CounterFact STR (clean + donors), clean only | 215 | 10.7 [6.5, 14.9] | -0.00 [-0.02, +0.02] | 48 | -0.28 | +5.81 |
| Qwen3-30B-A3B-Base | L44E069 | CounterFact STR (clean + donors), clean only | 215 | 87.9 [83.3, 92.1] | +0.81 [+0.61, +1.00] | 80 | 0.25 | +5.81 |
| Qwen3-30B-A3B-Base | L42E115 | CounterFact STR (clean + donors), clean only | 215 | 95.3 [92.6, 98.1] | +0.44 [+0.38, +0.50] | 87 | 0.36 | +5.81 |
| Qwen3-30B-A3B-Base | L41E117 | CounterFact scan (1,024 clean) | 1024 | 11.0 [9.1, 12.9] | -0.00 [-0.01, +0.01] | 45 | -0.08 | +5.19 |
| Qwen3-30B-A3B-Base | L44E069 | CounterFact scan (1,024 clean) | 1024 | 89.6 [87.7, 91.4] | +0.63 [+0.56, +0.70] | 75 | 0.38 | +5.19 |
| Qwen3-30B-A3B-Base | L42E115 | CounterFact scan (1,024 clean) | 1024 | 96.4 [95.2, 97.5] | +0.35 [+0.32, +0.38] | 81 | 0.47 | +5.19 |
| Qwen3-30B-A3B-Base | L41E117 | IOI clean | 1600 | 0.0 [0.0, 0.0] | n/a |  |  | +6.27 |
| Qwen3-30B-A3B-Base | L44E069 | IOI clean | 1600 | 10.9 [9.5, 12.6] | +0.00 [-0.00, +0.01] | 47 | 0.05 | +6.27 |
| Qwen3-30B-A3B-Base | L42E115 | IOI clean | 1600 | 100.0 [100.0, 100.0] | -0.02 [-0.02, -0.01] | 43 | 0.05 | +6.27 |
| Qwen3-30B-A3B-Base | L41E117 | wikitext-103 (final token of windows) | 1100 | 6.3 [4.9, 7.7] |  |  |  |  |
| Qwen3-30B-A3B-Base | L44E069 | wikitext-103 (final token of windows) | 1100 | 46.5 [43.6, 49.5] |  |  |  |  |
| Qwen3-30B-A3B-Base | L42E115 | wikitext-103 (final token of windows) | 1100 | 41.5 [38.5, 44.5] |  |  |  |  |
| Mixtral-8x7B (BOS) | L20E000 | WinoGrande main (512 prompts) | 512 | 96.1 [93.8, 98.2] | +0.32 [+0.30, +0.35] | 86 | 0.32 | +3.89 |
| Mixtral-8x7B (BOS) | L19E002 | WinoGrande main (512 prompts) | 512 | 2.9 [1.4, 4.7] | +0.04 [+0.01, +0.06] | 67 | 0.07 | +3.89 |
| Mixtral-8x7B (BOS) | L21E001 | WinoGrande main (512 prompts) | 512 | 3.7 [1.8, 6.1] | +0.06 [+0.04, +0.09] | 63 | 0.68 | +3.89 |
| Mixtral-8x7B (BOS) | L18E001 | WinoGrande main (512 prompts) | 512 | 1.2 [0.2, 2.5] | +0.04 [-0.03, +0.10] | 67 |  | +3.89 |
| Mixtral-8x7B (BOS) | L20E000 | WinoGrande local prompts (main pairs) | 512 | 95.1 [92.4, 97.5] | +0.06 [+0.03, +0.07] | 56 | 0.46 | +0.55 |
| Mixtral-8x7B (BOS) | L19E002 | WinoGrande local prompts (main pairs) | 512 | 6.2 [3.7, 9.2] | +0.00 [-0.01, +0.02] | 47 | 0.69 | +0.55 |
| Mixtral-8x7B (BOS) | L21E001 | WinoGrande local prompts (main pairs) | 512 | 2.1 [0.8, 3.7] | +0.00 [-0.02, +0.03] | 36 | 0.71 | +0.55 |
| Mixtral-8x7B (BOS) | L18E001 | WinoGrande local prompts (main pairs) | 512 | 1.2 [0.2, 2.5] | -0.03 [-0.09, +0.00] | 33 |  | +0.55 |
| Mixtral-8x7B (BOS) | L20E000 | WinoGrande margin pool (other pairs) | 1790 | 96.6 [95.5, 97.7] | +0.31 [+0.30, +0.33] | 86 | 0.39 | +3.66 |
| Mixtral-8x7B (BOS) | L19E002 | WinoGrande margin pool (other pairs) | 1790 | 4.2 [3.1, 5.4] | +0.05 [+0.03, +0.07] | 67 | 0.41 | +3.66 |
| Mixtral-8x7B (BOS) | L21E001 | WinoGrande margin pool (other pairs) | 1790 | 3.8 [2.7, 5.0] | +0.13 [+0.07, +0.19] | 71 | 0.22 | +3.66 |
| Mixtral-8x7B (BOS) | L18E001 | WinoGrande margin pool (other pairs) | 1790 | 1.2 [0.6, 2.0] | +0.08 [-0.00, +0.18] | 64 | 0.05 | +3.66 |
| Mixtral-8x7B (BOS) | L20E000 | local prompts (pool pairs) | 1790 | 94.2 [92.8, 95.7] | +0.05 [+0.04, +0.06] | 56 | 0.46 | +0.58 |
| Mixtral-8x7B (BOS) | L19E002 | local prompts (pool pairs) | 1790 | 9.1 [7.5, 10.8] | +0.00 [-0.01, +0.01] | 52 | 0.27 | +0.58 |
| Mixtral-8x7B (BOS) | L21E001 | local prompts (pool pairs) | 1790 | 3.1 [2.1, 4.1] | -0.01 [-0.03, +0.00] | 44 | 0.08 | +0.58 |
| Mixtral-8x7B (BOS) | L18E001 | local prompts (pool pairs) | 1790 | 1.6 [0.9, 2.5] | +0.01 [-0.02, +0.04] | 48 | 0.06 | +0.58 |
| Mixtral-8x7B (BOS) | L20E000 | CounterFact STR (clean + donors), clean only | 213 | 31.9 [25.8, 38.5] | +0.15 [+0.11, +0.20] | 81 | 0.21 | +6.56 |
| Mixtral-8x7B (BOS) | L19E002 | CounterFact STR (clean + donors), clean only | 213 | 61.0 [54.5, 67.6] | +0.41 [+0.34, +0.49] | 86 | 0.41 | +6.56 |
| Mixtral-8x7B (BOS) | L21E001 | CounterFact STR (clean + donors), clean only | 213 | 67.6 [61.5, 73.7] | +0.49 [+0.42, +0.56] | 90 | 0.30 | +6.56 |
| Mixtral-8x7B (BOS) | L18E001 | CounterFact STR (clean + donors), clean only | 213 | 77.0 [71.4, 83.1] | +0.14 [+0.11, +0.16] | 77 | 0.41 | +6.56 |
| Mixtral-8x7B (BOS) | L20E000 | CounterFact scan (1,024 clean) | 1024 | 32.7 [29.8, 35.7] | +0.14 [+0.11, +0.17] | 76 | 0.32 | +5.80 |
| Mixtral-8x7B (BOS) | L19E002 | CounterFact scan (1,024 clean) | 1024 | 58.4 [55.3, 61.1] | +0.34 [+0.31, +0.38] | 86 | 0.40 | +5.80 |
| Mixtral-8x7B (BOS) | L21E001 | CounterFact scan (1,024 clean) | 1024 | 67.0 [64.1, 69.9] | +0.39 [+0.36, +0.43] | 86 | 0.37 | +5.80 |
| Mixtral-8x7B (BOS) | L18E001 | CounterFact scan (1,024 clean) | 1024 | 77.6 [75.2, 80.2] | +0.12 [+0.11, +0.14] | 72 | 0.36 | +5.80 |
| Mixtral-8x7B (BOS) | L20E000 | IOI clean | 1600 | 0.1 [0.0, 0.3] | n/a | 50 |  | +5.55 |
| Mixtral-8x7B (BOS) | L19E002 | IOI clean | 1600 | 0.0 [0.0, 0.0] | n/a |  |  | +5.55 |
| Mixtral-8x7B (BOS) | L21E001 | IOI clean | 1600 | 94.4 [93.3, 95.5] | -0.02 [-0.02, -0.02] | 41 | 0.10 | +5.55 |
| Mixtral-8x7B (BOS) | L18E001 | IOI clean | 1600 | 99.8 [99.6, 100.0] | -0.01 [-0.01, -0.00] | 46 | 0.08 | +5.55 |
| Mixtral-8x7B (BOS) | L20E000 | wikitext-103 (final token of windows) | 1100 | 22.3 [19.9, 24.7] |  |  |  |  |
| Mixtral-8x7B (BOS) | L19E002 | wikitext-103 (final token of windows) | 1100 | 22.9 [20.6, 25.5] |  |  |  |  |
| Mixtral-8x7B (BOS) | L21E001 | wikitext-103 (final token of windows) | 1100 | 24.7 [22.2, 27.3] |  |  |  |  |
| Mixtral-8x7B (BOS) | L18E001 | wikitext-103 (final token of windows) | 1100 | 24.5 [22.0, 27.1] |  |  |  |  |

Final-position routing of the target experts in each prompt set (CounterFact STR: clean prompts) and the expert's own DLA there, (c_e . gamma) . (W_U[true] - W_U[foil]) / rms(h_final) with true / foil = own trigger / twin trigger (WinoGrande), true object / counterfactual object (CounterFact), IO / S (IOI). CIs: bootstrap over pairs (WinoGrande), cases or prompts.

### C2. WinoGrande: by the word before the trigger, full vs local prompt

| model | expert | final word | prompts | routed, full prompt (%) | routed, local prompt (%) |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | was | 1270 | 76.4 [73.5, 79.4] | 68.1 [64.9, 71.3] |
| Qwen3-30B-A3B-Base | L41E117 | too | 1154 | 99.9 [99.7, 100.0] | 100.0 [100.0, 100.0] |
| Qwen3-30B-A3B-Base | L41E117 | is | 396 | 94.9 [91.9, 97.2] | 65.2 [59.3, 70.7] |
| Qwen3-30B-A3B-Base | L41E117 | very | 214 | 99.5 [98.6, 100.0] | 98.1 [95.3, 100.0] |
| Qwen3-30B-A3B-Base | L41E117 | were | 160 | 97.5 [95.0, 100.0] | 85.6 [78.8, 91.9] |
| Qwen3-30B-A3B-Base | L41E117 | a | 72 | 38.9 [26.4, 51.4] | 11.1 [2.8, 22.2] |
| Qwen3-30B-A3B-Base | L41E117 | more | 68 | 97.1 [92.6, 100.0] | 94.1 [85.3, 100.0] |
| Qwen3-30B-A3B-Base | L41E117 | the | 60 | 38.3 [23.3, 53.4] | 3.3 [0.0, 10.0] |
| Qwen3-30B-A3B-Base | L41E117 | so | 58 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| Qwen3-30B-A3B-Base | L41E117 | are | 54 | 100.0 [100.0, 100.0] | 92.6 [83.3, 100.0] |
| Qwen3-30B-A3B-Base | L41E117 | be | 42 | 95.2 [88.1, 100.0] | 73.8 [54.7, 92.9] |
| Mixtral-8x7B (BOS) | L20E000 | too | 858 | 99.9 [99.7, 100.0] | 99.8 [99.4, 100.0] |
| Mixtral-8x7B (BOS) | L20E000 | was | 678 | 100.0 [100.0, 100.0] | 99.9 [99.4, 100.0] |
| Mixtral-8x7B (BOS) | L20E000 | is | 234 | 100.0 [100.0, 100.0] | 98.7 [97.0, 100.0] |
| Mixtral-8x7B (BOS) | L20E000 | very | 88 | 100.0 [100.0, 100.0] | 94.3 [86.4, 100.0] |
| Mixtral-8x7B (BOS) | L20E000 | were | 74 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |

WinoGrande prompts (main + margin pool) by the word before the trigger; local = the context-free prompt from the blank on.

**Paired main prompts: the same directed case as a full and as a local prompt**

| model | expert | routed full / local / both (%) | routed in local given routed in full (%) | DLA full / local when both (logits) | r(DLA full, DLA local) |
|---|---|---|---|---|---|
| Qwen3 | L41E117 | 84.4 / 86.3 / 76.6 | 90.7 | +0.58 / +0.09 | 0.50 |
| Qwen3 | L43E081 | 98.2 / 94.1 / 93.9 | 95.6 | +0.49 / +0.07 | 0.49 |
| Qwen3 | L39E071 | 80.5 / 37.7 / 36.7 | 45.6 | +0.25 / +0.03 | 0.23 |
| Mixtral (BOS) | L20E000 | 96.1 / 95.1 / 94.3 | 98.2 | +0.33 / +0.06 | 0.30 |
| Mixtral (BOS) | L19E006 | 98.2 / 97.7 / 97.7 | 99.4 | +0.22 / +0.03 | 0.43 |
| Mixtral (BOS) | L21E006 | 89.5 / 88.1 / 86.9 | 97.2 | +0.26 / +0.04 | 0.31 |

Local prompts carry little of the answer: Qwen3 mean Δ full +4.16, local +0.55 (local Δ > 0 in 53 %); Mixtral mean Δ full +3.89, local +0.55 (local Δ > 0 in 61 %).

### C3. CounterFact: by relation group

| model | expert | relation group | prompts | routed at final position (%) | DLA when routed (logits) |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | place | 459 | 3.9 [2.4, 5.7] | -0.00 [-0.02, +0.01] |
| Qwen3-30B-A3B-Base | L41E117 | language | 178 | 27.0 [20.8, 33.2] | +0.00 [-0.01, +0.02] |
| Qwen3-30B-A3B-Base | L41E117 | organisation | 152 | 0.0 [0.0, 0.0] | n/a |
| Qwen3-30B-A3B-Base | L41E117 | occupation / field | 200 | 23.5 [17.5, 29.5] | -0.00 [-0.02, +0.01] |
| Qwen3-30B-A3B-Base | L41E117 | other | 35 | 0.0 [0.0, 0.0] | n/a |
| Qwen3-30B-A3B-Base | L41E117 | STR clean / donor prompts | 215 | 10.7 [6.5, 14.9] / 10.2 [6.2, 14.7] |  |
| Qwen3-30B-A3B-Base | L44E069 | place | 459 | 99.6 [98.9, 100.0] | +1.08 [+0.96, +1.21] |
| Qwen3-30B-A3B-Base | L44E069 | language | 178 | 77.5 [71.3, 83.1] | +0.38 [+0.27, +0.51] |
| Qwen3-30B-A3B-Base | L44E069 | organisation | 152 | 98.0 [95.4, 100.0] | +0.12 [+0.09, +0.16] |
| Qwen3-30B-A3B-Base | L44E069 | occupation / field | 200 | 73.5 [67.0, 79.5] | +0.01 [-0.01, +0.02] |
| Qwen3-30B-A3B-Base | L44E069 | other | 35 | 77.1 [62.9, 88.6] | +0.57 [+0.21, +0.93] |
| Qwen3-30B-A3B-Base | L44E069 | STR clean / donor prompts | 215 | 87.9 [83.3, 92.1] / 87.1 [83.2, 91.0] |  |
| Qwen3-30B-A3B-Base | L42E115 | place | 459 | 99.8 [99.3, 100.0] | +0.41 [+0.37, +0.45] |
| Qwen3-30B-A3B-Base | L42E115 | language | 178 | 92.7 [88.8, 96.1] | +0.51 [+0.42, +0.60] |
| Qwen3-30B-A3B-Base | L42E115 | organisation | 152 | 99.3 [98.0, 100.0] | +0.33 [+0.27, +0.40] |
| Qwen3-30B-A3B-Base | L42E115 | occupation / field | 200 | 89.0 [84.5, 93.0] | +0.07 [+0.05, +0.09] |
| Qwen3-30B-A3B-Base | L42E115 | other | 35 | 100.0 [100.0, 100.0] | +0.33 [+0.19, +0.48] |
| Qwen3-30B-A3B-Base | L42E115 | STR clean / donor prompts | 215 | 95.3 [92.6, 98.1] / 96.0 [94.3, 97.6] |  |
| Mixtral-8x7B (BOS) | L20E000 | place | 497 | 33.4 [29.6, 37.6] | +0.17 [+0.14, +0.22] |
| Mixtral-8x7B (BOS) | L20E000 | language | 225 | 48.4 [42.2, 55.1] | +0.11 [+0.07, +0.15] |
| Mixtral-8x7B (BOS) | L20E000 | organisation | 105 | 2.9 [0.0, 5.7] | +0.03 [-0.08, +0.15] |
| Mixtral-8x7B (BOS) | L20E000 | occupation / field | 168 | 32.1 [25.6, 39.3] | +0.11 [+0.07, +0.15] |
| Mixtral-8x7B (BOS) | L20E000 | other | 29 | 10.3 [0.0, 20.7] | -0.12 [-0.24, +0.04] |
| Mixtral-8x7B (BOS) | L20E000 | STR clean / donor prompts | 213 | 31.9 [25.8, 38.5] / 33.0 [27.4, 39.2] |  |
| Mixtral-8x7B (BOS) | L19E002 | place | 497 | 82.7 [79.5, 85.9] | +0.40 [+0.36, +0.44] |
| Mixtral-8x7B (BOS) | L19E002 | language | 225 | 11.6 [7.6, 16.0] | +0.05 [-0.01, +0.11] |
| Mixtral-8x7B (BOS) | L19E002 | organisation | 105 | 75.2 [66.7, 82.9] | +0.28 [+0.22, +0.34] |
| Mixtral-8x7B (BOS) | L19E002 | occupation / field | 168 | 44.6 [36.9, 52.4] | +0.24 [+0.17, +0.30] |
| Mixtral-8x7B (BOS) | L19E002 | other | 29 | 24.1 [10.3, 38.0] | +0.10 [-0.04, +0.25] |
| Mixtral-8x7B (BOS) | L19E002 | STR clean / donor prompts | 213 | 61.0 [54.5, 67.6] / 55.1 [49.0, 61.5] |  |
| Mixtral-8x7B (BOS) | L21E001 | place | 497 | 80.9 [77.3, 84.3] | +0.38 [+0.33, +0.43] |
| Mixtral-8x7B (BOS) | L21E001 | language | 225 | 16.9 [12.0, 21.3] | +0.39 [+0.31, +0.48] |
| Mixtral-8x7B (BOS) | L21E001 | organisation | 105 | 78.1 [69.5, 84.8] | +0.50 [+0.40, +0.59] |
| Mixtral-8x7B (BOS) | L21E001 | occupation / field | 168 | 88.7 [83.9, 92.9] | +0.41 [+0.33, +0.48] |
| Mixtral-8x7B (BOS) | L21E001 | other | 29 | 51.7 [34.5, 69.0] | +0.14 [+0.05, +0.25] |
| Mixtral-8x7B (BOS) | L21E001 | STR clean / donor prompts | 213 | 67.6 [61.5, 73.7] / 64.5 [58.8, 70.7] |  |
| Mixtral-8x7B (BOS) | L18E001 | place | 497 | 98.4 [97.2, 99.4] | +0.15 [+0.13, +0.17] |
| Mixtral-8x7B (BOS) | L18E001 | language | 225 | 51.1 [44.9, 57.3] | +0.11 [+0.08, +0.14] |
| Mixtral-8x7B (BOS) | L18E001 | organisation | 105 | 100.0 [100.0, 100.0] | +0.08 [+0.05, +0.12] |
| Mixtral-8x7B (BOS) | L18E001 | occupation / field | 168 | 35.1 [28.0, 42.3] | +0.02 [+0.01, +0.04] |
| Mixtral-8x7B (BOS) | L18E001 | other | 29 | 93.1 [82.8, 100.0] | +0.05 [+0.00, +0.10] |
| Mixtral-8x7B (BOS) | L18E001 | STR clean / donor prompts | 213 | 77.0 [71.4, 83.1] / 73.4 [67.0, 79.5] |  |

CounterFact base-scan prompts by relation group (place = P17, P19, P20, P27, P30, P36, P131, P159, P190, P276, P495, P740, P937).

### C4. All-token routing inside the task prompts

| model | expert | WinoGrande main: final / other positions (%) | CounterFact STR: final / other positions (%) | IOI clean: final / other positions (%) | wikitext-103: final / other positions (%) | WG prompts, tokens ' was' ' too' ' is' ' very' ' were': final / earlier (%) |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | 84.4 / 6.7 | 10.3 / 2.1 | 0.0 / 0.0 | 6.3 / 7.0 | 86.3 / 80.3 (n=396) |
| Qwen3-30B-A3B-Base | L44E069 | 2.7 / 12.1 | 87.3 / 39.8 | 10.9 / 14.2 | 46.5 / 45.4 | 0.7 / 2.0 (n=396) |
| Qwen3-30B-A3B-Base | L42E115 | 2.0 / 22.1 | 95.9 / 29.7 | 100.0 / 34.9 | 41.5 / 40.2 | 0.9 / 1.3 (n=396) |
| Mixtral-8x7B (BOS) | L20E000 | 96.1 / 16.7 | 32.8 / 21.9 | 0.1 / 10.1 | 22.3 / 21.9 | 99.8 / 100.0 (n=396) |
| Mixtral-8x7B (BOS) | L19E002 | 2.9 / 17.4 | 56.3 / 31.5 | 0.0 / 21.2 | 22.9 / 24.0 | 2.9 / 15.9 (n=396) |
| Mixtral-8x7B (BOS) | L21E001 | 3.7 / 21.2 | 65.1 / 31.6 | 94.4 / 27.9 | 24.7 / 25.4 | 1.5 / 1.5 (n=396) |
| Mixtral-8x7B (BOS) | L18E001 | 1.2 / 19.6 | 74.1 / 15.3 | 99.8 / 19.4 | 24.5 / 26.6 | 0.4 / 2.8 (n=396) |

All-token routing (route_all_layers): rate at the final position vs every other position of the same prompts (Mixtral: BOS position excluded); last column: the same tokens as the WinoGrande final words when they occur earlier.

### C5. wikitext-103 contexts

| model | expert | base rate (%) | current = ' too' | current = degree adverb | current = copula (was / is / were / ...) | copula or degree, next word in WG trigger vocabulary | next word in WG trigger vocabulary | next word = CF place name | next word capitalised, not a CF object | current = ' in', next = CF place | current = ' in', next = other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | 7.01 | 87.5 (12.5x, n=24) | 59.3 (8.5x, n=580) | 66.0 (9.4x, n=3029) | 77.1 (11.0x, n=800) | 15.1 (2.2x, n=11696) | 0.4 (0.1x, n=824) | 0.5 (0.1x, n=17893) | 0.0 (0.0x, n=116) | 5.0 (0.7x, n=2466) |
| Qwen3-30B-A3B-Base | L44E069 | 45.38 | 4.2 (0.1x, n=24) | 26.9 (0.6x, n=580) | 21.2 (0.5x, n=3029) | 13.1 (0.3x, n=800) | 50.9 (1.1x, n=11696) | 93.7 (2.1x, n=824) | 71.7 (1.6x, n=17893) | 100.0 (2.2x, n=116) | 82.6 (1.8x, n=2466) |
| Qwen3-30B-A3B-Base | L42E115 | 40.20 | 4.2 (0.1x, n=24) | 19.8 (0.5x, n=580) | 13.4 (0.3x, n=3029) | 7.2 (0.2x, n=800) | 42.1 (1.0x, n=11696) | 95.4 (2.4x, n=824) | 82.3 (2.0x, n=17893) | 100.0 (2.5x, n=116) | 84.0 (2.1x, n=2466) |
| Mixtral-8x7B (BOS) | L20E000 | 21.93 | 91.3 (4.2x, n=23) | 65.1 (3.0x, n=558) | 94.4 (4.3x, n=2850) | 93.0 (4.2x, n=561) | 26.4 (1.2x, n=8440) | 39.9 (1.8x, n=771) | 26.9 (1.2x, n=16511) | 45.5 (2.1x, n=110) | 35.7 (1.6x, n=2408) |
| Mixtral-8x7B (BOS) | L19E002 | 24.01 | 17.4 (0.7x, n=23) | 5.6 (0.2x, n=558) | 13.4 (0.6x, n=2850) | 8.2 (0.3x, n=561) | 20.5 (0.9x, n=8440) | 46.3 (1.9x, n=771) | 22.5 (0.9x, n=16511) | 52.7 (2.2x, n=110) | 16.4 (0.7x, n=2408) |
| Mixtral-8x7B (BOS) | L21E001 | 25.38 | 4.3 (0.2x, n=23) | 18.6 (0.7x, n=558) | 7.9 (0.3x, n=2850) | 6.1 (0.2x, n=561) | 24.9 (1.0x, n=8440) | 66.8 (2.6x, n=771) | 38.4 (1.5x, n=16511) | 82.7 (3.3x, n=110) | 44.9 (1.8x, n=2408) |
| Mixtral-8x7B (BOS) | L18E001 | 26.57 | 4.3 (0.2x, n=23) | 9.9 (0.4x, n=558) | 5.7 (0.2x, n=2850) | 3.7 (0.1x, n=561) | 18.6 (0.7x, n=8440) | 61.6 (2.3x, n=771) | 50.1 (1.9x, n=16511) | 94.5 (3.6x, n=110) | 73.9 (2.8x, n=2408) |

wikitext-103 test windows (127 content tokens, consecutive): routing rate of the expert at the current token, in % (lift over the expert's base rate, n). Word classes are heuristic word lists; 'WG trigger vocabulary' = all sentence-final trigger words of the model's W1-W6 pairs; CF names = target_true / target_new strings of CounterFact relations.

| model | expert | highest-rate current tokens (rate, n) | most frequent current tokens among its routings (share) | highest-rate next words (rate, n) |
|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | ' may' 84% (38); ' are' 82% (213); ' considered' 77% (43); ' became' 74% (94); ' is' 74% (400); ' be' 72% (251); ' were' 70% (540); ' more' 68% (151) | ' was' 7.4%; ' the' 6.0%; ' a' 5.0%; ' were' 3.9%; ' to' 3.7%; ' and' 3.1% | important 79% (24); better 75% (20); too 74% (23); highly 71% (21); unable 70% (20); able 68% (28); significant 67% (24); seen 66% (29) |
| Qwen3-30B-A3B-Base | L44E069 | ' east' 100% (68); ' across' 100% (32); 'rd' 100% (32); ' Austrian' 100% (36); ' Battalion' 100% (49); ' South' 100% (51); ' NK' 100% (31); ' National' 100% (50) | ' the' 9.2%; '.' 4.4%; ',' 4.3%; ' of' 3.5%; ' in' 2.9%; ' ' 2.2% | york 100% (29); ny 100% (20); australia 100% (22); nk 100% (31); battleships 100% (28); mogadishu 100% (22); player 100% (21); pitcher 100% (55) |
| Qwen3-30B-A3B-Base | L42E115 | ' North' 100% (152); ' across' 100% (32); ' Austrian' 100% (36); ' against' 99% (110); ' Korean' 99% (83); ' American' 99% (72); ' As' 98% (53); ' South' 98% (51) | ' the' 10.1%; '.' 6.8%; ',' 6.2%; ' of' 4.4%; ' in' 3.4%; ' and' 3.0% | amphibians 100% (30); hms 100% (30); ny 100% (20); hed 100% (22); billboard 100% (20); episode 100% (29); philippines 100% (37); korean 100% (83) |
| Mixtral-8x7B (BOS) | L20E000 | ' St' 100% (40); ' en' 100% (33); ' became' 100% (91); ' Ar' 99% (74); ' been' 99% (139); ' un' 98% (64); ' be' 98% (224); ' Gu' 98% (48) | '.' 7.1%; ',' 4.7%; ' the' 4.5%; ' was' 3.3%; ' and' 2.8%; ' in' 2.5% | unable 100% (20); mph 100% (79); fu 98% (42); fu's 97% (29); able 93% (27); china 87% (39); too 86% (22); seen 86% (29) |
| Mixtral-8x7B (BOS) | L19E002 | ' As' 100% (52); ' wooden' 100% (32); ' Royal' 100% (31); ' Po' 100% (36); ' development' 100% (41); ' armor' 98% (51); ' Ar' 97% (74); ' Philippines' 97% (37) | ',' 5.2%; ' ' 3.0%; '0' 2.0%; ' of' 1.9%; ' the' 1.6%; '.' 1.5% | fu 88% (42); fu's 86% (29); status 79% (24); yongsan 77% (31); cyclones 75% (28); island 74% (34); video 71% (24); baseball 70% (20) |
| Mixtral-8x7B (BOS) | L21E001 | ' against' 96% (92); ' did' 94% (51); 'nd' 91% (86); 'rd' 90% (30); ' H' 90% (139); 'akt' 89% (53); ' New' 88% (52); ' peak' 88% (43) | ',' 5.9%; ' of' 5.2%; '.' 3.6%; ' the' 3.4%; ' in' 2.9%; ' a' 2.7% | battleships 100% (28); dodger 100% (23); amphibians 96% (27); helicopters 96% (22); york 93% (29); warships 92% (26); column 92% (25); soldiers 92% (24) |
| Mixtral-8x7B (BOS) | L18E001 | ' Independ' 100% (34); ' However' 100% (46); ' since' 100% (58); ' due' 100% (46); ' On' 99% (151); ' at' 98% (563); '".' 98% (97); ' $' 98% (48) | ' ' 9.7%; '.' 8.6%; ' of' 5.5%; ' in' 4.2%; ' and' 3.4%; ',' 3.3% | july 98% (50); december 92% (39); august 91% (65); november 90% (63); yongsan 90% (31); least 87% (30); october 86% (64); june 86% (42) |

wikitext-103: current tokens with the highest routing rate (≥ 30 occurrences), the most frequent current tokens among the expert's routings, and next words with the highest rate (≥ 20 occurrences).

### B1. Vocabulary projections: where r and r′ land

| model | expert | vectors | n | median rank of r (of V) | r in top-10 / top-100 (%) | median rank of r' from the bottom | median rank of r in its class (class size) | class z (own class) | most promoted tokens (share of vectors with it in the top-10) | most suppressed tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | δ_e, CounterFact STR rows | 95 | 71207 | 0 / 0 | 81704 | 51 (158) | cf_any: -0.03 | ' rich' 17%; '富' 13%; 'rich' 12%; ' varied' 12%; ' Rich' 11%; ' simple' 9% | ' clos' 11%; ' closer' 9%; ' easier' 8%; 'easy' 8%; ' cheapest' 7% |
| Qwen3-30B-A3B-Base | L41E117 | δ_e, WinoGrande STR rows | 432 | 98 | 30 / 50 | 98 | 12 (1648) | wg_trigger: +0.01 | ' small' 24%; 'small' 22%; ' smaller' 21%; '小' 20%; ' SMALL' 19%; ' Larger' 17% | ' small' 24%; 'small' 22%; ' smaller' 21%; '小' 20%; ' SMALL' 19% |
| Qwen3-30B-A3B-Base | L41E117 | c_e, CounterFact clean prompts | 23 | 60917 | 0 / 0 | 93610 | 39 (158) | cf_any: -0.01 | ' diverse' 39%; ' varied' 35%; ' leg' 26%; ' rich' 26%; ' divers' 26%; ' diversity' 22% | '更高的' 17%; ' stronger' 17%; 'мор' 17%; 'Easy' 13%; ' cheapest' 13% |
| Qwen3-30B-A3B-Base | L41E117 | c_e, WinoGrande prompts | 432 | 131 | 31 / 48 | 66798 | 11 (1648) | wg_trigger: -0.00 | ' small' 25%; '小' 22%; 'small' 21%; ' SMALL' 20%; ' smaller' 19%; '个小' 18% | '的大' 14%; '更大' 14%; ' Large' 13%; '做大' 13%; ' Larger' 13% |
| Qwen3-30B-A3B-Base | L41E117 | c_e, WinoGrande local prompts | 442 | 4826 | 16 / 25 | 139200 | 110 (1648) | wg_trigger: +0.12 | ' small' 28%; ' SMALL' 19%; 'small' 19%; ' smaller' 16%; '小' 14%; ' dear' 14% | ' cleaner' 17%; ' sufficiently' 17%; ' достат' 17%; '足够' 12%; '-clean' 12% |
| Qwen3-30B-A3B-Base | L44E069 | δ_e, CounterFact STR rows | 746 | 5758 | 15 / 29 | 11977 | 18 (168) | cf_any: -0.02 | ' New' 6%; ' Prov' 4%; ' Tur' 3%; ' Ant' 3%; '(New' 3%; ' Finnish' 3% | ' New' 3%; ' Paris' 3%; ' France' 3%; ' North' 3%; ' French' 3% |
| Qwen3-30B-A3B-Base | L44E069 | δ_e, WinoGrande STR rows | 14 | 80082 | 7 / 7 | 78230 | 934 (1648) | wg_trigger: +0.00 | ' Los' 21%; ' North' 21%; ' NORTH' 21%; 'Los' 14%; ' LA' 14%; 'North' 14% | ' North' 21%; ' NORTH' 21%; ' Los' 21%; 'North' 14%; 'Los' 14% |
| Qwen3-30B-A3B-Base | L44E069 | c_e, CounterFact clean prompts | 191 | 921 | 22 / 38 | 118393 | 10 (471) | cf_any: +0.30 | ' New' 6%; ' Tur' 4%; ' Prov' 4%; ' Norm' 4%; ' Southern' 3%; ' Japan' 3% | ' Spain' 3%; ' Italian' 3%; '西班牙' 3%; '葡萄牙' 3%; ' london' 3% |
| Qwen3-30B-A3B-Base | L44E069 | c_e, WinoGrande prompts | 14 | 71370 | 0 / 14 | 99641 | 763 (1648) | wg_trigger: +0.03 | ' US' 21%; ' Cleveland' 21%; ' Rochester' 21%; ' Los' 14%; ' Fresno' 14%; ' USPS' 14% | ' Vietnam' 21%; ' live' 14%; 'live' 14%; 'ui' 14%; ' Kut' 7% |
| Qwen3-30B-A3B-Base | L44E069 | c_e, WinoGrande local prompts | 16 | 39190 | 0 / 0 | 109820 | 390 (1648) | wg_trigger: -0.02 | ' London' 19%; ' Los' 19%; ' Sydney' 12%; ' Capitals' 12%; ' Cowboys' 12%; ' Nuggets' 12% | '北' 19%; '闻' 12%; 'zend' 12%; 'jie' 12%; '阶梯' 12% |
| Qwen3-30B-A3B-Base | L44E069 | c_e, IOI prompts | 45 | 34498 | 0 / 0 | 123014 |  | ioi_names: +0.77 | ' area' 22%; ' Central' 18%; ' Clintons' 18%; ' Area' 18%; ' high' 16%; ' UW' 13% | ' subsequ' 38%; '各省' 27%; ' Mad' 24%; 'adin' 11%; ' cav' 11% |
| Qwen3-30B-A3B-Base | L42E115 | δ_e, CounterFact STR rows | 821 | 1287 | 20 / 34 | 3412 | 7 (168) | cf_any: +0.03 | ' France' 5%; ' Swedish' 5%; ' Paris' 5%; ' Italian' 5%; '意大利' 4%; ' French' 4% | '俄罗斯' 6%; ' Russia' 5%; '俄' 5%; ' Russian' 5%; ' Russians' 5% |
| Qwen3-30B-A3B-Base | L42E115 | δ_e, WinoGrande STR rows | 11 | 1180 | 27 / 36 | 1180 | 11 (1648) | wg_trigger: +0.04 | '...' 18%; ' Spanish' 18%; 'Spanish' 18%; '’S' 9%; "'S" 9%; " '}" 9% | '...' 18%; ' Americans' 18%; ' Spanish' 18%; 'Spanish' 18%; ' ...' 9% |
| Qwen3-30B-A3B-Base | L42E115 | c_e, CounterFact clean prompts | 208 | 105 | 34 / 50 | 112411 | 3 (168) | cf_any: -0.09 | ' Catalan' 8%; ' French' 7%; ' Old' 5%; ' Hindi' 5%; ' Standard' 5%; ' Modern' 5% | 'Spain' 8%; 'Britain' 7%; '-China' 6%; 'China' 6%; 'Chinese' 6% |
| Qwen3-30B-A3B-Base | L42E115 | c_e, WinoGrande prompts | 11 | 577 | 36 / 36 | 120316 | 6 (1648) | wg_trigger: -0.04 | ' Wide' 18%; ' Mandarin' 18%; ' Portug' 18%; ' Span' 18%; ' Spanish' 18%; '.Span' 18% | '...' 55%; '....' 55%; '.....' 45%; '…' 36%; ' ...' 36% |
| Qwen3-30B-A3B-Base | L42E115 | c_e, WinoGrande local prompts | 11 | 20543 | 0 / 0 | 132431 | 366 (1648) | wg_trigger: +0.08 | 'Various' 18%; ' Various' 18%; 'pper' 18%; ' Span' 18%; ' Jama' 18%; ' Kling' 18% | 'Europe' 27%; 'Muslim' 18%; '琥' 18%; '\tside' 18%; '迪' 18% |
| Qwen3-30B-A3B-Base | L42E115 | c_e, IOI prompts | 400 | 59554 | 0 / 0 | 106870 |  | ioi_names: +0.26 | ' Lump' 42%; ' Bun' 16%; ' Len' 13%; ' Lob' 12%; ' Blur' 12%; ' Smarty' 12% | ' Państwo' 35%; 'enc' 28%; ' Câm' 23%; 'bsub' 22%; 'atak' 14% |
| Qwen3-30B-A3B-Base | L43E081 | δ_e, CounterFact STR rows | 324 | 67850 | 0 / 0 | 72011 | 42 (158) | cf_any: +0.01 | ' rather' 18%; 'rather' 17%; ' Rather' 13%; ' plutôt' 12%; '而不是' 11%; '而非' 11% | ' rather' 11%; 'rather' 10%; 'too' 10%; ' well' 10%; ' too' 9% |
| Qwen3-30B-A3B-Base | L43E081 | δ_e, WinoGrande STR rows | 503 | 3458 | 17 / 30 | 3458 | 84 (1648) | wg_trigger: -0.00 | ' small' 12%; ' much' 12%; '小' 11%; 'much' 11%; ' heavy' 10%; ' too' 9% | ' small' 12%; ' much' 12%; '小' 11%; 'much' 11%; ' heavy' 10% |
| Qwen3-30B-A3B-Base | L43E081 | c_e, CounterFact clean prompts | 79 | 51059 | 0 / 0 | 95233 | 42 (158) | cf_any: +0.24 | ' rather' 41%; 'rather' 37%; ' interesting' 27%; ' Rather' 27%; ' fascinating' 25%; ' plutôt' 23% | 'way' 15%; 'Way' 11%; 'WAY' 11%; 'more' 11%; '太' 10% |
| Qwen3-30B-A3B-Base | L43E081 | c_e, WinoGrande prompts | 503 | 30 | 42 / 54 | 151030 | 13 (1648) | wg_trigger: +0.54 | ' large' 34%; ' big' 34%; ' tall' 23%; ' heavy' 23%; ' small' 23%; ' deep' 19% | '非常好的' 21%; 'much' 18%; ' Too' 17%; ' Much' 15%; ' TOO' 15% |
| Qwen3-30B-A3B-Base | L43E081 | c_e, WinoGrande local prompts | 484 | 114 | 17 / 49 | 151784 | 34 (1648) | wg_trigger: +0.95 | ' heavy' 49%; 'heavy' 34%; ' hot' 32%; ' Heavy' 32%; ' big' 29%; ' large' 26% | '片刻' 16%; '一定的' 14%; 'GES' 11%; '槛' 9%; '很多' 8% |
| Qwen3-30B-A3B-Base | L43E081 | c_e, IOI prompts | 9 | 18223 | 0 / 0 | 132503 |  | ioi_names: +0.76 | ' even' 33%; ' Even' 33%; ' Too' 33%; ' too' 33%; 'too' 33%; ' extra' 22% | ' good' 67%; ' much' 56%; ' considerably' 44%; ' substantially' 44%; 'much' 44% |
| Mixtral-8x7B (BOS) | L20E000 | δ_e, CounterFact STR rows | 279 | 1712 | 11 / 23 | 6325 | 12 (153) | cf_any: +0.01 | ' /******/' 25%; ' unknown' 8%; ' uncertain' 6%; ' kennis' 5%; 'unknown' 5%; 'UNKNOWN' 4% | ' /******/' 21%; 'pshire' 4%; ' gepubliceerd' 4%; ' echo' 3%; ' oppon' 3% |
| Mixtral-8x7B (BOS) | L20E000 | δ_e, WinoGrande STR rows | 493 | 521 | 24 / 37 | 566 | 26 (764) | wg_trigger: -0.00 | ' /******/' 12%; ' small' 11%; ' smaller' 10%; ' empty' 8%; ' empt' 8%; 'Empty' 7% | ' /******/' 12%; ' small' 11%; ' smaller' 10%; ' empty' 8%; ' empt' 8% |
| Mixtral-8x7B (BOS) | L20E000 | c_e, CounterFact clean prompts | 68 | 628 | 12 / 32 | 21624 | 8 (406) | cf_any: -0.02 | ' /******/' 41%; 'WEBPACK' 13%; '丶' 13%; 'ityEngine' 13%; ' Станов' 12%; ' worth' 12% | ' gepubliceerd' 10%; ' oppon' 10%; 'borough' 7%; 'pshire' 6%; ' /******/' 6% |
| Mixtral-8x7B (BOS) | L20E000 | c_e, WinoGrande prompts | 493 | 10 | 51 / 79 | 31726 | 4 (764) | wg_trigger: +0.35 | ' large' 25%; ' small' 18%; ' crowded' 18%; ' tall' 18%; ' big' 15%; 'large' 14% | ' /******/' 30%; 'pread' 16%; ' oppon' 15%; 'impse' 12%; ' biologie' 11% |
| Mixtral-8x7B (BOS) | L20E000 | c_e, WinoGrande local prompts | 487 | 497 | 12 / 31 | 31001 | 53 (764) | wg_trigger: +0.42 | ' tempt' 21%; ' large' 20%; ' heavy' 18%; ' invented' 16%; ' originally' 15%; ' appealing' 13% | ' /******/' 41%; ' biologie' 32%; 'ntil' 23%; 'ursor' 19%; ' oppon' 19% |
| Mixtral-8x7B (BOS) | L19E002 | δ_e, CounterFact STR rows | 514 | 252 | 32 / 43 | 1087 | 5 (406) | cf_any: -0.00 | ' /******/' 20%; 'vscale' 7%; ' /***/' 6%; ' Станов' 5%; 'qpoint' 4%; '\ufeff' 4% | ' /******/' 18%; ' kennis' 5%; ' gepubliceerd' 4%; ' #!' 3%; 'úblic' 3% |
| Mixtral-8x7B (BOS) | L19E002 | δ_e, WinoGrande STR rows | 15 | 1710 | 0 / 20 | 19836 | 40 (764) | wg_trigger: -0.01 | ' /******/' 20%; 'onden' 20%; 'Ő' 13%; ' listade' 13%; ' Станов' 13%; 'eph' 13% | ' /******/' 27%; ' oppon' 20%; ' janu' 13%; 'Geplaatst' 13%; 'ipage' 13% |
| Mixtral-8x7B (BOS) | L19E002 | c_e, CounterFact clean prompts | 130 | 59 | 42 / 52 | 22848 | 2 (406) | cf_any: -0.13 | ' /******/' 49%; ' /***/' 19%; 'vscale' 17%; 'tcx' 14%; 'Geplaatst' 13%; 'Pyx' 13% | ' #!' 7%; 'sin' 5%; 'sis' 5%; 'ihood' 5%; ' gepubliceerd' 5% |
| Mixtral-8x7B (BOS) | L19E002 | c_e, WinoGrande prompts | 15 | 831 | 7 / 20 | 29335 | 29 (764) | wg_trigger: +0.03 | ' Станов' 27%; ' /******/' 20%; 'onden' 20%; 'Ő' 13%; ' listade' 13%; 'eph' 13% | ' /******/' 33%; ' oppon' 27%; 'ntil' 27%; 'decess' 20%; ' janu' 13% |
| Mixtral-8x7B (BOS) | L19E002 | c_e, WinoGrande local prompts | 32 | 12308 | 3 / 3 | 19398 | 268 (764) | wg_trigger: -0.06 | 'CodeAttribute' 50%; ' Станов' 50%; ' **_' 50%; ' /******/' 47%; 'multicol' 41%; '?;' 41% | ' ratio' 31%; ' sheer' 31%; ' oppon' 28%; ' patches' 28%; ' /******/' 25% |
| Mixtral-8x7B (BOS) | L21E001 | δ_e, CounterFact STR rows | 571 | 72 | 39 / 53 | 670 | 3 (406) | cf_any: -0.02 | ' /******/' 16%; ' French' 6%; ' jazz' 6%; ' Jazz' 5%; ' sa' 4%; ' trump' 4% | ' /******/' 15%; ' oppon' 5%; ' gepubliceerd' 5%; ' kennis' 4%; ' noten' 3% |
| Mixtral-8x7B (BOS) | L21E001 | δ_e, WinoGrande STR rows | 19 | 133 | 16 / 47 | 2769 | 7 (764) | wg_trigger: +0.09 | ' ones' 11%; ' sign' 11%; ' kennis' 11%; ' read' 5%; 'read' 5%; ' reading' 5% | 'decess' 11%; ' agre' 11%; 'criptor' 11%; ' oppon' 11%; ' kennis' 11% |
| Mixtral-8x7B (BOS) | L21E001 | c_e, CounterFact clean prompts | 145 | 7 | 52 / 73 | 21478 | 1 (406) | cf_any: -0.02 | ' /******/' 13%; '❶' 10%; ' listade' 8%; ' jazz' 6%; ' Jazz' 6%; ' French' 6% | ' oppon' 17%; ' /******/' 14%; ' gepubliceerd' 9%; 'peon' 6%; ' strugg' 6% |
| Mixtral-8x7B (BOS) | L21E001 | c_e, WinoGrande prompts | 19 | 45 | 26 / 58 | 30914 | 4 (764) | wg_trigger: +0.27 | ' drivers' 11%; ' typ' 11%; ' typing' 11%; ' ones' 11%; ' winning' 11%; ' rich' 11% | ' oppon' 53%; ' /******/' 32%; ' thous' 26%; ' toget' 16%; 'printStackTrace' 16% |
| Mixtral-8x7B (BOS) | L21E001 | c_e, WinoGrande local prompts | 10 | 1305 | 0 / 40 | 31420 | 58 (764) | wg_trigger: +0.32 | ' good' 40%; ' friends' 40%; ' friend' 40%; 'good' 40%; ' bab' 30%; ' hosts' 20% | ' oppon' 100%; ' /******/' 70%; ' thous' 50%; ' invånare' 40%; ' laug' 40% |
| Mixtral-8x7B (BOS) | L21E001 | c_e, IOI prompts | 383 | 21190 | 0 / 1 | 13064 |  | ioi_names: -0.09 | ' /******/' 27%; 'fem' 16%; ' everyone' 16%; ' dogs' 16%; ' everybody' 15%; ' dog' 15% | ' gepubliceerd' 47%; 'peon' 28%; 'ntil' 26%; ' Initialized' 25%; 'uminate' 21% |
| Mixtral-8x7B (BOS) | L18E001 | δ_e, CounterFact STR rows | 642 | 3303 | 9 / 20 | 4080 | 22 (406) | cf_any: +0.01 | ' /******/' 25%; ' /***/' 5%; '\ufeff' 5%; ' acknow' 5%; ' ==>' 4%; ' oppon' 4% | ' /******/' 22%; '\ufeff' 6%; ' /***/' 5%; '❶' 4%; ' noten' 4% |
| Mixtral-8x7B (BOS) | L18E001 | δ_e, WinoGrande STR rows | 6 | 8249 | 0 / 0 | 3742 | 210 (764) | wg_trigger: -0.06 | ' /******/' 50%; ' writing' 17%; ' write' 17%; ' Writing' 17%; ' Writ' 17%; ' writ' 17% | ' overflow' 17%; ' fa' 17%; 'Ї' 17%; ' Palmar' 17%; 'eral' 17% |
| Mixtral-8x7B (BOS) | L18E001 | c_e, CounterFact clean prompts | 164 | 544 | 16 / 31 | 25050 | 6 (406) | cf_any: -0.12 | ' /******/' 47%; 'tcx' 33%; 'IMPORTED' 24%; 'rdev' 24%; ' /***/' 21%; '－' 16% | 'istrzost' 12%; 'bros' 10%; 'ihood' 10%; ' gepubliceerd' 9%; 'iginal' 7% |
| Mixtral-8x7B (BOS) | L18E001 | c_e, WinoGrande prompts | 6 | 4006 | 0 / 0 | 22066 | 90 (764) | wg_trigger: -0.23 | '%%%%' 67%; ' /******/' 67%; ' /***/' 50%; 'igos' 33%; '－' 33%; '\ufeff' 33% | ' native' 33%; ' conc' 33%; ' workshops' 33%; '<0x0A>' 33%; 'riterion' 17% |
| Mixtral-8x7B (BOS) | L18E001 | c_e, WinoGrande local prompts | 6 | 8455 | 0 / 0 | 30692 | 144 (764) | wg_trigger: -0.22 | 'tcx' 67%; 'AtA' 67%; '\x08' 50%; 'ylv' 33%; '格' 33%; 'INCLUDING' 33% | 'urm' 33%; '╔' 33%; ' stern' 33%; 'area' 33%; ' pra' 33% |
| Mixtral-8x7B (BOS) | L18E001 | c_e, IOI prompts | 398 | 22564 | 0 / 0 | 10580 |  | ioi_names: -0.41 | ' /******/' 90%; 'tcx' 73%; 'fortunate' 48%; ' someone' 42%; ' oppon' 35%; 'čen' 31% | '💰' 43%; ' Initialized' 24%; 'zens' 20%; ' ass' 17%; 'rass' 13% |
| Mixtral-8x7B (BOS) | L19E006 | δ_e, CounterFact STR rows | 484 | 1607 | 14 / 27 | 4890 | 8 (153) | cf_any: -0.03 | ' /******/' 28%; ' /***/' 6%; '\ufeff' 6%; '－' 5%; ' kennis' 5%; ' Dutch' 5% | ' /******/' 15%; ' gepubliceerd' 5%; ' Dutch' 5%; ' kennis' 5%; ' /***/' 5% |
| Mixtral-8x7B (BOS) | L19E006 | δ_e, WinoGrande STR rows | 503 | 1364 | 14 / 26 | 1384 | 50 (764) | wg_trigger: -0.00 | ' /******/' 17%; ' too' 7%; ' crowded' 7%; ' wider' 7%; ' empty' 6%; ' faster' 5% | ' /******/' 17%; ' too' 7%; ' crowded' 7%; ' wider' 7%; ' empty' 6% |
| Mixtral-8x7B (BOS) | L19E006 | c_e, CounterFact clean prompts | 118 | 338 | 19 / 39 | 24488 | 4 (153) | cf_any: -0.15 | ' /******/' 74%; '\ufeff' 32%; '－' 29%; 'ityEngine' 25%; 'vscale' 25%; ' /***/' 24% | ' kennis' 5%; ' sophistic' 5%; ' Kenya' 4%; ' arrang' 4%; ' plug' 4% |
| Mixtral-8x7B (BOS) | L19E006 | c_e, WinoGrande prompts | 503 | 52 | 29 / 59 | 31284 | 9 (764) | wg_trigger: +0.14 | ' /******/' 28%; 'WEBPACK' 26%; ' Станов' 26%; ' small' 23%; ' listade' 20%; ' too' 19% | ' acknow' 16%; ' biologie' 14%; 'ipage' 13%; ' franç' 11%; ' availability' 10% |
| Mixtral-8x7B (BOS) | L19E006 | c_e, WinoGrande local prompts | 499 | 1239 | 12 / 28 | 30784 | 59 (764) | wg_trigger: +0.14 | ' Станов' 44%; ' /******/' 39%; '\ufeff' 36%; ' listade' 35%; ' too' 31%; 'too' 25% | ' differently' 18%; ' citiz' 16%; 'ntil' 15%; ' decisions' 14%; ' attempts' 13% |

Vocabulary projection of the expert's write at the final position, ((v ⊙ γ) @ W_U^T) / rms(h_final) with the norm frozen at the corrupted run (δ_e = c_e(clean) − c_e(corrupt)) or at the prompt's own run (c_e). Ranks over the full vocabulary (1 = most promoted; r' from the bottom: 1 = most suppressed); class = WinoGrande trigger vocabulary (WinoGrande rows) or the CounterFact objects of the case's relation group (CounterFact rows). Section: targets and each model's second WinoGrande expert; all experts of interest in `results/tables/ext11_B_ranks.md`.

### B2. Token classes

| model | expert | vectors | n | z wg_trigger | z cf_place | z cf_language | z cf_organisation | z cf_occupation / field | z ioi_names | top-50 share wg_trigger (base) | top-50 share cf_any (base) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | δ_e, CounterFact STR rows | 95 | +0.08 | -0.07 | -0.04 | -0.00 | +0.06 | +0.10 | 9.9 (1.08) | 0.9 (0.54) |
| Qwen3-30B-A3B-Base | L41E117 | δ_e, WinoGrande STR rows | 432 | +0.01 | +0.00 | +0.02 | +0.01 | +0.00 | -0.01 | 13.2 (1.08) | 0.1 (0.54) |
| Qwen3-30B-A3B-Base | L41E117 | c_e, CounterFact clean prompts | 23 | +0.09 | -0.05 | -0.11 | +0.05 | +0.09 | +0.07 | 10.3 (1.08) | 0.7 (0.54) |
| Qwen3-30B-A3B-Base | L41E117 | c_e, WinoGrande prompts | 432 | -0.00 | +0.12 | +0.18 | +0.16 | -0.12 | -0.15 | 12.9 (1.08) | 0.1 (0.54) |
| Qwen3-30B-A3B-Base | L41E117 | c_e, WinoGrande local prompts | 442 | +0.12 | +0.03 | +0.20 | +0.03 | -0.02 | -0.19 | 15.9 (1.08) | 0.2 (0.54) |
| Qwen3-30B-A3B-Base | L44E069 | δ_e, CounterFact STR rows | 746 | -0.00 | -0.04 | +0.01 | +0.00 | -0.01 | -0.01 | 1.3 (1.08) | 9.6 (0.54) |
| Qwen3-30B-A3B-Base | L44E069 | δ_e, WinoGrande STR rows | 14 | +0.00 | +0.34 | +0.01 | +0.22 | -0.00 | +0.00 | 3.7 (1.08) | 21.1 (0.54) |
| Qwen3-30B-A3B-Base | L44E069 | c_e, CounterFact clean prompts | 191 | +0.03 | +0.48 | -0.34 | +0.28 | +0.08 | -0.07 | 1.7 (1.08) | 15.1 (0.54) |
| Qwen3-30B-A3B-Base | L44E069 | c_e, WinoGrande prompts | 14 | +0.03 | +0.47 | -0.59 | +0.37 | +0.19 | -0.34 | 3.3 (1.08) | 22.6 (0.54) |
| Qwen3-30B-A3B-Base | L44E069 | c_e, WinoGrande local prompts | 16 | -0.02 | +1.81 | +0.52 | +1.09 | +0.15 | +0.10 | 2.2 (1.08) | 37.0 (0.54) |
| Qwen3-30B-A3B-Base | L44E069 | c_e, IOI prompts | 45 | -0.03 | +0.81 | -0.01 | +0.71 | +0.24 | +0.77 | 2.0 (1.08) | 10.0 (0.54) |
| Qwen3-30B-A3B-Base | L42E115 | δ_e, CounterFact STR rows | 821 | -0.01 | +0.07 | -0.07 | +0.03 | -0.02 | -0.02 | 1.5 (1.08) | 8.9 (0.54) |
| Qwen3-30B-A3B-Base | L42E115 | δ_e, WinoGrande STR rows | 11 | +0.04 | +0.12 | -0.12 | +0.15 | -0.04 | +0.16 | 3.3 (1.08) | 6.2 (0.54) |
| Qwen3-30B-A3B-Base | L42E115 | c_e, CounterFact clean prompts | 208 | +0.07 | -0.14 | -0.05 | -0.15 | -0.01 | +0.01 | 2.2 (1.08) | 9.9 (0.54) |
| Qwen3-30B-A3B-Base | L42E115 | c_e, WinoGrande prompts | 11 | -0.04 | -0.11 | -0.27 | -0.06 | -0.12 | +0.12 | 3.3 (1.08) | 4.4 (0.54) |
| Qwen3-30B-A3B-Base | L42E115 | c_e, WinoGrande local prompts | 11 | +0.08 | +0.13 | +0.17 | +0.08 | +0.16 | +0.03 | 2.4 (1.08) | 6.9 (0.54) |
| Qwen3-30B-A3B-Base | L42E115 | c_e, IOI prompts | 400 | -0.15 | -0.04 | +0.24 | +0.16 | +0.12 | +0.26 | 0.7 (1.08) | 1.8 (0.54) |
| Qwen3-30B-A3B-Base | L43E081 | δ_e, CounterFact STR rows | 324 | +0.10 | +0.01 | -0.02 | -0.01 | +0.05 | +0.12 | 10.4 (1.08) | 0.1 (0.54) |
| Qwen3-30B-A3B-Base | L43E081 | δ_e, WinoGrande STR rows | 503 | -0.00 | +0.00 | +0.00 | +0.00 | -0.00 | +0.00 | 14.7 (1.08) | 0.3 (0.54) |
| Qwen3-30B-A3B-Base | L43E081 | c_e, CounterFact clean prompts | 79 | +0.54 | +0.19 | +0.27 | +0.21 | +0.34 | +0.46 | 16.7 (1.08) | 0.1 (0.54) |
| Qwen3-30B-A3B-Base | L43E081 | c_e, WinoGrande prompts | 503 | +0.54 | -0.05 | +0.16 | +0.02 | +0.10 | +0.29 | 29.3 (1.08) | 0.1 (0.54) |
| Qwen3-30B-A3B-Base | L43E081 | c_e, WinoGrande local prompts | 484 | +0.95 | +0.02 | +0.09 | +0.02 | +0.14 | +0.30 | 36.5 (1.08) | 0.1 (0.54) |
| Qwen3-30B-A3B-Base | L43E081 | c_e, IOI prompts | 9 | -0.41 | +0.34 | +0.66 | +0.29 | +0.34 | +0.76 | 2.7 (1.08) | 0.0 (0.54) |
| Mixtral-8x7B (BOS) | L20E000 | δ_e, CounterFact STR rows | 279 | +0.02 | +0.01 | +0.05 | +0.03 | -0.02 | +0.01 | 4.3 (2.39) | 4.3 (2.18) |
| Mixtral-8x7B (BOS) | L20E000 | δ_e, WinoGrande STR rows | 493 | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | 11.0 (2.39) | 1.2 (2.18) |
| Mixtral-8x7B (BOS) | L20E000 | c_e, CounterFact clean prompts | 68 | +0.09 | -0.03 | -0.02 | -0.01 | +0.03 | -0.00 | 5.5 (2.39) | 4.2 (2.18) |
| Mixtral-8x7B (BOS) | L20E000 | c_e, WinoGrande prompts | 493 | +0.35 | +0.03 | +0.06 | +0.07 | -0.06 | +0.04 | 23.9 (2.39) | 0.8 (2.18) |
| Mixtral-8x7B (BOS) | L20E000 | c_e, WinoGrande local prompts | 487 | +0.42 | -0.10 | +0.16 | -0.00 | -0.01 | +0.12 | 21.6 (2.39) | 0.7 (2.18) |
| Mixtral-8x7B (BOS) | L19E002 | δ_e, CounterFact STR rows | 514 | -0.02 | -0.02 | -0.04 | +0.02 | +0.01 | +0.02 | 1.9 (2.39) | 4.9 (2.18) |
| Mixtral-8x7B (BOS) | L19E002 | δ_e, WinoGrande STR rows | 15 | -0.01 | +0.03 | +0.04 | -0.00 | -0.07 | +0.12 | 6.4 (2.39) | 1.6 (2.18) |
| Mixtral-8x7B (BOS) | L19E002 | c_e, CounterFact clean prompts | 130 | -0.16 | -0.13 | -0.27 | -0.08 | -0.16 | -0.16 | 1.4 (2.39) | 3.6 (2.18) |
| Mixtral-8x7B (BOS) | L19E002 | c_e, WinoGrande prompts | 15 | +0.03 | +0.09 | +0.01 | +0.07 | -0.10 | +0.18 | 7.2 (2.39) | 0.9 (2.18) |
| Mixtral-8x7B (BOS) | L19E002 | c_e, WinoGrande local prompts | 32 | -0.06 | -0.21 | -0.35 | -0.29 | -0.23 | -0.18 | 3.4 (2.39) | 0.4 (2.18) |
| Mixtral-8x7B (BOS) | L21E001 | δ_e, CounterFact STR rows | 571 | -0.03 | -0.02 | -0.05 | -0.02 | +0.00 | -0.04 | 2.4 (2.39) | 6.8 (2.18) |
| Mixtral-8x7B (BOS) | L21E001 | δ_e, WinoGrande STR rows | 19 | +0.09 | +0.02 | +0.05 | -0.02 | +0.06 | +0.01 | 5.9 (2.39) | 4.0 (2.18) |
| Mixtral-8x7B (BOS) | L21E001 | c_e, CounterFact clean prompts | 145 | -0.02 | -0.04 | -0.06 | -0.03 | +0.07 | -0.09 | 3.0 (2.39) | 7.6 (2.18) |
| Mixtral-8x7B (BOS) | L21E001 | c_e, WinoGrande prompts | 19 | +0.27 | +0.12 | +0.42 | +0.01 | +0.23 | +0.06 | 8.5 (2.39) | 5.4 (2.18) |
| Mixtral-8x7B (BOS) | L21E001 | c_e, WinoGrande local prompts | 10 | +0.32 | +0.07 | +0.22 | +0.12 | +0.51 | -0.12 | 13.2 (2.39) | 2.2 (2.18) |
| Mixtral-8x7B (BOS) | L21E001 | c_e, IOI prompts | 383 | +0.24 | +0.27 | +0.03 | +0.21 | +0.22 | -0.09 | 5.4 (2.39) | 2.8 (2.18) |
| Mixtral-8x7B (BOS) | L18E001 | δ_e, CounterFact STR rows | 642 | -0.01 | +0.01 | +0.03 | +0.01 | +0.01 | -0.01 | 2.1 (2.39) | 3.8 (2.18) |
| Mixtral-8x7B (BOS) | L18E001 | δ_e, WinoGrande STR rows | 6 | -0.06 | -0.02 | -0.07 | -0.07 | -0.14 | -0.02 | 2.3 (2.39) | 0.7 (2.18) |
| Mixtral-8x7B (BOS) | L18E001 | c_e, CounterFact clean prompts | 164 | -0.13 | -0.07 | -0.34 | -0.05 | -0.15 | -0.16 | 1.4 (2.39) | 2.7 (2.18) |
| Mixtral-8x7B (BOS) | L18E001 | c_e, WinoGrande prompts | 6 | -0.23 | -0.21 | -0.21 | -0.28 | -0.25 | -0.23 | 1.7 (2.39) | 0.3 (2.18) |
| Mixtral-8x7B (BOS) | L18E001 | c_e, WinoGrande local prompts | 6 | -0.22 | -0.14 | -0.02 | -0.24 | -0.21 | -0.08 | 0.7 (2.39) | 0.0 (2.18) |
| Mixtral-8x7B (BOS) | L18E001 | c_e, IOI prompts | 398 | -0.03 | +0.04 | +0.11 | -0.09 | -0.06 | -0.41 | 1.0 (2.39) | 0.3 (2.18) |
| Mixtral-8x7B (BOS) | L19E006 | δ_e, CounterFact STR rows | 484 | -0.04 | -0.03 | +0.06 | -0.04 | -0.03 | -0.07 | 2.8 (2.39) | 4.2 (2.18) |
| Mixtral-8x7B (BOS) | L19E006 | δ_e, WinoGrande STR rows | 503 | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | 9.3 (2.39) | 1.5 (2.18) |
| Mixtral-8x7B (BOS) | L19E006 | c_e, CounterFact clean prompts | 118 | -0.18 | -0.20 | +0.24 | -0.22 | -0.14 | -0.25 | 2.7 (2.39) | 4.4 (2.18) |
| Mixtral-8x7B (BOS) | L19E006 | c_e, WinoGrande prompts | 503 | +0.14 | +0.06 | +0.03 | +0.03 | -0.23 | +0.26 | 17.1 (2.39) | 0.7 (2.18) |
| Mixtral-8x7B (BOS) | L19E006 | c_e, WinoGrande local prompts | 499 | +0.14 | -0.06 | +0.10 | -0.06 | -0.24 | +0.11 | 10.9 (2.39) | 0.2 (2.18) |

Mean projection over the class tokens minus the vocabulary mean, in SD units of the vector's projection (z), and the share of the 50 most promoted tokens that belong to the class (base = class size / vocabulary size). Section: targets and each model's second WinoGrande expert; all experts of interest in `results/tables/ext11_B_classes.md`.

### B3. Exact final norm vs frozen norm for single experts

| model | expert | vectors | n | exact-norm mean (logits) | frozen-norm mean (logits) | exact / frozen | r | mean |exact - frozen| (logits) | sum |err| / sum |frozen| |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | δ_e, CounterFact STR rows | 95 | -0.001 | -0.001 | 1.722 | 0.999 | 0.002 | 0.047 |
| Qwen3-30B-A3B-Base | L41E117 | δ_e, WinoGrande STR rows | 432 | +1.101 | +1.103 | 0.998 | 1.000 | 0.003 | 0.003 |
| Qwen3-30B-A3B-Base | L41E117 | c_e, CounterFact clean prompts | 23 | -0.004 | -0.001 | 4.378 | 0.999 | 0.003 | 0.080 |
| Qwen3-30B-A3B-Base | L41E117 | c_e, WinoGrande prompts | 432 | +0.546 | +0.552 | 0.990 | 1.000 | 0.005 | 0.009 |
| Qwen3-30B-A3B-Base | L44E069 | δ_e, CounterFact STR rows | 746 | +1.385 | +1.384 | 1.001 | 1.000 | 0.026 | 0.018 |
| Qwen3-30B-A3B-Base | L44E069 | δ_e, WinoGrande STR rows | 14 | +0.183 | +0.182 | 1.003 | 1.000 | 0.003 | 0.014 |
| Qwen3-30B-A3B-Base | L44E069 | c_e, CounterFact clean prompts | 191 | +0.767 | +0.799 | 0.959 | 1.000 | 0.035 | 0.041 |
| Qwen3-30B-A3B-Base | L44E069 | c_e, WinoGrande prompts | 14 | +0.080 | +0.091 | 0.878 | 1.000 | 0.011 | 0.100 |
| Qwen3-30B-A3B-Base | L42E115 | δ_e, CounterFact STR rows | 821 | +0.855 | +0.858 | 0.996 | 1.000 | 0.009 | 0.011 |
| Qwen3-30B-A3B-Base | L42E115 | δ_e, WinoGrande STR rows | 11 | +0.888 | +0.885 | 1.004 | 1.000 | 0.004 | 0.005 |
| Qwen3-30B-A3B-Base | L42E115 | c_e, CounterFact clean prompts | 208 | +0.409 | +0.433 | 0.944 | 0.999 | 0.024 | 0.051 |
| Qwen3-30B-A3B-Base | L42E115 | c_e, WinoGrande prompts | 11 | +0.443 | +0.448 | 0.988 | 1.000 | 0.008 | 0.017 |
| Qwen3-30B-A3B-Base | L43E081 | δ_e, CounterFact STR rows | 324 | +0.009 | +0.009 | 1.047 | 0.991 | 0.004 | 0.089 |
| Qwen3-30B-A3B-Base | L43E081 | δ_e, WinoGrande STR rows | 503 | +0.951 | +0.952 | 0.999 | 1.000 | 0.007 | 0.007 |
| Qwen3-30B-A3B-Base | L43E081 | c_e, CounterFact clean prompts | 79 | -0.004 | +0.001 | -2.569 | 0.993 | 0.006 | 0.151 |
| Qwen3-30B-A3B-Base | L43E081 | c_e, WinoGrande prompts | 503 | +0.455 | +0.474 | 0.959 | 0.999 | 0.020 | 0.038 |
| Mixtral-8x7B (BOS) | L20E000 | δ_e, CounterFact STR rows | 279 | +0.258 | +0.260 | 0.990 | 0.996 | 0.013 | 0.049 |
| Mixtral-8x7B (BOS) | L20E000 | δ_e, WinoGrande STR rows | 493 | +0.644 | +0.648 | 0.994 | 1.000 | 0.005 | 0.008 |
| Mixtral-8x7B (BOS) | L20E000 | c_e, CounterFact clean prompts | 68 | +0.125 | +0.152 | 0.821 | 0.994 | 0.027 | 0.155 |
| Mixtral-8x7B (BOS) | L20E000 | c_e, WinoGrande prompts | 493 | +0.296 | +0.325 | 0.912 | 0.999 | 0.029 | 0.081 |
| Mixtral-8x7B (BOS) | L19E002 | δ_e, CounterFact STR rows | 514 | +0.666 | +0.674 | 0.988 | 0.999 | 0.016 | 0.023 |
| Mixtral-8x7B (BOS) | L19E002 | δ_e, WinoGrande STR rows | 15 | +0.063 | +0.061 | 1.023 | 0.999 | 0.003 | 0.039 |
| Mixtral-8x7B (BOS) | L19E002 | c_e, CounterFact clean prompts | 130 | +0.360 | +0.408 | 0.882 | 0.998 | 0.048 | 0.112 |
| Mixtral-8x7B (BOS) | L19E002 | c_e, WinoGrande prompts | 15 | +0.032 | +0.037 | 0.868 | 0.999 | 0.005 | 0.100 |
| Mixtral-8x7B (BOS) | L21E001 | δ_e, CounterFact STR rows | 571 | +0.864 | +0.871 | 0.992 | 0.999 | 0.016 | 0.019 |
| Mixtral-8x7B (BOS) | L21E001 | δ_e, WinoGrande STR rows | 19 | +0.111 | +0.109 | 1.018 | 0.996 | 0.004 | 0.032 |
| Mixtral-8x7B (BOS) | L21E001 | c_e, CounterFact clean prompts | 145 | +0.446 | +0.489 | 0.912 | 0.998 | 0.043 | 0.084 |
| Mixtral-8x7B (BOS) | L21E001 | c_e, WinoGrande prompts | 19 | +0.056 | +0.065 | 0.869 | 0.998 | 0.009 | 0.092 |
| Mixtral-8x7B (BOS) | L18E001 | δ_e, CounterFact STR rows | 642 | +0.255 | +0.258 | 0.988 | 0.999 | 0.008 | 0.028 |
| Mixtral-8x7B (BOS) | L18E001 | δ_e, WinoGrande STR rows | 6 | +0.093 | +0.092 | 1.010 | 1.000 | 0.002 | 0.019 |
| Mixtral-8x7B (BOS) | L18E001 | c_e, CounterFact clean prompts | 164 | +0.098 | +0.136 | 0.719 | 0.991 | 0.038 | 0.218 |
| Mixtral-8x7B (BOS) | L18E001 | c_e, WinoGrande prompts | 6 | +0.038 | +0.044 | 0.857 | 1.000 | 0.006 | 0.087 |
| Mixtral-8x7B (BOS) | L19E006 | δ_e, CounterFact STR rows | 484 | +0.240 | +0.241 | 0.999 | 0.999 | 0.007 | 0.029 |
| Mixtral-8x7B (BOS) | L19E006 | δ_e, WinoGrande STR rows | 503 | +0.445 | +0.448 | 0.994 | 1.000 | 0.004 | 0.009 |
| Mixtral-8x7B (BOS) | L19E006 | c_e, CounterFact clean prompts | 118 | +0.122 | +0.145 | 0.843 | 0.996 | 0.023 | 0.136 |
| Mixtral-8x7B (BOS) | L19E006 | c_e, WinoGrande prompts | 503 | +0.197 | +0.222 | 0.886 | 0.999 | 0.025 | 0.086 |

Exact final RMSNorm vs frozen norm for the single expert's direct effect: δ_e: Δ(norm(h_corrupt + δ_e)) − Δ(norm(h_corrupt)) vs δ_e·u / rms(h_corrupt); c_e: Δ(norm(h_clean)) − Δ(norm(h_clean − c_e)) vs c_e·u / rms(h_clean) (fp32, same path). Section: targets and each model's second WinoGrande expert; all experts of interest in `results/tables/ext11_B_norm.md`.

**Pooled over every δ_e vector of the experts of interest**

| model | task (δ_e rows) | vectors | Σ exact / Σ frozen | r | mean abs error (logits) | 95th pct abs error | Σ |err| / Σ |frozen| |
|---|---|---|---|---|---|---|---|
| Qwen3 | CounterFact | 6531 | 0.999 | 0.9997 | 0.0114 | 0.0461 | 0.020 |
| Qwen3 | WinoGrande | 4583 | 0.998 | 0.9999 | 0.0043 | 0.0154 | 0.009 |
| Mixtral (BOS) | CounterFact | 7069 | 0.996 | 0.9989 | 0.0132 | 0.0485 | 0.034 |
| Mixtral (BOS) | WinoGrande | 4723 | 0.998 | 0.9999 | 0.0045 | 0.0151 | 0.009 |

### Reading

**1. Qwen3: the localised experts are writers.** L41E117 writes its whole effect on WinoGrande directly (T +1.15 [+0.99, +1.32], D +1.20 [+1.02, +1.37] logits, r(D, T) across cases 0.87); so does L42E115 on CounterFact (share 0.96). L44E069 writes 1.59 logits but the patch only gains 1.21: the three layers after it remove about a quarter of its write (indirect effect negative). The same holds for every Qwen3 expert at or above the crossover layer (L43E081 1.42, L44E122 1.64, L44E006 1.33 ...), while experts a few layers below (L39E071 0.72, L34E119 0.52, L31E124 0.51 on WinoGrande) are partly computers whose effect later layers amplify.

**2. Mixtral: partly writers.** L20E000 writes 0.65 of its 1.15 logits (share 0.56 [0.52, 0.62]); L19E006, the second WinoGrande expert, 0.43; L16E007 0.20 (a computer). The CounterFact experts are closer to writers (L21E001 0.87, L19E002 0.71, L18E001 0.54). The late WinoGrande experts (L24E002 1.02, L26E002 1.46, L22E005, L25E002 > 1) write more than they gain. In Mixtral the band L16-L21, where the WinoGrande patches peak (ext7 W2: L20), does about half of its work through layers 22-31.

**3. Depth profile.** In every task the direct share rises with depth: experts in the first half of the network have total effects near zero and almost no direct effect (they act, if at all, through later layers); in the band just below the peak the share is 0.4-0.8 (later layers amplify); at and above a crossover layer the share is ≥ 1 (later layers partly cancel the write, the downstream self-repair known from dense models, McGrath et al. 2023); Mixtral CounterFact has no crossover (share 0.8-0.9 from L19 to L28). The experts of the last two or three layers (Qwen3 L46-L47, Mixtral L29-L31) push against the answer directly (negative D and T; at the very last layer T = D). Summed over all experts the share is Qwen3 CounterFact 0.81, Qwen3 WinoGrande 0.90, Mixtral CounterFact 0.67, Mixtral WinoGrande 0.63.

**4. Why the DLA ordering works (ext8).** Ranking by D recovers the top experts by T (top-10 overlap 10/10, 9/10, 10/10, 7/10), because the strongest experts sit at or above the crossover where D ≥ T. In Mixtral WinoGrande, D promotes the late over-writers (L26E002 first) and demotes the half-writers of L19-L21 (L19E006 rank 6 under D), which is where ext8 found DLA slightly below the oracle (AUC 0.55 vs 0.58). A good DLA ranking is therefore evidence that the top experts write, not that every expert does.

**5. The WinoGrande experts are slot detectors that write context.** Qwen3 L41E117 is routed at the final position of 84 % of the main WinoGrande prompts (100 % after 'too', 100 % after 'very', 76 % after 'was'), at the same tokens earlier in the same sentences (80 %), and at 86 % of the context-free local prompts; in wikitext at 66 % of copulas, 59 % of degree adverbs and 77 % of copula / degree tokens followed by a WinoGrande trigger word (base 7 %); its most selective next words in wikitext are predicative adjectives and participles (important, better, too, highly, unable, able, significant). It never fires at the IOI final token (0 %) and rarely at CounterFact's (11 %). Mixtral's L20E000 is the same kind of expert, less selective: 96 % of WinoGrande final positions, 95 % of local prompts, 94 % of wikitext copulas (base 22 %), and also 33 % of CounterFact final positions. The routing decision is therefore local (the token and its syntactic slot), but what the expert writes is not: its DLA toward the right trigger is +0.58 logits in the full prompt and +0.09 in the local prompt of the same case (Mixtral +0.33 / +0.06), where the model itself barely knows the answer (local Δ +0.55 vs +4.16). The expert reads the option-dependent information from its input residual and turns it into the trigger logit.

**6. The CounterFact experts are broad name-slot experts.** Qwen3 L44E069 / L42E115 are routed at 45 % / 40 % of all wikitext tokens, at 94 % / 95 % before a CounterFact place name and 72 % / 82 % before other capitalised words, and at 21 % / 13 % of copulas: yes, L44E069 fires before place names, but as part of a general 'a name follows' slot, not place-specifically. L42E115 is routed at 100 % of IOI final tokens (' to') and writes nothing there (DLA -0.02). What they write is relation-specific: L44E069's DLA when routed is +1.08 logits on place relations, +0.38 on languages, +0.12 on organisations and +0.01 on occupations (L42E115 +0.41 / +0.51 / +0.33 / +0.07). In Mixtral, L18E001 and L21E001 are preposition-slot experts (wikitext ' in' + place 95 % / 83 %; L18E001's most selective next words are months; IOI final ' to' 100 % / 94 % with DLA ≈ 0), L19E002 is a pre-entity expert that does not fire on IOI (0 %).

**7. What the writes say (vocabulary projections).** Projected on the whole vocabulary (151,936 tokens), the δ_e of the Qwen3 writers put the answer high: L41E117 median rank 98 (r in the top 100 in 50 % of rows, rank 12 of 1648 within the trigger vocabulary; r′ at median rank 98 from the bottom, the mirror image because the two directions of a pair carry δ_e of opposite sign), L42E115 1,287, L43E081 3,458, L44E069 5,758 (top 0.06 % to 3.8 %; within the CounterFact objects of the case's relation group L44E069 ranks r 18 of 168). The clean outputs c_e alone are already answer-like: L41E117's c_e in the WinoGrande prompts ranks the correct trigger at median 131 (most promoted ' small', '小', 'small', ' SMALL', ' smaller'), L43E081's at 30 with trigger words making up 29 % of its top 50 (base 1.1 %), L42E115's c_e on CounterFact at 105 (languages and nationalities: ' Catalan', ' French', ' Old', ' Hindi', ' Standard'), L44E069's at 921 (' New', ' Tur', ' Prov', ' Norm', ' Southern'). In the context-free local prompts L41E117's c_e still promotes the same adjective family (' small', ' SMALL', 'small', ' smaller') but ranks the specific trigger only at median 4826: the expert writes a property axis (size, weight, temperature, ...) whose direction is set by the context it receives, which is why its DLA toward r vs r′ collapses without the context (C2). The class means (z) are small because the trigger vocabulary (1,648 words) and the object classes are broad; the top-50 shares carry the class signal.

In Mixtral (vocabulary 32,000) the clean outputs of the targets rank the answer near the top: L20E000's c_e in the WinoGrande prompts at median 10 (top-10 in 51 %, top-100 in 79 %; most promoted ' large', ' small', ' crowded', ' tall', ' big'; δ_e 521), in the local prompts only at 497 (the same pattern as Qwen3); the CounterFact experts' c_e at 7 (L21E001), 59 (L19E002) and 544 (L18E001; δ_e 72, 252, 3,303), i.e. L18E001, the most indirect of the three in Part A, also writes the least answer-like vector. Many Mixtral vectors also promote a handful of answer-unrelated tokens (' /******/', byte-order mark, code fragments), a shared direction that is irrelevant to the logit difference (which reads only r and r′) and hence to DLA.

### Caveats

- T and D come from different passes (T: the ext6 / ext7 single-expert rows; D: ext8 pass 0). The pass-to-pass SD of a single-expert rescue is Qwen3 CounterFact 0.15, Qwen3 WinoGrande 0.30, Mixtral CounterFact 0.12, Mixtral WinoGrande 0.08 logits (ext8 in-pass singles of each first-donor row's top-10 vs the source rows; test-retest r 0.97, 0.83, 0.96, 0.97). Pairs with |T| below these values are noise; the bands' r over all pairs is dominated by them, the shares are not.
- D freezes the final RMSNorm at the corrupted run's scale. For the summed MoE writes the exact-norm direct effect is known (ext8 `addback_direct.parquet`) and differs by ≤ 0.006 of the drop on the mean; for single experts the last-layer check (T − D with no downstream computation) gives +0.053, +0.006, +0.002, +0.006 logits per expert on average and the exact computation of Part B (B3) Qwen3 Σ exact / Σ frozen 0.999 / 0.998; Mixtral Σ exact / Σ frozen 0.996 / 0.998.
- 'Indirect' includes interactions with every later layer at the final position only (the K/V of earlier positions are the corrupted run's); it does not separate later attention from later MoE layers (that needs path patching with downstream components frozen, not available in the engine).
- Direct share is a ratio of sums; for experts whose total is near zero (early layers, CounterFact experts on WinoGrande and vice versa, n active < 20) it is undefined or has very wide CIs.
- Final position only, STR only, Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS.
- Part B uses the ext9 engine's `DiagSpec.contrib_final_vectors` (E4c). Its OLMoE verification (`results/verify_ext9_engine_olmoe.json`): Σ slots vs the fp32 MoE output max |diff| 6.0e-08, vector norms vs `route_cnorm` 7.2e-07, per-expert vectors vs transformers hooks median relative norm difference 0.022 (max 0.13, n = 383; bf16 level). In-run checks: the frozen projection of c_e equals the engine's `contrib_dla` exactly, and the δ_e DLA reproduces ext8's DLA from a different pass (r Qwen3 0.999 CounterFact / 0.992 WinoGrande; Mixtral 0.999 CounterFact / 1.000 WinoGrande).
- Token classes are first tokens of words (CounterFact objects, triggers, names); a projection that promotes a different piece of the same word, another casing or a translation (' small', 'small', '小') counts only for the exact class token. Top-promoted tokens list the share of vectors with the token in their top 10.

### Files

- Code: `moetrace/ext11_writer.py` (loaders, cluster-bootstrap statistics, token classes), `scripts/ext11_writer_partA.py` (Part A), `scripts/ext11_writer_routing.py` + `scripts/ext11_chain_routing.sh` (Part C GPU passes), `scripts/ext11_writer_routing_analyze.py` (Part C analysis), `scripts/ext11_writer_vocab.py` + `scripts/ext11_chain_vocab.sh` (Part B GPU passes), `scripts/ext11_writer_vocab_analyze.py` (Part B analysis), `scripts/ext11_writer_text.py` (this section).
- Runs: `results/{qwen3,mixtral_bos}_writer_routing` (prompts_input, final_routing, token_events, run_meta), `results/{qwen3,mixtral_bos}_writer_vocab` (prompts, vec_rows = one row per projected vector, vec_top = top-20 promoted / suppressed ids, run_meta); raw arrays in `/opt/dlami/nvme/moe_ext11/` (routing chunks, vectors.pt; ephemeral).
- Numbers: `results/ext11_writer_summary.json` (keys A, C, B); tables `results/tables/ext11_*.md|csv`; figures `results/figures/ext11_*`.
- GPU time (gpu_queue jobs): Part C 8.0 min (Qwen3 176 s, Mixtral 303 s), Part B 4.0 min (Qwen3 96 s, Mixtral 146 s); Part A is CPU only.
