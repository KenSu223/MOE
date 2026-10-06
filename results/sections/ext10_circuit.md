**Summary.** With single attention heads as candidates next to experts, a handful of components restores the corrupted answer at the final position, and the ceiling is 1 instead of the all-MoE ceiling (Qwen3 CounterFact 0.53, Mixtral CounterFact 0.41, Qwen3 WinoGrande 0.86, Mixtral WinoGrande 0.79 of the drop). Adaptive greedy over heads ∪ experts (exact joint patches, 20 steps) restores, at k = 20, Qwen3 CounterFact **0.97 [0.95, 1.00]**, Mixtral CounterFact **1.00 [0.98, 1.03]**, Qwen3 WinoGrande **0.93 [0.91, 0.96]**, Mixtral WinoGrande **0.94 [0.93, 0.95]** of the drop (k = 10: Qwen3 CounterFact 0.88, Mixtral CounterFact 0.84, Qwen3 WinoGrande 0.80, Mixtral WinoGrande 0.80); k for 50 / 80 / 90 % of the drop: Qwen3 CounterFact 3/8/12, Mixtral CounterFact 4/9/13, Qwen3 WinoGrande 4/10/17, Mixtral WinoGrande 4/10/16. Experts alone (ext8 greedy) reach Qwen3 CounterFact 0.55, Mixtral CounterFact 0.56, Qwen3 WinoGrande 0.72, Mixtral WinoGrande 0.79 at k = 15, heads alone (head-only greedy) Qwen3 CounterFact 0.85, Mixtral CounterFact 0.87, Qwen3 WinoGrande 0.58, Mixtral WinoGrande 0.65 at k = 20. The answer is restored (Δ > 0) at k = 20 in Qwen3 CounterFact 0.94, Mixtral CounterFact 0.99, Qwen3 WinoGrande 0.96, Mixtral WinoGrande 1.00 of cases; rows with r ≥ 0.9 at k = 20: Qwen3 CounterFact 0.72, Mixtral CounterFact 0.81, Qwen3 WinoGrande 0.53, Mixtral WinoGrande 0.73. Heads among the first 10 mixed-greedy picks: Qwen3 CounterFact 6.2, Mixtral CounterFact 6.4, Qwen3 WinoGrande 3.6, Mixtral WinoGrande 2.8. The patch-free DLA ordering over heads ∪ experts reaches AUC (log k, 1..20) Qwen3 CounterFact 0.57, Mixtral CounterFact 0.52, Qwen3 WinoGrande 0.50, Mixtral WinoGrande 0.49 vs the single-patch oracle Qwen3 CounterFact 0.57, Mixtral CounterFact 0.53, Qwen3 WinoGrande 0.50, Mixtral WinoGrande 0.54 and greedy Qwen3 CounterFact 0.64, Mixtral CounterFact 0.61, Qwen3 WinoGrande 0.59, Mixtral WinoGrande 0.58. IOI (S2 → IO, Qwen3; attention-pole reference): Qwen3 IOI the mixed greedy picks heads almost exclusively (9.9 of the first 10), r(10) / r(20) = 0.78 / 0.94 [0.93, 0.96], 80 % at k = 11, while all MoE outputs give -0.27. Single-head patches at every layer (Step 1): the best single head per task is Qwen3 CounterFact L40H13 (+0.19 [+0.17, +0.22]), Mixtral CounterFact L18H4 (+0.15 [+0.13, +0.18]), Qwen3 WinoGrande L46H24 (+0.10 [+0.09, +0.11]), Mixtral WinoGrande L25H9 (+0.07 [+0.07, +0.08]) of the drop; per row the best head matches or beats the best expert on CounterFact and is weaker on WinoGrande (median best head / best expert Qwen3 CounterFact 0.19 / 0.19, Mixtral CounterFact 0.21 / 0.12, Qwen3 WinoGrande 0.15 / 0.20, Mixtral WinoGrande 0.11 / 0.19); Z8 (≥ 2 SD over all heads) detects Qwen3 CounterFact 15 positive / 5 negative of 1,536, Mixtral CounterFact 12 positive / 5 negative of 1,024, Qwen3 WinoGrande 15 positive / 12 negative of 1,536, Mixtral WinoGrande 18 positive / 5 negative of 1,024 heads. The head DLA agrees with the single-head patches at the population level (r over heads Qwen3 CounterFact 0.96, Mixtral CounterFact 0.92, Qwen3 WinoGrande 0.87, Mixtral WinoGrande 0.79; top-1 head agrees in Qwen3 CounterFact 0.71, Mixtral CounterFact 0.44, Qwen3 WinoGrande 0.41, Mixtral WinoGrande 0.47 of rows).

### What was run

**Rows.** Exactly the ext8 validation rows: CounterFact STR (Direction 6 donors, paper IDs and split; (case, donor) rows, donor mean primary, first donor sensitivity) and WinoGrande option swap (ext7, directed cases, bootstrap over pairs), Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS, bf16; plus IOI (i) S2 → IO (ext7-controls pairs, main validation family = 256 directed cases / 128 pairs) for Qwen3 as the attention-pole reference. Corruption = STR only; denoising (parent = corrupted run, source = clean run) at the final position.

**Step 1 passes (one chunk of validation rows per pass; every (layer, head) + one attn_layer row per layer per row)**

| Model | Task | Rows / cases | Passes | Spawn rows | Peak GB | Pass time (min) |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | CounterFact STR | 432 / 108 | 3 | 684,288 | 19.1 | 2.7 |
| Mixtral-8x7B (BOS) | CounterFact STR | 436 / 106 | 4 | 460,416 | 14.3 | 6.3 |
| Qwen3-30B-A3B-Base | WinoGrande STR | 256 / 256 | 2 | 405,504 | 17.1 | 2.5 |
| Mixtral-8x7B (BOS) | WinoGrande STR | 256 / 256 | 3 | 270,336 | 11.8 | 4.4 |
| Qwen3-30B-A3B-Base | IOI STR (S2 -> IO) | 256 / 256 | 2 | 405,504 | 17.0 | 2.0 |

**Step 2 passes (all tasks of a model share every pass: Qwen3 CounterFact + WinoGrande + IOI, Mixtral CounterFact + WinoGrande; ≤ 100k / 95k spawn rows per pass)**

| Model | Passes | Pass time (min) | Peak GB |
|---|---|---|---|
| qwen3 | 19 | 15.7 | 12.5 |
| mixtral | 19 | 28.2 | 16.1 |

GPU time of all ext10 queue jobs (incl. engine load, smoke tests): 79 min.


### Step 1. Single heads at every layer and the patch-free head DLA

Every (layer, head) of the model was patched alone at the final position (attn_head: the head's pre-o_proj output set to the clean run's, the rest of the corrupted attention output kept) on every ext8 validation row, with one attn_layer row per layer in the same pass. The head DLA uses the same per-head output differences without any patch: dla_h = (W_o[:, h] ΔH_h ⊙ γ) · (W_U[r] − W_U[r′]) / rms(h_final, corrupted run). Z8 = population effect ≥ 2 SD from the mean over all heads of the model, on all validation units and separately on two disjoint halves.

| Run | Rows / cases | Mean drop | Heads Z8 + / − (both halves) | Σ single heads / drop (all; positive pop. means) | Pop. heads for 50/80/90 % (additive) | Best head / best expert per row (median) | Rows best head > best expert | Heads in joint top-32 (median [IQR]) | DLA vs single, heads: row r / row ρ top-32 | top-1 / top-10 agree | pop. r | experts: row r | Σ head DLA vs ext8 attention DLA (r, ratio) | Σ heads vs attn_layer (row-layer r) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base, CounterFact STR | 432 / 108 | 11.66 | 15 / 5 (18) | 0.31, 1.09 | 7/32/69 | 0.189 / 0.186 | 0.53 | 14 [11, 19] | 0.63 / 0.17 | 0.71 / 0.65 | 0.96 | 0.72 | 0.999, 0.999 | 0.37 |
| Mixtral-8x7B (BOS), CounterFact STR | 436 / 106 | 12.98 | 12 / 5 (16) | 0.50, 1.03 | 6/27/55 | 0.214 / 0.123 | 0.75 | 17 [14, 20] | 0.76 / 0.35 | 0.44 / 0.70 | 0.92 | 0.74 | 0.999, 1.000 | 0.45 |
| Qwen3-30B-A3B-Base, WinoGrande STR | 256 / 256 | 8.16 | 15 / 12 (23) | 0.02, 1.36 | 21/78/115 | 0.151 / 0.204 | 0.22 | 12 [7, 18] | 0.33 / -0.39 | 0.41 / 0.28 | 0.87 | 0.53 | 0.989, 1.029 | 0.39 |
| Mixtral-8x7B (BOS), WinoGrande STR | 256 / 256 | 7.69 | 18 / 5 (19) | 0.95, 1.19 | 22/107/183 | 0.112 / 0.192 | 0.16 | 12 [10, 16] | 0.41 / -0.09 | 0.47 / 0.39 | 0.79 | 0.81 | 0.999, 0.998 | 0.31 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | 256 / 256 | 12.68 | 29 / 14 (40) | 1.41, 2.29 | 6/14/17 | 0.184 / 0.015 | 1.00 | 32 [31, 32] | 0.62 / 0.31 | 0.58 / 0.62 | 0.76 | 1.00 | 0.992, 1.003 | 0.41 |


'Σ single heads' = the sum of all single-head population effects (all heads at all layers jointly restore 1.00: the attention outputs are the only path of the corruption to the final position); 'Pop. heads for q' = the number of heads, taken in order of their population effect, whose single effects add up to q of the drop (an additive estimate; Step 2 measures joint patches); 'Heads in joint top-32' = how many of a row's 32 best single components (heads ∪ the row's clean-active experts) are heads. IOI (S2 → IO, Qwen3 only, the attention-pole reference): no single-expert patch table exists, so its expert columns use the expert DLA of the same pass ('experts: row r' = 1 by construction).

**Top and bottom heads (validation; donor mean for CounterFact)**

| Run | Head | Rank | Rescue / drop [95% CI] | Rescue (logits) | z (all / halves) | Z8 both halves | DLA / drop | Share of layer attn | Row top-5 / top-1 share |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base, CounterFact STR | L40H13 | 1 | +0.192 [+0.169, +0.216] | +2.23 | +28.6 / +29.8, +27.0 | yes | +0.115 | 0.57 | 0.84 / 0.65 |
| Qwen3-30B-A3B-Base, CounterFact STR | L40H15 | 2 | +0.079 [+0.072, +0.087] | +0.93 | +11.8 / +10.6, +13.0 | yes | +0.047 | 0.24 | 0.84 / 0.09 |
| Qwen3-30B-A3B-Base, CounterFact STR | L40H14 | 3 | +0.073 [+0.065, +0.082] | +0.85 | +10.9 / +10.4, +11.2 | yes | +0.047 | 0.22 | 0.66 / 0.16 |
| Qwen3-30B-A3B-Base, CounterFact STR | L43H11 | 4 | +0.056 [+0.046, +0.067] | +0.65 | +8.4 / +8.9, +7.7 | yes | +0.060 | 0.27 | 0.44 / 0.01 |
| Qwen3-30B-A3B-Base, CounterFact STR | L43H15 | 5 | +0.045 [+0.038, +0.054] | +0.53 | +6.8 / +7.3, +6.1 | yes | +0.057 | 0.21 | 0.37 / 0.00 |
| Qwen3-30B-A3B-Base, CounterFact STR | L43H28 | 6 | +0.035 [+0.028, +0.042] | +0.41 | +5.2 / +5.0, +5.4 | yes | +0.030 | 0.17 | 0.33 / 0.03 |
| Qwen3-30B-A3B-Base, CounterFact STR | L28H13 | 7 | +0.033 [+0.025, +0.041] | +0.38 | +4.9 / +4.4, +5.3 | yes | +0.025 | 0.46 | 0.15 / 0.00 |
| Qwen3-30B-A3B-Base, CounterFact STR | L40H11 | 8 | +0.032 [+0.028, +0.037] | +0.38 | +4.8 / +4.6, +5.0 | yes | +0.025 | 0.10 | 0.17 / 0.00 |
| Qwen3-30B-A3B-Base, CounterFact STR | L40H9 | 1536 | -0.063 [-0.069, -0.056] | -0.73 | -9.4 / -8.1, -10.7 | yes | -0.062 | -0.19 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, CounterFact STR | L40H8 | 1535 | -0.035 [-0.040, -0.032] | -0.41 | -5.3 / -5.2, -5.4 | yes | -0.030 | -0.10 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, CounterFact STR | L47H29 | 1534 | -0.034 [-0.038, -0.030] | -0.39 | -5.1 / -5.0, -5.1 | yes | -0.036 | -2.37 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), CounterFact STR | L18H4 | 1 | +0.154 [+0.126, +0.184] | +2.00 | +19.1 / +18.6, +19.6 | yes | +0.063 | 0.87 | 0.66 / 0.39 |
| Mixtral-8x7B (BOS), CounterFact STR | L24H22 | 2 | +0.147 [+0.128, +0.166] | +1.91 | +18.2 / +18.1, +18.3 | yes | +0.150 | 0.78 | 0.77 / 0.27 |
| Mixtral-8x7B (BOS), CounterFact STR | L19H29 | 3 | +0.061 [+0.052, +0.070] | +0.79 | +7.5 / +7.2, +7.8 | yes | +0.042 | 0.38 | 0.55 / 0.01 |
| Mixtral-8x7B (BOS), CounterFact STR | L29H2 | 4 | +0.051 [+0.045, +0.058] | +0.66 | +6.3 / +6.1, +6.4 | yes | +0.069 | 0.72 | 0.42 / 0.04 |
| Mixtral-8x7B (BOS), CounterFact STR | L24H21 | 5 | +0.046 [+0.037, +0.055] | +0.60 | +5.6 / +6.8, +4.5 | yes | +0.049 | 0.24 | 0.42 / 0.06 |
| Mixtral-8x7B (BOS), CounterFact STR | L24H23 | 6 | +0.042 [+0.033, +0.050] | +0.54 | +5.1 / +5.2, +5.1 | yes | +0.042 | 0.22 | 0.35 / 0.01 |
| Mixtral-8x7B (BOS), CounterFact STR | L15H1 | 7 | +0.035 [+0.029, +0.040] | +0.45 | +4.2 / +4.4, +4.0 | yes | +0.019 | 0.57 | 0.19 / 0.03 |
| Mixtral-8x7B (BOS), CounterFact STR | L31H25 | 8 | +0.031 [+0.028, +0.035] | +0.41 | +3.8 / +3.9, +3.7 | yes | +0.037 | 0.60 | 0.18 / 0.01 |
| Mixtral-8x7B (BOS), CounterFact STR | L24H20 | 1024 | -0.046 [-0.056, -0.038] | -0.60 | -5.8 / -6.5, -5.2 | yes | -0.048 | -0.25 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), CounterFact STR | L29H0 | 1023 | -0.024 [-0.028, -0.021] | -0.32 | -3.1 / -3.1, -3.1 | yes | -0.036 | -0.35 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), CounterFact STR | L18H6 | 1022 | -0.021 [-0.026, -0.017] | -0.28 | -2.7 / -2.7, -2.7 | yes | -0.021 | -0.12 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L46H24 | 1 | +0.101 [+0.088, +0.113] | +0.82 | +18.2 / +17.8, +17.3 | yes | +0.112 | -10.68 | 0.64 / 0.30 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L41H27 | 2 | +0.065 [+0.056, +0.075] | +0.53 | +11.8 / +11.8, +11.0 | yes | +0.033 | 3.05 | 0.45 / 0.21 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L38H18 | 3 | +0.041 [+0.036, +0.048] | +0.34 | +7.5 / +7.6, +6.8 | yes | +0.022 | 4.22 | 0.28 / 0.07 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L29H0 | 4 | +0.041 [+0.031, +0.052] | +0.34 | +7.4 / +6.3, +8.0 | yes | +0.001 | 5.41 | 0.26 / 0.10 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L38H21 | 5 | +0.035 [+0.029, +0.042] | +0.29 | +6.3 / +6.7, +5.5 | yes | +0.022 | 3.57 | 0.20 / 0.03 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L41H0 | 6 | +0.021 [+0.017, +0.027] | +0.18 | +3.9 / +3.5, +4.0 | yes | +0.002 | 1.00 | 0.07 / 0.02 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L42H30 | 7 | +0.021 [+0.017, +0.024] | +0.17 | +3.7 / +3.1, +4.1 | yes | +0.010 | 0.71 | 0.06 / 0.00 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L38H19 | 8 | +0.019 [+0.015, +0.022] | +0.15 | +3.4 / +2.7, +3.9 | yes | +0.009 | 1.90 | 0.07 / 0.00 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L46H25 | 1536 | -0.105 [-0.115, -0.095] | -0.86 | -19.0 / -18.6, -18.1 | yes | -0.126 | 11.17 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L38H16 | 1535 | -0.067 [-0.074, -0.060] | -0.54 | -12.0 / -11.4, -11.9 | yes | -0.048 | -6.77 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L41H24 | 1534 | -0.039 [-0.043, -0.035] | -0.32 | -7.1 / -6.9, -6.8 | yes | -0.020 | -1.83 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), WinoGrande STR | L25H9 | 1 | +0.073 [+0.066, +0.079] | +0.56 | +15.7 / +16.0, +15.0 | yes | +0.085 | 0.97 | 0.78 / 0.30 |
| Mixtral-8x7B (BOS), WinoGrande STR | L22H20 | 2 | +0.072 [+0.063, +0.081] | +0.55 | +15.5 / +15.5, +15.2 | yes | +0.043 | 2.18 | 0.73 / 0.24 |
| Mixtral-8x7B (BOS), WinoGrande STR | L19H13 | 3 | +0.047 [+0.042, +0.053] | +0.36 | +10.1 / +9.8, +10.2 | yes | +0.001 | 0.42 | 0.57 / 0.15 |
| Mixtral-8x7B (BOS), WinoGrande STR | L13H11 | 4 | +0.026 [+0.019, +0.034] | +0.20 | +5.5 / +5.8, +5.0 | yes | +0.001 | 0.19 | 0.16 / 0.04 |
| Mixtral-8x7B (BOS), WinoGrande STR | L13H18 | 5 | +0.026 [+0.021, +0.030] | +0.20 | +5.4 / +4.7, +5.9 | yes | -0.000 | 0.18 | 0.23 / 0.05 |
| Mixtral-8x7B (BOS), WinoGrande STR | L19H12 | 6 | +0.025 [+0.020, +0.030] | +0.19 | +5.2 / +4.2, +6.1 | yes | +0.008 | 0.22 | 0.23 / 0.02 |
| Mixtral-8x7B (BOS), WinoGrande STR | L13H4 | 7 | +0.022 [+0.019, +0.026] | +0.17 | +4.6 / +4.3, +4.9 | yes | -0.000 | 0.16 | 0.14 / 0.03 |
| Mixtral-8x7B (BOS), WinoGrande STR | L15H7 | 8 | +0.022 [+0.015, +0.030] | +0.17 | +4.5 / +4.1, +4.9 | yes | +0.009 | 0.30 | 0.12 / 0.02 |
| Mixtral-8x7B (BOS), WinoGrande STR | L22H23 | 1024 | -0.035 [-0.038, -0.031] | -0.27 | -7.7 / -8.1, -7.2 | yes | -0.025 | -1.05 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), WinoGrande STR | L25H11 | 1023 | -0.020 [-0.022, -0.018] | -0.15 | -4.5 / -4.3, -4.6 | yes | -0.023 | -0.26 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), WinoGrande STR | L20H25 | 1022 | -0.013 [-0.016, -0.010] | -0.10 | -3.1 / -2.7, -3.3 | yes | -0.001 | -0.47 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L42H11 | 1 | +0.178 [+0.171, +0.184] | +2.25 | +20.0 / +20.0, +19.9 | yes | +0.238 | 0.67 | 1.00 / 0.82 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L43H24 | 2 | +0.100 [+0.095, +0.105] | +1.27 | +11.2 / +11.0, +11.4 | yes | +0.090 | 0.47 | 0.82 / 0.02 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L40H2 | 3 | +0.071 [+0.068, +0.075] | +0.91 | +8.0 / +7.7, +8.2 | yes | +0.002 | 116.00 | 0.39 / 0.00 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L42H10 | 4 | +0.069 [+0.064, +0.074] | +0.88 | +7.7 / +7.7, +7.7 | yes | +0.214 | 0.26 | 0.43 / 0.00 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L43H29 | 5 | +0.063 [+0.060, +0.066] | +0.80 | +7.0 / +7.1, +6.9 | yes | -0.089 | 0.29 | 0.26 / 0.00 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L43H28 | 6 | +0.050 [+0.048, +0.053] | +0.64 | +5.6 / +5.8, +5.4 | yes | +0.045 | 0.24 | 0.05 / 0.00 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L47H26 | 7 | +0.043 [+0.040, +0.046] | +0.54 | +4.7 / +4.7, +4.7 | yes | +0.050 | 3.26 | 0.05 / 0.00 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L45H9 | 8 | +0.040 [+0.036, +0.045] | +0.51 | +4.5 / +4.4, +4.5 | yes | +0.115 | 0.12 | 0.10 / 0.00 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L44H15 | 1536 | -0.122 [-0.129, -0.115] | -1.55 | -13.9 / -13.6, -14.1 | yes | -0.069 | 0.80 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L47H29 | 1535 | -0.088 [-0.092, -0.084] | -1.12 | -10.1 / -10.2, -10.0 | yes | -0.099 | -6.74 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L44H28 | 1534 | -0.045 [-0.048, -0.042] | -0.57 | -5.2 / -5.0, -5.4 | yes | -0.022 | 0.29 | 0.00 / 0.00 |


**Heads found in earlier directions (F2 under GN on CounterFact, W5 under STR on WinoGrande at the W2 attention layers)**

| Run | Head | Earlier result | Rank (of all heads) | Rescue / drop | z | Z8 both halves | DLA / drop |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base, CounterFact STR | L40H13 | F2 GN mover (L40H13) | 1 | +0.192 [+0.169, +0.216] | +28.6 | yes | +0.115 |
| Mixtral-8x7B (BOS), CounterFact STR | L18H4 | F2 GN mover (L18H4) | 1 | +0.154 [+0.126, +0.184] | +19.1 | yes | +0.063 |
| Mixtral-8x7B (BOS), CounterFact STR | L24H22 | F2 GN (L24H22) | 2 | +0.147 [+0.128, +0.166] | +18.2 | yes | +0.150 |
| Mixtral-8x7B (BOS), CounterFact STR | L15H1 | F2 GN (L15H1) | 7 | +0.035 [+0.029, +0.040] | +4.2 | yes | +0.019 |
| Mixtral-8x7B (BOS), CounterFact STR | L15H3 | F2 GN (L15H3) | 11 | +0.018 [+0.014, +0.023] | +2.2 | yes | +0.013 |
| Mixtral-8x7B (BOS), CounterFact STR | L19H29 | F2 GN (L19H29) | 3 | +0.061 [+0.052, +0.070] | +7.5 | yes | +0.042 |
| Mixtral-8x7B (BOS), CounterFact STR | L19H30 | F2 GN (L19H30) | 9 | +0.029 [+0.025, +0.033] | +3.5 | yes | +0.020 |
| Mixtral-8x7B (BOS), CounterFact STR | L19H31 | F2 GN (L19H31) | 28 | +0.006 [+0.004, +0.009] | +0.7 | no | +0.005 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L38H18 | W5 L38H18 | 3 | +0.041 [+0.036, +0.048] | +7.5 | yes | +0.022 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L38H21 | W5 L38H21 | 5 | +0.035 [+0.029, +0.042] | +6.3 | yes | +0.022 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L38H16 | W5 L38H16 (negative) | 1535 | -0.067 [-0.074, -0.060] | -12.0 | yes | -0.048 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L42H30 | W5 L42H30 | 7 | +0.021 [+0.017, +0.024] | +3.7 | yes | +0.010 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L42H26 | W5 L42H26 (negative) | 1533 | -0.033 [-0.038, -0.027] | -5.9 | yes | -0.016 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L39H5 | W5 L39H5 | 9 | +0.017 [+0.014, +0.022] | +3.2 | yes | -0.000 |
| Mixtral-8x7B (BOS), WinoGrande STR | L25H9 | W5 L25H9 | 1 | +0.073 [+0.066, +0.079] | +15.7 | yes | +0.085 |
| Mixtral-8x7B (BOS), WinoGrande STR | L19H13 | W5 L19H13 | 3 | +0.047 [+0.042, +0.053] | +10.1 | yes | +0.001 |
| Mixtral-8x7B (BOS), WinoGrande STR | L13H18 | W5 L13H18 | 5 | +0.026 [+0.021, +0.030] | +5.4 | yes | -0.000 |
| Mixtral-8x7B (BOS), WinoGrande STR | L13H11 | W5 L13H11 | 4 | +0.026 [+0.019, +0.034] | +5.5 | yes | +0.001 |
| Mixtral-8x7B (BOS), WinoGrande STR | L19H12 | W5 L19H12 | 6 | +0.025 [+0.020, +0.030] | +5.2 | yes | +0.008 |
| Mixtral-8x7B (BOS), WinoGrande STR | L13H4 | W5 L13H4 | 7 | +0.022 [+0.019, +0.026] | +4.6 | yes | -0.000 |
| Mixtral-8x7B (BOS), WinoGrande STR | L25H11 | W5 L25H11 (negative) | 1023 | -0.020 [-0.022, -0.018] | -4.5 | yes | -0.023 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L42H11 | W5 IOI S2 reader L42H11 | 1 | +0.178 [+0.171, +0.184] | +20.0 | yes | +0.238 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L42H10 | W5 IOI name mover L42H10 | 4 | +0.069 [+0.064, +0.074] | +7.7 | yes | +0.214 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L43H24 | W5 IOI L43H24 | 2 | +0.100 [+0.095, +0.105] | +11.2 | yes | +0.090 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L43H29 | W5 IOI L43H29 | 5 | +0.063 [+0.060, +0.066] | +7.0 | yes | -0.089 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L42H14 | W5 IOI negative mover L42H14 | 1530 | -0.040 [-0.043, -0.037] | -4.7 | yes | -0.044 |


**The six layers with the largest attention-output patch: sum of single heads vs the layer patch**

| Run | Layer | attn_layer / drop | Σ heads / drop | r(Σ heads, attn_layer) rows | Top head | Top head / drop | Z8 + / − |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base, CounterFact STR | L40 | +0.337 [+0.309, +0.367] | +0.308 | 0.72 | L40H13 | +0.192 | 5 / 2 |
| Qwen3-30B-A3B-Base, CounterFact STR | L43 | +0.212 [+0.190, +0.233] | +0.176 | 0.70 | L43H11 | +0.056 | 5 / 0 |
| Qwen3-30B-A3B-Base, CounterFact STR | L28 | +0.072 [+0.056, +0.089] | +0.056 | 0.51 | L28H13 | +0.033 | 4 / 2 |
| Qwen3-30B-A3B-Base, CounterFact STR | L38 | +0.017 [+0.013, +0.022] | +0.009 | 0.33 | L38H8 | +0.007 | 0 / 0 |
| Qwen3-30B-A3B-Base, CounterFact STR | L16 | +0.017 [+0.010, +0.024] | +0.002 | 0.25 | L16H13 | +0.009 | 0 / 0 |
| Qwen3-30B-A3B-Base, CounterFact STR | L47 | +0.014 [+0.008, +0.020] | +0.011 | 0.46 | L47H26 | +0.024 | 1 / 1 |
| Mixtral-8x7B (BOS), CounterFact STR | L24 | +0.189 [+0.166, +0.212] | +0.186 | 0.75 | L24H22 | +0.147 | 3 / 1 |
| Mixtral-8x7B (BOS), CounterFact STR | L18 | +0.178 [+0.155, +0.203] | +0.183 | 0.68 | L18H4 | +0.154 | 3 / 1 |
| Mixtral-8x7B (BOS), CounterFact STR | L19 | +0.160 [+0.145, +0.176] | +0.112 | 0.42 | L19H29 | +0.061 | 2 / 0 |
| Mixtral-8x7B (BOS), CounterFact STR | L29 | +0.070 [+0.062, +0.079] | +0.073 | 0.50 | L29H2 | +0.051 | 1 / 1 |
| Mixtral-8x7B (BOS), CounterFact STR | L15 | +0.061 [+0.049, +0.074] | +0.044 | 0.38 | L15H1 | +0.035 | 2 / 1 |
| Mixtral-8x7B (BOS), CounterFact STR | L31 | +0.052 [+0.044, +0.062] | +0.053 | 0.54 | L31H25 | +0.031 | 1 / 1 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L42 | +0.029 [+0.022, +0.036] | +0.016 | 0.37 | L42H30 | +0.021 | 2 / 1 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L43 | +0.023 [+0.019, +0.027] | +0.009 | 0.16 | L43H28 | +0.011 | 0 / 0 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L40 | +0.023 [+0.015, +0.031] | +0.019 | 0.38 | L40H2 | +0.010 | 0 / 0 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L41 | +0.021 [+0.013, +0.030] | +0.021 | 0.30 | L41H27 | +0.065 | 2 / 3 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L39 | +0.016 [+0.008, +0.024] | +0.015 | 0.34 | L39H5 | +0.017 | 2 / 2 |
| Qwen3-30B-A3B-Base, WinoGrande STR | L26 | +0.015 [+0.002, +0.028] | +0.046 | 0.40 | L26H18 | +0.008 | 0 / 0 |
| Mixtral-8x7B (BOS), WinoGrande STR | L13 | +0.140 [+0.126, +0.155] | +0.102 | 0.34 | L13H11 | +0.026 | 3 / 0 |
| Mixtral-8x7B (BOS), WinoGrande STR | L19 | +0.113 [+0.103, +0.123] | +0.100 | 0.35 | L19H13 | +0.047 | 3 / 0 |
| Mixtral-8x7B (BOS), WinoGrande STR | L25 | +0.075 [+0.068, +0.082] | +0.087 | 0.44 | L25H9 | +0.073 | 2 / 1 |
| Mixtral-8x7B (BOS), WinoGrande STR | L15 | +0.072 [+0.063, +0.082] | +0.070 | 0.26 | L15H7 | +0.022 | 3 / 0 |
| Mixtral-8x7B (BOS), WinoGrande STR | L31 | +0.043 [+0.040, +0.047] | +0.058 | 0.46 | L31H22 | +0.021 | 2 / 1 |
| Mixtral-8x7B (BOS), WinoGrande STR | L16 | +0.036 [+0.029, +0.044] | +0.042 | 0.35 | L16H10 | +0.017 | 2 / 0 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L45 | +0.330 [+0.319, +0.340] | +0.337 | 0.52 | L45H9 | +0.040 | 7 / 0 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L42 | +0.264 [+0.253, +0.276] | +0.245 | 0.40 | L42H11 | +0.178 | 3 / 1 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L43 | +0.214 [+0.204, +0.224] | +0.219 | 0.38 | L43H24 | +0.100 | 5 / 3 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L39 | +0.149 [+0.140, +0.157] | +0.114 | 0.21 | L39H28 | +0.039 | 2 / 0 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L41 | +0.081 [+0.076, +0.085] | +0.090 | 0.30 | L41H20 | +0.035 | 3 / 0 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | L46 | +0.045 [+0.036, +0.054] | +0.051 | 0.49 | L46H14 | +0.031 | 2 / 2 |


**Additive top-k sums of single components per row (median, fraction of the drop; NOT joint patches)**

| Run | k | heads ∪ experts | heads only | experts only |
|---|---|---|---|---|
| Qwen3-30B-A3B-Base, CounterFact STR | 1 | 0.27 | 0.19 | 0.19 |
| Qwen3-30B-A3B-Base, CounterFact STR | 5 | 0.73 | 0.52 | 0.49 |
| Qwen3-30B-A3B-Base, CounterFact STR | 10 | 1.04 | 0.73 | 0.68 |
| Qwen3-30B-A3B-Base, CounterFact STR | 20 | 1.45 | 0.96 | 0.97 |
| Qwen3-30B-A3B-Base, CounterFact STR | 32 | 1.77 | 1.15 | 1.22 |
| Mixtral-8x7B (BOS), CounterFact STR | 1 | 0.23 | 0.21 | 0.12 |
| Mixtral-8x7B (BOS), CounterFact STR | 5 | 0.72 | 0.60 | 0.41 |
| Mixtral-8x7B (BOS), CounterFact STR | 10 | 1.05 | 0.79 | 0.61 |
| Mixtral-8x7B (BOS), CounterFact STR | 20 | 1.44 | 1.00 | 0.81 |
| Mixtral-8x7B (BOS), CounterFact STR | 32 | 1.73 | 1.17 | 0.89 |
| Qwen3-30B-A3B-Base, WinoGrande STR | 1 | 0.22 | 0.15 | 0.20 |
| Qwen3-30B-A3B-Base, WinoGrande STR | 5 | 0.74 | 0.45 | 0.66 |
| Qwen3-30B-A3B-Base, WinoGrande STR | 10 | 1.16 | 0.69 | 0.98 |
| Qwen3-30B-A3B-Base, WinoGrande STR | 20 | 1.77 | 1.07 | 1.50 |
| Qwen3-30B-A3B-Base, WinoGrande STR | 32 | 2.33 | 1.47 | 1.98 |
| Mixtral-8x7B (BOS), WinoGrande STR | 1 | 0.20 | 0.11 | 0.19 |
| Mixtral-8x7B (BOS), WinoGrande STR | 5 | 0.74 | 0.36 | 0.70 |
| Mixtral-8x7B (BOS), WinoGrande STR | 10 | 1.16 | 0.52 | 1.05 |
| Mixtral-8x7B (BOS), WinoGrande STR | 20 | 1.65 | 0.72 | 1.38 |
| Mixtral-8x7B (BOS), WinoGrande STR | 32 | 1.96 | 0.90 | 1.52 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | 1 | 0.18 | 0.18 | 0.01 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | 5 | 0.59 | 0.59 | 0.04 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | 10 | 0.89 | 0.88 | 0.06 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | 20 | 1.27 | 1.26 | 0.08 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | 32 | 1.54 | 1.53 | 0.10 |


![Single-head rescue, layer x head](figures/ext10_heads_heatmap.png)

![Head DLA vs single-head patch; layer curves](figures/ext10_heads_dla_layers.png)

### Step 2. Joint add-back over heads + experts

Every curve point is ONE exact joint patch (`multi` spawn, one step per layer carrying that layer's heads and experts: heads patched before the MoE, the row's own MoE recomputed on the head-patched input, the listed experts' contributions set to the clean run's), so later layers see earlier patches. Mixed greedy: pool = the row's top-32 heads by single-head rescue ∪ ext8's expert pool (Qwen3: the row's top-32 experts, Mixtral: all 64), 20 steps; head-only greedy: pool = the 32 heads; static orderings over ALL heads ∪ all clean-active experts by the row's single-patch rescue (oracle) or by DLA, and the same over heads only, on k = 1..10, 12, 16, 24, 32, 48, 64, 96, 128. ext8's expert-only greedy (15 steps, same rows) and its static expert curves are the comparison. r(k) = rescue / drop, ceiling 1 (all heads = all attention outputs = the clean final residual); the all-MoE ceiling is drawn for reference.

**Sanity checks (inside the passes)**

| Run | rows / cases | all heads (= 1) | all attention | all MoE | all heads: max |Δ − Δ_clean| | layer heads vs attn_layer: median / max |ΔΔ|, r | prefill SD across passes (median clean / corrupt) |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base, CounterFact STR | 432 / 108 | 0.999 | 1.000 | 0.530 [0.488, 0.571] | 1.875 | 0.062 / 0.625, 0.9994 | 0.094 / 0.096 |
| Mixtral-8x7B (BOS), CounterFact STR | 436 / 106 | 1.000 | 1.000 | 0.409 [0.366, 0.451] | 0.688 | 0.000 / 0.688, 0.9997 | 0.058 / 0.062 |
| Qwen3-30B-A3B-Base, WinoGrande STR | 256 / 256 | 1.006 | 1.005 | 0.856 [0.837, 0.875] | 2.625 | 0.125 / 2.000, 0.9964 | 0.183 / 0.183 |
| Mixtral-8x7B (BOS), WinoGrande STR | 256 / 256 | 1.001 | 1.001 | 0.788 [0.765, 0.810] | 0.250 | 0.000 / 0.375, 0.9992 | 0.062 / 0.062 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | 256 / 256 | 0.998 | 0.999 | -0.269 [-0.293, -0.244] | 0.500 | 0.125 / 0.500, 0.9978 | 0.155 / 0.151 |


**Curves (population r(k), validation; k50/80/90 = smallest k with r ≥ q of the drop; AUC over log k on the same k range for every curve; static curves interpolated in log k)**

| Run | Curve | r(1) | r(5) | r(10) | r(20) [95% CI] | r(128) | max r (k) | k50/80/90 of drop | AUC log k (1..15) | AUC log k (1..20) | all-MoE ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base, CounterFact STR | greedy_mix | 0.265 | 0.685 | 0.875 | 0.974 [0.953, 0.996] |  | 0.974 (20) | 3/8/12 | 0.611 | 0.644 | 0.530 |
| Qwen3-30B-A3B-Base, CounterFact STR | greedy_head | 0.210 | 0.639 | 0.797 | 0.855 [0.825, 0.884] |  | 0.855 (20) | 4/11/never | 0.551 | 0.580 | 0.530 |
| Qwen3-30B-A3B-Base, CounterFact STR | greedy_expert_ext8 | 0.200 | 0.443 | 0.523 | 0.548 (k=15) |  | 0.548 (15) | 8/never/never | 0.395 |  | 0.530 |
| Qwen3-30B-A3B-Base, CounterFact STR | static_mix_oracle | 0.263 | 0.604 | 0.754 |  | 0.953 | 0.953 (128) | 4/16/48 | 0.543 | 0.571 | 0.530 |
| Qwen3-30B-A3B-Base, CounterFact STR | static_mix_dla | 0.220 | 0.595 | 0.788 |  | 1.106 | 1.106 (128) | 4/12/24 | 0.535 | 0.571 | 0.530 |
| Qwen3-30B-A3B-Base, CounterFact STR | static_head_oracle | 0.210 | 0.579 | 0.727 |  | 0.881 | 0.881 (128) | 4/24/never | 0.509 | 0.536 | 0.530 |
| Qwen3-30B-A3B-Base, CounterFact STR | static_head_dla | 0.191 | 0.579 | 0.777 |  | 1.076 | 1.076 (128) | 4/12/24 | 0.509 | 0.545 | 0.530 |
| Qwen3-30B-A3B-Base, CounterFact STR | static_expert_oracle_ext8 | 0.201 | 0.407 | 0.466 |  | 0.589 | 0.607 (320) | 24/never/never | 0.367 | 0.380 | 0.530 |
| Qwen3-30B-A3B-Base, CounterFact STR | static_expert_dla_ext8 | 0.185 | 0.407 | 0.491 |  | 0.648 | 0.649 (192) | 12/never/never | 0.369 | 0.386 | 0.530 |
| Mixtral-8x7B (BOS), CounterFact STR | greedy_mix | 0.232 | 0.635 | 0.840 | 1.004 [0.983, 1.027] |  | 1.004 (20) | 4/9/13 | 0.573 | 0.612 | 0.409 |
| Mixtral-8x7B (BOS), CounterFact STR | greedy_head | 0.215 | 0.593 | 0.783 | 0.867 [0.838, 0.896] |  | 0.867 (20) | 4/11/never | 0.530 | 0.562 | 0.409 |
| Mixtral-8x7B (BOS), CounterFact STR | greedy_expert_ext8 | 0.132 | 0.393 | 0.509 | 0.561 (k=15) |  | 0.561 (15) | 10/never/never | 0.346 |  | 0.409 |
| Mixtral-8x7B (BOS), CounterFact STR | static_mix_oracle | 0.233 | 0.559 | 0.701 |  | 0.963 | 0.963 (128) | 4/24/48 | 0.507 | 0.535 | 0.409 |
| Mixtral-8x7B (BOS), CounterFact STR | static_mix_dla | 0.178 | 0.536 | 0.724 |  | 1.089 | 1.089 (128) | 5/16/24 | 0.484 | 0.520 | 0.409 |
| Mixtral-8x7B (BOS), CounterFact STR | static_head_oracle | 0.216 | 0.540 | 0.694 |  | 0.910 | 0.910 (128) | 5/24/128 | 0.487 | 0.515 | 0.409 |
| Mixtral-8x7B (BOS), CounterFact STR | static_head_dla | 0.167 | 0.505 | 0.710 |  | 1.089 | 1.089 (128) | 5/16/24 | 0.461 | 0.499 | 0.409 |
| Mixtral-8x7B (BOS), CounterFact STR | static_expert_oracle_ext8 | 0.132 | 0.348 | 0.431 |  |  | 0.517 (48) | 32/never/never | 0.307 | 0.324 | 0.409 |
| Mixtral-8x7B (BOS), CounterFact STR | static_expert_dla_ext8 | 0.103 | 0.326 | 0.440 |  |  | 0.533 (48) | 24/never/never | 0.289 | 0.310 | 0.409 |
| Qwen3-30B-A3B-Base, WinoGrande STR | greedy_mix | 0.223 | 0.622 | 0.800 | 0.931 [0.905, 0.960] |  | 0.931 (20) | 4/10/17 | 0.554 | 0.588 | 0.856 |
| Qwen3-30B-A3B-Base, WinoGrande STR | greedy_head | 0.145 | 0.413 | 0.518 | 0.580 [0.553, 0.606] |  | 0.580 (20) | 9/never/never | 0.362 | 0.382 | 0.856 |
| Qwen3-30B-A3B-Base, WinoGrande STR | greedy_expert_ext8 | 0.213 | 0.551 | 0.677 | 0.721 (k=15) |  | 0.721 (15) | 5/never/never | 0.486 |  | 0.856 |
| Qwen3-30B-A3B-Base, WinoGrande STR | static_mix_oracle | 0.226 | 0.533 | 0.647 |  | 0.898 | 0.898 (128) | 5/48/never | 0.476 | 0.499 | 0.856 |
| Qwen3-30B-A3B-Base, WinoGrande STR | static_mix_dla | 0.185 | 0.520 | 0.682 |  | 1.099 | 1.099 (128) | 5/24/32 | 0.465 | 0.498 | 0.856 |
| Qwen3-30B-A3B-Base, WinoGrande STR | static_head_oracle | 0.146 | 0.331 | 0.404 |  | 0.645 | 0.645 (128) | 32/never/never | 0.299 | 0.315 | 0.856 |
| Qwen3-30B-A3B-Base, WinoGrande STR | static_head_dla | 0.120 | 0.284 | 0.365 |  | 0.771 | 0.771 (128) | 32/never/never | 0.259 | 0.277 | 0.856 |
| Qwen3-30B-A3B-Base, WinoGrande STR | static_expert_oracle_ext8 | 0.214 | 0.469 | 0.553 |  | 0.781 | 0.846 (320) | 7/192/never | 0.422 | 0.439 | 0.856 |
| Qwen3-30B-A3B-Base, WinoGrande STR | static_expert_dla_ext8 | 0.174 | 0.464 | 0.606 |  | 0.884 | 0.888 (256) | 6/48/never | 0.420 | 0.447 | 0.856 |
| Mixtral-8x7B (BOS), WinoGrande STR | greedy_mix | 0.204 | 0.608 | 0.801 | 0.939 [0.926, 0.954] |  | 0.939 (20) | 4/10/16 | 0.541 | 0.577 | 0.788 |
| Mixtral-8x7B (BOS), WinoGrande STR | greedy_head | 0.120 | 0.420 | 0.591 | 0.652 [0.630, 0.674] |  | 0.652 (20) | 7/never/never | 0.369 | 0.395 | 0.788 |
| Mixtral-8x7B (BOS), WinoGrande STR | greedy_expert_ext8 | 0.192 | 0.557 | 0.718 | 0.791 (k=15) |  | 0.791 (15) | 4/never/never | 0.493 |  | 0.788 |
| Mixtral-8x7B (BOS), WinoGrande STR | static_mix_oracle | 0.205 | 0.571 | 0.730 |  | 0.955 | 0.955 (128) | 4/16/32 | 0.506 | 0.538 | 0.788 |
| Mixtral-8x7B (BOS), WinoGrande STR | static_mix_dla | 0.152 | 0.502 | 0.695 |  | 1.052 | 1.052 (128) | 5/16/32 | 0.449 | 0.485 | 0.788 |
| Mixtral-8x7B (BOS), WinoGrande STR | static_head_oracle | 0.121 | 0.374 | 0.519 |  | 0.750 | 0.750 (128) | 10/never/never | 0.334 | 0.358 | 0.788 |
| Mixtral-8x7B (BOS), WinoGrande STR | static_head_dla | 0.100 | 0.228 | 0.308 |  | 0.774 | 0.774 (128) | 32/never/never | 0.216 | 0.233 | 0.788 |
| Mixtral-8x7B (BOS), WinoGrande STR | static_expert_oracle_ext8 | 0.192 | 0.534 | 0.674 |  |  | 0.813 (48) | 5/48/never | 0.471 | 0.497 | 0.788 |
| Mixtral-8x7B (BOS), WinoGrande STR | static_expert_dla_ext8 | 0.147 | 0.472 | 0.647 |  |  | 0.821 (48) | 6/32/never | 0.422 | 0.454 | 0.788 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | greedy_mix | 0.184 | 0.561 | 0.784 | 0.943 [0.929, 0.957] |  | 0.943 (20) | 5/11/16 | 0.507 | 0.547 | -0.269 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | greedy_head | 0.184 | 0.560 | 0.784 | 0.935 [0.921, 0.950] |  | 0.935 (20) | 5/11/16 | 0.507 | 0.546 | -0.269 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | static_mix_oracle | 0.184 | 0.482 | 0.659 |  | 0.993 | 0.993 (128) | 6/16/32 | 0.445 | 0.481 | -0.269 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | static_mix_dla | 0.154 | 0.453 | 0.667 |  | 1.059 | 1.059 (128) | 6/16/24 | 0.425 | 0.465 | -0.269 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | static_head_oracle | 0.184 | 0.483 | 0.661 |  | 0.986 | 0.986 (128) | 6/16/32 | 0.446 | 0.482 | -0.269 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | static_head_dla | 0.154 | 0.453 | 0.669 |  | 1.049 | 1.049 (128) | 6/16/24 | 0.426 | 0.465 | -0.269 |


**Adaptive greedy: per-case thresholds, answer restored, composition, recurring components**

| Run | Greedy | case k50/80/90 (median; reached) | case k answer restored (median; reached) | cases restored k=1/5/10/20 | rows r ≥ 0.9 k=5/10/20 | rows top-1 at k=20 | heads among first 1/5/10/20 | most frequent in first 5 (share of rows) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base, CounterFact STR | mix | 3/8/12; 1.00/0.93/0.78 | 3; 0.94 | 0.09/0.73/0.92/0.94 | 0.10/0.42/0.72 | 0.29 | 0.5/3.3/6.2/10.4 | L40H13 0.77, L40H15 0.74, L40H14 0.60, L43H11 0.32, L44E069 0.22 |
| Qwen3-30B-A3B-Base, CounterFact STR | head | 4/11/never; 0.96/0.67/0.44 | 4; 0.88 | 0.05/0.69/0.81/0.87 | 0.08/0.30/0.42 | 0.23 | 1.0/5.0/10.0/20.0 | L40H15 0.90, L40H13 0.88, L40H14 0.83, L43H11 0.43, L43H15 0.33 |
| Mixtral-8x7B (BOS), CounterFact STR | mix | 3/8/13; 1.00/0.98/0.83 | 3; 0.99 | 0.06/0.83/0.96/0.99 | 0.09/0.40/0.81 | 0.43 | 0.7/3.4/6.4/11.1 | L24H22 0.67, L18H4 0.65, L19H29 0.47, L24H21 0.38, L21E001 0.19 |
| Mixtral-8x7B (BOS), CounterFact STR | head | 4/10/never; 0.99/0.74/0.45 | 4; 0.95 | 0.05/0.76/0.94/0.95 | 0.06/0.30/0.47 | 0.35 | 1.0/5.0/10.0/20.0 | L24H22 0.72, L18H4 0.67, L19H29 0.62, L24H21 0.44, L15H1 0.35 |
| Qwen3-30B-A3B-Base, WinoGrande STR | mix | 4/10/16; 1.00/0.82/0.58 | 4; 0.97 | 0.05/0.69/0.91/0.96 | 0.07/0.25/0.53 | 0.44 | 0.2/1.6/3.6/7.7 | L41E117 0.51, L46H24 0.40, L43E081 0.34, L41H27 0.29, L39E071 0.26 |
| Qwen3-30B-A3B-Base, WinoGrande STR | head | 8/never/never; 0.70/0.22/0.12 | 10; 0.66 | 0.02/0.29/0.50/0.61 | 0.01/0.04/0.09 | 0.19 | 1.0/5.0/10.0/20.0 | L46H24 0.61, L41H27 0.46, L38H18 0.32, L29H0 0.26, L38H21 0.22 |
| Mixtral-8x7B (BOS), WinoGrande STR | mix | 4/10/16; 1.00/0.96/0.75 | 4; 1.00 | 0.04/0.70/0.96/1.00 | 0.02/0.14/0.73 | 0.54 | 0.2/1.2/2.8/5.9 | L20E000 0.72, L19E006 0.71, L25H9 0.37, L21E006 0.30, L22H20 0.29 |
| Mixtral-8x7B (BOS), WinoGrande STR | head | 7/never/never; 0.80/0.31/0.12 | 7; 0.71 | 0.02/0.36/0.63/0.70 | 0.02/0.05/0.11 | 0.38 | 1.0/5.0/10.0/20.0 | L25H9 0.65, L22H20 0.62, L19H13 0.57, L13H18 0.36, L13H4 0.33 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | mix | 5/11/16; 1.00/0.97/0.74 | 5; 1.00 | 0.00/0.71/1.00/1.00 | 0.00/0.08/0.70 | 0.71 | 1.0/5.0/9.9/19.0 | L42H11 1.00, L43H24 0.71, L42H10 0.57, L40H2 0.50, L46H14 0.25 |
| Qwen3-30B-A3B-Base, IOI STR (S2 -> IO) | head | 5/11/16; 1.00/0.97/0.73 | 5; 1.00 | 0.00/0.71/1.00/1.00 | 0.00/0.08/0.68 | 0.71 | 1.0/5.0/10.0/20.0 | L42H11 1.00, L43H24 0.71, L42H10 0.58, L40H2 0.49, L46H14 0.25 |


![Joint add-back curves over heads + experts](figures/ext10_circuit_curves.png)

![Composition of the mixed greedy picks](figures/ext10_circuit_composition.png)

### Reading

**1. Full repair is reachable with few components.** Experts alone cannot exceed the all-MoE ceiling (Qwen3 CounterFact 0.53, Mixtral CounterFact 0.41, Qwen3 WinoGrande 0.86, Mixtral WinoGrande 0.79; ext8 greedy at k = 15: Qwen3 CounterFact 0.55, Mixtral CounterFact 0.56, Qwen3 WinoGrande 0.72, Mixtral WinoGrande 0.79). Admitting heads, adaptive greedy reaches Qwen3 CounterFact 0.88 / 0.97, Mixtral CounterFact 0.84 / 1.00, Qwen3 WinoGrande 0.80 / 0.93, Mixtral WinoGrande 0.80 / 0.94 of the drop at k = 10 / 20, i.e. 50 / 80 / 90 % of the drop with k = Qwen3 CounterFact 3/8/12, Mixtral CounterFact 4/9/13, Qwen3 WinoGrande 4/10/17, Mixtral WinoGrande 4/10/16 components (population curve; per case, median k for 80 %: Qwen3 CounterFact 8 (reached in 0.93), Mixtral CounterFact 8 (reached in 0.98), Qwen3 WinoGrande 10 (reached in 0.82), Mixtral WinoGrande 10 (reached in 0.96)). The answer itself (Δ > 0, donor mean) is restored at k = 20 in Qwen3 CounterFact 0.94, Mixtral CounterFact 0.99, Qwen3 WinoGrande 0.96, Mixtral WinoGrande 1.00 of cases (median k Qwen3 CounterFact 3, Mixtral CounterFact 3, Qwen3 WinoGrande 4, Mixtral WinoGrande 4), and r ≥ 0.9 at k = 20 holds for Qwen3 CounterFact 0.72, Mixtral CounterFact 0.81, Qwen3 WinoGrande 0.53, Mixtral WinoGrande 0.73 of rows; the true object is top-1 in Qwen3 CounterFact 0.29, Mixtral CounterFact 0.43, Qwen3 WinoGrande 0.44, Mixtral WinoGrande 0.54 of rows at k = 20 (clean top-1 rate Qwen3 CounterFact 0.28, Mixtral CounterFact 0.33, Qwen3 WinoGrande 0.39, Mixtral WinoGrande 0.50).

**2. Heads and experts are complements, not substitutes.** Head-only greedy (pool 32) reaches Qwen3 CounterFact 0.85, Mixtral CounterFact 0.87, Qwen3 WinoGrande 0.58, Mixtral WinoGrande 0.65 at k = 20, the mixed greedy Qwen3 CounterFact 0.97, Mixtral CounterFact 1.00, Qwen3 WinoGrande 0.93, Mixtral WinoGrande 0.94, experts alone (ext8, k = 15) Qwen3 CounterFact 0.55, Mixtral CounterFact 0.56, Qwen3 WinoGrande 0.72, Mixtral WinoGrande 0.79 (mixed at k = 15: Qwen3 CounterFact 0.94, Mixtral CounterFact 0.95, Qwen3 WinoGrande 0.89, Mixtral WinoGrande 0.89). Among the first 5 / 10 / 20 mixed picks the mean number of heads is Qwen3 CounterFact 3.3/6.2/10.4, Mixtral CounterFact 3.4/6.4/11.1, Qwen3 WinoGrande 1.6/3.6/7.7, Mixtral WinoGrande 1.2/2.8/5.9; the first pick is a head in Qwen3 CounterFact 0.53, Mixtral CounterFact 0.75, Qwen3 WinoGrande 0.22, Mixtral WinoGrande 0.16 of rows. Median layer of the heads / experts among the first 10 picks: Qwen3 CounterFact L40 / L42, Mixtral CounterFact L19 / L21, Qwen3 WinoGrande L38 / L39, Mixtral WinoGrande L22 / L20. The order differs by task: on CounterFact the greedy mostly starts with the mover heads and adds experts later (heads are about half of the first 10-20 picks), on WinoGrande it starts with the W6 experts and adds a few late heads (roughly a third of the picks); the median marginal gain of a head / expert step (mixed greedy, steps 2-20, fraction of the drop) is Qwen3 CounterFact 0.036 / 0.017, Mixtral CounterFact 0.034 / 0.026, Qwen3 WinoGrande 0.033 / 0.032, Mixtral WinoGrande 0.026 / 0.026. Neither component class alone gets there: the head-only curve flattens below the drop (most clearly on WinoGrande, where the final-position attention writes only ~5-30 % of the logit difference directly), the expert-only curve below the all-MoE ceiling.

**3. Which components recur.** Most frequent in the first five mixed-greedy picks (share of rows): Qwen3 CounterFact: L40H13 0.77, L40H15 0.74, L40H14 0.60, L43H11 0.32, L44E069 0.22, L43H15 0.22; Mixtral CounterFact: L24H22 0.67, L18H4 0.65, L19H29 0.47, L24H21 0.38, L21E001 0.19, L15H1 0.18; Qwen3 WinoGrande: L41E117 0.51, L46H24 0.40, L43E081 0.34, L41H27 0.29, L39E071 0.26, L34E119 0.14; Mixtral WinoGrande: L20E000 0.72, L19E006 0.71, L25H9 0.37, L21E006 0.30, L22H20 0.29, L16E007 0.29. The recurring components are the loci of earlier directions: the CounterFact mover heads of F2 (Qwen3 L40H13 / H14 / H15, L43H11; Mixtral L18H4, L24H22, L19H29) with the CounterFact experts (Qwen3 L44E069, Mixtral L21E001), and the WinoGrande experts of W6 / ext8 (Qwen3 L41E117, L43E081, L39E071; Mixtral L20E000, L19E006, L21E006) with the late WinoGrande heads found in Step 1 (Qwen3 L46H24, L41H27; Mixtral L25H9, L22H20).

**4. Greedy vs static and patch-free orderings.** AUC of r(k) over log k (k = 1..20; static orderings interpolated) mixed greedy / head-only greedy / static oracle (heads ∪ experts) / static DLA (heads ∪ experts) / static head oracle / static head DLA: Qwen3 CounterFact 0.64 / 0.58 / 0.57 / 0.57 / 0.54 / 0.54; Mixtral CounterFact 0.61 / 0.56 / 0.53 / 0.52 / 0.52 / 0.50; Qwen3 WinoGrande 0.59 / 0.38 / 0.50 / 0.50 / 0.31 / 0.28; Mixtral WinoGrande 0.58 / 0.40 / 0.54 / 0.49 / 0.36 / 0.23. Over k = 1..15, against ext8's expert-only greedy / static expert oracle / static expert DLA: Qwen3 CounterFact mixed greedy 0.61 vs 0.39 / 0.37 / 0.37; Mixtral CounterFact mixed greedy 0.57 vs 0.35 / 0.31 / 0.29; Qwen3 WinoGrande mixed greedy 0.55 vs 0.49 / 0.42 / 0.42; Mixtral WinoGrande mixed greedy 0.54 vs 0.49 / 0.47 / 0.42. Largest k on the static grid: r(128) oracle / DLA Qwen3 CounterFact 0.95 / 1.11, Mixtral CounterFact 0.96 / 1.09, Qwen3 WinoGrande 0.90 / 1.10, Mixtral WinoGrande 0.96 / 1.05. Adaptivity is worth 0.04-0.09 of AUC over the best static ordering; the patch-free DLA ordering over heads ∪ experts is as good as the single-patch oracle on CounterFact and Qwen3 WinoGrande and slightly worse on Mixtral WinoGrande, and it overshoots the drop at large k (r(128) > 1: clean components that lower LD are left out), whereas the oracle saturates below 1. For heads alone the DLA is a poor guide on WinoGrande (the decisive heads act indirectly, Reading 7).

**5. IOI (attention pole).** Qwen3 IOI mixed greedy r(10) / r(20) 0.78 / 0.94, head-only 0.93, heads among the first 10 picks 9.9, all-MoE -0.27; first-5 picks L42H11 1.00, L43H24 0.71, L42H10 0.57, L40H2 0.50. Expert orderings / pools on IOI use the expert DLA (no single-expert patch table exists for IOI).

**6. Single heads.** The best single head per row is as strong as the best single expert on CounterFact and weaker on WinoGrande (median best head / best expert Qwen3 CounterFact 0.19 / 0.19, Mixtral CounterFact 0.21 / 0.12, Qwen3 WinoGrande 0.15 / 0.20, Mixtral WinoGrande 0.11 / 0.19); a row's 32 best single components contain Qwen3 CounterFact 14, Mixtral CounterFact 17, Qwen3 WinoGrande 12, Mixtral WinoGrande 12 heads (median). The top heads are those of earlier directions: CounterFact STR recovers the F2 GN heads (Qwen3 L40H13, L43H11; Mixtral L18H4, L24H22, L19H29, L15H1), WinoGrande recovers every W5 head (Qwen3 L38H18 / L38H21 positive, L38H16 negative; Mixtral L25H9, L19H13, L13H18 / H11 / H4) and adds heads in layers W5 did not scan (Qwen3 WinoGrande L46H24 +0.10, L41H27 +0.07, L29H0 +0.04, L41H0 +0.02, L46H25 -0.11, Mixtral WinoGrande L22H20 +0.07, L22H23 -0.03). Opposing heads of one KV group cancel inside a layer (Qwen3 WinoGrande L46H24 / L46H25), so the layer patch hides them.

**7. The patch-free head DLA.** The head DLA ranks heads like the single-head patch at the population level (r over heads Qwen3 CounterFact 0.96, Mixtral CounterFact 0.92, Qwen3 WinoGrande 0.87, Mixtral WinoGrande 0.79, Qwen3 IOI 0.76; per row r Qwen3 CounterFact 0.63, Mixtral CounterFact 0.76, Qwen3 WinoGrande 0.33, Mixtral WinoGrande 0.41, Qwen3 IOI 0.62, top-1 agreement Qwen3 CounterFact 0.71, Mixtral CounterFact 0.44, Qwen3 WinoGrande 0.41, Mixtral WinoGrande 0.47, Qwen3 IOI 0.58, top-10 overlap Qwen3 CounterFact 0.65, Mixtral CounterFact 0.70, Qwen3 WinoGrande 0.28, Mixtral WinoGrande 0.39, Qwen3 IOI 0.62). The direct share of the top heads differs by task: CounterFact mover heads act partly through the downstream MoE (Qwen3 L40H13 single 0.19 vs DLA 0.12; Mixtral L18H4 0.15 vs 0.06), the WinoGrande heads of the early hand-off layers act almost entirely indirectly (Mixtral L19H13, L13 heads: DLA ≈ 0), late heads write directly (Mixtral L24H22, L25H9, Qwen3 L46H24: DLA ≈ single). Σ head DLA over all heads and layers equals ext8's linear attention DLA (ratio Qwen3 CounterFact 0.999, Mixtral CounterFact 1.000, Qwen3 WinoGrande 1.029, Mixtral WinoGrande 0.998, Qwen3 IOI 1.003).

### Caveats

- Final position only, denoising (sufficiency) only; components at the subject / option positions are not candidates. The ceiling 1 is reached by construction once all heads at all layers are patched (the corruption reaches the final position only through attention), so 'full repair' means: which small subset of final-position heads and experts suffices.
- bf16: single-head rescues are quantised at the logit spacing (one step = 0.125 at |logit| 16-32); identical single-head patches in different passes differ by a median of one step (this run vs the W5 head rows of ext7: Qwen3 WinoGrande r 0.64, median |diff| 0.125, Mixtral WinoGrande r 0.65, median |diff| 0.000, Qwen3 IOI r 0.91, median |diff| 0.125). Sums over the 32 heads of a layer therefore carry ~0.5 logit of noise per row (row-level r(Σ heads, attn_layer) Qwen3 CounterFact 0.37, Mixtral CounterFact 0.45, Qwen3 WinoGrande 0.39, Mixtral WinoGrande 0.31, Qwen3 IOI 0.41), while population means agree with the layer patch.
- All heads at all layers as one multi patch reproduce the clean Δ in the population (ratio Qwen3 CounterFact 0.999, Mixtral CounterFact 1.000, Qwen3 WinoGrande 1.006, Mixtral WinoGrande 1.001, Qwen3 IOI 0.998) and per row up to bf16 noise (median / max |Δ − Δ_clean| Qwen3 CounterFact 0.125 / 1.88, Mixtral CounterFact 0.000 / 0.69, Qwen3 WinoGrande 0.250 / 2.62, Mixtral WinoGrande 0.000 / 0.25, Qwen3 IOI 0.125 / 0.50; the same-pass all-attention patch (attn_layer steps) shows the same envelope: o_proj rounding at every layer plus batch-composition noise amplified by routing near-ties); all heads of one layer vs the attn_layer kind: max |ΔΔ| Qwen3 CounterFact 0.625, Mixtral CounterFact 0.688, Qwen3 WinoGrande 2.000, Mixtral WinoGrande 0.375, Qwen3 IOI 0.500. Prefill Δ of identical rows across passes: SD median Qwen3 CounterFact 0.09, Mixtral CounterFact 0.06, Qwen3 WinoGrande 0.18, Mixtral WinoGrande 0.06, Qwen3 IOI 0.16 logits. Greedy steps span passes (each step compares its candidates within one pass).
- Greedy pools and the oracle ordering use the row's own single-component patches (in-sample, as ext8); the pools are truncated (32 heads; Qwen3 32 experts, Mixtral all 64), so components outside a row's top-32 single heads can enter only through the static orderings (which use all heads and all clean-active experts).
- The ext8 comparison curves are ext8's own passes on the same rows (its drop reference); static ext8 curves come from results/tables/ext8_a1_curves.csv. Differences below ~0.02 of the drop are within bf16 pass-to-pass noise.
- Engine: Step 1 used the attn_head kind (unchanged by ext9; regression identical); Step 2 uses the ext9 multi steps attn_head / heads_experts, verified on OLMoE against transformers hooks (results/verify_ext9_engine_olmoe.json: mixed head + expert sets effect r 0.998, max |ΔΔ| 0.375; all heads at all layers r 0.9996; one head step = attn_head kind bit for bit).
- CounterFact rows are (case, donor) pairs (donor mean per case for population values); WinoGrande / IOI rows are directed cases with bootstrap over pairs. First-donor rows only (sensitivity), mixed greedy r(10) / r(20): Qwen3 CounterFact 0.86 / 0.96, Mixtral CounterFact 0.84 / 1.00.
- IOI was run for Qwen3 only (its rows share the Qwen3 Step-2 passes at little extra cost); Mixtral IOI would have needed ~25 extra GPU minutes of separate passes and was left out of the budget.

### Files

- Code: `moetrace/ext10_circuit.py` (run configs, task loading via `moetrace.ext8_addback`, IOI loader, step-2 candidates / pools / greedy state), `scripts/ext10_heads_run.py` (Step 1 driver: single heads + head / expert DLA, resumable, raw dump and CPU post-processing fallback, OOM chunk splitting), `scripts/ext10_heads_analyze.py`, `scripts/ext10_circuit_run.py` (Step 2 driver, both tasks of a model in every pass), `scripts/ext10_circuit_analyze.py`, `scripts/ext10_circuit_text.py`, chains `scripts/ext10_chain1.sh`, `scripts/ext10_chain2.sh`.
- Step 1 runs: `results/{qwen3_str,mixtral_bos_str,wino_qwen3_str,wino_mixtral_bos_str,ioi_qwen3_s2io}_circuit/` (`head_rows_pNN.parquet` single-head + attn_layer rows, `head_dla_pNN.parquet`, `expert_dla_pNN.parquet`, `head_prefill_pNN.parquet`, `heads_state.json`, `run_meta.json`).
- Step 2 runs: `results/{qwen3,mixtral}_circuit/` (`circuit_rows_pNN.parquet` spawn rows with task / fam / order / k / set / Δ / metrics, `circuit_prefill_pNN.parquet`, `circuit_state.pkl` = greedy paths, `run_meta.json`); smoke `results/qwen3_circuit_smoke/`.
- Tables `results/tables/ext10_heads_*`, `results/tables/ext10_circuit_*`; figures `results/figures/ext10_heads_*`, `results/figures/ext10_circuit_*`; numbers `results/ext10_circuit_summary.json` (keys step1, step2). GPU time of ext10 jobs ≈ 79 min.
