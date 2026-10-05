**Summary.** WinoGrande twins become symmetric-token-replacement (STR) pairs once the blank is filled with each twin's own answer and the model predicts the sentence-final trigger word ("… but the bag was too" → " small" / "… but the body was too" → " large"): the two prompts differ only in the filled option, each is an ordinary WinoGrande sentence with its own answer, and Δ = logit(r) − logit(r′) is Zhang & Nanda's logit difference. On 256 pairs that Qwen3 and Mixtral both solve with a 1-logit margin in both directions (128 discovery / 128 validation pairs, each used both ways; bootstrap over pairs; a disjoint 256-pair replication set), the answer to "MoE or attention?" is the opposite of the IOI hypothesis at the final position. Single-layer attention-output patches there carry at most 0.030 (Qwen3, L42) and 0.141 (Mixtral, L13) of the drop, the attention share of the positive layer-wise rescue is 0.16 [0.13, 0.20] and 0.34 [0.33, 0.36] — against 0.54 and 0.58 on CounterFact STR under the same definition (ext7-controls' sweep) — and the MoE-output peak per unit of drop is larger than on CounterFact (Qwen3 L41 0.218 [0.200, 0.238] vs 0.177 at L44; Mixtral L20 0.172 [0.162, 0.183] vs 0.086). Patching every final-position MoE output (W4) restores 0.845 [0.827, 0.861] (Qwen3) and 0.788 [0.765, 0.810] (Mixtral) of the drop while attention still reads the corrupted option, and on the direct paths to the logit difference the MoE outputs write 0.953 [0.925, 0.982] / 0.712 [0.687, 0.735] of the drop and the attention outputs 0.047 [0.018, 0.075] / 0.288 [0.265, 0.313] (validation). The position × layer grid explains the small attention patches: the option's identity reaches the final token directly from the option position (the intermediate token carries ≤ 0.12) and gradually, the final-position residual restoring 0.1 / 0.5 / 0.9 of the drop at L19 / L30 / L41 in Qwen3, so no single attention layer is a bottleneck. The MoE side localises to **one specific expert per model** — Qwen3 **L41E117** (validation rescue +1.002 [+0.844, +1.171], Spec +0.903 [+0.744, +1.071], gate-matched equal-norm Spec +0.450 [+0.373, +0.534]) and Mixtral **L20E000** (rescue +1.106 [+1.013, +1.203], Spec +0.940 [+0.836, +1.042], equal-norm Spec +0.160 [+0.132, +0.189]), pattern A, both re-selected on the replication set — and these are not the CounterFact STR selections (Qwen3 L44E069 / L42E115, Mixtral L19E002 / L21E001 / L18E001), which are routed at the WinoGrande final position in only 3–12 of 256 directed cases and rescue nothing.

### What was run

Pairs (W0, `moetrace/ext7_wino.py`, `scripts/ext7_wino_build.py`, funnel `results/tables/ext7_wino_funnel.md`): WinoGrande 1.1 train_xl twins whose two sentences differ only in their last word, with a single-token trigger after both prompts, token symmetry (equal length, the option at the same positions, all other tokens identical), the option not the final token, one pair per normalised context. Case set (decision (g), `data/wino_str/case_sets.json`): the 776 pairs that pass the STR margin (Δ_A ≥ 1, Δ_B ≤ −1) under Qwen3, Mixtral BOS and Mixtral no BOS, seed-0 shuffle → **main** 128 discovery + 128 validation pairs, **replication** 128 + 128 pairs, and per model a seed-1 sample of its own margin pool (**own pool**, 128 + 128, W2 only). Every pair is run in both directions (directed case 2·pair + d: d = 0 clean A / corrupted B / r = trigger of A; d = 1 the reverse); per-pair value = mean of the two directions; CIs = 5,000 pair-bootstrap resamples; expert recurrence and Spec on directed cases (gate 128 of 256 discovery directed cases). Models: Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1 with BOS (tokenizer defaults, decision (e)); bf16. Runner: `moetrace/ext7_pairs.py` + `scripts/ext7_wino_{sweep,expert,grid,heads,joint,dla}.py` (generic STR-pair runners, also used by ext7-controls for role swaps and IOI).

**WinoGrande STR pair sets (directed cases pooled; drop = Δ_clean − Δ_corrupt, pair means)**

| Model | Pair set | Pairs | Mean Δ clean | Mean Δ corrupt | Drop [95% CI] | Clean top-1 = r | Corrupt top-1 = r′ | Names | assoc | top-1 both | debiased | One-token option | Median T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | 256 | +4.16 | -4.16 | +8.31 [+7.97, +8.66] | 0.40 | 0.39 | 0.09 | 0.12 | 0.25 | 0.20 | 0.96 | 18 |
| Qwen3-30B-A3B-Base | rep | 256 | +4.19 | -4.21 | +8.39 [+8.07, +8.74] | 0.41 | 0.41 | 0.07 | 0.11 | 0.24 | 0.18 | 0.96 | 18 |
| Qwen3-30B-A3B-Base | own | 256 | +3.84 | -3.86 | +7.69 [+7.35, +8.04] | 0.27 | 0.27 | 0.25 | 0.07 | 0.14 | 0.15 | 0.97 | 18 |
| Mixtral-8x7B, BOS | main | 256 | +3.89 | -3.89 | +7.78 [+7.47, +8.09] | 0.48 | 0.48 | 0.09 | 0.10 | 0.33 | 0.20 | 0.92 | 20 |
| Mixtral-8x7B, BOS | rep | 256 | +4.09 | -4.09 | +8.17 [+7.86, +8.48] | 0.44 | 0.45 | 0.07 | 0.15 | 0.29 | 0.18 | 0.93 | 19 |
| Mixtral-8x7B, BOS | own | 256 | +3.63 | -3.64 | +7.27 [+6.94, +7.61] | 0.37 | 0.37 | 0.14 | 0.11 | 0.23 | 0.25 | 0.89 | 20 |

In 93 % of the directed cases the option is one or two tokens before the prediction position ("the bag was too" / "the bag was"; final word "too" in 48 %), and the clean prompt's top-1 is the trigger in 40% (Qwen3) and 48% (Mixtral) of the directed cases — margins are relative to the twin's trigger, not top-1 accuracy (top-1 in both directions is a sensitivity stratum).

Verification (W1, `scripts/ext7_wino_verify.py` → `results/verify_ext7_wino_olmoe.json`; OLMoE, 20 margin pairs = 40 directed cases, transformers hooks): final-position patches at every layer, rescue r = 0.995 (MoE), 0.989 (attention), 0.997 (block), 0.999 (residual), mean |ΔΔ| ≤ 0.071; per-head patches r = 0.905 (small effects, SD 0.15, mean |ΔΔ| 0.063); STR-position grid (suffix executor, every position from the option to the final token) r = 0.993 / 0.992 / 0.996 / 1.000 (MoE / attention / block / residual), window 5 r = 0.998; null invariant 0.156; the two directions of a pair are exact mirrors (antisymmetry 0.0). The first run of this check found that `moetrace/ext5_subject.py` (`run_subject`) turns `attn_layer` rows into `block` rows when the same pass contains a window > 1 row; production passes never mix them (the W3 windows run in separate passes, as in Direction 6b). GPU time of all W1–W6 passes on the big models: ≈ 53 min.

### W2. Final-position layer sweep: MoE, attention, block

**W2: final-position single-layer patches under STR (pairs; discovery argmax, validation value with pair-bootstrap CI)**

| Model | Pair set | Patched output | L* (discovery) | Discovery mean at L* | Validation rescue at L* [95% CI] | Normalised (rescue / drop) | Validation argmax | Discovery top 4 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | MoE output | L41 | +1.75 (gap +0.38 to L43) | +1.778 [+1.594, +1.972] | 0.218 [0.200, 0.238] | L41 +1.778 | L41 +1.75, L43 +1.37, L39 +1.25, L44 +0.80 |
| Qwen3-30B-A3B-Base | main | attention output | L38 | +0.33 (gap +0.04 to L42) | +0.076 [-0.042, +0.199] | 0.009 [-0.005, 0.025] | L42 +0.242 | L38 +0.33, L42 +0.29, L40 +0.24, L39 +0.22 |
| Qwen3-30B-A3B-Base | main | attention + MoE (block) | L41 | +1.93 (gap +0.42 to L43) | +1.940 [+1.753, +2.132] | 0.238 [0.219, 0.258] | L41 +1.940 | L41 +1.93, L43 +1.51, L39 +1.47, L42 +0.99 |
| Qwen3-30B-A3B-Base | replication | MoE output | L41 | +1.59 (gap +0.10 to L43) | +1.715 [+1.526, +1.911] | 0.200 [0.180, 0.219] | L41 +1.715 | L41 +1.59, L43 +1.49, L39 +1.21, L44 +0.83 |
| Qwen3-30B-A3B-Base | replication | attention output | L42 | +0.33 (gap +0.13 to L38) | +0.293 [+0.229, +0.360] | 0.034 [0.027, 0.042] | L42 +0.293 | L42 +0.33, L38 +0.20, L41 +0.19, L39 +0.17 |
| Qwen3-30B-A3B-Base | replication | attention + MoE (block) | L41 | +1.76 (gap +0.16 to L43) | +1.963 [+1.764, +2.177] | 0.228 [0.209, 0.248] | L41 +1.963 | L41 +1.76, L43 +1.60, L39 +1.37, L42 +0.98 |
| Qwen3-30B-A3B-Base | own pool | MoE output | L41 | +1.26 (gap +0.01 to L43) | +1.462 [+1.241, +1.695] | 0.184 [0.161, 0.207] | L41 +1.462 | L41 +1.26, L43 +1.25, L39 +0.97, L42 +0.91 |
| Qwen3-30B-A3B-Base | own pool | attention output | L38 | +0.37 (gap +0.01 to L41) | +0.397 [+0.248, +0.553] | 0.050 [0.031, 0.071] | L38 +0.397 | L38 +0.37, L41 +0.37, L42 +0.31, L40 +0.28 |
| Qwen3-30B-A3B-Base | own pool | attention + MoE (block) | L41 | +1.57 (gap +0.15 to L43) | +1.610 [+1.383, +1.842] | 0.203 [0.180, 0.225] | L41 +1.610 | L41 +1.57, L43 +1.43, L42 +1.20, L39 +1.09 |
| Qwen3-30B-A3B-Base | main, L ≤ L−5 | MoE output | L41 | +1.75 (gap +0.38 to L43) | +1.778 [+1.594, +1.972] | 0.218 [0.200, 0.238] | L41 +1.778 | L41 +1.75, L43 +1.37, L39 +1.25, L42 +0.73 |
| Qwen3-30B-A3B-Base | main, L ≤ L−5 | attention output | L38 | +0.33 (gap +0.04 to L42) | +0.076 [-0.042, +0.199] | 0.009 [-0.005, 0.025] | L42 +0.242 | L38 +0.33, L42 +0.29, L40 +0.24, L39 +0.22 |
| Qwen3-30B-A3B-Base | main, L ≤ L−5 | attention + MoE (block) | L41 | +1.93 (gap +0.42 to L43) | +1.940 [+1.753, +2.132] | 0.238 [0.219, 0.258] | L41 +1.940 | L41 +1.93, L43 +1.51, L39 +1.47, L42 +0.99 |
| Mixtral-8x7B, BOS | main | MoE output | L20 | +1.25 (gap +0.04 to L21) | +1.324 [+1.222, +1.429] | 0.172 [0.162, 0.183] | L20 +1.324 | L20 +1.25, L21 +1.21, L19 +1.18, L22 +0.84 |
| Mixtral-8x7B, BOS | main | attention output | L13 | +1.19 (gap +0.35 to L19) | +1.087 [+0.971, +1.209] | 0.141 [0.127, 0.157] | L13 +1.087 | L13 +1.20, L19 +0.84, L25 +0.57, L15 +0.56 |
| Mixtral-8x7B, BOS | main | attention + MoE (block) | L19 | +1.93 (gap +0.42 to L20) | +1.927 [+1.801, +2.058] | 0.251 [0.238, 0.264] | L19 +1.927 | L19 +1.93, L20 +1.51, L13 +1.49, L21 +1.25 |
| Mixtral-8x7B, BOS | replication | MoE output | L20 | +1.23 (gap +0.08 to L19) | +1.396 [+1.296, +1.502] | 0.165 [0.155, 0.175] | L20 +1.396 | L20 +1.23, L19 +1.16, L21 +1.09, L26 +0.87 |
| Mixtral-8x7B, BOS | replication | attention output | L13 | +1.10 (gap +0.32 to L19) | +1.132 [+1.004, +1.268] | 0.134 [0.119, 0.149] | L13 +1.132 | L13 +1.10, L19 +0.78, L25 +0.58, L15 +0.48 |
| Mixtral-8x7B, BOS | replication | attention + MoE (block) | L19 | +1.86 (gap +0.40 to L20) | +2.118 [+1.979, +2.269] | 0.250 [0.236, 0.264] | L19 +2.118 | L19 +1.86, L20 +1.46, L13 +1.39, L25 +1.21 |
| Mixtral-8x7B, BOS | own pool | MoE output | L20 | +1.15 (gap +0.03 to L21) | +1.163 [+1.059, +1.268] | 0.157 [0.146, 0.168] | L20 +1.163 | L20 +1.15, L21 +1.11, L19 +1.07, L26 +0.74 |
| Mixtral-8x7B, BOS | own pool | attention output | L13 | +1.01 (gap +0.24 to L19) | +0.972 [+0.837, +1.106] | 0.131 [0.114, 0.149] | L13 +0.972 | L13 +1.01, L19 +0.76, L25 +0.51, L15 +0.47 |
| Mixtral-8x7B, BOS | own pool | attention + MoE (block) | L19 | +1.76 (gap +0.41 to L20) | +1.708 [+1.576, +1.843] | 0.231 [0.216, 0.245] | L19 +1.708 | L19 +1.76, L20 +1.36, L13 +1.21, L21 +1.15 |
| Mixtral-8x7B, BOS | main, L ≤ L−5 | MoE output | L20 | +1.25 (gap +0.04 to L21) | +1.324 [+1.222, +1.429] | 0.172 [0.162, 0.183] | L20 +1.324 | L20 +1.25, L21 +1.21, L19 +1.18, L22 +0.84 |
| Mixtral-8x7B, BOS | main, L ≤ L−5 | attention output | L13 | +1.19 (gap +0.35 to L19) | +1.087 [+0.971, +1.209] | 0.141 [0.127, 0.157] | L13 +1.087 | L13 +1.20, L19 +0.84, L25 +0.57, L15 +0.56 |
| Mixtral-8x7B, BOS | main, L ≤ L−5 | attention + MoE (block) | L19 | +1.93 (gap +0.42 to L20) | +1.927 [+1.801, +2.058] | 0.251 [0.238, 0.264] | L19 +1.927 | L19 +1.93, L20 +1.51, L13 +1.49, L21 +1.25 |

**W2: attention share of the positive final-position rescue (Direction-2b definition: AUC+(attn) / (AUC+(attn) + AUC+(MoE)), sums over layers of the positive part of the mean curve)**

| Model | Set | Pairs | AUC+ attention | AUC+ MoE | AUC+ block | AUC+ attention / drop | AUC+ MoE / drop | Attention share [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 128 | 1.86 | 9.81 | 11.34 | 0.228 | 1.205 | 0.16 [0.13, 0.20] |
| Qwen3-30B-A3B-Base | main, all 256 pairs | 256 | 2.07 | 9.79 | 11.57 | 0.249 | 1.177 | 0.17 [0.15, 0.20] |
| Qwen3-30B-A3B-Base | replication, validation | 128 | 2.60 | 11.20 | 13.15 | 0.303 | 1.302 | 0.19 [0.17, 0.22] |
| Qwen3-30B-A3B-Base | own pool, validation | 128 | 2.84 | 9.28 | 11.51 | 0.358 | 1.169 | 0.23 [0.20, 0.27] |
| Qwen3-30B-A3B-Base | main val., A→B only | 128 | 1.90 | 10.53 | 11.96 | 0.233 | 1.294 | 0.15 [0.13, 0.21] |
| Qwen3-30B-A3B-Base | main val., B→A only | 128 | 1.94 | 9.26 | 10.88 | 0.238 | 1.137 | 0.17 [0.14, 0.23] |
| Qwen3-30B-A3B-Base | CounterFact STR, validation (ext7-controls' qwen3_str_attnsweep, donor mean) | 108 |  |  |  | 0.800 | 0.670 | 0.54 |
| Mixtral-8x7B, BOS | main, validation | 128 | 5.59 | 10.68 | 15.56 | 0.728 | 1.389 | 0.34 [0.33, 0.36] |
| Mixtral-8x7B, BOS | main, all 256 pairs | 256 | 5.68 | 10.55 | 15.63 | 0.730 | 1.356 | 0.35 [0.34, 0.36] |
| Mixtral-8x7B, BOS | replication, validation | 128 | 6.22 | 11.57 | 16.91 | 0.735 | 1.366 | 0.35 [0.34, 0.36] |
| Mixtral-8x7B, BOS | own pool, validation | 128 | 5.78 | 10.01 | 14.83 | 0.780 | 1.352 | 0.37 [0.35, 0.38] |
| Mixtral-8x7B, BOS | main val., A→B only | 128 | 5.78 | 10.85 | 15.78 | 0.751 | 1.411 | 0.35 [0.33, 0.36] |
| Mixtral-8x7B, BOS | main val., B→A only | 128 | 5.43 | 10.55 | 15.35 | 0.707 | 1.373 | 0.34 [0.32, 0.36] |
| Mixtral-8x7B, BOS | CounterFact STR, validation (ext7-controls' mixtral_bos_str_attnsweep, donor mean) | 106 |  |  |  | 0.842 | 0.615 | 0.58 |

**W2: attention vs MoE at the peak layers (validation pairs) and block additivity at the largest block layers**

| Model | Layer | Attention output rescue | MoE output rescue | Attention share at the layer / additivity |
|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41 (MoE output peak) | +0.163 [+0.092, +0.235] | +1.778 [+1.594, +1.972] | 0.08 [0.05, 0.12] |
| Qwen3-30B-A3B-Base | L38 (attention output peak) | +0.076 [-0.042, +0.199] | +0.739 [+0.627, +0.848] | 0.09 [-0.06, 0.21] |
| Qwen3-30B-A3B-Base | L41 (block +1.94) | +0.163 | +1.778 | block − (attn + MoE) -0.001 [-0.032, +0.030], r = 0.99 |
| Qwen3-30B-A3B-Base | L43 (block +1.53) | +0.188 | +1.375 | block − (attn + MoE) -0.038 [-0.060, -0.015], r = 0.99 |
| Qwen3-30B-A3B-Base | L39 (block +1.45) | +0.148 | +1.316 | block − (attn + MoE) -0.009 [-0.048, +0.030], r = 0.98 |
| Mixtral-8x7B, BOS | L20 (MoE output peak) | +0.222 [+0.183, +0.262] | +1.324 [+1.222, +1.429] | 0.14 [0.12, 0.16] |
| Mixtral-8x7B, BOS | L13 (attention output peak) | +1.087 [+0.971, +1.209] | +0.276 [+0.188, +0.374] | 0.80 [0.75, 0.85] |
| Mixtral-8x7B, BOS | L19 (block +1.93) | +0.873 | +1.120 | block − (attn + MoE) -0.066 [-0.103, -0.029], r = 0.96 |
| Mixtral-8x7B, BOS | L20 (block +1.53) | +0.222 | +1.324 | block − (attn + MoE) -0.013 [-0.034, +0.008], r = 0.99 |
| Mixtral-8x7B, BOS | L13 (block +1.25) | +1.087 | +0.276 | block − (attn + MoE) -0.110 [-0.158, -0.062], r = 0.97 |

![W2 curves](figures/ext7_wino_w2_curves.png)

- **Qwen3.** The MoE output of L41 restores +1.778 [+1.594, +1.972] logits = 0.218 [0.200, 0.238] of the drop (CounterFact STR: L44, 0.177); the band L39–L44 carries the effect (discovery top: L41 +1.75, L43 +1.37, L39 +1.25, L44 +0.80). No attention layer matters on its own (largest validation value 0.030 [0.022, 0.037] of the drop at L42; the discovery argmax L38 does not replicate on validation). Block ≈ attention + MoE at every layer (largest gap -0.038).
- **Mixtral.** MoE L20 0.172 [0.162, 0.183] (flat L19–L21 band as on CounterFact, where the same patch gives 0.085); attention has two discrete steps, L13 0.141 [0.127, 0.157] and L19 (validation 0.114), and block L19 0.251 [0.238, 0.264]. At L13 the attention output carries 0.80 of the layer's attention + MoE rescue, at L20 only 0.14.
- **Attention share** (Direction-2b AUC+ definition) 0.16 [0.13, 0.20] / 0.34 [0.33, 0.36] on main validation, 0.19 [0.17, 0.22] / 0.35 [0.34, 0.36] on the replication set, 0.23 [0.20, 0.27] / 0.37 [0.35, 0.38] on each model's own margin pool, the same in each single direction (A→B 0.15 [0.13, 0.21] / 0.35 [0.33, 0.36]); CounterFact STR 0.54 / 0.58. WinoGrande is less attention-dominated than CounterFact at the final position in both models, Qwen3 most clearly.

### W3. Position × layer grid: where the option's identity travels

**W3: position × layer grid under STR (main set, 256 pairs; positions before the option are exactly zero)**

| Model | Window | Patched quantity at p | Token group | Pairs | Peak layer | Peak rescue / drop [95% CI] | Sum over layers | Peak Δp (descriptive) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 1 | attention output | first option tok. | 9 | L4 | +0.020 [-0.001, +0.042] | -0.027 | +0.0001 (L1) |
| Qwen3-30B-A3B-Base | 1 | attention output | last option tok. | 256 | L34 | +0.013 [+0.009, +0.016] | +0.054 | +0.0013 (L3) |
| Qwen3-30B-A3B-Base | 1 | attention output | first subseq. | 157 | L26 | +0.005 [+0.001, +0.011] | +0.033 | +0.0004 (L26) |
| Qwen3-30B-A3B-Base | 1 | attention output | further | 18 | L38 | +0.009 [+0.003, +0.017] | -0.069 | +0.0013 (L39) |
| Qwen3-30B-A3B-Base | 1 | attention output | final token | 256 | L42 | +0.032 [+0.027, +0.037] | +0.213 | +0.0020 (L39) |
| Qwen3-30B-A3B-Base | 1 | MoE output | first option tok. | 9 | L0 | +0.052 [-0.071, +0.160] | +0.106 | +0.0024 (L0) |
| Qwen3-30B-A3B-Base | 1 | MoE output | last option tok. | 256 | L0 | +0.171 [+0.141, +0.204] | +0.396 | +0.0372 (L0) |
| Qwen3-30B-A3B-Base | 1 | MoE output | first subseq. | 157 | L39 | +0.024 [+0.019, +0.028] | +0.151 | +0.0016 (L39) |
| Qwen3-30B-A3B-Base | 1 | MoE output | further | 18 | L38 | +0.008 [+0.004, +0.012] | -0.094 | +0.0009 (L14) |
| Qwen3-30B-A3B-Base | 1 | MoE output | final token | 256 | L41 | +0.214 [+0.201, +0.227] | +1.050 | +0.0234 (L41) |
| Qwen3-30B-A3B-Base | 1 | residual (hidden state) | first option tok. | 9 | L0 | +0.484 [+0.226, +0.741] | +5.893 | +0.0325 (L0) |
| Qwen3-30B-A3B-Base | 1 | residual (hidden state) | last option tok. | 256 | L2 | +0.990 [+0.973, +1.003] | +27.333 | +0.2472 (L9) |
| Qwen3-30B-A3B-Base | 1 | residual (hidden state) | first subseq. | 157 | L36 | +0.121 [+0.108, +0.136] | +2.355 | +0.0139 (L35) |
| Qwen3-30B-A3B-Base | 1 | residual (hidden state) | further | 18 | L35 | +0.107 [+0.066, +0.155] | +1.721 | +0.0173 (L35) |
| Qwen3-30B-A3B-Base | 1 | residual (hidden state) | final token | 256 | L47 | +1.000 [+1.000, +1.000] | +17.603 | +0.2443 (L47) |
| Qwen3-30B-A3B-Base | 5 | MoE output | first option tok. | 9 | L2 | +0.398 [+0.150, +0.666] | +2.379 | +0.0354 (L2) |
| Qwen3-30B-A3B-Base | 5 | MoE output | last option tok. | 256 | L2 | +0.973 [+0.956, +0.987] | +6.574 | +0.2398 (L2) |
| Qwen3-30B-A3B-Base | 5 | MoE output | first subseq. | 157 | L38 | +0.066 [+0.058, +0.074] | +0.706 | +0.0049 (L37) |
| Qwen3-30B-A3B-Base | 5 | MoE output | further | 18 | L38 | +0.038 [+0.020, +0.056] | +0.361 | +0.0036 (L39) |
| Qwen3-30B-A3B-Base | 5 | MoE output | final token | 256 | L41 | +0.569 [+0.557, +0.580] | +5.622 | +0.1346 (L41) |
| Mixtral-8x7B, BOS | 1 | attention output | first option tok. | 21 | L12 | +0.015 [+0.007, +0.023] | +0.080 | +0.0087 (L7) |
| Mixtral-8x7B, BOS | 1 | attention output | middle option tok. | 3 | L0 | +0.066 [-0.006, +0.163] | +0.093 | +0.0007 (L0) |
| Mixtral-8x7B, BOS | 1 | attention output | last option tok. | 256 | L12 | +0.052 [+0.044, +0.061] | +0.169 | +0.0067 (L12) |
| Mixtral-8x7B, BOS | 1 | attention output | first subseq. | 157 | L13 | +0.022 [+0.018, +0.026] | +0.117 | +0.0022 (L13) |
| Mixtral-8x7B, BOS | 1 | attention output | further | 18 | L13 | +0.035 [+0.021, +0.051] | +0.100 | +0.0083 (L13) |
| Mixtral-8x7B, BOS | 1 | attention output | final token | 256 | L13 | +0.147 [+0.136, +0.158] | +0.736 | +0.0222 (L13) |
| Mixtral-8x7B, BOS | 1 | MoE output | first option tok. | 21 | L0 | +0.190 [+0.065, +0.345] | +0.250 | +0.0245 (L0) |
| Mixtral-8x7B, BOS | 1 | MoE output | middle option tok. | 3 | L0 | +0.149 [-0.025, +0.500] | +0.328 | +0.0029 (L7) |
| Mixtral-8x7B, BOS | 1 | MoE output | last option tok. | 256 | L0 | +0.900 [+0.874, +0.924] | +1.419 | +0.2705 (L0) |
| Mixtral-8x7B, BOS | 1 | MoE output | first subseq. | 157 | L21 | +0.027 [+0.024, +0.031] | +0.181 | +0.0025 (L21) |
| Mixtral-8x7B, BOS | 1 | MoE output | further | 18 | L20 | +0.017 [+0.009, +0.027] | +0.104 | +0.0023 (L17) |
| Mixtral-8x7B, BOS | 1 | MoE output | final token | 256 | L20 | +0.166 [+0.159, +0.174] | +1.101 | +0.0183 (L20) |
| Mixtral-8x7B, BOS | 1 | residual (hidden state) | first option tok. | 21 | L0 | +0.292 [+0.164, +0.437] | +1.851 | +0.0339 (L0) |
| Mixtral-8x7B, BOS | 1 | residual (hidden state) | middle option tok. | 3 | L1 | +0.199 [-0.025, +0.674] | +2.474 | +0.0043 (L12) |
| Mixtral-8x7B, BOS | 1 | residual (hidden state) | last option tok. | 256 | L2 | +0.985 [+0.970, +0.996] | +13.068 | +0.2863 (L2) |
| Mixtral-8x7B, BOS | 1 | residual (hidden state) | first subseq. | 157 | L15 | +0.120 [+0.107, +0.134] | +1.940 | +0.0198 (L13) |
| Mixtral-8x7B, BOS | 1 | residual (hidden state) | further | 18 | L16 | +0.106 [+0.079, +0.133] | +1.304 | +0.0206 (L13) |
| Mixtral-8x7B, BOS | 1 | residual (hidden state) | final token | 256 | L31 | +1.000 [+1.000, +1.000] | +16.609 | +0.2892 (L31) |
| Mixtral-8x7B, BOS | 5 | MoE output | first option tok. | 21 | L2 | +0.212 [+0.085, +0.365] | +1.410 | +0.0258 (L2) |
| Mixtral-8x7B, BOS | 5 | MoE output | middle option tok. | 3 | L2 | +0.212 [-0.050, +0.717] | +2.311 | +0.0081 (L10) |
| Mixtral-8x7B, BOS | 5 | MoE output | last option tok. | 256 | L2 | +0.984 [+0.968, +0.995] | +9.965 | +0.2870 (L2) |
| Mixtral-8x7B, BOS | 5 | MoE output | first subseq. | 157 | L18 | +0.091 [+0.082, +0.101] | +0.967 | +0.0121 (L18) |
| Mixtral-8x7B, BOS | 5 | MoE output | further | 18 | L19 | +0.064 [+0.046, +0.082] | +0.643 | +0.0079 (L19) |
| Mixtral-8x7B, BOS | 5 | MoE output | final token | 256 | L20 | +0.558 [+0.547, +0.568] | +5.406 | +0.1628 (L20) |

![W3 grid Qwen3](figures/ext7_wino_w3_grid_wino_qwen3_str.png)

![W3 grid Mixtral (BOS)](figures/ext7_wino_w3_grid_wino_mixtral_bos_str.png)

- **Qwen3.** Restoring the residual at the option token restores ≥ 0.5 of the drop up to L28 and ≤ 0.1 from L39 on (the option's identity has left the option position); at the final position the residual restoration rises from 0.1 at L19 through 0.5 at L30 to 0.9 at L41. The first subsequent token peaks at 0.121 (L36), further tokens at 0.107: the information is read from the option position mostly by the final token itself, with a minor relay. MoE output at the option token: peak L0 +0.171 [+0.141, +0.204] (token identity, Zhang & Nanda fn. 1 — not read as computation); at the final token L41 +0.214 [+0.201, +0.227]; attention output at the final token at most +0.032 [+0.027, +0.037] (L42). Last-token column vs the W2 sweep: curve r 0.9996 (MoE), 0.9920 (attention).
- **Mixtral (BOS).** Restoring the residual at the option token restores ≥ 0.5 of the drop up to L12 and ≤ 0.1 from L20 on (the option's identity has left the option position); at the final position the residual restoration rises from 0.1 at L10 through 0.5 at L13 to 0.9 at L23. The first subsequent token peaks at 0.120 (L15), further tokens at 0.106: the information is read from the option position mostly by the final token itself, with a minor relay. MoE output at the option token: peak L0 +0.900 [+0.874, +0.924] (token identity, Zhang & Nanda fn. 1 — not read as computation); at the final token L20 +0.166 [+0.159, +0.174]; attention output at the final token at most +0.147 [+0.136, +0.158] (L13). Last-token column vs the W2 sweep: curve r 1.0000 (MoE), 0.9999 (attention).

Window 5 (MoE output, secondary, Z6): sliding / summed single layers at the peak = Qwen3 last STR token 4.79; Qwen3 last token 0.84; Mixtral (BOS) last STR token 0.92; Mixtral (BOS) last token 0.83.

### W4. Joint decomposition of the final position (revised form)

The planned two-player Shapley split is degenerate: at the final position the token is shared and the MoE is a per-token function, so patching the attention output at every layer restores the clean final residual (A = 1 by construction; confirmed by ext8-addback and ext7-controls), and φ_attn = ½[A + (1 − M)] carries nothing beyond M. Reported instead (form shared by the three Phase-3 agents): A only as a sanity check of the `multi` path; M = all final-position MoE outputs patched, in the denoising direction (corrupted run, attention still reads the corrupted context: sufficiency) and in the noising direction (clean run, MoE outputs set to their corrupted values: necessity); and the direct-path split of h_clean − h_corrupt = Σ_l dAttn_l + Σ_l dMoE_l computed with ext7-controls' shared `direct_split_pairs` (= ext8's `direct_split`): A_direct = Δ(h_corrupt + Σ dAttn) − Δ_corrupt and M_direct = Δ(h_clean − Σ dAttn) − Δ_corrupt with the exact final norm, plus the linear DLA shares. For symmetric pairs used both ways the noising effect of direction d is, in exact arithmetic, the denoising effect of direction 1 − d (same intervention on the same prompt, metric sign-flipped), so M_noise and M_denoise coincide at the pair level; numerically the per-case values differ by bf16 recomputation noise amplified by top-k routing flips (last column: max per directed case), the population ratios by ≤ 0.002.

**W4 (revised form shared by ext7-wino / ext7-controls / ext8-addback): final-position joint patches and direct paths, all / drop (population ratios, pair bootstrap); direct split = moetrace/ext7_controls.direct_split_pairs**

| Model | Pair set | Pairs | M: all MoE outputs, denoise (sufficiency) | M: all MoE outputs, noise (necessity) | A_direct (direct path of attention outputs) | M_direct (direct path of MoE outputs) | DLA share attention | DLA share MoE | A: all attention (sanity, = 1 by construction) | Block all layers (sanity) | max |noise(d) − denoise(1−d)| (directed, logits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 128 | 0.845 [0.827, 0.861] | 0.843 [0.825, 0.859] | 0.047 [0.018, 0.075] | 0.953 [0.925, 0.982] | 0.04 [0.01, 0.07] | 0.96 [0.93, 0.99] | 0.992 [0.981, 1.004] | 1.000 [1.000, 1.000] | 1.88 |
| Qwen3-30B-A3B-Base | replication, validation | 128 | 0.835 [0.817, 0.852] | 0.834 [0.816, 0.851] | 0.048 [0.022, 0.075] | 0.952 [0.925, 0.978] | 0.05 [0.02, 0.07] | 0.95 [0.93, 0.98] | 0.985 [0.975, 0.993] | 1.000 [1.000, 1.000] | 1.69 |
| Qwen3-30B-A3B-Base | main, all 256 | 256 | 0.833 [0.820, 0.845] | 0.832 [0.820, 0.844] | 0.056 [0.037, 0.076] | 0.944 [0.924, 0.963] | 0.05 [0.03, 0.07] | 0.95 [0.93, 0.97] | 0.999 [0.992, 1.005] | 1.000 [1.000, 1.000] | 1.88 |
| Qwen3-30B-A3B-Base | replication, all 256 | 256 | 0.838 [0.826, 0.849] | 0.837 [0.826, 0.849] | 0.039 [0.021, 0.057] | 0.961 [0.943, 0.979] | 0.04 [0.02, 0.06] | 0.96 [0.94, 0.98] | 0.990 [0.984, 0.996] | 1.000 [1.000, 1.000] | 1.69 |
| Mixtral-8x7B, BOS | main, validation | 128 | 0.788 [0.765, 0.810] | 0.788 [0.765, 0.810] | 0.288 [0.265, 0.313] | 0.712 [0.687, 0.735] | 0.29 [0.27, 0.31] | 0.71 [0.69, 0.73] | 1.000 [0.999, 1.002] | 1.000 [1.000, 1.000] | 0.375 |
| Mixtral-8x7B, BOS | replication, validation | 128 | 0.765 [0.744, 0.784] | 0.765 [0.745, 0.785] | 0.308 [0.289, 0.329] | 0.692 [0.671, 0.711] | 0.31 [0.29, 0.33] | 0.69 [0.67, 0.71] | 1.000 [0.999, 1.001] | 1.000 [1.000, 1.000] | 0.5 |
| Mixtral-8x7B, BOS | main, all 256 | 256 | 0.774 [0.756, 0.791] | 0.774 [0.757, 0.791] | 0.300 [0.282, 0.318] | 0.700 [0.682, 0.718] | 0.30 [0.28, 0.32] | 0.70 [0.68, 0.72] | 1.000 [0.999, 1.001] | 1.000 [1.000, 1.000] | 0.375 |
| Mixtral-8x7B, BOS | replication, all 256 | 256 | 0.783 [0.769, 0.796] | 0.783 [0.769, 0.797] | 0.292 [0.278, 0.306] | 0.708 [0.694, 0.722] | 0.29 [0.28, 0.31] | 0.71 [0.69, 0.72] | 1.000 [0.999, 1.001] | 1.000 [1.000, 1.000] | 0.5 |

- All MoE outputs at the final position restore 0.845 [0.827, 0.861] of the drop in Qwen3 and 0.788 [0.765, 0.810] in Mixtral (replication 0.835 [0.817, 0.852] / 0.765 [0.744, 0.784]); ext8-addback's CounterFact STR value is reported in its section (preliminary log: ≈ 0.5 in Qwen3), so WinoGrande's final-position answer is more MoE-sufficient than factual recall's.
- Direct paths: MoE outputs 0.953 [0.925, 0.982] vs attention outputs 0.047 [0.018, 0.075] of the drop (Qwen3), 0.712 [0.687, 0.735] vs 0.288 [0.265, 0.313] (Mixtral); DLA shares attention 0.04 [0.01, 0.07] / 0.29 [0.27, 0.31]. Read with W3: attention transports the option's identity to the final position over many layers, the late MoE outputs write the answer.

**W4 strata (main set, all 256 pairs)**

| Model | Stratum | Value | Pairs | M denoise | M noise | A_direct | M_direct | DLA share attention |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | context-free association passes (assoc) | True | 31 | 0.85 [0.82, 0.88] | 0.85 [0.82, 0.88] | 0.08 [0.04, 0.12] | 0.92 [0.88, 0.96] | 0.07 [0.03, 0.11] |
| Qwen3-30B-A3B-Base | context-free association passes (assoc) | False | 225 | 0.83 [0.82, 0.84] | 0.83 [0.82, 0.84] | 0.05 [0.03, 0.07] | 0.95 [0.93, 0.97] | 0.05 [0.03, 0.07] |
| Qwen3-30B-A3B-Base | person-name options | True | 23 | 0.72 [0.66, 0.78] | 0.72 [0.66, 0.79] | 0.26 [0.20, 0.32] | 0.74 [0.68, 0.80] | 0.26 [0.20, 0.32] |
| Qwen3-30B-A3B-Base | person-name options | False | 233 | 0.84 [0.83, 0.85] | 0.84 [0.83, 0.85] | 0.04 [0.02, 0.06] | 0.96 [0.94, 0.98] | 0.03 [0.01, 0.05] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 64 | 0.78 [0.76, 0.81] | 0.79 [0.76, 0.81] | 0.01 [-0.03, 0.06] | 0.99 [0.94, 1.03] | 0.01 [-0.04, 0.06] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 192 | 0.85 [0.84, 0.86] | 0.85 [0.84, 0.86] | 0.07 [0.05, 0.09] | 0.93 [0.91, 0.95] | 0.07 [0.05, 0.09] |
| Qwen3-30B-A3B-Base | AfLite survivor (train_debiased) | True | 51 | 0.81 [0.78, 0.84] | 0.81 [0.78, 0.84] | 0.15 [0.11, 0.19] | 0.85 [0.81, 0.89] | 0.14 [0.10, 0.19] |
| Qwen3-30B-A3B-Base | AfLite survivor (train_debiased) | False | 205 | 0.84 [0.83, 0.85] | 0.84 [0.82, 0.85] | 0.03 [0.01, 0.06] | 0.97 [0.94, 0.99] | 0.03 [0.01, 0.05] |
| Qwen3-30B-A3B-Base | one-token option | True | 247 | 0.83 [0.82, 0.85] | 0.83 [0.82, 0.85] | 0.05 [0.03, 0.07] | 0.95 [0.93, 0.97] | 0.05 [0.03, 0.07] |
| Qwen3-30B-A3B-Base | trigger word in context | False | 250 | 0.84 [0.82, 0.85] | 0.83 [0.82, 0.85] | 0.05 [0.03, 0.07] | 0.95 [0.93, 0.97] | 0.05 [0.03, 0.07] |
| Qwen3-30B-A3B-Base | direction | A→B | 256 | 0.83 [0.82, 0.85] | 0.83 [0.82, 0.85] | 0.06 [0.04, 0.08] | 0.94 [0.92, 0.96] | 0.05 [0.03, 0.07] |
| Qwen3-30B-A3B-Base | direction | B→A | 256 | 0.83 [0.82, 0.85] | 0.83 [0.82, 0.84] | 0.06 [0.04, 0.08] | 0.94 [0.92, 0.96] | 0.05 [0.03, 0.07] |
| Mixtral-8x7B, BOS | context-free association passes (assoc) | True | 25 | 0.76 [0.71, 0.81] | 0.76 [0.71, 0.81] | 0.31 [0.27, 0.37] | 0.69 [0.63, 0.73] | 0.31 [0.27, 0.37] |
| Mixtral-8x7B, BOS | context-free association passes (assoc) | False | 231 | 0.78 [0.76, 0.79] | 0.78 [0.76, 0.79] | 0.30 [0.28, 0.32] | 0.70 [0.68, 0.72] | 0.30 [0.28, 0.32] |
| Mixtral-8x7B, BOS | person-name options | True | 23 | 0.61 [0.48, 0.72] | 0.61 [0.48, 0.72] | 0.49 [0.39, 0.60] | 0.51 [0.40, 0.61] | 0.49 [0.39, 0.60] |
| Mixtral-8x7B, BOS | person-name options | False | 233 | 0.79 [0.77, 0.80] | 0.79 [0.77, 0.80] | 0.28 [0.27, 0.30] | 0.72 [0.70, 0.73] | 0.28 [0.27, 0.30] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 85 | 0.76 [0.72, 0.79] | 0.76 [0.72, 0.79] | 0.31 [0.28, 0.34] | 0.69 [0.66, 0.72] | 0.31 [0.28, 0.34] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 171 | 0.79 [0.77, 0.80] | 0.79 [0.77, 0.80] | 0.29 [0.27, 0.31] | 0.71 [0.69, 0.73] | 0.29 [0.27, 0.31] |
| Mixtral-8x7B, BOS | AfLite survivor (train_debiased) | True | 51 | 0.74 [0.69, 0.78] | 0.74 [0.68, 0.78] | 0.35 [0.31, 0.39] | 0.65 [0.61, 0.69] | 0.35 [0.31, 0.40] |
| Mixtral-8x7B, BOS | AfLite survivor (train_debiased) | False | 205 | 0.78 [0.76, 0.80] | 0.78 [0.76, 0.80] | 0.29 [0.27, 0.31] | 0.71 [0.69, 0.73] | 0.29 [0.27, 0.31] |
| Mixtral-8x7B, BOS | one-token option | True | 235 | 0.78 [0.76, 0.79] | 0.78 [0.76, 0.79] | 0.30 [0.28, 0.31] | 0.70 [0.69, 0.72] | 0.30 [0.28, 0.31] |
| Mixtral-8x7B, BOS | one-token option | False | 21 | 0.74 [0.66, 0.81] | 0.74 [0.65, 0.81] | 0.34 [0.27, 0.42] | 0.66 [0.58, 0.73] | 0.34 [0.28, 0.42] |
| Mixtral-8x7B, BOS | trigger word in context | False | 250 | 0.78 [0.76, 0.79] | 0.78 [0.76, 0.79] | 0.29 [0.28, 0.31] | 0.71 [0.69, 0.72] | 0.29 [0.28, 0.31] |
| Mixtral-8x7B, BOS | direction | A→B | 256 | 0.78 [0.76, 0.79] | 0.77 [0.75, 0.79] | 0.30 [0.28, 0.32] | 0.70 [0.68, 0.72] | 0.30 [0.28, 0.32] |
| Mixtral-8x7B, BOS | direction | B→A | 256 | 0.77 [0.75, 0.79] | 0.78 [0.76, 0.79] | 0.30 [0.28, 0.32] | 0.70 [0.68, 0.72] | 0.30 [0.28, 0.32] |

**W4 cross-check with ext7-wino's own implementation (scripts/ext7_wino_dla.py; DLA linearised at each run's own final RMS, direct = exact final norm; all / drop, main and replication families, all pairs) and the per-layer DLA peaks**

| Model | Pair set | DLA attention / drop | DLA MoE / drop | DLA embedding (norm scale) / drop | Attention share of DLA | Direct: corrupt + ΣdAttn | Direct: corrupt + ΣdMoE | Direct: clean − ΣdAttn (damage) | Direct: clean − ΣdMoE (damage) | Direct Shapley φ_attn | Top attention layers (DLA / drop) | Top MoE layers (DLA / drop) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | 0.053 [0.033, 0.074] | 0.947 [0.926, 0.967] | 0.000 [-0.000, 0.000] | 0.05 [0.03, 0.07] | 0.057 [0.038, 0.078] | 0.943 [0.923, 0.963] | 0.057 [0.037, 0.077] | 0.943 [0.922, 0.962] | 0.06 [0.04, 0.08] | L42 +0.020, L43 +0.015, L38 +0.010 | L43 +0.216, L41 +0.215, L44 +0.136 |
| Qwen3-30B-A3B-Base | rep | 0.036 [0.018, 0.055] | 0.964 [0.945, 0.982] | 0.000 [-0.000, 0.000] | 0.04 [0.02, 0.06] | 0.041 [0.022, 0.059] | 0.960 [0.942, 0.978] | 0.040 [0.022, 0.058] | 0.959 [0.941, 0.978] | 0.04 [0.02, 0.06] | L42 +0.024, L43 +0.015, L38 +0.011 | L43 +0.245, L41 +0.194, L44 +0.143 |
| Mixtral-8x7B, BOS | main | 0.301 [0.283, 0.319] | 0.699 [0.681, 0.716] | -0.000 [-0.000, 0.000] | 0.30 [0.28, 0.32] | 0.299 [0.282, 0.318] | 0.700 [0.682, 0.717] | 0.299 [0.282, 0.318] | 0.700 [0.682, 0.717] | 0.30 [0.28, 0.32] | L25 +0.084, L31 +0.049, L29 +0.025 | L26 +0.165, L22 +0.125, L21 +0.107 |
| Mixtral-8x7B, BOS | rep | 0.293 [0.279, 0.306] | 0.708 [0.694, 0.722] | -0.000 [-0.000, 0.000] | 0.29 [0.28, 0.31] | 0.292 [0.278, 0.306] | 0.709 [0.694, 0.723] | 0.292 [0.278, 0.306] | 0.708 [0.694, 0.722] | 0.29 [0.28, 0.31] | L25 +0.082, L31 +0.050, L29 +0.024 | L26 +0.168, L22 +0.120, L21 +0.106 |

### W5. Attention heads at the W2 attention layers

**W5: top six and bottom three heads by validation rescue (z over all scanned heads of the model; detection = |z| ≥ 2 on discovery AND validation, Zhang & Nanda §3)**

| Model | Head | Val rescue [95% CI] | Disc. mean | z (val / disc) | ≥ 2 SD both splits | Spec vs other heads | Share of attn_layer |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L38H18 | +0.342 [+0.301, +0.388] | +0.380 | +4.7 / +5.2 | yes | +0.339 [+0.298, +0.384] | 3.58 |
| Qwen3-30B-A3B-Base | L38H21 | +0.289 [+0.241, +0.342] | +0.291 | +4.0 / +3.9 | yes | +0.284 [+0.237, +0.336] | 3.02 |
| Qwen3-30B-A3B-Base | L38H19 | +0.164 [+0.138, +0.191] | +0.139 | +2.2 / +1.8 | no | +0.155 [+0.131, +0.181] | 1.72 |
| Qwen3-30B-A3B-Base | L42H30 | +0.163 [+0.135, +0.195] | +0.210 | +2.2 / +2.8 | yes | +0.157 [+0.129, +0.188] | 0.65 |
| Qwen3-30B-A3B-Base | L39H5 | +0.147 [+0.113, +0.182] | +0.202 | +1.9 / +2.7 | no | +0.141 [+0.109, +0.176] | 0.93 |
| Qwen3-30B-A3B-Base | L38H20 | +0.146 [+0.120, +0.174] | +0.107 | +1.9 / +1.4 | no | +0.136 [+0.112, +0.162] | 1.52 |
| Qwen3-30B-A3B-Base | L39H9 | -0.129 [-0.162, -0.096] | -0.127 | -1.9 / -1.9 | no | -0.143 [-0.177, -0.111] | -0.81 |
| Qwen3-30B-A3B-Base | L42H26 | -0.248 [-0.304, -0.198] | -0.293 | -3.6 / -4.2 | yes | -0.267 [-0.323, -0.218] | -0.99 |
| Qwen3-30B-A3B-Base | L38H16 | -0.547 [-0.602, -0.497] | -0.468 | -7.9 / -6.6 | yes | -0.579 [-0.633, -0.529] | -5.71 |
| Mixtral-8x7B, BOS | L25H9 | +0.553 [+0.499, +0.608] | +0.521 | +8.3 / +7.8 | yes | +0.554 [+0.500, +0.610] | 0.95 |
| Mixtral-8x7B, BOS | L19H13 | +0.364 [+0.319, +0.410] | +0.367 | +5.4 / +5.4 | yes | +0.352 [+0.308, +0.399] | 0.42 |
| Mixtral-8x7B, BOS | L13H18 | +0.193 [+0.159, +0.229] | +0.236 | +2.8 / +3.4 | yes | +0.181 [+0.147, +0.216] | 0.18 |
| Mixtral-8x7B, BOS | L13H11 | +0.191 [+0.138, +0.250] | +0.178 | +2.7 / +2.5 | yes | +0.179 [+0.125, +0.240] | 0.18 |
| Mixtral-8x7B, BOS | L19H12 | +0.188 [+0.154, +0.229] | +0.179 | +2.7 / +2.5 | yes | +0.171 [+0.137, +0.211] | 0.22 |
| Mixtral-8x7B, BOS | L13H4 | +0.178 [+0.150, +0.208] | +0.223 | +2.5 / +3.2 | yes | +0.165 [+0.138, +0.195] | 0.17 |
| Mixtral-8x7B, BOS | L28H14 | -0.027 [-0.039, -0.016] | -0.035 | -0.6 / -0.8 | no | -0.030 [-0.040, -0.020] | -1.02 |
| Mixtral-8x7B, BOS | L19H3 | -0.059 [-0.073, -0.046] | -0.061 | -1.1 / -1.2 | no | -0.084 [-0.098, -0.071] | -0.07 |
| Mixtral-8x7B, BOS | L25H11 | -0.159 [-0.179, -0.139] | -0.157 | -2.6 / -2.7 | yes | -0.180 [-0.201, -0.160] | -0.27 |

**W5: per layer, additivity of the head patches and greedy additive minimal head sets (order by discovery, validation sums)**

| Model | Layer | Attention output (val) | Sum of heads | r(sum, attn) pairs | MoE output | Block | Top head (disc.) | Heads for 50 % | Heads for 80 % | Top-3 share |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L38 | +0.096 [-0.023, +0.218] | +0.439 [+0.159, +0.707] | 0.47 | +0.742 [+0.642, +0.846] | +0.821 [+0.665, +0.975] | H18 | 1 | 1 | 8.31 |
| Qwen3-30B-A3B-Base | L39 | +0.158 [+0.091, +0.230] | +0.314 [+0.067, +0.562] | 0.39 | +1.321 [+1.164, +1.484] | +1.459 [+1.296, +1.630] | H5 | 1 | 1 | 1.76 |
| Qwen3-30B-A3B-Base | L40 | +0.190 [+0.121, +0.259] | +0.280 [+0.002, +0.554] | 0.48 | +0.504 [+0.411, +0.599] | +0.683 [+0.567, +0.799] | H15 | 2 | 2 | 1.09 |
| Qwen3-30B-A3B-Base | L42 | +0.250 [+0.189, +0.308] | +0.361 [+0.112, +0.614] | 0.55 | +0.678 [+0.584, +0.782] | +0.904 [+0.791, +1.026] | H30 | 1 | 2 | 1.36 |
| Qwen3-30B-A3B-Base | L45 (null) | +0.005 [-0.010, +0.022] | +0.030 [-0.178, +0.232] | 0.46 | -0.075 [-0.176, +0.028] | -0.064 [-0.164, +0.039] | H25 | 7 | 8 | -0.77 |
| Mixtral-8x7B, BOS | L13 | +1.073 [+0.958, +1.190] | +0.572 [+0.336, +0.799] | 0.27 | +0.267 [+0.177, +0.365] | +1.247 [+1.073, +1.444] | H18 | 3 | not reached | 0.52 |
| Mixtral-8x7B, BOS | L15 | +0.551 [+0.485, +0.621] | +0.306 [+0.068, +0.531] | 0.33 | +0.256 [+0.209, +0.307] | +0.827 [+0.747, +0.908] | H8 | 2 | not reached | 0.64 |
| Mixtral-8x7B, BOS | L19 | +0.869 [+0.788, +0.953] | +0.719 [+0.508, +0.923] | 0.16 | +1.106 [+1.029, +1.189] | +1.920 [+1.789, +2.052] | H13 | 2 | 3 | 0.81 |
| Mixtral-8x7B, BOS | L25 | +0.583 [+0.525, +0.641] | +0.515 [+0.318, +0.708] | 0.32 | +0.641 [+0.558, +0.730] | +1.198 [+1.093, +1.306] | H9 | 1 | 1 | 1.18 |
| Mixtral-8x7B, BOS | L28 (null) | +0.027 [+0.013, +0.041] | +0.046 [-0.120, +0.212] | 0.42 | +0.211 [+0.128, +0.298] | +0.245 [+0.162, +0.331] | H13 | 1 | 1 | 2.09 |

**W5: final-position attention mass of the top heads by position class (str = the filled option; ment_filled / ment_other = first mention of the candidate filled in the clean prompt / of the other one)**

| Model | Head | Val rescue | final (clean → corrupt) | str (clean → corrupt) | ment_filled (clean → corrupt) | ment_other (clean → corrupt) | pos0 (clean → corrupt) | other (clean → corrupt) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L38H18 | +0.342 | 0.09 → 0.09 | 0.12 → 0.12 | 0.06 → 0.04 | 0.04 → 0.06 | 0.48 → 0.48 | 0.21 → 0.22 |
| Qwen3-30B-A3B-Base | L38H21 | +0.289 | 0.06 → 0.06 | 0.07 → 0.07 | 0.05 → 0.04 | 0.04 → 0.05 | 0.47 → 0.47 | 0.31 → 0.31 |
| Qwen3-30B-A3B-Base | L38H19 | +0.164 | 0.09 → 0.09 | 0.11 → 0.11 | 0.02 → 0.02 | 0.02 → 0.02 | 0.41 → 0.41 | 0.36 → 0.36 |
| Qwen3-30B-A3B-Base | L42H30 | +0.163 | 0.15 → 0.15 | 0.24 → 0.24 | 0.04 → 0.02 | 0.02 → 0.04 | 0.21 → 0.21 | 0.34 → 0.34 |
| Qwen3-30B-A3B-Base | L39H5 | +0.147 | 0.03 → 0.03 | 0.32 → 0.32 | 0.14 → 0.05 | 0.05 → 0.14 | 0.28 → 0.27 | 0.18 → 0.18 |
| Qwen3-30B-A3B-Base | L38H20 | +0.146 | 0.06 → 0.06 | 0.16 → 0.17 | 0.04 → 0.04 | 0.04 → 0.04 | 0.29 → 0.29 | 0.41 → 0.41 |
| Mixtral-8x7B, BOS | L25H9 | +0.553 | 0.12 → 0.12 | 0.21 → 0.21 | 0.04 → 0.01 | 0.01 → 0.04 | 0.18 → 0.18 | 0.44 → 0.44 |
| Mixtral-8x7B, BOS | L19H13 | +0.364 | 0.03 → 0.03 | 0.51 → 0.51 | 0.12 → 0.03 | 0.03 → 0.12 | 0.10 → 0.10 | 0.21 → 0.21 |
| Mixtral-8x7B, BOS | L13H18 | +0.193 | 0.03 → 0.03 | 0.27 → 0.27 | 0.08 → 0.04 | 0.04 → 0.08 | 0.18 → 0.18 | 0.40 → 0.40 |
| Mixtral-8x7B, BOS | L13H11 | +0.191 | 0.03 → 0.03 | 0.34 → 0.34 | 0.06 → 0.03 | 0.03 → 0.06 | 0.31 → 0.31 | 0.24 → 0.24 |
| Mixtral-8x7B, BOS | L19H12 | +0.188 | 0.04 → 0.04 | 0.32 → 0.32 | 0.14 → 0.03 | 0.03 → 0.14 | 0.22 → 0.23 | 0.25 → 0.25 |
| Mixtral-8x7B, BOS | L13H4 | +0.178 | 0.02 → 0.02 | 0.60 → 0.60 | 0.06 → 0.02 | 0.02 → 0.06 | 0.07 → 0.07 | 0.22 → 0.22 |

![W5 heads](figures/ext7_wino_w5_heads.png)

- **Qwen3** (layers L38, L39, L40, L42, L45; null layer L45): 5 of 160 heads at |z| ≥ 2 on both splits — positive L38H18 +0.342 [+0.301, +0.388], L38H21 +0.289 [+0.241, +0.342], L42H30 +0.163 [+0.135, +0.195]; negative L42H26 -0.248, L38H16 -0.547. L38: attention output +0.096, sum of single heads +0.439 (per-pair r 0.47). L39: attention output +0.158, sum of single heads +0.314 (per-pair r 0.39). L40: attention output +0.190, sum of single heads +0.280 (per-pair r 0.48). L42: attention output +0.250, sum of single heads +0.361 (per-pair r 0.55). Single-head patches are far from additive (per-pair r 0.39–0.55), so additive minimal head sets are not interpreted. Attention of the top heads at the final position (clean → corrupted): L38H18 option 0.12 → 0.12, first mention of the clean filler 0.06 → 0.04, of the other candidate 0.04 → 0.06, position 0 0.48; L38H21 option 0.07 → 0.07, first mention of the clean filler 0.05 → 0.04, of the other candidate 0.04 → 0.05, position 0 0.47; L38H19 option 0.11 → 0.11, first mention of the clean filler 0.02 → 0.02, of the other candidate 0.02 → 0.02, position 0 0.41.
- **Mixtral (BOS)** (layers L13, L15, L19, L25, L28; null layer L28): 7 of 160 heads at |z| ≥ 2 on both splits — positive L25H9 +0.553 [+0.499, +0.608], L19H13 +0.364 [+0.319, +0.410], L13H18 +0.193 [+0.159, +0.229], L13H11 +0.191 [+0.138, +0.250], L19H12 +0.188 [+0.154, +0.229], L13H4 +0.178 [+0.150, +0.208]; negative L25H11 -0.159. L13: attention output +1.073, sum of single heads +0.572 (per-pair r 0.27). L15: attention output +0.551, sum of single heads +0.306 (per-pair r 0.33). L19: attention output +0.869, sum of single heads +0.719 (per-pair r 0.16). L25: attention output +0.583, sum of single heads +0.515 (per-pair r 0.32). Single-head patches are far from additive (per-pair r 0.16–0.42), so additive minimal head sets are not interpreted. Attention of the top heads at the final position (clean → corrupted): L25H9 option 0.21 → 0.21, first mention of the clean filler 0.04 → 0.01, of the other candidate 0.01 → 0.04, position 0 0.18; L19H13 option 0.51 → 0.51, first mention of the clean filler 0.12 → 0.03, of the other candidate 0.03 → 0.12, position 0 0.10; L13H18 option 0.27 → 0.27, first mention of the clean filler 0.08 → 0.04, of the other candidate 0.04 → 0.08, position 0 0.18.

### W6. Experts: two-stage selection, joint search, CounterFact experts

**W6: paper two-stage selection on WinoGrande STR (MoE-layer argmax on discovery → recurrence-first expert; validation pairs)**

| Model | Pair set | Rule | Layer | Layer rescue (val) [95% CI] | Layer / drop | Selected expert | Disc. active | Val rescue | Spec | Spec / drop | Clean top-k coalition | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | two-stage | L41 | +1.778 [+1.594, +1.972] | 0.218 [0.200, 0.238] | L41E117 | 209/256 (≥ 128), 6 cand. | +1.002 [+0.844, +1.171] | +0.903 [+0.744, +1.071] | 0.111 [0.092, 0.131] | +1.800 [+1.613, +1.998] | A |
| Qwen3-30B-A3B-Base | main | interior (≤ L−5) | L41 | +1.778 [+1.594, +1.972] | 0.218 [0.200, 0.238] | L41E117 | 209/256 (≥ 128), 6 cand. | +1.002 [+0.844, +1.171] | +0.903 [+0.744, +1.071] | 0.111 [0.092, 0.131] | +1.800 [+1.613, +1.998] | A |
| Qwen3-30B-A3B-Base | replication | two-stage | L41 | +1.715 [+1.526, +1.911] | 0.200 [0.180, 0.219] | L41E117 | 218/256 (≥ 128), 6 cand. | +0.809 [+0.668, +0.960] | +0.693 [+0.548, +0.844] | 0.081 [0.064, 0.097] | +1.676 [+1.488, +1.865] | A |
| Qwen3-30B-A3B-Base | replication | interior (≤ L−5) | L41 | +1.715 [+1.526, +1.911] | 0.200 [0.180, 0.219] | L41E117 | 218/256 (≥ 128), 6 cand. | +0.809 [+0.668, +0.960] | +0.693 [+0.548, +0.844] | 0.081 [0.064, 0.097] | +1.676 [+1.488, +1.865] | A |
| Mixtral-8x7B, BOS | main | two-stage | L20 | +1.324 [+1.222, +1.429] | 0.172 [0.162, 0.183] | L20E000 | 246/256 (≥ 128), 1 cand. | +1.106 [+1.013, +1.203] | +0.940 [+0.836, +1.042] | 0.122 [0.109, 0.135] | +1.322 [+1.222, +1.427] | A |
| Mixtral-8x7B, BOS | main | interior (≤ L−5) | L20 | +1.324 [+1.222, +1.429] | 0.172 [0.162, 0.183] | L20E000 | 246/256 (≥ 128), 1 cand. | +1.106 [+1.013, +1.203] | +0.940 [+0.836, +1.042] | 0.122 [0.109, 0.135] | +1.322 [+1.222, +1.427] | A |
| Mixtral-8x7B, BOS | replication | two-stage | L20 | +1.396 [+1.296, +1.502] | 0.165 [0.155, 0.175] | L20E000 | 252/256 (≥ 128), 1 cand. | +1.166 [+1.067, +1.269] | +0.974 [+0.867, +1.084] | 0.115 [0.104, 0.126] | +1.381 [+1.282, +1.487] | A |
| Mixtral-8x7B, BOS | replication | interior (≤ L−5) | L20 | +1.396 [+1.296, +1.502] | 0.165 [0.155, 0.175] | L20E000 | 252/256 (≥ 128), 1 cand. | +1.166 [+1.067, +1.269] | +0.974 [+0.867, +1.084] | 0.115 [0.104, 0.126] | +1.381 [+1.282, +1.487] | A |

**W6: equal-norm check at the selected layer (Qwen3: gate-matched control, Table 9; Mixtral: other active expert, Table 11)**

| Model | Expert | Control | n (anchor-active val, directed) | Raw Spec vs control | Selected, equal norm | Control, equal norm | Equal-norm Spec |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | gate_matched | 223 | +1.034 [+0.866, +1.212] | +0.534 [+0.458, +0.617] | +0.084 [+0.055, +0.116] | +0.450 [+0.373, +0.534] |
| Mixtral-8x7B, BOS | L20E000 | active_pair | 246 | +0.996 [+0.900, +1.088] | +0.313 [+0.276, +0.351] | +0.152 [+0.129, +0.177] | +0.160 [+0.132, +0.189] |

**W6: joint (layer, expert) search over all layers (recurrence gate = half of the discovery directed cases)**

| Model | Pair set | Rank | (layer, expert) | Disc. active (directed) | Disc. all-case | Val rescue | Spec |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | 1 | L41E117 | 209 | +0.788 | +1.002 [+0.844, +1.171] | +0.903 [+0.744, +1.071] |
| Qwen3-30B-A3B-Base | main | 2 | L43E081 | 251 | +0.649 | +0.701 [+0.595, +0.806] | +0.607 [+0.496, +0.718] |
| Qwen3-30B-A3B-Base | main | 3 | L39E071 | 205 | +0.448 | +0.451 [+0.363, +0.546] | +0.348 [+0.258, +0.444] |
| Qwen3-30B-A3B-Base | main | 4 | L44E122 | 217 | +0.358 | +0.297 [+0.228, +0.374] | +0.251 [+0.178, +0.332] |
| Qwen3-30B-A3B-Base | main | 5 | L41E053 | 244 | +0.352 | +0.265 [+0.211, +0.323] | +0.041 [-0.031, +0.115] |
| Qwen3-30B-A3B-Base | replication | 1 | L43E081 | 256 | +0.745 | +0.715 [+0.592, +0.841] | +0.603 [+0.475, +0.736] |
| Qwen3-30B-A3B-Base | replication | 2 | L41E117 | 218 | +0.699 | +0.809 [+0.668, +0.960] | +0.693 [+0.548, +0.844] |
| Qwen3-30B-A3B-Base | replication | 3 | L39E071 | 192 | +0.420 | +0.446 [+0.362, +0.533] | +0.346 [+0.258, +0.439] |
| Qwen3-30B-A3B-Base | replication | 4 | L44E122 | 224 | +0.339 | +0.333 [+0.252, +0.425] | +0.266 [+0.182, +0.362] |
| Qwen3-30B-A3B-Base | replication | 5 | L41E053 | 252 | +0.326 | +0.310 [+0.250, +0.372] | +0.130 [+0.060, +0.203] |
| Mixtral-8x7B, BOS | main | 1 | L20E000 | 246 | +1.053 | +1.106 [+1.013, +1.203] | +0.940 [+0.836, +1.042] |
| Mixtral-8x7B, BOS | main | 2 | L19E006 | 248 | +1.002 | +0.966 [+0.893, +1.039] | +0.841 [+0.765, +0.919] |
| Mixtral-8x7B, BOS | main | 3 | L21E006 | 228 | +0.738 | +0.630 [+0.553, +0.711] | +0.162 [+0.048, +0.276] |
| Mixtral-8x7B, BOS | main | 4 | L26E002 | 235 | +0.562 | +0.583 [+0.489, +0.682] | +0.261 [+0.153, +0.377] |
| Mixtral-8x7B, BOS | main | 5 | L18E005 | 234 | +0.441 | +0.354 [+0.277, +0.440] | +0.043 [-0.029, +0.123] |
| Mixtral-8x7B, BOS | replication | 1 | L20E000 | 252 | +1.033 | +1.166 [+1.067, +1.269] | +0.974 [+0.867, +1.084] |
| Mixtral-8x7B, BOS | replication | 2 | L19E006 | 256 | +0.969 | +1.078 [+0.990, +1.172] | +0.922 [+0.831, +1.018] |
| Mixtral-8x7B, BOS | replication | 3 | L21E006 | 241 | +0.630 | +0.719 [+0.640, +0.797] | +0.284 [+0.177, +0.387] |
| Mixtral-8x7B, BOS | replication | 4 | L26E002 | 242 | +0.597 | +0.648 [+0.546, +0.755] | +0.330 [+0.205, +0.455] |
| Mixtral-8x7B, BOS | replication | 5 | L24E002 | 192 | +0.428 | +0.510 [+0.427, +0.600] | +0.292 [+0.213, +0.375] |

**W6: the CounterFact STR experts (Direction 6) as fixed hypotheses on WinoGrande validation pairs**

| Model | CounterFact expert | Disc. active | Val active | Val rescue | Spec | Rescue / drop |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L44E069 | 6/256 | 7 | +0.009 [-0.001, +0.025] | -0.085 [-0.108, -0.061] | 0.001 [-0.000, 0.003] |
| Qwen3-30B-A3B-Base | L42E115 | 5/256 | 7 | +0.033 [+0.000, +0.079] | -0.050 [-0.089, -0.003] | 0.004 [0.000, 0.010] |
| Mixtral-8x7B, BOS | L19E002 | 9/256 | 6 | +0.004 [-0.002, +0.015] | -0.545 [-0.610, -0.481] | 0.001 [-0.000, 0.002] |
| Mixtral-8x7B, BOS | L21E001 | 8/256 | 12 | +0.006 [+0.001, +0.014] | -0.583 [-0.662, -0.510] | 0.001 [0.000, 0.002] |
| Mixtral-8x7B, BOS | L18E001 | 3/256 | 3 | +0.006 [+0.000, +0.019] | -0.343 [-0.400, -0.291] | 0.001 [0.000, 0.003] |
| Mixtral-8x7B, BOS | L19E006 | 248/256 | 256 | +0.966 [+0.893, +1.039] | +0.841 [+0.765, +0.919] | 0.126 [0.117, 0.134] |

- **Qwen3 L41E117**: active in 209 of 256 discovery directed cases; carries 56% of the L41 MoE rescue; rank 1 among the case's active experts in 178 of 223 anchor-active validation cases; at equal norm with its gate-matched partner it still wins by +0.450 [+0.373, +0.534] (raw +1.034 [+0.866, +1.212]). Replication set: L41E117 again (Spec +0.693 [+0.548, +0.844]). The joint search over all 48 layers puts it first on main and second on the replication set behind L43E081 (the main set's second locus).
- **Mixtral L20E000**: active in 246 of 256; Spec +0.940 [+0.836, +1.042]; equal norm +0.160 [+0.132, +0.189] (the other active expert as control, Table 11 analogue); replicated (L20E000, Spec +0.974 [+0.867, +1.084]); second locus L19E006, the expert that carried Mixtral's sink-state final tokens without BOS (Direction 3) and is an ordinary, positively specific content expert here.
- The CounterFact STR selections (Qwen3 L44E069 / L42E115; Mixtral BOS L19E002 / L21E001 / L18E001) are routed at the WinoGrande final position in only 3–12 of 256 discovery directed cases and rescue ≈ 0 (table above); only L19E006 — the paper's Mixtral expert, negatively specific on CounterFact — is routed here (248 of 256) and is WinoGrande's second locus. Factual recall and WinoGrande's trigger prediction use different late experts in both models.

### Strata and robustness

**W2 strata (all 256 main pairs, layers fixed by the main discovery argmax; descriptive)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Block at L*_block / drop | Attention share | Peak attention / peak MoE (normalised) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | context-free association passes (assoc) | True | 31 | 10.10 | 0.218 [0.186, 0.250] | 0.013 [-0.002, 0.028] | 0.246 [0.214, 0.276] | 0.21 [0.19, 0.26] | 0.047 / 0.218 |
| Qwen3-30B-A3B-Base | context-free association passes (assoc) | False | 225 | 8.07 | 0.211 [0.197, 0.226] | 0.026 [0.013, 0.039] | 0.231 [0.216, 0.245] | 0.17 [0.15, 0.20] | 0.032 / 0.211 |
| Qwen3-30B-A3B-Base | person-name options | True | 23 | 7.93 | 0.109 [0.078, 0.148] | 0.132 [0.071, 0.194] | 0.149 [0.111, 0.193] | 0.30 [0.26, 0.38] | 0.132 / 0.169 |
| Qwen3-30B-A3B-Base | person-name options | False | 233 | 8.35 | 0.222 [0.208, 0.236] | 0.014 [0.004, 0.024] | 0.241 [0.228, 0.254] | 0.16 [0.14, 0.19] | 0.030 / 0.222 |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 64 | 8.68 | 0.199 [0.177, 0.221] | 0.009 [-0.006, 0.025] | 0.213 [0.189, 0.238] | 0.17 [0.15, 0.21] | 0.036 / 0.199 |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 192 | 8.19 | 0.217 [0.200, 0.234] | 0.030 [0.015, 0.045] | 0.240 [0.224, 0.255] | 0.18 [0.15, 0.21] | 0.030 / 0.217 |
| Qwen3-30B-A3B-Base | AfLite survivor (train_debiased) | True | 51 | 8.01 | 0.187 [0.161, 0.215] | 0.050 [0.025, 0.080] | 0.222 [0.196, 0.249] | 0.28 [0.24, 0.33] | 0.050 / 0.187 |
| Qwen3-30B-A3B-Base | AfLite survivor (train_debiased) | False | 205 | 8.39 | 0.218 [0.203, 0.233] | 0.018 [0.006, 0.031] | 0.235 [0.220, 0.251] | 0.16 [0.14, 0.19] | 0.031 / 0.218 |
| Qwen3-30B-A3B-Base | one-token option | True | 247 | 8.34 | 0.215 [0.201, 0.229] | 0.023 [0.012, 0.035] | 0.234 [0.220, 0.248] | 0.17 [0.15, 0.20] | 0.031 / 0.215 |
| Qwen3-30B-A3B-Base | one-token option | False | 9 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base | trigger word in context | True | 6 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base | trigger word in context | False | 250 | 8.32 | 0.215 [0.201, 0.229] | 0.022 [0.010, 0.033] | 0.235 [0.221, 0.249] | 0.17 [0.15, 0.20] | 0.031 / 0.215 |
| Mixtral-8x7B, BOS | context-free association passes (assoc) | True | 25 | 9.61 | 0.173 [0.149, 0.196] | 0.091 [0.068, 0.115] | 0.262 [0.241, 0.284] | 0.36 [0.34, 0.39] | 0.129 / 0.181 |
| Mixtral-8x7B, BOS | context-free association passes (assoc) | False | 231 | 7.58 | 0.165 [0.157, 0.172] | 0.154 [0.143, 0.166] | 0.246 [0.236, 0.256] | 0.35 [0.34, 0.36] | 0.154 / 0.165 |
| Mixtral-8x7B, BOS | person-name options | True | 23 | 6.63 | 0.124 [0.094, 0.155] | 0.146 [0.112, 0.180] | 0.255 [0.217, 0.295] | 0.43 [0.39, 0.48] | 0.146 / 0.169 |
| Mixtral-8x7B, BOS | person-name options | False | 233 | 7.90 | 0.169 [0.161, 0.176] | 0.147 [0.135, 0.159] | 0.247 [0.238, 0.257] | 0.34 [0.33, 0.35] | 0.147 / 0.169 |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 85 | 8.88 | 0.152 [0.140, 0.164] | 0.174 [0.153, 0.195] | 0.222 [0.205, 0.238] | 0.34 [0.33, 0.36] | 0.174 / 0.153 |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 171 | 7.23 | 0.174 [0.165, 0.183] | 0.130 [0.119, 0.142] | 0.264 [0.254, 0.274] | 0.35 [0.34, 0.36] | 0.130 / 0.174 |
| Mixtral-8x7B, BOS | AfLite survivor (train_debiased) | True | 51 | 7.15 | 0.163 [0.145, 0.181] | 0.125 [0.097, 0.159] | 0.246 [0.224, 0.267] | 0.37 [0.34, 0.39] | 0.125 / 0.163 |
| Mixtral-8x7B, BOS | AfLite survivor (train_debiased) | False | 205 | 7.94 | 0.166 [0.158, 0.174] | 0.151 [0.139, 0.163] | 0.248 [0.238, 0.259] | 0.35 [0.34, 0.36] | 0.151 / 0.166 |
| Mixtral-8x7B, BOS | one-token option | True | 235 | 7.83 | 0.167 [0.159, 0.175] | 0.146 [0.135, 0.157] | 0.248 [0.238, 0.258] | 0.35 [0.34, 0.36] | 0.146 / 0.167 |
| Mixtral-8x7B, BOS | one-token option | False | 21 | 7.28 | 0.152 [0.123, 0.180] | 0.154 [0.104, 0.212] | 0.247 [0.222, 0.274] | 0.37 [0.33, 0.41] | 0.154 / 0.160 |
| Mixtral-8x7B, BOS | trigger word in context | True | 6 |  |  |  |  |  |  |
| Mixtral-8x7B, BOS | trigger word in context | False | 250 | 7.83 | 0.167 [0.160, 0.175] | 0.147 [0.136, 0.158] | 0.249 [0.239, 0.258] | 0.35 [0.34, 0.36] | 0.147 / 0.167 |

**W6 strata: the main two-stage expert on all 256 main pairs by stratum (descriptive; layer and expert fixed)**

| Model | Expert | Stratum | Value | Pairs | Layer rescue | Expert rescue | Spec |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | context-free association passes (assoc) | True | 31 | +2.203 [+1.754, +2.664] | +1.028 [+0.738, +1.342] | +0.883 [+0.583, +1.203] |
| Qwen3-30B-A3B-Base | L41E117 | context-free association passes (assoc) | False | 225 | +1.706 [+1.563, +1.851] | +0.877 [+0.760, +1.002] | +0.776 [+0.657, +0.900] |
| Qwen3-30B-A3B-Base | L41E117 | person-name options | True | 23 | +0.867 [+0.587, +1.215] | +0.046 [-0.008, +0.114] | -0.060 [-0.139, +0.016] |
| Qwen3-30B-A3B-Base | L41E117 | person-name options | False | 233 | +1.855 [+1.712, +2.000] | +0.979 [+0.861, +1.098] | +0.873 [+0.754, +0.993] |
| Qwen3-30B-A3B-Base | L41E117 | top-1 in both directions | True | 64 | +1.723 [+1.499, +1.967] | +0.609 [+0.400, +0.824] | +0.480 [+0.271, +0.699] |
| Qwen3-30B-A3B-Base | L41E117 | top-1 in both directions | False | 192 | +1.780 [+1.618, +1.950] | +0.990 [+0.856, +1.126] | +0.892 [+0.756, +1.029] |
| Qwen3-30B-A3B-Base | L41E117 | AfLite survivor (train_debiased) | True | 51 | +1.499 [+1.239, +1.798] | +0.675 [+0.483, +0.890] | +0.559 [+0.357, +0.787] |
| Qwen3-30B-A3B-Base | L41E117 | AfLite survivor (train_debiased) | False | 205 | +1.832 [+1.672, +1.992] | +0.950 [+0.822, +1.083] | +0.846 [+0.717, +0.978] |
| Qwen3-30B-A3B-Base | L41E117 | one-token option | True | 247 | +1.790 [+1.645, +1.940] | +0.914 [+0.798, +1.034] | +0.807 [+0.690, +0.927] |
| Qwen3-30B-A3B-Base | L41E117 | trigger word in context | False | 250 | +1.788 [+1.647, +1.936] | +0.908 [+0.791, +1.033] | +0.800 [+0.684, +0.924] |
| Mixtral-8x7B, BOS | L20E000 | context-free association passes (assoc) | True | 25 | +1.660 [+1.390, +1.930] | +1.391 [+1.121, +1.656] | +1.206 [+0.930, +1.473] |
| Mixtral-8x7B, BOS | L20E000 | context-free association passes (assoc) | False | 231 | +1.248 [+1.175, +1.322] | +1.046 [+0.975, +1.115] | +0.877 [+0.801, +0.949] |
| Mixtral-8x7B, BOS | L20E000 | person-name options | True | 23 | +0.823 [+0.584, +1.092] | +0.405 [+0.253, +0.573] | +0.190 [-0.038, +0.416] |
| Mixtral-8x7B, BOS | L20E000 | person-name options | False | 233 | +1.334 [+1.263, +1.407] | +1.146 [+1.080, +1.214] | +0.980 [+0.912, +1.049] |
| Mixtral-8x7B, BOS | L20E000 | top-1 in both directions | True | 85 | +1.348 [+1.226, +1.474] | +1.064 [+0.947, +1.181] | +0.847 [+0.716, +0.975] |
| Mixtral-8x7B, BOS | L20E000 | top-1 in both directions | False | 171 | +1.259 [+1.168, +1.351] | +1.088 [+1.001, +1.176] | +0.939 [+0.852, +1.030] |
| Mixtral-8x7B, BOS | L20E000 | AfLite survivor (train_debiased) | True | 51 | +1.165 [+0.984, +1.343] | +1.001 [+0.830, +1.178] | +0.855 [+0.675, +1.043] |
| Mixtral-8x7B, BOS | L20E000 | AfLite survivor (train_debiased) | False | 205 | +1.319 [+1.242, +1.396] | +1.099 [+1.026, +1.174] | +0.922 [+0.845, +1.001] |
| Mixtral-8x7B, BOS | L20E000 | one-token option | True | 235 | +1.305 [+1.230, +1.382] | +1.094 [+1.023, +1.167] | +0.924 [+0.847, +0.999] |
| Mixtral-8x7B, BOS | L20E000 | one-token option | False | 21 | +1.107 [+0.842, +1.354] | +0.917 [+0.661, +1.170] | +0.744 [+0.461, +1.015] |
| Mixtral-8x7B, BOS | L20E000 | trigger word in context | False | 250 | +1.307 [+1.234, +1.380] | +1.095 [+1.025, +1.167] | +0.922 [+0.848, +0.997] |

Replication set, own margin pools and single directions reproduce the W2 peaks and attention shares (tables above) and the W6 selections. Person-name pairs ("Brett bought Kevin dinner … Brett felt very" → " generous" / " thankful"; 9 % of the main set) behave differently: their drop is carried less by the selected experts (W6 strata) and their attention share is 0.30 (Qwen3) and 0.43 (Mixtral); on the direct paths (W4 strata) the attention outputs write 0.26 of the drop for names vs 0.04 for objects (Qwen3), 0.49 of the drop for names vs 0.28 for objects (Mixtral) — the social items are the attention-leaning subset (W7, the role swap of the two names, is in ext7-controls).

### Reading

- **WinoGrande (option swap) is not IOI-like at the final position.** In both models the answer is written by the MoE outputs of
  one late band (Qwen3 L39–L44, Mixtral L19–L21) and, within it, by one positively specific expert (pattern A: Qwen3 L41E117,
  Mixtral L20E000, both replicated), with a larger drop-normalised MoE peak than CounterFact's. Attention is necessary (it is the
  only route by which the option's identity reaches the final token; all-attention = 1) but it is not a localised bottleneck at
  the final position in Qwen3, and only partly in Mixtral.
- **The two models move the option differently.** Qwen3 transports it gradually (the final-position residual restoration rises over
  some twenty layers, W3); its strongest single heads (L38H18, L38H21 positive, L38H16 negative; all three in the same GQA key/value group, heads 16–23)
  cancel within the layer, so no
  attention layer patch exceeds 0.03 of the drop. Mixtral has discrete transport steps at L13, L19 and L25 (the residual hand-off
  crosses half of the drop at L13), carried by a few heads that attend to the filled option and shift their attention to the first
  mention of the candidate the option names (e.g. L19H13: 0.12 vs 0.03 of its mass on that antecedent in the clean vs corrupted
  run) — a coreference-like read of the antecedent, the closest WinoGrande analogue of an IOI mover, but sub-additive and
  shared by several heads.
- **Three measurements, one direction.** Single-layer patches (attention share 0.16 / 0.34 vs CounterFact 0.54 / 0.58), joint
  patches (all MoE outputs restore about 0.8 of the drop; CounterFact Qwen3 ≈ 0.5 in ext8-addback's log, +6.2 of a ≈ 12.2 drop,
  final value in its section) and direct paths (MoE
  outputs write about 0.95 / 0.7 of the logit difference) all put WinoGrande further on the MoE side than factual recall; Mixtral
  is the more attention-involved of the two models on both tasks.
- **Experts are task-specific.** The CounterFact STR selections are idle on WinoGrande (3–12 of 256 directed cases routed) and the
  WinoGrande experts are new ones in the same late band; they carry person-name pairs much less (Qwen3 not at all, W6 strata),
  and those social pairs are also the attention-leaning subset (W4 strata: direct attention path 0.26 vs 0.04 of the drop in Qwen3,
  0.49 vs 0.28 in Mixtral, names vs objects; 23 name pairs only). Expert-level
  localisation is a property of the late read-out of a task, not of a model-wide store.

### Caveats

- The STR corruption swaps the filled option only (decision (a)); the role swap of the two candidates' earlier mentions (W7, Z7)
  and IOI (W8) are run by ext7-controls with these runners. The 256 main pairs are mostly physical items (9 % names) because the
  shared margin pool requires all three protocols to solve the pair.
- Margins are relative to the twin's trigger; the clean top-1 is the trigger in only 40–48 % of directed cases (strata
  `top1_both`).
- `assoc` pairs (12 % / 10 %) are solvable from the local context alone; they are kept and reported as a stratum.
- Per-head patches are small (bf16 noise floor ≈ 0.06 logit per row on OLMoE); detections use both splits (|z| ≥ 2 on discovery
  and validation) next to pair-bootstrap CIs.
- moetrace/engine.py was replaced by ext8's verified version at 07:36Z (additive; regression identical). The Qwen3 W2 sweep ran
  before, all other passes after.
- Mixtral without BOS (the paper's protocol) is not run (decision (e)); the case set keeps it possible on identical pairs.
- The direct split uses ext7-controls' shared implementation (fp32 final norm on the bf16 residuals; its Δ_clean differs from the
  engine's by ≤ 0.14 logit); ext7-wino's own implementation (`scripts/ext7_wino_dla.py`, bf16 norm as the engine) agrees to ≤ 0.002
  of the drop (last table of W4).
- All-attention sanity is 0.99 rather than exactly 1 in Qwen3 (the MoE of the final position is recomputed inside the wavefront
  row in a different batch, bf16), exactly 1 in Mixtral and OLMoE.

### Files

- Code: `moetrace/ext7_pairs.py` (generic STR-pair runner helpers, pair-level statistics, `analysis.ModelData` adapter),
  `scripts/ext7_wino_{sweep,expert,grid,heads,joint,dla,verify,analyze,text}.py`, chains `scripts/ext7_wino_chain{1,2,3,4}.sh` (chain 1 died at the Mixtral sweep with a CUDA OOM, chain 2 resumed with `--wf-chunk 2048`).
- Runs: `results/wino_qwen3_str/`, `results/wino_mixtral_bos_str/` (`str_sweep_rows`, `str_sweep_routing`, `sweep_cases`,
  `str_expert_rows` (ext6 schema; used by ext8 for add-back), `str_grid_w{1,5}_rows`, `head_rows`, `head_attn_final.npz`,
  `head_positions`, `joint_rows`, `direct_split` (shared W4 split), `dla_rows`, `dla_cases`, `case_sets.json` (families), `run_meta.json`).
- Verification: `results/verify_ext7_wino_olmoe.json`. Numbers: `results/ext7_wino_summary.json`. Tables `results/tables/ext7_wino_*`,
  figures `results/figures/ext7_wino_*`.
