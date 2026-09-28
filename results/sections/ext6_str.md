**Summary.** Zhang & Nanda (2024, arXiv:2309.16042) recommend symmetric token replacement (STR) over Gaussian noising (GN) as the corruption for activation patching, because GN can put the model off-distribution and inflate localisation. The paper (and our reproduction) uses GN on the subject embeddings. We re-ran the paper's full two-stage procedure with STR: the subject is replaced by another CounterFact subject of the same relation whose true object is the case's foil (same template, identical token positions), so the paper's metric Δ = logit(true) − logit(foil) becomes Zhang & Nanda's logit difference LD(r, r′) with r′ the corrupted prompt's answer. On 215, 212, 213 of 256 paper cases (Qwen3, Mixtral no BOS, Mixtral BOS) with up to five known donor facts per case, **every selection of the paper survives**: 
Qwen3 selects L44 and L44E069 (validation Spec +0.953 [+0.673, +1.276] logits; 0.082 [0.058, 0.108] of the drop vs 0.075 [0.048, 0.106] under GN), with L42E115 as the second locus; Mixtral under the paper's no-BOS protocol selects L19 and L19E006, whose Spec is again negative (-0.286 [-0.406, -0.168]) while the clean top-2 coalition recovers the block (+0.796 [+0.627, +0.964] vs +0.818 [+0.644, +0.992]). 
STR roughly doubles the drop (the corrupted run now prefers the foil instead of being indifferent), the STR and GN layer curves correlate at r = 0.99, 0.94, 0.94, and the drop-normalised block rescue at the selected layer is 0.177, 0.077, 0.085 under STR vs 0.167, 0.089, 0.113 under GN: no GN inflation in Qwen3, a mild one (13–25 % lower under STR) in Mixtral. With BOS, Mixtral's flat L19–L21 band makes the layer argmax move to L21 (discovery gap to L19 +0.04), where the selected L21E001 is positively specific; L19E002 keeps a positive Spec (+0.302 [+0.099, +0.547]). The single-donor sensitivity run reproduces the selections of Qwen3, Mixtral no BOS; it differs only where the layer argmax is a tie inside a flat band: Mixtral BOS: donor mean L21E001, first donor L19E002 (discovery L21 +1.13 vs L19 +1.13 in the first-donor run).

### Why and how

Zhang & Nanda compare GN (the ROME/causal-tracing corruption: N(0, (3σ)²) added to the subject embeddings) with STR (the key tokens are swapped for tokens of the same kind so that the corrupted prompt is an ordinary in-distribution prompt with its own answer r′), and logit difference (normalised by LD_clean − LD_corrupt) with probability and KL. Their recommendations: STR whenever possible, logit difference rather than probability, single-layer before sliding-window patching, and trying several corruption sites. The paper follows the metric and the single-layer patch but uses GN; with GN the foil is not the corrupted run's answer, so Δ_noised ≈ 0 ("indifferent") rather than negative ("prefers the other fact").

Construction (`moetrace/ext6_str.py`). For a case with template t, subject s, true object o and foil o_f (CounterFact `target_new`), a donor is any other CounterFact subject s′ with the same relation whose `target_true` is o_f, such that t.format(s′) has the same length, the subject at the same token positions, identical template tokens and the same single-token continuation ids for o and o_f. The corrupted run is t.format(s′) as a plain prefill row; every patch is the paper's (final-position MoE output, expert update δ_e = c_e(clean) − c_e(corrupt), coalitions) with the donor run as parent. User decisions (2026-09-28): the paper's case IDs and discovery/validation split, restricted to cases with a donor, recurrence threshold = half of the retained discovery cases; a donor qualifies when the model knows its fact, logit(o_f) − logit(o) ≥ 1.0 on the donor prompt (mirror of the clean-margin filter); up to five qualifying donors per case, taken in a fixed random order (`random.Random(2000 + case_id)`), per-case values = donor means, with the first donor alone as the sensitivity run; protocols Qwen3, Mixtral without BOS (the paper's) and Mixtral with BOS. GN baselines are the existing runs (`*_bos_alllayers`, `mixtral_nobos_alllayers`: base sweep + all-layer expert pass) restricted to exactly the same cases and split, and on the full paper set.

Verification (`scripts/ext6_str_verify.py`, `results/verify_ext6_str_olmoe.json`): on OLMoE, 12 (case, donor) pairs against transformers hooks: prefill Δ max diff 0.125 (one bf16 ulp); layer patch max 0.31, mean |diff| 0.052, 98 % within 0.25, rescue r = 0.994; literal expert patches (111 rows) max 0.19, r = 0.992; identity 0.125. The GN layer patch on the same model agrees with transformers to max 1.12, r = 0.983, so STR adds no new numerical error (it has no noise injection to amplify bf16 differences).

**STR construction and descriptors (same cases for STR and GN)**

| Model / protocol | Symmetric candidates | Cases with a candidate | Cases kept (disc/val) | Qualify rate | Donor rows | Donor top-1 = foil | Mean Δ clean | Mean Δ corrupt (STR) | Mean drop STR / GN | r(val curve STR, GN) | Per-case r at paper layer | Donor dispersion at paper layer (median SD) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 3883 | 221/256 | 215 (107/108) | 0.86 | 852 (4.0/case) | 0.28 | +5.79 | -6.39 | +12.18 / +5.69 | 0.987 | 0.77 | 0.42 (|mean| 1.68); sign agreement 0.95 |
| Mixtral-8x7B, no BOS (paper protocol) | 3397 | 218/256 | 212 (106/106) | 0.88 | 848 (4.0/case) | 0.26 | +5.43 | -5.09 | +10.52 / +4.72 | 0.937 | 0.60 | 0.34 (|mean| 0.51); sign agreement 0.89 |
| Mixtral-8x7B, BOS (tokenizer default) | 3397 | 218/256 | 213 (107/106) | 0.89 | 858 (4.0/case) | 0.30 | +6.57 | -6.25 | +12.82 / +5.03 | 0.935 | 0.67 | 0.42 (|mean| 0.85); sign agreement 0.90 |

Donor dispersion = median over validation cases with ≥ 2 donors of the SD of the block rescue at the paper's layer across the case's donors (|mean| = median absolute case mean); sign agreement = mean fraction of a case's donors whose rescue has the majority sign.

### Layer level

**Layer-level tracing under STR vs GN (validation; normalised = mean rescue / mean drop, paired bootstrap)**

| Model / protocol | Corruption | n disc/val | L* (disc) | Val rescue at L* | Val argmax (2nd, gap) | Val rescue at paper layer | Mean drop | Normalised at L* | Normalised at paper layer | Discovery top 5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | STR (donor mean) | 107/108 | L44 | +2.063 [+1.763, +2.390] | L44 (2nd L43, gap +0.807) | +2.063 [+1.763, +2.390] | +12.18 | 0.177 [0.157, 0.200] | 0.177 [0.157, 0.200] | L44 +2.26, L43 +1.43, L42 +1.14, L40 +0.93, L41 +0.45 |
| Qwen3-30B-A3B-Base | STR (first donor) | 107/108 | L44 | +2.048 [+1.741, +2.376] | L44 (2nd L43, gap +0.872) | +2.048 [+1.741, +2.376] | +12.19 | 0.178 [0.157, 0.199] | 0.178 [0.157, 0.199] | L44 +2.31, L43 +1.50, L42 +1.20, L40 +0.95, L41 +0.49 |
| Qwen3-30B-A3B-Base | GN (same cases) | 107/108 | L44 | +0.916 [+0.729, +1.116] | L44 (2nd L43, gap +0.275) | +0.916 [+0.729, +1.116] | +5.69 | 0.167 [0.140, 0.196] | 0.167 [0.140, 0.196] | L44 +0.86, L43 +0.67, L42 +0.58, L40 +0.41, L41 +0.24 |
| Qwen3-30B-A3B-Base | GN (paper set) | 128/128 | L44 | +0.941 [+0.778, +1.124] | L44 (2nd L42, gap +0.316) | +0.941 [+0.778, +1.124] | +5.92 | 0.167 [0.143, 0.193] | 0.167 [0.143, 0.193] | L44 +0.97, L43 +0.63, L42 +0.57, L40 +0.46, L41 +0.24 |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | 106/106 | L19 | +0.819 [+0.645, +0.993] | L21 (2nd L19, gap +0.125) | +0.819 [+0.645, +0.993] | +10.52 | 0.077 [0.062, 0.092] | 0.077 [0.062, 0.092] | L19 +0.81, L20 +0.75, L21 +0.69, L22 +0.50, L25 +0.37 |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | 106/106 | L19 | +0.836 [+0.657, +1.015] | L21 (2nd L19, gap +0.093) | +0.836 [+0.657, +1.015] | +10.51 | 0.079 [0.064, 0.094] | 0.079 [0.064, 0.094] | L19 +0.75, L20 +0.74, L21 +0.67, L22 +0.52, L18 +0.36 |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | 106/106 | L19 | +0.429 [+0.287, +0.571] | L21 (2nd L19, gap +0.101) | +0.429 [+0.287, +0.571] | +4.72 | 0.089 [0.062, 0.116] | 0.089 [0.062, 0.116] | L19 +0.51, L21 +0.45, L20 +0.34, L18 +0.30, L22 +0.28 |
| Mixtral-8x7B, no BOS (paper protocol) | GN (paper set) | 128/128 | L19 | +0.446 [+0.318, +0.569] | L21 (2nd L19, gap +0.086) | +0.446 [+0.318, +0.569] | +4.80 | 0.091 [0.066, 0.114] | 0.091 [0.066, 0.114] | L19 +0.44, L21 +0.40, L20 +0.30, L22 +0.28, L18 +0.27 |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | 107/106 | L21 | +1.102 [+0.916, +1.289] | L19 (2nd L21, gap +0.010) | +1.111 [+0.877, +1.381] | +12.82 | 0.085 [0.071, 0.098] | 0.086 [0.069, 0.105] | L21 +1.11, L20 +1.07, L19 +1.07, L22 +0.67, L23 +0.55 |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | 107/106 | L19 | +1.021 [+0.795, +1.272] | L21 (2nd L19, gap +0.063) | +1.021 [+0.795, +1.272] | +12.77 | 0.079 [0.063, 0.097] | 0.079 [0.063, 0.097] | L19 +1.13, L21 +1.13, L20 +1.07, L22 +0.68, L23 +0.60 |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | 107/106 | L19 | +0.565 [+0.434, +0.700] | L19 (2nd L20, gap +0.080) | +0.565 [+0.434, +0.700] | +5.03 | 0.113 [0.092, 0.135] | 0.113 [0.092, 0.135] | L19 +0.67, L21 +0.54, L20 +0.50, L22 +0.30, L18 +0.30 |
| Mixtral-8x7B, BOS (tokenizer default) | GN (paper set) | 128/128 | L19 | +0.571 [+0.461, +0.692] | L19 (2nd L20, gap +0.074) | +0.571 [+0.461, +0.692] | +4.99 | 0.115 [0.096, 0.135] | 0.115 [0.096, 0.135] | L19 +0.62, L21 +0.48, L20 +0.45, L22 +0.31, L18 +0.29 |

![STR vs GN layer curves: raw (top) and normalised by the mean validation drop (bottom)](figures/ext6_str_layers.png)

- Qwen3: STR L*44 (val +2.063 [+1.763, +2.390]; validation argmax L44, 2nd L43, gap +0.81) vs GN L*44 (val +0.916 [+0.729, +1.116]; validation argmax L44, gap +0.27); per-case r at the paper layer 0.77
- Mixtral no BOS: STR L*19 (val +0.819 [+0.645, +0.993]; validation argmax L21, 2nd L19, gap +0.13) vs GN L*19 (val +0.429 [+0.287, +0.571]; validation argmax L21, gap +0.10); per-case r at the paper layer 0.60
- Mixtral BOS: STR L*21 (val +1.102 [+0.916, +1.289]; validation argmax L19, 2nd L21, gap +0.01) vs GN L*19 (val +0.565 [+0.434, +0.700]; validation argmax L19, gap +0.08); per-case r at the paper layer 0.67

Raw rescues are about twice as large under STR because the drop is (mean Δ_clean +5.4 to +6.6; Δ_corrupt -5.1 to -6.4 under STR vs Δ_noised +0.1 to +1.5 under GN). Normalised by the drop, the layer effect is the same in Qwen3 and 13–25 % smaller under STR in Mixtral — the direction Zhang & Nanda report for GPT-2 XL (GN peaks 2–5× STR's), but far weaker. The Mixtral no-BOS and BOS curves are flat over L19–L21 under both corruptions, so the layer argmax is not a stable fingerprint there (the paper already noted the small gap, Table 6).

### Expert level

**Recurrence-first expert selection at L* and at the paper's layer (validation rescue and Spec)**

| Model / protocol | Corruption | Layer | Selected | Disc. active | Candidates | Val rescue | Spec | Spec / drop | Clean top-k coalition | Layer (same pass) | Top candidates (disc. all-case, active) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | STR (donor mean) | L42 | L42E115 | 104/107 | 5 (>= 53) | +0.838 [+0.687, +0.998] | +0.782 [+0.624, +0.954] | 0.067 [0.054, 0.081] | +1.183 [+1.010, +1.366] | +1.197 [+1.017, +1.390] | E115 +0.85 (104), E055 +0.02 (54), E101 +0.01 (80), E023 +0.01 (62) |
| Qwen3-30B-A3B-Base | STR (donor mean) | L44 | L44E069 | 94/107 | 6 (>= 53) | +1.071 [+0.802, +1.387] | +0.953 [+0.673, +1.276] | 0.082 [0.058, 0.108] | +1.970 [+1.673, +2.282] | +2.066 [+1.764, +2.392] | E069 +1.12 (94), E006 +0.25 (55), E027 +0.05 (62), E054 +0.02 (90) |
| Qwen3-30B-A3B-Base | STR (first donor) | L42 | L42E115 | 104/107 | 5 (>= 53) | +0.803 [+0.652, +0.966] | +0.744 [+0.584, +0.913] | 0.065 [0.052, 0.078] | +1.138 [+0.954, +1.331] | +1.163 [+0.973, +1.362] | E115 +0.85 (104), E055 +0.04 (54), E023 +0.02 (62), E101 +0.01 (80) |
| Qwen3-30B-A3B-Base | STR (first donor) | L44 | L44E069 | 94/107 | 6 (>= 53) | +1.113 [+0.848, +1.417] | +0.997 [+0.722, +1.309] | 0.087 [0.063, 0.111] | +1.969 [+1.670, +2.289] | +2.043 [+1.737, +2.373] | E069 +1.19 (94), E006 +0.24 (55), E027 +0.04 (62), E054 +0.03 (90) |
| Qwen3-30B-A3B-Base | GN (same cases) | L42 | L42E115 | 105/107 | 5 (>= 53) | +0.449 [+0.358, +0.550] | +0.419 [+0.324, +0.521] | 0.076 [0.060, 0.094] | +0.642 [+0.535, +0.753] | +0.635 [+0.527, +0.749] | E115 +0.43 (105), E055 +0.02 (53), E101 +0.01 (79), E065 +0.01 (62) |
| Qwen3-30B-A3B-Base | GN (same cases) | L44 | L44E069 | 96/107 | 6 (>= 53) | +0.472 [+0.314, +0.655] | +0.413 [+0.253, +0.595] | 0.075 [0.048, 0.106] | +0.887 [+0.702, +1.089] | +0.917 [+0.732, +1.116] | E069 +0.42 (96), E006 +0.09 (57), E027 +0.02 (63), E054 +0.01 (90) |
| Qwen3-30B-A3B-Base | GN (paper set) | L42 | L42E115 | 126/128 | 5 (>= 64) | +0.447 [+0.363, +0.537] | +0.423 [+0.339, +0.510] | 0.075 [0.061, 0.091] | +0.622 [+0.520, +0.724] | +0.621 [+0.517, +0.728] | E115 +0.46 (126), E055 +0.01 (65), E101 +0.01 (98), E023 +0.00 (70) |
| Qwen3-30B-A3B-Base | GN (paper set) | L44 | L44E069 | 114/128 | 5 (>= 64) | +0.499 [+0.357, +0.659] | +0.443 [+0.302, +0.603] | 0.079 [0.055, 0.104] | +0.916 [+0.756, +1.097] | +0.941 [+0.779, +1.124] | E069 +0.48 (114), E027 +0.03 (73), E054 +0.02 (111), E071 +0.01 (101) |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | L18 | L18E001 | 66/106 | 2 (>= 53) | +0.210 [+0.144, +0.279] | +0.151 [+0.083, +0.223] | 0.014 [0.008, 0.020] | +0.317 [+0.231, +0.403] | +0.328 [+0.242, +0.413] | E001 +0.21 (66), E006 +0.02 (64) |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | L19 | L19E006 | 72/106 | 1 (>= 53) | +0.127 [+0.078, +0.181] | -0.286 [-0.406, -0.168] | -0.027 [-0.038, -0.016] | +0.796 [+0.627, +0.964] | +0.818 [+0.644, +0.992] | E006 +0.18 (72) |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | L18 | L18E001 | 66/106 | 2 (>= 53) | +0.212 [+0.131, +0.295] | +0.140 [+0.057, +0.224] | 0.013 [0.005, 0.021] | +0.307 [+0.209, +0.406] | +0.322 [+0.221, +0.423] | E001 +0.23 (66), E006 +0.01 (64) |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | L19 | L19E006 | 72/106 | 1 (>= 53) | +0.126 [+0.059, +0.203] | -0.296 [-0.420, -0.169] | -0.028 [-0.039, -0.016] | +0.820 [+0.645, +0.991] | +0.842 [+0.664, +1.019] | E006 +0.18 (72) |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | L18 | L18E001 | 66/106 | 2 (>= 53) | +0.134 [+0.066, +0.212] | +0.094 [+0.025, +0.169] | 0.020 [0.005, 0.034] | +0.173 [+0.085, +0.266] | +0.146 [+0.047, +0.251] | E001 +0.16 (66), E006 +0.03 (64) |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | L19 | L19E006 | 72/106 | 1 (>= 53) | +0.054 [-0.026, +0.137] | -0.169 [-0.274, -0.064] | -0.035 [-0.056, -0.014] | +0.426 [+0.292, +0.563] | +0.407 [+0.270, +0.546] | E006 +0.13 (72) |
| Mixtral-8x7B, no BOS (paper protocol) | GN (paper set) | L18 | L18E001 | 76/128 | 2 (>= 64) | +0.139 [+0.081, +0.205] | +0.098 [+0.040, +0.162] | 0.020 [0.008, 0.033] | +0.199 [+0.118, +0.284] | +0.187 [+0.095, +0.284] | E001 +0.16 (76), E006 +0.03 (80) |
| Mixtral-8x7B, no BOS (paper protocol) | GN (paper set) | L19 | L19E006 | 91/128 | 1 (>= 64) | +0.063 [-0.009, +0.134] | -0.159 [-0.252, -0.065] | -0.032 [-0.051, -0.013] | +0.442 [+0.321, +0.561] | +0.427 [+0.305, +0.547] | E006 +0.07 (91) |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | L18 | L18E001 | 81/107 | 2 (>= 53) | +0.373 [+0.277, +0.474] | +0.289 [+0.191, +0.387] | 0.022 [0.015, 0.029] | +0.464 [+0.351, +0.578] | +0.471 [+0.358, +0.583] | E001 +0.27 (81), E006 +0.04 (60) |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | L19 | L19E002 | 61/107 | 2 (>= 53) | +0.597 [+0.397, +0.842] | +0.302 [+0.099, +0.547] | 0.023 [0.008, 0.041] | +1.086 [+0.853, +1.357] | +1.106 [+0.872, +1.375] | E002 +0.56 (61), E006 +0.16 (58) |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | L21 | L21E001 | 76/107 | 1 (>= 53) | +0.642 [+0.483, +0.802] | +0.396 [+0.232, +0.567] | 0.031 [0.018, 0.043] | +1.064 [+0.889, +1.238] | +1.099 [+0.914, +1.283] | E001 +0.71 (76) |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | L18 | L18E001 | 81/107 | 2 (>= 53) | +0.370 [+0.252, +0.486] | +0.292 [+0.175, +0.407] | 0.023 [0.014, 0.031] | +0.481 [+0.338, +0.615] | +0.484 [+0.341, +0.619] | E001 +0.31 (81), E006 +0.03 (60) |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | L19 | L19E002 | 61/107 | 2 (>= 53) | +0.551 [+0.358, +0.781] | +0.263 [+0.059, +0.495] | 0.020 [0.005, 0.038] | +1.016 [+0.789, +1.267] | +1.022 [+0.799, +1.274] | E002 +0.59 (61), E006 +0.15 (58) |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | L18 | L18E001 | 81/107 | 2 (>= 53) | +0.225 [+0.148, +0.306] | +0.175 [+0.097, +0.257] | 0.035 [0.020, 0.050] | +0.282 [+0.193, +0.374] | +0.277 [+0.187, +0.371] | E001 +0.20 (81), E006 +0.04 (60) |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | L19 | L19E002 | 61/107 | 2 (>= 53) | +0.371 [+0.259, +0.494] | +0.212 [+0.100, +0.328] | 0.042 [0.021, 0.064] | +0.554 [+0.425, +0.688] | +0.575 [+0.445, +0.711] | E002 +0.39 (61), E006 +0.12 (58) |
| Mixtral-8x7B, BOS (tokenizer default) | GN (paper set) | L18 | L18E001 | 98/128 | 2 (>= 64) | +0.244 [+0.177, +0.320] | +0.182 [+0.111, +0.261] | 0.037 [0.023, 0.051] | +0.319 [+0.239, +0.406] | +0.315 [+0.234, +0.403] | E001 +0.19 (98), E006 +0.04 (70) |
| Mixtral-8x7B, BOS (tokenizer default) | GN (paper set) | L19 | L19E002 | 76/128 | 2 (>= 64) | +0.363 [+0.267, +0.471] | +0.192 [+0.094, +0.296] | 0.039 [0.020, 0.058] | +0.559 [+0.451, +0.679] | +0.580 [+0.468, +0.700] | E002 +0.38 (76), E006 +0.10 (71) |

**Fixed-hypothesis validation of the experts named in the earlier directions**

| Model / protocol | Corruption | Expert | Disc. active | Val active | Val rescue | Spec | Rescue / drop | Spec / drop |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | STR (donor mean) | L44E069 | 94/107 | 96 | +1.071 [+0.802, +1.387] | +0.953 [+0.673, +1.276] | 0.092 [0.070, 0.117] | 0.082 [0.058, 0.108] |
| Qwen3-30B-A3B-Base | STR (donor mean) | L42E115 | 104/107 | 102 | +0.838 [+0.687, +0.998] | +0.782 [+0.624, +0.954] | 0.072 [0.059, 0.086] | 0.067 [0.054, 0.081] |
| Qwen3-30B-A3B-Base | STR (first donor) | L44E069 | 94/107 | 96 | +1.113 [+0.848, +1.417] | +0.997 [+0.722, +1.309] | 0.097 [0.075, 0.121] | 0.087 [0.063, 0.111] |
| Qwen3-30B-A3B-Base | STR (first donor) | L42E115 | 104/107 | 102 | +0.803 [+0.652, +0.966] | +0.744 [+0.584, +0.913] | 0.070 [0.058, 0.083] | 0.065 [0.052, 0.078] |
| Qwen3-30B-A3B-Base | GN (same cases) | L44E069 | 96/107 | 96 | +0.472 [+0.314, +0.655] | +0.413 [+0.253, +0.595] | 0.086 [0.060, 0.116] | 0.075 [0.048, 0.106] |
| Qwen3-30B-A3B-Base | GN (same cases) | L42E115 | 105/107 | 103 | +0.449 [+0.358, +0.550] | +0.419 [+0.324, +0.521] | 0.082 [0.066, 0.099] | 0.076 [0.060, 0.094] |
| Qwen3-30B-A3B-Base | GN (paper set) | L44E069 | 114/128 | 116 | +0.499 [+0.357, +0.659] | +0.443 [+0.302, +0.603] | 0.089 [0.065, 0.114] | 0.079 [0.055, 0.104] |
| Qwen3-30B-A3B-Base | GN (paper set) | L42E115 | 126/128 | 123 | +0.447 [+0.363, +0.537] | +0.423 [+0.339, +0.510] | 0.080 [0.066, 0.095] | 0.075 [0.061, 0.091] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | L19E006 | 72/106 | 72 | +0.127 [+0.078, +0.181] | -0.286 [-0.406, -0.168] | 0.012 [0.007, 0.017] | -0.027 [-0.038, -0.016] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | L19E002 | 50/106 | 56 | +0.383 [+0.262, +0.506] | +0.140 [+0.015, +0.269] | 0.036 [0.026, 0.046] | 0.013 [0.001, 0.025] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | L18E001 | 66/106 | 62 | +0.210 [+0.144, +0.279] | +0.151 [+0.083, +0.223] | 0.020 [0.014, 0.026] | 0.014 [0.008, 0.020] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | L19E006 | 72/106 | 72 | +0.126 [+0.059, +0.203] | -0.296 [-0.420, -0.169] | 0.012 [0.006, 0.019] | -0.028 [-0.039, -0.016] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | L19E002 | 50/106 | 56 | +0.384 [+0.266, +0.508] | +0.158 [+0.029, +0.292] | 0.036 [0.026, 0.046] | 0.015 [0.003, 0.027] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | L18E001 | 66/106 | 62 | +0.212 [+0.131, +0.295] | +0.140 [+0.057, +0.224] | 0.020 [0.013, 0.027] | 0.013 [0.005, 0.021] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | L19E006 | 72/106 | 70 | +0.054 [-0.026, +0.137] | -0.169 [-0.274, -0.064] | 0.011 [-0.006, 0.029] | -0.035 [-0.056, -0.014] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | L19E002 | 50/106 | 56 | +0.218 [+0.127, +0.313] | +0.079 [-0.037, +0.192] | 0.045 [0.027, 0.063] | 0.016 [-0.008, 0.040] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | L18E001 | 66/106 | 61 | +0.134 [+0.066, +0.212] | +0.094 [+0.025, +0.169] | 0.028 [0.014, 0.042] | 0.020 [0.005, 0.034] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (paper set) | L19E006 | 91/128 | 83 | +0.063 [-0.009, +0.134] | -0.159 [-0.252, -0.065] | 0.013 [-0.002, 0.027] | -0.032 [-0.051, -0.013] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (paper set) | L19E002 | 59/128 | 71 | +0.218 [+0.141, +0.304] | +0.066 [-0.029, +0.161] | 0.044 [0.029, 0.061] | 0.013 [-0.006, 0.033] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (paper set) | L18E001 | 76/128 | 76 | +0.139 [+0.081, +0.205] | +0.098 [+0.040, +0.162] | 0.028 [0.017, 0.041] | 0.020 [0.008, 0.033] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | L19E002 | 61/107 | 68 | +0.597 [+0.397, +0.842] | +0.302 [+0.099, +0.547] | 0.046 [0.032, 0.063] | 0.023 [0.008, 0.041] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | L19E006 | 58/107 | 59 | +0.164 [+0.108, +0.224] | -0.416 [-0.650, -0.233] | 0.013 [0.008, 0.018] | -0.032 [-0.049, -0.018] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | L18E001 | 81/107 | 83 | +0.373 [+0.277, +0.474] | +0.289 [+0.191, +0.387] | 0.029 [0.021, 0.036] | 0.022 [0.015, 0.029] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | L19E002 | 61/107 | 68 | +0.551 [+0.358, +0.781] | +0.263 [+0.059, +0.495] | 0.042 [0.028, 0.060] | 0.020 [0.005, 0.038] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | L19E006 | 58/107 | 59 | +0.170 [+0.104, +0.239] | -0.382 [-0.609, -0.200] | 0.013 [0.008, 0.019] | -0.029 [-0.046, -0.016] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | L18E001 | 81/107 | 83 | +0.370 [+0.252, +0.486] | +0.292 [+0.175, +0.407] | 0.029 [0.020, 0.037] | 0.023 [0.014, 0.031] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | L19E002 | 61/107 | 68 | +0.371 [+0.259, +0.494] | +0.212 [+0.100, +0.328] | 0.074 [0.055, 0.094] | 0.042 [0.021, 0.064] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | L19E006 | 58/107 | 59 | +0.085 [+0.050, +0.123] | -0.220 [-0.323, -0.124] | 0.017 [0.010, 0.025] | -0.044 [-0.063, -0.026] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | L18E001 | 81/107 | 83 | +0.225 [+0.148, +0.306] | +0.175 [+0.097, +0.257] | 0.045 [0.031, 0.060] | 0.035 [0.020, 0.050] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (paper set) | L19E002 | 76/128 | 84 | +0.363 [+0.267, +0.471] | +0.192 [+0.094, +0.296] | 0.073 [0.056, 0.091] | 0.039 [0.020, 0.058] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (paper set) | L19E006 | 71/128 | 67 | +0.079 [+0.048, +0.110] | -0.246 [-0.341, -0.162] | 0.016 [0.010, 0.023] | -0.049 [-0.067, -0.033] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (paper set) | L18E001 | 98/128 | 103 | +0.244 [+0.177, +0.320] | +0.182 [+0.111, +0.261] | 0.049 [0.037, 0.062] | 0.037 [0.023, 0.051] |

Spec / drop and rescue / drop are population ratios (mean over validation cases / mean drop, paired bootstrap), the scale on which STR and GN are comparable.

- **Qwen3.** L44E069 is selected under STR as under GN (discovery active 94/107), with a larger and still specific effect: rescue +1.071 [+0.802, +1.387], Spec +0.953 [+0.673, +1.276]; per unit of drop 0.082 [0.058, 0.108] vs GN 0.075 [0.048, 0.106]. L42E115 is the L42 winner (Spec +0.782 [+0.624, +0.954]). The clean top-8 coalition carries 95% of the L44 block.
- **Mixtral, no BOS (paper protocol).** L19E006 is again the only recurrent L19 candidate (72/106) and again not specific: rescue +0.127 [+0.078, +0.181], Spec -0.286 [-0.406, -0.168] (GN same cases -0.169 [-0.274, -0.064]); the clean top-2 coalition +0.796 [+0.627, +0.964] recovers the block +0.818 [+0.644, +0.992]. L18E001 stays positively specific (+0.151 [+0.083, +0.223]); L19E002 is specific (+0.140 [+0.015, +0.269]) but below the recurrence gate (50/106), exactly the GN picture of Directions 1 and 3.
- **Mixtral, BOS.** At L19 the content expert E002 is selected with a positive Spec +0.302 [+0.099, +0.547] (GN +0.212 [+0.100, +0.328]); E006 is negatively specific (-0.416 [-0.650, -0.233]). Because the discovery argmax moves to L21, the two-stage rule now reports L21E001 (Spec +0.396 [+0.232, +0.567]); both are single, positive, specific experts (pattern A of Direction 2).

**Equal-norm active-pair check (Mixtral L19, paper Table 11 analogue).**

**Mixtral active-pair equal-norm check at L19 (paper Table 11 analogue): both patch vectors scaled to the smaller norm**

| Model / protocol | Corruption | Expert | n (anchor-active val) | Raw active-pair Spec | Selected, equal norm | Other active, equal norm | Equal-norm Spec |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | L19E006 | 72 | -0.142 [-0.275, -0.019] | +0.144 [+0.090, +0.202] | +0.151 [+0.088, +0.223] | -0.008 [-0.055, +0.039] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | L19E006 | 70 | -0.131 [-0.267, +0.002] | +0.074 [+0.027, +0.121] | +0.141 [+0.059, +0.221] | -0.066 [-0.139, +0.007] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | L19E002 | 56 | +0.436 [+0.247, +0.628] | +0.247 [+0.168, +0.328] | +0.249 [+0.164, +0.343] | -0.002 [-0.058, +0.052] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | L19E002 | 56 | +0.280 [+0.104, +0.459] | +0.226 [+0.147, +0.315] | +0.113 [+0.051, +0.177] | +0.114 [+0.035, +0.201] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | L19E002 | 68 | +0.640 [+0.366, +0.977] | +0.286 [+0.210, +0.371] | +0.236 [+0.167, +0.309] | +0.050 [-0.002, +0.106] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | L19E002 | 68 | +0.428 [+0.287, +0.586] | +0.182 [+0.116, +0.253] | +0.108 [+0.057, +0.160] | +0.074 [+0.026, +0.128] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | L19E006 | 59 | -0.322 [-0.714, -0.053] | +0.231 [+0.153, +0.323] | +0.210 [+0.129, +0.306] | +0.021 [-0.036, +0.075] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | L19E006 | 59 | -0.119 [-0.276, +0.008] | +0.095 [+0.049, +0.145] | +0.100 [+0.039, +0.168] | -0.004 [-0.054, +0.041] |

Under STR every L19 expert is indistinguishable from its co-active partner once both patch vectors have the same norm: E006 -0.008 [-0.055, +0.039] and E002 -0.002 [-0.058, +0.052] without BOS, E002 +0.050 [-0.002, +0.106] and E006 +0.021 [-0.036, +0.075] with BOS. Under GN on the same cases E002 kept a small direction advantage (+0.114 [+0.035, +0.201] / +0.074 [+0.026, +0.128]) and E006 a small deficit without BOS (-0.066 [-0.139, +0.007]; paper Table 11: −0.062 [−0.130, −0.003]). So in Mixtral the raw Spec differences at L19 (E006 negative, E002 positive) are, under STR, differences in how much each expert's update changes between the two facts, not in the direction of the update. The Qwen3 gate-matched / equal-norm control (Table 9) needs pair rows at L44, which the STR expert pass did not record.

### Joint layer × expert search

**Joint (layer, expert) search over the layers of the STR expert pass (top 5, recurrence gate = half of discovery)**

| Model / protocol | Corruption | Rank | (layer, expert) | Disc. active | Disc. all-case | Val rescue | Spec |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | STR (donor mean) | 1 | L44E069 | 94 | +1.117 | +1.071 [+0.802, +1.387] | +0.953 [+0.673, +1.276] |
| Qwen3-30B-A3B-Base | STR (donor mean) | 2 | L42E115 | 104 | +0.849 | +0.838 [+0.687, +0.998] | +0.782 [+0.624, +0.954] |
| Qwen3-30B-A3B-Base | STR (donor mean) | 3 | L43E046 | 72 | +0.395 | +0.238 [+0.114, +0.382] | +0.103 [-0.041, +0.258] |
| Qwen3-30B-A3B-Base | STR (donor mean) | 4 | L43E005 | 62 | +0.299 | +0.253 [+0.130, +0.414] | +0.180 [+0.044, +0.347] |
| Qwen3-30B-A3B-Base | STR (donor mean) | 5 | L40E127 | 76 | +0.262 | +0.215 [+0.143, +0.304] | +0.120 [+0.029, +0.219] |
| Qwen3-30B-A3B-Base | STR (first donor) | 1 | L44E069 | 94 | +1.192 | +1.113 [+0.848, +1.417] | +0.997 [+0.722, +1.309] |
| Qwen3-30B-A3B-Base | STR (first donor) | 2 | L42E115 | 104 | +0.851 | +0.803 [+0.652, +0.966] | +0.744 [+0.584, +0.913] |
| Qwen3-30B-A3B-Base | STR (first donor) | 3 | L43E046 | 72 | +0.425 | +0.281 [+0.148, +0.430] | +0.170 [+0.020, +0.325] |
| Qwen3-30B-A3B-Base | STR (first donor) | 4 | L41E001 | 94 | +0.270 | +0.203 [+0.137, +0.269] | +0.169 [+0.102, +0.235] |
| Qwen3-30B-A3B-Base | STR (first donor) | 5 | L40E127 | 76 | +0.268 | +0.215 [+0.142, +0.304] | +0.128 [+0.040, +0.230] |
| Qwen3-30B-A3B-Base | GN (same cases) | 1 | L42E115 | 105 | +0.430 | +0.449 [+0.358, +0.550] | +0.419 [+0.324, +0.521] |
| Qwen3-30B-A3B-Base | GN (same cases) | 2 | L44E069 | 96 | +0.420 | +0.472 [+0.314, +0.655] | +0.413 [+0.253, +0.595] |
| Qwen3-30B-A3B-Base | GN (same cases) | 3 | L43E046 | 73 | +0.161 | +0.070 [+0.027, +0.120] | -0.012 [-0.066, +0.045] |
| Qwen3-30B-A3B-Base | GN (same cases) | 4 | L43E005 | 62 | +0.157 | +0.157 [+0.077, +0.259] | +0.104 [+0.022, +0.208] |
| Qwen3-30B-A3B-Base | GN (same cases) | 5 | L40E127 | 76 | +0.150 | +0.133 [+0.082, +0.198] | +0.090 [+0.034, +0.157] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | 1 | L21E001 | 54 | +0.398 | +0.623 [+0.475, +0.778] | +0.419 [+0.271, +0.569] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | 2 | L20E005 | 55 | +0.238 | +0.247 [+0.170, +0.331] | -0.073 [-0.178, +0.033] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | 3 | L18E001 | 66 | +0.207 | +0.210 [+0.144, +0.279] | +0.151 [+0.083, +0.223] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | 4 | L19E006 | 72 | +0.179 | +0.127 [+0.078, +0.181] | -0.286 [-0.406, -0.168] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | 5 | L22E001 | 82 | +0.165 | +0.215 [+0.130, +0.308] | +0.032 [-0.090, +0.152] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | 1 | L21E001 | 54 | +0.376 | +0.626 [+0.480, +0.786] | +0.426 [+0.277, +0.581] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | 2 | L20E005 | 55 | +0.233 | +0.272 [+0.186, +0.361] | -0.064 [-0.177, +0.051] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | 3 | L18E001 | 66 | +0.225 | +0.212 [+0.131, +0.295] | +0.140 [+0.057, +0.224] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | 4 | L22E001 | 82 | +0.188 | +0.199 [+0.102, +0.302] | +0.023 [-0.103, +0.146] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | 5 | L19E006 | 72 | +0.176 | +0.126 [+0.059, +0.203] | -0.296 [-0.420, -0.169] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | 1 | L21E001 | 53 | +0.268 | +0.306 [+0.195, +0.422] | +0.180 [+0.075, +0.289] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | 2 | L18E001 | 66 | +0.163 | +0.134 [+0.066, +0.212] | +0.094 [+0.025, +0.169] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | 3 | L19E006 | 72 | +0.127 | +0.054 [-0.026, +0.137] | -0.169 [-0.274, -0.064] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | 4 | L22E001 | 81 | +0.125 | +0.087 [+0.020, +0.159] | +0.041 [-0.031, +0.115] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | 5 | L20E005 | 56 | +0.096 | +0.125 [+0.072, +0.187] | -0.076 [-0.157, +0.003] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | 1 | L21E001 | 76 | +0.709 | +0.642 [+0.483, +0.802] | +0.396 [+0.232, +0.567] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | 2 | L19E002 | 61 | +0.561 | +0.597 [+0.397, +0.842] | +0.302 [+0.099, +0.547] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | 3 | L20E005 | 62 | +0.348 | +0.316 [+0.224, +0.416] | -0.074 [-0.201, +0.061] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | 4 | L18E001 | 81 | +0.274 | +0.373 [+0.277, +0.474] | +0.289 [+0.191, +0.387] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | 5 | L22E001 | 78 | +0.259 | +0.400 [+0.285, +0.526] | +0.122 [-0.027, +0.271] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | 1 | L21E001 | 76 | +0.700 | +0.617 [+0.463, +0.772] | +0.401 [+0.241, +0.558] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | 2 | L19E002 | 61 | +0.588 | +0.551 [+0.358, +0.781] | +0.263 [+0.059, +0.495] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | 3 | L20E005 | 62 | +0.340 | +0.292 [+0.196, +0.395] | -0.070 [-0.195, +0.065] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | 4 | L18E001 | 81 | +0.306 | +0.370 [+0.252, +0.486] | +0.292 [+0.175, +0.407] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | 5 | L22E001 | 78 | +0.289 | +0.373 [+0.228, +0.524] | +0.096 [-0.067, +0.259] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | 1 | L19E002 | 61 | +0.388 | +0.371 [+0.259, +0.494] | +0.212 [+0.100, +0.328] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | 2 | L21E001 | 78 | +0.374 | +0.282 [+0.190, +0.380] | +0.139 [+0.047, +0.236] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | 3 | L18E001 | 81 | +0.204 | +0.225 [+0.148, +0.306] | +0.175 [+0.097, +0.257] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | 4 | L20E005 | 63 | +0.155 | +0.160 [+0.104, +0.223] | -0.047 [-0.121, +0.031] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | 5 | L22E001 | 78 | +0.136 | +0.167 [+0.101, +0.244] | +0.059 [-0.017, +0.142] |

Layers covered by the STR expert pass: Qwen3 48, Mixtral no BOS 32, Mixtral BOS 32 (all layers when 48 / 32); the GN column is restricted to the same layers. Joint top-1 (STR / GN, same cases): Qwen3 L44E069 / L42E115; Mixtral no BOS L21E001 / L21E001; Mixtral BOS L21E001 / L19E002. 
The recurrence gate is half of the retained discovery cases (53–54 instead of the paper's 64/128). Mixtral no BOS L21E001 is active in 59/128 discovery cases of the full paper set (gate 64), i.e. it enters the ranking only because the retained set is smaller — under the paper's gate on the full set, the GN joint winner is L18E001 (Direction 1), which STR also finds positively specific. Under STR the paper's two-stage choice is the joint top-1 in Qwen3 (under GN on the same cases L42E115 and L44E069 tie on discovery, Direction 1's second-locus result). Joint top-1 identical under STR and GN: Qwen3 no, Mixtral no BOS yes, Mixtral BOS no; where it differs (discovery all-case rescue of the two leaders under each corruption: Qwen3 STR L44E069 +1.12 vs L42E115 +0.85; GN L42E115 +0.43 vs L44E069 +0.42; Mixtral BOS STR L21E001 +0.71 vs L19E002 +0.56; GN L19E002 +0.39 vs L21E001 +0.37).

### Donor sensitivity

First donor only (one replacement per case, the analogue of the paper's single noise draw). The single-donor sensitivity run reproduces the selections of Qwen3, Mixtral no BOS; it differs only where the layer argmax is a tie inside a flat band: Mixtral BOS: donor mean L21E001, first donor L19E002 (discovery L21 +1.13 vs L19 +1.13 in the first-donor run). Across a case's donors the block rescue at the paper's layer varies with a median SD of 0.42, 0.34, 0.42 logits (median |case mean| 1.68, 0.51, 0.85), and 95 %, 89 %, 90 % of a case's donors agree on the sign: the per-case effect is mostly a property of the clean fact, not of which other fact replaces it.

### Reading

1. The paper's conclusions do not depend on its corruption. Under the corruption Zhang & Nanda recommend, Qwen3 still localises to L44 and to a positive specific expert L44E069, and Mixtral (paper protocol) still shows a validated mid-layer block whose single selected expert L19E006 is not specific while the routed coalition recovers it. The earlier extension findings also hold: L42E115 as Qwen3's second locus, L18E001 as a positive Mixtral expert, and the BOS dependence of Mixtral's L19 selection (E006 without BOS, E002 with). The one selection that changes is outside the paper's protocol: with BOS, STR ranks L21E001 above L19E002 in both the two-stage and the joint search, where GN had L19E002 first by a hair; both are positive, specific single experts inside the flat L19–L21 band, their validation intervals overlap, and the single-donor run returns L19E002 — a tie rather than a reversal.
2. GN does not inflate the localisation here the way it does in GPT-2 XL. Normalised block rescue is unchanged in Qwen3 and 13–25 % lower under STR in Mixtral; STR and GN layer curves correlate at r ≥ 0.94. A plausible reason is the patch site: the paper patches the final position late in the network, whereas Zhang & Nanda's GN/STR gap is at the last subject token in early MLPs, where GN's off-distribution embeddings act directly.
3. STR changes the meaning of the corrupted run, and that matters for interpretation more than for selection: the corrupted run prefers the foil (Δ_corrupt ≈ −6), so a rescue measures how much of the *switch between two facts* a component carries; components that compute the same thing for both facts (e.g. "the answer is a country") cancel by construction. Qwen3's E069 keeps its share of the drop under this stricter reading, Mixtral's experts lose some of theirs, and at equal norm Mixtral's L19 experts no longer differ from their co-active partner (the specificity that remains is a magnitude effect).
4. Protocol recommendation for this project: keep GN as the paper's protocol for comparability, report STR alongside it as the best-practice check, and report drop-normalised rescue for any comparison across corruptions.

Caveats. STR covers 83–84 % of the paper cases (cases without a same-relation subject whose true object is the foil, or none of equal token length, drop out), so all comparisons use the GN runs on exactly the same cases. Donors are CounterFact facts the model prefers by ≥ 1 logit, not facts it necessarily outputs (the foil is the donor prompt's top-1 in 26–30 % of selected donors, as the true object is for 29 % of clean prompts). The relaxed-filter and our own strict sets were not re-run under STR.

Files: `moetrace/ext6_str.py`; `scripts/ext6_str_{filter,sweep,expert,verify,analyze,text}.py`, `scripts/ext6_str_chain.sh`; runs `results/{qwen3_str,mixtral_nobos_str,mixtral_bos_str}` (`str_candidates.parquet` with every symmetric candidate and its Δ_donor, `str_sweep_rows.parquet` / `str_expert_rows.parquet` at donor level, `str_sweep_routing.parquet`, `sweep_cases.parquet`, `case_sets.json`, `run_meta.json`); tables `results/tables/ext6_str_*`; figure `results/figures/ext6_str_layers.*`; numbers `results/ext6_str_summary.json`.
