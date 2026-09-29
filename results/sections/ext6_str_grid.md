**Summary.** Zhang & Nanda (2024, Section 4.1 / Figure 4) extend the MLP patch from the last subject token to every position and plot layer × position heatmaps under STR, to test whether the last subject token is special and whether logit difference and probability agree. We ran the same grid for the paper's MoE-output patch with the STR corruption of Direction 6 (every selected donor; Qwen3 215 cases, Mixtral BOS 213 cases), single-layer patches at every position from the first subject token to the final token and every layer, and the 5-layer sliding window of their Figure 4. The picture is the two-site pattern of Meng et al.: Qwen3 early site at the last subject token (peak L0 +0.187 of the drop, 1.27 drops summed over layers) and late site at the final position (peak L44 +0.176, the paper's layer band); Mixtral BOS early site at the last subject token (peak L0 +0.304 of the drop, 1.60 drops summed over layers) and late site at the final position (peak L21 +0.086, the paper's layer band); the tokens between the subject and the final position carry almost nothing (≤ 0.007). Among subject tokens the last one dominates (Zhang & Nanda's last / middle ratio: Qwen3 logit difference 2.87× [2.40, 3.44], probability 2.65× [1.30, 5.95]; Mixtral BOS logit difference 4.79× [4.20, 5.48], probability 6.54× [3.40, 13.53]); with single-layer patches the two metrics draw the same map, unlike GPT-2 XL (1.22× vs 4.33×). With the 5-layer window of their Figure 4, probability does emphasise the last subject token more (Qwen3 logit difference 3.18× [2.75, 3.67], probability 3.99× [2.68, 6.57]; Mixtral BOS logit difference 5.04× [4.47, 5.70], probability 14.88× [9.74, 23.83]) and the window is super-additive only on the probability scale (joint / summed single-layer peak Qwen3 0.83–0.97 on the logit difference vs 3.53–3.71 on Δp; Mixtral BOS 0.60–1.09 on the logit difference vs 2.76–6.08 on Δp): Zhang & Nanda's two GPT-2 XL observations reappear exactly where a window and the probability metric are combined. The early site is where the corruption matters: compared with the GN runs of F4 at the same position and cases, Qwen3 GN peaks at L4 instead of L0 and its layer-summed rescue is 1.06× STR's; Mixtral BOS GN peaks at L4 instead of L0 and its layer-summed rescue is 1.77× STR's — Zhang & Nanda's GN inflation, which at the paper's final-position site (Direction 6) was absent or small.

### What was run

For every retained STR case (Direction 6: paper IDs and split, donors = same-relation subjects whose true object is the foil, up to five per case) and every position p from the first subject token to the final token, a clean prefill row and one row per donor record at p; per donor row and layer l one suffix wavefront row (`moetrace/ext5_subject.py`) carries the MoE output at p set to the clean one (window 1) or at layers l−2..l+2 clipped to the network (window 5, centred, the setting of Zhang & Nanda's Figure 4 and Meng et al.'s causal-tracing plots), and the model runs on from there. Positions before the first subject token are not patched: clean and donor prompts share that prefix, so the patch is exactly zero. Metrics at the final position: the paper's Δ (its rescue normalised by the drop Δ_clean − Δ_corrupt, the normalised logit difference of Zhang & Nanda) and Δp = p_patched(true) − p_corrupt(true) (their probability metric; full softmax). Aggregation as in ROME: donor mean per (case, position, layer), then mean over the positions of each token group of a case, then mean over the cases that have the group (one-token subjects have no first/middle token; the first subsequent token counts as "last token" when it is the final position).

Engine additions (backward compatible, in `moetrace/ext5_subject.py`): `SubjectSpawn.window` (joint restoration of the MoE output at p over consecutive layers) and `run_subject(metrics=True)`. The F4 verification re-run gives results identical to the saved copy in all 70 non-timing fields. New verification against transformers hooks on OLMoE (`results/verify_ext6_str_grid_olmoe.json`, 8 STR pairs, 50 (case, position) units, all layers): window 1 max |ΔΔ| 0.59, mean 0.077, 93 % within 0.25, rescue r = 0.991; window 5 max 0.50, r = 0.998; p(true) mean |diff| 0.0002 / 0.0005; null invariant 0.125 (the GN patch at p in F4: max 0.80, r = 0.952). Consistency: the grid's last-token column reproduces the final-position STR sweep of Direction 6 (Qwen3: curve r = 0.9999, max |diff| 0.015, per-case r = 0.985); (Mixtral BOS: curve r = 0.9999, max |diff| 0.015, per-case r = 0.992).

### Heatmaps

![Qwen3 STR layer x token-group heatmaps](figures/ext6_str_grid_qwen3_str.png)

![Mixtral BOS STR layer x token-group heatmaps](figures/ext6_str_grid_mixtral_bos_str.png)

![Single-layer layer curves per token group](figures/ext6_str_grid_curves.png)

**STR layer x position grid: peak of each token group's layer curve (mean over retained cases)**

| Model | Window | Metric | Token group | n cases | Peak layer | Peak value [95% CI] | Sum over layers |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 1 | LD / drop | first subject token | 213 | L0 | +0.079 [+0.062, +0.097] | +0.068 |
| Qwen3-30B-A3B-Base | 1 | LD / drop | middle subject tokens | 188 | L0 | +0.167 [+0.147, +0.188] | +0.426 |
| Qwen3-30B-A3B-Base | 1 | LD / drop | last subject token | 215 | L0 | +0.187 [+0.165, +0.211] | +1.267 |
| Qwen3-30B-A3B-Base | 1 | LD / drop | first subsequent token | 176 | L27 | +0.003 [+0.002, +0.005] | +0.029 |
| Qwen3-30B-A3B-Base | 1 | LD / drop | further tokens | 139 | L28 | +0.003 [+0.002, +0.005] | +0.043 |
| Qwen3-30B-A3B-Base | 1 | LD / drop | last token | 215 | L44 | +0.176 [+0.161, +0.192] | +0.578 |
| Qwen3-30B-A3B-Base | 1 | Δp | first subject token | 213 | L0 | +0.0060 [+0.0021, +0.0109] | +0.0059 |
| Qwen3-30B-A3B-Base | 1 | Δp | middle subject tokens | 188 | L2 | +0.0023 [+0.0006, +0.0047] | +0.0090 |
| Qwen3-30B-A3B-Base | 1 | Δp | last subject token | 215 | L3 | +0.0062 [+0.0023, +0.0115] | +0.0327 |
| Qwen3-30B-A3B-Base | 1 | Δp | first subsequent token | 176 | L24 | +0.0001 [-0.0000, +0.0002] | -0.0004 |
| Qwen3-30B-A3B-Base | 1 | Δp | further tokens | 139 | L28 | +0.0001 [+0.0000, +0.0002] | +0.0005 |
| Qwen3-30B-A3B-Base | 1 | Δp | last token | 215 | L44 | +0.0059 [+0.0034, +0.0090] | +0.0176 |
| Qwen3-30B-A3B-Base | 5 | LD / drop | first subject token | 213 | L0 | +0.096 [+0.077, +0.116] | +0.329 |
| Qwen3-30B-A3B-Base | 5 | LD / drop | middle subject tokens | 188 | L2 | +0.341 [+0.308, +0.378] | +1.757 |
| Qwen3-30B-A3B-Base | 5 | LD / drop | last subject token | 215 | L2 | +0.592 [+0.549, +0.635] | +5.906 |
| Qwen3-30B-A3B-Base | 5 | LD / drop | first subsequent token | 176 | L26 | +0.011 [+0.008, +0.014] | +0.223 |
| Qwen3-30B-A3B-Base | 5 | LD / drop | further tokens | 139 | L28 | +0.009 [+0.006, +0.013] | +0.181 |
| Qwen3-30B-A3B-Base | 5 | LD / drop | last token | 215 | L42 | +0.475 [+0.455, +0.495] | +3.092 |
| Qwen3-30B-A3B-Base | 5 | Δp | first subject token | 213 | L1 | +0.0071 [+0.0032, +0.0121] | +0.0219 |
| Qwen3-30B-A3B-Base | 5 | Δp | middle subject tokens | 188 | L1 | +0.0192 [+0.0113, +0.0286] | +0.0937 |
| Qwen3-30B-A3B-Base | 5 | Δp | last subject token | 215 | L2 | +0.0715 [+0.0524, +0.0921] | +0.5028 |
| Qwen3-30B-A3B-Base | 5 | Δp | first subsequent token | 176 | L26 | +0.0001 [+0.0000, +0.0002] | +0.0008 |
| Qwen3-30B-A3B-Base | 5 | Δp | further tokens | 139 | L32 | +0.0002 [+0.0000, +0.0004] | +0.0027 |
| Qwen3-30B-A3B-Base | 5 | Δp | last token | 215 | L42 | +0.0484 [+0.0352, +0.0630] | +0.1686 |
| Mixtral-8x7B, BOS | 1 | LD / drop | first subject token | 212 | L0 | +0.197 [+0.171, +0.224] | +0.254 |
| Mixtral-8x7B, BOS | 1 | LD / drop | middle subject tokens | 196 | L0 | +0.181 [+0.165, +0.198] | +0.327 |
| Mixtral-8x7B, BOS | 1 | LD / drop | last subject token | 213 | L0 | +0.304 [+0.279, +0.330] | +1.599 |
| Mixtral-8x7B, BOS | 1 | LD / drop | first subsequent token | 165 | L21 | +0.005 [+0.003, +0.006] | +0.049 |
| Mixtral-8x7B, BOS | 1 | LD / drop | further tokens | 139 | L19 | +0.007 [+0.005, +0.009] | +0.047 |
| Mixtral-8x7B, BOS | 1 | LD / drop | last token | 213 | L21 | +0.086 [+0.077, +0.095] | +0.500 |
| Mixtral-8x7B, BOS | 1 | Δp | first subject token | 212 | L0 | +0.0132 [+0.0061, +0.0225] | +0.0165 |
| Mixtral-8x7B, BOS | 1 | Δp | middle subject tokens | 196 | L0 | +0.0024 [+0.0015, +0.0035] | +0.0082 |
| Mixtral-8x7B, BOS | 1 | Δp | last subject token | 213 | L3 | +0.0156 [+0.0090, +0.0235] | +0.0601 |
| Mixtral-8x7B, BOS | 1 | Δp | first subsequent token | 165 | L6 | +0.0000 [-0.0000, +0.0001] | +0.0001 |
| Mixtral-8x7B, BOS | 1 | Δp | further tokens | 139 | L20 | +0.0001 [+0.0000, +0.0002] | +0.0004 |
| Mixtral-8x7B, BOS | 1 | Δp | last token | 213 | L20 | +0.0032 [+0.0017, +0.0055] | +0.0145 |
| Mixtral-8x7B, BOS | 5 | LD / drop | first subject token | 212 | L2 | +0.211 [+0.184, +0.239] | +1.024 |
| Mixtral-8x7B, BOS | 5 | LD / drop | middle subject tokens | 196 | L2 | +0.239 [+0.218, +0.260] | +1.362 |
| Mixtral-8x7B, BOS | 5 | LD / drop | last subject token | 213 | L4 | +0.663 [+0.622, +0.704] | +6.933 |
| Mixtral-8x7B, BOS | 5 | LD / drop | first subsequent token | 165 | L20 | +0.025 [+0.020, +0.031] | +0.252 |
| Mixtral-8x7B, BOS | 5 | LD / drop | further tokens | 139 | L19 | +0.026 [+0.022, +0.031] | +0.227 |
| Mixtral-8x7B, BOS | 5 | LD / drop | last token | 213 | L20 | +0.383 [+0.364, +0.403] | +2.798 |
| Mixtral-8x7B, BOS | 5 | Δp | first subject token | 212 | L2 | +0.0138 [+0.0065, +0.0230] | +0.0764 |
| Mixtral-8x7B, BOS | 5 | Δp | middle subject tokens | 196 | L2 | +0.0094 [+0.0063, +0.0135] | +0.0565 |
| Mixtral-8x7B, BOS | 5 | Δp | last subject token | 213 | L4 | +0.1188 [+0.0926, +0.1463] | +0.9203 |
| Mixtral-8x7B, BOS | 5 | Δp | first subsequent token | 165 | L20 | +0.0002 [+0.0001, +0.0002] | +0.0014 |
| Mixtral-8x7B, BOS | 5 | Δp | further tokens | 139 | L19 | +0.0004 [+0.0002, +0.0006] | +0.0026 |
| Mixtral-8x7B, BOS | 5 | Δp | last token | 213 | L20 | +0.0572 [+0.0420, +0.0743] | +0.2930 |

### Is the last subject token special?

**Zhang & Nanda's statistic: sum over layers of the effect at the last subject token / at the middle subject tokens (GPT-2 XL, STR, window 5: LD 1.22x, probability 4.33x)**

| Model | Window | Cases with >= 3 subject tokens | Ratio last / middle subject tokens, LD | Ratio, probability |
|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 1 | 188 | 2.87 [2.40, 3.44] | 2.65 [1.30, 5.95] |
| Qwen3-30B-A3B-Base | 5 | 188 | 3.18 [2.75, 3.67] | 3.99 [2.68, 6.57] |
| Mixtral-8x7B, BOS | 1 | 196 | 4.79 [4.20, 5.48] | 6.54 [3.40, 13.53] |
| Mixtral-8x7B, BOS | 5 | 196 | 5.04 [4.47, 5.70] | 14.88 [9.74, 23.83] |

Within the subject, yes: in Qwen3 the last subject token's layer-summed rescue is 1.27 drops against 0.43 for the middle and 0.07 for the first subject token; above 0.05 of the drop the last subject token stays up to L7, the middle tokens up to L2 and the first up to L0; in Mixtral BOS the last subject token's layer-summed rescue is 1.60 drops against 0.33 for the middle and 0.25 for the first subject token; above 0.05 of the drop the last subject token stays up to L17, the middle tokens up to L0 and the first up to L0. The first and middle subject tokens matter only in the first MoE layers (peak L0), where the MoE output at a subject position still carries that token's identity — the analogue of the MLP0 effect Zhang & Nanda set aside for GPT-2. Across the whole prompt, the last subject token and the final position are the two loci; per layer the final-position peak is 0.94× the last-subject-token peak in Qwen3, 0.28× the last-subject-token peak in Mixtral BOS. With single-layer patches, unlike GPT-2 XL (probability 4.33×, logit difference 1.22× more effect on the last than on the middle subject tokens), both metrics give the same ordering; the probability ratio is noisier because Δp is tiny (clean p(true) has a median of about 0.04, Direction 5-F5). With the 5-layer window, Zhang & Nanda's setting, the probability ratio rises above the logit-difference ratio (Qwen3 3.99× [2.68, 6.57] vs 3.18× [2.75, 3.67]; Mixtral BOS 14.88× [9.74, 23.83] vs 5.04× [4.47, 5.70]): a joint patch at the last subject token moves the logit difference far enough to reach the steep part of the softmax, a middle-token patch does not.

### STR vs GN at the last subject token

**Single-layer MoE patch at the last subject token: STR (this grid) vs GN (F4 runs *_subject) on the same cases, each normalised by its own drop**

| Model | n cases | STR peak (rescue / drop) | GN peak (rescue / drop) | STR sum over layers | GN sum over layers | GN / STR (sum) | Mean drop STR / GN |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 215 | L0 +0.187 [+0.165, +0.211] | L4 +0.160 [+0.124, +0.199] | +1.27 | +1.34 | 1.06 | +12.20 / +5.72 |
| Mixtral-8x7B, BOS | 213 | L0 +0.304 [+0.279, +0.330] | L4 +0.487 [+0.427, +0.544] | +1.60 | +2.83 | 1.77 | +12.81 / +5.01 |

![STR vs GN at the last subject token](figures/ext6_str_grid_gn_subject.png)

Normalised by their own drops, Qwen3: GN peaks at L4 (+0.160), STR at L0 (+0.187); layer sums 1.34 vs 1.27; Mixtral BOS: GN peaks at L4 (+0.487), STR at L0 (+0.304); layer sums 2.83 vs 1.60. At L0 the patch rescues +0.187 under STR vs +0.066 under GN in Qwen3; +0.304 under STR vs +0.175 under GN in Mixtral BOS. GN noises the subject embeddings, so the residual at the subject keeps the off-distribution embedding and restoring the first MoE output repairs less, with the repair peaking a few layers later; STR swaps in another real subject, so the first MoE layer's output already restores much of the token identity. Layer-summed GN / STR: Qwen3 1.06×; Mixtral BOS 1.77×. Where this ratio is well above 1 it is the inflation Zhang & Nanda report for GPT-2 XL (GN peaks 2–5× STR's); near 1 the total is unchanged and only the layer profile shifts.

### Sliding window vs single layer

**Sliding-window (joint) patch vs the sum of single-layer patches over the same 5-layer window (Zhang & Nanda Section 5: 1.40-1.75x in GPT-2 XL)**

| Model | Metric | Token group | Sliding window 5 peak | Sum of single layers over the window, peak | Sliding / adding |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | LD / drop | first subject token | +0.0962 (L0) | +0.0863 (L1) | 1.12 |
| Qwen3-30B-A3B-Base | LD / drop | middle subject tokens | +0.3415 (L2) | +0.3711 (L2) | 0.92 |
| Qwen3-30B-A3B-Base | LD / drop | last subject token | +0.5922 (L2) | +0.7167 (L2) | 0.83 |
| Qwen3-30B-A3B-Base | LD / drop | first subsequent token | +0.0108 (L26) | +0.0080 (L40) | 1.34 |
| Qwen3-30B-A3B-Base | LD / drop | further tokens | +0.0093 (L28) | +0.0100 (L17) | 0.94 |
| Qwen3-30B-A3B-Base | LD / drop | last token | +0.4749 (L42) | +0.4892 (L42) | 0.97 |
| Qwen3-30B-A3B-Base | Δp | first subject token | +0.0071 (L1) | +0.0063 (L1) | 1.14 |
| Qwen3-30B-A3B-Base | Δp | middle subject tokens | +0.0192 (L1) | +0.0062 (L2) | 3.10 |
| Qwen3-30B-A3B-Base | Δp | last subject token | +0.0715 (L2) | +0.0193 (L3) | 3.71 |
| Qwen3-30B-A3B-Base | Δp | first subsequent token | +0.0001 (L26) | +0.0001 (L26) | 0.80 |
| Qwen3-30B-A3B-Base | Δp | further tokens | +0.0002 (L32) | +0.0003 (L30) | 0.73 |
| Qwen3-30B-A3B-Base | Δp | last token | +0.0484 (L42) | +0.0137 (L42) | 3.53 |
| Mixtral-8x7B, BOS | LD / drop | first subject token | +0.2109 (L2) | +0.2302 (L2) | 0.92 |
| Mixtral-8x7B, BOS | LD / drop | middle subject tokens | +0.2387 (L2) | +0.2835 (L2) | 0.84 |
| Mixtral-8x7B, BOS | LD / drop | last subject token | +0.6635 (L4) | +1.0968 (L2) | 0.60 |
| Mixtral-8x7B, BOS | LD / drop | first subsequent token | +0.0253 (L20) | +0.0186 (L20) | 1.36 |
| Mixtral-8x7B, BOS | LD / drop | further tokens | +0.0262 (L19) | +0.0245 (L19) | 1.07 |
| Mixtral-8x7B, BOS | LD / drop | last token | +0.3831 (L20) | +0.3502 (L21) | 1.09 |
| Mixtral-8x7B, BOS | Δp | first subject token | +0.0138 (L2) | +0.0168 (L2) | 0.82 |
| Mixtral-8x7B, BOS | Δp | middle subject tokens | +0.0094 (L2) | +0.0057 (L2) | 1.64 |
| Mixtral-8x7B, BOS | Δp | last subject token | +0.1188 (L4) | +0.0430 (L2) | 2.76 |
| Mixtral-8x7B, BOS | Δp | first subsequent token | +0.0002 (L20) | +0.0001 (L20) | 2.13 |
| Mixtral-8x7B, BOS | Δp | further tokens | +0.0004 (L19) | +0.0003 (L19) | 1.40 |
| Mixtral-8x7B, BOS | Δp | last token | +0.0572 (L20) | +0.0094 (L20) | 6.08 |

- Qwen3: window-5 peaks last subject token L2 +0.592, final position L42 +0.475; joint / summed single-layer peak 0.83 (last subject token) and 0.97 (final position)
- Mixtral BOS: window-5 peaks last subject token L4 +0.663, final position L20 +0.383; joint / summed single-layer peak 0.60 (last subject token) and 1.09 (final position)

A ratio below 1 means the five layers restore overlapping information (the joint patch is smaller than the sum of its parts), above 1 that they act together (Zhang & Nanda's GPT-2 XL: 1.40–1.75 with probability as the metric; they suspect non-linear effects). On the logit-difference scale the window is additive or sub-additive here (Qwen3 0.83–0.97; Mixtral BOS 0.60–1.09: adjacent layers at the subject restore overlapping information, as the cross-layer redundancy of Direction 5-F1 suggested), so sliding windows do not inflate the localisation in log-odds. On the probability scale the same windows are strongly super-additive (Qwen3 3.53–3.71; Mixtral BOS 2.76–6.08): p(true) is a convex function of the log-odds near p ≈ 0, so five small single-layer moves add up to much less than one large joint move. The non-linearity Zhang & Nanda suspected is, in these models, the softmax of the metric rather than the network. The window heatmaps are smoothed versions of the single-layer ones (same two sites), so the single-layer grid is the one to quote for layer-level claims.

### Reading

1. The paper's choice of patch site (the final position) sees only the late site. Under the best-practice corruption the per-layer peak at the last subject token relative to the final-position peak is 1.06× in Qwen3, 3.53× in Mixtral BOS, and the early site is spread over several layers instead of one. This is the STR version of Direction 5-F4, where patching at the last subject token found the early site under GN and no shared expert there.
2. Zhang & Nanda's warnings, checked one by one in these MoE models. Metric: with single-layer patches logit difference and probability draw the same map; probability over-weights the last subject token only once windows are used, because joint patches reach the steep part of the softmax. Window: sliding windows barely inflate the logit difference (joint / summed singles Qwen3 0.83–0.97; Mixtral BOS 0.60–1.09 at the two sites) but strongly inflate probability. Corruption: GN inflates the early site in Mixtral, changes only its layer profile in Qwen3, and barely matters at the final position, which is where the paper's claims live (Direction 6). Following their recommendations (STR, logit difference, single layer first) therefore changes the picture at the subject, not at the paper's site.
3. Nothing between the subject and the final position matters for the MoE output: relation tokens are not a third site, consistent with attention moving the subject information to the final position (Directions 2b and 5-F2: mover heads read the last subject token).

Caveats. MoE-output patches only (Meng et al. also plot hidden states and attention; the executor supports `resid` and `attn_layer` at p, not run here). Token groups have different case counts; means are over the cases that have the group. The layer-summed values add single-layer effects and over-count shared information (Direction 5-F1); they rank groups, they are not joint effects. Mixtral without BOS (the paper's protocol) is not in this grid yet.

Files: `scripts/ext6_str_grid.py` (grid passes), `scripts/ext6_str_grid_verify.py`, `scripts/ext6_str_grid_analyze.py`, `scripts/ext6_str_grid_text.py`, `scripts/ext6_str_grid_chain.sh`; rows `results/{qwen3_str,mixtral_bos_str}/str_grid_w{1,5}_rows.parquet` (donor level: case, position, token group, layer, window bounds, Δ, rescue, p(true), rank) and `str_grid_w*_prefill.parquet`; tables `results/tables/ext6_str_grid_*`; figures `results/figures/ext6_str_grid_*`; numbers `results/ext6_str_grid_summary.json`.
