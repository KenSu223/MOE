**Summary.** Expert knockout on clean prompts (engine route mask at every position, reroute: the expert is removed from the router's menu and the next-best expert takes the slot) asks whether the experts that the final-position STR patches select are NECESSARY, and only for their own task (F = fraction of the clean margin lost). Necessary and task-specific (lower CI bound of the own-task F above max(0.005, 3 x |noise-floor F|), other-task effect below a quarter of it): Qwen3 L41E117, Qwen3 L42E115, Qwen3 L44E069, Mixtral (BOS) L18E001, Mixtral (BOS) L20E000; barely necessary: Mixtral (BOS) L19E002 (+0.004), Mixtral (BOS) L21E001 (+0.004). Each carries only a few per cent of the margin. Qwen3: L41E117 (WinoGrande) costs **+0.033 [+0.028, +0.037]** of the WinoGrande margin and -0.001 of CounterFact; L44E069 (CounterFact) costs **+0.033 [+0.026, +0.039]** of CounterFact and +0.001 of WinoGrande (L42E115 +0.024 for the other CounterFact selections); same-layer rank on the own task 1/17 and 1/17; double-dissociation contrast **+0.065** [+0.057, +0.073]. Mixtral (BOS): L20E000 (WinoGrande) costs **+0.036 [+0.033, +0.040]** of the WinoGrande margin and -0.001 of CounterFact; L21E001 (CounterFact) costs **+0.004 [+0.001, +0.008]** of CounterFact and +0.000 of WinoGrande (L18E001 +0.015, L19E002 +0.004 for the other CounterFact selections); same-layer rank on the own task 1/8 and 3/8; double-dissociation contrast **+0.041** [+0.036, +0.047]. Joint knockout of each task's population top-10 (ext8 ranking): Qwen3 WinoGrande top-10 +0.148 on WinoGrande (pair accuracy 0.997 -> 0.944) / -0.004 on CounterFact, CounterFact top-10 +0.144 / -0.000 (frequency-matched random 10-sets +0.001 / -0.006 on the respective own task); Mixtral (BOS) WinoGrande top-10 +0.154 on WinoGrande (pair accuracy 1.000 -> 0.978) / +0.013 on CounterFact, CounterFact top-10 +0.038 / +0.047 (frequency-matched random 10-sets +0.011 / +0.003 on the respective own task). Sufficiency is not necessity: a knockout moves Δ by only 0.04-0.25 of the logits that the same expert restores when patched under STR; dropping the slot without replacement (zero mode) costs about as much as rerouting, so it is not the replacement expert that compensates, and the damage is confined to prompts whose final position routes to the expert; a knockout at the final position only reproduces Qwen3 L41E117 0.91, Qwen3 L44E069 0.93, Mixtral (BOS) L20E000 0.96 of the all-position effect, i.e. the necessity of these experts sits at the final position, where the patches found them. Noise floor (unmasked baseline recomputed in other passes, WinoGrande / CounterFact): Qwen3 +0.008 / -0.003; Mixtral (BOS) +0.001 / +0.000.

### What was run

**Engine E4 (additive, `moetrace/engine.py`).** (a) `PrefillSpec.route_mask` = ((layer, expert), ...) with `route_mask_pos` all | final and `route_mask_mode` reroute | zero: reroute sets the masked experts' router logits to -inf before the softmax (top-k and renormalisation as usual; for Qwen3 and Mixtral, which renormalise the top-k, this is exactly removing the expert from the menu; OLMoE's unrenormalised weights grow by 1/(1 - p_masked)), zero keeps the routing and drops the masked contributions; recorded routing is the masked routing. (b) `multi` steps `(layer, 'attn_head', heads)` and `(layer, 'heads_experts', (heads, experts))`: v_heads = sum_h W_o[:, h](H_h_source - H_h_own) added before the MoE, the row's own MoE recomputed on the head-patched input, then the listed experts set to the source's. (c) `DiagSpec.contrib_final_vectors`: per-slot c_e at the final position. Verification on OLMoE against transformers hooks (`scripts/ext9_engine_verify.py`, 12 STR units): route masks vs a masked HF router, all / final x reroute / zero (288 rows): |ΔΔ| max 0.812, mean 0.079 (unmasked baseline 0.188 / 0.062), effect r all:reroute 0.979, all:zero 0.966, final:reroute 0.950, final:zero 0.945; masked-layer routing identical to HF in 0.941 of (row, layer); final-only masks leave every earlier position bit-identical (1.00); zero mode keeps the first masked layer's routing (1.00) and zeroes the masked slots (1.00). Head steps: one head step = the attn_head kind (max |ΔΔ| 0, vector 0); all heads of a layer vs attn_layer max |ΔΔ| 0.125; all heads at every layer vs the source Δ max 0.125; mixed head / expert / layer / attention configurations vs HF max |ΔΔ| 0.656, mean 0.079, effect r 0.9992. Contribution vectors sum to the fp32 MoE output to 8.2e-08 (relative). After the merge `verify_olmoe.json` (23 result fields), `verify_ext5_engine_olmoe.json` (225) and `verify_ext8_engine_olmoe.json` (496) are identical to their pre-merge copies in every result field (`scripts/ext9_regress_compare.py`); only verify_ext8's GPU peak-memory fields differ (< 1 MB), which also happens between two runs of the unchanged engine.

**Design.** Models Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1 with BOS (tokenizer defaults), bf16. Targets = the patching selections (Qwen3 WinoGrande L41E117, CounterFact L44E069 / L42E115; Mixtral WinoGrande L20E000, CounterFact L19E002 / L21E001 / L18E001) and the top-1/3/5/10 sets of each task's ext8 population ranking (discovery means of the single-expert STR rescue). Clean prompts only, items used for selection excluded: WinoGrande own margin pool minus the 128 main discovery pairs (primary; both prompts of a pair, Δ toward each prompt's own trigger) and every W1-W6 pair minus discovery; CounterFact clean scan with Δ_clean >= 1 minus the paper discovery IDs; IOI clean prompts (logit(IO) - logit(S); primary subset baseline Δ >= 1); 128 wikitext-103 windows of 127 tokens (per-token NLL of tokens 1..126). Controls on a fixed subsample (512 WinoGrande margin pairs, 512 CounterFact cases, 256 IOI prompts, 64 windows): same-layer experts (Mixtral the 7 others; Qwen3 the 16 experts of the layer most often routed at the final position on the target task's items) and 5 random sets per population set, matched member by member on final-position routing frequency within the member's layer. Primary intervention: mask at all positions, reroute; secondary: final position only (links to the final-position patches); sensitivity: zero mode. F = fraction of the clean margin lost = (Σ Δ_base - Σ Δ_ko) / Σ Δ_base; percentile bootstrap (5,000) over WinoGrande pairs, CounterFact cases, IOI prompts, windows, with one resample matrix per item set (paired across conditions); dissociation contrast (F_WG-expert(WG) - F_WG-expert(CF)) - (F_CF-expert(WG) - F_CF-expert(CF)) with WinoGrande pairs and CounterFact cases resampled independently.

| Model | WinoGrande margin pool | WinoGrande all pairs | CounterFact | IOI (Δ >= 1) | wikitext | GPU (passes) |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 1971 pairs (3942 prompts), Δ 3.81, pair acc 0.997 | 4386 pairs, pair acc 0.605 | 746, Δ 6.12, top-1 0.25 | 1597, Δ 6.29 | 128 windows, NLL 2.276 | 33 min |
| Mixtral-8x7B-v0.1 | 1023 pairs (2046 prompts), Δ 3.68, pair acc 1.000 | 2293 pairs, pair acc 0.608 | 816, Δ 6.33, top-1 0.27 | 1599, Δ 5.55 | 128 windows, NLL 2.103 | 55 min |

### 1. Targets and population sets (full scope)

| model | expert | selected on | WG F | WG pair acc | CF F | CF acc | IOI F | wiki dNLL | WG all F |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | WG | +0.033 [+0.028, +0.037] | 0.997 -> 0.995 | -0.001 [-0.002, +0.001] | 1.000 -> 1.000 | -0.000 [-0.002, +0.002] | +0.0000 [-0.0018, +0.0019] | +0.027 [+0.024, +0.031] |
| Qwen3-30B-A3B-Base | L42E115 | CF | -0.001 [-0.004, +0.002] | 0.997 -> 0.998 | +0.024 [+0.019, +0.029] | 1.000 -> 0.999 | -0.018 [-0.020, -0.016] | +0.0038 [+0.0013, +0.0063] | +0.000 [-0.002, +0.003] |
| Qwen3-30B-A3B-Base | L44E069 | CF | +0.001 [-0.002, +0.003] | 0.997 -> 0.997 | +0.033 [+0.026, +0.039] | 1.000 -> 0.999 | +0.001 [-0.001, +0.003] | +0.0011 [-0.0008, +0.0029] | +0.003 [+0.000, +0.005] |
| Qwen3-30B-A3B-Base | WG top-3 | WG | +0.079 [+0.072, +0.085] | 0.997 -> 0.985 | -0.002 [-0.004, -0.000] | 1.000 -> 1.000 | -0.003 [-0.004, -0.001] | +0.0020 [-0.0000, +0.0041] | +0.069 [+0.064, +0.074] |
| Qwen3-30B-A3B-Base | WG top-5 | WG | +0.101 [+0.094, +0.108] | 0.997 -> 0.979 | -0.002 [-0.004, +0.001] | 1.000 -> 0.999 | -0.003 [-0.005, -0.001] | +0.0028 [+0.0008, +0.0049] | +0.087 [+0.081, +0.093] |
| Qwen3-30B-A3B-Base | WG top-10 | WG | +0.148 [+0.139, +0.158] | 0.997 -> 0.944 | -0.004 [-0.006, -0.001] | 1.000 -> 0.999 | -0.005 [-0.008, -0.003] | +0.0075 [+0.0049, +0.0102] | +0.127 [+0.119, +0.135] |
| Qwen3-30B-A3B-Base | CF top-3 | CF | +0.001 [-0.002, +0.004] | 0.997 -> 0.997 | +0.075 [+0.065, +0.085] | 1.000 -> 0.992 | -0.016 [-0.018, -0.013] | +0.0068 [+0.0034, +0.0103] | +0.002 [-0.001, +0.005] |
| Qwen3-30B-A3B-Base | CF top-5 | CF | +0.001 [-0.003, +0.004] | 0.997 -> 0.998 | +0.093 [+0.082, +0.104] | 1.000 -> 0.991 | -0.015 [-0.018, -0.013] | +0.0112 [+0.0075, +0.0153] | +0.002 [-0.001, +0.005] |
| Qwen3-30B-A3B-Base | CF top-10 | CF | -0.000 [-0.004, +0.003] | 0.997 -> 0.998 | +0.144 [+0.130, +0.158] | 1.000 -> 0.988 | -0.013 [-0.017, -0.010] | +0.0447 [+0.0359, +0.0545] | +0.002 [-0.002, +0.004] |
| Mixtral-8x7B-v0.1 | L18E001 | CF | +0.002 [+0.000, +0.003] | 1.000 -> 1.000 | +0.015 [+0.010, +0.020] | 1.000 -> 1.000 | -0.016 [-0.018, -0.013] | +0.0016 [-0.0014, +0.0048] | +0.002 [+0.001, +0.003] |
| Mixtral-8x7B-v0.1 | L19E002 | CF | -0.001 [-0.002, -0.000] | 1.000 -> 1.000 | +0.004 [+0.001, +0.008] | 1.000 -> 1.000 | -0.001 [-0.002, +0.000] | +0.0036 [+0.0008, +0.0069] | -0.000 [-0.001, +0.001] |
| Mixtral-8x7B-v0.1 | L20E000 | WG | +0.036 [+0.033, +0.040] | 1.000 -> 0.997 | -0.001 [-0.003, +0.001] | 1.000 -> 1.000 | +0.001 [-0.000, +0.002] | +0.0077 [+0.0054, +0.0101] | +0.035 [+0.031, +0.038] |
| Mixtral-8x7B-v0.1 | L21E001 | CF | +0.000 [-0.001, +0.001] | 1.000 -> 1.000 | +0.004 [+0.001, +0.008] | 1.000 -> 1.000 | +0.003 [+0.002, +0.005] | +0.0023 [+0.0000, +0.0048] | +0.000 [-0.001, +0.001] |
| Mixtral-8x7B-v0.1 | WG top-3 | WG | +0.095 [+0.088, +0.102] | 1.000 -> 0.987 | +0.007 [+0.003, +0.011] | 1.000 -> 1.000 | +0.022 [+0.021, +0.024] | +0.0328 [+0.0281, +0.0376] | +0.089 [+0.083, +0.095] |
| Mixtral-8x7B-v0.1 | WG top-5 | WG | +0.116 [+0.109, +0.124] | 1.000 -> 0.984 | +0.006 [+0.002, +0.010] | 1.000 -> 1.000 | +0.026 [+0.024, +0.027] | +0.0382 [+0.0323, +0.0444] | +0.102 [+0.096, +0.109] |
| Mixtral-8x7B-v0.1 | WG top-10 | WG | +0.154 [+0.145, +0.164] | 1.000 -> 0.978 | +0.013 [+0.007, +0.018] | 1.000 -> 1.000 | +0.024 [+0.023, +0.026] | +0.0650 [+0.0574, +0.0727] | +0.137 [+0.130, +0.145] |
| Mixtral-8x7B-v0.1 | CF top-3 | CF | +0.000 [-0.001, +0.002] | 1.000 -> 1.000 | +0.009 [+0.003, +0.014] | 1.000 -> 1.000 | +0.018 [+0.016, +0.021] | +0.0128 [+0.0086, +0.0176] | +0.001 [-0.001, +0.002] |
| Mixtral-8x7B-v0.1 | CF top-5 | CF | +0.000 [-0.002, +0.002] | 1.000 -> 1.000 | +0.024 [+0.016, +0.033] | 1.000 -> 0.999 | +0.006 [+0.002, +0.009] | +0.0194 [+0.0138, +0.0256] | +0.001 [-0.001, +0.002] |
| Mixtral-8x7B-v0.1 | CF top-10 | CF | +0.047 [+0.042, +0.052] | 1.000 -> 0.998 | +0.038 [+0.029, +0.048] | 1.000 -> 0.999 | +0.041 [+0.037, +0.045] | +0.0522 [+0.0438, +0.0609] | +0.044 [+0.040, +0.048] |


### 2. Against same-layer controls

| model | target | task of target | subset | target effect | controls mean | controls max | max control | rank | z |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | WG | wg_margin | 0.0346 | 0.0057 | 0.0093 | L41E053 | 1/17 | 7.9356 |
| Qwen3-30B-A3B-Base | L41E117 | WG | cf | -0.0013 | -0.0008 | 0.0017 | L41E121 | 11/17 | -0.532 |
| Qwen3-30B-A3B-Base | L41E117 | WG | ioi | -0.0002 | 0.0001 | 0.0075 | L41E018 | 12/17 | -0.0626 |
| Qwen3-30B-A3B-Base | L41E117 | WG | wiki | -0.0003 | 0.0006 | 0.0029 | L41E112 | 13/17 | -0.6775 |
| Qwen3-30B-A3B-Base | L44E069 | CF | wg_margin | 0.003 | 0.005 | 0.009 | L44E022 | 14/17 | -0.9075 |
| Qwen3-30B-A3B-Base | L44E069 | CF | cf | 0.0373 | -0.0003 | 0.0101 | L44E006 | 1/17 | 8.6623 |
| Qwen3-30B-A3B-Base | L44E069 | CF | ioi | 0.0004 | -0.0013 | 0.0044 | L44E006 | 10/17 | 0.1694 |
| Qwen3-30B-A3B-Base | L44E069 | CF | wiki | 0.0014 | 0.0049 | 0.0324 | L44E104 | 10/17 | -0.3797 |
| Qwen3-30B-A3B-Base | L42E115 | CF | wg_margin | 0.0071 | 0.006 | 0.0118 | L42E037 | 6/17 | 0.4174 |
| Qwen3-30B-A3B-Base | L42E115 | CF | cf | 0.0246 | -0.0008 | 0.002 | L42E003 | 1/17 | 19.5571 |
| Qwen3-30B-A3B-Base | L42E115 | CF | ioi | -0.0179 | 0.0027 | 0.0137 | L42E065 | 17/17 | -4.9538 |
| Qwen3-30B-A3B-Base | L42E115 | CF | wiki | 0.0039 | 0.001 | 0.0029 | L42E014 | 1/17 | 2.382 |
| Mixtral-8x7B-v0.1 | L20E000 | WG | wg_margin | 0.0393 | 0.0008 | 0.0033 | L20E001 | 1/8 | 27.0124 |
| Mixtral-8x7B-v0.1 | L20E000 | WG | cf | 0.0003 | 0.0006 | 0.0021 | L20E006 | 5/8 | -0.2423 |
| Mixtral-8x7B-v0.1 | L20E000 | WG | ioi | 0.0015 | 0.0023 | 0.0152 | L20E005 | 4/8 | -0.1288 |
| Mixtral-8x7B-v0.1 | L20E000 | WG | wiki | 0.0078 | 0.0022 | 0.0081 | L20E005 | 2/8 | 1.4982 |
| Mixtral-8x7B-v0.1 | L19E002 | CF | wg_margin | -0.0015 | 0.0071 | 0.0429 | L19E006 | 8/8 | -0.5481 |
| Mixtral-8x7B-v0.1 | L19E002 | CF | cf | 0.0048 | 0.0022 | 0.0128 | L19E006 | 2/8 | 0.5493 |
| Mixtral-8x7B-v0.1 | L19E002 | CF | ioi | -0.001 | 0.001 | 0.0247 | L19E006 | 5/8 | -0.1616 |
| Mixtral-8x7B-v0.1 | L19E002 | CF | wiki | 0.0036 | 0.0082 | 0.0234 | L19E006 | 7/8 | -0.6396 |
| Mixtral-8x7B-v0.1 | L21E001 | CF | wg_margin | 0.0006 | 0.0028 | 0.0225 | L21E006 | 5/8 | -0.2412 |
| Mixtral-8x7B-v0.1 | L21E001 | CF | cf | 0.0003 | -0.0005 | 0.002 | L21E007 | 3/8 | 0.4925 |
| Mixtral-8x7B-v0.1 | L21E001 | CF | ioi | 0.0028 | 0.001 | 0.0049 | L21E000 | 2/8 | 0.7701 |
| Mixtral-8x7B-v0.1 | L21E001 | CF | wiki | 0.0008 | 0.0044 | 0.0064 | L21E000 | 8/8 | -3.3765 |
| Mixtral-8x7B-v0.1 | L18E001 | CF | wg_margin | 0.0021 | 0.0024 | 0.008 | L18E003 | 5/8 | -0.1064 |
| Mixtral-8x7B-v0.1 | L18E001 | CF | cf | 0.0132 | -0.0002 | 0.0022 | L18E005 | 1/8 | 7.6784 |
| Mixtral-8x7B-v0.1 | L18E001 | CF | ioi | -0.0149 | 0.0009 | 0.027 | L18E003 | 7/8 | -1.1387 |
| Mixtral-8x7B-v0.1 | L18E001 | CF | wiki | -0.0021 | 0.0036 | 0.0059 | L18E003 | 8/8 | -2.7955 |


![Targets vs same-layer controls](figures/ext9_knockout_controls.png)

### 3. Population sets against frequency-matched random sets

| model | set | subset | population set | random mean | random max | rank | z | population set (full scope) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | wg:k1 | wg_margin | 0.0346 | 0.0064 | 0.0082 | 1/6 | 28.3157 | 0.0325 |
| Qwen3-30B-A3B-Base | wg:k1 | cf | -0.0013 | -0.0019 | -0.0019 | 1/6 | 8.4971 | -0.0005 |
| Qwen3-30B-A3B-Base | wg:k1 | ioi | -0.0002 | -0.0004 | -0.0002 | 1/6 | 0.6505 | -0.0004 |
| Qwen3-30B-A3B-Base | wg:k1 | wiki | -0.0003 | -0.0007 | -0.0001 | 2/6 | 1.1579 | 0.0 |
| Qwen3-30B-A3B-Base | wg:k3 | wg_margin | 0.0816 | 0.0151 | 0.0201 | 1/6 | 19.7389 | 0.0787 |
| Qwen3-30B-A3B-Base | wg:k3 | cf | -0.0026 | -0.0023 | -0.0015 | 5/6 | -0.5228 | -0.002 |
| Qwen3-30B-A3B-Base | wg:k3 | ioi | -0.001 | -0.0007 | 0.0063 | 3/6 | -0.0807 | -0.0026 |
| Qwen3-30B-A3B-Base | wg:k3 | wiki | 0.0026 | 0.001 | 0.0019 | 1/6 | 1.4381 | 0.002 |
| Qwen3-30B-A3B-Base | wg:k5 | wg_margin | 0.1056 | 0.0111 | 0.0172 | 1/6 | 21.0538 | 0.1014 |
| Qwen3-30B-A3B-Base | wg:k5 | cf | -0.0027 | -0.0048 | -0.0017 | 2/6 | 0.8469 | -0.0019 |
| Qwen3-30B-A3B-Base | wg:k5 | ioi | -0.0021 | 0.0075 | 0.013 | 6/6 | -2.1347 | -0.0032 |
| Qwen3-30B-A3B-Base | wg:k5 | wiki | 0.0035 | 0.0024 | 0.0039 | 2/6 | 0.8731 | 0.0028 |
| Qwen3-30B-A3B-Base | wg:k10 | wg_margin | 0.1525 | 0.0013 | 0.0114 | 1/6 | 19.2485 | 0.1484 |
| Qwen3-30B-A3B-Base | wg:k10 | cf | -0.0051 | -0.0018 | 0.0003 | 6/6 | -2.2889 | -0.0035 |
| Qwen3-30B-A3B-Base | wg:k10 | ioi | -0.0054 | 0.0082 | 0.0161 | 6/6 | -1.8823 | -0.0055 |
| Qwen3-30B-A3B-Base | wg:k10 | wiki | 0.0061 | 0.0062 | 0.0076 | 5/6 | -0.0514 | 0.0075 |
| Qwen3-30B-A3B-Base | cf:k1 | wg_margin | 0.003 | 0.0076 | 0.0086 | 6/6 | -2.9807 | 0.0007 |
| Qwen3-30B-A3B-Base | cf:k1 | cf | 0.0373 | -0.0044 | -0.0015 | 1/6 | 15.676 | 0.0325 |
| Qwen3-30B-A3B-Base | cf:k1 | ioi | 0.0004 | 0.0023 | 0.0037 | 6/6 | -1.3112 | 0.0006 |
| Qwen3-30B-A3B-Base | cf:k1 | wiki | 0.0014 | 0.0002 | 0.001 | 1/6 | 1.0209 | 0.0011 |
| Qwen3-30B-A3B-Base | cf:k3 | wg_margin | 0.0033 | 0.0094 | 0.0111 | 6/6 | -2.3972 | 0.0011 |
| Qwen3-30B-A3B-Base | cf:k3 | cf | 0.08 | -0.0041 | 0.0034 | 1/6 | 17.4013 | 0.075 |
| Qwen3-30B-A3B-Base | cf:k3 | ioi | -0.015 | 0.0005 | 0.0051 | 6/6 | -5.0149 | -0.0158 |
| Qwen3-30B-A3B-Base | cf:k3 | wiki | 0.0042 | 0.0029 | 0.0063 | 2/6 | 0.5587 | 0.0068 |
| Qwen3-30B-A3B-Base | cf:k5 | wg_margin | 0.0051 | 0.0126 | 0.0146 | 6/6 | -4.3378 | 0.0005 |
| Qwen3-30B-A3B-Base | cf:k5 | cf | 0.1001 | -0.0022 | 0.0028 | 1/6 | 25.7276 | 0.093 |
| Qwen3-30B-A3B-Base | cf:k5 | ioi | -0.0142 | 0.0073 | 0.0202 | 6/6 | -2.4234 | -0.0155 |
| Qwen3-30B-A3B-Base | cf:k5 | wiki | 0.0088 | 0.0065 | 0.0112 | 2/6 | 0.6793 | 0.0112 |
| Qwen3-30B-A3B-Base | cf:k10 | wg_margin | 0.0046 | 0.0187 | 0.0373 | 6/6 | -1.2926 | -0.0004 |
| Qwen3-30B-A3B-Base | cf:k10 | cf | 0.1497 | -0.0061 | -0.0014 | 1/6 | 40.5283 | 0.1441 |
| Qwen3-30B-A3B-Base | cf:k10 | ioi | -0.0149 | 0.0067 | 0.034 | 6/6 | -1.1768 | -0.0134 |
| Qwen3-30B-A3B-Base | cf:k10 | wiki | 0.0597 | 0.0164 | 0.0191 | 1/6 | 24.639 | 0.0447 |
| Mixtral-8x7B-v0.1 | wg:k1 | wg_margin | 0.0393 | 0.0017 | 0.0033 | 1/6 | 23.8857 | 0.0364 |
| Mixtral-8x7B-v0.1 | wg:k1 | cf | 0.0003 | 0.0004 | 0.001 | 5/6 | -0.2242 | -0.0006 |
| Mixtral-8x7B-v0.1 | wg:k1 | ioi | 0.0015 | 0.0099 | 0.0152 | 5/6 | -1.1494 | 0.0006 |
| Mixtral-8x7B-v0.1 | wg:k1 | wiki | 0.0078 | 0.0057 | 0.0081 | 4/6 | 0.627 | 0.0077 |
| Mixtral-8x7B-v0.1 | wg:k3 | wg_margin | 0.0982 | 0.0035 | 0.0052 | 1/6 | 53.9666 | 0.0947 |
| Mixtral-8x7B-v0.1 | wg:k3 | cf | 0.0081 | -0.0007 | 0.0007 | 1/6 | 10.2549 | 0.007 |
| Mixtral-8x7B-v0.1 | wg:k3 | ioi | 0.024 | 0.0016 | 0.0071 | 1/6 | 3.3448 | 0.0225 |
| Mixtral-8x7B-v0.1 | wg:k3 | wiki | 0.0369 | 0.0134 | 0.016 | 1/6 | 12.5915 | 0.0328 |
| Mixtral-8x7B-v0.1 | wg:k5 | wg_margin | 0.1191 | 0.0041 | 0.0105 | 1/6 | 28.0147 | 0.1162 |
| Mixtral-8x7B-v0.1 | wg:k5 | cf | 0.0078 | -0.0031 | -0.0003 | 1/6 | 5.6289 | 0.0062 |
| Mixtral-8x7B-v0.1 | wg:k5 | ioi | 0.0278 | 0.009 | 0.0403 | 2/6 | 0.9752 | 0.0257 |
| Mixtral-8x7B-v0.1 | wg:k5 | wiki | 0.0511 | 0.0281 | 0.0366 | 1/6 | 4.4117 | 0.0382 |
| Mixtral-8x7B-v0.1 | wg:k10 | wg_margin | 0.1586 | 0.0109 | 0.0164 | 1/6 | 29.6209 | 0.1543 |
| Mixtral-8x7B-v0.1 | wg:k10 | cf | 0.0144 | 0.0049 | 0.0125 | 1/6 | 1.6956 | 0.0126 |
| Mixtral-8x7B-v0.1 | wg:k10 | ioi | 0.0264 | 0.0129 | 0.0506 | 2/6 | 0.4628 | 0.0244 |
| Mixtral-8x7B-v0.1 | wg:k10 | wiki | 0.0753 | 0.0478 | 0.0551 | 1/6 | 5.0301 | 0.065 |
| Mixtral-8x7B-v0.1 | cf:k1 | wg_margin | 0.0006 | 0.0027 | 0.0225 | 4/6 | -0.1793 | 0.0001 |
| Mixtral-8x7B-v0.1 | cf:k1 | cf | 0.0003 | -0.0014 | -0.001 | 1/6 | 1.9912 | 0.0045 |
| Mixtral-8x7B-v0.1 | cf:k1 | ioi | 0.0028 | 0.0018 | 0.0049 | 3/6 | 0.3435 | 0.0032 |
| Mixtral-8x7B-v0.1 | cf:k1 | wiki | 0.0008 | 0.0045 | 0.0064 | 6/6 | -2.0844 | 0.0023 |
| Mixtral-8x7B-v0.1 | cf:k3 | wg_margin | 0.0003 | 0.0081 | 0.024 | 4/6 | -0.5234 | 0.0003 |
| Mixtral-8x7B-v0.1 | cf:k3 | cf | 0.0056 | -0.0015 | 0.0004 | 1/6 | 4.1616 | 0.0085 |
| Mixtral-8x7B-v0.1 | cf:k3 | ioi | 0.0171 | -0.0151 | -0.01 | 1/6 | 7.3336 | 0.0185 |
| Mixtral-8x7B-v0.1 | cf:k3 | wiki | 0.0161 | 0.0154 | 0.0164 | 4/6 | 0.3995 | 0.0128 |
| Mixtral-8x7B-v0.1 | cf:k5 | wg_margin | 0.0002 | 0.0175 | 0.0372 | 5/6 | -0.9669 | 0.0001 |
| Mixtral-8x7B-v0.1 | cf:k5 | cf | 0.0204 | 0.0015 | 0.0046 | 1/6 | 6.4397 | 0.0243 |
| Mixtral-8x7B-v0.1 | cf:k5 | ioi | 0.0066 | -0.0055 | 0.0065 | 1/6 | 1.0889 | 0.0057 |
| Mixtral-8x7B-v0.1 | cf:k5 | wiki | 0.0217 | 0.02 | 0.0246 | 4/6 | 0.3024 | 0.0194 |
| Mixtral-8x7B-v0.1 | cf:k10 | wg_margin | 0.0473 | 0.0163 | 0.0497 | 2/6 | 1.5747 | 0.0471 |
| Mixtral-8x7B-v0.1 | cf:k10 | cf | 0.0351 | 0.0033 | 0.0071 | 1/6 | 9.6352 | 0.0384 |
| Mixtral-8x7B-v0.1 | cf:k10 | ioi | 0.0422 | -0.0091 | 0.018 | 1/6 | 3.0317 | 0.0412 |
| Mixtral-8x7B-v0.1 | cf:k10 | wiki | 0.0582 | 0.0479 | 0.0582 | 2/6 | 1.3623 | 0.0522 |


![Sets vs random sets](figures/ext9_knockout_sets.png)

### 4. Double dissociation

| model | WG expert(s) | CF expert(s) | F WGexp on WG | F WGexp on CF | F CFexp on WG | F CFexp on CF | contrast [95% CI] |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | L44E069 | 0.0325 | -0.0005 | 0.0007 | 0.0325 | +0.065 [+0.057, +0.073] |
| Qwen3-30B-A3B-Base | L41E117 | L42E115 | 0.0325 | -0.0005 | -0.0007 | 0.0243 | +0.058 [+0.052, +0.064] |
| Qwen3-30B-A3B-Base | pop:wg:k3 | pop:cf:k3 | 0.0787 | -0.002 | 0.0011 | 0.075 | +0.155 [+0.143, +0.167] |
| Qwen3-30B-A3B-Base | pop:wg:k5 | pop:cf:k5 | 0.1014 | -0.0019 | 0.0005 | 0.093 | +0.196 [+0.183, +0.209] |
| Qwen3-30B-A3B-Base | pop:wg:k10 | pop:cf:k10 | 0.1484 | -0.0035 | -0.0004 | 0.1441 | +0.296 [+0.279, +0.314] |
| Mixtral-8x7B-v0.1 | L20E000 | L19E002 | 0.0364 | -0.0006 | -0.0012 | 0.0045 | +0.043 [+0.037, +0.048] |
| Mixtral-8x7B-v0.1 | L20E000 | L21E001 | 0.0364 | -0.0006 | 0.0001 | 0.0045 | +0.041 [+0.036, +0.047] |
| Mixtral-8x7B-v0.1 | L20E000 | L18E001 | 0.0364 | -0.0006 | 0.0016 | 0.0148 | +0.050 [+0.044, +0.057] |
| Mixtral-8x7B-v0.1 | pop:wg:k3 | pop:cf:k3 | 0.0947 | 0.007 | 0.0003 | 0.0085 | +0.096 [+0.086, +0.105] |
| Mixtral-8x7B-v0.1 | pop:wg:k5 | pop:cf:k5 | 0.1162 | 0.0062 | 0.0001 | 0.0243 | +0.134 [+0.122, +0.147] |
| Mixtral-8x7B-v0.1 | pop:wg:k10 | pop:cf:k10 | 0.1543 | 0.0126 | 0.0471 | 0.0384 | +0.133 [+0.120, +0.146] |


### 5. Routing: which expert takes the slot

| model | expert | task | routed at final (baseline) | routed, all positions | clean one-for-one swap | top replacements (share) |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | wg | 0.8247 | 0.1089 | 0.9824 | L41E079 (0.131), L41E121 (0.1001), L41E018 (0.074) |
| Qwen3-30B-A3B-Base | L41E117 | cf | 0.1072 | 0.0269 | 0.975 | L41E085 (0.1375), L41E106 (0.125), L41E001 (0.1125) |
| Qwen3-30B-A3B-Base | L41E117 | ioi | 0.0 | 0.0001 | 0.0 |  |
| Qwen3-30B-A3B-Base | L42E115 | wg | 0.0345 | 0.2342 | 0.967 | L42E002 (0.0957), L42E022 (0.0957), L42E016 (0.0858) |
| Qwen3-30B-A3B-Base | L42E115 | cf | 0.9718 | 0.3789 | 0.9903 | L42E016 (0.091), L42E023 (0.0869), L42E071 (0.0703) |
| Qwen3-30B-A3B-Base | L42E115 | ioi | 1.0 | 0.388 | 0.9838 | L42E031 (0.1819), L42E123 (0.1719), L42E074 (0.1169) |
| Qwen3-30B-A3B-Base | L44E069 | wg | 0.0421 | 0.1107 | 0.981 | L44E006 (0.1274), L44E027 (0.122), L44E022 (0.0705) |
| Qwen3-30B-A3B-Base | L44E069 | cf | 0.9062 | 0.4747 | 0.9956 | L44E022 (0.0932), L44E027 (0.074), L44E081 (0.0651) |
| Qwen3-30B-A3B-Base | L44E069 | ioi | 0.1181 | 0.1403 | 0.9894 | L44E006 (0.3069), L44E116 (0.1217), L44E122 (0.0952) |
| Mixtral-8x7B-v0.1 | L18E001 | wg | 0.017 | 0.1804 | 1.0 | L18E006 (0.4359), L18E005 (0.2179), L18E004 (0.1282) |
| Mixtral-8x7B-v0.1 | L18E001 | cf | 0.7966 | 0.2124 | 1.0 | L18E006 (0.3185), L18E003 (0.3015), L18E005 (0.2708) |
| Mixtral-8x7B-v0.1 | L18E001 | ioi | 0.9981 | 0.229 | 1.0 | L18E003 (0.6493), L18E006 (0.3456), L18E005 (0.0038) |
| Mixtral-8x7B-v0.1 | L19E002 | wg | 0.0667 | 0.1811 | 1.0 | L19E000 (0.4542), L19E004 (0.3595), L19E005 (0.0784) |
| Mixtral-8x7B-v0.1 | L19E002 | cf | 0.5956 | 0.2964 | 1.0 | L19E004 (0.323), L19E006 (0.1996), L19E000 (0.1934) |
| Mixtral-8x7B-v0.1 | L19E002 | ioi | 0.0 | 0.1875 | 0.0 |  |
| Mixtral-8x7B-v0.1 | L20E000 | wg | 0.9344 | 0.2509 | 0.9998 | L20E007 (0.3162), L20E005 (0.2474), L20E001 (0.189) |
| Mixtral-8x7B-v0.1 | L20E000 | cf | 0.31 | 0.3117 | 1.0 | L20E004 (0.3439), L20E002 (0.2569), L20E005 (0.1502) |
| Mixtral-8x7B-v0.1 | L20E000 | ioi | 0.0013 | 0.1467 | 1.0 | L20E006 (0.5), L20E002 (0.5) |
| Mixtral-8x7B-v0.1 | L21E001 | wg | 0.0602 | 0.2076 | 1.0 | L21E004 (0.3188), L21E000 (0.1775), L21E003 (0.163) |
| Mixtral-8x7B-v0.1 | L21E001 | cf | 0.674 | 0.3324 | 1.0 | L21E006 (0.2091), L21E004 (0.1818), L21E000 (0.1709) |
| Mixtral-8x7B-v0.1 | L21E001 | ioi | 0.9456 | 0.3011 | 1.0 | L21E002 (0.534), L21E000 (0.1765), L21E006 (0.1659) |


### 6. Final position only and zero mode

| model | condition | subset | all_reroute | final_reroute | all_zero |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | wg_margin | 0.0346 | 0.0314 | 0.0341 |
| Qwen3-30B-A3B-Base | L41E117 | cf | -0.0013 | -0.0021 | -0.0023 |
| Qwen3-30B-A3B-Base | L41E117 | ioi | -0.0002 | 0.0008 | 0.0 |
| Qwen3-30B-A3B-Base | L41E117 | wiki | -0.0003 | nan | 0.0015 |
| Qwen3-30B-A3B-Base | L42E115 | wg_margin | 0.0071 | 0.0031 | 0.0002 |
| Qwen3-30B-A3B-Base | L42E115 | cf | 0.0246 | 0.0211 | 0.0243 |
| Qwen3-30B-A3B-Base | L42E115 | ioi | -0.0179 | -0.0159 | -0.0068 |
| Qwen3-30B-A3B-Base | L42E115 | wiki | 0.0039 | nan | 0.0056 |
| Qwen3-30B-A3B-Base | L44E069 | wg_margin | 0.003 | 0.0038 | 0.0005 |
| Qwen3-30B-A3B-Base | L44E069 | cf | 0.0373 | 0.0347 | 0.0338 |
| Qwen3-30B-A3B-Base | L44E069 | ioi | 0.0004 | 0.0001 | -0.002 |
| Qwen3-30B-A3B-Base | L44E069 | wiki | 0.0014 | nan | 0.0025 |
| Qwen3-30B-A3B-Base | pop:wg:k3 | wg_margin | 0.0816 | 0.0724 | nan |
| Qwen3-30B-A3B-Base | pop:wg:k3 | cf | -0.0026 | -0.002 | nan |
| Qwen3-30B-A3B-Base | pop:wg:k3 | ioi | -0.001 | 0.0024 | nan |
| Qwen3-30B-A3B-Base | pop:wg:k5 | wg_margin | 0.1056 | 0.1039 | nan |
| Qwen3-30B-A3B-Base | pop:wg:k5 | cf | -0.0027 | -0.0016 | nan |
| Qwen3-30B-A3B-Base | pop:wg:k5 | ioi | -0.0021 | 0.0002 | nan |
| Qwen3-30B-A3B-Base | pop:wg:k10 | wg_margin | 0.1525 | 0.136 | nan |
| Qwen3-30B-A3B-Base | pop:wg:k10 | cf | -0.0051 | -0.0031 | nan |
| Qwen3-30B-A3B-Base | pop:wg:k10 | ioi | -0.0054 | 0.0024 | nan |
| Qwen3-30B-A3B-Base | pop:cf:k3 | wg_margin | 0.0033 | 0.0015 | nan |
| Qwen3-30B-A3B-Base | pop:cf:k3 | cf | 0.08 | 0.0749 | nan |
| Qwen3-30B-A3B-Base | pop:cf:k3 | ioi | -0.015 | -0.0154 | nan |
| Qwen3-30B-A3B-Base | pop:cf:k5 | wg_margin | 0.0051 | 0.0072 | nan |
| Qwen3-30B-A3B-Base | pop:cf:k5 | cf | 0.1001 | 0.0934 | nan |
| Qwen3-30B-A3B-Base | pop:cf:k5 | ioi | -0.0142 | -0.0158 | nan |
| Qwen3-30B-A3B-Base | pop:cf:k10 | wg_margin | 0.0046 | 0.0066 | nan |
| Qwen3-30B-A3B-Base | pop:cf:k10 | cf | 0.1497 | 0.1437 | nan |
| Qwen3-30B-A3B-Base | pop:cf:k10 | ioi | -0.0149 | -0.0229 | nan |
| Mixtral-8x7B-v0.1 | L18E001 | wg_margin | 0.0021 | 0.0009 | 0.0015 |
| Mixtral-8x7B-v0.1 | L18E001 | cf | 0.0132 | 0.0124 | 0.0182 |
| Mixtral-8x7B-v0.1 | L18E001 | ioi | -0.0149 | -0.014 | 0.014 |
| Mixtral-8x7B-v0.1 | L18E001 | wiki | -0.0021 | nan | -0.0012 |
| Mixtral-8x7B-v0.1 | L19E002 | wg_margin | -0.0015 | 0.0 | 0.0004 |
| Mixtral-8x7B-v0.1 | L19E002 | cf | 0.0048 | 0.0036 | 0.0069 |
| Mixtral-8x7B-v0.1 | L19E002 | ioi | -0.001 | 0.0006 | -0.0028 |
| Mixtral-8x7B-v0.1 | L19E002 | wiki | 0.0036 | nan | 0.0047 |
| Mixtral-8x7B-v0.1 | L20E000 | wg_margin | 0.0393 | 0.0377 | 0.0377 |
| Mixtral-8x7B-v0.1 | L20E000 | cf | 0.0003 | 0.0005 | 0.0001 |
| Mixtral-8x7B-v0.1 | L20E000 | ioi | 0.0015 | 0.0004 | 0.0028 |
| Mixtral-8x7B-v0.1 | L20E000 | wiki | 0.0078 | nan | 0.0079 |
| Mixtral-8x7B-v0.1 | L21E001 | wg_margin | 0.0006 | -0.0009 | 0.0 |
| Mixtral-8x7B-v0.1 | L21E001 | cf | 0.0003 | -0.0009 | 0.0024 |
| Mixtral-8x7B-v0.1 | L21E001 | ioi | 0.0028 | 0.0056 | -0.0007 |
| Mixtral-8x7B-v0.1 | L21E001 | wiki | 0.0008 | nan | 0.0013 |
| Mixtral-8x7B-v0.1 | pop:wg:k3 | wg_margin | 0.0982 | 0.0743 | nan |
| Mixtral-8x7B-v0.1 | pop:wg:k3 | cf | 0.0081 | -0.0005 | nan |
| Mixtral-8x7B-v0.1 | pop:wg:k3 | ioi | 0.024 | 0.0008 | nan |
| Mixtral-8x7B-v0.1 | pop:wg:k5 | wg_margin | 0.1191 | 0.0941 | nan |
| Mixtral-8x7B-v0.1 | pop:wg:k5 | cf | 0.0078 | 0.0005 | nan |
| Mixtral-8x7B-v0.1 | pop:wg:k5 | ioi | 0.0278 | 0.0 | nan |
| Mixtral-8x7B-v0.1 | pop:wg:k10 | wg_margin | 0.1586 | 0.1255 | nan |
| Mixtral-8x7B-v0.1 | pop:wg:k10 | cf | 0.0144 | 0.0051 | nan |
| Mixtral-8x7B-v0.1 | pop:wg:k10 | ioi | 0.0264 | 0.0002 | nan |
| Mixtral-8x7B-v0.1 | pop:cf:k3 | wg_margin | 0.0003 | 0.0009 | nan |
| Mixtral-8x7B-v0.1 | pop:cf:k3 | cf | 0.0056 | 0.0033 | nan |
| Mixtral-8x7B-v0.1 | pop:cf:k3 | ioi | 0.0171 | 0.0181 | nan |
| Mixtral-8x7B-v0.1 | pop:cf:k5 | wg_margin | 0.0002 | -0.0006 | nan |
| Mixtral-8x7B-v0.1 | pop:cf:k5 | cf | 0.0204 | 0.0157 | nan |
| Mixtral-8x7B-v0.1 | pop:cf:k5 | ioi | 0.0066 | 0.005 | nan |
| Mixtral-8x7B-v0.1 | pop:cf:k10 | wg_margin | 0.0473 | 0.0329 | nan |
| Mixtral-8x7B-v0.1 | pop:cf:k10 | cf | 0.0351 | 0.0192 | nan |
| Mixtral-8x7B-v0.1 | pop:cf:k10 | ioi | 0.0422 | 0.0028 | nan |


### Reading

**1. The patch-selected experts are necessary for their own task and only for it, but each carries a few per cent of the margin.** Fraction of the clean margin lost by a single all-position knockout (full scope): Qwen3: L41E117 (WG) +0.033 [+0.028, +0.037] on its own task vs -0.001 on the other (IOI -0.000, wiki +0.000 nats; rank 1/17 among same-layer experts on its own task); L42E115 (CF) +0.024 [+0.019, +0.029] on its own task vs -0.001 on the other (IOI -0.018, wiki +0.004 nats; rank 1/17 among same-layer experts on its own task); L44E069 (CF) +0.033 [+0.026, +0.039] on its own task vs +0.001 on the other (IOI +0.001, wiki +0.001 nats; rank 1/17 among same-layer experts on its own task). Mixtral (BOS): L18E001 (CF) +0.015 [+0.010, +0.020] on its own task vs +0.002 on the other (IOI -0.016, wiki +0.002 nats; rank 1/8 among same-layer experts on its own task); L19E002 (CF) +0.004 [+0.001, +0.008] on its own task vs -0.001 on the other (IOI -0.001, wiki +0.004 nats; rank 2/8 among same-layer experts on its own task); L20E000 (WG) +0.036 [+0.033, +0.040] on its own task vs -0.001 on the other (IOI +0.001, wiki +0.008 nats; rank 1/8 among same-layer experts on its own task); L21E001 (CF) +0.004 [+0.001, +0.008] on its own task vs +0.000 on the other (IOI +0.003, wiki +0.002 nats; rank 3/8 among same-layer experts on its own task).

**Strongest non-target expert in the control sets (sub scope).** Qwen3 WinoGrande: L42E037 +0.012 (wiki +0.001 nats; best target +0.035); Qwen3 CounterFact: L44E006 +0.010 (wiki +0.022 nats; best target +0.037); Mixtral (BOS) WinoGrande: L19E006 +0.043 (wiki +0.023 nats; best target +0.039); Mixtral (BOS) CounterFact: L19E006 +0.013 (wiki +0.023 nats; best target +0.013). A control can be as necessary as a target: in Mixtral L19E006 (a same-layer control of L19E002 and WinoGrande rank 2 in the ext8 population ranking) is the single most damaging WinoGrande knockout and also the most generically damaging one.

**2. Sufficiency is not necessity.** Patched alone into the corrupted run at the final position (Directions 6 / 7, validation), each expert restores a sizeable part of the STR drop; removed from the clean run, it costs a fraction of a logit (knockout Δ change / patch rescue = 0.04-0.25). The reroute knockout replaces the expert one-for-one in almost every prompt (spread over several replacement experts), but dropping the slot with no replacement (zero mode) costs about the same, so the small necessity is not compensation by the replacement: the knockout removes the expert's clean contribution, whereas the patch swaps the corrupted run's contribution for the clean one and lets later layers respond (the direct / indirect split of the patch effect is Direction 11). The final-position-only knockout reproduces most of the all-position effect: the necessity sits where the patches found the experts. F columns on the sub scope (same items for the three interventions).

| Model | Expert | Task | STR patch rescue, logits (fraction of drop) | knockout Δ change, logits | F all positions | F final only | F zero mode | routed at final | one-for-one swap | top replacement (share) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3 | L41E117 | WG | +1.00 (0.123) | -0.124 | +0.035 | +0.031 | +0.034 | 0.825 | 0.982 | L41E079 (0.13) |
| Qwen3 | L42E115 | CF | +0.84 (0.072) | -0.148 | +0.025 | +0.021 | +0.024 | 0.972 | 0.990 | L42E016 (0.09) |
| Qwen3 | L44E069 | CF | +1.07 (0.092) | -0.199 | +0.037 | +0.035 | +0.034 | 0.906 | 0.996 | L44E022 (0.09) |
| Mixtral (BOS) | L18E001 | CF | +0.37 (0.029) | -0.093 | +0.013 | +0.012 | +0.018 | 0.797 | 1.000 | L18E006 (0.32) |
| Mixtral (BOS) | L19E002 | CF | +0.60 (0.046) | -0.029 | +0.005 | +0.004 | +0.007 | 0.596 | 1.000 | L19E004 (0.32) |
| Mixtral (BOS) | L20E000 | WG | +1.11 (0.144) | -0.134 | +0.039 | +0.038 | +0.038 | 0.934 | 1.000 | L20E007 (0.32) |
| Mixtral (BOS) | L21E001 | CF | +0.64 (0.049) | -0.028 | +0.000 | -0.001 | +0.002 | 0.674 | 1.000 | L21E006 (0.21) |

**3. Sets: the damage grows with k and stays on the task where the rankings do not overlap.** Qwen3 WG top-10: +0.148 [+0.139, +0.158] on its own task (pair accuracy 0.997 -> 0.944), -0.004 on the other, wiki +0.0075 [+0.0049, +0.0102]; on the sub scope +0.152 vs +0.001 (max +0.011) for 5 frequency-matched random sets; Qwen3 CF top-10: +0.144 [+0.130, +0.158] on its own task (accuracy 1.000 -> 0.988), -0.000 on the other, wiki +0.0447 [+0.0359, +0.0545]; on the sub scope +0.150 vs -0.006 (max -0.001) for 5 frequency-matched random sets; Mixtral (BOS) WG top-10: +0.154 [+0.145, +0.164] on its own task (pair accuracy 1.000 -> 0.978), +0.013 on the other, wiki +0.0650 [+0.0574, +0.0727]; on the sub scope +0.159 vs +0.011 (max +0.016) for 5 frequency-matched random sets; Mixtral (BOS) CF top-10: +0.038 [+0.029, +0.048] on its own task (accuracy 1.000 -> 0.999), +0.047 on the other, wiki +0.0522 [+0.0438, +0.0609]; on the sub scope +0.035 vs +0.003 (max +0.007) for 5 frequency-matched random sets. Experts shared by the two population top-10 lists: Qwen3: none; Mixtral (BOS): L19E006, L22E005. Mixtral's CounterFact top-10 contains two WinoGrande experts (L19E006 = WinoGrande rank 2, L22E005 = rank 8), which is why it damages WinoGrande more than CounterFact; L19E006 is also the expert that the BOS token always routes to at L19 (Direction 3), so knocking it out at all positions removes it from the attention sink as well, a likely source of the larger generic (wikitext) damage of every Mixtral set that contains it.

**IOI (third task, never used for selection).** Single knockouts move the IOI margin by Qwen3 L41E117 -0.000, Qwen3 L42E115 -0.018, Qwen3 L44E069 +0.001, Mixtral (BOS) L18E001 -0.016, Mixtral (BOS) L19E002 -0.001, Mixtral (BOS) L20E000 +0.001, Mixtral (BOS) L21E001 +0.003. Negative values (margin gains) come from CounterFact experts that are routed at the IOI final position: in IOI the clean final-position MoE output works against the indirect object (Direction 7b: all-MoE patch M = -0.27 in Qwen3), so removing such an expert helps.

**Where the damage is.** The loss concentrates on prompts whose FINAL position routes to the expert in the baseline: Qwen3 L41E117: F +0.038 on the 3306 own-task prompts that route it at the final position vs +0.004 on the 636 that do not; Qwen3 L42E115: F +0.025 on the 725 own-task prompts that route it at the final position vs +0.008 on the 21 that do not; Qwen3 L44E069: F +0.035 on the 676 own-task prompts that route it at the final position vs +0.000 on the 70 that do not; Mixtral (BOS) L18E001: F +0.017 on the 650 own-task prompts that route it at the final position vs +0.001 on the 166 that do not; Mixtral (BOS) L19E002: F +0.007 on the 486 own-task prompts that route it at the final position vs -0.000 on the 330 that do not; Mixtral (BOS) L20E000: F +0.038 on the 1977 own-task prompts that route it at the final position vs -0.003 on the 69 that do not; Mixtral (BOS) L21E001: F +0.005 on the 550 own-task prompts that route it at the final position vs +0.003 on the 266 that do not. Some CounterFact experts are routed at the final position of almost every IOI prompt and their removal INCREASES the IOI margin: Qwen3 L42E115 (routed at the IOI final position in 1597 prompts, F -0.018), Mixtral (BOS) L18E001 (routed at the IOI final position in 1596 prompts, F -0.016).

**4. Noise floor.** The unmasked baseline recomputed in other pass compositions (condition `null`, sub scope; bf16 results depend on the batch composition at the 0.1-logit level) gives F = Qwen3 WinoGrande +0.008, CounterFact -0.003, IOI -0.001, wiki -0.002 nats; Mixtral (BOS) WinoGrande +0.001, CounterFact +0.000, IOI +0.000, wiki -0.002 nats.

### Caveats

- A knockout is a distribution shift the model was not trained on; the reroute variant keeps k experts per token (the model's own routing rule), the zero variant drops the slot. Generic damage is measured on wikitext, not removed.
- Rows are not bit-reproducible across pass compositions (bf16, batch-size-dependent kernels): baseline and knockout rows sit in different passes; the `null` condition measures this floor.
- Controls and random sets use a fixed subsample (512 WinoGrande margin pairs, 512 CounterFact cases, 256 IOI prompts, 64 windows); random sets are matched member by member within the member's layer, so with Mixtral's 8 experts per layer several of the 5 draws coincide.
- WinoGrande evaluation excludes only the 128 main discovery pairs (the selection set of the WinoGrande experts and of the population ranking); the main validation and replication pairs are part of the margin pool. CounterFact excludes the paper discovery IDs (the STR selections used the paper split).
- Necessity is measured on clean prompts at every position; the patches measured sufficiency at the final position under STR. The final-only knockout is the bridge between the two.

### Files

Code: `moetrace/engine.py` (E4, docstring 'ext9'), `moetrace/ext9_knockout.py` (items, conditions, controls, pass planner), `scripts/ext9_engine_verify.py`, `scripts/ext9_knockout_run.py`, `scripts/ext9_knockout_analyze.py`, `scripts/ext9_knockout_text.py`, `scripts/ext9_chain.sh`. Runs: `results/qwen3_knockout`, `results/mixtral_bos_knockout` (items.parquet, pop_rank.json, controls.json, plan_<phase>.json, ko_rows_*.parquet row level, ko_wiki_*.npz per-token NLL, ko_basefinal_* / ko_baseallpos_* baseline routing, run_meta.json). Numbers: `results/ext9_knockout_summary.json`; tables `results/tables/ext9_knockout_*`; figures `results/figures/ext9_knockout_*`; engine verification `results/verify_ext9_engine_olmoe.json`; API note `logs/ext9_engine_api.md`.
