# Engine E4 (ext9) — API for the Phase-4 agents

Merged into `moetrace/engine.py` by ext9-knockout (developed in `moetrace/engine_ext9_dev.py`). Everything is additive,
behind new fields with defaults; existing kinds, steps and outputs are unchanged (regression: `results/verify_olmoe.json`,
`results/verify_ext5_engine_olmoe.json`, `results/verify_ext8_engine_olmoe.json` identical in every non-timing field to the
`*_before_ext9.json` copies). Verification vs transformers hooks on OLMoE: `results/verify_ext9_engine_olmoe.json`
(`scripts/ext9_engine_verify.py`). The module docstring of `moetrace/engine.py` ("ext9 (Phase 4, E4) additions") is the
normative description; this file is the quick reference.

## E4a — route masks (expert knockout) on PREFILL rows

```python
from moetrace.engine import Engine, PrefillSpec
PrefillSpec(ids, true_id, foil_id,
            route_mask=((41, 117), (44, 69)),   # (layer, expert) pairs knocked out in this row
            route_mask_pos="all",               # "all" (every position of the row, default) | "final" (final position only)
            route_mask_mode="reroute")          # "reroute" (default) | "zero"
```

* `reroute`: the masked experts' router logits are set to -inf before the fp32 softmax; top-k and the model's own
  renormalisation then apply as usual, so the token still uses k experts and the next-best expert takes the slot.
  Qwen3-MoE / Mixtral renormalise the top-k, so this is exactly "remove the expert from the menu" (the surviving weights
  are the original probabilities renormalised over the k chosen experts). OLMoE (norm_topk_prob = False): the -inf also
  renormalises the full softmax over the remaining experts, so surviving weights grow by 1 / (1 - p_masked) even when the
  masked expert was not in the top-k (verified: `final_unrouted_reroute_weight_ratio_relerr_max`).
* `zero`: routing unchanged (indices and weights), the masked experts' contributions c_e are 0 (slot dropped, no
  replacement).
* At most E - k experts per (row, layer). Several rows of one pass may carry different masks / positions / modes; rows
  without a mask are computed exactly as before.
* Recorded routing (`PassResult.route_idx / route_w / route_cnorm`, `DiagSpec.route_all_layers`) is the MASKED routing:
  reroute -> the replacement expert sits in the slot; zero -> original indices and weights, `route_cnorm` = 0 and zero
  contribution vectors in the masked slots. Router logits in the diagnostics are the raw router outputs (before the -inf).
* Wavefront rows are never masked: a spawn whose PARENT carries a route mask raises `ValueError`. A masked row may be a
  spawn's source (`clean`): its recorded components are those of the knocked-out run.
* `PassResult.extra["route_masked_rows"]` = number of masked prefill rows.

## E4b — per-head steps in `multi`

```python
SpawnSpec(layer0, parent, src, "multi", steps=(
    (l1, "attn_head", (3, 7)),                      # heads only (int or tuple of head indices)
    (l2, "heads_experts", ((5,), (17, 101))),       # heads AND experts at one layer
    (l3, "coalition_set", (12,)),                   # existing kinds mix freely; layers strictly increasing
))
```

With H_h the final-position head-h output BEFORE o_proj and W_o[:, h] its o_proj column block:

    v_heads = sum_{h in heads} W_o[:, h] (H_h_source - H_h_own)      (fp32, heads ascending)
    h_mid   = h_in_own + bf16(Attn_own + v_heads)
    h_out   = h_mid + bf16(MoE_own(h_mid) + sum_{e in experts} (c_e_source - c_e_own))   (heads_experts only)

* "own" = the parent prefill row for a FIRST step (spawned before the MoE of its layer, like the `attn_head` kind) and the
  live wavefront row for a LATER step; the row's own MoE of that layer is recomputed on h_mid.
* A one-head first step equals the `attn_head` kind bit for bit (Delta and vector); `heads_experts` with no experts equals
  `attn_head`; all heads of a layer equal an `attn_layer` step up to the bf16 rounding of the o_proj outputs.
* Both directions: denoising parent = corrupted row, `clean` = clean row; noising parent = clean row, `clean` = corrupted.
* `sp_vnorm` (first step) / `extra["multi_vnorm"][i]` (one entry per later step): |v_heads| (attn_head) or
  |v_heads + v_experts| (heads_experts); `DiagSpec(spawn_vectors=True)` stores the same vectors, one list entry per step.

## E4c — contribution vectors at the final position

```python
res = eng.run(pre, spawns, diag=DiagSpec(contrib_final_vectors=(40, 41, 44)))
cv = res.extra["diag"]["contrib_final_vectors"][41]   # fp32 [B, k, H] CPU tensor, slot s <-> res.route_idx[41, b, s]
mo = res.extra["diag"]["moe_out_final_fp32"][41]      # fp32 [B, H] MoE output (= cv.sum(1) up to fp32 summation order)
```

c_e = w_e E_e(x) of the prefill rows only (0 for a zero-masked slot). For a vocabulary projection with the final norm frozen
at row b's scale use `DiagSpec(contrib_dla=True)`'s `final_rms[b]` (ext8) or the row's own final residual.

## Numbers (OLMoE vs transformers hooks; dev smoke run, same code as the merged engine)

* Route masks (288 masked rows: 6 mask sets x all/final x reroute/zero on 12 STR units' clean prompts): |dDelta| vs a masked
  HF router max 0.81, mean 0.079 (unmasked baseline engine vs HF: max 0.19, mean 0.063); effect r 0.98 / 0.97 (all,
  reroute / zero), 0.95 / 0.94 (final); masked-layer routing = HF in 0.94 of (row, layer) (unmasked final routing agreement
  0.94); invariants all 1.0: masked experts absent from the recorded routing (reroute), zero mode keeps the first masked
  layer's routing and has route_cnorm = 0 / zero vectors in the masked slots, final-only masks leave every earlier
  position's token log-prob bit-identical, zero mode with an expert not routed at the final position = baseline bit for
  bit; OLMoE reroute weight inflation 1/(1 - p_masked) to 0.6 % (bf16 weights); spawn on a masked parent raises.
* Batch composition: the same unmasked prompt in two different passes differs by up to 0.31 in Delta (mean 0.08), the
  usual bf16 batch dependence of this engine (rows are not bit-reproducible across different pass compositions).
* Head steps: one-head step == attn_head kind (Delta and vector bit for bit); heads_experts without experts == attn_head;
  heads_experts without heads == coalition_set (bit for bit); all heads of a layer vs attn_layer (first / later step)
  max |dDelta| 0.375, mean 0.03 (vectors to 0.2 % relative = o_proj bf16 rounding); all heads at every layer vs the
  source Delta max 0.06; mixed head / expert / MoE / attention configurations vs HF (both directions) max 0.69, mean
  0.071, effect r 0.9993 (the established attn_layer kind on the same units: max 0.64, mean 0.085).
* contrib_final_vectors: sum over slots = fp32 MoE output to 8e-8 relative; norms = route_cnorm to 7e-7; vs HF
  per-expert contributions median 2 % relative (bf16 upstream differences).
* Regression after the merge: verify_olmoe (23 result fields), verify_ext5_engine (225), verify_ext8_engine (496)
  identical to the `*_before_ext9.json` copies (scripts/ext9_regress_compare.py); only verify_ext8's GPU peak-memory
  fields differ (< 1 MB; nondeterministic run to run with an unchanged engine).
* Full verification on the merged engine (6 mixed configurations per unit and direction, 10,000-row head-step stress
  pass): results/verify_ext9_engine_olmoe.json (2026-10-05 07:11Z). Route-mask numbers as above (identical); head steps:
  one-head / heads_experts-without-experts / heads_experts-without-heads identities exact; all heads vs attn_layer max
  |dDelta| 0.125; mixed configurations vs HF max 0.66, mean 0.079, effect r 0.9992 (attn_layer kind calibration max 0.25,
  mean 0.097); stress 10,000 all-layer head/expert rows: 4.8 s, 5.1 GB peak on OLMoE, batch independence mean |dDelta| 0.057.

## Cost notes

* A route mask is a property of a PREFILL row: every masked condition is a full forward of the prompt (no wavefront
  shortcut). Rows with different masks share a pass freely.
* Head steps add one [m_h, head_dim] x [head_dim, H] fp32 matmul per (layer, head) present in the pass; per-head
  pre-o_proj outputs of the wavefront rows are kept only at layers with later head steps.
