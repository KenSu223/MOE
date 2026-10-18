# Engine ext14 (Phase 6, p6-engine) — API for the Phase-6 agents

Status (2026-10-17 06:50Z): all handshakes written (HEADMASK READY, NEURONS READY, MODEL READY for all seven Phase-6 models,
PROTOCOL fp32 for qwen2_57b and qwen2_7b); see `logs/agent_status_p6-engine.md`. Developed in the worktree
`/home/ubuntu/MOE_wt_p6engine` (branch `p6-engine`), merged into `moetrace/{arch,weights,engine,models}.py` only after the
regression was identical. Everything is additive, behind new families / fields / kinds with defaults: the three original
families (qwen3_moe, mixtral, olmoe) and all existing kinds, steps and outputs are unchanged. The module docstring of
`moetrace/engine.py` ("ext14 (Phase 6, p6-engine) additions") is the normative description; this file is the quick reference.

## Model keys (`moetrace/models.py`)

| key | repo | family | L | experts / top-k | n_controls | proto (Phase 6) | prompts |
|---|---|---|---|---|---|---|---|
| `mistral7b` | mistralai/Mistral-7B-v0.1 | mistral (dense) | 32 | 1 / 1 | 0 | `mistral7b_bos` | tokenizer default = BOS (as `mixtral_bos`) |
| `qwen2_7b` | Qwen/Qwen2-7B | qwen2 (dense, q/k/v bias) | 28 | 1 / 1 | 0 | `qwen2_7b` | no BOS; **fp32** (`act_dtype`) |
| `qwen2_57b` | Qwen/Qwen2-57B-A14B | qwen2_moe | 28 | 64 / 8 + shared | 3 | `qwen2_57b` | no BOS; **fp32** (`act_dtype`) |
| `qwen3_1p7b` | Qwen/Qwen3-1.7B-Base | qwen3 (dense, tied) | 28 | 1 / 1 | 0 | `qwen3_1p7b` | no BOS |
| `qwen3_4b` | Qwen/Qwen3-4B-Base | qwen3 (dense, tied) | 36 | 1 / 1 | 0 | `qwen3_4b` | no BOS |
| `qwen3_8b` | Qwen/Qwen3-8B-Base | qwen3 (dense) | 36 | 1 / 1 | 0 | `qwen3_8b` | no BOS |
| `qwen3_14b` | Qwen/Qwen3-14B-Base | qwen3 (dense) | 40 | 1 / 1 | 0 | `qwen3_14b` | no BOS |
| `mixtral22` | mistralai/Mixtral-8x22B-v0.1 | mixtral | 56 | 8 / 2 | 1 | `mixtral22_bos` | entry only (not downloaded; Stage 3 gated) |

Entries also carry `"proto"` and `"dense"` (informational). The engine does not add BOS: token ids come from the pair
files, so BOS is decided when the pairs are built (Mistral: with BOS; Qwen2 / Qwen3: the tokenizer adds none).
Tokenizers: Qwen2 and Qwen3 share `vocab.json` (merges differ only in the header line); Mistral-7B's `tokenizer.model` is
identical to Mixtral-8x7B's; the verification re-tokenised every unit prompt with the model's own tokenizer and found the
stored ids identical (counts in the verification JSONs).

## Dense families (`mistral`, `qwen2`, `qwen3`)

* The MLP is ONE expert with weight 1: `spec.n_experts == 1`, `spec.top_k == 1`, `spec.is_dense == True`, no router.
* Kinds: `layer` = MLP output patch; `expert` with `expert=0` and `coalition_clean / coalition_union / coalition_set((0,))`
  are the same patch (identical Delta); `attn_layer`, `attn_head`, `block`, `resid`, `multi` (all step kinds) as before.
* Recorded routing is trivial: `route_idx == 0`, `route_w == 1`, `route_cnorm = |MLP output|` (final position);
  `DiagSpec.router_logits_final` / `route_all_layers` give zeros / ones. `contrib_dla[l, b, 0]` = DLA of the MLP output.
* `PrefillSpec.route_mask` raises `ValueError` (no router).
* Mistral has `sliding_window = 4096`: the engine asserts that every prompt is shorter (no sliding-window mask).

## Qwen2-MoE (`qwen2_57b`)

* q/k/v biases; 64 routed experts, top-8, `norm_topk_prob = False` (weights are the raw softmax probabilities, cast to
  bf16, as HF); plus the always-on shared expert (intermediate 20480) with weight `sigmoid(w_sg . x)` (bf16, as HF).
* **The shared expert has engine index `E = spec.n_experts` (= 64)** for spawns:
  * `SpawnSpec(l, parent, src, "expert", expert=64)` patches the shared expert's contribution;
  * `coalition_set` may contain 64; `multi` steps `(l, "expert", 64)`, `(l, "coalition_set", (..., 64))` (first and later
    steps, i.e. ext8-style add-back steps) and `heads_experts` steps accept 64 (verified vs transformers, tiny fp32:
    |dDelta| <= 1.7e-6); `expert_neurons` with `expert=64`;
  * `layer` patches routed + shared (the whole MoE block output);
  * `coalition_clean` / `coalition_union` INCLUDE the shared expert (it is active in every run). For a routed-only
    coalition use `coalition_set` with the routed experts.
  * `Engine.n_e_all` = 65 = number of addressable expert indices.
* **Records keep k = 8 routed slots**: `route_idx / route_w / route_cnorm` [L, B, 8], `contrib_dla` [L, B, 8],
  `contrib_final_vectors[l]` [B, 8, H] contain routed experts only. The shared expert is recorded separately:
  * `PassResult.extra["shared_w"]` fp32 [L, B] (final-position gate value) and `extra["shared_cnorm"]` [L, B] (|c_shared|)
    whenever `record_routing=True`;
  * `diag["contrib_dla_shared"]` [L, B] with `DiagSpec(contrib_dla=True)`;
  * `diag["contrib_final_vectors_shared"][l]` [B, H] with `DiagSpec(contrib_final_vectors=...)`;
  * `diag["moe_out_final_fp32"][l]` = routed + shared.
  * Consequence for analyses that sum routed slots (e.g. "all-MoE" DLA, add-back candidate pools, concentration): the
    shared expert must be added explicitly, otherwise the routed sum falls short of the MoE output.
* Route masks on routed experts (0..63): reroute renormalises the full softmax (norm_topk False, like OLMoE), so surviving
  weights grow by 1 / (1 - p_masked) — not comparable to the Qwen3 / Mixtral reroute semantics.
* **Shared-expert knockout** (coordinator request 2026-10-16 20:47Z): `route_mask` may contain `(layer, 64)` with
  `route_mask_mode="zero"`: the shared expert's contribution is 0 at the masked positions (all / final); the recorded gate
  value `extra["shared_w"]` is unchanged and `extra["shared_cnorm"]` is 0 (like a zero-masked routed slot); neuron
  captures get weight 0. `route_mask_mode="reroute"` with expert 64 raises `ValueError` (the shared expert is not routed).
  Routed and shared entries mix freely in one mask (zero mode); with `route_mask_inherit=True` wavefront rows inherit the
  shared knockout like the routed entries. Verified on the tiny Qwen2-MoE against transformers with a forward hook that
  zeroes `mlp.shared_expert`'s output: max |dDelta| < 1.4e-6 for layer / attn_layer (at and below the masked layer) /
  multi / all-MoE / shared-expert patches (masked and unmasked layer); the knockout changes Delta by 0.76 on average.

## Qwen2 numerics protocol (fp32; 2026-10-17)

* `moetrace/models.py`: `qwen2_57b` and `qwen2_7b` carry `"act_dtype": "fp32"`. `Engine(repo)` reads the MODELS entry of
  the repo (`engine.model_protocol`) and then runs fp32 weights (upcast from the bf16 checkpoint at load) and fp32
  activations; `Engine(repo, dtype=torch.bfloat16)` overrides; `PassResult.extra["engine_protocol"]` and `Engine.protocol`
  record what ran. A second flag `"router_fp32": True` (router logits in fp32, rest bf16) exists but is NOT used: it does
  not fix Qwen2-57B (below).
* Why: Qwen2-57B-A14B's routers are nearly uniform (top-8 of 64 experts hold ~13 % of the probability; 8th-9th gaps ~1e-5),
  so bf16 rounding anywhere upstream of a router flips its routed set. On the same prompts, transformers bf16 vs
  transformers fp32 (layers chained one at a time): routed sets agree at 8 % of (layer, position) pairs, |dDelta| mean 3.9,
  sanity answers rank [1, 1, 47, 1, 40874] vs all rank 1 in fp32; fp32 router logits alone: agreement 0.19, ranks
  [1, 1, 42, 1, 124639] (results/verify_ext14_qwen2_57b_chain.json, _router.json). Per layer on identical inputs, a bf16
  layer's routed sets differ from the fp32 layer's at ~98 % of positions at layer 1 although its update differs by 1 %.
  The engine reproduces both behaviours: engine fp32 = transformers fp32 (|dDelta| <= 7e-5, routed sets identical at all
  28 layers, sanity 5/5 rank 1) and engine bf16 = transformers bf16 within transformers' own bf16 noise.
* Qwen2-7B runs in fp32 too (coordinator decision: precision must not be a confound between the twins; its bf16
  engine-vs-transformers deviation is large as well, prefill mean |dDelta| 0.22 vs 0.03 for Mistral-7B).
* Cost and budgets (measured 2026-10-17): Qwen2-57B fp32 pass 143-168 s vs 126-132 s in bf16 (I/O-bound), peak GPU memory
  19.5 GB vs 9.9 GB with 128 prefill rows at T <= 26 + 5,568 spawn rows (19.1 GB with 12 + 520): ~20 GB are resident fp32
  weights (two layers + embeddings), ~0.1 MB per extra spawn row at T ~ 26. Budgets: <= 12,000 spawn rows and <= 256 prefill
  rows per pass at T <= 32 (scale ~ 32 / T), expandable segments, retry at 3/4 on OOM; calibrate on the first cell.
  Qwen2-7B fp32: pass ~10 s for 64 units, peak 7.6 GB; usual budgets.
* Runner caveats: `scripts/ext7_wino_dla.py:89` casts the residual to bf16 before the final norm (DLA readout only);
  `moetrace/ext5_subject.py` (T4 grid) allocates bf16 wavefront rows and would fail with an fp32 engine (grids are out of
  scope after the 2026-10-17 scope cut).

## Head masks (R-causal) — `PrefillSpec.head_mask`

```python
PrefillSpec(ids, true_id, foil_id,
            head_mask=((40, 13), (42, 10), (42, 11)),   # (layer, head) knocked out at EVERY position of this row
            head_mask_mode="zero",                       # "zero" (default) | "mean"
            head_mask_mean=None)                         # mean mode: {(layer, head): vector [head_dim]} or a sequence
                                                         # aligned with head_mask (rounded to bf16)
```

* The per-head attention output BEFORE o_proj is replaced (zero / mean vector) at every position of the row, so
  `attn_heads_final`, `attn_head` patches and head steps see the masked values; o_proj then sums the heads as usual.
* **Unlike route masks, a head-masked row can be a spawn's parent.** Wavefront rows inherit the parent row's mask (the
  listed heads are replaced at the patched final position too). Usage for R-causal: mask the same heads in the clean and
  the corrupted rows, then the usual spawns (parent = masked corrupted row, clean = masked clean row). An `attn_head` patch
  of a masked head is then the identity (verified).
* Rows with different masks / modes share a pass freely; rows without a mask are computed exactly as before.
* Mean vectors: `DiagSpec(attn_heads_mean=(layers...))` → `diag["attn_heads_mean"][l]` fp32 [B, n_heads, head_dim] = each
  prefill row's per-head output averaged over its valid positions (average over rows yourself);
  `DiagSpec(attn_heads_final=...)` gives the final-position values (bf16 [B, n_heads, head_dim]).
* `PassResult.extra["head_masked_rows"]` = number of head-masked prefill rows.
* NOT supported by `ext5_subject.SubjectEngine.run_subject` (the T4 position grid): it ignores `head_mask` (and
  `route_mask`). Use `Engine.run` (T2 / T3 runners) for knockout re-runs.

## Route-mask inheritance (R-causal, coordinator request 2026-10-16 19:22Z)

```python
PrefillSpec(ids, true_id, foil_id, route_mask=((41, 117), (44, 69)), route_mask_pos="all", route_mask_mode="reroute",
            route_mask_inherit=True)   # default False = the ext9 behaviour (a spawn on a route-masked parent raises)
```

* With `route_mask_inherit=True` the row may be a spawn's parent; every wavefront row spawned from it inherits the row's
  mask at the final position (both `route_mask_pos` values mask the final position) at every layer at which the wavefront
  row computes its own MoE: the spawn layer for kinds spawned before the MoE (`attn_layer`, `attn_head`, first
  `heads_experts` steps) and every later layer. reroute -> the masked experts' router logits are -inf for the wavefront
  row too; zero -> their contributions are 0. Mask the clean (source) row identically (`route_mask_inherit` is only needed
  on rows used as parents).
* The default stays False so that existing behaviour (and the ext9 verification field `spawn_on_masked_parent_raises`)
  is unchanged; opting in is explicit per prefill row (`scripts/ext14_causal_run.py` already sets it).
* Head masks need no flag: a head-masked row can always be a parent and its wavefront rows always inherit the mask.
* Verified against transformers hooks with the masks applied in the hooked model (o_proj pre-hooks for heads, router
  forward hooks for reroute / zero at every position): `layer`, `attn_layer`, `multi` (attention + MoE + block steps,
  all-layer MoE) spawns under head masks (zero, mean) and route masks (reroute, zero). Tiny Qwen2-MoE fp32: max |dDelta|
  2.6e-6; negative control with the inheritance disabled: |dDelta| up to 1.27 (attn_layer at the masked layer), 0.42
  (layer), 0.34 (multi). OLMoE bf16: see Verification.

## Neurons (C1) — `DiagSpec.expert_neurons` and kind `expert_neurons`

```python
res = eng.run(pre, spawns, diag=DiagSpec(expert_neurons=((44, 69), (42, 115))))
dn = res.extra["diag"]["expert_neurons"][(44, 69)]
dn["act"]     # fp32 [B, inter_e] = SiLU(W_gate x) * (W_up x) at the final position (bf16 values)
dn["w"]       # fp32 [B] weight applied to the expert output (0 = not routed; dense 1; shared = sigmoid gate)
dn["routed"]  # bool [B]; rows where the expert is not routed get the activations it WOULD compute (routed False)

SpawnSpec(44, parent=corrupt_row, clean=clean_row, kind="expert_neurons", expert=69, neurons=(3, 17, 250, ...))
```

* `v = sum_{i in N} (w_src a_i,src - w_own a_i,own) W_down[:, i]` (fp32), added like the `expert` kind
  (`h = h_pre_moe_own + bf16(MoE_own + v)`); everything else as in the parent. Single-layer kind (not a multi step);
  works in both directions (parent / clean as for every kind).
* N = all neurons equals the `expert` kind up to the bf16 rounding of the expert's down projection; v is exactly linear in N
  (fp32). Dense: `expert=0` = the MLP (Mistral 14336, Qwen2-7B 18944, Qwen3 9728 / 12288 / 17408 neurons); Qwen2-MoE:
  routed experts 2560 neurons, `expert=64` = shared expert 20480 neurons; Qwen3-30B 768; Mixtral 14336.
* Neuron scores need W_down rows: `eng.store.expert_tensors(layer, expert)` → (W_gate, W_up, W_down) CPU tensors
  (dense: expert 0; Qwen2-MoE: 64 = shared).
* Cost: one [m, inter] x [inter, H] fp32 matmul per expert per layer with expert_neurons spawns (chunks of 4096 rows).

## Other additions

* `DiagSpec.resid_layers=(l, ...)`: full residual stream after layer l (bf16 [B, T, H], CPU); -1 = embeddings. Memory-heavy;
  used by the per-layer verification.
* `Engine(repo, dtype=torch.float32)`: fp32 activations / weights (tiny-model verification only).
* `moe_forward(..., extras=dict)` receives the shared expert outputs and neuron captures (internal; the 5-tuple return is
  unchanged, so `ext5_subject` keeps working).
* `ArchSpec`: `is_dense`, `has_shared`, `shared_inter`, `sliding_window`, bias / shared-expert key templates.

## Verification (scripts/ext14_engine_verify.py, gate scripts/ext14_engine_gate.py; JSONs results/verify_ext14_*.json)

Method: transformers 5.16 (eager attention) with forward hooks implementing every configuration (final-position
self_attn / mlp / decoder-layer replacements, o_proj pre-hooks for head steps and head masks, router hooks for route masks,
mlp hooks for expert sets and neuron subsets) vs one engine pass. Per unit ~70-100 configurations: layer / attn_layer /
block / resid at 5 layers (denoising) and 2 layers (noising), single heads, expert 0 (dense) or a clean-routed expert and
the shared expert, coalition_set, expert_neurons (random half, all), multi (attention + MoE, MoE + block + attention, head
steps + heads_experts, all-layer MoE), zero; head masks (zero, mean) and route masks (reroute, zero; inherited) followed by
layer / attn_layer / attn_head / multi spawns. Metrics (log p, p, rank, KL) and diagnostics (attn_heads_mean, resid_layers,
expert_neurons) compared too. Per-layer mode: engine identities on the real checkpoint, sanity prompts, transformers decoder
layers applied to the engine's own residual stream (all positions; routing at every position).

| check | result |
|---|---|
| Tiny random checkpoints, fp32 (mistral, qwen2, qwen3 tied, qwen2_moe + shared), 8 units, 424-584 spawns each | max |dDelta| 2.2e-6 - 4.2e-6 (prefill 1.7e-6 - 5.4e-6); every kind, mask and diagnostic exact to fp32; per-layer harness update relerr <= 3.5e-7, routing identical |
| OLMoE-1B-7B bf16 (real MoE family; 12 STR units, 888 spawns) | rescue r 0.996 overall, max |dDelta| 0.47, mean |dDelta| per group 0.04-0.11 (prefill rows 0.05 = the bf16 noise floor); groups with effects >= 0.3: r 0.99-1.00; head-mask multi r 0.993 / 0.995, route-mask multi r 0.996 / 0.997, all-MoE under route masks r 0.999 / 1.000; masks change Delta by 0.20 (heads) / 0.27-0.31 (routes); neuron linearity 2e-7, neurons(all) vs expert |dDelta| <= 0.06 |
| Regression (main tree after the merge, 2026-10-16 20:01Z) | verify_olmoe (23 fields), verify_ext5_engine (225), verify_ext8_engine (496), verify_ext9_engine (316): identical to the committed JSONs (only GPU peak-memory fields differ); Phase-5 sweep passes p5_f2e_qwen3 (18,688 rows) and p5_f2e_mixtral_bos (7,056 rows) incl. routing: bit for bit. Copies: results/verify_ext14_regress_main_*.json, results/verify_ext14_regress_sweep_*_main.json |
| Checkpoints (bf16, vs transformers bf16) | table below; gates scripts/ext14_engine_gate.py (logs/ext14_engine_gate{,3,4}_<key>.log) |
| fp32 exactness on real weights | engine fp32 vs transformers fp32: Qwen2-7B full spawn plan (offloaded, 168 spawns) rescue r 1.0000000, max |dDelta| 3e-5; Qwen3-1.7B full plan r 1.0000; chain mode (forward + routing, transformers layers chained in fp32): Qwen2-57B |dDelta| <= 7e-5, routed sets identical at all 28 layers; Qwen3-14B 1.4e-5; Qwen2-7B 7.6e-6 |
| bf16 noise floor (transformers bf16 vs transformers fp32, same prompts) | Qwen3-14B |dDelta| mean 0.085 (engine bf16 vs fp32 0.062); Qwen2-7B 0.56 (engine 0.43), spawn rescues r vs fp32 transformers 0.970 / engine 0.979 (results/verify_ext14_qwen2_7b_noise.json); Qwen2-57B 3.86 (engine 3.99), routed-set agreement 0.08 / 0.08 |
| Precision effect on one cell (e1_a2e, 64 units, engine bf16 vs engine fp32) | Qwen2-7B: rescue r 0.91, attention share 0.580 vs 0.577; Qwen2-57B: rescue r -0.04, attention share 0.678 vs 0.582, final routed sets equal 0.000 (results/verify_ext14_qwen2_{7b,57b}_vsfp32_e1a2e.json) -> both Qwen2 twins run in fp32 (PROTOCOL lines 2026-10-17 06:46Z) |

Checkpoint verifications (bf16; the Qwen2 rows are the pre-protocol bf16 checks):

| model | full mode vs transformers (spawns; rescue r; max / mean-of-groups |dDelta|; prefill max) | per-layer update relerr (max; routing) | engine identities | sanity (answer ranks) |
|---|---|---|---|---|
| mistral7b | 684; r 0.9996; 0.188 / 0.029; 0.0625 | L1 0.00052, L16 0.0095, L31 0.0039 | union/set(all) vs layer 0; neurons(all) vs expert 0.062; linearity 3.3e-07 | Paris#4 Paris#1 oxygen#1 cold#1 Thursday#1 |
| qwen2_7b | 168; r 0.9930; 0.562 / 0.149; 0.375 | L1 0.01, L14 0.01, L27 0.096 | union/set(all) vs layer 0; neurons(all) vs expert 0.062; linearity 2.9e-07 | Paris#1 Paris#1 oxygen#1 cold#1 Thursday#1 |
| qwen3_1p7b | 684; r 0.9894; 0.25 / 0.074; 0.25 | L1 0.0046, L14 0.015, L27 0.0044 | union/set(all) vs layer 0; neurons(all) vs expert 0.12; linearity 2.1e-07 | Paris#1 Paris#1 oxygen#1 cold#1 Thursday#1 |
| qwen3_4b | 684; r 0.9931; 0.375 / 0.071; 0.375 | L1 0.005, L18 0.017, L35 0.006 | union/set(all) vs layer 0; neurons(all) vs expert 0.12; linearity 5.1e-07 | Paris#1 Paris#1 oxygen#1 cold#1 Thursday#1 |
| qwen3_8b | 684; r 0.9902; 0.375 / 0.079; 0.25 | L1 0.0069, L18 0.018, L35 0.0038 | union/set(all) vs layer 0; neurons(all) vs expert 0.12; linearity 3.0e-07 | Paris#1 Paris#1 oxygen#1 cold#1 Thursday#1 |
| qwen3_14b | 252; r 0.9973; 0.375 / 0.086; 0.25 (HF offloaded to CPU RAM, light plan) | L1 0.0095, L20 0.014, L39 0.018 | union/set(all) vs layer 0; neurons(all) vs expert 0.12; linearity 2.9e-07 | Paris#1 Paris#1 oxygen#1 cold#1 Thursday#1 |
| qwen2_57b | n/a (per-layer only) | L1 0.0059 (0.08), L13 0.065 (0.14), L27 0.014 (0.93) | union/set(all) vs layer 0; neurons(all) vs expert 0.047; linearity 1.5e-07; shared expert vs set 0, shared neurons(all) 0.19 | Paris#1 Paris#1 oxygen#50 cold#1 Thursday#86 |

Notes: Qwen3-14B full mode with transformers offloaded to CPU RAM (light plan; its first attempt crashed on a light-plan bug,
fixed). Qwen2-57B has no full-mode transformers reference (107 GB); its evidence is the tiny fp32 model (every kind), the
per-layer check (with an fp32 reference layer: engine-vs-fp32 relerr = transformers-vs-fp32 relerr at L1 / L13 / L27) and
the chain mode. Its bf16 per-layer routing agreements (0.08 / 0.14 at L1 / L13) and sanity ranks (oxygen #50, Thursday #86)
are the model's bf16 fragility, not an engine error (transformers bf16 behaves the same); in fp32 all sanity answers rank 1.
Qwen3-1.7B: rescue r 0.989 overall in bf16 (just under 0.99), r 0.9894 on the competent units; accepted with the noise-scaled
gate (transformers bf16 vs fp32 noise floor from its chain run) and exact fp32 agreement.

## Runner compatibility (audit 2026-10-16, read-only; line numbers of the main tree at 4143e09)

Resolved inside the engine (no runner change needed):
* `moetrace/arch.py` dense configs have no `num_experts_per_tok` (KeyError in `load_spec`) — fixed (top_k = 1).
* `moetrace/ext5_subject.py:286-342` (T4 grid executor) has its own layer loop but uses `Engine._qk` and `moe_forward`;
  q/k/v biases, the dense MLP and the Qwen2-MoE shared expert live inside those two functions, so the grid computes the
  new families correctly (layer / attention / block / resid patches include the shared expert).
* `engine._set_mask` width: expert index 64 (shared) is accepted (`Engine.n_e_all`).

Still open (owners: p6-data / p6-analysis; severity: B = blocks, S = silently wrong / not comparable, C = cosmetic):
* `ext5_subject.run_subject` ignores `PrefillSpec.head_mask` / `route_mask` (S if used for knockouts; use `Engine.run`);
  `_build_v_at` expert lookup `topi_r == ex` never matches the shared expert 64 (S: a shared-expert patch at a non-final
  position is silently zero; `layer` patches are fine).
* Model / protocol tables: `scripts/ext7_wino_scan.py:33,53` PROTOS (B), `ext7_ioi_scan.py:27`, `ext7_ioi_attn.py:30`,
  `moetrace/ext7_controls.py:52,373,795` (B for IOI), `ext13_p5_scan.py:24,67` (B), `ext13ab_run.py:137` choices qwen3|mixtral
  (B), `moetrace/ext13ab.py:43-47` PROTO_MODEL / MODEL_PROTO / POOL / NOISE_ROWS (B), `ext13ab.py:124,187` new protocols
  dropped silently (registry empty), `scripts/ext13ab_analyze.py:366-368,546` (B), `moetrace/ext13ci.py:48-50,137-145,159`
  PROTOS / MODEL_OF / `[p5-data] BASE READY` regex / p5 paths (B), `scripts/ext13ci_heads_run.py:42,58`,
  `ext13ci_circuit_run.py:67-114` (rows per pass and budget fall through to the Mixtral values; `EXPERT_POOL[model]` B),
  `moetrace/ext10_circuit.py:187,325` EXPERT_POOL (B), `moetrace/ext13ko.py:55-60,101-104` PROTO (B),
  `ext13wr.py:38,53`, `scripts/ext13wr_analyze.py:286`, `ext13wr_text.py:45`, `ext13ko_analyze.py:313,570` (B; `wg=None`
  -> TypeError), `moetrace/ext9_knockout.py:63-79` (B), `scripts/ext13_p5_jobs.py:12,37` L = {qwen3: 48, mixtral: 32} (B).
  `ext7_wino_scan` / `ext7_wino_casesets.py:161-173` / `ext7_pairs.py:102` own pools only for qwen3 / mixtral_bos (B for
  `--own-pool <new>`). `moetrace/ext14_analysis.py` (p6-analysis) already maps the new protocols.
* The generic runners `scripts/ext7_wino_{sweep,joint,dla,grid,heads,expert}.py` take `--model <MODELS key>` (no choices;
  unknown key -> KeyError) and read L / nH / k from the spec: they run on every new model. Pass the MODELS key
  (`mistral7b`, not `mistral7b_bos`). Default row budgets are not model-scaled (sweep 60000, joint 30000, heads 50000):
  calibrate on Qwen3-14B (hidden 5120, MLP 17408) and Qwen2-57B (64 experts + shared 20480) (C / OOM risk).
* Dense models (n_experts = 1, top_k = 1):
  * `scripts/ext7_wino_expert.py:221-232` emits identical rows (layer = expert 0 = coalition_*) and no `expert_scaled`
    rows (wasteful, OK); expert-level analyses are degenerate.
  * `scripts/ext7_wino_analyze.py:225` -> `analysis.all_active_rank` raises KeyError with one active expert
    (`analysis.py:200`) (B); `:186,210-213` n_controls = 0 (C); `protocol.active_controls` with n = 0 gives Spec = NaN
    (pattern never "A") (S).
  * `moetrace/ext8_addback.py:96,214` K = L * top_k = L candidates (the MLP layers): AUC over log K on a different scale (S);
    `ext13ko.py:366-369` `rng.choice([])` IndexError (knockouts need a router; B). Dense tier-2 = neurons (C1).
* Qwen2-57B (64 experts top-8 + shared):
  * expert-level tables never contain the shared expert (`ext7_wino_expert.py`, `analysis.py`, `ext1_analysis.py`);
    `coalition_*` rows include it, so coalition != sum of routed experts (S: report the shared expert separately, e.g.
    `expert=64` rows).
  * `ext8_addback.py` candidates are routed experts only while the all-MoE ceiling includes the shared expert, so greedy /
    oracle / all-active curves end short of the ceiling (S: add 64 to the candidate pool or subtract the shared part).
  * `scripts/ext13ko_run.py:217` `moe_dla_masked` sums routed slots only (C: base - knockout difference unaffected);
    reroute semantics with norm_topk False differ from Qwen3 / Mixtral (S, interpretive).
* Interior-layer rules use a fixed offset `L - 5` / `L - 8` / `L - 4` (`ext7_wino_analyze.py:100,197`,
  `ext8_addback.py:266-269`, `ext11_writer.py:240`): 18 % of 28 layers vs 10 % of 48 (S across models; use l / L).
* `moetrace/ext11_writer.py:487` maps every non-qwen3 model to `mixtral_bos` vocabulary ids (correct for mistral7b, S for
  qwen2_7b / qwen2_57b / qwen3_*); `scripts/ext13wr_analyze.py:180` BPE renderer only for `qwen3` (C).
