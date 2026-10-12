## Direction 13-AtP (Phase 5, P5-5): attribution patching against activation patching on OLMoE

**Summary.** On OLMoE-1B-7B-0125 (bf16), for two STR tasks (WinoGrande option swap and IOI, 64 + 64 pairs = 256 directed
cases each), the gradient estimate AtP of single final-position patches agrees with the engine's activation patching
(pooled r) at 0.96 / 0.97 for experts,
0.96 / 0.99 for MoE layers,
0.83 / 0.95 for heads and
0.86 / 0.98 for attention layers
(WinoGrande / IOI). A second exact patch with other bf16 numerics (transformers instead of the engine) reaches
0.97 / 0.96 (experts) and
0.92 / 0.94 (heads), so on IOI AtP is as good
as the noise floor for experts, heads and MoE layers, and on WinoGrande it falls short, most for heads and attention layers. The failures are systematic, not
random: AtP is nearly exact in the last quarter of the network (r 0.95-1.00, slope 0.85-1.01) and underestimates effects
that pass through several later layers (slope 0.01-0.32 in L0-3, 0.11-0.63 in L4-7); the strongest WinoGrande head, L6H4 (+0.93 logits per row), is only
AtP's rank 6. Routing switches are the MoE-specific part of the error: 0.34 / 0.32
of single-expert patches change at least one downstream top-k slot, and freezing the downstream routing removes
0.48 / 0.44 of AtP's squared error for experts
(0.32 / 0.30 for heads). AtP* (the next
attention softmax and the next router computed exactly) helps little; EAP-IG is worse than AtP for single patches. As a
static add-back ordering AtP equals the oracle (each row's own single-patch ranking): expert AUC(k ≤ 16)
0.669 vs 0.670 (WinoGrande), 0.198 vs 0.184 (IOI); heads + experts 0.742 vs 0.764,
0.852 vs 0.855. bf16 is not a limitation (AtP bf16 vs fp32 r ≥ 0.998).
**Recommendation: do not build reverse layer streaming for the final-position sweeps** (activation patching there is
cheap in the engine and exact); build it only if the project needs attribution at every position or for edges, or 10×
larger case sets, and then use AtP as a screen for experts and late components with activation patching as the check.

### What was run

- **Model and protocol.** OLMoE-1B-7B-0125 (16 layers, 16 heads, 64 experts, top-8 without renormalisation), bf16,
  tokenizer defaults (no BOS), STR pairs (moetrace/ext7_pairs.py contract), denoising at the final position: one component
  of the corrupted run is set to its clean value, Δ = logit(true) − logit(foil), rescue = Δ_patched − Δ_corrupt.
- **Tasks** (competence ≥ 64 / 64 pairs with margin ≥ 1 in both directions):
  - WinoGrande option swap in the Phase-3 format: 1325 / 3809 OLMoE margin pairs
    (results/wino_olmoe); 64 discovery + 64 validation pairs drawn from the Phase-3 discovery / validation pairs (Phase-3
    split kept, random.Random(13)); mean drop 8.05 logits.
  - IOI with the S2 → IO corruption: the 1,600 Phase-3 items re-tokenised for OLMoE (all pass the build rules),
    1571 / 1600 margin pairs; 64 + 64 pairs from the Phase-3 IOI discovery / validation pairs; mean drop 8.87.
  - Data: results/p5_atp_data/ (pairs_ioi_s2io_olmoe.parquet, scan_ioi_olmoe.parquet, case_sets_{wino,ioi}.json).
- **Components per directed case** (all at the final position): every expert routed in the clean or the corrupted run at
  each layer (about 140 per row; others have a zero write), all 256 heads, 16 attention-sublayer outputs, 16 MoE outputs.
- **Truth.** Engine activation patching, kinds expert, attn_head, attn_layer, layer (scripts/ext13atp_engine.py truth;
  110k / 108k spawn rows per task). A second exact patch in transformers with an fp32 read-out of Δ from the final-norm output
  (no bf16 logit rounding) serves as the noise ceiling: two exact computations that differ only in bf16 numerics.
- **Estimates** (scripts/ext13atp_grad.py, moetrace/ext13atp.py; transformers, eager attention and experts, bf16 weights,
  gradients w.r.t. activations only): with δ_n = a_n(clean) − a_n(corrupt) the node's write into the residual stream,
  - AtP = δ_n · ∂Δ/∂h at the corrupted run (h = residual after the attention sublayer for heads / attention layers, after the
    MoE sublayer for experts / MoE layers);
  - EAP-IG = δ_n · mean over α = k/8 (k = 1..8) of the gradient on the straight line between the corrupted and the clean input
    embeddings (Hanna et al. 2024); m = 16 and the midpoint rule were also computed (tables);
  - AtP*: Kramár et al.'s QK fix computes the attention softmax that a query / key node feeds exactly and linearises after
    it. Our nodes are sublayer outputs; the softmaxes they feed are the next attention sublayer's at the final position and,
    in an MoE, the next router (softmax + top-k). AtP* here = the node's write propagated exactly through the next
    attention sublayer and the next router / MoE (QK + router fix), linearised after that; the full curve over the number of
    exactly computed sublayers is reported (cut curve). GradDrop was not run.
  - DLA = δ_n · γ ⊙ (W_U[true] − W_U[foil]) / rms(h_final, corrupt) (direct path only).
- **Routing.** Every node was also patched in transformers with the final position's top-k recorded at every layer; a node
  "switches" when any downstream top-k set differs from the unpatched corrupted row of the same batch. Switching nodes were
  re-patched with the downstream top-k frozen to the corrupted run's (weights recomputed from the router softmax), so
  exact − AtP = (exact − frozen) [routing switch, invisible to a gradient] + (frozen − AtP) [smooth non-linearity].
- **Numerics.** fp32 check on CPU with the checkpoint's native fp32 weights: gradients and estimates for 16 validation cases
  per task, fp32 activation patches for 8.
- **Add-back** (scripts/ext13atp_engine.py addback): validation rows, engine `multi` spawns, static orderings oracle (the
  row's own engine single-patch rescue), AtP, AtP*, EAP-IG, DLA, pop (discovery mean truth) and 3 random permutations;
  pools = clean-active experts (ext8 convention, 128 per row) and heads + clean-active experts (Direction 10, 384 per row);
  k = 1..64; r(k) = rescue / drop; AUC = mean r(k) over k = 1..16 (linear interpolation between grid points).

Checks: engine vs transformers Δ mean absolute difference 0.081 / 0.114 logits (WinoGrande /
IOI); Σ expert AtP of a layer = MoE-layer AtP (r 1.00000), Σ head AtP = attention-layer AtP
(r 1.00000); the patched run's first cut equals the gradient AtP (r 0.9997).

### 1. Agreement per component type

Pooled over all (row, component) pairs (WinoGrande 256 rows, IOI 256); truth = engine activation patching; ceiling = the
transformers activation patch scored against the engine; slope = OLS slope of AtP on the truth; top-k overlap and recovery
(Σ truth over AtP's top k / Σ truth over the truth's top k) per row, k = 10 for experts / heads and 3 for layers; population
= mean over the 256 rows per component (table results/tables/ext13atp_agreement_main.md; per-row Kendall, Spearman CIs and
all estimator variants in ext13atp_agreement_rows.md, ext13atp_agreement_pooled.md, ext13atp_agreement_population.md).

| Task | Component | Ceiling r (2nd exact patch) | AtP r / slope | AtP* r | EAP-IG r | DLA r | AtP per-row Spearman (ceiling) | AtP top-k overlap / recovery | AtP pop. Spearman / top-k |
|---|---|---|---|---|---|---|---|---|---|
| WinoGrande | experts | 0.973 | 0.956 / 0.87 | 0.961 | 0.917 | 0.819 | 0.79 (0.80) | top-10 0.86 / 0.98 | 0.77 / 0.90 |
| WinoGrande | heads | 0.921 | 0.833 / 0.55 | 0.857 | 0.786 | 0.351 | 0.58 (0.56) | top-10 0.79 / 0.93 | 0.82 / 1.00 |
| WinoGrande | attention layers | 0.992 | 0.856 / 0.56 | 0.908 | 0.825 | 0.283 | 0.89 (0.97) | top-3 0.71 / 0.89 | 0.95 / 0.67 |
| WinoGrande | MoE layers | 0.997 | 0.958 / 0.88 | 0.970 | 0.942 | 0.814 | 0.93 (0.98) | top-3 0.89 / 0.98 | 0.98 / 1.00 |
| IOI | experts | 0.957 | 0.970 / 0.95 | 0.970 | 0.951 | 0.918 | 0.57 (0.47) | top-10 0.60 / 0.85 | 0.50 / 1.00 |
| IOI | heads | 0.937 | 0.946 / 0.86 | 0.945 | 0.931 | 0.782 | 0.49 (0.40) | top-10 0.93 / 0.99 | 0.76 / 1.00 |
| IOI | attention layers | 0.997 | 0.978 / 0.94 | 0.980 | 0.983 | 0.879 | 0.90 (0.92) | top-3 0.92 / 0.99 | 0.89 / 1.00 |
| IOI | MoE layers | 0.992 | 0.990 / 0.99 | 0.992 | 0.971 | 0.930 | 0.89 (0.89) | top-3 0.86 / 0.96 | 0.80 / 1.00 |

- Per-row Spearman over all 140 experts or 256 heads is low for every method, the ceiling included (heads
  0.56 / 0.40): most components have
  effects at the bf16 resolution of the logits (40 % of the engine's WinoGrande head truths are exactly 0). The informative
  metrics are top-k overlap, recovery and the pooled r; on all of them AtP is within a few points of the ceiling on IOI and
  for WinoGrande experts / MoE layers.
- Per-row top-1: AtP's best expert is the truth's best in 0.86 / 0.88 of rows and
  in its top 3 in 1.00 / 0.96; best head
  0.61 / 0.90 (top 3: 0.93 /
  1.00); the second exact patch reaches 0.94 / 0.98
  for heads (results/tables/ext13atp_top1_hits.md).
- AtP underestimates large effects (slope 0.55 for WinoGrande heads,
  0.56 for attention layers; scatter results/figures/ext13atp_scatter_atp.png).

### 2. Where AtP fails: distance from the read-out

Pooled r / slope of AtP against the engine truth by layer band (results/tables/ext13atp_by_layer.md):

| Task | Component | L0-3 r / slope | L4-7 r / slope | L8-11 r / slope | L12-15 r / slope | mean abs truth L0-3 / L12-15 |
|---|---|---|---|---|---|---|
| WinoGrande | experts | 0.28 / 0.08 | 0.65 / 0.44 | 0.94 / 0.75 | 0.98 / 0.95 | 0.050 / 0.274 |
| WinoGrande | heads | 0.24 / 0.06 | 0.66 / 0.31 | 0.91 / 0.63 | 0.95 / 0.85 | 0.045 / 0.065 |
| WinoGrande | attention layers | 0.56 / 0.32 | 0.72 / 0.41 | 0.84 / 0.55 | 0.97 / 0.89 | 0.093 / 0.517 |
| WinoGrande | MoE layers | 0.54 / 0.27 | 0.74 / 0.63 | 0.92 / 0.75 | 0.97 / 0.94 | 0.092 / 1.862 |
| IOI | experts | 0.14 / 0.02 | 0.44 / 0.21 | 0.85 / 0.71 | 1.00 / 1.01 | 0.069 / 0.182 |
| IOI | heads | 0.09 / 0.01 | 0.32 / 0.11 | 0.93 / 0.75 | 0.99 / 0.97 | 0.068 / 0.186 |
| IOI | attention layers | 0.36 / 0.11 | 0.61 / 0.39 | 0.94 / 0.70 | 0.99 / 0.98 | 0.091 / 2.105 |
| IOI | MoE layers | 0.29 / 0.11 | 0.74 / 0.55 | 0.94 / 0.87 | 1.00 / 1.01 | 0.092 / 1.200 |

- AtP is close to exact in the last four layers and degrades with every layer between the component and the read-out.
- Early-layer single-component effects are small and partly below the bf16 noise of activation patching itself: in the
  fp32 patches (8 cases per task) the L0-3 expert / head effects are 0.01-0.02 logits on average, the bf16 truth correlates
  with the fp32 truth at only 0.07-0.50 there, and fp32 AtP still underestimates them (slope 0.1-0.3). Neither method
  resolves them in bf16.
- Mid-layer components with large indirect effects are the costly case. WinoGrande population heads (mean truth per row):
  L6H4 0.93 (AtP 0.28, rank 6; EAP-IG 0.60, rank 2); L11H6 0.92 (AtP 0.62, rank 1; EAP-IG 0.72, rank 1); L8H2 0.72 (AtP 0.36, rank 5; EAP-IG 0.56, rank 3); L9H7 0.68 (AtP 0.37, rank 4; EAP-IG 0.51, rank 5). AtP's population top 5 misses the top head; EAP-IG ranks it 2nd
  (results/tables/ext13atp_population_top.md). Experts: the WinoGrande top-3 experts (L12E11, L13E40, L11E42) and the IOI
  top-5 keep their ranks under AtP; L9E10 drops from rank 4 to 9.

### 3. Routing switches (the MoE caveat)

Transformers patches (bf16), all 256 rows; "large switch effect" = |exact − frozen| > max(0.1, 0.25 |exact|) (results/tables/ext13atp_routing.md):

| Task | Component | Nodes that switch ≥ 1 downstream top-k slot | Slots switched per switching node | r(AtP, exact) all / no switch / switch | r(AtP, frozen-routing patch) | Share of AtP's squared error removed by freezing | Switching nodes with a large switch effect |
|---|---|---|---|---|---|---|---|
| WinoGrande | experts | 0.34 | 1.6 | 0.958 / 0.985 / 0.933 | 0.978 | 0.48 | 0.27 |
| WinoGrande | heads | 0.25 | 1.7 | 0.843 / 0.923 / 0.809 | 0.884 | 0.32 | 0.26 |
| WinoGrande | attention layers | 0.68 | 2.6 | 0.856 / 0.871 / 0.860 | 0.866 | 0.11 | 0.23 |
| WinoGrande | MoE layers | 0.71 | 2.2 | 0.958 / 0.978 / 0.950 | 0.963 | 0.15 | 0.23 |
| IOI | experts | 0.32 | 1.6 | 0.977 / 0.988 / 0.961 | 0.987 | 0.44 | 0.35 |
| IOI | heads | 0.27 | 1.6 | 0.955 / 0.970 / 0.942 | 0.968 | 0.30 | 0.33 |
| IOI | attention layers | 0.62 | 2.0 | 0.979 / 0.989 / 0.976 | 0.980 | 0.07 | 0.32 |
| IOI | MoE layers | 0.62 | 1.9 | 0.992 / 0.997 / 0.987 | 0.995 | 0.37 | 0.36 |

- A third of single-expert and a quarter of single-head patches switch downstream routing (1.6 slots on average among the
  switching nodes); layer-sized patches switch in 62-71 % of rows. Switch rates fall with depth (WinoGrande experts: 0.47 at L0 to 0.10 at L14), since
  fewer routers lie downstream.
- Freezing the downstream top-k removes 29-49 % of AtP's squared error on experts and heads (38-65 % among the switching
  nodes); for whole layers the smooth non-linearity dominates (7-37 % removed).
- Part of the switching is bf16 noise amplification: in fp32 fewer nodes switch (see section 6). A gradient sees routing
  weights but not set changes; the frozen-routing patch is what AtP approximates.

### 4. AtP* and the cut curve

Pooled r with the transformers patch after s exactly computed sublayers (s = 0 AtP, s = 2 AtP*, s = 8):
WinoGrande experts 0.96 → 0.96 → 0.99, heads 0.84 → 0.88 → 0.95, attention layers 0.86 → 0.91 → 0.98, MoE layers 0.96 → 0.97 → 0.99; IOI experts 0.98 → 0.98 → 0.99, heads 0.96 → 0.96 → 0.98, attention layers 0.98 → 0.98 → 1.00, MoE layers 0.99 → 1.00 → 1.00 (results/figures/ext13atp_cut_curve.png). The QK + router fix gains
0.00-0.05 in r; the non-linearity is spread over many downstream layers, so a local fix cannot remove it. AtP* per-row top-1
for WinoGrande heads 0.69 (AtP 0.61).

### 5. EAP-IG

EAP-IG (input-embedding interpolation, m = 8) agrees less with single activation patches than AtP for every component type
except IOI attention layers (e.g. WinoGrande experts r 0.92 vs 0.96;
IOI MoE layers per-row Spearman 0.66 vs 0.89). Its gradients
are taken in increasingly clean contexts, whereas a single patch keeps the corrupted context downstream; the gradient at the
clean run (AtP at α = 1) is worst of all (WinoGrande experts r 0.79).
EAP-IG does better on large mid-layer components (WinoGrande L6H4) and as an add-back ordering (section 7), i.e. for joint
patches.

### 6. Numerics (bf16 vs fp32)

- WinoGrande: AtP bf16 vs fp32 r 0.9983 (relative L2 0.056), EAP-IG r 0.9989, gradient cosine at the corrupted run median 0.9996 (min 0.9968); fp32 patches (8 cases): r(AtP, exact) fp32 / bf16 experts 0.979 / 0.977, heads 0.833 / 0.834; engine vs fp32 truth r experts 0.983, heads 0.952; switching nodes fp32 / bf16 experts 0.25 / 0.29, heads 0.15 / 0.23.
- IOI: AtP bf16 vs fp32 r 0.9994 (relative L2 0.033), EAP-IG r 0.9996, gradient cosine at the corrupted run median 0.9983 (min 0.9953); fp32 patches (8 cases): r(AtP, exact) fp32 / bf16 experts 0.986 / 0.960, heads 0.984 / 0.944; engine vs fp32 truth r experts 0.969, heads 0.967; switching nodes fp32 / bf16 experts 0.22 / 0.35, heads 0.11 / 0.29.

bf16 gradients are accurate; agreement with the truth is limited by the method (WinoGrande heads 0.83 in both precisions),
and the bf16 noise of the truth itself (IOI: AtP fp32 agrees with the fp32 truth at 0.98, better than the bf16 engine does).

### 7. AtP as an add-back ordering

AUC = mean restored fraction r(k) over k = 1..16 on the validation rows, pair-bootstrap CI (results/tables/ext13atp_addback_auc.md,
results/figures/ext13atp_addback.png):

| Task | Pool | oracle | AtP | AtP* | EAP-IG | DLA | pop (discovery truth) | random (3 perms) |
|---|---|---|---|---|---|---|---|---|
| WinoGrande | experts | 0.670 [0.645, 0.694] | 0.669 [0.644, 0.693] | 0.670 [0.645, 0.693] | 0.677 [0.650, 0.702] | 0.659 [0.633, 0.684] | 0.544 [0.502, 0.582] | 0.084 |
| WinoGrande | heads + experts | 0.764 [0.748, 0.778] | 0.742 [0.726, 0.757] | 0.748 [0.733, 0.763] | 0.758 [0.744, 0.773] | 0.703 [0.686, 0.720] | 0.648 [0.610, 0.683] | 0.045 |
| IOI | experts | 0.184 [0.171, 0.196] | 0.198 [0.187, 0.210] | 0.195 [0.184, 0.206] | 0.187 [0.174, 0.200] | 0.085 [0.072, 0.097] | 0.162 [0.149, 0.175] | -0.010 |
| IOI | heads + experts | 0.855 [0.833, 0.878] | 0.852 [0.830, 0.877] | 0.852 [0.830, 0.876] | 0.866 [0.844, 0.891] | 0.789 [0.775, 0.803] | 0.813 [0.801, 0.826] | 0.024 |

- AtP, AtP* and EAP-IG orderings are at the oracle (AUC differences within ±0.022, CIs overlapping); DLA is 0.01-0.07 below
  the oracle except for IOI experts (half the oracle), which act indirectly; the discovery-population ordering is below the
  row-wise orderings.
- At k = 10: WinoGrande heads + experts oracle 0.86, AtP 0.84,
  EAP-IG 0.86; IOI 0.99 / 0.98 /
  0.99. All-MoE ceiling WinoGrande 0.89,
  IOI -0.12 (patching every MoE output at the final position hurts IOI).

### 8. Recommendation on reverse layer streaming for Qwen3 / Mixtral

- **What AtP would buy at the final position.** Our engine already runs every single-component patch at the final
  position cheaply: on OLMoE 110,072 truth spawns took 18-104 s per task. On Qwen3 the same sweep costs ≈ 5 GPU-min per
  256 rows (Phase 5 logs: all-layer single experts, 631,536 spawns for 1,024 rows, ≈ 11 min; all 1,536 single heads for 256
  rows, 2 passes, ≈ 2 min), about what a forward plus a reverse stream (two checkpoint reads) would cost (estimate). AtP would replace these with one forward and one reverse
  stream, but is less accurate exactly where the open questions are (mid-layer heads and attention, components far from the
  read-out) and blind to routing switches (a third of single-expert patches). Not worth several days of engine work for
  this use.
- **Where it would pay.** Attribution at every position (T × components; the F4 / 6b grids are T-fold more expensive by
  activation patching), edge-level attribution (EAP; infeasible by activation patching), or case sets of thousands. Then:
  - experts and MoE / attention layers late in the network: AtP is reliable in the last quarter (r ≥ 0.95, slope ≥ 0.85) and
    usable in the third (r 0.84-0.94, slope 0.55-0.87); its best expert per row is within the truth's top 3 in
    1.00 / 0.96 of rows; use it as the ranking and verify the top N with activation patching;
  - heads: reliable in the last quarter only; mid-layer heads are underestimated 2-3× (WinoGrande L6H4); screen with AtP
    or EAP-IG and keep activation patching for the final claim;
  - early layers (first quarter): no method resolves single components in bf16; use layer-level or joint patches.
- **Add-back orderings.** No reverse streaming is needed: the engine's single-patch oracle is as cheap, and DLA (already in
  the engine) is a reasonable gradient-free substitute where the components write directly (WinoGrande); AtP would only
  matter if per-row single patches became unaffordable.
- **Scaling caveat.** OLMoE has 16 layers; Qwen3 (48) and Mixtral (32) put more layers between mid-depth components and the
  read-out, and AtP degrades with that distance, so the pilot's numbers are an upper bound for mid-layer components there.

### Caveats

- One model, two tasks, final position only; components are single-node patches (no AtP for edges or positions).
- The truth is bf16 activation patching with bf16 logits (0.03-0.06 logit resolution); the transformers patch with an fp32
  read-out is the noise ceiling, an fp32 truth exists for 8 cases per task only.
- "Routing switch" is measured in transformers bf16 batches against the same-batch unpatched row; near-tie routers make it
  partly a numerics effect (fp32 switch rates are lower).
- AtP* here is an adaptation (sublayer-output nodes; QK fix applied at the next attention's final-position query / key, plus
  an analogous router fix), not Kramár et al.'s per-query / per-key node set; GradDrop was not run.
- GPU time 28 min of queue jobs (engine scan, truth and add-back ≈ 5 min including a failed first attempt;
  transformers estimates and patches ≈ 23 min); fp32 checks ≈ 30 CPU-min. The transformers patches dominate; the gradients alone are
  ≈ 10 % of that time (CPU timing ratio).

### Files

moetrace/ext13atp.py (tasks, transformers hooks, gradient / IG / patched passes), moetrace/ext13atp_analysis.py (joined table,
metrics); scripts/ext13atp_build.py, ext13atp_engine.py (scan, truth, add-back), ext13atp_grad.py (transformers stage, fp32
check), ext13atp_analyze.py, ext13atp_text.py, ext13atp_chain.sh; run dirs results/p5_atp_wino, results/p5_atp_ioi
(truth_rows / truth_cases, hf_nodes / hf_cases, hf_fp32*_nodes, addback_rows, run_meta.json), results/p5_atp_data;
results/ext13atp_summary.json; tables results/tables/ext13atp_*; figures results/figures/ext13atp_{scatter_atp,cut_curve,addback}.png.
