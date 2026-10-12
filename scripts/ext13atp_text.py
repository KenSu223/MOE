"""p5-atp section writer: results/sections/ext13_atp.md from results/ext13atp_summary.json and results/tables/ext13atp_*.

Usage: python scripts/ext13atp_text.py
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext13atp as A

R = A.RESULTS
S = json.load(open(os.path.join(R, "ext13atp_summary.json")))
M = pd.DataFrame(S["main"])
RT = pd.DataFrame(S["routing"])
AU = pd.DataFrame(S["addback_auc"])
BL = pd.DataFrame(S["by_layer"])
PT = pd.DataFrame(S["population_top"])
PH = pd.DataFrame(S["top1_hits"])
TL = {"wino": "WinoGrande", "ioi": "IOI"}
TYPES = ("expert", "head", "attn", "moe")
TN = {"expert": "experts", "head": "heads", "attn": "attention layers", "moe": "MoE layers"}


def m(task, group, est, col):
    x = M[(M.task == task) & (M.group == group) & (M.est == est)]
    return float(x[col].iloc[0])


def rt(task, group, subset, col):
    x = RT[(RT.task == task) & (RT.group == group) & (RT.subset == subset)]
    return float(x[col].iloc[0])


def auc(task, pool, order):
    x = AU[(AU.task == task) & (AU.pool == pool) & (AU.order == order)]
    return float(x.auc16.iloc[0]), float(x.ci_lo.iloc[0]), float(x.ci_hi.iloc[0])


def bl(task, ct, band, est, col):
    x = BL[(BL.task == task) & (BL.ctype == ct) & (BL.band == band) & (BL.est == est)]
    return float(x[col].iloc[0])


def hit(task, ct, est, col="top1_hit"):
    x = PH[(PH.task == task) & (PH.ctype == ct) & (PH.est == est)]
    return float(x[col].iloc[0])


def f(x, d=2):
    return f"{x:.{d}f}"


def agreement_table():
    rows = ["| Task | Component | Ceiling r (2nd exact patch) | AtP r / slope | AtP* r | EAP-IG r | DLA r | AtP per-row Spearman (ceiling) | AtP top-k overlap / recovery | AtP pop. Spearman / top-k |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for t in ("wino", "ioi"):
        for g in TYPES:
            kk = "3" if g in ("attn", "moe") else "10"
            rows.append(f"| {TL[t]} | {TN[g]} | {f(m(t, g, 'truth_hf', 'pearson_pooled'), 3)} | {f(m(t, g, 'atp', 'pearson_pooled'), 3)} / {f(m(t, g, 'atp', 'slope'))} | "
                        f"{f(m(t, g, 'atpstar', 'pearson_pooled'), 3)} | {f(m(t, g, 'ig8', 'pearson_pooled'), 3)} | {f(m(t, g, 'dla', 'pearson_pooled'), 3)} | "
                        f"{f(m(t, g, 'atp', 'spearman_row_median'))} ({f(m(t, g, 'truth_hf', 'spearman_row_median'))}) | "
                        f"top-{kk} {f(m(t, g, 'atp', 'topk_overlap'))} / {f(m(t, g, 'atp', 'recovery'))} | "
                        f"{f(m(t, g, 'atp', 'spearman_population'))} / {f(m(t, g, 'atp', 'top10_population'))} |")
    return "\n".join(rows)


def depth_table():
    rows = ["| Task | Component | " + " | ".join(f"{b} r / slope" for b in ("L0-3", "L4-7", "L8-11", "L12-15")) + " | mean abs truth L0-3 / L12-15 |",
            "|---|---|---|---|---|---|---|"]
    for t in ("wino", "ioi"):
        for g in TYPES:
            cells = [f"{f(bl(t, g, b, 'atp', 'pearson'))} / {f(bl(t, g, b, 'atp', 'slope'))}" for b in ("L0-3", "L4-7", "L8-11", "L12-15")]
            rows.append(f"| {TL[t]} | {TN[g]} | " + " | ".join(cells) +
                        f" | {f(bl(t, g, 'L0-3', 'atp', 'mean_abs_truth'), 3)} / {f(bl(t, g, 'L12-15', 'atp', 'mean_abs_truth'), 3)} |")
    return "\n".join(rows)


def routing_table():
    rows = ["| Task | Component | Nodes that switch ≥ 1 downstream top-k slot | Slots switched per switching node | r(AtP, exact) all / no switch / switch | r(AtP, frozen-routing patch) | Share of AtP's squared error removed by freezing | Switching nodes with a large switch effect |",
            "|---|---|---|---|---|---|---|---|"]
    for t in ("wino", "ioi"):
        for g in TYPES:
            rows.append(f"| {TL[t]} | {TN[g]} | {f(rt(t, g, 'switch', 'share'))} | {f(rt(t, g, 'switch', 'mean_switch_slots'), 1)} | "
                        f"{f(rt(t, g, 'all', 'r_atp_truth'), 3)} / {f(rt(t, g, 'no switch', 'r_atp_truth'), 3)} / {f(rt(t, g, 'switch', 'r_atp_truth'), 3)} | "
                        f"{f(rt(t, g, 'all', 'r_atp_frozen'), 3)} | {f(rt(t, g, 'all', 'sse_share_removed_by_freeze'))} | {f(rt(t, g, 'switch', 'frac_big_switch'))} |")
    return "\n".join(rows)


def addback_table():
    rows = ["| Task | Pool | oracle | AtP | AtP* | EAP-IG | DLA | pop (discovery truth) | random (3 perms) |", "|---|---|---|---|---|---|---|---|---|"]
    for t in ("wino", "ioi"):
        for pool in ("experts", "heads_experts"):
            cells = []
            for o in ("oracle", "atp", "atpstar", "ig8", "dla", "pop"):
                a, lo, hi = auc(t, pool, o)
                cells.append(f"{f(a, 3)} [{f(lo, 3)}, {f(hi, 3)}]")
            rnd = np.mean([auc(t, pool, f"rand{i}")[0] for i in range(3)])
            rows.append(f"| {TL[t]} | {pool.replace('_', ' + ')} | " + " | ".join(cells) + f" | {f(rnd, 3)} |")
    return "\n".join(rows)


def curve_val(task, pool, order, k):
    c = pd.DataFrame(S["addback_curves"][task])
    x = c[(c.pool == pool) & (c.order == order) & (c.k == k)]
    return float(x["mean"].iloc[0])


def pop_heads_text():
    w = PT[(PT.task == "wino") & (PT.ctype == "head")].head(4)
    return "; ".join(f"{r.comp} {f(r.truth)} (AtP {f(r.atp)}, rank {r.rank_atp}; EAP-IG {f(r.ig8)}, rank {r.rank_ig8})" for r in w.itertuples())


def numerics():
    out = []
    for t in ("wino", "ioi"):
        fp = S["tasks"][t]["fp32"]
        out.append(f"{TL[t]}: AtP bf16 vs fp32 r {f(fp['atp']['r'], 4)} (relative L2 {f(fp['atp']['rel_l2'], 3)}), EAP-IG r {f(fp['ig8']['r'], 4)}, "
                   f"gradient cosine at the corrupted run median {f(fp['grad_cosine_alpha0']['median'], 4)} (min {f(fp['grad_cosine_alpha0']['min'], 4)}); "
                   f"fp32 patches (8 cases): r(AtP, exact) fp32 / bf16 experts {f(fp['fp32_patch']['expert']['r_atp32_truth32'], 3)} / {f(fp['fp32_patch']['expert']['r_atp16_truth16'], 3)}, "
                   f"heads {f(fp['fp32_patch']['head']['r_atp32_truth32'], 3)} / {f(fp['fp32_patch']['head']['r_atp16_truth16'], 3)}; "
                   f"engine vs fp32 truth r experts {f(fp['fp32_patch']['expert']['r_truth32_eng'], 3)}, heads {f(fp['fp32_patch']['head']['r_truth32_eng'], 3)}; "
                   f"switching nodes fp32 / bf16 experts {f(fp['fp32_patch']['expert']['switch_frac_32'])} / {f(fp['fp32_patch']['expert']['switch_frac_16'])}, "
                   f"heads {f(fp['fp32_patch']['head']['switch_frac_32'])} / {f(fp['fp32_patch']['head']['switch_frac_16'])}")
    return out


def cut_text(task):
    c = pd.DataFrame(S["cut_curves"][task])
    g = lambda grp, s: float(c[(c.group == grp) & (c.s == s)].pearson.iloc[0])
    return ", ".join(f"{TN[x]} {f(g(x, 0))} → {f(g(x, 2))} → {f(g(x, 8))}" for x in TYPES)


def main():
    sw = S["tasks"]["wino"]["sanity"]
    si = S["tasks"]["ioi"]["sanity"]
    scan = json.load(open(os.path.join(A.DATA, "scan_ioi_olmoe.json")))
    wscan = json.load(open(os.path.join(R, "wino_olmoe", "scan_summary.json")))
    gpu = {t: S["tasks"][t]["gpu_s"] for t in ("wino", "ioi")}
    import re
    secs = [int(x) for x in re.findall(r"END   p5atp-[\w-]+ rc=\d+ \((\d+)s\)", open(os.path.join(A.ROOT, "logs", "gpu_queue.log")).read())]
    gpu_min = sum(secs) / 60
    a_w, a_i = auc("wino", "experts", "atp"), auc("ioi", "experts", "atp")
    o_w, o_i = auc("wino", "experts", "oracle"), auc("ioi", "experts", "oracle")
    ha_w, ha_i = auc("wino", "heads_experts", "atp"), auc("ioi", "heads_experts", "atp")
    ho_w, ho_i = auc("wino", "heads_experts", "oracle"), auc("ioi", "heads_experts", "oracle")
    txt = f"""## Direction 13-AtP (Phase 5, P5-5): attribution patching against activation patching on OLMoE

**Summary.** On OLMoE-1B-7B-0125 (bf16), for two STR tasks (WinoGrande option swap and IOI, 64 + 64 pairs = 256 directed
cases each), the gradient estimate AtP of single final-position patches agrees with the engine's activation patching
(pooled r) at {f(m('wino', 'expert', 'atp', 'pearson_pooled'))} / {f(m('ioi', 'expert', 'atp', 'pearson_pooled'))} for experts,
{f(m('wino', 'moe', 'atp', 'pearson_pooled'))} / {f(m('ioi', 'moe', 'atp', 'pearson_pooled'))} for MoE layers,
{f(m('wino', 'head', 'atp', 'pearson_pooled'))} / {f(m('ioi', 'head', 'atp', 'pearson_pooled'))} for heads and
{f(m('wino', 'attn', 'atp', 'pearson_pooled'))} / {f(m('ioi', 'attn', 'atp', 'pearson_pooled'))} for attention layers
(WinoGrande / IOI). A second exact patch with other bf16 numerics (transformers instead of the engine) reaches
{f(m('wino', 'expert', 'truth_hf', 'pearson_pooled'))} / {f(m('ioi', 'expert', 'truth_hf', 'pearson_pooled'))} (experts) and
{f(m('wino', 'head', 'truth_hf', 'pearson_pooled'))} / {f(m('ioi', 'head', 'truth_hf', 'pearson_pooled'))} (heads), so on IOI AtP is as good
as the noise floor for experts, heads and MoE layers, and on WinoGrande it falls short, most for heads and attention layers. The failures are systematic, not
random: AtP is nearly exact in the last quarter of the network (r 0.95-1.00, slope 0.85-1.01) and underestimates effects
that pass through several later layers (slope 0.01-0.32 in L0-3, 0.11-0.63 in L4-7); the strongest WinoGrande head, L6H4 (+0.93 logits per row), is only
AtP's rank 6. Routing switches are the MoE-specific part of the error: {f(rt('wino', 'expert', 'switch', 'share'))} / {f(rt('ioi', 'expert', 'switch', 'share'))}
of single-expert patches change at least one downstream top-k slot, and freezing the downstream routing removes
{f(rt('wino', 'expert', 'all', 'sse_share_removed_by_freeze'))} / {f(rt('ioi', 'expert', 'all', 'sse_share_removed_by_freeze'))} of AtP's squared error for experts
({f(rt('wino', 'head', 'all', 'sse_share_removed_by_freeze'))} / {f(rt('ioi', 'head', 'all', 'sse_share_removed_by_freeze'))} for heads). AtP* (the next
attention softmax and the next router computed exactly) helps little; EAP-IG is worse than AtP for single patches. As a
static add-back ordering AtP equals the oracle (each row's own single-patch ranking): expert AUC(k ≤ 16)
{f(a_w[0], 3)} vs {f(o_w[0], 3)} (WinoGrande), {f(a_i[0], 3)} vs {f(o_i[0], 3)} (IOI); heads + experts {f(ha_w[0], 3)} vs {f(ho_w[0], 3)},
{f(ha_i[0], 3)} vs {f(ho_i[0], 3)}. bf16 is not a limitation (AtP bf16 vs fp32 r ≥ {f(min(S['tasks'][t]['fp32']['atp']['r'] for t in ('wino', 'ioi')), 3)}).
**Recommendation: do not build reverse layer streaming for the final-position sweeps** (activation patching there is
cheap in the engine and exact); build it only if the project needs attribution at every position or for edges, or 10×
larger case sets, and then use AtP as a screen for experts and late components with activation patching as the check.

### What was run

- **Model and protocol.** OLMoE-1B-7B-0125 (16 layers, 16 heads, 64 experts, top-8 without renormalisation), bf16,
  tokenizer defaults (no BOS), STR pairs (moetrace/ext7_pairs.py contract), denoising at the final position: one component
  of the corrupted run is set to its clean value, Δ = logit(true) − logit(foil), rescue = Δ_patched − Δ_corrupt.
- **Tasks** (competence ≥ 64 / 64 pairs with margin ≥ 1 in both directions):
  - WinoGrande option swap in the Phase-3 format: {wscan['margin']} / {wscan['n_pairs']} OLMoE margin pairs
    (results/wino_olmoe); 64 discovery + 64 validation pairs drawn from the Phase-3 discovery / validation pairs (Phase-3
    split kept, random.Random(13)); mean drop {f(sw['drop_mean'])} logits.
  - IOI with the S2 → IO corruption: the 1,600 Phase-3 items re-tokenised for OLMoE (all pass the build rules),
    {scan['margin']} / {scan['n_pairs']} margin pairs; 64 + 64 pairs from the Phase-3 IOI discovery / validation pairs; mean drop {f(si['drop_mean'])}.
  - Data: results/p5_atp_data/ (pairs_ioi_s2io_olmoe.parquet, scan_ioi_olmoe.parquet, case_sets_{{wino,ioi}}.json).
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

Checks: engine vs transformers Δ mean absolute difference {f(sw['eng_vs_hf_delta_corrupt_mae'], 3)} / {f(si['eng_vs_hf_delta_corrupt_mae'], 3)} logits (WinoGrande /
IOI); Σ expert AtP of a layer = MoE-layer AtP (r {f(sw['linearity']['expert_sum_vs_moe']['r'], 5)}), Σ head AtP = attention-layer AtP
(r {f(sw['linearity']['head_sum_vs_attn']['r'], 5)}); the patched run's first cut equals the gradient AtP (r {f(sw['cut0_vs_atp_r'], 4)}).

### 1. Agreement per component type

Pooled over all (row, component) pairs (WinoGrande {256} rows, IOI {256}); truth = engine activation patching; ceiling = the
transformers activation patch scored against the engine; slope = OLS slope of AtP on the truth; top-k overlap and recovery
(Σ truth over AtP's top k / Σ truth over the truth's top k) per row, k = 10 for experts / heads and 3 for layers; population
= mean over the 256 rows per component (table results/tables/ext13atp_agreement_main.md; per-row Kendall, Spearman CIs and
all estimator variants in ext13atp_agreement_rows.md, ext13atp_agreement_pooled.md, ext13atp_agreement_population.md).

{agreement_table()}

- Per-row Spearman over all 140 experts or 256 heads is low for every method, the ceiling included (heads
  {f(m('wino', 'head', 'truth_hf', 'spearman_row_median'))} / {f(m('ioi', 'head', 'truth_hf', 'spearman_row_median'))}): most components have
  effects at the bf16 resolution of the logits (40 % of the engine's WinoGrande head truths are exactly 0). The informative
  metrics are top-k overlap, recovery and the pooled r; on all of them AtP is within a few points of the ceiling on IOI and
  for WinoGrande experts / MoE layers.
- Per-row top-1: AtP's best expert is the truth's best in {f(hit('wino', 'expert', 'atp'))} / {f(hit('ioi', 'expert', 'atp'))} of rows and
  in its top 3 in {f(hit('wino', 'expert', 'atp', 'top1_in_truth_top3'))} / {f(hit('ioi', 'expert', 'atp', 'top1_in_truth_top3'))}; best head
  {f(hit('wino', 'head', 'atp'))} / {f(hit('ioi', 'head', 'atp'))} (top 3: {f(hit('wino', 'head', 'atp', 'top1_in_truth_top3'))} /
  {f(hit('ioi', 'head', 'atp', 'top1_in_truth_top3'))}); the second exact patch reaches {f(hit('wino', 'head', 'truth_hf'))} / {f(hit('ioi', 'head', 'truth_hf'))}
  for heads (results/tables/ext13atp_top1_hits.md).
- AtP underestimates large effects (slope {f(m('wino', 'head', 'atp', 'slope'))} for WinoGrande heads,
  {f(m('wino', 'attn', 'atp', 'slope'))} for attention layers; scatter results/figures/ext13atp_scatter_atp.png).

### 2. Where AtP fails: distance from the read-out

Pooled r / slope of AtP against the engine truth by layer band (results/tables/ext13atp_by_layer.md):

{depth_table()}

- AtP is close to exact in the last four layers and degrades with every layer between the component and the read-out.
- Early-layer single-component effects are small and partly below the bf16 noise of activation patching itself: in the
  fp32 patches (8 cases per task) the L0-3 expert / head effects are 0.01-0.02 logits on average, the bf16 truth correlates
  with the fp32 truth at only 0.07-0.50 there, and fp32 AtP still underestimates them (slope 0.1-0.3). Neither method
  resolves them in bf16.
- Mid-layer components with large indirect effects are the costly case. WinoGrande population heads (mean truth per row):
  {pop_heads_text()}. AtP's population top 5 misses the top head; EAP-IG ranks it 2nd
  (results/tables/ext13atp_population_top.md). Experts: the WinoGrande top-3 experts (L12E11, L13E40, L11E42) and the IOI
  top-5 keep their ranks under AtP; L9E10 drops from rank 4 to 9.

### 3. Routing switches (the MoE caveat)

Transformers patches (bf16), all 256 rows; "large switch effect" = |exact − frozen| > max(0.1, 0.25 |exact|) (results/tables/ext13atp_routing.md):

{routing_table()}

- A third of single-expert and a quarter of single-head patches switch downstream routing (1.6 slots on average among the
  switching nodes); layer-sized patches switch in 62-71 % of rows. Switch rates fall with depth (WinoGrande experts: 0.47 at L0 to 0.10 at L14), since
  fewer routers lie downstream.
- Freezing the downstream top-k removes 29-49 % of AtP's squared error on experts and heads (38-65 % among the switching
  nodes); for whole layers the smooth non-linearity dominates (7-37 % removed).
- Part of the switching is bf16 noise amplification: in fp32 fewer nodes switch (see section 6). A gradient sees routing
  weights but not set changes; the frozen-routing patch is what AtP approximates.

### 4. AtP* and the cut curve

Pooled r with the transformers patch after s exactly computed sublayers (s = 0 AtP, s = 2 AtP*, s = 8):
WinoGrande {cut_text('wino')}; IOI {cut_text('ioi')} (results/figures/ext13atp_cut_curve.png). The QK + router fix gains
0.00-0.05 in r; the non-linearity is spread over many downstream layers, so a local fix cannot remove it. AtP* per-row top-1
for WinoGrande heads {f(hit('wino', 'head', 'atpstar'))} (AtP {f(hit('wino', 'head', 'atp'))}).

### 5. EAP-IG

EAP-IG (input-embedding interpolation, m = 8) agrees less with single activation patches than AtP for every component type
except IOI attention layers (e.g. WinoGrande experts r {f(m('wino', 'expert', 'ig8', 'pearson_pooled'))} vs {f(m('wino', 'expert', 'atp', 'pearson_pooled'))};
IOI MoE layers per-row Spearman {f(m('ioi', 'moe', 'ig8', 'spearman_row_median'))} vs {f(m('ioi', 'moe', 'atp', 'spearman_row_median'))}). Its gradients
are taken in increasingly clean contexts, whereas a single patch keeps the corrupted context downstream; the gradient at the
clean run (AtP at α = 1) is worst of all (WinoGrande experts r {f(m('wino', 'expert', 'atp_clean', 'pearson_pooled'))}).
EAP-IG does better on large mid-layer components (WinoGrande L6H4) and as an add-back ordering (section 7), i.e. for joint
patches.

### 6. Numerics (bf16 vs fp32)

{chr(10).join('- ' + x + '.' for x in numerics())}

bf16 gradients are accurate; agreement with the truth is limited by the method (WinoGrande heads 0.83 in both precisions),
and the bf16 noise of the truth itself (IOI: AtP fp32 agrees with the fp32 truth at 0.98, better than the bf16 engine does).

### 7. AtP as an add-back ordering

AUC = mean restored fraction r(k) over k = 1..16 on the validation rows, pair-bootstrap CI (results/tables/ext13atp_addback_auc.md,
results/figures/ext13atp_addback.png):

{addback_table()}

- AtP, AtP* and EAP-IG orderings are at the oracle (AUC differences within ±0.022, CIs overlapping); DLA is 0.01-0.07 below
  the oracle except for IOI experts (half the oracle), which act indirectly; the discovery-population ordering is below the
  row-wise orderings.
- At k = 10: WinoGrande heads + experts oracle {f(curve_val('wino', 'heads_experts', 'oracle', 10))}, AtP {f(curve_val('wino', 'heads_experts', 'atp', 10))},
  EAP-IG {f(curve_val('wino', 'heads_experts', 'ig8', 10))}; IOI {f(curve_val('ioi', 'heads_experts', 'oracle', 10))} / {f(curve_val('ioi', 'heads_experts', 'atp', 10))} /
  {f(curve_val('ioi', 'heads_experts', 'ig8', 10))}. All-MoE ceiling WinoGrande {f(float(pd.DataFrame(S['addback_curves']['wino']).query("order == 'all_moe'")['mean'].iloc[0]))},
  IOI {f(float(pd.DataFrame(S['addback_curves']['ioi']).query("order == 'all_moe'")['mean'].iloc[0]))} (patching every MoE output at the final position hurts IOI).

### 8. Recommendation on reverse layer streaming for Qwen3 / Mixtral

- **What AtP would buy at the final position.** Our engine already runs every single-component patch at the final
  position cheaply: on OLMoE {110072:,} truth spawns took 18-104 s per task. On Qwen3 the same sweep costs ≈ 5 GPU-min per
  256 rows (Phase 5 logs: all-layer single experts, 631,536 spawns for 1,024 rows, ≈ 11 min; all 1,536 single heads for 256
  rows, 2 passes, ≈ 2 min), about what a forward plus a reverse stream (two checkpoint reads) would cost (estimate). AtP would replace these with one forward and one reverse
  stream, but is less accurate exactly where the open questions are (mid-layer heads and attention, components far from the
  read-out) and blind to routing switches (a third of single-expert patches). Not worth several days of engine work for
  this use.
- **Where it would pay.** Attribution at every position (T × components; the F4 / 6b grids are T-fold more expensive by
  activation patching), edge-level attribution (EAP; infeasible by activation patching), or case sets of thousands. Then:
  - experts and MoE / attention layers late in the network: AtP is reliable in the last quarter (r ≥ 0.95, slope ≥ 0.85) and
    usable in the third (r 0.84-0.94, slope 0.55-0.87); its best expert per row is within the truth's top 3 in
    {f(hit('wino', 'expert', 'atp', 'top1_in_truth_top3'))} / {f(hit('ioi', 'expert', 'atp', 'top1_in_truth_top3'))} of rows; use it as the ranking and verify the top N with activation patching;
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
- GPU time {f(gpu_min, 0)} min of queue jobs (engine scan, truth and add-back ≈ 5 min including a failed first attempt;
  transformers estimates and patches ≈ 23 min); fp32 checks ≈ 30 CPU-min. The transformers patches dominate; the gradients alone are
  ≈ 10 % of that time (CPU timing ratio).

### Files

moetrace/ext13atp.py (tasks, transformers hooks, gradient / IG / patched passes), moetrace/ext13atp_analysis.py (joined table,
metrics); scripts/ext13atp_build.py, ext13atp_engine.py (scan, truth, add-back), ext13atp_grad.py (transformers stage, fp32
check), ext13atp_analyze.py, ext13atp_text.py, ext13atp_chain.sh; run dirs results/p5_atp_wino, results/p5_atp_ioi
(truth_rows / truth_cases, hf_nodes / hf_cases, hf_fp32*_nodes, addback_rows, run_meta.json), results/p5_atp_data;
results/ext13atp_summary.json; tables results/tables/ext13atp_*; figures results/figures/ext13atp_{{scatter_atp,cut_curve,addback}}.png.
"""
    os.makedirs(os.path.join(R, "sections"), exist_ok=True)
    open(os.path.join(R, "sections", "ext13_atp.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
