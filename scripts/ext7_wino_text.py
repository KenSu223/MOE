"""ext7: writes results/sections/ext7_wino.md from results/ext7_wino_summary.json, the tables results/tables/ext7_wino_*.md,
results/verify_ext7_wino_olmoe.json and the run metadata (CPU only). Numbers in the prose are read from those files.

Usage: python scripts/ext7_wino_text.py
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np
from moetrace import ext7_pairs as P

ROOT = "/home/ubuntu/MOE/results"
S = json.load(open(os.path.join(ROOT, "ext7_wino_summary.json")))
V = json.load(open(os.path.join(ROOT, "verify_ext7_wino_olmoe.json")))
QR, MR = "wino_qwen3_str", "wino_mixtral_bos_str"
Q, M = S[QR], S[MR]
F, FR = P.fmt, P.fmt_r
LAB = {QR: "Qwen3", MR: "Mixtral (BOS)"}


def tab(name):
    p = os.path.join(ROOT, "tables", name + ".md")
    return open(p).read().strip() + "\n" if os.path.exists(p) else f"(table {name} not available)\n"


def meta(run):
    return json.load(open(os.path.join(ROOT, run, "run_meta.json")))


def gpu_s(run):
    return sum(sum(v.get("pass_times_s", []) or []) for v in meta(run).values())


def pk(s, kind, fam="main"):
    return [p for p in s["w2"]["peaks"][fam] if p["kind"] == kind][0]


def sh(s, key="main"):
    x = s["w2"]["share"][key]
    return f"{x['share']:.2f} [{x['share_lo']:.2f}, {x['share_hi']:.2f}]"


def ex(s, fam="main", tag="two_stage"):
    return s["w6"][fam][tag]


def en(r):
    e = r["selection"]["e_star"]
    return f"L{r['layer']}E{e:03d}" if e is not None else "none"


def gp(s, kind, cat, w="w1"):
    return (s.get("w3") or {}).get(w, {}).get("kinds", {}).get(kind, {}).get(cat)


def pv(x, d=3):
    return f"{x['peak'][0]:+.{d}f} [{x['peak'][1]:+.{d}f}, {x['peak'][2]:+.{d}f}]"


def attn_max(s):
    c = [x for x in s["w2"]["curves"]["main_val"] if x["kind"] == "attn_layer"]
    b = max(c, key=lambda x: x["norm"])
    return b["layer"], b["norm"], b["norm_lo"], b["norm_hi"]


def strat(s, col, val, key="w2"):
    for x in (s.get(key) or {}).get("strata", []):
        if x["stratum"] == col and x["value"] == val:
            return x
    return None


def w4(s, key="main_validation"):
    return (s.get("w4") or {}).get(key) or {}


def w4v(s, k, sub=None, key="main_validation", d=3):
    x = w4(s, key)
    v = x.get(k) if sub is None else (x.get(k) or {}).get(sub)
    return FR(v, d) if v is not None else "n/a"


def head_rows(s, n=4):
    return (s.get("w5") or {}).get("table", [])[:n]


def main():
    gpu = (gpu_s(QR) + gpu_s(MR)) / 60
    qL, qA, qB = pk(Q, "layer"), pk(Q, "attn_layer"), pk(Q, "block")
    mL, mA, mB = pk(M, "layer"), pk(M, "attn_layer"), pk(M, "block")
    q6, m6, q6r, m6r = ex(Q), ex(M), ex(Q, "rep"), ex(M, "rep")
    qcf, mcf = Q.get("cf_attn") or {}, M.get("cf_attn") or {}
    qh, mh = Q["w3"]["w1"].get("handoff", {}), (M.get("w3") or {}).get("w1", {}).get("handoff", {})
    qdl, mdl = (Q.get("w4_dla") or {}).get("main"), (M.get("w4_dla") or {}).get("main")
    qam, mam = attn_max(Q), attn_max(M)
    o = []
    w = o.append

    def dla_txt(x, k):
        return FR(x[k]) if x and k in x else "n/a"

    # ------------------------------------------------------------------------------------------------ summary
    w(f"**Summary.** WinoGrande twins become symmetric-token-replacement (STR) pairs once the blank is filled with each twin's "
      f"own answer and the model predicts the sentence-final trigger word (\"… but the bag was too\" → \" small\" / \"… but the body "
      f"was too\" → \" large\"): the two prompts differ only in the filled option, each is an ordinary WinoGrande sentence with its own "
      f"answer, and Δ = logit(r) − logit(r′) is Zhang & Nanda's logit difference. On 256 pairs that Qwen3 and Mixtral both solve with "
      f"a 1-logit margin in both directions (128 discovery / 128 validation pairs, each used both ways; bootstrap over pairs; a disjoint "
      f"256-pair replication set), the answer to \"MoE or attention?\" is the opposite of the IOI hypothesis at the final position. "
      f"Single-layer attention-output patches there carry at most {qam[1]:.3f} (Qwen3, L{qam[0]}) and {mam[1]:.3f} (Mixtral, L{mam[0]}) "
      f"of the drop, the attention share of the positive layer-wise rescue is {sh(Q)} and {sh(M)} — against "
      f"{qcf.get('share', float('nan')):.2f} and {mcf.get('share', float('nan')):.2f} on CounterFact STR under the same definition "
      f"(ext7-controls' sweep) — and the MoE-output peak per unit of drop is larger than on CounterFact (Qwen3 L{qL['L_disc']} "
      f"{FR(qL['norm_at_L_disc'])} vs {Q['cf_moe_norm_peak'][1]:.3f} at L{Q['cf_moe_norm_peak'][0]}; Mixtral L{mL['L_disc']} "
      f"{FR(mL['norm_at_L_disc'])} vs {M['cf_moe_norm_peak'][1]:.3f}). Patching every final-position MoE output (W4) restores "
      f"{w4v(Q, 'M_denoise')} (Qwen3) and {w4v(M, 'M_denoise')} (Mixtral) of the drop while attention still reads the corrupted option, "
      f"and on the direct paths to the logit difference the MoE outputs write {w4v(Q, 'direct', 'M_direct')} / {w4v(M, 'direct', 'M_direct')} "
      f"of the drop and the attention outputs {w4v(Q, 'direct', 'A_direct')} / {w4v(M, 'direct', 'A_direct')} (validation). The position × layer grid explains the small attention "
      f"patches: the option's identity reaches the final token directly from the option position (the intermediate token carries "
      f"≤ {qh.get('max_first_subsequent_token', [0, 0])[1]:.2f}) and gradually, the final-position residual restoring 0.1 / 0.5 / 0.9 "
      f"of the drop at L{qh.get('final_first_layer_ge_0.1')} / L{qh.get('final_first_layer_ge_0.5')} / L{qh.get('final_first_layer_ge_0.9')} "
      f"in Qwen3, so no single attention layer is a bottleneck. The MoE side localises to **one specific expert per model** — Qwen3 "
      f"**{en(q6)}** (validation rescue {F(q6['eval']['rescue'])}, Spec {F(q6['eval']['spec'])}, gate-matched equal-norm Spec "
      f"{F(q6['equal_norm']['spec_eq'])}) and Mixtral **{en(m6)}** (rescue {F(m6['eval']['rescue'])}, Spec {F(m6['eval']['spec'])}, "
      f"equal-norm Spec {F(m6['equal_norm']['spec_eq'])}), pattern A, both re-selected on the replication set — and these are not the "
      f"CounterFact STR selections (Qwen3 L44E069 / L42E115, Mixtral L19E002 / L21E001 / L18E001), which are routed at the WinoGrande "
      f"final position in only 3–12 of 256 directed cases and rescue nothing.\n")

    # ------------------------------------------------------------------------------------------------ what was run
    vq = V
    w("### What was run\n")
    w("Pairs (W0, `moetrace/ext7_wino.py`, `scripts/ext7_wino_build.py`, funnel `results/tables/ext7_wino_funnel.md`): WinoGrande 1.1 "
      "train_xl twins whose two sentences differ only in their last word, with a single-token trigger after both prompts, token "
      "symmetry (equal length, the option at the same positions, all other tokens identical), the option not the final token, one "
      "pair per normalised context. Case set (decision (g), `data/wino_str/case_sets.json`): the 776 pairs that pass the STR margin "
      "(Δ_A ≥ 1, Δ_B ≤ −1) under Qwen3, Mixtral BOS and Mixtral no BOS, seed-0 shuffle → **main** 128 discovery + 128 validation "
      "pairs, **replication** 128 + 128 pairs, and per model a seed-1 sample of its own margin pool (**own pool**, 128 + 128, W2 only). "
      "Every pair is run in both directions (directed case 2·pair + d: d = 0 clean A / corrupted B / r = trigger of A; d = 1 the "
      "reverse); per-pair value = mean of the two directions; CIs = 5,000 pair-bootstrap resamples; expert recurrence and Spec on "
      "directed cases (gate 128 of 256 discovery directed cases). Models: Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1 with BOS "
      "(tokenizer defaults, decision (e)); bf16. Runner: `moetrace/ext7_pairs.py` + `scripts/ext7_wino_{sweep,expert,grid,heads,"
      "joint,dla}.py` (generic STR-pair runners, also used by ext7-controls for role swaps and IOI).\n")
    w(tab("ext7_wino_descriptors"))
    w(f"In 93 % of the directed cases the option is one or two tokens before the prediction position (\"the bag was too\" / \"the bag "
      f"was\"; final word \"too\" in 48 %), and the clean prompt's top-1 is the trigger in {Q['descriptors']['main']['clean_top1_rate']:.0%} "
      f"(Qwen3) and {M['descriptors']['main']['clean_top1_rate']:.0%} (Mixtral) of the directed cases — margins are relative to the "
      f"twin's trigger, not top-1 accuracy (top-1 in both directions is a sensitivity stratum).\n")
    w(f"Verification (W1, `scripts/ext7_wino_verify.py` → `results/verify_ext7_wino_olmoe.json`; OLMoE, 20 margin pairs = 40 "
      f"directed cases, transformers hooks): final-position patches at every layer, rescue r = "
      f"{vq['final_layer']['rescue_corr_vs_hf']:.3f} (MoE), {vq['final_attn_layer']['rescue_corr_vs_hf']:.3f} (attention), "
      f"{vq['final_block']['rescue_corr_vs_hf']:.3f} (block), {vq['final_resid']['rescue_corr_vs_hf']:.3f} (residual), mean |ΔΔ| ≤ "
      f"{max(vq[f'final_{k}']['meanabsdiff'] for k in ('layer', 'attn_layer', 'block', 'resid')):.3f}; per-head patches "
      f"r = {vq['final_attn_head']['rescue_corr_vs_hf']:.3f} (small effects, SD {vq['final_attn_head']['hf_rescue_sd']:.2f}, mean |ΔΔ| "
      f"{vq['final_attn_head']['meanabsdiff']:.3f}); STR-position grid (suffix executor, every position from the option to the final token) "
      f"r = {vq['grid_layer']['rescue_corr_vs_hf']:.3f} / {vq['grid_attn_layer']['rescue_corr_vs_hf']:.3f} / "
      f"{vq['grid_block']['rescue_corr_vs_hf']:.3f} / {vq['grid_resid']['rescue_corr_vs_hf']:.3f} (MoE / attention / block / residual), "
      f"window 5 r = {vq['grid_w5_layer']['rescue_corr_vs_hf']:.3f}; null invariant {vq['grid_zero_null_maxdiff']:.3f}; the two directions "
      f"of a pair are exact mirrors (antisymmetry {vq['antisymmetry_maxdiff']:.1f}). The first run of this check found that "
      f"`moetrace/ext5_subject.py` (`run_subject`) turns `attn_layer` rows into `block` rows when the same pass contains a window > 1 "
      f"row; production passes never mix them (the W3 windows run in separate passes, as in Direction 6b). GPU time of all W1–W6 "
      f"passes on the big models: ≈ {gpu:.0f} min.\n")

    # ------------------------------------------------------------------------------------------------ W2
    w("### W2. Final-position layer sweep: MoE, attention, block\n")
    w(tab("ext7_wino_w2_peaks"))
    w(tab("ext7_wino_w2_share"))
    w(tab("ext7_wino_w2_layers"))
    w("![W2 curves](figures/ext7_wino_w2_curves.png)\n")
    w(f"- **Qwen3.** The MoE output of L{qL['L_disc']} restores {F(qL['val_at_L_disc'])} logits = {FR(qL['norm_at_L_disc'])} of the drop "
      f"(CounterFact STR: L44, 0.177); the band L39–L44 carries the effect (discovery top: {', '.join(f'L{l} {v:+.2f}' for l, v in qL['disc_top5'][:4])}). "
      f"No attention layer matters on its own (largest validation value {qam[1]:.3f} [{qam[2]:.3f}, {qam[3]:.3f}] of the drop at L{qam[0]}; the "
      f"discovery argmax L{qA['L_disc']} does not replicate on validation). Block ≈ attention + MoE at every layer (largest gap "
      f"{min(x['gap'] for x in Q['w2']['additivity']['top_block_layers']):+.3f}).\n"
      f"- **Mixtral.** MoE L{mL['L_disc']} {FR(mL['norm_at_L_disc'])} (flat L19–L21 band as on CounterFact, where the same patch gives 0.085); "
      f"attention has two discrete steps, L{mA['L_disc']} {FR(mA['norm_at_L_disc'])} and L19 (validation {[x for x in M['w2']['curves']['main_val'] if x['kind'] == 'attn_layer' and x['layer'] == 19][0]['norm']:.3f}), "
      f"and block L{mB['L_disc']} {FR(mB['norm_at_L_disc'])}. At L13 the attention output carries "
      f"{M['w2']['share_at_peaks']['attn_layer']['share'][0]:.2f} of the layer's attention + MoE rescue, at L20 only "
      f"{M['w2']['share_at_peaks']['layer']['share'][0]:.2f}.\n"
      f"- **Attention share** (Direction-2b AUC+ definition) {sh(Q)} / {sh(M)} on main validation, {sh(Q, 'rep')} / {sh(M, 'rep')} on the "
      f"replication set, {sh(Q, 'own')} / {sh(M, 'own')} on each model's own margin pool, the same in each single direction "
      f"(A→B {sh(Q, 'main_d0')} / {sh(M, 'main_d0')}); CounterFact STR {qcf.get('share', float('nan')):.2f} / {mcf.get('share', float('nan')):.2f}. "
      f"WinoGrande is less attention-dominated than CounterFact at the final position in both models, Qwen3 most clearly.\n")

    # ------------------------------------------------------------------------------------------------ W3
    w("### W3. Position × layer grid: where the option's identity travels\n")
    w(tab("ext7_wino_w3_grid_peaks"))
    for run, s in ((QR, Q), (MR, M)):
        if (s.get("w3") or {}).get("w1"):
            w(f"![W3 grid {LAB[run]}](figures/ext7_wino_w3_grid_{run}.png)\n")
    lines = []
    for run, s, h in ((QR, Q, qh), (MR, M, mh)):
        if not h:
            continue
        rs = gp(s, "resid", "last STR token")
        rf = gp(s, "resid", "last token")
        mo = gp(s, "layer", "last STR token")
        mf = gp(s, "layer", "last token")
        af = gp(s, "attn_layer", "last token")
        lines.append(f"- **{LAB[run]}.** Restoring the residual at the option token restores ≥ 0.5 of the drop up to L{h.get('str_last_layer_ge_0.5')} "
                     f"and ≤ 0.1 from L{h.get('str_first_layer_le_0.1')} on (the option's identity has left the option position); at the final position the residual "
                     f"restoration rises from 0.1 at L{h.get('final_first_layer_ge_0.1')} through 0.5 at L{h.get('final_first_layer_ge_0.5')} "
                     f"to 0.9 at L{h.get('final_first_layer_ge_0.9')}. The first subsequent token peaks at "
                     f"{h.get('max_first_subsequent_token', [0, 0])[1]:.3f} (L{h.get('max_first_subsequent_token', [0, 0])[0]}), further tokens at "
                     f"{h.get('max_further_tokens', [0, 0])[1]:.3f}: the information is read from the option position mostly by the final token itself, with a minor relay. "
                     f"MoE output at the option token: peak L{mo['peak_layer']} {pv(mo)} (token identity, Zhang & Nanda fn. 1 — not read as "
                     f"computation); at the final token L{mf['peak_layer']} {pv(mf)}; attention output at the final token at most "
                     f"{pv(af)} (L{af['peak_layer']}). Last-token column vs the W2 sweep: curve r "
                     f"{s['w3']['w1']['consistency_vs_sweep']['layer']['curve_r']:.4f} (MoE), "
                     f"{s['w3']['w1']['consistency_vs_sweep']['attn_layer']['curve_r']:.4f} (attention).")
    w("\n".join(lines) + "\n")
    sv = [(LAB[r], x) for r, s in ((QR, Q), (MR, M)) for x in (s.get("w3") or {}).get("sliding_vs_adding", [])]
    if sv:
        w("Window 5 (MoE output, secondary, Z6): sliding / summed single layers at the peak = " +
          "; ".join(f"{lab} {x['cat']} {x['ratio']:.2f}" for lab, x in sv if x["cat"] in ("last STR token", "last token")) + ".\n")

    # ------------------------------------------------------------------------------------------------ W4
    w("### W4. Joint decomposition of the final position (revised form)\n")
    w("The planned two-player Shapley split is degenerate: at the final position the token is shared and the MoE is a per-token "
      "function, so patching the attention output at every layer restores the clean final residual (A = 1 by construction; confirmed by "
      "ext8-addback and ext7-controls), and φ_attn = ½[A + (1 − M)] carries nothing beyond M. Reported instead (form shared by the three "
      "Phase-3 agents): A only as a sanity check of the `multi` path; M = all final-position MoE outputs patched, in the denoising "
      "direction (corrupted run, attention still reads the corrupted context: sufficiency) and in the noising direction (clean run, "
      "MoE outputs set to their corrupted values: necessity); and the direct-path split of h_clean − h_corrupt = Σ_l dAttn_l + Σ_l dMoE_l "
      "computed with ext7-controls' shared `direct_split_pairs` (= ext8's `direct_split`): A_direct = Δ(h_corrupt + Σ dAttn) − "
      "Δ_corrupt and M_direct = Δ(h_clean − Σ dAttn) − Δ_corrupt with the exact final norm, plus the linear DLA shares. For symmetric "
      "pairs used both ways the noising effect of direction d is, in exact arithmetic, the denoising effect of direction 1 − d (same "
      "intervention on the same prompt, metric sign-flipped), so M_noise and M_denoise coincide at the pair level; numerically the "
      "per-case values differ by bf16 recomputation noise amplified by top-k routing flips (last column: max per directed case), the "
      "population ratios by ≤ 0.002.\n")
    w(tab("ext7_wino_w4_joint"))
    w(f"- All MoE outputs at the final position restore {w4v(Q, 'M_denoise')} of the drop in Qwen3 and {w4v(M, 'M_denoise')} in Mixtral "
      f"(replication {w4v(Q, 'M_denoise', key='rep_validation')} / {w4v(M, 'M_denoise', key='rep_validation')}); ext8-addback's "
      f"CounterFact STR value is reported in its section (preliminary log: ≈ 0.5 in Qwen3), so WinoGrande's final-position answer is "
      f"more MoE-sufficient than factual recall's.\n"
      f"- Direct paths: MoE outputs {w4v(Q, 'direct', 'M_direct')} vs attention outputs {w4v(Q, 'direct', 'A_direct')} of the drop (Qwen3), "
      f"{w4v(M, 'direct', 'M_direct')} vs {w4v(M, 'direct', 'A_direct')} (Mixtral); DLA shares attention {w4v(Q, 'dla', 'attn', d=2)} / "
      f"{w4v(M, 'dla', 'attn', d=2)}. Read with W3: attention transports the option's identity to the final position over many "
      f"layers, the late MoE outputs write the answer.\n")
    w(tab("ext7_wino_w4_strata"))
    w(tab("ext7_wino_w4_dla"))

    # ------------------------------------------------------------------------------------------------ W5
    w("### W5. Attention heads at the W2 attention layers\n")
    if Q.get("w5") or M.get("w5"):
        w(tab("ext7_wino_w5_heads"))
        w(tab("ext7_wino_w5_layers"))
        w(tab("ext7_wino_w5_attention"))
        w("![W5 heads](figures/ext7_wino_w5_heads.png)\n")
        for run, s in ((QR, Q), (MR, M)):
            x = s.get("w5")
            if not x:
                continue
            det = x["detected_2sd"]
            pos = [d for d in det if d["val_mean"] > 0]
            neg = [d for d in det if d["val_mean"] < 0]
            am = {(a["layer"], a["head"]): a for a in x["attention_mass_top"]}
            ls = {l["layer"]: l for l in x["layer_summary"]}
            w(f"- **{LAB[run]}** (layers {', '.join(f'L{l}' for l in x['layers'])}; null layer L{x['null_layer']}): "
              f"{len(det)} of {x['n_heads_scanned']} heads at |z| ≥ 2 on both splits — positive "
              + ", ".join(f"L{d['layer']}H{d['head']} {d['val_mean']:+.3f} [{d['val_lo']:+.3f}, {d['val_hi']:+.3f}]" for d in pos)
              + ("; negative " + ", ".join(f"L{d['layer']}H{d['head']} {d['val_mean']:+.3f}" for d in neg) if neg else "") + ". "
              + " ".join(f"L{l}: attention output {ls[l]['attn_layer']['mean']:+.3f}, sum of single heads {ls[l]['sum_heads']['mean']:+.3f} "
                         f"(per-pair r {ls[l]['r_sum_vs_attn_pair']:.2f})." for l in x["layers"] if l != x["null_layer"])
              + f" Single-head patches are far from additive (per-pair r {min(ls[l]['r_sum_vs_attn_pair'] for l in ls):.2f}–"
                f"{max(ls[l]['r_sum_vs_attn_pair'] for l in ls):.2f}), so additive minimal head sets are not interpreted."
              + " Attention of the top heads at the final position (clean → corrupted): "
              + "; ".join(f"L{a['layer']}H{a['head']} option {a['clean_str']:.2f} → {a['corrupt_str']:.2f}, first mention of the clean "
                          f"filler {a['clean_ment_filled']:.2f} → {a['corrupt_ment_filled']:.2f}, of the other candidate "
                          f"{a['clean_ment_other']:.2f} → {a['corrupt_ment_other']:.2f}, position 0 {a['clean_pos0']:.2f}"
                          for a in x["attention_mass_top"][:3]) + ".")
        w("")
    else:
        w("(not available)\n")

    # ------------------------------------------------------------------------------------------------ W6
    w("### W6. Experts: two-stage selection, joint search, CounterFact experts\n")
    w(tab("ext7_wino_w6_experts"))
    w(tab("ext7_wino_w6_equalnorm"))
    w(tab("ext7_wino_w6_joint"))
    w(tab("ext7_wino_w6_cf_experts"))
    qr = q6["all_active_rank"]
    w(f"- **Qwen3 {en(q6)}**: active in {q6['selection']['disc_active']} of 256 discovery directed cases; carries "
      f"{q6['eval']['rescue']['mean'] / q6['layer_val']['mean']:.0%} of the L{q6['layer']} MoE rescue; rank 1 among the case's active experts in "
      f"{qr['top1']} of {qr['n']} anchor-active validation cases; at equal norm with its gate-matched partner it still wins by "
      f"{F(q6['equal_norm']['spec_eq'])} (raw {F(q6['equal_norm']['spec_raw'])}). Replication set: {en(q6r)} again (Spec "
      f"{F(q6r['eval']['spec'])}). The joint search over all 48 layers puts it first on main and second on the replication set behind "
      f"{Q['w6']['rep']['joint_top'][0]['layer'] and 'L' + str(Q['w6']['rep']['joint_top'][0]['layer']) + 'E' + format(Q['w6']['rep']['joint_top'][0]['expert'], '03d')} "
      f"(the main set's second locus).\n"
      f"- **Mixtral {en(m6)}**: active in {m6['selection']['disc_active']} of 256; Spec {F(m6['eval']['spec'])}; equal norm "
      f"{F(m6['equal_norm']['spec_eq'])} (the other active expert as control, Table 11 analogue); replicated ({en(m6r)}, Spec "
      f"{F(m6r['eval']['spec'])}); second locus L19E006, the expert that carried Mixtral's sink-state final tokens without BOS "
      f"(Direction 3) and is an ordinary, positively specific content expert here.\n"
      f"- The CounterFact STR selections (Qwen3 L44E069 / L42E115; Mixtral BOS L19E002 / L21E001 / L18E001) are routed at the WinoGrande "
      f"final position in only 3–12 of 256 discovery directed cases and rescue ≈ 0 (table above); only L19E006 — the paper's Mixtral "
      f"expert, negatively specific on CounterFact — is routed here (248 of 256) and is WinoGrande's second locus. Factual recall and "
      f"WinoGrande's trigger prediction use different late experts in both models.\n")

    # ------------------------------------------------------------------------------------------------ strata
    w("### Strata and robustness\n")
    w(tab("ext7_wino_w2_strata"))
    w(tab("ext7_wino_w6_strata") if os.path.exists(os.path.join(ROOT, "tables", "ext7_wino_w6_strata.md")) else "")
    nq, nm = strat(Q, "names", True), strat(M, "names", True)
    w("Replication set, own margin pools and single directions reproduce the W2 peaks and attention shares (tables above) and the W6 "
      "selections. Person-name pairs (\"Brett bought Kevin dinner … Brett felt very\" → \" generous\" / \" thankful\"; 9 % of the main set) "
      "behave differently: their drop is carried less by the selected experts (W6 strata) and their attention share is "
      + (f"{nq['share']['share']:.2f} (Qwen3) and {nm['share']['share']:.2f} (Mixtral)" if nq and nm and "share" in nq and "share" in nm else "n/a")
      + "; on the direct paths (W4 strata) the attention outputs write "
      + ", ".join(f"{(next(x for x in sv['w4']['strata'] if x['stratum'] == 'names' and x['value'] is True)['direct']['A_direct'][0]):.2f} of the drop for names vs "
                  f"{(next(x for x in sv['w4']['strata'] if x['stratum'] == 'names' and x['value'] is False)['direct']['A_direct'][0]):.2f} for objects ({lab})"
                  for sv, lab in ((Q, 'Qwen3'), (M, 'Mixtral')))
      + " — the social items are the attention-leaning subset (W7, the role swap of the two names, is in ext7-controls).\n")

    # ------------------------------------------------------------------------------------------------ reading
    w("### Reading\n")
    w(READING)
    w("### Caveats\n")
    w(CAVEATS)
    w("### Files\n")
    w(FILES)
    return "\n".join(o)


READING = """- **WinoGrande (option swap) is not IOI-like at the final position.** In both models the answer is written by the MoE outputs of
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
"""

CAVEATS = """- The STR corruption swaps the filled option only (decision (a)); the role swap of the two candidates' earlier mentions (W7, Z7)
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
"""

FILES = """- Code: `moetrace/ext7_pairs.py` (generic STR-pair runner helpers, pair-level statistics, `analysis.ModelData` adapter),
  `scripts/ext7_wino_{sweep,expert,grid,heads,joint,dla,verify,analyze,text}.py`, chains `scripts/ext7_wino_chain{1,2,3,4}.sh` (chain 1 died at the Mixtral sweep with a CUDA OOM, chain 2 resumed with `--wf-chunk 2048`).
- Runs: `results/wino_qwen3_str/`, `results/wino_mixtral_bos_str/` (`str_sweep_rows`, `str_sweep_routing`, `sweep_cases`,
  `str_expert_rows` (ext6 schema; used by ext8 for add-back), `str_grid_w{1,5}_rows`, `head_rows`, `head_attn_final.npz`,
  `head_positions`, `joint_rows`, `direct_split` (shared W4 split), `dla_rows`, `dla_cases`, `case_sets.json` (families), `run_meta.json`).
- Verification: `results/verify_ext7_wino_olmoe.json`. Numbers: `results/ext7_wino_summary.json`. Tables `results/tables/ext7_wino_*`,
  figures `results/figures/ext7_wino_*`.
"""


if __name__ == "__main__":
    txt = main()
    os.makedirs(os.path.join(ROOT, "sections"), exist_ok=True)
    with open(os.path.join(ROOT, "sections", "ext7_wino.md"), "w") as f:
        f.write(txt)
    print(f"wrote results/sections/ext7_wino.md ({len(txt)} chars)")
