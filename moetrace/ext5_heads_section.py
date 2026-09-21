"""ext5 F2: writes results/sections/ext5_f2_heads.md from the summary produced by scripts/ext5_heads_analyze.py."""
from __future__ import annotations

import json
import os

from .models import RESULTS
from .ext5_heads import POS_CLASSES, POS_LABEL, RUNS


SUMMARY = """**Summary.** Per-head patching of the final position's attention output (engine kind `attn_head`, verified against transformers hooks on OLMoE) shows that the attention rescue of Extension 2b is carried by a few *mover heads* that read the last subject token. Qwen3 L40 (attention +1.58): head 13 alone gives +0.95 (60%, Spec +0.93); two heads (13, 15) reach 80%. Qwen3 L43 (+1.10) is distributed (h11, h15, h28; four heads for 80%). Mixtral L18 (+0.81 no BOS / +0.99 BOS): head 4 gives +0.61 / +0.79 (76% / 80%) and suffices alone in 57% / 55% of cases; L24 (+0.90 / +0.86): head 22 +0.78 / +0.70 (86% / 82%); L19, the paper's MoE-peak layer where attention rescues +0.84 / +0.93, needs heads 29–31 plus one or two more; L15: heads 1 and 3. The same heads win with and without BOS. The top head's Spec equals its rescue because the other heads average zero; two Qwen3 L40 heads oppose the recall (h9 −0.50). Mover heads put 0.4–0.5 of their clean attention on the last subject token; noise halves it (Qwen3 h13 0.50 → 0.23, Mixtral h4 0.39 → 0.17) and moves it to the relation tokens (no BOS) or the position-0 sink (BOS). Σ heads equals the attention rescue on the mean at every layer (all gap CIs cover 0) but per-case r is only 0.3–0.7, so minimal sets are additive estimates. At Qwen3 L44 attention rescues +0.04 and no head exceeds +0.014: the paper's MoE peak is a pure-MoE layer.

"""


def _ci(m, lo, hi, d=3):
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def _verif_text() -> str:
    p = os.path.join(RESULTS, "verify_ext5_engine_olmoe.json")
    if not os.path.exists(p):
        return ""
    v = json.load(open(p))
    a = v["attn_head"]
    return (f"**Verification (OLMoE-1B-7B-0125, {v['n_cases']} cases, layers {v['head_layers']}, `scripts/ext5_engine_verify.py`, "
            f"`results/verify_ext5_engine_olmoe.json`).** Linearity: the sum over heads of the engine's head vectors equals "
            f"W_o(H_clean − H_noised) in fp32 to {a['headsum_vs_fp32_linear_maxabs']:.1e} (max abs) and the `attn_layer` vector "
            f"(difference of the bf16 o_proj outputs) to {a['headsum_vs_attn_layer_vec_mean_relnorm'] * 100:.2f}% relative norm (bf16 rounding). "
            f"Against transformers hooks that replace the head-h slice of the o_proj input at the final position: per-row |Δ_engine − Δ_HF| "
            f"mean {a['attn_head_vs_hf']['meanabs']:.3f} (max {a['attn_head_vs_hf']['maxabs']:.2f}, {a['attn_head_vs_hf_frac_within_0.25'] * 100:.0f}% within 0.25), "
            f"the same floor as the whole-attention replace ({a['attn_layer_vs_hf_oin_replace']['meanabs']:.3f}) and the `layer` kind in "
            f"`verify_olmoe.json`; a head patch spawned on the clean run with itself as donor reproduces the clean logits "
            f"(max |Δ| {a['identity_attn_head_on_clean']['maxabs']:.3f}, mean {a['identity_attn_head_on_clean']['meanabs']:.3f}). "
            f"At the rescue level the 16 OLMoE heads do not add up per case (Σ_h rescue_h vs attention rescue: mean |gap| "
            f"{a['headsum_rescue_vs_attn_layer_rescue']['meanabs']:.2f}, r {a['headsum_rescue_vs_attn_layer_rescue_corr']:.2f} on 40 rows whose "
            f"attention rescue averages {a['attn_layer_rescue_mean']:+.2f}): the sum of 16 single-head rows carries 16 times the per-row bf16 "
            f"noise (±0.1–0.6), so per-case additivity can only be assessed on the large models with a sizeable attention rescue (below). "
            f"`results/verify_olmoe.json` is bit-identical to `results/verify_olmoe_before_ext5.json` on every non-timing metric.\n\n")


def write_section(summ: dict, path: str):
    L = []
    L.append("## Extension 5, F2: attention heads at the final token\n")
    L.append(SUMMARY)
    L.append("**Question.** Extension 2b showed that the attention sublayer carries about half of the positive rescue in both models "
             "(Qwen3 L40 +1.59, Mixtral L18 +0.99 vs the paper's MoE peaks +0.93 / +0.56). Which heads carry it, are they specific "
             "in the sense the paper uses for experts, how many are needed, and what do they attend to in the clean versus the noised run?\n")
    L.append("**Method.** New engine kind `attn_head` (`moetrace/engine.py`): with H_h the head-h output of the final position before "
             "`o_proj` and W_o[:, h] the matching column block, v_h = W_o[:, h]·(H_h^clean − H_h^noised) (fp32) is added to the noised "
             "attention output before the MoE of the same layer, h = h_pre^noised + bf16(Attn^noised + v_h); the MoE of that layer and "
             "all later layers recompute. Because `o_proj` is linear, Σ_h v_h equals the `attn_layer` vector, so the head patches decompose "
             "the attention-output patch exactly at the vector level; at the rescue level additivity is an empirical question. Rescue = "
             "Δ_patched − Δ_noised as in Table 1. One pass per run (`scripts/ext5_heads_sweep.py`): every head at the requested layers "
             "plus `attn_layer`, `layer` (MoE) and `block` reference rows on the paper's 256 cases, and the final position's attention "
             "distribution over positions (`DiagSpec.attn_final`) for the clean and noised prefill rows. Heads are ranked by validation "
             "rescue (128 cases; the discovery rank is reported for stability); Spec_h = rescue_h − mean of the other heads of the layer, "
             "per case; the minimal head set uses the additive approximation (heads ordered by discovery rescue, cumulative validation "
             "rescue against 80% of the `attn_layer` validation rescue) and is therefore an estimate, not an exact joint patch; attention "
             "masses are summed over position classes with priority final > last subject token > other subject tokens > position 0 > "
             "other (relation) tokens. Rows: `results/<run>/head_rows.parquet`; code `moetrace/ext5_heads.py`, "
             "`scripts/ext5_heads_analyze.py`.\n")
    L.append(_verif_text())
    for run, s in summ.items():
        cfg = RUNS.get(run, {})
        short = cfg.get("short", run)
        L.append(f"### {s['label']} (`results/{run}`)\n")
        for l, e in s["per_layer"].items():
            a, pop, pc, st = e["additivity"], e["minimal_population"], e["minimal_per_case"], e["stability"]
            t = e["ranking_top5"]
            top = ", ".join(f"h{int(r['head'])} {_ci(r['val_mean'], r['val_ci_lo'], r['val_ci_hi'])} (Spec {r['spec']:+.3f}, disc. rank {int(r['disc_rank'])})"
                            for r in t[:3])
            role = ""
            if cfg.get("moe_peak") == int(l):
                role = (" (the paper's MoE-peak layer; the attention output rescues nothing here, a null)" if a["attn_layer_mean"] < 0.1
                        else f" (the paper's MoE-peak layer, where the attention output also rescues {a['attn_layer_mean']:+.2f})")
            elif int(l) in cfg.get("attn_peaks", ()):
                role = " (attention peak from Extension 2b)"
            L.append(f"- **L{l}**{role}: attention-output rescue {_ci(a['attn_layer_mean'], *a['attn_layer_ci'])}, MoE {a['moe_layer_mean']:+.3f}, "
                     f"block {a['block_mean']:+.3f}. Top heads (validation): {top}. {a['n_heads_positive_mean']} of {s['n_heads']} heads have a positive "
                     f"mean rescue; discovery/validation rank agreement ρ = {st['spearman_disc_val']:.2f}, top-3 overlap {st['top3_overlap']}/3. "
                     f"Additivity: Σ heads {_ci(a['sum_heads_mean'], *a['sum_heads_ci'])} vs attention {a['attn_layer_mean']:+.3f} "
                     f"(gap {_ci(a['gap_mean'], *a['gap_ci'])}, per-case r = {a['per_case_r']:.2f}, per-case gap SD {a['gap_sd']:.2f}). "
                     + (f"Minimal set (additive estimate): {pop['k_attn_layer']} heads for 80% of the attention rescue ({pop.get('set_attn_layer')})"
                        if pop['k_attn_layer'] is not None else
                        "Minimal set (additive estimate): 80% of the attention rescue is not reached by any additive head set") +
                     f", top-3 heads carry {pop['top3_share_of_attn'] * 100:.0f}%; per case (attention rescue > 0.25, n = {pc['n_cases_used']}) the "
                     f"median number of heads is {pc['k_median']:.0f} (IQR {pc['k_q25']:.0f}–{pc['k_q75']:.0f}), one head suffices in "
                     f"{pc['frac_k_eq_1'] * 100:.0f}% and ≤ 3 in {pc['frac_k_le_3'] * 100:.0f}% of cases.")
            if "attention_top3" in e:
                at = e["attention_top3"]
                parts = []
                for r in at:
                    parts.append(f"h{int(r['head'])}: subject {r['clean_subject']:.2f} → {r['noised_subject']:.2f} (last subject token "
                                 f"{r['clean_subj_last']:.2f} → {r['noised_subj_last']:.2f}), position 0 {r['clean_pos0']:.2f} → {r['noised_pos0']:.2f}, "
                                 f"final {r['clean_final']:.2f} → {r['noised_final']:.2f}, relation {r['clean_other']:.2f} → {r['noised_other']:.2f}")
                c = e["attention_corr"]
                L.append(f"  Attention of the top-3 heads (validation mean mass, clean → noised): " + "; ".join(parts) + ". Across all heads of the "
                         f"layer the mean rescue correlates with the clean subject mass at ρ = {c['spearman_rescue_vs_clean_subject_mass']:.2f} "
                         f"(last subject token ρ = {c['spearman_rescue_vs_clean_subj_last_mass']:.2f}; with the noise-induced drop of subject mass "
                         f"ρ = {c['spearman_rescue_vs_subject_mass_drop']:.2f}); layer-mean subject mass {c['layer_mean_clean_subject_mass']:.2f} clean, "
                         f"{c['layer_mean_noised_subject_mass']:.2f} noised.")
        L.append("")
        L.append(f"![ext5 heads {short}](../figures/ext5_heads_{short}.png)\n")
        L.append(f"Figure E5-F2-{short}: per layer, left: validation rescue of every head with 95% bootstrap CIs (top-3 labelled); middle: "
                 f"per-case sum of the single-head rescues against the attention-output patch (grey diagonal = additivity); right: attention "
                 f"mass of the top-3 heads over position classes, clean (left bar) vs noised (right bar, hatched final-token class). Tables: "
                 f"`results/tables/ext5_heads_ranking_{short}_L<l>.md/csv` (all heads in `_all.csv`), `ext5_heads_additivity_{short}.md`, "
                 f"`ext5_heads_minimal_{short}.md`, `ext5_heads_attention_{short}.md`.\n")
        for l in s["per_layer"]:
            p = os.path.join(RESULTS, "tables", f"ext5_heads_ranking_{short}_L{l}.md")
            if os.path.exists(p):
                L.append(f"**{s['label']}: top-8 heads at L{l} by validation rescue ({s['n_val']} validation cases; Spec = rescue minus the mean of the other heads)**\n")
                L.append(open(p).read())
        for name in ("additivity", "minimal", "attention"):
            p = os.path.join(RESULTS, "tables", f"ext5_heads_{name}_{short}.md")
            if os.path.exists(p):
                title = {"additivity": "additivity of head patches (validation means)", "minimal": "minimal head sets (additive approximation)",
                         "attention": "attention distribution of the top-3 heads per layer (validation mean mass per position class)"}[name]
                L.append(f"**{s['label']}: {title}**\n")
                L.append(open(p).read())
    L.append("### Reading across runs\n")
    L.append(_cross_text(summ))
    L.append("\n**Caveats and open questions.** (1) Head Spec has no recurrence gate (every head is always active), so the analogue of the "
             "paper's expert Spec is a contrast against the other heads only. (2) Minimal sets are additive estimates; an exact joint head "
             "patch (`attn_head_set`, the head analogue of `coalition_set`) is a one-line engine extension and would settle the sub-additivity "
             "seen at the peaks. (3) The MoE-side cross-check of the plan (patch the L40 heads and read E069's routing/contribution at L44) "
             "needs routing recorded for wavefront rows and is left for wave 2. (4) The position-0 class is the BOS sink only in the BOS run of "
             "Mixtral; in Qwen3 and in Mixtral without BOS it is the first prompt token and is absorbed by the subject classes when the prompt "
             "starts with the subject.\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


def _cross_text(summ: dict) -> str:
    lines = []
    for run, s in summ.items():
        cfg = RUNS.get(run, {})
        for l, e in s["per_layer"].items():
            a, pop = e["additivity"], e["minimal_population"]
            t0 = e["ranking_top5"][0]
            lines.append(f"- {s['label']} L{l}: attention {a['attn_layer_mean']:+.2f}; best head h{int(t0['head'])} {t0['val_mean']:+.2f} "
                         f"({(t0['val_mean'] / a['attn_layer_mean'] * 100) if a['attn_layer_mean'] > 0.05 else float('nan'):.0f}% of the attention rescue); "
                         f"top-3 share {pop['top3_share_of_attn'] * 100 if a['attn_layer_mean'] > 0.05 else float('nan'):.0f}%; Σ heads {a['sum_heads_mean']:+.2f}, "
                         f"r = {a['per_case_r']:.2f}.")
    return "\n".join(lines)
