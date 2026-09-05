"""Compose results/REPORT.md from the analysis results, verification JSONs, logs and the table files."""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from .models import MODELS, RESULTS
from .stats import fmt


def _load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _pass_times() -> list[tuple[str, str]]:
    out = []
    for lg in sorted(glob.glob(os.path.join("/home/ubuntu/MOE/logs", "*.log"))):
        name = os.path.basename(lg)
        if not any(name.startswith(p) for p in ("filter_", "sweep_", "expert_", "noise_", "funnel", "verify_", "hf_ref")):
            continue
        txt = open(lg).read()
        m = re.findall(r"pass done: (\d+) prefill rows \(T=(\d+)\), (\d+) spawn rows, ([\d.]+)s", txt)
        for pre, T, sp, sec in m:
            out.append((name, f"{pre} prefill rows (T={T}), {sp} spawn rows, {float(sec):.0f} s"))
    return out


def write(res: dict, tables: dict, alt_table: str = "", extra_verdict: str = "", extra_sections: str = "") -> str:
    q = res.get("qwen3")
    x = res.get("mixtral")
    L = []
    A = L.append
    A("# Reproduction report: Expert-Aware Causal Tracing of Factual Recall in Sparse MoE Language Models (arXiv 2606.03780)\n")
    A("Generated automatically by the overnight agent (see logs/PROGRESS.md for the timeline). All numbers below are ours unless labelled paper.\n")

    # ---------------------------------------------------------------- verdict
    A("## 1. Verdict\n")
    for mk, r in res.items():
        s = r["sets"]["paper"]
        la = s["layer"]
        p = {"qwen3": (44, 0.901, "L44E069", 0.463, 0.400), "mixtral": (19, 0.457, "L19E006", 0.099, -0.175)}[mk]
        e = s.get("selection", {}).get("e_star")
        ev = s.get("eval")
        line = f"- **{MODELS[mk]['label']}** (paper case IDs): discovery selects **L{la['L_star']}** (paper L{p[0]}); validation layer rescue {fmt(la['val_at_Lstar'])} (paper {p[1]:+.3f}). "
        if ev:
            line += (f"Recurrence-first selection picks **L{la['L_star']}E{e:03d}** (paper {p[2]}); validation expert rescue {fmt(ev['rescue_all'])} "
                     f"(paper {p[3]:+.3f}), specificity {fmt(ev['spec_all'])} (paper {p[4]:+.3f}), sign-flip p = {ev['spec_all']['p']:.4f}.")
        A(line)
    if extra_verdict:
        A(extra_verdict)
    A("")
    A("Qualitative pattern of the paper: Qwen3's layer-level signal localises to one positive, specific expert; Mixtral's selected expert has negative "
      "specificity while routed coalitions recover the layer-level effect. See the tables below for whether each number falls inside the paper's CI.\n")

    # ---------------------------------------------------------------- what was run
    A("## 2. What was run\n")
    A("Engine: a layer-streaming PyTorch executor (`moetrace/engine.py`) that keeps one decoder layer on the GPU, runs clean/noised prefill rows and "
      "single-token 'wavefront' rows for every intervention in the same pass, and builds intervention vectors in-pass from the recorded final-position "
      "expert contributions. One pass reads the full checkpoint once (Qwen3 61 GB about 60 s, Mixtral 93 GB about 95 s).\n")
    A("Passes per model: (1) filter scan in chunks of 1,024 tokenizable records; (2) layer sweep over the union of the paper's 256 IDs, our strict 256 and "
      "relaxed 512 (clean, noised, MoE-block patch at every layer, plus per-layer routing tables); (3) expert pass at the selected layer(s): every clean-active "
      "expert, every noised-only-active expert, all ordered equal-norm pairs, clean-top-k and routing-union coalitions, and the layer patch; (4) Qwen3 noise-scale "
      "pass (sigma multipliers 1, 2, 4); (5) verification passes. Row-level parquet files are under results/<model>/.\n")
    A("| log | pass |\n|---|---|")
    for name, desc in _pass_times():
        A(f"| {name} | {desc} |")
    A("")

    # ---------------------------------------------------------------- verification
    A("## 3. Verification\n")
    v1 = _load_json(os.path.join(RESULTS, "verify_olmoe.json"))
    v2 = _load_json(os.path.join(RESULTS, "verify_olmoe_fp32.json"))
    if v1:
        A("**OLMoE-1B-7B-0125 pilot vs transformers 5.16.1 (bf16, eager attention), 50 CounterFact cases** (results/verify_olmoe.json):\n")
        keys = ["embed_std_engine", "embed_std_hf", "routing_set_agreement", "top1_agreement_clean", "delta_clean_maxdiff_vs_hf", "delta_clean_frac_within_0.1",
                "delta_noised_maxdiff_vs_hf", "delta_noised_frac_within_0.1", "layer_patch_vs_hf_n", "layer_patch_vs_hf_maxdiff", "layer_patch_vs_hf_frac_within_0.1",
                "layer_rescue_vs_hf_maxdiff", "expert_patch_vs_hf_n", "expert_patch_vs_hf_maxdiff", "expert_patch_vs_hf_frac_within_0.1", "zero_on_noised_maxdiff",
                "zero_on_clean_maxdiff", "union_vs_layer_maxdiff", "delta_bmm_vs_fullmatmul_frac_equal", "pass_total_s"]
        A("| check | value |\n|---|---|")
        for k in keys:
            if k in v1:
                v = v1[k]
                A(f"| {k} | {v if not isinstance(v, float) else round(v, 4)} |")
        A("")
    if v2:
        A("**bf16 noise floor (results/verify_olmoe_fp32.json, 12 cases; HF fp32 on CPU as ground truth):** the engine's bf16 error is the same size as HF's own "
          "bf16 error, and two HF attention backends differ from each other by as much as either differs from us.\n")
        A("| implementation | clean Delta: mean abs err / max | noised Delta: mean / max | L12 patch: mean / max | L12 rescue: mean / max |\n|---|---|---|---|---|")
        for k in ("engine_bf16", "hf_bf16_eager", "hf_bf16_sdpa"):
            d = v2[k]
            A(f"| {k} | {d['clean_vs_fp32_mean_abs']:.3f} / {d['clean_vs_fp32_max']:.3f} | {d['noised_vs_fp32_mean_abs']:.3f} / {d['noised_vs_fp32_max']:.3f} | "
              f"{d['patchL12_vs_fp32_mean_abs']:.3f} / {d['patchL12_vs_fp32_max']:.3f} | {d['rescueL12_vs_fp32_mean_abs']:.3f} / {d['rescueL12_vs_fp32_max']:.3f} |")
        A(f"\nEngine vs HF-eager max diffs: {json.dumps(v2['engine_vs_hf_eager'])}; HF-eager vs HF-sdpa: {json.dumps(v2['hf_eager_vs_hf_sdpa'])}.\n")
        A("Interpretation: Delta is a difference of two bf16 logits of magnitude 10 to 30 (ulp 0.06 to 0.25); noised prompts amplify accumulation-order "
          "differences. Per-case rescue therefore carries roughly 0.1 to 0.6 of bf16 noise in any implementation (including the authors'); means over 128 "
          "cases carry about 0.02 to 0.05. The HANDOFF's expectation of |diff| < 0.1 on almost all prompts was too optimistic for noised prompts, and the same "
          "holds for HF against itself.\n")
    A("Internal invariances (by construction and checked in the pilot): sum of recorded expert contributions equals the fp32 block output; the routing-union "
      "coalition equals the MoE-block patch bit-for-bit; bmm and full-matmul logits agree on 100% of rows; a zero-vector wavefront row reproduces the parent's "
      "logits to within the bf16 floor above (not bit-exact because single-token rows use different matmul shapes than the prefill rows).\n")
    for mk in ("qwen3", "mixtral"):
        h = _load_json(os.path.join(RESULTS, mk, "hf_reference_check.json"))
        if h:
            A(f"**{MODELS[mk]['label']} vs transformers with CPU/disk offload ({h['n']} validation prompts):** max |Delta diff| {h['max_abs_delta_diff']:.3f}, "
              f"mean {h['mean_abs_delta_diff']:.3f}, top-1 agreement {h['top1_agree']}/{h['n']}.\n")
            A("| case | prompt | Delta engine | Delta HF | top-1 engine / HF | HF forward s |\n|---|---|---|---|---|---|")
            for row in h["rows"]:
                A(f"| {row['case_id']} | {row['prompt']} | {row['delta_engine']:+.3f} | {row['delta_hf']:+.3f} | {row['top1_engine']} / {row['top1_hf']} | {row['forward_s']:.0f} |")
            A("")
        else:
            A(f"**{MODELS[mk]['label']} vs transformers with offload:** not available (see logs/hf_ref_{mk}.log).\n")

    # ---------------------------------------------------------------- funnel
    A("## 4. Filtering funnel vs the paper's Table 8 case IDs\n")
    for mk in ("qwen3", "mixtral"):
        cs = _load_json(os.path.join(RESULTS, mk, "case_sets.json"))
        if not cs:
            continue
        pc, sc = cs["paper_check"], cs["scan"]
        A(f"**{MODELS[mk]['label']}:** scanned {sc['records_scanned']} shuffled records ({sc['tokenizable']} tokenizable, {sc['rejected']} rejected: {sc['reject_reasons']}); "
          f"strict pass rate {sc['strict_pass_rate_tokenizable']:.2f}, relaxed {sc['relaxed_pass_rate_tokenizable']:.2f}. All 256 paper IDs lie within the first "
          f"{pc['paper_ids_scan_rank_max'] + 1} records of our seed-0 shuffle (median rank {pc['paper_ids_scan_rank_median']}), so the paper's record order is the same "
          f"`random.Random(0).shuffle`. Paper IDs passing our strict filter: {pc['paper_ids_strict_in_scan']}/256 (relaxed {pc['paper_ids_relaxed_in_scan']}); overlap of "
          f"the paper's 256 with our first-256 strict set: {pc['overlap_paper_vs_our_strict256']}; with our relaxed 512: {pc['overlap_paper_vs_our_relaxed512']}.\n")
    fh = _load_json(os.path.join(RESULTS, "qwen3", "funnel_hypotheses2.json"))
    if fh:
        A("Why the paper kept only 256 of the first 390 Qwen3 records while about 80% of tokenizable records pass our filter: we tested object-token conventions "
          "over all 390 records (results/qwen3/funnel_hypotheses2.json). Agreement = fraction of records where (passes our filter) == (is a paper case):\n")
        A("| object-token rule | records defined | paper cases passing | non-paper passing | agreement | first-256 overlap with paper |\n|---|---|---|---|---|---|")
        names = {"space_single": "continuation token with leading space (HANDOFF default, used for all main results)",
                 "space_first": "first token of ' '+obj (no single-token filter)", "nospace_first": "first token of obj (no space)",
                 "nospace_single_else_space": "tok(obj) if single token, else leading-space token", "nospace_single_only": "tok(obj) single token only (subset)"}
        for k, v in fh.items():
            A(f"| {names.get(k, k)} | {v['defined']} | {v['paper_pass']}/{v['paper_defined']} | {v['nonpaper_pass']}/{v['nonpaper_defined']} | {v['agreement']:.3f} | {v['overlap_first256']} |")
        A("\nThe rule 'use tok(obj) when it is a single token, otherwise the leading-space token' reproduces the paper's case set far better (0.949) than the "
          "HANDOFF default (0.774). BOS handling made no difference (results/qwen3/funnel_hypotheses.json). We therefore believe the paper resolved object tokens "
          "without a leading space first. Because the primary case set is the paper's own IDs, this affects only which token pair defines Delta for the roughly "
          "30% of cases whose objects are single tokens without a space; a secondary run with that rule is reported in section 6.\n")

    # ---------------------------------------------------------------- tables
    A("## 5. Paper vs ours, table by table\n")
    A("Paper numbers are copied from paper_src/acl_latex.tex. Bracketed intervals are 95% percentile-bootstrap CIs of the mean (5,000 resamples).\n")
    order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "12b", 13, 14, 15, 16, "sets"]
    notes = {
        3: "Zero rows: the paper reports one count (26 / 54) whose definition is not stated; we report the number of validation cases where the selected expert is not clean-active, the number of selected-expert rows with exactly zero rescue (not-active plus bf16-quantised zeros), and the same count over selected plus control rows.",
        8: "Our strict/relaxed sets are the first 256/512 records passing the respective filter in the seed-0 order (split with random.Random(0)); the paper's IDs are in data/paper_case_ids.json.",
        12: "Relation folds: relation ids shuffled with random.Random(0) and dealt round-robin into 5 folds; selection threshold = half of the selection cases; the paper's fold assignment is unknown.",
        "12b": "Splits for the stability grid use random.Random(seed).shuffle over the paper's 256 IDs; the paper's split function is unknown, so individual cells differ while the selected expert can be compared.",
        13: "sigma = 3.0 row comes from the main expert pass (same rows as Table 1); other rows from the noise pass; Active = validation cases where L44E069 is clean-active (independent of sigma).",
        14: "Relaxed set = first 512 records passing (margin >= 0.5, drop >= 0.25) in the seed-0 order, split 256/256 with random.Random(0), recurrence threshold 128.",
        15: "The paper reports only 15 of 256 relaxed validation cases overlapping its strict set; with a single shuffled scan, the strict 256 are necessarily a large subset of the first 512 relaxed cases, so the paper must have built its relaxed set from a different record order. We report the split against both the paper's IDs and our strict set. n_E in the paper (137/130) is unexplained; ours uses all validation cases.",
        16: "Coalition rows for Qwen3 are an extra (the paper reports Mixtral only). The routing-union coalition equals the MoE-block patch exactly; the same-pass layer patch is shown for reference (the sweep-pass value differs only by bf16 noise).",
    }
    for k in order:
        if k in tables:
            A(tables[k])
            if k in notes:
                A(f"_Note:_ {notes[k]}\n")
    A("Figure 1: results/figures/fig1.pdf and fig1.png (a: validation rescue by layer for both models with bootstrap bands; b: selected expert, "
      "active-random controls and specificity; c: coalition patches vs the MoE-block patch at the selected layer).\n")
    A("![Figure 1](figures/fig1.png)\n")

    # ---------------------------------------------------------------- secondary run
    A("## 6. Secondary run with the paper-like object-token rule\n")
    if alt_table:
        A(alt_table)
        A("These runs use the paper's case IDs but define Delta with tok(obj) when that is a single token (see section 4). The main tables above use the "
          "HANDOFF default (leading-space continuation token).\n")
    else:
        A("Not available.\n")
    if extra_sections:
        A(extra_sections)

    # ---------------------------------------------------------------- deviations
    A("## 7. Deviations, assumptions and open points\n")
    A("- **Object token convention**: HANDOFF default (leading-space continuation token) used throughout; the paper most likely used tok(obj) first (section 4). Secondary run in section 6.")
    A("- **Not-clean-active selected expert**: rescue set to 0 for that case (paper's convention); the literal patch -c_e(noised) for noised-only-active experts is recorded in expert_rows.parquet (kind expert_noised_only) but not used in the tables.")
    A("- **Active-random controls**: drawn with random.Random(1000 + case_id) from the other clean-active experts (Qwen3: 3; Mixtral: the unique other). The paper's draws cannot be recovered; the gate-matched, all-active-rank and equal-norm controls do not depend on draws.")
    A("- **Relaxed set construction**: single seed-0 scan, first 512 relaxed passes; the paper evidently used a different order (its relaxed validation overlaps its strict set in only 15 cases). Our relaxed-new subsets are smaller (Table 15).")
    A("- **Precision**: bf16 weights/activations with fp32 accumulation, matching HF; routing weights cast to bf16 for Qwen3 (HF behaviour) and kept in fp32 for Mixtral (transformers 5.16 behaviour). Expert contributions are the exact fp32 products w_e * E_e(x); the block output is their fp32 sum cast to bf16 (HF accumulates in bf16 in expert-index order; difference is at the bf16 rounding level).")
    A("- **Delta from bf16 logits** of the true and foil tokens (bmm on the two lm_head rows, fp32 accumulation, identical to the full-vocab matmul on every tested row).")
    A("- **Zero rows, n_E in Table 15, fold assignment in Table 12, split function in Appendix D**: not specified in the paper; our definitions are stated in the table notes.")
    A("- **Layer sweep on relaxed set** re-selects the layer (Appendix F reading in PLAN.md section 9); it selected the same layer as the strict set for both models, so no separate fixed-layer table was needed.")
    A("- **HF reference check on the big models**: see section 3 (device_map=auto with CPU/disk offload; slow, run last).")
    A("")

    # ---------------------------------------------------------------- honest statement
    A("## 8. What did and did not reproduce\n")
    for mk, r in res.items():
        s = r["sets"]["paper"]
        la = s["layer"]
        ev = s.get("eval")
        e = s.get("selection", {}).get("e_star")
        pl, pe = MODELS[mk]["paper_layer"], MODELS[mk]["paper_expert"]
        bits = [f"layer selection {'matches' if la['L_star'] == pl else 'differs from'} the paper (L{la['L_star']} vs L{pl})"]
        if ev:
            bits.append(f"expert selection {'matches' if e == pe else 'differs from'} the paper (E{e:03d} vs E{pe:03d})")
            pr = {"qwen3": ((0.752, 1.053), (0.344, 0.590), (0.276, 0.533)), "mixtral": ((0.331, 0.579), (0.018, 0.175), (-0.284, -0.072))}[mk]
            inside = [pr[0][0] <= la["val_at_Lstar"]["mean"] <= pr[0][1], pr[1][0] <= ev["rescue_all"]["mean"] <= pr[1][1], pr[2][0] <= ev["spec_all"]["mean"] <= pr[2][1]]
            bits.append(f"our validation means fall inside the paper's CIs for {sum(inside)}/3 of (layer rescue, expert rescue, specificity)")
            bits.append(f"specificity sign {'matches' if np.sign(ev['spec_all']['mean']) == np.sign(pr[2][0] + pr[2][1]) else 'DIFFERS'} (ours {ev['spec_all']['mean']:+.3f})")
        A(f"- **{MODELS[mk]['label']}**: " + "; ".join(bits) + ".")
    A("")
    A("The reproduction is an independent re-implementation from the paper text (no code was released); the funnel analysis shows one protocol detail "
      "(object-token convention) where our default differs from what the paper most likely did. Everything else follows PLAN.md / HANDOFF.md.\n")
    txt = "\n".join(L)
    with open(os.path.join(RESULTS, "REPORT.md"), "w") as f:
        f.write(txt)
    return txt
