"""ext13syn: section results/sections/ext13_curves.md (Task B, P5-4), generated from results/ext13syn_summary.json."""
import os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np
from moetrace.models import RESULTS

TAB = os.path.join(RESULTS, "tables")


def fci(x, d=2, sign=True):
    if not x or x[0] is None or (isinstance(x[0], float) and not np.isfinite(x[0])):
        return "n/a"
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    if len(x) < 3 or x[1] is None or x[2] is None:
        return f.format(x[0])
    return f"{f.format(x[0])} [{f.format(x[1])}, {f.format(x[2])}]"


def neg(x):
    return [-x[0], -x[2], -x[1]]


def tab(name):
    p = os.path.join(TAB, name + ".md")
    return open(p).read() if os.path.exists(p) else ""


def write(summ, SEC):
    cur = summ["curves"]
    C = cur["conditions"]
    X = cur.get("across", {})
    g = lambda k: C.get(k)
    main = [e for e in C.values() if e["family"] == "main"]
    over = sorted(main, key=lambda e: -e["overshoot"][0])
    big = [e for e in main if e["overshoot"][0] >= 0.15]
    small = [e for e in main if e["overshoot"][0] < 0.05]
    late_frac = [e["neg_late_frac"] for e in main if e.get("neg_late_frac") is not None and e["ratios"]["N"][0] < -0.1]
    dsh = [e["neg_direct_share"]["last 4"] for e in main if (e.get("neg_direct_share") or {}).get("last 4") is not None]
    xb = X.get("both", {})
    rl = (xb.get("late") or {}).get("r")
    ra = (xb.get("A_direct") or {}).get("r")
    rn = (xb.get("-N_late") or {}).get("r")
    pr = [e["per_case"]["r(overshoot_c, late_c)"][0] for e in big]
    comp = [e for e in main if e.get("compensation")]
    ba = [e["compensation"]["b_attn"][0] for e in comp if e["level"] not in ("cf",)]
    bm = [e["compensation"]["b_moe_early"][0] for e in comp if e["level"] not in ("cf",)]
    nba = sum(1 for e in comp if e["level"] != "cf" and e["compensation"]["b_attn"][2] < 0)
    nbm = sum(1 for e in comp if e["level"] != "cf" and e["compensation"]["b_moe_early"][2] < 0)
    f1q, f1m, oq, om = g("f1_qwen3"), g("f1_mixtral_bos"), g("wino_qwen3"), g("wino_mixtral")
    cfq, cfm = g("cf_qwen3"), g("cf_mixtral")
    gpu = {k: e["gpucheck"] for k, e in C.items() if e.get("gpucheck")}
    L = []
    s = ("**Summary.** Why the expert add-back curves have different shapes (P5-4; agent p5-synth; validation cases of every add-back run, "
         "Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS, STR, final position). The overshoot of the oracle add-back curve over the all-MoE ceiling M is, "
         f"condition by condition, the late-MoE term M(≤ L−5) − M: across the {xb.get('n', 0)} main-family conditions of both models Pearson r = "
         f"{fci(rl)} (slope {xb.get('late', {}).get('slope', float('nan')):.2f}), and inside every condition that overshoots by ≥ 0.15 the per-case "
         f"overshoot tracks the per-case late term (r {min(pr):.2f}–{max(pr):.2f}). ")
    if late_frac:
        s += (f"Every task has experts with a negative single-expert effect, and the cross-fitted negative mass sits in the last four layers "
              f"(last-four-layer over total negative mass: median {np.median(late_frac):.2f} over the conditions with a clear negative mass; the "
              "earlier layers net to about zero), "
              f"where it writes against the answer directly (DLA / total of the negative set in the last four layers {min(dsh):.2f}–{max(dsh):.2f}). ")
    s += (f"The option swap has such experts too (Mixtral −N in the last four layers {fci(neg(om['ratios']['N|last 4']))}), yet it does not overshoot "
          f"({fci(om['overshoot'])}), so the negative experts alone do not decide the shape. What decides it is where the answer evidence they react to "
          "comes from: the late MoE writes against the answer in proportion to the evidence already in the residual (per-case regression of the "
          f"last-four-layer MoE DLA: median coefficient {np.median(ba):+.2f} per logit of attention DLA, CI below zero in {nba} of {len(ba)} conditions; "
          f"{np.median(bm):+.2f} per logit of earlier-MoE DLA, CI below zero in {nbm} of {len(bm)}; CounterFact excluded). An expert-only add-back "
          "restores the MoE-written evidence but not the attention-written "
          "evidence, so when attention writes much of the answer (attribute → entity, CounterFact) the ceiling M contains the late "
          "experts' clean reaction to evidence the patched run does not carry, and any subset that leaves the late negatives at their recomputed values "
          f"overshoots M (f1 {fci(f1q['overshoot'])} Qwen3, {fci(f1m['overshoot'])} Mixtral); when the MoE writes the answer (option swap) the recomputed late "
          f"experts react as in the clean run and nothing is gained (Qwen3 {fci(oq['overshoot'])}). Across conditions the overshoot also tracks the "
          f"direct attention share (r {fci(ra)}). ")
    if "f1_qwen3" in gpu:
        r = gpu["f1_qwen3"]["r"]
        d = gpu["f1_qwen3"]["diff"]
        s += (f"GPU check (Qwen3 f1): adding back every expert except the discovery-negative ones gives {fci(r['noneg'], 3, False)} vs M "
              f"{fci(r['all_moe'], 3, False)} (oracle maximum {f1q['max_r'][0]:.3f}); excluding only the last-four-layer negatives gives "
              f"{fci(r['noneg_late'], 3, False)}, excluding only the earlier ones {fci(r['noneg_early'], 3, False)}; and the late MoE's mismatch "
              f"(recomputed − clean) is {fci(d['late mismatch | MoE upstream (int_moe - all_moe)'], 3)} when only the MoE upstream is restored but "
              f"{fci(d['late mismatch | both upstream (intblock - intblock_latemoe)'], 3)} when attention and MoE upstream are restored.")
        if "dir_opt_qwen3" in gpu:
            r2, d2 = gpu["dir_opt_qwen3"]["r"], gpu["dir_opt_qwen3"]["diff"]
            s += (f" In the option swap (dir_opt) the same negative set patched alone costs {fci(r2['negonly'], 3)}, yet leaving it out of the "
                  f"full expert patch changes nothing ({fci(d2['noneg - all_moe'], 3)}): recomputed in a run whose earlier experts are clean, these "
                  "experts produce their clean anti-answer output again.")
    L.append(s)
    L.append("")
    L.append("### What was computed")
    L.append("")
    L.append(
        "Runs: every add-back run with a complete A0 / A1 set (`results/p5_<cond>_<proto>[_rep]_ab`, Phase 3 `results/{qwen3,mixtral_bos}_str_addback` "
        "(CounterFact) and `results/wino_{qwen3,mixtral_bos}_str_addback` (option swap)), their source runs' single-expert rows (`str_expert_rows`, "
        "all layers, discovery and validation), the add-back runs' expert DLA (`addback_dla`) and the source runs' per-layer DLA (`dla_rows`). "
        "Per condition, on the validation cases (ratios = sum over cases / sum of drops; unit-cluster bootstrap, 5,000 resamples, seed 0):")
    L.append("")
    for k, v in cur["definitions"].items():
        L.append(f"- **{k}**: {v}.")
    L.append(
        "- Why cross-fitted: a single-expert rescue carries bf16 noise (SD ≈ 0.17 logits per single in Qwen3, ≈ 0.12 in Mixtral's early layers) and a "
        "Qwen3 row has 384 singles, so 'the sum of a row's negative singles' is mostly noise (the 'Nrow' column of the CSV: −1.7 to −4.6 of the drop in "
        "Qwen3). Fixing each (layer, expert)'s sign on the discovery cases and summing on the validation cases removes that selection bias; experts unseen "
        "in discovery (U) carry ≤ 0.02 of the drop.")
    L.append(
        "- GPU check (`scripts/ext13syn_gpucheck.py`, one queued job, Qwen3 f1 / dir_f1 / dir_opt validation rows, engine kind `multi` read-only): every "
        "expert := clean except the discovery-negative set (noneg), the same restricted to the last four or to the earlier layers, the negative set alone, "
        "the row's own negative singles, the last four MoE layers alone, and attention (+ MoE) restored below L−4 with or without the late MoE at "
        "clean values (reaction test). All values same pass.")
    L.append("")
    L.append("### 1. The overshoot is the late-MoE term")
    L.append("")
    L.append(tab("ext13syn_curves_summary"))
    L.append("![overshoot vs predictors](../figures/ext13syn_curves_overshoot.png)")
    L.append("")
    lines = []
    for scope, lab in (("both", "both models"), ("qwen3", "Qwen3"), ("mixtral_bos", "Mixtral")):
        r = X.get(scope)
        if not r:
            continue
        lines.append(f"- {lab} ({r['n']} conditions): " + "; ".join(f"{k} r {fci(v['r'])}" for k, v in r.items() if k != "n") + ".")
    L += lines
    L.append("")
    L.append(
        f"Reading. M is the all-MoE ceiling and the oracle maximum sits at k* = 320 of 384 (Qwen3) or 48 of 64 (Mixtral), i.e. just before the "
        "most negative singles of the row enter. M(≤ L−5) − M compares the same expert patch with and without the last four MoE layers forced to their "
        "clean values; it is the overshoot's counterpart in a single joint patch. The two agree within about 0.05 everywhere except Qwen3 CounterFact, where "
        f"the last four layers also hold the positive factual expert L44E069 (late term {fci(cfq['late'])}, but the negative experts inside that band still "
        f"give −N {fci(neg(cfq['ratios']['N|last 4']))} and the overshoot {fci(cfq['overshoot'])}).")
    L.append("")
    L.append("### 2. Where the negative experts are, and how they write")
    L.append("")
    L.append(tab("ext13syn_curves_bands"))
    L.append("![layer profiles](../figures/ext13syn_curves_layers.png)")
    L.append("")
    L.append(
        "The cross-fitted negative mass is concentrated in the last four layers in every condition and model (Qwen3 L44–L47, Mixtral L28–L31); "
        "earlier bands net to about zero once noise is not selected. Its DLA is about as large as or larger than its total effect "
        f"(D/T {min(dsh):.2f}–{max(dsh):.2f}), so these experts write against the answer directly at the final position, and the layers after them "
        "do not add to it (if anything they undo a part). "
        "The same experts are present in the option swap: Mixtral's last four layers write "
        f"{om['ratios']['T|last 4'][0]:+.2f} of the drop (all experts; DLA {om['ratios']['DT|last 4'][0]:+.2f}) in the option swap and "
        f"{f1m['ratios']['T|last 4'][0]:+.2f} (DLA {f1m['ratios']['DT|last 4'][0]:+.2f}) in f1. This matches the writer section's depth bands "
        "(`ext13_writer.md`: the last four layers' summed writes are negative with D/T ≈ 1–1.7 wherever their net write is clearly negative) and "
        "Phase 3's observation of a small negative last-layer MoE in every task (ext7_controls reading 2).")
    L.append("")
    L.append("### 3. Per case")
    L.append("")
    L.append(tab("ext13syn_curves_rows"))
    ov = [e for e in main if e["overshoot"][0] >= 0.05]
    rl_ = [e["per_case"]["r(overshoot_c, late_c)"][0] for e in ov]
    rn_ = [e["per_case"]["r(overshoot_c, -N_late_c)"][0] for e in ov]
    ra_pos = [e for e in main if (e["per_case"].get("r(overshoot_c, A_direct_c)") or [0, 0, 0])[1] is not None
              and (e["per_case"].get("r(overshoot_c, A_direct_c)") or [0, 0, 0])[1] > 0]
    ra_not = [f"{e['level']} {e['model']} ({fci(e['per_case'].get('r(overshoot_c, A_direct_c)'))})" for e in main if e not in ra_pos]
    L.append(
        f"Within a condition, the case-level overshoot at k* is almost the case-level late term (r {min(rl_):.2f}–{max(rl_):.2f} in the "
        f"{len(ov)} conditions that overshoot by ≥ 0.05), and both correlate only moderately with the case's own cross-fitted late negative mass "
        f"(r {min(rn_):.2f}–{max(rn_):.2f}): which experts react varies across cases, but the late band as a whole carries the overshoot. The "
        f"case-level direct attention share predicts the case-level overshoot (CI above zero) in {len(ra_pos)} of {len(main)} conditions; the "
        f"exceptions are Qwen3 entity → attribute conditions with almost no attention write (A_direct ≤ 0.13): {'; '.join(ra_not)}.")
    L.append("")
    L.append("### 4. Do the late experts react to the answer evidence? (compensation regression)")
    L.append("")
    L.append(tab("ext13syn_curves_compensation"))
    L.append("![compensation](../figures/ext13syn_curves_compensation.png)")
    L.append("")
    neg_both = [e for e in comp if e["compensation"]["b_attn"][2] < 0 and e["compensation"]["b_moe_early"][2] < 0]
    exc = []
    for e in comp:
        for k, lab in (("b_attn", "attention"), ("b_moe_early", "earlier MoE")):
            v = e["compensation"][k]
            if v[2] >= 0:
                exc.append(f"{e['level']} {e['model']} b({lab}) {fci(v)}")
    L.append(
        f"Per case (logits, clean minus corrupted): the last-four-layer MoE DLA falls with both the attention DLA and the earlier-MoE DLA (both CIs below "
        f"zero) in {len(neg_both)} of {len(comp)} conditions — the late band suppresses the answer in proportion to how much evidence for it is already "
        "in the residual, whichever "
        "sublayer wrote it (a confidence-regulation / anti-overconfidence pattern; compare the late 'erasure' MLPs of McGrath et al. 2023, "
        "arXiv:2307.15771, the suppression neurons of the final layers in Gurnee et al. 2024, arXiv:2401.12181, and Stolfo et al. 2024, "
        "arXiv:2406.16254). Coefficients whose CI reaches zero or above: " + ("; ".join(exc) if exc else "none") + ". The entity → attribute "
        "conditions of Qwen3 have a small attention DLA range (A_direct ≈ 0.05–0.13), and Qwen3 CounterFact's band contains the positive factual "
        "writer L44E069 (CounterFact rows use the add-back DLA approximation). This is correlational: items with stronger evidence could differ in "
        "other ways; the GPU check below tests the reaction directly.")
    L.append("")
    L.append("### 5. GPU check")
    L.append("")
    if gpu:
        L.append(tab("ext13syn_curves_gpucheck"))
        L.append("![gpu check](../figures/ext13syn_curves_gpucheck.png)")
        L.append("")
        for k, gg in gpu.items():
            r, d = gg["r"], gg["diff"]
            e = C[k]
            L.append(
                f"- **{k}** ({gg['n_cases']} validation cases; oracle maximum of the add-back run {e['max_r'][0]:.3f}, overshoot {fci(e['overshoot'], 3)}): "
                f"M {fci(r['all_moe'], 3, False)}; all but the negatives {fci(r['noneg'], 3, False)} (gain {fci(d['noneg - all_moe'], 3)}), "
                f"late negatives only excluded {fci(d['noneg_late - all_moe'], 3)}, earlier ones only {fci(d['noneg_early - all_moe'], 3)}, the row's own "
                f"negatives excluded {fci(d['noneg_row - all_moe'], 3)}; negatives alone {fci(r['negonly'], 3)}; last four MoE layers alone "
                f"{fci(r['lateonly'], 3)}. Late-MoE mismatch (recomputed − clean): MoE upstream restored {fci(d['late mismatch | MoE upstream (int_moe - all_moe)'], 3)}, "
                f"attention upstream {fci(d['late mismatch | attention upstream (intattn - intattn_latemoe)'], 3)}, both "
                f"{fci(d['late mismatch | both upstream (intblock - intblock_latemoe)'], 3)}"
                + (f"; additivity negonly + noneg − M {fci(gg['additivity'], 3)}" if gg.get("additivity") else "") + ".")
        L.append("")
        L.append(
            "Reading. Leaving out the discovery-negative experts reproduces the oracle maximum, and all of the gain comes from the last four layers. "
            "The effects are not additive: patched alone into the corrupted run the negatives cost about twice what leaving them out gains in "
            "attribute → entity, and in the option swap leaving them out gains nothing although alone they cost about a quarter of the drop — "
            "recomputed in a run whose earlier experts are clean, they write part (attribute → entity) or all (option swap) of their clean "
            "anti-answer output again. The reaction test separates the sources: with only the MoE upstream restored, the recomputed late MoE falls "
            "short of its clean anti-answer write by about a quarter to a third of the drop in attribute → entity but not in the option swap. With "
            "the attention upstream restored the late MoE matches its clean values; that part is close to guaranteed, because at the final position "
            "restoring every attention output below L−4 also restores the earlier MoE outputs exactly (the MoE is per-token), so the late MoE inputs "
            "differ from the clean ones only through the recomputed late attention. The informative contrast is therefore the MoE-only one: the late "
            "experts' clean anti-answer write depends on the part of the residual that attention writes, and the add-back ceiling M contains that "
            "write although the expert-only patch does not restore its input.")
    else:
        L.append("Pending: `results/ext13syn_gpucheck/rows_<task>.parquet` not present yet (job `ext13syn-gpucheck` in `logs/gpu_queue.log`); "
                 "the section is regenerated by `python scripts/ext13syn_all.py --parts curves,text`.")
    L.append("")
    L.append("### Readings")
    L.append("")
    L.append(
        "1. **Two shapes, one mechanism.** The rise-to-ceiling shape (option swap) and the overshoot-and-fall shape (CounterFact, attribute → entity, Mixtral's "
        "copy-prefix templates) differ in the late-MoE term, not in whether answer-opposing experts exist. The fall at the end of the curve is the last "
        "four layers' experts entering at their clean values.")
    L.append(
        "2. **The hypothesis holds, with a refinement.** Overshoot appears when the answer is written by attention (across conditions r with A_direct "
        f"{fci(ra)}; per case in {len(ra_pos)} of {len(main)} conditions). The late experts do not counter attention specifically: they also counter "
        "MoE-written evidence (regression; in dir_opt the recomputed negatives reproduce their clean output in a MoE-restored run), but only "
        "attention-written evidence is missing from an expert-only add-back, so only then is their clean reaction mismatched. When experts write the "
        "answer (option swap), the same late experts push back in the patched run as in the clean run and the curve saturates at M.")
    L.append(
        "3. **Consequences for reading add-back curves.** The all-MoE ceiling understates what the early and middle experts deliver whenever attention "
        "writes part of the answer; 'k for 80 % of M' then counts experts toward a depressed target. The ≤ L−5 ceiling (or the oracle maximum) is the "
        "better reference for attention-written tasks, and the late band is better read as a regulator of confidence than as part of the circuit "
        "that computes the answer.")
    L.append("")
    L.append("### Caveats")
    L.append("")
    L.append("- Correlations across conditions use ~20 condition points (two models, overlapping item pools); the bootstrap over conditions treats them "
             "as independent.")
    L.append("- The negative set is defined by population sign on the discovery split; experts that help some cases and hurt others are assigned by "
             "their net effect. Template conditions have 8–18 seed clusters per split.")
    L.append("- CounterFact's compensation regression uses the add-back run's expert DLA for the late band (clean-active experts only; experts routed "
             "only in the corrupted run are missing) because the ext6 runs have no per-layer DLA.")
    L.append("- The GPU check covers three Qwen3 conditions; Mixtral is inferred from the CPU analyses (its oracle maximum at k* = 48 of 64 already "
             "excludes the late band's two most negative experts per layer).")
    L.append("- f2e and any later condition enter when their add-back runs are complete (`python scripts/ext13syn_all.py`).")
    L.append("")
    L.append("Files: `moetrace/ext13syn_curves.py`, `scripts/ext13syn_curves_report.py`, `scripts/ext13syn_gpucheck.py`, tables "
             "`results/tables/ext13syn_curves_{summary,bands,rows,compensation,across,gpucheck}.{md,csv}`, figures "
             "`results/figures/ext13syn_curves_{overshoot,layers,compensation,gpucheck}.{png,pdf}`, GPU rows `results/ext13syn_gpucheck/`, numbers "
             "`results/ext13syn_summary.json` (key curves).")
    open(os.path.join(SEC, "ext13_curves.md"), "w").write("\n".join(L) + "\n")
