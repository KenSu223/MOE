"""Build a markdown section for a secondary Mixtral run (e.g. mixtral_nobos) on the paper case set:
layer curve summary, per-expert table at L19, selection, evaluation of the selected expert and of the paper's E006,
active-pair equal-norm check, coalitions, relation breakdown, stability grid, relation-held-out.
Usage: python scripts/mixtral_run_section.py <run_name> "<title>" [layer]
Writes results/tables/<run_name>_section.md and prints it."""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import analysis as A
from moetrace.models import MODELS, RESULTS
from moetrace.report import md_table, fmt, TABLES

run = sys.argv[1]; title = sys.argv[2] if len(sys.argv) > 2 else run
L = int(sys.argv[3]) if len(sys.argv) > 3 else 19
PAPER_E, NC = 6, MODELS["mixtral"]["n_controls"]
md = A.load_model(run)
disc, val = md.ids("paper", "discovery"), md.ids("paper", "validation")
la = A.layer_analysis(md, "paper")
out = [f"### {title}\n"]
out.append(f"Run directory `results/{run}`; paper case IDs (128 discovery / 128 validation); paper IDs passing the strict filter under this tokenisation: "
           f"{int(md.cases.strict.sum())}/{len(md.cases)}; mean clean Delta {md.cases.delta_clean.mean():+.3f}, mean subject-noise drop {(md.cases.delta_clean - md.cases.delta_noised).mean():+.3f}.\n")
sh = la["sharpness"]
out.append(md_table(["Quantity", "Ours", "Paper"],
    [["Discovery-selected layer", f"L{la['L_star']}", "L19"],
     ["Validation rescue at L19", fmt(A.summarize(md.R.loc[val, 19].values)), "+0.457 [+0.331, +0.579]"],
     ["Validation top layer / rescue", f"L{sh['top_layer']} {sh['top_rescue']:+.3f}", "L21 +0.496"],
     ["Validation next layer / rescue", f"L{sh['next_layer']} {sh['next_rescue']:+.3f}", "L19 +0.457"],
     ["Discovery top-5 layers", ", ".join(f"L{l} {v:+.3f}" for l, v in la["disc_curve_top5"]), ""]],
    os.path.join(TABLES, f"{run}_layer"), f"{title}: layer level"))
if md.expert_rows is None or L not in set(md.expert_rows.layer):
    out.append("Expert pass not available for this run.\n")
else:
    et = A.expert_table(md, L)
    rows = []
    for e in range(8):
        sub = et[et.expert == e]
        rows.append([f"E{e:03d}", f"{int(sub.case_id.isin(disc).sum())}/128", f"{int(sub.case_id.isin(val).sum())}/128",
                     f"{sub[sub.case_id.isin(disc)].rescue.sum() / len(disc):+.3f}", f"{sub[sub.case_id.isin(val)].rescue.sum() / len(val):+.3f}",
                     f"{sub[sub.case_id.isin(val)].rescue.mean():+.3f}" if sub.case_id.isin(val).any() else "n/a"])
    out.append(md_table(["Expert", "Disc. active", "Val. active", "Disc. all-case mean rescue", "Val. all-case mean rescue", "Val. active-only mean rescue"],
                        rows, os.path.join(TABLES, f"{run}_per_expert_L{L}"), f"{title}: every expert at L{L} (paper: E006 active 91/128 disc, 83/128 val)"))
    sel = A.select_expert(et, disc, len(disc) // 2)
    out.append(f"Recurrence-first selection (>= 64 of 128 discovery cases, highest all-case mean rescue): **E{sel['e_star']:03d}** "
               f"({sel['n_candidates']} candidates; paper: E006).\n")
    rows = []
    for e in sorted({sel["e_star"], PAPER_E}):
        ev = A.evaluate_expert(md, L, e, val, NC)
        ap = A.active_pair_equal_norm(md, L, e, val)
        rows.append([f"L{L}E{e:03d}" + (" (selected)" if e == sel["e_star"] else "") + (" (paper's expert)" if e == PAPER_E else ""),
                     f"{ev['val_active']}/128", fmt(ev["rescue_all"]), fmt(ev["control_all"]), fmt(ev["spec_all"]),
                     f"{ev['spec_all']['p']:.4f}" if "p" in ev["spec_all"] else "", f"{ap['n']}", fmt(ap["spec_raw"]), fmt(ap["selected_eq"]), fmt(ap["other_eq"]), fmt(ap["spec_eq"])])
    rows.append(["Paper L19E006", "83/128", "+0.099 [+0.018, +0.175]", "", "-0.175 [-0.284, -0.072]", "", "83", "-0.081 [-0.214, +0.042]",
                 "+0.046 [-0.007, +0.096]", "+0.108 [+0.047, +0.172]", "-0.062 [-0.130, -0.003]"])
    out.append(md_table(["Expert", "Val. active", "Expert rescue (all-case)", "Active-random control", "Spec", "Spec sign-flip p",
                         "Anchor-active n", "Raw active-pair spec", "Selected equal-norm", "Other equal-norm", "Equal-norm spec"],
                        rows, os.path.join(TABLES, f"{run}_experts_L{L}"), f"{title}: validation results (Tables 1, 11 analogues)"))
    co = A.coalitions(md, L, val)
    out.append(md_table(["Patch", "Rescue (ours)", "Paper"],
        [["Clean top-2 coalition", fmt(co["coalition_clean"]), "+0.461 [+0.343, +0.572]"],
         ["Routing-union coalition", fmt(co["coalition_union"]), "+0.490 [+0.367, +0.613]"],
         ["L19 MoE-block patch (same pass)", fmt(co["layer"]) if "layer" in co else "", "+0.457 [+0.331, +0.579]"]],
        os.path.join(TABLES, f"{run}_coalitions_L{L}"), f"{title}: coalition patching (Table 16 analogue)"))
    ev6 = A.evaluate_expert(md, L, PAPER_E, val, NC)
    rb = A.relation_breakdown(md, ev6)
    paper5 = {"P103": ("+0.312", "+0.206", "0.67"), "P17": ("+0.187", "-0.272", "0.09"), "P1412": ("+0.031", "-0.076", "0.33"), "P27": ("+0.049", "-0.132", "0.22"),
              "P740": ("+0.146", "-0.167", "0.56"), "P495": ("+0.094", "-0.358", "0.11"), "P178": ("+0.000", "-0.368", "0.00"), "P136": ("+0.111", "-0.562", "0.11"),
              "P106": ("+0.223", "+0.179", "0.71"), "P131": ("-0.016", "-0.557", "0.00")}
    rows = [[r.relation, int(r.n), f"{r.expert_rescue:+.3f}", f"{r.specificity:+.3f}", f"{r.pos_frac:.2f}", *paper5.get(r.relation, ("", "", ""))] for r in rb.itertuples()]
    out.append(md_table(["Relation", "n", "E006 rescue", "E006 spec", "Pos. frac.", "Paper rescue", "Paper spec", "Paper pos."], rows,
                        os.path.join(TABLES, f"{run}_relations_E006"), f"{title}: relation-wise validation for L19E006 (Table 5 analogue)"))
    sg = A.stability_grid(md, L, "paper", NC, reference_expert=PAPER_E)
    rh = A.relation_heldout(md, L, "paper", NC, reference_expert=PAPER_E)
    out.append(md_table(["Check", "Ours", "Paper"],
        [["Stability grid: E006 selected", f"{sg['n_reference_selected']}/{sg['n_settings']} (selected counts {sg['selected_counts']})", "(Qwen3 only in the paper)"],
         ["Stability grid: mean val rescue / spec", f"{sg['mean_val_rescue']:+.3f} / {sg['mean_val_spec']:+.3f}", ""],
         ["Relation-held-out: E006 selected", f"{rh['n_selected_reference']}/5 folds; active {rh['active']}/{rh['n_heldout']}", ""],
         ["Relation-held-out: rescue / spec", f"{fmt(rh['rescue'])} / {fmt(rh['spec'])}", ""]],
        os.path.join(TABLES, f"{run}_stability"), f"{title}: selection stability (Appendix D analogue)"))
txt = "\n".join(out)
open(os.path.join(TABLES, f"{run}_section.md"), "w").write(txt)
print(txt)
