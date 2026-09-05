"""Tables 1-16 (markdown + CSV), Figure 1 and summary.json from the row-level results."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import analysis as A
from .models import MODELS, RESULTS
from .stats import fmt, summarize

TABLES = os.path.join(RESULTS, "tables")
FIGS = os.path.join(RESULTS, "figures")

# ---- paper numbers (from paper_src/acl_latex.tex) ---------------------------------------------------------------
P = {
    "qwen3": {"layer": 44, "layer_rescue": "+0.901 [+0.752, +1.053]", "expert": "L44E069", "expert_rescue": "+0.463 [+0.344, +0.590]",
              "spec": "+0.400 [+0.276, +0.533]", "disc_active": "112/128", "val_active": "116/128", "zero": 26,
              "sharp": ("L44", "+0.901", "L42", "+0.592", "+0.309"),
              "ratio_expert": "+0.515 [+0.411, +0.613]", "ratio_spec": "+0.444 [+0.323, +0.560]",
              "relations": "P17:21, P103:16, P136:14, P178:13, P27:13, P131:12, P159:12, P30:12, P20:11, P176:11, P740:9, P364:9",
              "relaxed": ("+0.846 [+0.760, +0.934]", "230/256", "+0.454 [+0.367, +0.545]", "+0.413 [+0.318, +0.511]"),
              "relaxed_split": {"all": ("256", "+0.833 [+0.714, +0.962]", "137", "+0.400 [+0.293, +0.516]", "+0.360 [+0.239, +0.487]"),
                                "new": ("241", "+0.860 [+0.739, +0.989]", "130", "+0.420 [+0.310, +0.541]", "+0.380 [+0.259, +0.509]")}},
    "mixtral": {"layer": 19, "layer_rescue": "+0.457 [+0.331, +0.579]", "expert": "L19E006", "expert_rescue": "+0.099 [+0.018, +0.175]",
                "spec": "-0.175 [-0.284, -0.072]", "disc_active": "91/128", "val_active": "83/128", "zero": 54,
                "sharp": ("L21", "+0.496", "L19", "+0.457", "+0.038"),
                "ratio_expert": "+0.216 [+0.044, +0.388]", "ratio_spec": "-0.383 [-0.645, -0.159]",
                "relations": "P17:28, P103:26, P27:16, P178:15, P495:15, P136:15, P740:15, P106:14, P1412:14, P131:11, P20:10, P364:9",
                "relaxed": ("+0.421 [+0.352, +0.489]", "174/256", "+0.114 [+0.041, +0.187]", "-0.155 [-0.247, -0.067]"),
                "relaxed_split": {"all": ("256", "+0.454 [+0.360, +0.557]", "137", "+0.119 [-0.003, +0.240]", "-0.124 [-0.270, +0.014]"),
                                  "new": ("241", "+0.459 [+0.360, +0.561]", "130", "+0.123 [-0.010, +0.250]", "-0.128 [-0.280, +0.014]")}},
}
P_REL = {  # Table 5 paper rows: (model, relation) -> (n, rescue, spec, pos)
    ("qwen3", "P103"): (12, "+0.427", "+0.406", "0.83"), ("qwen3", "P27"): (9, "+0.826", "+0.780", "1.00"), ("qwen3", "P136"): (7, "+0.018", "-0.000", "0.57"),
    ("qwen3", "P131"): (6, "+1.240", "+1.160", "0.83"), ("qwen3", "P17"): (6, "+1.073", "+0.910", "0.67"), ("qwen3", "P30"): (6, "+0.292", "+0.312", "0.50"),
    ("qwen3", "P1412"): (6, "+0.198", "+0.163", "0.67"), ("qwen3", "P413"): (6, "+0.042", "-0.174", "0.17"), ("qwen3", "P937"): (5, "+1.387", "+1.350", "1.00"),
    ("qwen3", "P159"): (5, "+1.012", "+0.988", "0.80"),
    ("mixtral", "P103"): (12, "+0.312", "+0.206", "0.67"), ("mixtral", "P17"): (11, "+0.187", "-0.272", "0.09"), ("mixtral", "P1412"): (9, "+0.031", "-0.076", "0.33"),
    ("mixtral", "P27"): (9, "+0.049", "-0.132", "0.22"), ("mixtral", "P740"): (9, "+0.146", "-0.167", "0.56"), ("mixtral", "P495"): (9, "+0.094", "-0.358", "0.11"),
    ("mixtral", "P178"): (9, "+0.000", "-0.368", "0.00"), ("mixtral", "P136"): (9, "+0.111", "-0.562", "0.11"), ("mixtral", "P106"): (7, "+0.223", "+0.179", "0.71"),
    ("mixtral", "P131"): (6, "-0.016", "-0.557", "0.00"),
}
P_GATE = {"sel": ("+0.513 [+0.383, +0.651]", "+0.239 [+0.177, +0.302]"), "ctrl": ("+0.054 [+0.017, +0.091]", "+0.051 [+0.016, +0.086]"),
          "spec": ("+0.459 [+0.322, +0.607]", "+0.188 [+0.119, +0.259]"), "n": 116}
P_RANK = {"n": 115, "top1": 53, "top2": 74, "mean_rank": 2.90, "pct": 0.73, "diff": "+0.441 [+0.309, +0.584]"}
P_PAIR = {"n": 83, "spec_raw": "-0.081 [-0.214, +0.042]", "sel_eq": "+0.046 [-0.007, +0.096]", "other_eq": "+0.108 [+0.047, +0.172]", "spec_eq": "-0.062 [-0.130, -0.003]"}
P_HELD = {"folds": 5, "selected": "5 / 5", "n": 256, "active": "229 / 256", "rescue": "+0.443 [+0.343, +0.545]", "spec": "+0.388 [+0.297, +0.484]"}
P_NOISE = {1.0: ("115/128", "+1.259", "+0.098 [+0.040, +0.165]", "0.43"), 2.0: ("115/128", "+4.909", "+0.406 [+0.280, +0.546]", "0.55"),
           3.0: ("115/128", "+5.697", "+0.459 [+0.339, +0.587]", "0.66"), 4.0: ("115/128", "+6.004", "+0.478 [+0.356, +0.605]", "0.67")}
P_COAL = {"coalition_clean": ("+0.461 [+0.343, +0.572]", "0.75"), "coalition_union": ("+0.490 [+0.367, +0.613]", "0.75")}
P_STAB = {"selected": "25/25", "rescue": "+0.398", "spec": "+0.344"}


def md_table(header: list[str], rows: list[list], path: str, caption: str = "") -> str:
    lines = []
    if caption:
        lines.append(f"**{caption}**\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    txt = "\n".join(lines) + "\n"
    with open(path + ".md", "w") as f:
        f.write(txt)
    pd.DataFrame(rows, columns=header).to_csv(path + ".csv", index=False)
    return txt


def f3(x):
    return f"{x:+.3f}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "n/a"


# ---------------------------------------------------------------------------------------------------------------
def analyze_model(model: str) -> dict:
    md = A.load_model(model)
    m = MODELS[model]
    out = {"model": model, "sets": {}}
    for s in A.set_names(md.sets) if hasattr(A, "set_names") else [k for k in ("paper", "strict", "relaxed") if k in md.sets]:
        r = {}
        la = A.layer_analysis(md, s)
        r["layer"] = la
        disc, val = md.ids(s, "discovery"), md.ids(s, "validation")
        L = la["L_star"]
        th = len(disc) // 2
        if md.expert_rows is not None and L in set(md.expert_rows.layer):
            et = A.expert_table(md, L)
            sel = A.select_expert(et, disc, th)
            r["selection"] = sel
            if sel["e_star"] is not None:
                e = sel["e_star"]
                ev = A.evaluate_expert(md, L, e, val, m["n_controls"])
                r["eval"] = {k: v for k, v in ev.items() if k != "per_case"}
                r["eval_per_case"] = ev["per_case"]
                r["relations"] = A.relation_breakdown(md, ev)
                r["ratios"] = A.ratios(md, L, ev, val)
                r["coalitions"] = A.coalitions(md, L, val)
                if m["n_controls"] >= 3:
                    r["gate"] = {k: v for k, v in A.gate_matched_control(md, L, e, val).items() if k != "per_case"}
                    r["rank"] = {k: v for k, v in A.all_active_rank(md, L, e, val).items() if k != "per_case"}
                else:
                    r["pair"] = {k: v for k, v in A.active_pair_equal_norm(md, L, e, val).items() if k != "per_case"}
                    r["rank"] = {k: v for k, v in A.all_active_rank(md, L, e, val).items() if k != "per_case"}
            # paper's expert evaluated as a fixed hypothesis (if different)
            if m["paper_expert"] is not None and sel.get("e_star") != m["paper_expert"] and L == m["paper_layer"]:
                evp = A.evaluate_expert(md, L, m["paper_expert"], val, m["n_controls"])
                r["eval_paper_expert"] = {k: v for k, v in evp.items() if k != "per_case"}
        # paper's layer evaluated as a fixed hypothesis if our L* differs
        if m["paper_layer"] is not None and L != m["paper_layer"]:
            r["paper_layer_val"] = summarize(md.R.loc[val, m["paper_layer"]].values)
            if md.expert_rows is not None and m["paper_layer"] in set(md.expert_rows.layer):
                et2 = A.expert_table(md, m["paper_layer"])
                sel2 = A.select_expert(et2, disc, th)
                r["selection_at_paper_layer"] = sel2
                if sel2["e_star"] is not None:
                    ev2 = A.evaluate_expert(md, m["paper_layer"], sel2["e_star"], val, m["n_controls"])
                    r["eval_at_paper_layer"] = {k: v for k, v in ev2.items() if k != "per_case"}
                    r["coalitions_at_paper_layer"] = A.coalitions(md, m["paper_layer"], val)
                if m["paper_expert"] is not None:
                    ev3 = A.evaluate_expert(md, m["paper_layer"], m["paper_expert"], val, m["n_controls"])
                    r["eval_paper_layer_paper_expert"] = {k: v for k, v in ev3.items() if k != "per_case"}
        out["sets"][s] = r
    # appendix D on the paper set (and strict as extra)
    if md.expert_rows is not None and "paper" in out["sets"] and "selection" in out["sets"]["paper"]:
        L = out["sets"]["paper"]["layer"]["L_star"]
        e = out["sets"]["paper"]["selection"]["e_star"]
        out["stability"] = A.stability_grid(md, L, "paper", m["n_controls"], reference_expert=e)
        out["heldout"] = A.relation_heldout(md, L, "paper", m["n_controls"], reference_expert=e)
        out["noise"] = A.noise_table(md, L, e, md.ids("paper", "validation"))
    out["funnel"] = A.funnel_check(md)
    out["cases"] = md.cases
    out["R"] = md.R
    out["sets_raw"] = md.sets
    return out


# ---------------------------------------------------------------------------------------------------------------
def build(models=("qwen3", "mixtral")) -> dict:
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    res = {}
    for mk in models:
        try:
            res[mk] = analyze_model(mk)
        except FileNotFoundError as ex:
            print("skipping", mk, ex)
    tables = {}
    # ---- Table 1
    rows = []
    for mk, r in res.items():
        s = r["sets"]["paper"]
        L = s["layer"]["L_star"]
        e = s.get("selection", {}).get("e_star")
        ev = s.get("eval")
        rows.append([MODELS[mk]["label"], f"L{L} (paper L{P[mk]['layer']})", fmt(s["layer"]["val_at_Lstar"]), P[mk]["layer_rescue"],
                     f"L{L}E{e:03d}" if e is not None else "n/a", P[mk]["expert"], fmt(ev["rescue_all"]) if ev else "n/a", P[mk]["expert_rescue"],
                     fmt(ev["spec_all"]) if ev else "n/a", P[mk]["spec"]])
    tables[1] = md_table(["Model", "Layer (ours)", "Layer rescue (ours)", "Layer rescue (paper)", "Expert (ours)", "Expert (paper)",
                          "Expert rescue (ours)", "Expert rescue (paper)", "Spec (ours)", "Spec (paper)"], rows,
                         os.path.join(TABLES, "table_01"), "Table 1: main validation results (128 held-out cases per model; paper case IDs)")
    # ---- Table 2 protocol
    rows = [["Record order", "random.Random(0).shuffle over the 21,919 CounterFact records (verified: all paper IDs lie in the first 390/580 records)"],
            ["Cases", "Paper's Table 8 IDs (primary); our own strict 256 and relaxed 512 as secondary sets"],
            ["Filtering", "single-token true/foil (continuation tokenisation with leading space); clean margin >= 1.0; drop >= 0.5 (relaxed 0.5 / 0.25)"],
            ["Subject noise", "one Gaussian draw per case, torch.Generator seed 0 + case_id, scale 3.0 x embedding-matrix std (fp32), added to subject-token embeddings"],
            ["Layer selection", "argmax mean rescue over discovery cases; fixed on validation"],
            ["Expert selection", "clean-active in >= 64 of 128 discovery cases (half the split); highest all-case mean rescue"],
            ["Active-random", "Qwen3: 3 other clean-active experts sampled with random.Random(1000 + case_id); Mixtral: the unique other expert"],
            ["Statistics", "5,000 percentile-bootstrap resamples (seed 0); 10,000 two-sided sign-flip samples (seed 0)"],
            ["Precision", "bf16 weights and activations (as HF), fp32 accumulation; Delta from bf16 logits"]]
    tables[2] = md_table(["Item", "Setting (ours)"], rows, os.path.join(TABLES, "table_02"), "Table 2: reproducibility protocol as implemented")
    # ---- Table 3
    rows = []
    for mk, r in res.items():
        s = r["sets"]["paper"]
        if "eval" not in s:
            continue
        sel, ev = s["selection"], s["eval"]
        rows.append([MODELS[mk]["label"], f"L{s['layer']['L_star']}E{sel['e_star']:03d}", f"{sel['disc_active']}/{sel['n_disc']}", P[mk]["disc_active"],
                     f"{ev['val_active']}/{ev['n_val']}", P[mk]["val_active"], ev["not_active_rows"], ev["zero_rows_selected_only"],
                     ev["zero_rows_selected_plus_controls"], P[mk]["zero"]])
    tables[3] = md_table(["Model", "Expert", "Disc. active (ours)", "Disc. (paper)", "Val. active (ours)", "Val. (paper)", "Val. not-active rows",
                          "Val. exact-zero selected rows", "Val. exact-zero selected+control rows", "Zero (paper)"], rows,
                         os.path.join(TABLES, "table_03"), "Table 3: selected-expert activity and zero-rescue rows")
    # ---- Table 4 relations
    rows = []
    for mk, r in res.items():
        ct = r["cases"]
        for s in ("paper", "strict"):
            if f"in_{s}" not in ct:
                continue
            vc = ct[ct[f"in_{s}"]].relation.value_counts().head(12)
            rows.append([MODELS[mk]["label"], s, ", ".join(f"{k}:{v}" for k, v in vc.items()), P[mk]["relations"] if s == "paper" else ""])
    tables[4] = md_table(["Model", "Case set", "Most frequent relations (ours)", "Paper"], rows, os.path.join(TABLES, "table_04"),
                         "Table 4: most frequent retained relations (256 cases)")
    # ---- Table 5
    rows = []
    for mk, r in res.items():
        s = r["sets"]["paper"]
        if "relations" not in s:
            continue
        rel = s["relations"].head(10)
        for _, x in rel.iterrows():
            p = P_REL.get((mk, x.relation), ("", "", "", ""))
            rows.append([MODELS[mk]["label"], x.relation, int(x.n), f3(x.expert_rescue), f3(x.specificity), f"{x.pos_frac:.2f}",
                         p[0], p[1], p[2], p[3]])
    tables[5] = md_table(["Model", "Relation", "n", "Expert rescue", "Spec", "Pos. frac.", "n (paper)", "Rescue (paper)", "Spec (paper)", "Pos. (paper)"],
                         rows, os.path.join(TABLES, "table_05"), "Table 5: relation-wise expert-level validation (paper case set)")
    # ---- Table 6
    rows = []
    for mk, r in res.items():
        sh = r["sets"]["paper"]["layer"]["sharpness"]
        p = P[mk]["sharp"]
        rows.append([MODELS[mk]["label"], f"L{sh['top_layer']}", f3(sh["top_rescue"]), f"L{sh['next_layer']}", f3(sh["next_rescue"]), f3(sh["gap"]),
                     p[0], p[1], p[2], p[3], p[4]])
    tables[6] = md_table(["Model", "Top layer", "Top rescue", "Next layer", "Next rescue", "Gap", "Top (paper)", "Rescue (paper)", "Next (paper)",
                          "Next rescue (paper)", "Gap (paper)"], rows, os.path.join(TABLES, "table_06"), "Table 6: layer-sweep sharpness on validation cases")
    # ---- Table 7
    rows = []
    for mk, r in res.items():
        s = r["sets"]["paper"]
        if "ratios" not in s:
            continue
        rt = s["ratios"]
        rows.append([MODELS[mk]["label"], "Layer rescue", f3(rt["layer_rescue"]), P[mk]["layer_rescue"].split(" ")[0]])
        rows.append([MODELS[mk]["label"], "Expert/layer", f"{rt['expert_over_layer'][0]:+.3f} [{rt['expert_over_layer'][1]:+.3f}, {rt['expert_over_layer'][2]:+.3f}]", P[mk]["ratio_expert"]])
        rows.append([MODELS[mk]["label"], "Spec/layer", f"{rt['spec_over_layer'][0]:+.3f} [{rt['spec_over_layer'][1]:+.3f}, {rt['spec_over_layer'][2]:+.3f}]", P[mk]["ratio_spec"]])
    tables[7] = md_table(["Model", "Quantity", "Ours", "Paper"], rows, os.path.join(TABLES, "table_07"), "Table 7: expert-level effects relative to selected-layer rescue")
    # ---- Table 8 case ids (ours) + overlap
    rows = []
    for mk, r in res.items():
        sets = r["sets_raw"]
        pap = set(sets["paper"]["discovery"] + sets["paper"]["validation"])
        for s in ("strict", "relaxed"):
            if s not in sets:
                continue
            ids_all = sets[s]["discovery"] + sets[s]["validation"]
            rows.append([MODELS[mk]["label"], s, "discovery", len(sets[s]["discovery"]), len(set(sets[s]["discovery"]) & pap), ", ".join(map(str, sets[s]["discovery"]))])
            rows.append([MODELS[mk]["label"], s, "validation", len(sets[s]["validation"]), len(set(sets[s]["validation"]) & pap), ", ".join(map(str, sets[s]["validation"]))])
    tables[8] = md_table(["Model", "Set", "Split", "n", "Overlap with paper IDs", "Case IDs"], rows, os.path.join(TABLES, "table_08"),
                         "Table 8: our own case sets (paper IDs are in data/paper_case_ids.json)")
    # ---- Table 9 gate matched (qwen3)
    rows = []
    if "qwen3" in res and "gate" in res["qwen3"]["sets"]["paper"]:
        g = res["qwen3"]["sets"]["paper"]["gate"]
        rows = [["Selected rescue", fmt(g["selected_raw"]), fmt(g["selected_eq"]), P_GATE["sel"][0], P_GATE["sel"][1]],
                ["Matched control", fmt(g["control_raw"]), fmt(g["control_eq"]), P_GATE["ctrl"][0], P_GATE["ctrl"][1]],
                ["Specificity", fmt(g["spec_raw"]), fmt(g["spec_eq"]), P_GATE["spec"][0], P_GATE["spec"][1]],
                ["n cases", g["n"], g["n"], P_GATE["n"], P_GATE["n"]]]
    tables[9] = md_table(["Quantity", "Raw (ours)", "Equal-norm (ours)", "Raw (paper)", "Equal-norm (paper)"], rows, os.path.join(TABLES, "table_09"),
                         "Table 9: Qwen3 gate-weight-matched and equal-norm control (validation cases where the selected expert is clean-active)")
    # ---- Table 10 rank (qwen3)
    rows = []
    if "qwen3" in res and "rank" in res["qwen3"]["sets"]["paper"]:
        k = res["qwen3"]["sets"]["paper"]["rank"]
        rows = [["Ranked validation cases", k["n"], P_RANK["n"]], ["Top-1 among active experts", f"{k['top1']} / {k['n']}", f"{P_RANK['top1']} / {P_RANK['n']}"],
                ["Top-2 among active experts", f"{k['top2']} / {k['n']}", f"{P_RANK['top2']} / {P_RANK['n']}"], ["Mean rank", f"{k['mean_rank']:.2f}", P_RANK["mean_rank"]],
                ["Mean percentile", f"{k['mean_percentile']:.2f}", P_RANK["pct"]], ["Selected minus all-other active", fmt(k["sel_minus_all_other"]), P_RANK["diff"]]]
    tables[10] = md_table(["Metric", "Ours", "Paper"], rows, os.path.join(TABLES, "table_10"), "Table 10: Qwen3 selected-expert rank among all clean-active experts")
    # ---- Table 11 mixtral pair
    rows = []
    if "mixtral" in res and "pair" in res["mixtral"]["sets"]["paper"]:
        p = res["mixtral"]["sets"]["paper"]["pair"]
        rows = [["Raw active-pair specificity", p["n"], fmt(p["spec_raw"]), P_PAIR["n"], P_PAIR["spec_raw"]],
                ["Selected equal-norm rescue", p["n"], fmt(p["selected_eq"]), P_PAIR["n"], P_PAIR["sel_eq"]],
                ["Other active expert equal-norm rescue", p["n"], fmt(p["other_eq"]), P_PAIR["n"], P_PAIR["other_eq"]],
                ["Equal-norm active-pair specificity", p["n"], fmt(p["spec_eq"]), P_PAIR["n"], P_PAIR["spec_eq"]]]
    tables[11] = md_table(["Metric", "n (ours)", "Ours", "n (paper)", "Paper"], rows, os.path.join(TABLES, "table_11"), "Table 11: Mixtral active-pair equal-norm check (anchor-active validation cases)")
    # ---- Table 12 relation held-out (qwen3, mixtral extra) + stability grid text
    rows = []
    for mk, r in res.items():
        if "heldout" not in r:
            continue
        h = r["heldout"]
        e = r["sets"]["paper"]["selection"]["e_star"]
        L = r["sets"]["paper"]["layer"]["L_star"]
        pap = P_HELD if mk == "qwen3" else {"folds": "", "selected": "", "n": "", "active": "", "rescue": "", "spec": ""}
        rows += [[MODELS[mk]["label"], "Folds", len(h["folds"]), pap["folds"]], [MODELS[mk]["label"], f"Selected L{L}E{e:03d}", f"{h['n_selected_reference']} / {len(h['folds'])}", pap["selected"]],
                 [MODELS[mk]["label"], "Held-out cases", h["n_heldout"], pap["n"]], [MODELS[mk]["label"], "Active cases", f"{h['active']} / {h['n_heldout']}", pap["active"]],
                 [MODELS[mk]["label"], "Rescue", fmt(h["rescue"]), pap["rescue"]], [MODELS[mk]["label"], "Specificity", fmt(h["spec"]), pap["spec"]],
                 [MODELS[mk]["label"], "Per-fold selections", ", ".join(f"E{f['e_star']:03d}" for f in h["folds"]), ""]]
    tables[12] = md_table(["Model", "Metric", "Ours", "Paper"], rows, os.path.join(TABLES, "table_12"), "Table 12: relation-held-out expert-selection check (5 relation folds)")
    # stability grid (Appendix D text) -> table_12b
    rows = []
    for mk, r in res.items():
        if "stability" not in r:
            continue
        st = r["stability"]
        e = r["sets"]["paper"]["selection"]["e_star"]
        rows.append([MODELS[mk]["label"], f"E{e:03d} selected in {st['n_reference_selected']}/{st['n_settings']} settings", f3(st["mean_val_rescue"]), f3(st["mean_val_spec"]),
                     json.dumps({f"E{int(k):03d}": int(v) for k, v in st["selected_counts"].items()}),
                     P_STAB["selected"] if mk == "qwen3" else "", P_STAB["rescue"] if mk == "qwen3" else "", P_STAB["spec"] if mk == "qwen3" else ""])
        st["grid"].to_csv(os.path.join(TABLES, f"stability_grid_{mk}.csv"), index=False)
    tables["12b"] = md_table(["Model", "Selection stability (seeds 0-4 x thresholds 32,48,64,80,96)", "Mean val rescue", "Mean val spec", "Selected experts",
                             "Paper selected", "Paper rescue", "Paper spec"], rows, os.path.join(TABLES, "table_12b"), "Appendix D: expert-selection stability grid")
    # ---- Table 13 noise (qwen3)
    rows = []
    if "qwen3" in res and "noise" in res["qwen3"]:
        for _, x in res["qwen3"]["noise"].iterrows():
            p = P_NOISE.get(float(x.sigma), ("", "", "", ""))
            rows.append([f"{x.sigma:.1f}", f"{int(x.active)}/{int(x.n)}", f3(x["drop"]), f"{x.rescue:+.3f} [{x.ci_lo:+.3f}, {x.ci_hi:+.3f}]", f"{x.pos_frac:.2f}",
                         p[0], p[1], p[2], p[3]])
    tables[13] = md_table(["sigma", "Active", "Drop", "Rescue", "Pos.", "Active (paper)", "Drop (paper)", "Rescue (paper)", "Pos. (paper)"], rows,
                          os.path.join(TABLES, "table_13"), "Table 13: Qwen3 fixed-hypothesis noise-scale sensitivity for L44E069 (validation cases)")
    # ---- Table 14 relaxed
    rows = []
    for mk, r in res.items():
        if "relaxed" not in r["sets"]:
            continue
        s = r["sets"]["relaxed"]
        p = P[mk]["relaxed"]
        sel, ev = s.get("selection", {}), s.get("eval")
        rows.append([MODELS[mk]["label"], f"L{s['layer']['L_star']}", fmt(s["layer"]["val_at_Lstar"]), p[0],
                     f"L{s['layer']['L_star']}E{sel['e_star']:03d}" if sel.get("e_star") is not None else "n/a", P[mk]["expert"],
                     f"{sel.get('disc_active', 'n/a')}/{sel.get('n_disc', '')}", p[1], fmt(ev["rescue_all"]) if ev else "n/a", p[2], fmt(ev["spec_all"]) if ev else "n/a", p[3]])
    tables[14] = md_table(["Model", "Layer", "Layer rescue", "Paper", "Expert", "Paper expert", "Disc. active", "Paper", "Expert rescue", "Paper", "Spec", "Paper"], rows,
                          os.path.join(TABLES, "table_14"), "Table 14: relaxed-filter scale-up (512 cases: margin >= 0.5, drop >= 0.25; 256 discovery / 256 validation)")
    # ---- Table 15 relaxed split by overlap with the strict set
    rows = []
    for mk, r in res.items():
        if "relaxed" not in r["sets"] or "eval_per_case" not in r["sets"]["relaxed"]:
            continue
        s = r["sets"]["relaxed"]
        L = s["layer"]["L_star"]
        pc = s["eval_per_case"].set_index("case_id")
        R = r["R"]
        val = list(pc.index)
        sets = r["sets_raw"]
        for ref_name in ("paper", "strict"):
            ref = set(sets[ref_name]["discovery"] + sets[ref_name]["validation"])
            for sub_name, ids in (("All relaxed val.", val), (f"Relaxed-new val. (not in {ref_name} set)", [c for c in val if c not in ref])):
                lay = summarize(R.loc[ids, L].values, with_p=False)
                er = summarize(pc.loc[ids].rescue.values, with_p=False)
                sp = summarize(pc.loc[ids].spec.dropna().values, with_p=False)
                pp = P[mk]["relaxed_split"]["all" if sub_name.startswith("All") else "new"] if ref_name == "paper" else ("", "", "", "", "")
                rows.append([MODELS[mk]["label"], sub_name, len(ids), fmt(lay), len(ids), fmt(er), fmt(sp), pp[0], pp[1], pp[2], pp[3], pp[4]])
    tables[15] = md_table(["Model", "Subset", "n_L", "Layer", "n_E", "Expert", "Spec", "n_L (paper)", "Layer (paper)", "n_E (paper)", "Expert (paper)", "Spec (paper)"], rows,
                          os.path.join(TABLES, "table_15"), "Table 15: relaxed-filter validation split by overlap with the strict set")
    # ---- Table 16 coalitions
    rows = []
    for mk, r in res.items():
        s = r["sets"]["paper"]
        if "coalitions" not in s:
            continue
        co = s["coalitions"]
        L = s["layer"]["L_star"]
        for kind, label in (("coalition_clean", f"Clean top-{MODELS[mk]['n_controls'] and ''}k coalition"), ("coalition_union", "Routing-union coalition"), ("layer", f"L{L} MoE-block patch (same pass)")):
            p = P_COAL.get(kind, ("", "")) if mk == "mixtral" else ("", "")
            rows.append([MODELS[mk]["label"], label.replace("top-k", "top-8" if mk == "qwen3" else "top-2"), fmt(co[kind]), f"{co[kind]['pos_frac']:.2f}", p[0], p[1]])
    tables[16] = md_table(["Model", "Patch", "Rescue (ours)", "Pos. frac. (ours)", "Rescue (paper)", "Pos. frac. (paper)"], rows, os.path.join(TABLES, "table_16"),
                          "Table 16: multi-expert coalition patching at the selected layer (validation cases)")
    # ---- extra: per-set summary and layer curves CSV
    rows = []
    for mk, r in res.items():
        for s, v in r["sets"].items():
            la = v["layer"]
            sel = v.get("selection", {})
            ev = v.get("eval")
            rows.append([MODELS[mk]["label"], s, la["n_disc"], la["n_val"], f"L{la['L_star']}", fmt(la["val_at_Lstar"]),
                         f"E{sel['e_star']:03d}" if sel.get("e_star") is not None else "n/a", sel.get("n_candidates", ""),
                         fmt(ev["rescue_all"]) if ev else "", fmt(ev["spec_all"]) if ev else "", f"{ev['val_active']}/{ev['n_val']}" if ev else ""])
        curves = pd.DataFrame([dict(model=mk, set=s, **c) for s, v in r["sets"].items() for c in v["layer"]["curve"]])
        curves.to_csv(os.path.join(TABLES, f"layer_curves_{mk}.csv"), index=False)
    tables["sets"] = md_table(["Model", "Case set", "n disc", "n val", "L*", "Layer rescue (val)", "e*", "# candidates", "Expert rescue (val)", "Spec (val)", "Val active"], rows,
                              os.path.join(TABLES, "table_sets"), "All case sets: layer and expert selections")
    figure1(res)
    # summary json (drop DataFrames)
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if not isinstance(v, pd.DataFrame)}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    summ = {mk: clean({k: v for k, v in r.items() if k not in ("cases", "R")}) for mk, r in res.items()}
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=str)
    return {"res": res, "tables": tables}


def alt_summary(models=("qwen3", "mixtral"), suffix: str = "_alt", table_name: str = "table_alt_token_rule",
                title: str = "Secondary run: paper_like object-token rule (no-space token if single, else leading-space), paper case IDs") -> str:
    """Secondary run summary for results/<model><suffix> (paper case set only). Default: the paper_like object-token rule."""
    rows = []
    for mk in models:
        name = f"{mk}{suffix}"
        if not os.path.exists(os.path.join(RESULTS, name, "sweep_rows.parquet")):
            continue
        md = A.load_model(name)
        la = A.layer_analysis(md, "paper")
        disc, val = md.ids("paper", "discovery"), md.ids("paper", "validation")
        L = la["L_star"]
        row = [MODELS[mk]["label"], f"L{L}", fmt(la["val_at_Lstar"]), f"L{la['sharpness']['next_layer']} {la['sharpness']['next_rescue']:+.3f}"]
        if md.expert_rows is not None and L in set(md.expert_rows.layer):
            et = A.expert_table(md, L)
            sel = A.select_expert(et, disc, len(disc) // 2)
            if sel["e_star"] is not None:
                ev = A.evaluate_expert(md, L, sel["e_star"], val, MODELS[mk]["n_controls"])
                co = A.coalitions(md, L, val)
                row += [f"E{sel['e_star']:03d} ({sel['n_candidates']} cand.)", f"{sel['disc_active']}/{sel['n_disc']}", f"{ev['val_active']}/{ev['n_val']}",
                        fmt(ev["rescue_all"]), fmt(ev["spec_all"]), fmt(co["coalition_clean"]), fmt(co["coalition_union"])]
            else:
                row += ["no candidate"] + [""] * 6
        else:
            row += [""] * 7
        ct = md.cases
        row += [f"{int(ct.strict.sum())}/{len(ct)}"]
        rows.append(row)
        pd.DataFrame([dict(model=mk, **c) for c in la["curve"]]).to_csv(os.path.join(TABLES, f"layer_curves_{mk}{suffix}.csv"), index=False)
    if not rows:
        return ""
    return md_table(["Model", "L*", "Layer rescue (val)", "Next layer", "e*", "Disc. active", "Val. active", "Expert rescue (val)", "Spec (val)",
                     "Clean top-k coalition", "Union coalition", "Paper IDs passing strict"], rows, os.path.join(TABLES, table_name), title)


# ---------------------------------------------------------------------------------------------------------------
def figure1(res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"qwen3": "#2a78d6", "mixtral": "#eb6834"}
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    # (a) layer sweep validation curves
    ax = axes[0]
    for mk, r in res.items():
        cur = pd.DataFrame(r["sets"]["paper"]["layer"]["curve"])
        ax.plot(cur.layer, cur.val_mean, color=C[mk], lw=1.6, label=f"{MODELS[mk]['label']} (n={r['sets']['paper']['layer']['n_val']})")
        ax.fill_between(cur.layer, cur.ci_lo, cur.ci_hi, color=C[mk], alpha=0.15, lw=0)
        L = r["sets"]["paper"]["layer"]["L_star"]
        v = r["sets"]["paper"]["layer"]["val_at_Lstar"]["mean"]
        ax.plot([L], [v], "o", color=C[mk], ms=5)
        ax.annotate(f"L{L} {v:+.2f}", (L, v), textcoords="offset points", xytext=(-6, 6), ha="right", fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("Validation rescue (Delta patched - Delta noised)")
    ax.set_title("(a) MoE-block output patching across layers", fontsize=9, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    # (b) selected-expert specificity
    ax = axes[1]
    labels, x = [], 0
    for mk, r in res.items():
        s = r["sets"]["paper"]
        if "eval" not in s:
            continue
        ev = s["eval"]
        e = s["selection"]["e_star"]
        L = s["layer"]["L_star"]
        for key, lab in (("rescue_all", f"L{L}E{e:03d}"), ("control_all", "active-random"), ("spec_all", "Spec")):
            st = ev[key]
            ax.bar(x, st["mean"], width=0.7, color=C[mk], alpha=1.0 if key != "control_all" else 0.45, edgecolor="none")
            ax.errorbar(x, st["mean"], yerr=[[st["mean"] - st["ci_lo"]], [st["ci_hi"] - st["mean"]]], color=INK2, lw=1, capsize=2)
            labels.append((x, lab))
            x += 1
        x += 0.6
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([p for p, _ in labels])
    ax.set_xticklabels([l for _, l in labels], rotation=35, ha="right", fontsize=7.5)
    ax.set_title("(b) Selected expert vs active-random controls", fontsize=9, loc="left")
    ax.set_ylabel("Validation rescue")
    ax.grid(axis="y", color=GRID, lw=0.6)
    # (c) Mixtral coalitions vs layer rescue (Qwen3 as thin comparison if present)
    ax = axes[2]
    x = 0
    labels = []
    for mk in ("mixtral", "qwen3"):
        if mk not in res or "coalitions" not in res[mk]["sets"]["paper"]:
            continue
        s = res[mk]["sets"]["paper"]
        co = s["coalitions"]
        L = s["layer"]["L_star"]
        e = s["selection"]["e_star"]
        ev = s["eval"]
        items = [(ev["rescue_all"], f"E{e:03d}"), (co["coalition_clean"], "clean top-k"), (co["coalition_union"], "routing union")]
        xs = []
        for st, lab in items:
            ax.bar(x, st["mean"], width=0.7, color=C[mk], edgecolor="none")
            ax.errorbar(x, st["mean"], yerr=[[st["mean"] - st["ci_lo"]], [st["ci_hi"] - st["mean"]]], color=INK2, lw=1, capsize=2)
            labels.append((x, lab))
            xs.append(x)
            x += 1
        lr = co["layer_from_sweep"]["mean"]
        ax.plot([xs[0] - 0.45, xs[-1] + 0.45], [lr, lr], ls="--", color=C[mk], lw=1.2)
        ax.annotate(f"L{L} MoE-block patch {lr:+.2f}", (xs[0] - 0.45, lr), textcoords="offset points", xytext=(0, 4), fontsize=7.5, ha="left", va="bottom", color=INK)
        x += 0.6
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([p for p, _ in labels])
    ax.set_xticklabels([l for _, l in labels], rotation=35, ha="right", fontsize=7.5)
    ax.set_title("(c) Coalition patching vs MoE-block patch", fontsize=9, loc="left")
    ax.set_ylabel("Validation rescue")
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1.pdf"))
    fig.savefig(os.path.join(FIGS, "fig1.png"), dpi=200)
    plt.close(fig)
