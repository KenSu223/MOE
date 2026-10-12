"""ext13 Phase-5 behaviour tables (CPU, agent p5-data): probe outcome, association strata per condition, restated format
(F2R) behaviour, original-name memorisation check (mem), behaviour of the new scans (templates F1 / copy / natural copy),
condition sizes. Inputs: results/wino_ref_<proto>/{p5_probe_outcome.json, scan_pairs_*.parquet}, data/wino_ref/
{pairs_p5_*, case_sets_p5_*, p5_conditions.json}.
Outputs results/tables/ext13base_{probe,probe_strata,f2r_behaviour,mem_behaviour,scan_behaviour,conditions,cp_respectively,cp_options}.{md,csv},
results/ext13base_behaviour.json
Usage: python scripts/ext13_p5_behaviour.py
"""
import json, os, re, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import RESULTS

D = "/home/ubuntu/MOE/data/wino_ref"
TAB = os.path.join(RESULTS, "tables")
PROTOS = ("qwen3", "mixtral_bos")
LABEL = {"qwen3": "Qwen3-30B-A3B-Base", "mixtral_bos": "Mixtral-8x7B, BOS"}
CONDS = ["f1", "dir_f1", "dir_opt", "cp_nat", "cp_vt_orig", "cp_vt_neu", "vt_e2a_orig", "vt_e2a_neu", "vt_a2e_orig", "vt_a2e_neu", "f2e"]


def md_table(header, rows, name, caption=None):
    lines = ([f"**{caption}**", ""] if caption else []) + ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    txt = "\n".join(lines) + "\n"
    open(os.path.join(TAB, name + ".md"), "w").write(txt)
    pd.DataFrame(rows, columns=header).to_csv(os.path.join(TAB, name + ".csv"), index=False)
    return txt


def boot_mean(x, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 3
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def cluster_mean(x, units, n=5000, seed=0):
    """mean of x with a cluster bootstrap over units."""
    x, units = np.asarray(x, float), np.asarray(units)
    u, inv = np.unique(units, return_inverse=True)
    idx = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    cnt = np.zeros((n, len(u)))
    np.add.at(cnt, (np.repeat(np.arange(n), len(u)), idx.ravel()), 1.0)
    W = cnt[:, inv]
    b = (W @ x) / W.sum(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def f3(x, d=2):
    return f"{x[0]:.{d}f} [{x[1]:.{d}f}, {x[2]:.{d}f}]"


def first_mentioned(sent, a, b):
    ia = re.search(r"\b" + re.escape(a) + r"\b", sent)
    ib = re.search(r"\b" + re.escape(b) + r"\b", sent)
    if not ia or not ib:
        return None
    return a if ia.start() < ib.start() else b


def main():
    out = {}
    # ---- probe outcome
    rows = []
    for p in PROTOS:
        f = f"{RESULTS}/wino_ref_{p}/p5_probe_outcome.json"
        if not os.path.exists(f):
            continue
        o = json.load(open(f))
        out.setdefault(p, {})["probe"] = o
        for v in ("v1", "v2"):
            bf = o["frac_ge1_by_format"][v]
            rows.append([LABEL[p], v, o["n_margin_pairs_unique_twins"], f"{o['frac_ge1'][v]:.3f}", f"{o['frac_ge1_names'][v]:.3f}" if o["frac_ge1_names"][v] is not None else "",
                         f"{o['frac_ge1_objects'][v]:.3f}" if o["frac_ge1_objects"][v] is not None else "",
                         " / ".join(f"{bf[k]:.3f}" if bf[k] is not None else "–" for k in ("F1", "F2E", "F2R")), f"{o['sd'][v]:.2f}",
                         "yes" if v in o["responding"] else "no", f"{o['templates_frac_ge1'][v]:.3f}"])
    md_table(["Model", "Probe", "Margin pairs (unique twins)", "Share |probe| ≥ 1", "Names", "Objects", "F1 / F2E / F2R", "SD", "Responds (≥ 0.10)",
              "Templates (random names): share |probe| ≥ 1"], rows, "ext13base_probe",
             "Association probes v1 / v2 on the margin pairs of the valid attribute → entity formats (trigger probes of both twins; pre-registered rule: a probe responds if ≥ 10 % reach |probe| ≥ 1)")
    rows = []
    for p in PROTOS:
        o = out.get(p, {}).get("probe")
        if o:
            rows.append([LABEL[p], f"{o['r_v1_v2']:.2f}", ", ".join(o["responding"]) or "none", ", ".join(o["R"]) + (" (flag: none responds)" if o["flag_none_responds"] else ""),
                         " / ".join(str(o["margin_strata"].get(k, 0)) for k in ("associative", "one-sided", "reverse", "neutral", "no_probe")),
                         " / ".join(str(o["margin_strata_names"].get(k, 0)) for k in ("associative", "one-sided", "reverse", "neutral")),
                         " / ".join(str(o["margin_strata_objects"].get(k, 0)) for k in ("associative", "one-sided", "reverse", "neutral")), o["margin_conflicts"]])
    md_table(["Model", "r(v1, v2)", "Responding", "R (used)", "Margin pairs: assoc. / one-sided / reverse / neutral / no probe", "Names", "Objects", "Conflicts (reverse & own)"],
             rows, "ext13base_probe_strata_overall", "Probe outcome and association strata of the margin pairs (F1 / F2E / F2R pooled, unique twins)")
    # ---- strata per condition (all pairs of the condition's pool and of its case sets)
    reg = json.load(open(f"{D}/p5_conditions.json"))["conditions"] if os.path.exists(f"{D}/p5_conditions.json") else {}
    rows = []
    for c in CONDS:
        for p in PROTOS:
            e = reg.get(c, {}).get(p)
            if not e or e.get("rebuild_pending") or not os.path.exists(os.path.join("/home/ubuntu/MOE", e["case_sets"])):
                continue
            cs = json.load(open(os.path.join("/home/ubuntu/MOE", e["case_sets"])))
            for sp, st in cs["strata"].items():
                rows.append([c, LABEL[p], sp, st["pairs"], st["names"], st["objects"], st.get("assoc_associative", 0), st.get("assoc_one-sided", 0),
                             st.get("assoc_reverse", 0), st.get("assoc_neutral", 0), st.get("assoc_no_probe", 0), st.get("local_assoc", ""), st.get("seeds", ""),
                             "behaviour only" if e["behaviour_only"] else ""])
    md_table(["Condition", "Model", "Split", "Pairs", "Names", "Objects", "Associative", "One-sided", "Reverse", "Neutral", "No probe", "Local assoc", "Seeds", "Flag"],
             rows, "ext13base_conditions", "Phase-5 conditions: split sizes and strata (data/wino_ref/case_sets_p5_<cond>_<proto>.json)")
    # ---- F2R behaviour (restated format)
    rows = []
    for p in PROTOS:
        f = f"{RESULTS}/wino_ref_{p}/scan_pairs_F2R.parquet"
        if not os.path.exists(f):
            continue
        s = pd.read_parquet(f)
        s = s[s.main]
        fm = [first_mentioned(x, a, b) for x, a, b in zip(s.sent_a, s.ans_a, s.ans_b)]
        s = s.assign(first=fm)
        s = s[s.first.notna()]
        pred_a = np.where(s.d_clean_a > 0, s.ans_a, s.ans_b)
        pred_b = np.where(s.d_clean_b > 0, s.ans_a, s.ans_b)
        s = s.assign(first_pref=((pred_a == s["first"]).astype(float) + (pred_b == s["first"]).astype(float)) / 2,
                     correct_one=((s.d_clean_a > 0).astype(float) + (s.d_clean_b < 0).astype(float)) / 2,
                     signed_margin=(s.d_clean_a - s.d_clean_b) / 2)
        res = {}
        for nm, g in (("names", s[s.names]), ("objects", s[~s.names]), ("all", s)):
            if len(g) == 0:
                continue
            res[nm] = {"pairs": int(len(g)), "correct_prompt": boot_mean(g.correct_one), "correct_both": boot_mean(g.correct_both.astype(float)),
                       "margin": boot_mean(g.margin.astype(float)), "mean_signed_margin": boot_mean(g.signed_margin), "first_pref": boot_mean(g.first_pref)}
            x = res[nm]
            rows.append([LABEL[p], nm, x["pairs"], f3(x["correct_prompt"]), f3(x["correct_both"]), f3(x["margin"]), f3(x["mean_signed_margin"]), f3(x["first_pref"])])
        out.setdefault(p, {})["f2r"] = res
    md_table(["Model", "Items", "Pairs (content swaps)", "Correct per prompt", "Correct in both twins", "Margin ≥ 1 both ways", "Mean Δ toward the answer (logits)",
              "First-mentioned candidate predicted"], rows, "ext13base_f2r_behaviour",
             "Restated format (F2R; behaviour only): whole sentence with the blank → 'one of them', the predicate restated, entity readout (95% bootstrap CIs over pairs)")
    # ---- mem: original names vs IOI names (same seeds; seed-clustered CIs)
    rows = []
    for p in PROTOS:
        fm, fv = f"{RESULTS}/wino_ref_{p}/scan_pairs_p5_mem.parquet", f"{RESULTS}/wino_ref_{p}/scan_pairs_verbs_ent2attr.parquet"
        if not (os.path.exists(fm) and os.path.exists(fv)):
            continue
        m, v = pd.read_parquet(fm), pd.read_parquet(fv)
        res = {}
        for var in ("orig", "neutral", "unrelated"):
            mm = m[m.variant == var]
            vv = v[(v.variant == var) & v.seed.isin(set(mm.seed))]
            ent = {}
            for tag, g in (("original names", mm), ("IOI names (same seeds)", vv)):
                ent[tag] = {"pairs": int(len(g)), "seeds": int(g.seed.nunique()), "margin": cluster_mean(g.margin.astype(float), g.seed),
                            "correct_both": cluster_mean(g.correct_both.astype(float), g.seed), "drop": cluster_mean(g.drop_str, g.seed)}
                x = ent[tag]
                rows.append([LABEL[p], var, tag, x["pairs"], x["seeds"], f3(x["margin"]), f3(x["correct_both"]), f3(x["drop"])])
            res[var] = ent
        out.setdefault(p, {})["mem"] = res
    md_table(["Model", "Verb", "Names", "Pairs", "Seeds", "Margin ≥ 1 both ways", "Correct in both twins", "Mean drop (logits)"], rows, "ext13base_mem_behaviour",
             "Memorisation check (task 7): the 55 entity → attribute seeds with their ORIGINAL WinoGrande names vs the IOI-name fills of the same seeds, "
             "orig / neutral / unrelated verb (Phase-3 format; seed-clustered bootstrap 95% CIs)")
    # ---- copy prefixes: void 'respectively' prefix by congruence (finding) and the options prefix by order
    from moetrace.ext7_wino import load_twins
    opt1 = {t.pair_id: t.opt1 for t in load_twins("train_xl")[0]}
    rows = []
    for p in PROTOS:
        for cond in ("cp_vt", "cp_nat"):
            f = f"{RESULTS}/wino_ref_{p}/scan_pairs_p5_{cond}_respectively.parquet"
            if not os.path.exists(f):
                continue
            s_ = pd.read_parquet(f)
            x = s_.name_x if cond == "cp_vt" else s_.base_pair_id.map(opt1)
            s_["congruent"] = np.where(s_.order == 1, s_.ans_a == x, s_.ans_a != x)
            unit = s_.seed if cond == "cp_vt" else s_.base_pair_id
            grp = ["variant", "congruent"] if cond == "cp_vt" else ["congruent"]
            for k, g in s_.groupby(grp):
                k = k if isinstance(k, tuple) else (k,)
                lab = (f"{k[0]}, " if cond == "cp_vt" else "") + ("congruent (X ↔ t1)" if k[-1] else "incongruent")
                u = unit.loc[g.index]
                rows.append([LABEL[p], cond, lab, len(g), f3(cluster_mean(g.margin.astype(float), u)), f3(cluster_mean(g.correct_both.astype(float), u)),
                             f3(cluster_mean(g.drop_str, u))])
    md_table(["Model", "Build", "Subset", "Pairs", "Margin ≥ 1 both ways", "Correct in both twins", "Mean drop"], rows, "ext13base_cp_respectively",
             "VOID copy prefix 'Of X and Y, one <frame> t1 and the other <frame> t2.' (behaviour record): twin A's answer is option 1 (X) in every item, so the "
             "order with t1 = A's attribute is 'respectively'-congruent (X ↔ t1 answers both twins without the sentence); cluster bootstrap over seeds / twins")
    rows = []
    for p in PROTOS:
        for cond in ("cp_vt", "cp_nat"):
            f = f"{RESULTS}/wino_ref_{p}/scan_pairs_p5_{cond}.parquet"
            if not os.path.exists(f):
                continue
            s_ = pd.read_parquet(f)
            if "prefix_style" in s_ and not (s_.prefix_style == "options").all():
                continue
            unit = s_.seed if cond == "cp_vt" else s_.base_pair_id
            item = (s_.seed + "|" + s_.name_j.astype(str)) if cond == "cp_vt" else s_.base_pair_id
            s_ = s_.assign(item=item.values)
            both = s_.groupby(([s_.variant] if cond == "cp_vt" else []) + [s_.item]).margin.all()
            for k, g in (s_.groupby(["variant", "order"]) if cond == "cp_vt" else s_.groupby("order")):
                k = k if isinstance(k, tuple) else (k,)
                u = unit.loc[g.index]
                rows.append([LABEL[p], cond, (f"{k[0]}, " if cond == "cp_vt" else "") + f"order {k[-1]}", len(g), f3(cluster_mean(g.margin.astype(float), u)),
                             f3(cluster_mean(g.correct_both.astype(float), u)), f3(cluster_mean(g.drop_str, u)), ""])
            if cond == "cp_vt":
                for v in ("orig", "neutral", "unrelated"):
                    b = both.loc[v]
                    rows.append([LABEL[p], cond, f"{v}, both orders pass", len(b), f"{b.mean():.3f} of items", "", "", int(b.sum())])
            else:
                rows.append([LABEL[p], cond, "both orders pass", len(both), f"{both.mean():.3f} of twins", "", "", int(both.sum())])
    md_table(["Model", "Build", "Subset", "Pairs / items", "Margin ≥ 1 both ways", "Correct in both twins", "Mean drop", "Items passing both orders"], rows,
             "ext13base_cp_options", "Copy prefix 'Options: t1, t2.' + newline (current; coordinator decision 2026-10-11): pass rates per attribute order (cluster bootstrap over seeds / twins)")
    # ---- scan behaviour of the new builds
    rows = []
    for p in PROTOS:
        for cond, grp in (("vt_a2e", "variant"), ("cp_vt", "variant"), ("cp_nat", "order"), ("verbs_ent2attr", "variant")):
            f = f"{RESULTS}/wino_ref_{p}/scan_pairs_{'p5_' if cond != 'verbs_ent2attr' else ''}{cond}.parquet"
            if not os.path.exists(f):
                continue
            s = pd.read_parquet(f)
            unit = s.seed if "seed" in s else s.base_pair_id
            for k, g in s.groupby(grp):
                u = unit.loc[g.index]
                rows.append([LABEL[p], cond, f"{grp} {k}", len(g), int(u.nunique()), f3(cluster_mean(g.margin.astype(float), u)),
                             f3(cluster_mean(g.correct_both.astype(float), u)), f3(cluster_mean(g.drop_str, u))])
    md_table(["Model", "Build", "Subset", "Pairs", "Units (seeds / twins)", "Margin ≥ 1 both ways", "Correct in both twins", "Mean drop"], rows, "ext13base_scan_behaviour",
             "Competence scans of the Phase-5 builds (cluster bootstrap over seeds / Phase-3 twins); verbs_ent2attr = the existing IOI-name entity → attribute scan")
    json.dump(out, open(os.path.join(RESULTS, "ext13base_behaviour.json"), "w"), indent=1, default=float)
    print("behaviour tables written")


if __name__ == "__main__":
    main()
