"""ext13 post-hoc analyses (coordinator, 2026-10-13/14; CPU only, existing Phase-3 / Phase-5 rows).

Answers follow-up questions after the Phase-5 synthesis:
  A  position x layer grid by site: attention vs MoE at the swapped word, the tokens after it and the final token; residual
     hand-off layer (last layer at which the swapped-word residual still restores >= 0.5 of the drop, first layer at which the
     final-token residual does); validation pairs, single-position single-layer patches (not additive across positions).
  B  name-site attention: the same at the swapped name of the verb templates, and names vs objects in the Phase-3 option swap
     and the fill-in-the-blank pool.
  C  fill-in-the-blank (attribute -> entity) competence by item type and pool; behaviour on the name templates (same name in both
     twins, first-mentioned-name preference).
  D  template stability: pass rate per template over its 48 name pairs; share of the drop variance between templates vs names.
  E  margin descriptives: the 1-logit rule is logit(answer) - logit(twin's answer) (log-odds of the two candidates), not a
     top-1 rule; top-1 rates and margin quantiles of the main sets.

Usage: python scripts/ext13_posthoc.py   -> results/tables/ext13_posthoc_{sites,namesite,f1_competence,templates,margins}.md
"""
import json
import os

import numpy as np
import pandas as pd

R = "/home/ubuntu/MOE/results"
D = "/home/ubuntu/MOE/data"
PROTOS = ("qwen3", "mixtral_bos")
LABEL = {"qwen3": "Qwen3", "mixtral_bos": "Mixtral BOS"}


def md(df: pd.DataFrame, path: str, title: str):
    with open(path, "w") as f:
        f.write(f"**{title}**\n\n")
        f.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for _, r in df.iterrows():
            f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
    print("written", path)


def grid(run):
    f = f"{R}/{run}/str_grid_w1_rows.parquet"
    if not os.path.exists(f):
        return None
    r = pd.read_parquet(f, columns=["case_id", "split", "cat", "kind", "layer", "rescue", "delta_clean", "delta_corrupt"])
    v = r[r.split.astype(str).str.contains("validation")]
    return v if len(v) else r


def drop_of(x):
    return (x.groupby("case_id").delta_clean.first() - x.groupby("case_id").delta_corrupt.first()).mean()


def curves(x, drop):
    return x.groupby(["kind", "layer"]).rescue.mean().unstack(0) / drop


def part_a():
    runs = [("Phase-3 option swap (objects mostly)", "wino_{p}_str"), ("same twins, ask attribute (dir_opt)", "p5_dir_opt_{p}"),
            ("same twins, ask entity (dir_f1)", "p5_dir_f1_{p}"), ("ask entity, large pool (f1)", "p5_f1_{p}"),
            ("ask attribute + Options prefix (cp_nat)", "p5_cp_nat_{p}"), ("name templates, ask attribute (vt_e2a_orig)", "p5_vt_e2a_orig_{p}"),
            ("name templates, ask entity (vt_a2e_orig)", "p5_vt_a2e_orig_{p}"), ("natural sentence, ask entity (f2e)", "p5_f2e_{p}")]
    rows = []
    for p in PROTOS:
        for lab, t in runs:
            r = grid(t.format(p=p))
            if r is None:
                continue
            drop = drop_of(r)
            site = {}
            for cat in ("last STR token", "last token"):
                g = curves(r[r.cat == cat], drop)
                site[cat] = g
            s, f = site["last STR token"], site["last token"]
            hand_s = s.resid[s.resid >= 0.5].index.max()
            hand_f = f.resid[f.resid >= 0.5].index.min()
            mid = r[~r.cat.isin(["last STR token", "last token", "first STR token", "middle STR tokens", "between STR tokens"])]
            gm = curves(mid, drop) if len(mid) else None
            rows.append({"Model": LABEL[p], "Condition": lab, "Pairs": r.case_id.nunique() // 2,
                         "Swapped word: attn peak": f"L{s.attn_layer.idxmax()} {s.attn_layer.max():.2f}",
                         "Swapped word: MoE peak": f"L{s.layer.idxmax()} {s.layer.max():.2f}",
                         "Tokens in between: max peak": f"{max(gm.attn_layer.max(), gm.layer.max()):.2f}" if gm is not None else "–",
                         "Final: attn peak": f"L{f.attn_layer.idxmax()} {f.attn_layer.max():.2f}",
                         "Final: MoE peak": f"L{f.layer.idxmax()} {f.layer.max():.2f}",
                         "Hand-off (word ≥0.5 until → final ≥0.5 from)": f"L{hand_s} → L{hand_f}"})
    md(pd.DataFrame(rows), f"{R}/tables/ext13_posthoc_sites.md",
       "Position x layer grid by site (validation; single-position, single-layer patch rescue / drop; not additive across positions)")


def part_b():
    rows = []
    for p in PROTOS:
        for c in ("vt_e2a_orig", "vt_e2a_neu", "cp_vt_orig", "cp_vt_neu"):
            r = grid(f"p5_{c}_{p}")
            if r is None:
                continue
            drop = drop_of(r)
            g = curves(r[r.cat == "last STR token"], drop)
            f = curves(r[r.cat == "last token"], drop)
            rows.append({"Model": LABEL[p], "Items": f"name templates ({c})", "Pairs": r.case_id.nunique() // 2,
                         "Name site: attn peak": f"L{g.attn_layer.idxmax()} {g.attn_layer.max():.2f}",
                         "Name site: attn L4–L9 sum": f"{g.attn_layer.loc[4:9].clip(lower=0).sum():.2f}",
                         "Name site: MoE peak": f"L{g.layer.idxmax()} {g.layer.max():.2f}",
                         "Final: attn peak": f"{f.attn_layer.max():.2f}"})
        for run, pf, cs, lab in ((f"wino_{p}_str", f"{D}/wino_str/pairs_train_xl_{p}.parquet", f"{D}/wino_str/case_sets.json", "Phase-3 option swap"),):
            r = grid(run)
            c = json.load(open(cs))
            inv = {v: k for k, v in c["pair_idx_of"].items()}
            names = pd.read_parquet(pf, columns=["pair_id", "names"]).set_index("pair_id").names
            r = r.assign(names=r.case_id.map(lambda k: bool(names.get(inv.get(k // 2), False))))
            for nm, x in r.groupby("names"):
                drop = drop_of(x)
                g = curves(x[x.cat == "last STR token"], drop)
                f = curves(x[x.cat == "last token"], drop)
                rows.append({"Model": LABEL[p], "Items": f"{lab}, {'names' if nm else 'objects'}", "Pairs": x.case_id.nunique() // 2,
                             "Name site: attn peak": f"L{g.attn_layer.idxmax()} {g.attn_layer.max():.2f}",
                             "Name site: attn L4–L9 sum": f"{g.attn_layer.loc[4:9].clip(lower=0).sum():.2f}",
                             "Name site: MoE peak": f"L{g.layer.idxmax()} {g.layer.max():.2f}",
                             "Final: attn peak": f"{f.attn_layer.max():.2f}"})
    md(pd.DataFrame(rows), f"{R}/tables/ext13_posthoc_namesite.md",
       "Attention at the swapped name (entity -> attribute; 'site' = the swapped name or noun; validation)")


def part_c():
    p3 = json.load(open(f"{D}/wino_str/case_sets.json"))
    inv = {v: k for k, v in p3["pair_idx_of"].items()}
    p3ids = {inv[i] for v in p3["pairs"].values() for i in v}
    rows = []
    for p in PROTOS:
        f = pd.read_parquet(f"{R}/wino_ref_{p}/scan_pairs_F1.parquet")
        f = f[f.main]
        sets = [("all twins · objects", f[~f.names]), ("all twins · names", f[f.names]),
                ("Phase-3 776 · objects", f[f.pair_id.isin(p3ids) & ~f.names]), ("Phase-3 776 · names", f[f.pair_id.isin(p3ids) & f.names])]
        v = pd.read_parquet(f"{R}/wino_ref_{p}/scan_pairs_p5_vt_a2e.parquet")
        e = pd.read_parquet(f"{R}/wino_ref_{p}/scan_pairs_verbs_ent2attr.parquet")
        sets += [(f"name templates, ask entity · {k}", x) for k, x in v.groupby("variant")]
        sets += [(f"name templates, ask attribute · {k}", x) for k, x in e.groupby("variant")]
        for lab, x in sets:
            row = {"Model": LABEL[p], "Items": lab, "Pairs": len(x), "Correct in both twins": f"{x.correct_both.mean():.2f}",
                   "Pass margin": f"{x.margin.mean():.2f}", "Mean drop (logits)": f"{x.drop_str.mean():.2f}", "Same name in both twins": "",
                   "Prefers first-mentioned name": ""}
            if lab.startswith("name templates, ask entity") and "d_clean_a" in x:
                fa = x.apply(lambda r: r.sent_a.find(r.ans_a) < r.sent_a.find(r.ans_b), axis=1).to_numpy()
                pa = np.where(fa, x.d_clean_a > 0, x.d_clean_a < 0)
                pb = np.where(fa, x.d_clean_b > 0, x.d_clean_b < 0)
                row["Same name in both twins"] = f"{((x.d_clean_a > 0) == (x.d_clean_b > 0)).mean():.2f}"
                row["Prefers first-mentioned name"] = f"{np.concatenate([pa, pb]).mean():.2f}"
            rows.append(row)
    md(pd.DataFrame(rows), f"{R}/tables/ext13_posthoc_f1_competence.md",
       "Competence by question direction, item type and pool (chance for 'correct in both twins' = 0.25 if the trigger is ignored and each twin is guessed independently, 0 if the same name is always chosen)")


def part_d():
    rows = []
    for p in PROTOS:
        e = pd.read_parquet(f"{R}/wino_ref_{p}/scan_pairs_verbs_ent2attr.parquet")
        e = e[e.variant == "orig"]
        g = e.groupby("seed").margin.mean()
        within = e.groupby("seed").drop_str.var().mean() / e.drop_str.var()
        rows.append({"Model": LABEL[p], "Templates": g.size, "Name pairs per template": int(e.groupby("seed").size().median()),
                     "Templates passing ≤20 % of names": int((g <= 0.2).sum()), "20–80 %": int(((g > 0.2) & (g <= 0.8)).sum()),
                     ">80 %": int((g > 0.8).sum()), "Drop variance between templates": f"{1 - within:.2f}", "within (names)": f"{within:.2f}"})
    md(pd.DataFrame(rows), f"{R}/tables/ext13_posthoc_templates.md",
       "Verb templates (entity -> attribute, original verb): stability across the 48 name pairs of a template")


def part_e():
    rows = []
    for p in PROTOS:
        s = pd.read_parquet(f"{R}/wino_ref_{p}/scan_pairs_F1.parquet")
        cs = json.load(open(f"{D}/wino_ref/case_sets_p5_f1_{p}.json"))
        inv = {v: k for k, v in cs["pair_idx_of"].items()}
        x = s[s.pair_id.isin({inv[i] for i in cs["pairs"]["validation"]})]
        d = pd.concat([x.d_clean_a, -x.d_clean_b])
        rows.append({"Model": LABEL[p], "Set": "ask entity (f1), validation", "Pairs": len(x), "Answer is top-1 in both twins": f"{x.top1_both.mean():.2f}",
                     "Clean margin median (logits)": f"{d.median():.2f}", "10th percentile": f"{d.quantile(0.1):.2f}"})
    md(pd.DataFrame(rows), f"{R}/tables/ext13_posthoc_margins.md",
       "Margin rule = logit(answer) - logit(twin's answer) >= 1 in both twins (1 logit = probability ratio e ≈ 2.7, 73 % in a two-way choice); top-1 is recorded, not required")


if __name__ == "__main__":
    part_a(); part_b(); part_c(); part_d(); part_e()
