"""ext13 Phase-5 base analysis per condition and model (CPU, agent p5-data), on results/p5_<cond>_<proto>.

Reuses the Phase-3 analysis functions of scripts/ext7_wino_analyze.py per run (as scripts/ext12_analyze.py part 4c):
W2 layer curves / attention share (AUC+), W4 all-MoE + direct split, W4 DLA, W6 two-stage expert selection with Spec /
equal norm / replication / joint search / fixed experts, W3 grid, W5 heads; strata = names / objects and the association
strata of the pre-registered probe rule (+ the Phase-3 local-context assoc flag where defined). Additions:
  - cluster bootstrap of the headline numbers with the condition's independent unit (templates: seed; cp_nat: Phase-3 twin
    (both attribute orders); otherwise: pair), 5,000 resamples, seed 0;
  - grid by site (F1 formats: STR trigger, blank, candidate mentions after the STR site, answer cue, final);
  - copy conditions: final-position attention of the top heads on the prefix attributes (in-context answer vs the other)
    and the mentions;
  - selected_expert (main two-stage selection) and pop_top10 (top 10 (layer, expert) by discovery all-case single-expert
    rescue, ext8's population ranking) written to data/wino_ref/p5_conditions.json; status.base = true; PROGRESS line
    "[p5-data] BASE READY <cond>/<proto>".
Outputs results/p5_<cond>_<proto>/base_summary.json, results/ext13base_summary.json (all analysed runs),
results/tables/ext13base_<cond>_*.{md,csv}, results/figures/ext13base_<cond>_*.{png,pdf}.
Usage: python scripts/ext13_p5_analyze.py <cond> <proto> [--no-ready] [--tables-only]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, pandas as pd
from moetrace import ext7_pairs as P
from moetrace.models import MODELS, RESULTS

ROOT = "/home/ubuntu/MOE"
TAB, FIG = os.path.join(RESULTS, "tables"), os.path.join(RESULTS, "figures")
REG = os.path.join(ROOT, "data/wino_ref/p5_conditions.json")
SUMM = os.path.join(RESULTS, "ext13base_summary.json")
LABEL = {"qwen3": "Qwen3-30B-A3B-Base", "mixtral_bos": "Mixtral-8x7B, BOS"}
FIXED = {"qwen3": [(41, 117), (44, 69), (42, 115)], "mixtral": [(20, 0), (19, 2), (21, 1), (18, 1), (19, 6)]}
NB = 5000
F = P.fmt


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jl(x):
    return json.loads(x) if isinstance(x, str) else list(x)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def md_table(header, rows, name, caption=None):
    lines = ([f"**{caption}**", ""] if caption else []) + ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(TAB, name + ".md"), "w") as f:
        f.write(txt)
    pd.DataFrame(rows, columns=header).to_csv(os.path.join(TAB, name + ".csv"), index=False)
    return txt


# ---------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------
BOOL_STRATA = [("names", "person-name options"), ("s_neutral", "probe-neutral (both triggers |probe| < 1)"),
               ("s_associative", "probe-associative"), ("s_onesided", "probe one-sided"), ("s_reverse", "probe reverse"),
               ("local_assoc", "Phase-3 local-context assoc"), ("congruent", "copy prefix 'respectively'-congruent order"), ("debiased", "AfLite survivor (train_debiased)"), ("top1_both", "top-1 in both directions")]


RUN_OVERRIDE = {}


def load(cond, proto, ent):
    cfg = {"run": RUN_OVERRIDE.get("run", f"p5_{cond}_{proto}"), "model": ent["model"], "label": LABEL[proto], "cf_experts": FIXED[ent["model"]]}
    run = P.PairRun(cfg["run"])
    pairs = pd.read_parquet(os.path.join(ROOT, ent["pair_file"]))
    pi = pairs.set_index("pair_idx")
    c = run.cases
    for col in ("stratum_assoc", "stratum_local_assoc", "seed", "variant", "order", "top1_both", "debiased", "base_pair_id", "phase3_pair_idx", "congruent"):
        if col in pi.columns:
            c[col] = c.pair_idx.map(pi[col])
    sa = c.get("stratum_assoc", pd.Series("no_probe", index=c.index)).fillna("no_probe")
    c["s_neutral"], c["s_associative"] = sa == "neutral", sa == "associative"
    c["s_onesided"], c["s_reverse"] = sa == "one-sided", sa == "reverse"
    if "stratum_local_assoc" in c:
        c["local_assoc"] = c.stratum_local_assoc.astype("boolean").fillna(False).astype(bool)
    for col in ("top1_both", "debiased", "names", "congruent"):
        if col in c:
            c[col] = c[col].astype("boolean").fillna(False).astype(bool)
    # columns the Phase-3 functions read
    c["assoc"] = c.s_associative
    if "one_token_option" not in c:
        c["one_token_option"] = np.nan
    if "trigger_in_context" not in c:
        c["trigger_in_context"] = False
    # unit of the cluster bootstrap
    if "seed" in c and c.seed.notna().all():
        c["unit"], unit_name = c.seed, "seed"
    elif "phase3_pair_idx" in c and c.phase3_pair_idx.notna().all():
        c["unit"], unit_name = c.phase3_pair_idx.astype(int), "Phase-3 twin"
    else:
        c["unit"], unit_name = c.pair_idx, "pair"
    return cfg, run, pairs, unit_name


def strata_for(run):
    out = []
    for col, lab in BOOL_STRATA:
        if col not in run.cases:
            continue
        v = run.cases[col].astype(bool)
        if v.sum() >= 20 or (~v).sum() >= 20:
            out.append((col, lab))
    return out


def descriptors(run, unit_name):
    out = {}
    for f, sp in run.fam.items():
        ids = [x for v in sp.values() for x in v]
        c = run.cases.loc[ids]
        e = {"n_pairs": len(ids) // 2, "n_directed": len(ids), "n_units": int(c.unit.nunique()), "unit": unit_name,
             "delta_clean": float(c.delta_clean.mean()), "delta_corrupt": float(c.delta_corrupt.mean()), "drop": P.summ(c["drop"], with_p=False),
             "clean_top1_rate": float((c.top1_clean == c.true_id).mean()), "corrupt_top1_foil_rate": float((c.top1_corrupt == c.foil_id).mean()),
             "names": float(c.names.mean()), "T_median": float(c["T"].median())}
        for s in ("neutral", "associative", "onesided", "reverse"):
            e[f"assoc_{s}"] = float(c[f"s_{s}"].mean())
        if "local_assoc" in c:
            e["local_assoc"] = float(c.local_assoc.mean())
        if "seed" in c:
            e["seeds"] = int(c.seed.nunique())
        out[f] = e
    return out


# ---------------------------------------------------------------------------------------------------------------
# cluster bootstrap of the headline numbers
# ---------------------------------------------------------------------------------------------------------------
def unit_weights(units, n_boot=NB, seed=0):
    """units: array of unit labels per PAIR -> weight matrix [n_boot + 1, n_pairs] (row 0 = point estimate)."""
    u, inv = np.unique(units, return_inverse=True)
    idx = np.random.default_rng(seed).integers(0, len(u), size=(n_boot, len(u)))
    cnt = np.zeros((n_boot, len(u)))
    np.add.at(cnt, (np.repeat(np.arange(n_boot), len(u)), idx.ravel()), 1.0)
    W = np.vstack([np.ones(len(units)), cnt[:, inv]])
    return W, len(u)


def ci(x):
    return [float(x[0]), float(np.percentile(x[1:], 2.5)), float(np.percentile(x[1:], 97.5))]


def cluster_stats(run, ids, cfg):
    pairs = sorted({c // 2 for c in ids})
    units = run.cases.groupby("pair_idx").unit.first().loc[pairs].to_numpy()
    W, nu = unit_weights(units)
    ws = W.sum(1)

    def pm(series):
        return P.pair_means(series).reindex(pairs).to_numpy(dtype=np.float64)
    drop = pm(run.drop.loc[ids])
    dd = W @ drop / ws
    out = {"n_pairs": len(pairs), "n_units": nu}
    A = P.matrix_pairs(run.mat("attn_layer", ids)).reindex(pairs).to_numpy(np.float64)
    M = P.matrix_pairs(run.mat("layer", ids)).reindex(pairs).to_numpy(np.float64)
    a = np.clip(W @ A / ws[:, None], 0, None).sum(1)
    m = np.clip(W @ M / ws[:, None], 0, None).sum(1)
    out["share"] = ci(a / (a + m))
    out["auc_attn_norm"], out["auc_moe_norm"] = ci(a / dd), ci(m / dd)
    pk = {}
    for k in ("layer", "attn_layer", "block"):
        if k in run.kinds:
            Kd = P.matrix_pairs(run.mat(k, ids)).reindex(pairs)
            cur = Kd.mean(axis=0) / drop.mean()
            l = int(cur.idxmax())
            pk[k] = {"layer": l, "norm": ci((W @ Kd[l].to_numpy(np.float64)) / ws / dd)}
    out["peaks_val_argmax"] = pk
    dp = os.path.join(run.dir, "direct_split.parquet")
    if os.path.exists(dp):
        x = pd.read_parquet(dp).set_index("case_id")
        x = x.loc[[c for c in ids if c in x.index]]
        if len(x) == len(ids):
            den = pm(x.d_clean_fp32 - x.d_corrupt_fp32)
            for name, s in (("A_direct", x.d_attn_direct - x.d_corrupt_fp32), ("M_direct", x.d_moe_direct - x.d_corrupt_fp32)):
                out[name] = ci((W @ pm(s)) / (W @ den))
    jp = os.path.join(run.dir, "joint_rows.parquet")
    if os.path.exists(jp):
        j = pd.read_parquet(jp)
        j = j[(j.span == "all") & (j.direction == "denoise") & (j.kind == "all_moe")].set_index("case_id")
        if set(ids) <= set(j.index):
            out["M_all_moe"] = ci((W @ pm(j.effect.loc[ids])) / (W @ pm(j["drop"].loc[ids])))
    return out


# ---------------------------------------------------------------------------------------------------------------
# grid by site, copy-head attention
# ---------------------------------------------------------------------------------------------------------------
def site_labels(run, pairs, cond, tok):
    """directed case -> {pos: site} for positions >= first STR position."""
    pi = pairs.set_index("pair_idx")
    out = {}
    for c, r in run.cases.iterrows():
        p = pi.loc[int(r.pair_idx)]
        T = int(r["T"])
        str_pos = set(jl(r.str_pos))
        lab = {}
        blank = set(jl(p["blank_pos"])) if "blank_pos" in pi.columns and isinstance(p.get("blank_pos"), str) else set()
        ment = set(jl(p["mention_a_pos"])) | set(jl(p["mention_b_pos"])) if "mention_a_pos" in pi.columns else set()
        cue = set()
        if tok is not None and "\nAnswer:" in p["prompt_a"]:
            j = p["prompt_a"].rindex("\nAnswer:")
            enc = tok(p["prompt_a"], return_offsets_mapping=True, add_special_tokens=True)
            cue = {i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > j and b > a}
        first = min(str_pos)
        for q in range(first, T):
            if q == T - 1:
                s = "final"
            elif q in str_pos:
                s = "STR"
            elif q in blank:
                s = "blank"
            elif q in ment:
                s = "mention"
            elif q in cue:
                s = "answer cue"
            else:
                s = "other"
            lab[q] = s
        out[int(c)] = lab
    return out


SITES = ("STR", "blank", "mention", "other", "answer cue", "final")


def grid_sites(run, pairs, cond, tok):
    p = os.path.join(run.dir, "str_grid_w1_rows.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    lab = site_labels(run, pairs, cond, tok)
    df["site"] = [lab[int(c)].get(int(q), "other") for c, q in zip(df.case_id, df.pos)]
    L = int(df.layer.max()) + 1
    drop = run.drop
    out = {}
    for k, g in df.groupby("kind"):
        cc = g.groupby(["case_id", "site", "layer"], as_index=False).rescue.mean()  # mean over positions of the site
        kk = {}
        for s in SITES:
            sub = cc[cc.site == s]
            if sub.empty:
                continue
            Mx = sub.pivot(index="case_id", columns="layer", values="rescue").reindex(columns=range(L))
            nc = P.norm_curve(Mx, drop)
            l = int(nc.norm.idxmax())
            kk[s] = {"n_dir": int(len(Mx)), "n_pairs": int(len({c // 2 for c in Mx.index})), "curve_norm": nc.norm.round(5).tolist(),
                     "peak_layer": l, "peak": [float(nc.norm[l]), float(nc.norm_lo[l]), float(nc.norm_hi[l])],
                     "positions_per_case": float(g[g.site == s].groupby("case_id").pos.nunique().mean())}
        out[k] = kk
    return out


def copy_attention(run, pairs, hr_summary, cfg):
    """cp_*: clean / corrupted final-position attention mass of the top heads on the prefix attribute that is the clean
    answer (attr_true), the other prefix attribute (attr_other), the prefix names, the sentence mentions and the STR site."""
    p = os.path.join(run.dir, "head_attn_final.npz")
    if not os.path.exists(p) or "attr_a_pos" not in pairs.columns:
        return None
    hr = P.HeadRun(cfg["run"])
    pi = pairs.set_index("pair_idx")
    val = run.ids("main", "validation")
    pos = {c: i for i, c in enumerate(hr.attn_ids)}
    n = len(hr.attn_ids)
    classes = ("attr_true", "attr_other", "prefix_names", "ment_filled", "ment_other", "str", "final")

    def masks(c):
        r = run.cases.loc[c]
        q = pi.loc[int(r.pair_idx)]
        aa, ab = jl(q["attr_a_pos"]), jl(q["attr_b_pos"])
        at, ao = (aa, ab) if int(r.d) == 0 else (ab, aa)
        hp = hr.pos.loc[c]
        return {"attr_true": at, "attr_other": ao, "prefix_names": jl(q["prefix_ment_a_pos"]) + jl(q["prefix_ment_b_pos"]),
                "ment_filled": jl(hp["ment_filled"]), "ment_other": jl(hp["ment_other"]), "str": jl(hp["str_pos"]), "final": [int(hp["T"]) - 1]}
    tab = hr_summary["table"] if hr_summary else None
    out = {"layers": hr.layers, "top_heads": [], "layer_mean": []}
    mk = {c: masks(c) for c in val}
    for l in hr.layers:
        Al = hr.attn[l].astype(np.float32)
        res_c = {k: [] for k in classes}
        res_k = {k: [] for k in classes}
        for c in val:
            i = pos[c]
            for k in classes:
                res_c[k].append(Al[i][:, mk[c][k]].sum(-1) if mk[c][k] else np.zeros(Al.shape[1]))
                res_k[k].append(Al[n + i][:, mk[c][k]].sum(-1) if mk[c][k] else np.zeros(Al.shape[1]))
        mc = {k: np.mean(np.stack(v), 0) for k, v in res_c.items()}
        mn = {k: np.mean(np.stack(v), 0) for k, v in res_k.items()}
        out["layer_mean"].append({"layer": l, **{f"clean_{k}": float(mc[k].mean()) for k in classes}})
        if tab is not None:
            for _, r in tab[(tab.layer == l)].sort_values("val_mean", ascending=False).head(3).iterrows():
                h = int(r["head"])
                out["top_heads"].append({"layer": l, "head": h, "val_mean": float(r.val_mean),
                                         **{f"clean_{k}": float(mc[k][h]) for k in classes}, **{f"corrupt_{k}": float(mn[k][h]) for k in classes}})
    out["top_heads"] = sorted(out["top_heads"], key=lambda x: -x["val_mean"])[:8]
    return out


# ---------------------------------------------------------------------------------------------------------------
def pop_top10(cfg, run):
    p = os.path.join(run.dir, "str_expert_rows.parquet")
    if not os.path.exists(p):
        return None
    e = pd.read_parquet(p, columns=["case_id", "layer", "kind", "expert", "rescue", "clean_active"])
    disc = run.ids("main", "discovery")
    s = e[(e.kind == "expert") & e.clean_active & e.case_id.isin(disc)]
    pop = (s.groupby(["layer", "expert"]).rescue.sum() / len(disc)).sort_values(ascending=False)
    return [[int(l), int(x), round(float(v), 4)] for (l, x), v in pop.head(10).items()]


def analyse(cond, proto, ent):
    import ext7_wino_analyze as W
    cfg, run, pairs, unit_name = load(cond, proto, ent)
    W.STRATA = strata_for(run)
    meta = run.meta
    s = {"cond": cond, "proto": proto, "label": LABEL[proto], "L": run.L, "unit": unit_name, "strata_used": W.STRATA,
         "descriptors": descriptors(run, unit_name)}
    log(cond, proto, "W2")
    s["w2"] = W.w2(run, cfg)
    log("cluster bootstrap")
    s["cluster"] = {"main_validation": cluster_stats(run, run.ids("main", "validation"), cfg),
                    "main_all": cluster_stats(run, run.ids("main", "discovery") + run.ids("main", "validation"), cfg)}
    if "rep" in run.fam:
        s["cluster"]["rep_validation"] = cluster_stats(run, run.ids("rep", "validation"), cfg)
    for nm in ("names", "s_neutral", "congruent"):
        if nm in run.cases:
            ids_all = run.ids("main", "discovery") + run.ids("main", "validation")
            for v in (True, False):
                ids = [c for c in ids_all if bool(run.cases.at[c, nm]) == v]
                if len({c // 2 for c in ids}) >= 16:
                    s["cluster"][f"{nm}_{v}"] = cluster_stats(run, ids, cfg)
    if meta.get("expert", {}).get("complete"):
        log("W6")
        s["w6"] = W.w6(run, cfg, s["w2"])
        s["pop_top10"] = pop_top10(cfg, run)
    if meta.get("grid_w1", {}).get("complete"):
        log("W3")
        s["w3"] = W.grid(run, cfg)
        tok = None
        if "\nAnswer:" in str(pairs.prompt_a.iloc[0]):
            from transformers import AutoTokenizer
            from moetrace.arch import snapshot_dir
            tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[cfg["model"]]["repo"]))
        s["w3_sites"] = grid_sites(run, pairs, cond, tok)
    if meta.get("heads", {}).get("complete"):
        log("W5")
        s["w5"] = W.heads(run, cfg)
        if cond.startswith("cp_"):
            s["w5_copy"] = copy_attention(run, pairs, s["w5"], cfg)
    log("W4")
    s["w4"] = W.joint(run, cfg)
    s["w4_dla"] = W.dla(run, cfg)
    return s


def to_json(s):
    s2 = dict(s)
    s2["w2"] = dict(s["w2"])
    s2["w2"]["curves"] = {k: v.to_dict("records") for k, v in s["w2"]["curves"].items()}
    s2["w2"]["additivity"] = {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in s["w2"]["additivity"].items()}
    if s.get("w5"):
        s2["w5"] = dict(s["w5"])
        s2["w5"]["table"] = s["w5"]["table"].head(40).to_dict("records")
        s2["w5"]["table_bottom"] = s["w5"]["table"].tail(5).to_dict("records")
    return clean(s2)


def selected(s):
    w6 = (s.get("w6") or {}).get("main", {}).get("two_stage")
    if not w6:
        return None
    ev = w6.get("eval")
    e = w6["selection"].get("e_star")
    if e is None:
        return {"layer": int(w6["layer"]), "expert": None, "pattern": w6.get("pattern")}
    out = {"layer": int(w6["layer"]), "expert": int(e), "name": f"L{int(w6['layer'])}E{int(e):03d}", "pattern": w6.get("pattern"),
           "spec": [ev["spec"]["mean"], ev["spec"]["ci_lo"], ev["spec"]["ci_hi"]], "rescue": [ev["rescue"]["mean"], ev["rescue"]["ci_lo"], ev["rescue"]["ci_hi"]],
           "spec_norm": list(ev["spec_norm"]), "val_active": ev["val_active"]}
    rep = (s.get("w6") or {}).get("rep", {}).get("two_stage")
    if rep and rep.get("eval"):
        out["rep"] = {"layer": int(rep["layer"]), "expert": int(rep["selection"]["e_star"]), "spec": [rep["eval"]["spec"]["mean"], rep["eval"]["spec"]["ci_lo"], rep["eval"]["spec"]["ci_hi"]]}
    return out


# ---------------------------------------------------------------------------------------------------------------
# tables and figures per condition (all analysed protos)
# ---------------------------------------------------------------------------------------------------------------
def fr(x, d=3):
    return f"{x[0]:.{d}f} [{x[1]:.{d}f}, {x[2]:.{d}f}]" if x else "n/a"


def tables(cond, allsum):
    runs = [(p, allsum[f"{cond}/{p}"]) for p in ("qwen3", "mixtral_bos") if f"{cond}/{p}" in allsum]
    if not runs:
        return
    rows = []
    for p, s in runs:
        for f, d in s["descriptors"].items():
            rows.append([s["label"], f, d["n_pairs"], f"{d['n_units']} {d['unit']}s", f"{d['delta_clean']:+.2f}", F(d["drop"], 2), f"{d['clean_top1_rate']:.2f}",
                         f"{d['names']:.2f}", f"{d['assoc_neutral']:.2f} / {d['assoc_associative']:.2f} / {d['assoc_onesided']:.2f} / {d['assoc_reverse']:.2f}",
                         f"{d.get('local_assoc', float('nan')):.2f}", f"{d['T_median']:.0f}"])
    md_table(["Model", "Family", "Pairs", "Units", "Mean Δ clean", "Drop [95% CI]", "Clean top-1", "Names", "Probe strata neutral / assoc. / one-sided / reverse",
              "Local assoc", "Median T"], rows, f"ext13base_{cond}_descriptors", f"{cond}: pair sets (directed cases pooled; pair means)")
    rows = []
    for p, s in runs:
        for key, lab in (("main_validation", "main, validation"), ("main_all", "main, all"), ("rep_validation", "replication, validation"),
                         ("names_True", "names (main, all)"), ("names_False", "objects (main, all)"), ("s_neutral_True", "probe-neutral (main, all)"),
                         ("s_neutral_False", "not neutral (main, all)"), ("congruent_True", "copy order congruent (main, all)"),
                         ("congruent_False", "copy order incongruent (main, all)")):
            x = s["cluster"].get(key)
            if not x:
                continue
            pk = x["peaks_val_argmax"]
            rows.append([s["label"], lab, x["n_pairs"], x["n_units"], fr(x["share"], 2), fr(x["auc_attn_norm"]), fr(x["auc_moe_norm"]),
                         fr(x.get("A_direct")), fr(x.get("M_direct")), fr(x.get("M_all_moe")),
                         f"L{pk['layer']['layer']} {pk['layer']['norm'][0]:.3f}" if "layer" in pk else "", f"L{pk['attn_layer']['layer']} {pk['attn_layer']['norm'][0]:.3f}" if "attn_layer" in pk else ""])
    md_table(["Model", "Set", "Pairs", f"Units", "Attention share (AUC+)", "AUC+ attention / drop", "AUC+ MoE / drop", "A_direct / drop", "M_direct / drop",
              "All-MoE M / drop", "MoE peak (layer, / drop)", "Attention peak"], rows, f"ext13base_{cond}_headline",
             f"{cond}: headline numbers with cluster-bootstrap 95% CIs (unit = {runs[0][1]['unit']}; 5,000 resamples)")
    # W2 peaks (Phase-3 format)
    rows = []
    for p, s in runs:
        for f in ("main", "rep", "main_interior"):
            for x in s["w2"]["peaks"].get(f, []):
                rows.append([s["label"], f, x["kind"], f"L{x['L_disc']}", f"{x['disc_mean']:+.2f}", F(x["val_at_L_disc"]), P.fmt_r(x["norm_at_L_disc"]),
                             ", ".join(f"L{l} {v:+.2f}" for l, v in x["disc_top5"][:4])])
    md_table(["Model", "Family", "Patched output", "L* (disc.)", "Disc. mean", "Validation rescue at L* [95% CI]", "Normalised", "Discovery top 4"], rows,
             f"ext13base_{cond}_w2_peaks", f"{cond} W2: final-position single-layer patches (pair bootstrap)")
    rows = []
    for p, s in runs:
        for x in s["w2"]["strata"]:
            if "share" not in x:
                rows.append([s["label"], x["label"], x["value"], x["n_pairs"], "", "", "", ""])
                continue
            rows.append([s["label"], x["label"], x["value"], x["n_pairs"], f"{x['drop']:.2f}", P.fmt_r(x["norm_layer"]), P.fmt_r(x["norm_attn_layer"]),
                         f"{x['share']['share']:.2f} [{x['share']['share_lo']:.2f}, {x['share']['share_hi']:.2f}]"])
    md_table(["Model", "Stratum", "Value", "Pairs", "Mean drop", "MoE at L*_MoE / drop", "Attention at L*_attn / drop", "Attention share"], rows,
             f"ext13base_{cond}_w2_strata", f"{cond} W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)")
    # W6
    rows, rows_f, rows_j = [], [], []
    for p, s in runs:
        w6 = s.get("w6")
        if not w6:
            continue
        for f in ("main", "rep"):
            if f not in w6:
                continue
            for tag in ("two_stage", "interior"):
                r = w6[f][tag]
                sel, ev = r["selection"], r.get("eval")
                rows.append([s["label"], f, tag, f"L{r['layer']}", F(r["layer_val"]), P.fmt_r(r["layer_norm"]),
                             f"L{r['layer']}E{sel['e_star']:03d}" if sel["e_star"] is not None else "none",
                             f"{sel.get('disc_active', sel.get('max_activity'))} (≥ {sel['threshold']})", F(ev["rescue"]) if ev else "", F(ev["spec"]) if ev else "",
                             F(r["coalitions"]["coalition_clean"]), r["pattern"],
                             F(r["equal_norm"]["spec_eq"]) if isinstance(r.get("equal_norm"), dict) and "spec_eq" in r["equal_norm"] else ""])
            for x in w6[f]["joint_top"][:5]:
                rows_j.append([s["label"], f, x["rank"], f"L{x['layer']}E{x['expert']:03d}", x["disc_active"], f"{x['disc_allcase_mean']:+.3f}", F(x["val_rescue"]), F(x["val_spec"])])
        for x in w6["main"].get("cf_experts", []):
            rows_f.append([s["label"], f"L{x['layer']}E{x['expert']:03d}", x["disc_active"], x["val_active"], F(x["rescue"]), F(x["spec"]), P.fmt_r(x["rescue_norm"])])
        if s.get("pop_top10"):
            rows_f.append([s["label"], "pop top-10: " + ", ".join(f"L{l}E{e:03d}" for l, e, _ in s["pop_top10"]), "", "", "", "", ""])
    md_table(["Model", "Family", "Rule", "Layer", "Layer rescue (val)", "Layer / drop", "Selected expert", "Disc. active", "Val rescue", "Spec", "Clean top-k coalition",
              "Pattern", "Equal-norm Spec"], rows, f"ext13base_{cond}_w6_experts", f"{cond} W6: two-stage expert selection (recurrence gate half of the discovery directed cases)")
    md_table(["Model", "Family", "Rank", "(layer, expert)", "Disc. active", "Disc. all-case", "Val rescue", "Spec"], rows_j, f"ext13base_{cond}_w6_joint", f"{cond} W6: joint (layer, expert) search")
    md_table(["Model", "Fixed expert", "Disc. active", "Val active", "Val rescue", "Spec", "Rescue / drop"], rows_f, f"ext13base_{cond}_w6_fixed",
             f"{cond} W6: Phase-3 WinoGrande / CounterFact experts as fixed hypotheses (validation), and the population top-10")
    # W4 / DLA
    rows = []
    for p, s in runs:
        w4 = s.get("w4") or {}
        for key in ("main_validation", "main_all", "rep_validation"):
            x = w4.get(key)
            if not x:
                continue
            g = lambda k, sub=None: (P.fmt_r(x.get(k)) if sub is None else P.fmt_r((x.get(k) or {}).get(sub))) if (x.get(k) if sub is None else (x.get(k) or {}).get(sub)) else "n/a"
            rows.append([s["label"], key, x["n_pairs"], g("M_denoise"), g("M_noise"), g("direct", "A_direct"), g("direct", "M_direct"), g("dla", "attn"), g("dla", "moe")])
    md_table(["Model", "Set", "Pairs", "All-MoE M denoise", "All-MoE M noise", "A_direct", "M_direct", "DLA share attention", "DLA share MoE"], rows,
             f"ext13base_{cond}_w4", f"{cond} W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)")
    rows = []
    for p, s in runs:
        x = (s.get("w4_dla") or {}).get("main")
        if x:
            rows.append([s["label"], P.fmt_r(x["attn_total"]), P.fmt_r(x["moe_total"]), P.fmt_r(x["attn_share_of_direct"], 2),
                         ", ".join(f"L{l} {v:+.3f}" for l, v in x["attn_top5"][:3]), ", ".join(f"L{l} {v:+.3f}" for l, v in x["moe_top5"][:3])])
    md_table(["Model", "DLA attention / drop", "DLA MoE / drop", "Attention share of DLA", "Top attention layers", "Top MoE layers"], rows, f"ext13base_{cond}_dla",
             f"{cond} W4 DLA: direct logit attribution of the final-position sublayer outputs (main family)")
    # W5
    rows, rows_c = [], []
    for p, s in runs:
        w5 = s.get("w5")
        if not w5:
            continue
        for r in w5["table"][:6] + w5.get("table_bottom", [])[-2:]:
            rows.append([s["label"], f"L{int(r['layer'])}H{int(r['head'])}", f"{r['val_mean']:+.3f} [{r['val_lo']:+.3f}, {r['val_hi']:+.3f}]", f"{r['disc_mean']:+.3f}",
                         f"{r['z_val']:+.1f} / {r['z_disc']:+.1f}", "yes" if r["detected_2sd"] else "no", f"{r['share_attn_layer']:.2f}"])
        for r in (s.get("w5_copy") or {}).get("top_heads", []):
            rows_c.append([s["label"], f"L{r['layer']}H{r['head']}", f"{r['val_mean']:+.3f}"] +
                          [f"{r['clean_' + k]:.2f} → {r['corrupt_' + k]:.2f}" for k in ("attr_true", "attr_other", "prefix_names", "ment_filled", "ment_other", "str")])
    md_table(["Model", "Head", "Val rescue [95% CI]", "Disc. mean", "z (val / disc)", "≥ 2 SD both", "Share of attn_layer"], rows, f"ext13base_{cond}_w5_heads",
             f"{cond} W5: top six and bottom two heads (final position; layers = top-4 attention layers + one null layer)")
    if rows_c:
        md_table(["Model", "Head", "Val rescue", "attr (answer) clean → corrupt", "attr (other)", "prefix names", "mention filled", "mention other", "STR site"], rows_c,
                 f"ext13base_{cond}_w5_copy", f"{cond} W5: final-position attention mass of the top heads on the copy prefix (validation; clean → corrupted run)")
    # W3
    rows = []
    for p, s in runs:
        g = s.get("w3_sites") or {}
        for k, kk in g.items():
            for site in SITES:
                if site in kk:
                    x = kk[site]
                    rows.append([s["label"], k, site, x["n_pairs"], f"{x['positions_per_case']:.1f}", f"L{x['peak_layer']}", f"{x['peak'][0]:+.3f} [{x['peak'][1]:+.3f}, {x['peak'][2]:+.3f}]"])
    md_table(["Model", "Patched quantity", "Site", "Pairs", "Positions / case", "Peak layer", "Peak rescue / drop [95% CI]"], rows, f"ext13base_{cond}_w3_sites",
             f"{cond} W3: position × layer grid by site (main family, window 1; site mean over its positions)")
    figures(cond, runs)


def figures(cond, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"layer": "#d95f02", "attn_layer": "#1b9e77", "block": "#7570b3", "resid": "#555555"}
    lab = {"layer": "MoE output", "attn_layer": "attention output", "block": "block", "resid": "residual"}
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 3.6), squeeze=False)
    for ax, (p, s) in zip(axes[0], runs):
        cur = pd.DataFrame(s["w2"]["curves"]["main_val"])
        for k, g in cur.groupby("kind"):
            ax.plot(g.layer, g.norm, color=cols.get(k), label=lab.get(k, k), lw=1.6)
            ax.fill_between(g.layer, g.norm_lo, g.norm_hi, color=cols.get(k), alpha=0.18)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("layer"); ax.set_ylabel("rescue / mean drop")
        ax.set_title(f"{s['label']}: {cond}, final position (validation, {int(cur.n.iloc[0])} pairs)", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext13base_{cond}_w2_curves.{ext}"), dpi=150)
    plt.close(fig)
    for p, s in runs:
        g = s.get("w3_sites")
        if not g:
            continue
        kinds = [k for k in ("layer", "attn_layer", "resid") if k in g]
        fig, axes = plt.subplots(1, len(kinds), figsize=(3.4 * len(kinds), 4.8), squeeze=False)
        for ax, k in zip(axes[0], kinds):
            sites = [x for x in SITES if x in g[k]]
            H = np.array([g[k][x]["curve_norm"] for x in sites]).T
            vmax = np.nanmax(np.abs(H)) or 1.0
            im = ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest", origin="lower")
            ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel("layer"); ax.set_title(lab[k], fontsize=8.5)
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        fig.suptitle(f"{s['label']}: {cond}, rescue / mean drop by layer and site", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"ext13base_{cond}_w3_grid_{p}.{ext}"), dpi=150)
        plt.close(fig)
    have = [(p, s) for p, s in runs if s.get("w5")]
    if have:
        fig, axes = plt.subplots(1, len(have), figsize=(7 * len(have), 3.2), squeeze=False)
        for ax, (p, s) in zip(axes[0], have):
            t = pd.DataFrame(s["w5"]["table"] + s["w5"].get("table_bottom", []))
            # full table is not stored; heat map of the stored rows only would be sparse -> bar of the top 40
            t = t.drop_duplicates(["layer", "head"]).sort_values("val_mean", ascending=False)
            ax.bar(range(len(t)), t.val_mean, color="#1b9e77")
            ax.set_xticks(range(len(t))); ax.set_xticklabels([f"L{int(a)}H{int(b)}" for a, b in zip(t.layer, t["head"])], rotation=90, fontsize=5)
            ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("validation rescue (logits)")
            ax.set_title(f"{s['label']}: {cond}, per-head attention patches (top 40 + bottom)", fontsize=8.5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"ext13base_{cond}_w5_heads.{ext}"), dpi=150)
        plt.close(fig)
    have = [(p, s) for p, s in runs if (s.get("w4_dla") or {}).get("main")]
    if have:
        fig, axes = plt.subplots(1, len(have), figsize=(6 * len(have), 3.2), squeeze=False)
        for ax, (p, s) in zip(axes[0], have):
            x = s["w4_dla"]["main"]
            ax.bar(np.arange(len(x["moe_curve"])) - 0.2, x["moe_curve"], width=0.4, color="#d95f02", label="MoE output")
            ax.bar(np.arange(len(x["attn_curve"])) + 0.2, x["attn_curve"], width=0.4, color="#1b9e77", label="attention output")
            ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("layer"); ax.set_ylabel("DLA difference / mean drop")
            ax.set_title(f"{s['label']}: {cond}, DLA of the final-position sublayer outputs", fontsize=8.5)
            ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"ext13base_{cond}_w4_dla.{ext}"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond")
    ap.add_argument("proto")
    ap.add_argument("--no-ready", action="store_true")
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--test-run", default=None, help="analyse another run dir with this condition's pair file; writes only to the scratch dir")
    a = ap.parse_args()
    if a.test_run:
        global TAB, FIG, SUMM, REG
        REG = os.environ.get("P5_TEST_REG", REG)
        RUN_OVERRIDE["run"] = a.test_run
        scratch = os.environ.get("P5_SCRATCH", "/tmp/p5_test")
        TAB, FIG, SUMM = (os.path.join(scratch, "tables"), os.path.join(scratch, "figures"), os.path.join(scratch, "summary.json"))
        os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
        ent = json.load(open(REG))["conditions"][a.cond][a.proto]
        s = analyse(a.cond, a.proto, ent)
        js = to_json(s)
        json.dump({f"{a.cond}/{a.proto}": js}, open(SUMM, "w"), indent=1, default=str)
        log("selected", selected(js), "pop", js.get("pop_top10"))
        tables(a.cond, json.load(open(SUMM)))
        return
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    reg = json.load(open(REG))
    ent = reg["conditions"][a.cond][a.proto]
    allsum = json.load(open(SUMM)) if os.path.exists(SUMM) else {}
    if not a.tables_only:
        s = analyse(a.cond, a.proto, ent)
        js = to_json(s)
        with open(os.path.join(RESULTS, f"p5_{a.cond}_{a.proto}", "base_summary.json"), "w") as f:
            json.dump(js, f, indent=1, default=str)
        allsum = json.load(open(SUMM)) if os.path.exists(SUMM) else {}
        allsum[f"{a.cond}/{a.proto}"] = js
        with open(SUMM, "w") as f:
            json.dump(allsum, f, indent=1, default=str)
        reg = json.load(open(REG))  # re-read (other conditions may have been written meanwhile)
        e = reg["conditions"][a.cond][a.proto]
        e["selected_expert"] = selected(js)
        e["pop_top10"] = js.get("pop_top10")
        e["status"]["base"] = bool(js.get("w6")) and bool(js.get("w4"))
        e["base_summary"] = f"results/p5_{a.cond}_{a.proto}/base_summary.json"
        with open(REG, "w") as f:
            json.dump(reg, f, indent=1, default=int)
        log("registry updated", json.dumps({k: e[k] for k in ("selected_expert", "pop_top10", "status")}))
        if e["status"]["base"] and not a.no_ready:
            with open(os.path.join(ROOT, "logs/PROGRESS.md"), "a") as f:
                f.write(f"- {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())} [p5-data] BASE READY {a.cond}/{a.proto}\n")
    tables(a.cond, json.load(open(SUMM)))
    log("tables / figures written for", a.cond)
    if not a.test_run:  # refresh the behaviour tables and the section draft (cheap; failures do not matter here)
        import subprocess
        for sc in ("ext13_p5_behaviour.py", "ext13_p5_text.py"):
            r = subprocess.run([sys.executable, "-W", "ignore", os.path.join(ROOT, "scripts", sc)], capture_output=True, text=True)
            log(sc, "rc", r.returncode, (r.stderr or "")[-300:])


if __name__ == "__main__":
    main()
