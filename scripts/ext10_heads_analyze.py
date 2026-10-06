"""ext10 step 1 analysis: single-head rankings, Z8 detection, head DLA vs single-patch agreement, layer additivity,
heads vs experts (single components), consistency with W5 / F2 heads and with ext8's direct-path split.

Normalisation as ext8: rescue = Delta_patched - Delta_corrupt (same pass); population values are ratios of means over
validation cases (CounterFact: donor means per case; WinoGrande: directed cases) with a percentile bootstrap over units
(CounterFact cases, WinoGrande pairs). Z8 (Zhang & Nanda Sec. 3): a head is detected when its population effect is >= 2 SD
from the mean over all heads of the model; reported on all validation units and on two disjoint halves of the units.

Writes results/tables/ext10_heads_*.md|csv, results/figures/ext10_heads_*.png, key "step1" of
results/ext10_circuit_summary.json.

Usage: python scripts/ext10_heads_analyze.py [--runs cf_qwen3,...] [--suffix _smoke]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from moetrace.models import RESULTS
from moetrace import ext8_addback as X
from moetrace import ext10_circuit as C

TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
SUMMARY = os.path.join(RESULTS, "ext10_circuit_summary.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def save(df, name):
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    with open(os.path.join(TAB, f"{name}.md"), "w") as f:
        f.write(md_table(df))
    log(f"wrote {name} ({len(df)} rows)")


def hname(l, h):
    return f"L{int(l)}H{int(h)}"


def rc_dict(v):
    return {"ratio": float(v.ratio), "lo": float(v.lo), "hi": float(v.hi)}


def row_spearman(A, B):
    ra = rankdata(A, axis=1)
    rb = rankdata(B, axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def topk_overlap(A, B, k):
    ta = np.argsort(-A, axis=1)[:, :k]
    tb = np.argsort(-B, axis=1)[:, :k]
    return np.array([len(set(a) & set(b)) / k for a, b in zip(ta, tb)])


def med_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {"median": float(np.median(x)), "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)), "n": int(len(x))}


# ------------------------------------------------------------------------------------------------------------------
def analyse(key, suffix=""):
    c = C.cfg(key)
    od = os.path.join(RESULTS, c["out"] + suffix)
    t = C.load_task(key)
    hd = C._cat(od, "head_rows_p")
    hdla = C._cat(od, "head_dla_p")
    edla = C._cat(od, "expert_dla_p")
    pf = C._cat(od, "head_prefill_p")
    L = int(hd.layer.max()) + 1
    nH = int(hd["head"].max()) + 1
    NC = L * nH
    rows = np.sort(hd.row_id.unique())
    ri = {int(r): i for i, r in enumerate(rows)}
    R = len(rows)
    H = hd[hd.kind == "attn_head"]
    S = np.full((R, NC), np.nan, dtype=np.float64)
    S[H.row_id.map(ri).values, (H.layer.values.astype(int) * nH + H["head"].values.astype(int))] = H.rescue.values
    Dm = np.full((R, NC), np.nan, dtype=np.float64)
    Dm[hdla.row_id.map(ri).values, (hdla.layer.values.astype(int) * nH + hdla["head"].values.astype(int))] = hdla.dla.values
    assert not np.isnan(S).any() and not np.isnan(Dm).any()
    Al = hd[hd.kind == "attn_layer"].pivot(index="row_id", columns="layer", values="rescue").loc[rows].values  # [R, L]
    rinfo = hd.drop_duplicates("row_id").set_index("row_id").loc[rows]
    drop_row = (rinfo.delta_clean - rinfo.delta_corrupt).values
    case_row = rinfo.case_id.values.astype(int)
    first = t.rows.set_index("row_id")["first"].loc[rows].values
    cases = t.cases.set_index("case_id")
    unit = cases.unit
    ucases = list(dict.fromkeys(case_row.tolist()))
    # donor-mean case matrices
    cidx = pd.Series(np.arange(R)).groupby(case_row).apply(list)
    def case_mean(M):
        return np.stack([M[cidx[cc]].mean(0) for cc in ucases])
    Sc, Dc, Alc = case_mean(S), case_mean(Dm), case_mean(Al)
    drop_c = pd.Series(np.array([drop_row[cidx[cc]].mean() for cc in ucases]), index=ucases)
    cols = [hname(l, h) for l in range(L) for h in range(nH)]
    pop = X.ratio_cols(pd.DataFrame(Sc, index=ucases, columns=cols), drop_c, unit)
    popd = X.ratio_cols(pd.DataFrame(Dc, index=ucases, columns=cols), drop_c, unit)
    m = pop.ratio.values
    z = (m - m.mean()) / m.std()
    # halves of the units
    us = np.array(sorted(set(unit.loc[ucases])))
    half = {u: i % 2 for i, u in enumerate(us)}
    hz = []
    for hb in (0, 1):
        sel = [j for j, cc in enumerate(ucases) if half[unit.loc[cc]] == hb]
        mh = Sc[sel].sum(0) / drop_c.values[sel].sum()
        hz.append((mh - mh.mean()) / mh.std())
    hz = np.stack(hz)
    # first-donor sensitivity (CounterFact)
    if c["task"] == "cf":
        fr = np.nonzero(first)[0]
        mf = S[fr].sum(0) / drop_row[fr].sum()
        rho_first = float(pd.Series(m).corr(pd.Series(mf), method="spearman"))
        top5_first = [cols[j] for j in np.argsort(-mf)[:5]]
    else:
        rho_first, top5_first = None, None
    lay_attn = X.ratio_cols(pd.DataFrame(Alc, index=ucases, columns=[f"L{l}" for l in range(L)]), drop_c, unit)
    sumh_c = Sc.reshape(len(ucases), L, nH).sum(-1)
    lay_sum = X.ratio_cols(pd.DataFrame(sumh_c, index=ucases, columns=[f"L{l}" for l in range(L)]), drop_c, unit)
    # row-level additivity within a layer
    sum_row = S.reshape(R, L, nH).sum(-1)
    r_layer = float(np.corrcoef(sum_row.ravel(), Al.ravel())[0, 1])
    # per-layer r over rows for the top attention layers
    top_layers = list(np.argsort(-lay_attn.ratio.values)[:6])
    # recurrence: share of rows in which the head is among the row's top-5 heads
    top5 = np.argsort(-S, axis=1)[:, :5]
    rec = np.bincount(top5.ravel(), minlength=NC) / R
    top1 = np.bincount(np.argmax(S, axis=1), minlength=NC) / R
    # DLA vs single patch
    rho_row = row_spearman(S, Dm)
    pear_row = np.array([np.corrcoef(S[i], Dm[i])[0, 1] for i in range(R)])
    # Spearman restricted to the union of each row's top-32 heads by either ranking (most heads are ~0 + bf16 noise)
    rho_top = []
    for i in range(R):
        u = np.union1d(np.argsort(-S[i])[:32], np.argsort(-Dm[i])[:32])
        rho_top.append(pd.Series(S[i, u]).corr(pd.Series(Dm[i, u]), method="spearman"))
    top1_agree = float((np.argmax(S, 1) == np.argmax(Dm, 1)).mean())
    ov10 = topk_overlap(S, Dm, 10)
    ov32 = topk_overlap(S, Dm, 32)
    r_pop = float(np.corrcoef(m, popd.ratio.values)[0, 1])
    # head DLA total vs ext8 direct split (attention DLA, linear, same definition)
    hsum = Dm.sum(1)
    if c.get("ext8"):
        dd = pd.read_parquet(os.path.join(RESULTS, c["ext8"], "addback_direct.parquet")).set_index("row_id")
        ref = dd.loc[rows].dla_attn.values
    else:  # IOI: the ext7-controls direct split (one row per directed case = row)
        dd = pd.read_parquet(os.path.join(RESULTS, c["src"], "direct_split.parquet")).drop_duplicates("case_id").set_index("case_id")
        ref = dd.loc[case_row].dla_attn.values
    r_direct = float(np.corrcoef(hsum, ref)[0, 1])
    mad_direct = float(np.median(np.abs(hsum - ref)))
    ratio_direct = float(hsum.sum() / ref.sum())
    # sum of all single-head effects vs drop (all-attention = 1 by construction)
    sum_all = float(Sc.sum() / drop_c.sum())
    sum_pos = float(m[m > 0].sum())  # population means (case-level clipping would add bf16 noise)
    # experts: single-expert rescue of the source run (ext8 Task.single) and expert DLA of this pass
    K = t.K
    E1 = np.zeros((R, K))
    ED = np.zeros((R, K))
    ekeys = {}
    edd = {int(r_): dict(zip(zip(g.layer.astype(int), g.expert.astype(int)), g.dla)) for r_, g in edla.groupby("row_id")}
    for i, r_ in enumerate(rows):
        cand = t.cand[int(case_row[i])]
        ED[i] = [edd[int(r_)][p] for p in cand]
        if t.single:
            sv = t.single[int(r_)]
            E1[i] = [sv[p] for p in cand]
        else:  # IOI: no single-expert patches; expert DLA as the proxy (flagged)
            E1[i] = ED[i]
        ekeys[i] = cand
    rho_e = row_spearman(E1, ED)
    pear_e = np.array([np.corrcoef(E1[i], ED[i])[0, 1] for i in range(R)])
    best_head = S.max(1) / drop_row
    best_exp = E1.max(1) / drop_row
    # joint top-32 of heads U experts by single rescue: number of heads
    nh32 = []
    add = {k: [] for k in (1, 5, 10, 20, 32)}
    add_h = {k: [] for k in (1, 5, 10, 20, 32)}
    add_e = {k: [] for k in (1, 5, 10, 20, 32)}
    for i in range(R):
        u = np.concatenate([S[i], E1[i]])
        o = np.argsort(-u)
        nh32.append(int((o[:32] < NC).sum()))
        for k in add:
            add[k].append(u[o[:k]].sum() / drop_row[i])
            add_h[k].append(np.sort(S[i])[::-1][:k].sum() / drop_row[i])
            add_e[k].append(np.sort(E1[i])[::-1][:k].sum() / drop_row[i])
    # top heads table
    order = np.argsort(-m)
    def head_rec(j):
        l, h = divmod(int(j), nH)
        la = lay_attn.ratio.values[l]
        return {"head": cols[j], "layer": l, "h": h, "rescue_over_drop": rc_dict(pop.iloc[j]), "rescue_mean_logits": float(Sc[:, j].mean()),
                "z": float(z[j]), "z_half": [float(hz[0, j]), float(hz[1, j])], "z8_all": bool(abs(z[j]) >= 2),
                "z8_both_halves": bool((abs(hz[:, j]) >= 2).all() and np.sign(hz[0, j]) == np.sign(hz[1, j])),
                "dla_over_drop": rc_dict(popd.iloc[j]), "share_of_layer_attn": float(m[j] / la) if abs(la) > 1e-9 else None,
                "top5_share": float(rec[j]), "top1_share": float(top1[j]), "rank": int(np.nonzero(order == j)[0][0]) + 1}
    top = [head_rec(j) for j in order[:12]]
    bottom = [head_rec(j) for j in order[::-1][:4]]
    known = []
    for (l, h), lab in C.KNOWN_HEADS.get(key, {}).items():
        if l < L and h < nH:
            r_ = head_rec(l * nH + h)
            r_["label"] = lab
            known.append(r_)
    # recurring heads by top-5 share
    rec_order = np.argsort(-rec)[:8]
    recurring = [{"head": cols[j], "top5_share": float(rec[j]), "top1_share": float(top1[j]), "rescue_over_drop": float(m[j])} for j in rec_order]
    # heads needed (population ranking, additive) for q of the sum of positive population effects / of the drop
    srt = np.sort(m)[::-1]
    cum = np.cumsum(srt)
    def k_for(thr):
        hit = np.nonzero(cum >= thr)[0]
        return int(hit[0]) + 1 if len(hit) else None
    det_pos = int(((z >= 2)).sum())
    det_neg = int(((z <= -2)).sum())
    det_both = int((((hz >= 2).all(0)) | ((hz <= -2).all(0))).sum())
    layers = []
    for l in range(L):
        sl = slice(l * nH, (l + 1) * nH)
        jb = l * nH + int(np.argmax(m[sl]))
        layers.append({"layer": l, "attn_layer": rc_dict(lay_attn.iloc[l]), "sum_heads": rc_dict(lay_sum.iloc[l]),
                       "r_rows": float(np.corrcoef(sum_row[:, l], Al[:, l])[0, 1]) if Al[:, l].std() > 0 else None,
                       "top_head": cols[jb], "top_head_ratio": float(m[jb]), "n_z8_pos": int((z[sl] >= 2).sum()),
                       "n_z8_neg": int((z[sl] <= -2).sum())})
    out = {"key": key, "label": c["label"], "task": c["task_label"], "L": L, "nH": nH, "expert_single_is_dla": not bool(t.single), "n_rows": R, "n_cases": len(ucases),
           "mean_drop": float(drop_c.mean()), "passes": int(pf.chunk.nunique()),
           "top": top, "bottom": bottom, "known": known, "recurring": recurring,
           "z8": {"n_pos": det_pos, "n_neg": det_neg, "n_both_halves": det_both, "mean_over_heads": float(m.mean()), "sd_over_heads": float(m.std())},
           "k_heads_pop_additive": {"50pct_drop": k_for(0.5), "80pct_drop": k_for(0.8), "90pct_drop": k_for(0.9)},
           "sum_all_heads_over_drop": sum_all, "sum_positive_heads_over_drop": sum_pos,
           "first_donor": {"spearman_pop": rho_first, "top5": top5_first},
           "layer_additivity": {"r_rows_all_layers": r_layer, "top_layers": [int(l) for l in top_layers]},
           "layers": layers,
           "dla_vs_single": {"row_spearman": med_iqr(rho_row), "row_pearson": med_iqr(pear_row), "row_spearman_top32": med_iqr(rho_top), "top1_agree": top1_agree, "top10_overlap": float(ov10.mean()),
                             "top32_overlap": float(ov32.mean()), "pop_pearson": r_pop,
                             "pop_spearman": float(pd.Series(m).corr(pd.Series(popd.ratio.values), method="spearman")),
                             "sum_single_over_sum_dla_top20": float(m[order[:20]].sum() / popd.ratio.values[order[:20]].sum())},
           "head_dla_total_vs_ext8_direct": {"r": r_direct, "median_abs_diff": mad_direct, "ratio_of_sums": ratio_direct},
           "experts": {"row_spearman_dla_single": med_iqr(rho_e), "row_pearson_dla_single": med_iqr(pear_e), "best_head_over_drop": med_iqr(best_head),
                       "best_expert_over_drop": med_iqr(best_exp), "frac_rows_best_head_gt_best_expert": float((best_head > best_exp).mean()),
                       "n_heads_in_joint_top32": med_iqr(nh32),
                       "additive_topk_mixed": {k: med_iqr(v) for k, v in add.items()},
                       "additive_topk_heads": {k: med_iqr(v) for k, v in add_h.items()},
                       "additive_topk_experts": {k: med_iqr(v) for k, v in add_e.items()}},
           "prefill": {"mean_drop_rows": float(drop_row.mean())}}
    # consistency with earlier head runs
    if key in C.W5_RUNS:
        w5 = pd.read_parquet(os.path.join(RESULTS, C.W5_RUNS[key], "head_rows.parquet"))
        w5 = w5[w5.kind == "attn_head"]
        mine = H[["case_id", "layer", "head", "rescue"]]
        j = mine.merge(w5[["case_id", "layer", "head", "rescue"]], on=["case_id", "layer", "head"], suffixes=("", "_w5"))
        out["w5_rowmatch"] = {"n": int(len(j)), "r": float(np.corrcoef(j.rescue, j.rescue_w5)[0, 1]) if len(j) else None,
                              "median_abs_diff": float(np.median(np.abs(j.rescue - j.rescue_w5))) if len(j) else None,
                              "p99_abs_diff": float(np.percentile(np.abs(j.rescue - j.rescue_w5), 99)) if len(j) else None,
                              "layers": sorted(int(x) for x in j.layer.unique())}
    if key in C.F2_RUNS:
        f2 = pd.read_parquet(os.path.join(RESULTS, C.F2_RUNS[key], "head_rows.parquet"))
        f2 = f2[f2.kind == "attn_head"]
        f2d = f2.drop_duplicates("case_id")
        gn_drop = float((f2d.delta_clean - f2d.delta_noised).mean())
        g = f2.groupby(["layer", "head"]).rescue.mean() / gn_drop
        comp = []
        for l in sorted(f2.layer.unique()):
            gl = g.loc[l]
            mine_l = m[int(l) * nH:(int(l) + 1) * nH]
            comp.append({"layer": int(l), "gn_top": hname(l, gl.idxmax()), "gn_top_ratio": float(gl.max()),
                         "str_top": hname(l, int(np.argmax(mine_l))), "str_top_ratio": float(mine_l.max()),
                         "spearman_heads": float(pd.Series(gl.values).corr(pd.Series(mine_l), method="spearman"))})
        out["f2_gn_compare"] = comp
    arrays = {"m": m, "md": popd.ratio.values, "z": z, "lay_attn": lay_attn.ratio.values, "lay_sum": lay_sum.ratio.values,
              "rho_row": rho_row, "best_head": best_head, "best_exp": best_exp}
    return out, arrays


def tables(res, suffix=""):
    tl = []
    for key, r in res.items():
        for kind, lst in (("top", r["top"][:8]), ("bottom", r["bottom"][:3])):
            for h in lst:
                tl.append({"Run": f"{r['label']}, {r['task']}", "Head": h["head"], "Rank": h["rank"],
                           "Rescue / drop [95% CI]": f"{h['rescue_over_drop']['ratio']:+.3f} [{h['rescue_over_drop']['lo']:+.3f}, {h['rescue_over_drop']['hi']:+.3f}]",
                           "Rescue (logits)": f"{h['rescue_mean_logits']:+.2f}", "z (all / halves)": f"{h['z']:+.1f} / {h['z_half'][0]:+.1f}, {h['z_half'][1]:+.1f}",
                           "Z8 both halves": "yes" if h["z8_both_halves"] else "no",
                           "DLA / drop": f"{h['dla_over_drop']['ratio']:+.3f}",
                           "Share of layer attn": "" if h["share_of_layer_attn"] is None else f"{h['share_of_layer_attn']:.2f}",
                           "Row top-5 / top-1 share": f"{h['top5_share']:.2f} / {h['top1_share']:.2f}"})
    save(pd.DataFrame(tl), f"ext10_heads_top{suffix}")
    kn = []
    for key, r in res.items():
        for h in r["known"]:
            kn.append({"Run": f"{r['label']}, {r['task']}", "Head": h["head"], "Earlier result": h["label"], "Rank (of all heads)": h["rank"],
                       "Rescue / drop": f"{h['rescue_over_drop']['ratio']:+.3f} [{h['rescue_over_drop']['lo']:+.3f}, {h['rescue_over_drop']['hi']:+.3f}]",
                       "z": f"{h['z']:+.1f}", "Z8 both halves": "yes" if h["z8_both_halves"] else "no", "DLA / drop": f"{h['dla_over_drop']['ratio']:+.3f}"})
    if kn:
        save(pd.DataFrame(kn), f"ext10_heads_known{suffix}")
    sm = []
    for key, r in res.items():
        dv, ex = r["dla_vs_single"], r["experts"]
        sm.append({"Run": f"{r['label']}, {r['task']}", "Rows / cases": f"{r['n_rows']} / {r['n_cases']}", "Mean drop": f"{r['mean_drop']:.2f}",
                   "Heads Z8 + / − (both halves)": f"{r['z8']['n_pos']} / {r['z8']['n_neg']} ({r['z8']['n_both_halves']})",
                   "Σ single heads / drop (all; positive pop. means)": f"{r['sum_all_heads_over_drop']:.2f}, {r['sum_positive_heads_over_drop']:.2f}",
                   "Pop. heads for 50/80/90 % (additive)": "/".join(str(r['k_heads_pop_additive'][q]) for q in ('50pct_drop', '80pct_drop', '90pct_drop')),
                   "Best head / best expert per row (median)": f"{ex['best_head_over_drop']['median']:.3f} / {ex['best_expert_over_drop']['median']:.3f}",
                   "Rows best head > best expert": f"{ex['frac_rows_best_head_gt_best_expert']:.2f}",
                   "Heads in joint top-32 (median [IQR])": f"{ex['n_heads_in_joint_top32']['median']:.0f} [{ex['n_heads_in_joint_top32']['q25']:.0f}, {ex['n_heads_in_joint_top32']['q75']:.0f}]",
                   "DLA vs single, heads: row r / row ρ top-32": f"{dv['row_pearson']['median']:.2f} / {dv['row_spearman_top32']['median']:.2f}",
                   "top-1 / top-10 agree": f"{dv['top1_agree']:.2f} / {dv['top10_overlap']:.2f}",
                   "pop. r": f"{dv['pop_pearson']:.2f}",
                   "experts: row r": f"{ex['row_pearson_dla_single']['median']:.2f}",
                   "Σ head DLA vs ext8 attention DLA (r, ratio)": f"{r['head_dla_total_vs_ext8_direct']['r']:.3f}, {r['head_dla_total_vs_ext8_direct']['ratio_of_sums']:.3f}",
                   "Σ heads vs attn_layer (row-layer r)": f"{r['layer_additivity']['r_rows_all_layers']:.2f}"})
    save(pd.DataFrame(sm), f"ext10_heads_summary{suffix}")
    ly = []
    for key, r in res.items():
        for l in r["layer_additivity"]["top_layers"]:
            e = r["layers"][l]
            ly.append({"Run": f"{r['label']}, {r['task']}", "Layer": f"L{l}",
                       "attn_layer / drop": f"{e['attn_layer']['ratio']:+.3f} [{e['attn_layer']['lo']:+.3f}, {e['attn_layer']['hi']:+.3f}]",
                       "Σ heads / drop": f"{e['sum_heads']['ratio']:+.3f}", "r(Σ heads, attn_layer) rows": f"{e['r_rows']:.2f}" if e["r_rows"] is not None else "",
                       "Top head": e["top_head"], "Top head / drop": f"{e['top_head_ratio']:+.3f}", "Z8 + / −": f"{e['n_z8_pos']} / {e['n_z8_neg']}"})
    save(pd.DataFrame(ly), f"ext10_heads_layers{suffix}")
    ad = []
    for key, r in res.items():
        ex = r["experts"]
        for k in ("1", "5", "10", "20", "32"):
            kk = k if k in ex["additive_topk_mixed"] else int(k)
            ad.append({"Run": f"{r['label']}, {r['task']}", "k": k,
                       "heads ∪ experts": f"{ex['additive_topk_mixed'][kk]['median']:.2f}",
                       "heads only": f"{ex['additive_topk_heads'][kk]['median']:.2f}",
                       "experts only": f"{ex['additive_topk_experts'][kk]['median']:.2f}"})
    save(pd.DataFrame(ad), f"ext10_heads_additive{suffix}")


def figures(res, arrs, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(res)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.8), squeeze=False)
    for ax, k in zip(axes[0], keys):
        r, a = res[k], arrs[k]
        M = a["m"].reshape(r["L"], r["nH"])
        v = np.abs(M).max()
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
        ax.set_title(f"{r['label']}\n{r['task']}", fontsize=9)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        for h in r["top"][:3]:
            ax.text(h["h"], h["layer"], h["head"], fontsize=6, ha="left", va="bottom")
        plt.colorbar(im, ax=ax, fraction=0.046, label="single-head rescue / drop")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"ext10_heads_heatmap{suffix}.png"), dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, len(keys), figsize=(4.2 * len(keys), 7.6), squeeze=False)
    for j, k in enumerate(keys):
        r, a = res[k], arrs[k]
        ax = axes[0, j]
        ax.scatter(a["md"], a["m"], s=4, alpha=0.5)
        lim = max(np.abs(a["md"]).max(), np.abs(a["m"]).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.8)
        ax.set_xlabel("head DLA / drop (population)")
        ax.set_ylabel("single-head rescue / drop")
        ax.set_title(f"{r['label']}, {r['task']}\nr = {r['dla_vs_single']['pop_pearson']:.2f}", fontsize=9)
        ax = axes[1, j]
        x = np.arange(r["L"])
        ax.plot(x, a["lay_attn"], label="attn_layer patch")
        ax.plot(x, a["lay_sum"], label="Σ single heads", ls="--")
        mx = a["m"].reshape(r["L"], r["nH"]).max(1)
        ax.plot(x, mx, label="best head", ls=":")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("layer")
        ax.set_ylabel("rescue / drop")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"ext10_heads_dla_layers{suffix}.png"), dpi=140)
    plt.close(fig)
    log("figures written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(C.RUN_ORDER))
    ap.add_argument("--suffix", default="", help="run-dir suffix (smoke runs)")
    ap.add_argument("--tag", default=None, help="suffix of the output tables / figures / summary (default = --suffix)")
    args = ap.parse_args()
    tag = args.suffix if args.tag is None else args.tag
    res, arrs = {}, {}
    for k in args.runs.split(","):
        od = os.path.join(RESULTS, C.cfg(k)["out"] + args.suffix)
        sp = os.path.join(od, "heads_state.json")
        st = json.load(open(sp)) if os.path.exists(sp) else None
        if not st or len(st["done"]) < len(st["chunks"]):
            log(f"{k}: no complete step-1 run, skipped")
            continue
        log(f"analysing {k}")
        res[k], arrs[k] = analyse(k, args.suffix)
    tables(res, tag)
    figures(res, arrs, tag)
    if not tag:
        cur = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {}
        cur.setdefault("step1", {}).update(res)
        cur["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(SUMMARY, "w") as f:
            json.dump(cur, f, indent=1, default=float)
        log(f"wrote {SUMMARY}")
    else:
        with open(os.path.join(RESULTS, f"ext10_circuit_summary{tag}.json"), "w") as f:
            json.dump({"step1": res}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
