"""ext13ci analysis (Phase 5, p5-circuit): Direction-10 step 1 (single heads, head DLA, attention mass of the top heads on
the position classes) and step 2 (joint head + expert add-back curves) for every Phase-5 condition, per-condition figures
(both models as columns) and the cross-condition overview panel with Phase 4's CounterFact / WinoGrande / IOI values.

Normalisation as ext10: rescue = Delta_patched - Delta_corrupt (same pass); population values = ratio of means over the
validation directed cases with a percentile bootstrap over clusters (WinoGrande twin = HIT id of q_a; verb templates: the
seed; 5,000 resamples). Z8: |population effect - mean over heads| >= 2 SD over all heads, on all units and on two disjoint
halves of the clusters.

Attention mass (step 1): per row the final position's attention (clean and corrupted prefill rows) summed over the
positions of each class (moetrace.ext13ci.case_positions); per head the mean over validation rows. Head labels from the
clean-run mass, ignoring pos0 / final: 'copy (attribute)' if the largest class is an attribute class (name contains
'attr' or 'prefix') with mass >= 0.2, 'mover (mention)' for a mention class (contains 'ment', 'name' or 'ent'), 'STR site'
for str, 'sink' if pos0 >= 0.5, else 'diffuse'.

Writes results/tables/ext13ci_{heads,attn,circuit}_*.md|csv, results/figures/ext13ci_*.png, results/ext13ci_summary.json
(keys step1, step2, step2_runs, reference).

Usage: python scripts/ext13ci_analyze.py [--step 1|2|all] [--keys f1/qwen3,...] [--suffix _smoke] [--tag _smoke]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from moetrace.models import RESULTS
from moetrace import ext8_addback as X
from moetrace import ext10_circuit as C
from moetrace import ext13ci as E13

TAB, FIG = E13.TAB, E13.FIG
Q = (0.5, 0.8, 0.9)
KG = 20


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
    if not len(x):
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan"), "n": 0}
    return {"median": float(np.median(x)), "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)), "n": int(len(x))}


def run_label(cond, proto):
    return f"{E13.MODEL_LABEL[proto]}, {E13.COND_INFO.get(cond, {}).get('label', cond)}"


def class_kind(c: str) -> str:
    base = c.rsplit("_", 1)[0] if c.endswith(("_clean", "_corr")) else c
    if c in ("pos0", "final", "str"):
        return c
    if "attr" in base or "trig" in base:
        return "attribute"
    if "ment" in base or "name" in base or "ent" in base:
        return "mention"
    return base


def head_attn_label(mass_row: dict, direction: str = "") -> str:
    """mass_row: class -> mean clean-run mass of one head. Rule (in this order): the largest content class (STR site,
    mentions, in-context attributes, prefix names, blank) if >= 0.2; 'sink (pos0)' if pos0 >= 0.4; 'self (final token)' if
    final >= 0.3; 'other context tokens' if the mass outside every class is >= 0.5; else 'diffuse'."""
    content = {c: v for c, v in mass_row.items() if c not in ("pos0", "final")}
    rest = max(0.0, 1.0 - sum(mass_row.values()))
    if content:
        c = max(content, key=content.get)
        if content[c] >= 0.2:
            k = class_kind(c)
            side = ": answer" if c.endswith("_clean") else (": other candidate" if c.endswith("_corr") else "")
            base = c.rsplit("_", 1)[0] if side else c
            if k == "attribute":
                return f"copy ({base}{side})"
            if k == "mention":
                return f"mover ({base}{side})"
            if k == "str":
                return "STR site (" + ("trigger" if direction == "a2e" else "filled entity" if direction == "e2a" else "swap") + ")"
            return c
    if mass_row.get("pos0", 0) >= 0.4:
        return "sink (pos0)"
    if mass_row.get("final", 0) >= 0.3:
        return "self (final token)"
    if rest >= 0.5:
        return "other context tokens"
    return "diffuse"


# ======================================================================================================================
# step 1
# ======================================================================================================================
def analyse1(cond, proto, suffix=""):
    od = E13.step1_dir(cond, proto, suffix)
    t = E13.load_task(cond, proto)
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
    S = np.full((R, NC), np.nan)
    S[H.row_id.map(ri).values, (H.layer.values.astype(int) * nH + H["head"].values.astype(int))] = H.rescue.values
    Dm = np.full((R, NC), np.nan)
    Dm[hdla.row_id.map(ri).values, (hdla.layer.values.astype(int) * nH + hdla["head"].values.astype(int))] = hdla.dla.values
    assert not np.isnan(S).any() and not np.isnan(Dm).any()
    Al = hd[hd.kind == "attn_layer"].pivot(index="row_id", columns="layer", values="rescue").loc[rows].values
    rinfo = hd.drop_duplicates("row_id").set_index("row_id").loc[rows]
    drop_row = (rinfo.delta_clean - rinfo.delta_corrupt).values
    case_row = rinfo.case_id.values.astype(int)
    cases = t.cases.set_index("case_id")
    unit = cases.unit
    ucases = list(case_row)  # one row per directed case
    drop_c = pd.Series(drop_row, index=ucases)
    cols = [hname(l, h) for l in range(L) for h in range(nH)]
    pop = X.ratio_cols(pd.DataFrame(S, index=ucases, columns=cols), drop_c, unit)
    popd = X.ratio_cols(pd.DataFrame(Dm, index=ucases, columns=cols), drop_c, unit)
    m = pop.ratio.values
    z = (m - m.mean()) / m.std()
    us = np.array(sorted(set(unit.loc[ucases])))
    half = {u: i % 2 for i, u in enumerate(us)}
    hz = []
    for hb in (0, 1):
        sel = [j for j, cc in enumerate(ucases) if half[unit.loc[cc]] == hb]
        mh = S[sel].sum(0) / drop_row[sel].sum()
        hz.append((mh - mh.mean()) / mh.std())
    hz = np.stack(hz)
    lay_attn = X.ratio_cols(pd.DataFrame(Al, index=ucases, columns=[f"L{l}" for l in range(L)]), drop_c, unit)
    sum_row = S.reshape(R, L, nH).sum(-1)
    lay_sum = X.ratio_cols(pd.DataFrame(sum_row, index=ucases, columns=[f"L{l}" for l in range(L)]), drop_c, unit)
    r_layer = float(np.corrcoef(sum_row.ravel(), Al.ravel())[0, 1])
    top_layers = list(np.argsort(-lay_attn.ratio.values)[:6])
    top5 = np.argsort(-S, axis=1)[:, :5]
    rec = np.bincount(top5.ravel(), minlength=NC) / R
    top1 = np.bincount(np.argmax(S, axis=1), minlength=NC) / R
    rho_row = row_spearman(S, Dm)
    pear_row = np.array([np.corrcoef(S[i], Dm[i])[0, 1] for i in range(R)])
    rho_top = []
    for i in range(R):
        u = np.union1d(np.argsort(-S[i])[:32], np.argsort(-Dm[i])[:32])
        rho_top.append(pd.Series(S[i, u]).corr(pd.Series(Dm[i, u]), method="spearman"))
    top1_agree = float((np.argmax(S, 1) == np.argmax(Dm, 1)).mean())
    ov10 = topk_overlap(S, Dm, 10)
    ov32 = topk_overlap(S, Dm, 32)
    r_pop = float(np.corrcoef(m, popd.ratio.values)[0, 1])
    hsum = Dm.sum(1)
    dpath = os.path.join(RESULTS, t.run, "direct_split.parquet")
    direct = None
    if os.path.exists(dpath):
        dd = pd.read_parquet(dpath).drop_duplicates("case_id").set_index("case_id")
        if "dla_attn" in dd.columns and set(case_row) <= set(dd.index):
            ref = dd.loc[case_row].dla_attn.values
            direct = {"r": float(np.corrcoef(hsum, ref)[0, 1]), "median_abs_diff": float(np.median(np.abs(hsum - ref))),
                      "ratio_of_sums": float(hsum.sum() / ref.sum())}
    sum_all = float(S.sum() / drop_row.sum())
    sum_pos = float(m[m > 0].sum())
    # experts: single-expert rescue of the base run (DLA fallback for missing entries), expert DLA of this pass
    K = t.K
    E1 = np.zeros((R, K))
    ED = np.zeros((R, K))
    edd = {int(r_): dict(zip(zip(g.layer.astype(int), g.expert.astype(int)), g.dla)) for r_, g in edla.groupby("row_id")}
    n_proxy = 0
    for i, r_ in enumerate(rows):
        cand = t.cand[int(case_row[i])]
        ED[i] = [edd[int(r_)][p] for p in cand]
        sv = t.single.get(int(r_), {}) if t.single else {}
        E1[i] = [sv.get(p, np.nan) for p in cand]
        miss = np.isnan(E1[i])
        n_proxy += int(miss.sum())
        E1[i, miss] = ED[i, miss]
    rho_e = row_spearman(E1, ED)
    pear_e = np.array([np.corrcoef(E1[i], ED[i])[0, 1] for i in range(R)])
    best_head = S.max(1) / drop_row
    best_exp = E1.max(1) / drop_row
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
    order = np.argsort(-m)
    # attention mass (clean / corrupted run) per head and class
    am = E13.load_attn_mass(cond, proto, suffix)
    amass = None
    if am:
        ami = {int(r_): i for i, r_ in enumerate(am["row_ids"])}
        sel = [ami[int(r_)] for r_ in rows]
        M5 = am["mass"][sel]  # [R, 2, L, nH, C]
        npos = am["n_pos"][sel]
        has = (npos > 0)  # [R, C]
        # mean over rows that have the class (NaN-safe)
        Mc = np.where(has[:, None, None, None, :], M5, np.nan)
        mean_clean = np.nanmean(Mc[:, 0], axis=0).reshape(NC, -1)  # [NC, C]
        mean_corr = np.nanmean(Mc[:, 1], axis=0).reshape(NC, -1)
        amass = {"classes": am["classes"], "clean": mean_clean, "corr": mean_corr, "n_rows_class": has.sum(0).tolist(),
                 "M": M5}

    def head_rec(j):
        l, h = divmod(int(j), nH)
        la = lay_attn.ratio.values[l]
        rr = {"head": cols[j], "layer": l, "h": h, "rescue_over_drop": rc_dict(pop.iloc[j]), "rescue_mean_logits": float(S[:, j].mean()),
              "z": float(z[j]), "z_half": [float(hz[0, j]), float(hz[1, j])], "z8_all": bool(abs(z[j]) >= 2),
              "z8_both_halves": bool((abs(hz[:, j]) >= 2).all() and np.sign(hz[0, j]) == np.sign(hz[1, j])),
              "dla_over_drop": rc_dict(popd.iloc[j]), "share_of_layer_attn": float(m[j] / la) if abs(la) > 1e-9 else None,
              "top5_share": float(rec[j]), "top1_share": float(top1[j]), "rank": int(np.nonzero(order == j)[0][0]) + 1}
        if amass is not None:
            cl = {c: float(amass["clean"][j, k]) for k, c in enumerate(amass["classes"]) if np.isfinite(amass["clean"][j, k])}
            co = {c: float(amass["corr"][j, k]) for k, c in enumerate(amass["classes"]) if np.isfinite(amass["corr"][j, k])}
            rr["attn_clean"], rr["attn_corr"] = cl, co
            rr["attn_label"] = head_attn_label(cl, E13.COND_INFO.get(cond, {}).get("direction", ""))
        return rr
    top = [head_rec(j) for j in order[:12]]
    bottom = [head_rec(j) for j in order[::-1][:4]]
    rec_order = np.argsort(-rec)[:8]
    recurring = [{"head": cols[j], "top5_share": float(rec[j]), "top1_share": float(top1[j]), "rescue_over_drop": float(m[j])} for j in rec_order]
    srt = np.sort(m)[::-1]
    cum = np.cumsum(srt)

    def k_for(thr):
        hit = np.nonzero(cum >= thr)[0]
        return int(hit[0]) + 1 if len(hit) else None
    layers = []
    for l in range(L):
        sl = slice(l * nH, (l + 1) * nH)
        jb = l * nH + int(np.argmax(m[sl]))
        layers.append({"layer": l, "attn_layer": rc_dict(lay_attn.iloc[l]), "sum_heads": rc_dict(lay_sum.iloc[l]),
                       "r_rows": float(np.corrcoef(sum_row[:, l], Al[:, l])[0, 1]) if Al[:, l].std() > 0 and sum_row[:, l].std() > 0 else None,
                       "top_head": cols[jb], "top_head_ratio": float(m[jb]), "n_z8_pos": int((z[sl] >= 2).sum()),
                       "n_z8_neg": int((z[sl] <= -2).sum())})
    meta = json.load(open(os.path.join(od, "run_meta.json")))
    out = {"key": E13.key_of(cond, proto), "cond": cond, "proto": proto, "label": E13.MODEL_LABEL[proto],
           "task": E13.COND_INFO.get(cond, {}).get("label", cond), "info": E13.COND_INFO.get(cond, {}), "L": L, "nH": nH,
           "n_rows": R, "n_units": int(len(us)), "cluster": t.cluster_src, "mean_drop": float(drop_c.mean()),
           "mean_delta_clean": float(rinfo.delta_clean.mean()), "passes": int(pf.chunk.nunique()),
           "gpu_s": float(sum(n.get("pass_s") or 0 for n in meta.get("pass_notes", []))),
           "peak_GB": float(max([n.get("peak_GB") or 0 for n in meta.get("pass_notes", [])] or [0])),
           "top": top, "bottom": bottom, "recurring": recurring,
           "z8": {"n_pos": int((z >= 2).sum()), "n_neg": int((z <= -2).sum()),
                  "n_both_halves": int((((hz >= 2).all(0)) | ((hz <= -2).all(0))).sum()), "mean_over_heads": float(m.mean()),
                  "sd_over_heads": float(m.std())},
           "k_heads_pop_additive": {"50pct_drop": k_for(0.5), "80pct_drop": k_for(0.8), "90pct_drop": k_for(0.9)},
           "sum_all_heads_over_drop": sum_all, "sum_positive_heads_over_drop": sum_pos,
           "layer_additivity": {"r_rows_all_layers": r_layer, "top_layers": [int(l) for l in top_layers]}, "layers": layers,
           "dla_vs_single": {"row_spearman": med_iqr(rho_row), "row_pearson": med_iqr(pear_row), "row_spearman_top32": med_iqr(rho_top),
                             "top1_agree": top1_agree, "top10_overlap": float(ov10.mean()), "top32_overlap": float(ov32.mean()),
                             "pop_pearson": r_pop, "pop_spearman": float(pd.Series(m).corr(pd.Series(popd.ratio.values), method="spearman"))},
           "head_dla_total_vs_direct_split": direct,
           "experts": {"n_single_proxied_by_dla": n_proxy, "row_spearman_dla_single": med_iqr(rho_e), "row_pearson_dla_single": med_iqr(pear_e),
                       "best_head_over_drop": med_iqr(best_head), "best_expert_over_drop": med_iqr(best_exp),
                       "frac_rows_best_head_gt_best_expert": float((best_head > best_exp).mean()),
                       "n_heads_in_joint_top32": med_iqr(nh32),
                       "additive_topk_mixed": {str(k): med_iqr(v) for k, v in add.items()},
                       "additive_topk_heads": {str(k): med_iqr(v) for k, v in add_h.items()},
                       "additive_topk_experts": {str(k): med_iqr(v) for k, v in add_e.items()}}}
    # attention-mass population summaries
    if amass is not None:
        cls = amass["classes"]
        pos_idx = np.nonzero(m > 0)[0]
        w = m[pos_idx] / m[pos_idx].sum()
        wmass = {c: float(np.nansum(w * amass["clean"][pos_idx, k])) for k, c in enumerate(cls)}
        topw = order[:10]
        wt = np.clip(m[topw], 0, None)
        wt = wt / wt.sum() if wt.sum() > 0 else wt
        wmass10 = {c: float(np.nansum(wt * amass["clean"][topw, k])) for k, c in enumerate(cls)}
        allh = {c: float(np.nanmean(amass["clean"][:, k])) for k, c in enumerate(cls)}
        # correlation over heads: single-head rescue vs clean-run mass per class (layers of the second half)
        late = np.repeat(np.arange(L), nH) >= L // 2
        corr = {}
        for k, c in enumerate(cls):
            v = amass["clean"][late, k]
            ok = np.isfinite(v)
            corr[c] = float(np.corrcoef(m[late][ok], v[ok])[0, 1]) if ok.sum() > 3 and v[ok].std() > 0 else None
        # per-row: top-1 head of the row (by rescue): attention mass on its classes in that row (clean / corrupt)
        M5 = amass["M"]
        jb = np.argmax(S, axis=1)
        lb, hb = jb // nH, jb % nH
        per_row = {c: {"clean": float(np.nanmean(M5[np.arange(R), 0, lb, hb, k])), "corr": float(np.nanmean(M5[np.arange(R), 1, lb, hb, k]))}
                   for k, c in enumerate(cls)}
        labels = {}
        for hr in top[:10]:
            labels[hr["attn_label"]] = labels.get(hr["attn_label"], 0) + 1
        out["attn"] = {"classes": cls, "n_rows_with_class": dict(zip(cls, amass["n_rows_class"])),
                       "rescue_weighted_mass_positive_heads": wmass, "rescue_weighted_mass_top10": wmass10,
                       "mean_mass_all_heads": allh, "corr_rescue_vs_mass_late_layers": corr,
                       "row_top1_head_mass": per_row, "top10_labels": labels}
    # consistency with the base run's W5 head rows (if any)
    w5p = os.path.join(RESULTS, t.run, "head_rows.parquet")
    if os.path.exists(w5p):
        w5 = pd.read_parquet(w5p)
        w5 = w5[w5.kind == "attn_head"]
        mine = H[["case_id", "layer", "head", "rescue"]]
        j = mine.merge(w5[["case_id", "layer", "head", "rescue"]], on=["case_id", "layer", "head"], suffixes=("", "_w5"))
        if len(j):
            out["w5_rowmatch"] = {"n": int(len(j)), "r": float(np.corrcoef(j.rescue, j.rescue_w5)[0, 1]),
                                  "median_abs_diff": float(np.median(np.abs(j.rescue - j.rescue_w5))),
                                  "layers": sorted(int(x) for x in j.layer.unique())}
    arrays = {"m": m, "md": popd.ratio.values, "z": z, "lay_attn": lay_attn.ratio.values, "lay_sum": lay_sum.ratio.values}
    if amass is not None:
        arrays["attn_clean"] = amass["clean"]
        arrays["attn_corr"] = amass["corr"]
        arrays["classes"] = amass["classes"]
    return out, arrays


def tables1(res, tag=""):
    tl = []
    for key, r in res.items():
        for kind, lst in (("top", r["top"][:8]), ("bottom", r["bottom"][:3])):
            for h in lst:
                tl.append({"Run": f"{r['label']}, {r['task']}", "Head": h["head"], "Rank": h["rank"],
                           "Rescue / drop [95% CI]": f"{h['rescue_over_drop']['ratio']:+.3f} [{h['rescue_over_drop']['lo']:+.3f}, {h['rescue_over_drop']['hi']:+.3f}]",
                           "Rescue (logits)": f"{h['rescue_mean_logits']:+.2f}", "z (all / halves)": f"{h['z']:+.1f} / {h['z_half'][0]:+.1f}, {h['z_half'][1]:+.1f}",
                           "Z8 both halves": "yes" if h["z8_both_halves"] else "no", "DLA / drop": f"{h['dla_over_drop']['ratio']:+.3f}",
                           "Row top-5 / top-1 share": f"{h['top5_share']:.2f} / {h['top1_share']:.2f}",
                           "Attention (clean run)": h.get("attn_label", "")})
    save(pd.DataFrame(tl), f"ext13ci_heads_top{tag}")
    sm = []
    for key, r in res.items():
        dv, ex = r["dla_vs_single"], r["experts"]
        dr = r.get("head_dla_total_vs_direct_split")
        sm.append({"Run": f"{r['label']}, {r['task']}", "Rows / units": f"{r['n_rows']} / {r['n_units']}", "Mean drop": f"{r['mean_drop']:.2f}",
                   "Heads Z8 + / − (both halves)": f"{r['z8']['n_pos']} / {r['z8']['n_neg']} ({r['z8']['n_both_halves']})",
                   "Σ single heads / drop (all; positive)": f"{r['sum_all_heads_over_drop']:.2f}, {r['sum_positive_heads_over_drop']:.2f}",
                   "Pop. heads for 50/80/90 % (additive)": "/".join(str(r['k_heads_pop_additive'][q]) for q in ('50pct_drop', '80pct_drop', '90pct_drop')),
                   "Best head / best expert per row (median)": f"{ex['best_head_over_drop']['median']:.3f} / {ex['best_expert_over_drop']['median']:.3f}",
                   "Rows best head > best expert": f"{ex['frac_rows_best_head_gt_best_expert']:.2f}",
                   "Heads in joint top-32": f"{ex['n_heads_in_joint_top32']['median']:.0f} [{ex['n_heads_in_joint_top32']['q25']:.0f}, {ex['n_heads_in_joint_top32']['q75']:.0f}]",
                   "DLA vs single heads: pop. r / top-1 agree": f"{dv['pop_pearson']:.2f} / {dv['top1_agree']:.2f}",
                   "experts: row r (DLA vs single)": f"{ex['row_pearson_dla_single']['median']:.2f}",
                   "Σ head DLA vs direct split (r, ratio)": "" if dr is None else f"{dr['r']:.3f}, {dr['ratio_of_sums']:.3f}",
                   "Σ heads vs attn_layer (row-layer r)": f"{r['layer_additivity']['r_rows_all_layers']:.2f}"})
    save(pd.DataFrame(sm), f"ext13ci_heads_summary{tag}")
    ly = []
    for key, r in res.items():
        for l in r["layer_additivity"]["top_layers"]:
            e = r["layers"][l]
            ly.append({"Run": f"{r['label']}, {r['task']}", "Layer": f"L{l}",
                       "attn_layer / drop": f"{e['attn_layer']['ratio']:+.3f} [{e['attn_layer']['lo']:+.3f}, {e['attn_layer']['hi']:+.3f}]",
                       "Σ heads / drop": f"{e['sum_heads']['ratio']:+.3f}", "r(Σ heads, attn_layer) rows": "" if e["r_rows"] is None else f"{e['r_rows']:.2f}",
                       "Top head": e["top_head"], "Top head / drop": f"{e['top_head_ratio']:+.3f}", "Z8 + / −": f"{e['n_z8_pos']} / {e['n_z8_neg']}"})
    save(pd.DataFrame(ly), f"ext13ci_heads_layers{tag}")
    at, ap_ = [], []

    def groups(cls):
        """(label, [classes]) display groups: paired bases as 'clean side / other side', single classes alone."""
        out, seen = [], set()
        for c in cls:
            if c in seen:
                continue
            if c.endswith("_clean") and c[:-6] + "_corr" in cls:
                b = c[:-6]
                out.append((f"{b} (answer side / other)", [c, b + "_corr"]))
                seen |= {c, b + "_corr"}
            else:
                out.append((c, [c]))
                seen.add(c)
        return out

    def fmt_mass(d, cs):
        v = [d.get(c) for c in cs]
        return "" if any(x is None for x in v) else " / ".join(f"{x:.2f}" for x in v)
    for key, r in res.items():
        if "attn" not in r:
            continue
        cls = list(r["attn"]["classes"])
        gr = groups(cls)
        for h in r["top"][:8] + r["bottom"][:2]:
            row = {"Run": f"{r['label']}, {r['task']}", "Head": h["head"], "Rescue / drop": f"{h['rescue_over_drop']['ratio']:+.3f}",
                   "Label": h.get("attn_label", "")}
            for lab, cs in gr:
                row[lab] = fmt_mass(h["attn_clean"], cs)
            row["rest"] = f"{max(0.0, 1 - sum(h['attn_clean'].values())):.2f}"
            at.append(row)
        a = r["attn"]
        for nm, d in (("all heads", a["mean_mass_all_heads"]), ("positive heads, rescue-weighted", a["rescue_weighted_mass_positive_heads"]),
                      ("top-10 heads, rescue-weighted", a["rescue_weighted_mass_top10"]),
                      ("best head of each row", {c: v["clean"] for c, v in a["row_top1_head_mass"].items()})):
            row = {"Run": f"{r['label']}, {r['task']}", "Heads": nm}
            for lab, cs in gr:
                row[lab] = fmt_mass(d, cs)
            ap_.append(row)
        row = {"Run": f"{r['label']}, {r['task']}", "Heads": "labels of the top-10 heads"}
        row[gr[0][0]] = ", ".join(f"{k} {v}" for k, v in sorted(a["top10_labels"].items(), key=lambda kv: -kv[1]))
        ap_.append(row)
    if at:
        save(pd.DataFrame(at).fillna(""), f"ext13ci_attn_heads{tag}")
        save(pd.DataFrame(ap_).fillna(""), f"ext13ci_attn_population{tag}")


def figures1(res, arrs, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = list(dict.fromkeys(r["cond"] for r in res.values()))
    for cond in conds:
        keys = [k for k in res if res[k]["cond"] == cond]
        n = len(keys)
        # heatmap
        fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.8), squeeze=False)
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
        fig.savefig(os.path.join(FIG, f"ext13ci_heads_heatmap_{cond}{tag}.png"), dpi=130)
        plt.close(fig)
        # DLA vs single + layer curves
        fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 7.6), squeeze=False)
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
            ax.plot(x, a["m"].reshape(r["L"], r["nH"]).max(1), label="best head", ls=":")
            ax.plot(x, a["md"].reshape(r["L"], r["nH"]).sum(1), label="Σ head DLA", ls="-.", lw=0.8)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("layer")
            ax.set_ylabel("rescue / drop")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"ext13ci_heads_dla_layers_{cond}{tag}.png"), dpi=130)
        plt.close(fig)
        # attention mass of the top heads
        ka = [k for k in keys if "attn" in res[k]]
        if ka:
            fig, axes = plt.subplots(1, len(ka), figsize=(5.2 * len(ka), 4.2), squeeze=False)
            for ax, k in zip(axes[0], ka):
                r = res[k]
                cls = r["attn"]["classes"]
                hs = r["top"][:6] + r["bottom"][:2]
                Mx = np.array([[h["attn_clean"].get(c, np.nan) for c in cls] for h in hs])
                im = ax.imshow(Mx, aspect="auto", cmap="viridis", vmin=0, vmax=1)
                ax.set_xticks(range(len(cls)))
                ax.set_xticklabels(cls, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(hs)))
                ax.set_yticklabels([f"{h['head']} ({h['rescue_over_drop']['ratio']:+.2f})" for h in hs], fontsize=7)
                for i_ in range(Mx.shape[0]):
                    for j_ in range(Mx.shape[1]):
                        if np.isfinite(Mx[i_, j_]):
                            ax.text(j_, i_, f"{Mx[i_, j_]:.2f}", ha="center", va="center", fontsize=6,
                                    color="w" if Mx[i_, j_] < 0.5 else "k")
                ax.set_title(f"{r['label']}, {r['task']}\nfinal-position attention (clean run)", fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG, f"ext13ci_heads_attn_{cond}{tag}.png"), dpi=130)
            plt.close(fig)
    log("step-1 figures written")


# ======================================================================================================================
# step 2
# ======================================================================================================================
def pop_curve(M, drop_c, unit):
    return X.ratio_cols(M, drop_c, unit)


def auc_upto(ks, r, kmax):
    gk = np.arange(1, kmax + 1)
    return X.auc_log(gk, np.interp(np.log(gk), np.log(np.asarray(ks, dtype=float)), np.asarray(r, dtype=float)))


def k_first(ks, r, thr):
    for k, v in zip(ks, r):
        if v >= thr:
            return float(k)
    return float("inf")


def cname(c):
    return f"L{c[1]}E{c[2]:03d}" if c[0] == "e" else f"L{c[1]}H{c[2]}"


def addback_reference(cond, proto, vrows, case_of):
    """p5-addback's expert-only greedy on the same directed cases, if a compatible ext8-style state exists."""
    od = os.path.join(RESULTS, f"p5_{cond}_{proto}_ab")
    sp = os.path.join(od, "addback_state.pkl")
    if not os.path.exists(sp):
        return None
    try:
        st8 = X.load_state(sp)
        pfs = sorted(f for f in os.listdir(od) if f.startswith("addback_prefill_p"))
        p8 = pd.read_parquet(os.path.join(od, pfs[0]))
        dcl = p8[p8.kind == "clean"].groupby("case_id").delta.first()
        dco = p8[p8.kind == "corrupt"].groupby("row_id").delta.first()
        rid8 = p8[p8.kind == "corrupt"].groupby("case_id").row_id.first()
        out = {}
        for r_ in vrows:
            c = case_of[r_]
            if c not in rid8.index:
                return None
            r8 = int(rid8.loc[c])
            g = st8.greedy.get(r8)
            if g is None or not g.resc:
                return None
            out[r_] = (list(g.resc), float(dcl.loc[c] - dco.loc[r8]))
        return out
    except Exception as ex:  # incompatible format: skip the comparison
        log(f"addback reference for {cond}/{proto} skipped: {ex}")
        return None


def analyse_task2(key, st, rows_all, pf0):
    cond, proto = E13.parse_key(key)
    t = E13.load_task(cond, proto)
    cases = t.cases.set_index("case_id")
    unit = cases.unit
    R = rows_all[rows_all.task == key]
    P = pf0[pf0.task == key]
    dclean = P[P.kind == "clean"].set_index("case_id").delta
    dcor = P[P.kind == "corrupt"].set_index("row_id").delta
    vrows = sorted(int(x) for x in P[P.kind == "corrupt"].row_id)
    case_of = {r_: int(t.rows.case_id.iloc[r_]) for r_ in vrows}
    drop_ref = pd.Series({r_: float(dclean.loc[case_of[r_]] - dcor.loc[r_]) for r_ in vrows})
    ucases = [case_of[r_] for r_ in vrows]
    drop_c = pd.Series(drop_ref.values, index=ucases)
    out = {"key": key, "cond": cond, "proto": proto, "label": E13.MODEL_LABEL[proto], "task": E13.COND_INFO.get(cond, {}).get("label", cond),
           "info": E13.COND_INFO.get(cond, {}), "n_rows": len(vrows), "n_units": int(unit.loc[ucases].nunique()), "mean_drop": float(drop_c.mean())}
    ce = R[R.fam == "ceil"].copy()
    ce["rescue"] = ce.delta - ce.delta_corrupt
    cm = ce.groupby(["case_id", "order"]).rescue.mean().unstack("order").loc[ucases]
    rc = X.ratio_cols(cm, drop_c, unit)
    out["ceilings"] = {k: {"ratio": float(v.ratio), "lo": float(v.lo), "hi": float(v.hi)} for k, v in rc.iterrows()}
    ah = ce[ce.order == "all_heads"]
    out["sanity_all_heads"] = {"max_abs_dev": float((ah.delta - ah.delta_clean).abs().max()),
                               "median_abs_dev": float((ah.delta - ah.delta_clean).abs().median()),
                               "ratio_mean": float(ah.delta.mean() / ah.delta_clean.mean())}
    sa = R[R.fam == "sanity"].copy()
    if len(sa):
        sa["layer"] = sa.set.str.split(":").str[1].astype(int)
        pv = sa.pivot_table(index=["row_id", "layer"], columns="order", values="delta", aggfunc="first").dropna()
        dv = (pv["layer_heads"] - pv["attn_layer"]).abs()
        out["sanity_layer_heads"] = {"n": int(len(pv)), "max_abs_dev": float(dv.max()), "median_abs_dev": float(dv.median()),
                                     "frac_within_0.25": float((dv <= 0.25).mean()),
                                     "r": float(np.corrcoef(pv["layer_heads"], pv["attn_layer"])[0, 1])}
    curves = {}
    G = {}
    for kind in C.GREEDY_KINDS:
        res_m = pd.DataFrame({r_: st.greedy[(key, kind, r_)].resc[:KG] for r_ in vrows}).T
        res_m.columns = np.arange(1, res_m.shape[1] + 1)
        val_m = pd.DataFrame({r_: st.greedy[(key, kind, r_)].vals[:KG] for r_ in vrows}).T
        val_m.columns = res_m.columns
        G[kind] = (res_m, val_m)
        curves[f"greedy_{kind}"] = pop_curve(pd.DataFrame(res_m.values, index=ucases, columns=res_m.columns), drop_c, unit)
    ab = addback_reference(cond, proto, vrows, case_of)
    if ab:
        e8 = pd.DataFrame({r_: ab[r_][0] for r_ in vrows}).T
        e8.columns = np.arange(1, e8.shape[1] + 1)
        d8 = pd.Series([ab[r_][1] for r_ in vrows], index=ucases)
        curves["greedy_expert_ab"] = pop_curve(pd.DataFrame(e8.values, index=ucases, columns=e8.columns), d8, unit)
    sr = R[R.fam == "static"].copy()
    sr["rescue"] = sr.delta - sr.delta_corrupt
    for o in C.STATIC_ORDERS:
        M = sr[sr.order == o].pivot_table(index="row_id", columns="k", values="rescue", aggfunc="first").loc[vrows]
        curves[f"static_{o}"] = pop_curve(pd.DataFrame(M.values, index=ucases, columns=M.columns), drop_c, unit)
    cs = {}
    for name, pc in curves.items():
        ks = np.array(pc.index, dtype=float)
        r = pc.ratio.values
        e = {"k": [int(k) for k in ks], "r": [float(v) for v in r], "lo": [float(v) for v in pc.lo.values], "hi": [float(v) for v in pc.hi.values],
             "auc_log_k15": auc_upto(ks, r, 15) if ks.max() >= 15 else None, "auc_log_k20": auc_upto(ks, r, 20) if ks.max() >= 20 else None,
             "k_q": {str(q): k_first(ks, r, q) for q in Q}, "max_r": float(r.max()), "k_max_r": int(ks[int(np.argmax(r))])}
        for kk in (1, 2, 5, 10, 15, 20, 32, 64, 128):
            if kk in pc.index:
                e[f"r{kk}"] = {"ratio": float(pc.ratio.loc[kk]), "lo": float(pc.lo.loc[kk]), "hi": float(pc.hi.loc[kk])}
        e["r_last"] = {"k": int(ks[-1]), "ratio": float(r[-1]), "lo": float(pc.lo.values[-1]), "hi": float(pc.hi.values[-1])}
        cs[name] = e
    out["curves"] = cs
    for kind in C.GREEDY_KINDS:
        res_m, val_m = G[kind]
        rr = res_m.div(drop_ref.loc[res_m.index], axis=0)
        ks = np.arange(1, res_m.shape[1] + 1)
        e = cs[f"greedy_{kind}"]
        e["case_k_q"] = {str(q): X.med_iqr(X.first_k(ks, rr.values, q)) for q in Q}
        e["case_k_restored"] = X.med_iqr(X.first_k(ks, val_m.values, 1e-9))
        e["frac_cases_restored"] = {str(k): float((val_m[k] > 0).mean()) for k in (1, 5, 10, 20) if k in val_m.columns}
        e["frac_rows_r_ge_0.9"] = {str(k): float((rr[k] >= 0.9).mean()) for k in (5, 10, 20) if k in rr.columns}
        comp = {}
        picks = {r_: st.greedy[(key, kind, r_)].S[:KG] for r_ in vrows}
        for kk in (1, 3, 5, 10, 20):
            nh = [sum(1 for c_ in p[:kk] if c_[0] == "h") for p in picks.values()]
            comp[str(kk)] = {"mean_heads": float(np.mean(nh)), "mean_experts": float(min(kk, np.mean([len(p[:kk]) for p in picks.values()])) - np.mean(nh)),
                             "frac_rows_first_pick_head": float(np.mean([p[0][0] == "h" for p in picks.values()])) if kk == 1 else None}
        lay_h = [c_[1] for p in picks.values() for c_ in p[:10] if c_[0] == "h"]
        lay_e = [c_[1] for p in picks.values() for c_ in p[:10] if c_[0] == "e"]
        comp["layers_first10"] = {"heads": X.med_iqr(lay_h) if lay_h else None, "experts": X.med_iqr(lay_e) if lay_e else None}
        e["composition"] = comp
        cnt, cnt5, cnt1 = {}, {}, {}
        for p in picks.values():
            for c_ in set(p[:10]):
                cnt[c_] = cnt.get(c_, 0) + 1
            for c_ in set(p[:5]):
                cnt5[c_] = cnt5.get(c_, 0) + 1
            cnt1[p[0]] = cnt1.get(p[0], 0) + 1
        e["recurring_first10"] = [{"comp": cname(c_), "kind": c_[0], "share": v / len(picks)} for c_, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:12]]
        e["recurring_first5"] = [{"comp": cname(c_), "kind": c_[0], "share": v / len(picks)} for c_, v in sorted(cnt5.items(), key=lambda kv: -kv[1])[:8]]
        e["first_picks"] = [{"comp": cname(c_), "kind": c_[0], "share": v / len(picks)} for c_, v in sorted(cnt1.items(), key=lambda kv: -kv[1])[:6]]
        if kind == "mix":
            gains_h, gains_e = [], []
            for r_ in vrows:
                g = st.greedy[(key, kind, r_)]
                rs = np.array(g.resc) / drop_ref.loc[r_]
                for j in range(1, min(KG, len(g.S))):
                    (gains_h if g.S[j][0] == "h" else gains_e).append(rs[j] - rs[j - 1])
            e["marginal_gain"] = {"heads": X.med_iqr(gains_h) if gains_h else None, "experts": X.med_iqr(gains_e) if gains_e else None}
    gr = R[R.fam.str.startswith("greedy")]
    for kind in C.GREEDY_KINDS:
        g20 = []
        idx = gr[gr.fam == f"greedy_{kind}"].set_index(["row_id", "set"])
        for r_ in vrows:
            g = st.greedy[(key, kind, r_)]
            s = C.set_str(tuple(g.S[:KG]))
            try:
                rk = idx.loc[(r_, s)].rank_true
                rk = float(rk.iloc[-1]) if hasattr(rk, "iloc") else float(rk)
            except KeyError:
                rk = np.nan
            g20.append(rk)
        cs[f"greedy_{kind}"]["frac_rows_top1_kmax"] = float(np.nanmean(np.array(g20) == 1))
    cl = pf0[(pf0.task == key) & (pf0.kind == "clean")]
    out["clean_top1_rate"] = float((cl.rank_true == 1).mean())
    return out


def analyse_group(proto, group, suffix=""):
    od = E13.step2_dir(proto, group, suffix)
    st = C.load_cstate(os.path.join(od, "circuit_state.pkl"))
    rows = C._cat(od, "circuit_rows_p")
    pf = C._cat(od, "circuit_prefill_p")
    pf0 = pf[pf.pass_ == pf.pass_.min()]
    meta = json.load(open(os.path.join(od, "run_meta.json")))
    sdc = pf[pf.kind == "clean"].groupby(["task", "case_id"]).delta.std()
    sdj = pf[pf.kind == "corrupt"].groupby(["task", "row_id"]).delta.std()
    res = {}
    for key in meta["tasks"]:
        log(f"analysing {key} ({group})")
        res[key] = analyse_task2(key, st, rows, pf0)
        res[key]["group"] = group
        res[key]["passes"] = int(pf.pass_.nunique())
        res[key]["complete"] = bool(meta.get("complete"))
        res[key]["prefill_sd"] = {"clean_median": float(sdc.loc[key].median()) if pf.pass_.nunique() > 1 else None,
                                  "corrupt_median": float(sdj.loc[key].median()) if pf.pass_.nunique() > 1 else None}
    notes = [n for n in st.notes if "pass" in n]
    return res, {"group": group, "proto": proto, "tasks": meta["tasks"], "passes": len(notes), "complete": bool(meta.get("complete")),
                 "pass_s_sum": float(sum(n.get("pass_s", 0) for n in notes)),
                 "peak_GB_max": float(max(n.get("peak_GB", 0) for n in notes)) if notes else None}


def fk(x):
    return "never" if x is None or not np.isfinite(x) else f"{int(x)}"


def tables2(res, tag=""):
    tb = []
    for key, r in res.items():
        cs, ce = r["curves"], r["ceilings"]
        for name in ("greedy_mix", "greedy_head", "greedy_expert_ab", "static_mix_oracle", "static_mix_dla", "static_head_oracle", "static_head_dla"):
            e = cs.get(name)
            if e is None:
                continue
            rk = lambda kk: f"{e[f'r{kk}']['ratio']:.3f}" if f"r{kk}" in e else ""
            tb.append({"Run": f"{r['label']}, {r['task']}", "Curve": name, "r(1)": rk(1), "r(5)": rk(5), "r(10)": rk(10),
                       "r(20) [95% CI]": f"{e['r20']['ratio']:.3f} [{e['r20']['lo']:.3f}, {e['r20']['hi']:.3f}]" if "r20" in e else
                       (f"{e['r15']['ratio']:.3f} (k=15)" if "r15" in e else ""),
                       "r(128)": rk(128), "max r (k)": f"{e['max_r']:.3f} ({e['k_max_r']})",
                       "k50/80/90 of drop": "/".join(fk(e["k_q"][str(q)]) for q in Q),
                       "AUC log k (1..20)": "" if e["auc_log_k20"] is None else f"{e['auc_log_k20']:.3f}",
                       "all-MoE ceiling": f"{ce['all_moe']['ratio']:.3f}"})
    save(pd.DataFrame(tb), f"ext13ci_circuit_curves{tag}")
    gt = []
    for key, r in res.items():
        for kind in C.GREEDY_KINDS:
            e = r["curves"][f"greedy_{kind}"]
            cq, comp = e["case_k_q"], e["composition"]
            gt.append({"Run": f"{r['label']}, {r['task']}", "Greedy": kind,
                       "row k50/80/90 (median; reached)": "/".join(fk(cq[str(q)]["median"]) for q in Q) + "; " +
                       "/".join(f"{cq[str(q)]['frac_finite']:.2f}" for q in Q),
                       "rows restored k=1/5/10/20": "/".join(f"{e['frac_cases_restored'].get(str(k), float('nan')):.2f}" for k in (1, 5, 10, 20)),
                       "rows r ≥ 0.9 k=5/10/20": "/".join(f"{e['frac_rows_r_ge_0.9'].get(str(k), float('nan')):.2f}" for k in (5, 10, 20)),
                       "rows top-1 at last step": f"{e['frac_rows_top1_kmax']:.2f}",
                       "heads among first 1/5/10/20": "/".join(f"{comp[str(k)]['mean_heads']:.1f}" for k in (1, 5, 10, 20)),
                       "first picks (share)": ", ".join(f"{x['comp']} {x['share']:.2f}" for x in e["first_picks"][:3]),
                       "most frequent in first 5": ", ".join(f"{x['comp']} {x['share']:.2f}" for x in e["recurring_first5"][:5])})
    save(pd.DataFrame(gt), f"ext13ci_circuit_greedy{tag}")
    sn = []
    for key, r in res.items():
        ce, sl = r["ceilings"], r.get("sanity_layer_heads", {})
        sn.append({"Run": f"{r['label']}, {r['task']}", "rows / units": f"{r['n_rows']} / {r['n_units']}",
                   "all heads (= 1)": f"{ce['all_heads']['ratio']:.3f}", "all attention": f"{ce['all_attn']['ratio']:.3f}",
                   "all MoE": f"{ce['all_moe']['ratio']:.3f} [{ce['all_moe']['lo']:.3f}, {ce['all_moe']['hi']:.3f}]",
                   "all heads: max |Δ − Δ_clean|": f"{r['sanity_all_heads']['max_abs_dev']:.3f}",
                   "layer heads vs attn_layer: median / max |ΔΔ|, r": f"{sl.get('median_abs_dev', float('nan')):.3f} / {sl.get('max_abs_dev', float('nan')):.3f}, {sl.get('r', float('nan')):.4f}",
                   "passes": r["passes"], "complete": "yes" if r["complete"] else "no"})
    save(pd.DataFrame(sn), f"ext13ci_circuit_sanity{tag}")


STY = {"greedy_mix": ("C3", "-", "greedy heads+experts"), "greedy_head": ("C0", "-", "greedy heads"),
       "greedy_expert_ab": ("C2", "-", "greedy experts (p5-addback)"), "static_mix_oracle": ("C3", "--", "oracle heads+experts"),
       "static_mix_dla": ("C3", ":", "DLA heads+experts"), "static_head_oracle": ("C0", "--", "oracle heads"),
       "static_head_dla": ("C0", ":", "DLA heads")}


def figures2(res, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = list(dict.fromkeys(r["cond"] for r in res.values()))
    for cond in conds:
        keys = [k for k in res if res[k]["cond"] == cond]
        n = len(keys)
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
        for ax, k in zip(axes[0], keys):
            r = res[k]
            for name, (col, ls, lab) in STY.items():
                e = r["curves"].get(name)
                if e is None:
                    continue
                ax.plot(e["k"], e["r"], color=col, ls=ls, lw=1.4 if name.startswith("greedy") else 1.0, label=lab)
                if name.startswith("greedy"):
                    ax.fill_between(e["k"], e["lo"], e["hi"], color=col, alpha=0.12)
            ax.axhline(1.0, color="k", lw=0.8)
            ax.axhline(r["ceilings"]["all_moe"]["ratio"], color="C2", lw=0.8, ls="-.", label="all-MoE ceiling")
            ax.set_xscale("log")
            ax.set_xlabel("components patched jointly (k)")
            ax.set_ylabel("r(k) = rescue / drop")
            ax.set_title(f"{r['label']}\n{r['task']}", fontsize=9)
            ax.set_ylim(min(-0.05, r["ceilings"]["all_moe"]["ratio"] - 0.05), 1.15)
        axes[0, 0].legend(fontsize=6, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"ext13ci_circuit_curves_{cond}{tag}.png"), dpi=130)
        plt.close(fig)
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.4), squeeze=False)
        for ax, k in zip(axes[0], keys):
            e = res[k]["curves"]["greedy_mix"]["composition"]
            ks = [1, 3, 5, 10, 20]
            h = [e[str(x)]["mean_heads"] for x in ks]
            x_ = [e[str(x)]["mean_experts"] for x in ks]
            ax.bar(range(len(ks)), h, color="C0", label="heads")
            ax.bar(range(len(ks)), x_, bottom=h, color="C2", label="experts")
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels([str(x) for x in ks])
            ax.set_xlabel("first k greedy picks")
            ax.set_ylabel("mean count")
            ax.set_title(f"{res[k]['label']}, {res[k]['task']}", fontsize=9)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"ext13ci_circuit_composition_{cond}{tag}.png"), dpi=130)
        plt.close(fig)
    log("step-2 figures written")


def overview(res2, ref, tag=""):
    """Cross-condition panel: heads among the first 10 mixed-greedy picks and r(20), per condition and model, with Phase 4's
    CounterFact / WinoGrande (option swap) / IOI values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for k, v in ref.items():
        m = "qwen3" if "qwen3" in k else "mixtral_bos"
        rows.append({"name": f"P4 {v['task'].replace(' STR', '').replace(' (S2 -> IO)', '')}", "proto": m, "heads10": v["heads_first10"],
                     "r20": v["r20"], "lo": v["r20_lo"], "hi": v["r20_hi"], "k80": v["k80"], "all_moe": v["all_moe"], "ref": True,
                     "direction": "ref", "order": -10 + list(ref).index(k)})
    for key, r in res2.items():
        cv = r["curves"]["greedy_mix"]
        rk = cv.get("r20") or cv["r_last"]
        rows.append({"name": r["cond"], "proto": r["proto"], "heads10": cv["composition"]["10"]["mean_heads"], "r20": rk["ratio"],
                     "lo": rk["lo"], "hi": rk["hi"], "k80": cv["k_q"]["0.8"], "all_moe": r["ceilings"]["all_moe"]["ratio"], "ref": False,
                     "direction": r["info"].get("direction", ""), "order": E13.COND_ORDER.index(r["cond"]) if r["cond"] in E13.COND_ORDER else 99})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # table
    tb = []
    for (name, ref_), g in df.sort_values("order").groupby(["name", "ref"], sort=False):
        e = {"Condition": name if ref_ else f"{name} ({E13.COND_INFO.get(name, {}).get('label', '')})"}
        for p in E13.PROTOS:
            x = g[g.proto == p]
            if len(x):
                x = x.iloc[0]
                e[f"{E13.MODEL_LABEL[p]}: heads in first 10 / r(20) [CI] / k80 / all-MoE"] = (
                    f"{x.heads10:.1f} / {x.r20:.2f} [{x.lo:.2f}, {x.hi:.2f}] / {fk(x.k80)} / {x.all_moe:.2f}")
            else:
                e[f"{E13.MODEL_LABEL[p]}: heads in first 10 / r(20) [CI] / k80 / all-MoE"] = ""
        tb.append(e)
    save(pd.DataFrame(tb), f"ext13ci_overview{tag}")
    names = list(dict.fromkeys(df.sort_values("order").name))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(names) + 1.8), sharey=True)
    y = np.arange(len(names))
    for j, (p, off, col) in enumerate((("qwen3", -0.18, "C0"), ("mixtral_bos", 0.18, "C1"))):
        sub = df[df.proto == p].set_index("name")
        yy = [y[names.index(nm)] + off for nm in sub.index]
        axes[0].barh(yy, sub.heads10.values, height=0.34, color=col, label=E13.MODEL_LABEL[p],
                     edgecolor=["k" if rf else col for rf in sub.ref.values])
        axes[1].errorbar(sub.r20.values, yy, xerr=[sub.r20.values - sub.lo.values, sub.hi.values - sub.r20.values], fmt="o", color=col,
                         ms=4, label=f"{E13.MODEL_LABEL[p]}: mixed greedy r(20)")
        axes[1].scatter(sub.all_moe.values, yy, marker="|", s=80, color=col, alpha=0.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([n if n.startswith("P4") else f"{n}: {E13.COND_INFO.get(n, {}).get('label', '')}" for n in names], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("heads among the first 10 mixed-greedy picks")
    axes[0].set_xlim(0, 10)
    axes[0].axvline(5, color="k", lw=0.5, ls=":")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].set_xlabel("r(20) of the drop (dots, 95% CI) | all-MoE ceiling (ticks)")
    axes[1].axvline(1, color="k", lw=0.5)
    axes[1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Heads + experts add-back at the final position: Phase-5 conditions vs Phase 4 (CounterFact / WinoGrande option swap / IOI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"ext13ci_overview{tag}.png"), dpi=140)
    plt.close(fig)
    log("overview written")
    return df


# ----------------------------------------------------------------------------------------------------------------------
# contrasts (paired over shared clusters: WinoGrande twins / template seeds)
# ----------------------------------------------------------------------------------------------------------------------
CONTRASTS = (
    ("dir_f1", "dir_opt", "direction on identical twins: attribute→entity (F1) − entity→attribute (option swap)"),
    ("vt_a2e_orig", "vt_e2a_orig", "direction on the verb templates, original verb (a→e − e→a)"),
    ("vt_a2e_neu", "vt_e2a_neu", "direction on the verb templates, neutral verb (a→e − e→a)"),
    ("cp_vt_orig", "vt_e2a_orig", "copy availability on the templates, original verb (C2 − C1)"),
    ("cp_vt_neu", "vt_e2a_neu", "copy availability on the templates, neutral verb (C2 − C1)"),
    ("cp_nat", "P3wino", "copy availability on natural pairs (C2 − Phase-3 option swap C1)"),
    ("vt_a2e_orig", "cp_vt_orig", "direction with copy fixed, original verb (C3 − C2)"),
    ("vt_a2e_neu", "cp_vt_neu", "direction with copy fixed, neutral verb (C3 − C2)"),
    ("dir_f1", "cp_nat", "direction with copy fixed on natural pairs (C3 − C2)"),
    ("f1", "P3wino", "F1 attribute→entity vs Phase-3 option swap (shared twins only)"),
)


def row_table(key, cur, suffix=""):
    """Per validation row of a Phase-5 condition (or 'P3wino/<proto>' = Phase 4's WinoGrande option-swap rows): cluster label,
    drop, mixed-greedy rescue at k = 10 / 20, heads among the first 10 picks."""
    cond, proto = E13.parse_key(key)
    if cond == "P3wino":
        model = E13.MODEL_OF[proto]
        od = os.path.join(RESULTS, f"{model}_circuit")
        sp = os.path.join(od, "circuit_state.pkl")
        if not os.path.exists(sp):
            return None
        from moetrace import ext7_pairs as P
        st = C.load_cstate(sp)
        tk = f"wino_{model}"
        t = C.load_task(tk)
        cfg = C.cfg(tk)
        pairs = P.load_pairs(os.path.join(E13.ROOT, cfg["pairs"]), json.load(open(os.path.join(E13.ROOT, cfg["case_sets"]))))
        lab = lambda c: str(pairs.loc[int(c) // 2].q_a).rsplit("-", 1)[0]
        pf = C._cat(od, "circuit_prefill_p")
    else:
        g = next((v["group"] for k_, v in cur.get("step2_runs", {}).items() if key in v.get("tasks", [])), None)
        if g is None:
            return None
        od = E13.step2_dir(proto, g, suffix)
        st = C.load_cstate(os.path.join(od, "circuit_state.pkl"))
        tk = key
        t = E13.load_task(cond, proto)
        ul = dict(zip(t.cases.case_id, t.cases.unit_label))
        lab = lambda c: ul[int(c)]
        pf = C._cat(od, "circuit_prefill_p")
    p0 = pf[(pf.pass_ == pf.pass_.min()) & (pf.task == tk)]
    dcl = p0[p0.kind == "clean"].set_index("case_id").delta
    dco = p0[p0.kind == "corrupt"].set_index("row_id").delta
    out = []
    for r_ in sorted(int(x) for x in p0[p0.kind == "corrupt"].row_id):
        g = st.greedy.get((tk, "mix", r_))
        if g is None or not g.resc:
            continue
        c = int(t.rows.case_id.iloc[r_])
        res = list(g.resc[:KG])
        out.append({"row_id": r_, "case_id": c, "cluster": lab(c), "drop": float(dcl.loc[c] - dco.loc[r_]),
                    "resc10": res[min(10, len(res)) - 1], "resc20": res[-1], "heads10": sum(1 for x in g.S[:10] if x[0] == "h")})
    return pd.DataFrame(out)


def contrasts(cur, suffix=""):
    out = []
    rng = np.random.default_rng(0)
    for a, b, lab in CONTRASTS:
        for proto in E13.PROTOS:
            ka, kb = E13.key_of(a, proto), E13.key_of(b, proto)
            if ka not in cur.get("step2", {}) or (b != "P3wino" and kb not in cur.get("step2", {})):
                continue
            A, B = row_table(ka, cur, suffix), row_table(kb, cur, suffix)
            if A is None or B is None or A.empty or B.empty:
                continue
            ga = A.groupby("cluster").agg(resc20=("resc20", "sum"), resc10=("resc10", "sum"), drop=("drop", "sum"), heads10=("heads10", "mean"))
            gb = B.groupby("cluster").agg(resc20=("resc20", "sum"), resc10=("resc10", "sum"), drop=("drop", "sum"), heads10=("heads10", "mean"))
            sh = sorted(set(ga.index) & set(gb.index))
            e = {"a": a, "b": b, "proto": proto, "label": lab, "n_clusters_a": int(len(ga)), "n_clusters_b": int(len(gb)),
                 "n_shared": int(len(sh)), "rows_a": int(len(A)), "rows_b": int(len(B))}
            for nm, fa, fb in (("r20", ga.resc20 / ga["drop"], gb.resc20 / gb["drop"]), ("r10", ga.resc10 / ga["drop"], gb.resc10 / gb["drop"]),
                               ("heads10", ga.heads10, gb.heads10)):
                e[f"{nm}_a_all"] = float(fa.mean() if nm == "heads10" else ga[nm.replace('r', 'resc')].sum() / ga["drop"].sum())
                e[f"{nm}_b_all"] = float(fb.mean() if nm == "heads10" else gb[nm.replace('r', 'resc')].sum() / gb["drop"].sum())
                if len(sh) >= 8:
                    d = (fa.loc[sh] - fb.loc[sh]).values
                    bs = d[rng.integers(0, len(d), size=(5000, len(d)))].mean(1)
                    e[f"{nm}_diff"] = {"mean": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                                       "a_shared": float(fa.loc[sh].mean()), "b_shared": float(fb.loc[sh].mean())}
            out.append(e)
    return out


def tables_contrasts(cs, tag=""):
    rows = []
    for e in cs:
        f = lambda nm: (f"{e[nm + '_diff']['a_shared']:.2f} vs {e[nm + '_diff']['b_shared']:.2f}: {e[nm + '_diff']['mean']:+.2f} "
                        f"[{e[nm + '_diff']['lo']:+.2f}, {e[nm + '_diff']['hi']:+.2f}]") if nm + "_diff" in e else \
            f"{e[nm + '_a_all']:.2f} vs {e[nm + '_b_all']:.2f} (unpaired)"
        rows.append({"Contrast": e["label"], "Model": E13.MODEL_LABEL[e["proto"]], "A − B": f"{e['a']} − {e['b']}",
                     "shared clusters (A / B)": f"{e['n_shared']} ({e['n_clusters_a']} / {e['n_clusters_b']})",
                     "r(20): A vs B: diff [95% CI]": f("r20"), "r(10)": f("r10"), "heads among first 10": f("heads10")})
    if rows:
        save(pd.DataFrame(rows), f"ext13ci_contrasts{tag}")


def all_keys(step, suffix=""):
    regs = E13.load_conditions()
    out = []
    for c in list(E13.COND_ORDER) + sorted({c for c, _ in regs if c not in E13.COND_ORDER}):
        for p in E13.PROTOS:
            if (c, p) not in regs:
                continue
            if step == 1 and E13.step1_complete(c, p, suffix):
                out.append(E13.key_of(c, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["1", "2", "all"])
    ap.add_argument("--keys", default=None, help="comma list of cond/proto (step 1; default: every complete step-1 run)")
    ap.add_argument("--groups", default=None, help="comma list of <proto>/<group> (step 2; default: every step-2 run)")
    ap.add_argument("--suffix", default="", help="run-dir suffix (smoke)")
    ap.add_argument("--tag", default=None, help="suffix of tables / figures / summary (default = --suffix)")
    args = ap.parse_args()
    tag = args.suffix if args.tag is None else args.tag
    spath = E13.SUMMARY if not tag else os.path.join(RESULTS, f"ext13ci_summary{tag}.json")
    import fcntl
    lockf = open(spath + ".lock", "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)  # the two per-model chains may analyse at the same time
    cur = json.load(open(spath)) if os.path.exists(spath) else {}
    if args.step in ("1", "all"):
        keys = args.keys.split(",") if args.keys else all_keys(1, args.suffix)
        res, arrs = {}, {}
        for k in keys:
            c, p = E13.parse_key(k)
            if not E13.step1_complete(c, p, args.suffix):
                log(f"{k}: step 1 incomplete, skipped")
                continue
            log(f"step 1: {k}")
            res[k], arrs[k] = analyse1(c, p, args.suffix)
        if res:
            cur.setdefault("step1", {}).update(res)
            figures1(res, arrs, tag)
        if cur.get("step1"):
            order = [E13.key_of(c, p) for c in list(E13.COND_ORDER) + ["F1pre"] for p in E13.PROTOS]
            allres = {k: cur["step1"][k] for k in order if k in cur["step1"]}
            allres.update({k: v for k, v in cur["step1"].items() if k not in allres})
            tables1(allres, tag)
    if args.step in ("2", "all"):
        base = os.path.join(RESULTS, f"p5_{{}}_ci{args.suffix}")
        want = set(args.groups.split(",")) if args.groups else None
        res2, runs = {}, {}
        for p in E13.PROTOS:
            d = base.format(p)
            if not os.path.isdir(d):
                continue
            for g in sorted(os.listdir(d)):
                if want and f"{p}/{g}" not in want:
                    continue
                if os.path.exists(os.path.join(d, g, "circuit_state.pkl")):
                    r, info = analyse_group(p, g, args.suffix)
                    res2.update(r)
                    runs[f"{p}/{g}"] = info
        cur.setdefault("step2", {}).update(res2)
        cur.setdefault("step2_runs", {}).update(runs)
        if res2:
            figures2(res2, tag)
        order = [E13.key_of(c, p) for c in list(E13.COND_ORDER) + ["F1pre"] for p in E13.PROTOS]
        all2 = {k: cur["step2"][k] for k in order if k in cur["step2"]}
        if all2:
            tables2(all2, tag)
            ref = E13.ext10_reference()
            cur["reference"] = ref
            overview(all2, ref, tag)
            try:
                cur["contrasts"] = contrasts(cur, args.suffix)
                tables_contrasts(cur["contrasts"], tag)
            except Exception as ex:
                log(f"contrasts failed: {ex!r}")
    cur["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(spath, "w") as f:
        json.dump(cur, f, indent=1, default=float)
    log(f"wrote {spath}")


if __name__ == "__main__":
    main()
