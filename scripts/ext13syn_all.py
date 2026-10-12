"""ext13syn (Phase 5, agent p5-synth): the two cross-condition syntheses, re-runnable in one command (CPU only).

  Task A  the task ladder (RESEARCH_PLAN.md "P5-2"): one table / figure across tasks and both models, ordered by the
          attention share; paired contrasts on identical items with cluster-bootstrap CIs; association strata and the
          probe score within levels; the confounds of the ladder.
  Task B  why the add-back curves differ ("P5-4"): overshoot of the expert add-back curve over the all-MoE ceiling against
          the cross-fitted negative single-expert mass, its layers and direct (DLA) share, the late-MoE term, attention
          shares and writer bands; per-row analysis; the compensation regression; the optional GPU check
          (scripts/ext13syn_gpucheck.py, read if present).

Reads whatever exists (p5_conditions.json status, run dirs, summaries) and skips missing conditions; f2e and later
conditions appear automatically when their base / add-back runs are complete.

Usage: python scripts/ext13syn_all.py [--parts ladder,curves,text] [--nb 5000]
Outputs: results/ext13syn_summary.json; results/tables/ext13syn_ladder*.{md,csv}, ext13syn_curves_*.{md,csv};
results/figures/ext13syn_ladder*.{png,pdf}, ext13syn_curves_*.{png,pdf}; results/sections/ext13_ladder.md, ext13_curves.md.
"""
import argparse, json, os, sys, time, warnings
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from moetrace import ext13syn as S
from moetrace.models import RESULTS

TAB, FIG, SEC = os.path.join(RESULTS, "tables"), os.path.join(RESULTS, "figures"), os.path.join(RESULTS, "sections")
SUMM = os.path.join(RESULTS, "ext13syn_summary.json")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def save_table(df: pd.DataFrame, name: str, caption: str = "") -> str:
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    cols = list(df.columns)
    lines = ([f"**{caption}**", ""] if caption else []) + ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and not np.isfinite(v)) else str(v) for v in r.values) + " |")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(TAB, name + ".md"), "w").write(txt)
    return txt


def fci(x, d=2, sign=False):
    if x is None or x[0] is None or not np.isfinite(x[0]):
        return "n/a"
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    if len(x) < 3 or x[1] is None or not np.isfinite(x[1]):
        return f.format(x[0])
    return f"{f.format(x[0])} [{f.format(x[1])}, {f.format(x[2])}]"


# ==================================================================================================================
# Task A: ladder
# ==================================================================================================================
_CI = {}


def ci_rows(key):
    """Per-row heads among the first 10 mixed greedy picks (p5-circuit's analysis code, imported read-only)."""
    if "mod" not in _CI:
        try:
            import ext13ci_analyze as CI
            _CI["mod"] = CI
            _CI["cur"] = json.load(open(os.path.join(RESULTS, "ext13ci_summary.json")))
        except Exception as e:  # noqa
            log("circuit rows unavailable:", e)
            _CI["mod"] = None
    if _CI["mod"] is None:
        return None
    try:
        return _CI["mod"].row_table(key, _CI["cur"])
    except Exception as e:  # noqa
        log(f"row_table({key}) failed: {e}")
        return None


def level_available(level, proto, reg):
    if level == "cf":
        return S.cf_reference(proto) is not None
    if level in S.P5_CONDS:
        return S.base_ready(level, proto, reg)
    run = S.run_of(level, proto)
    return run is not None and os.path.exists(os.path.join(RESULTS, run, "str_sweep_rows.parquet"))


def pairdata(level, proto, splits=("validation",), family="main"):
    d = S.PairData(S.run_of(level, proto), family, splits, S.pairs_file(level, proto), proto)
    if level == "p3_opt":
        d.set_local_assoc_from_scan(f"wino_{proto}")
    return d


def heads_key(level, proto):
    if level == "p3_opt":
        return f"P3wino/{proto}"
    if level in S.P5_CONDS:
        return f"{level}/{proto}"
    return None


def part_ladder(nb):
    reg = S.registry()
    kt = S.ko_table()
    rows, cache = [], {}
    for level, meta in S.LEVELS.items():
        for proto in S.PROTOS:
            if not level_available(level, proto, reg):
                continue
            log(f"ladder {level}/{proto}")
            base = {"level": level, "proto": proto, "model": S.MSHORT[proto], **{k: meta[k] for k in ("label", "direction", "fmt", "answer", "ctx", "items")}}
            if level == "cf":
                s = S.cf_reference(proto)
                h = S.ci_reference("cf_qwen3" if proto == "qwen3" else "cf_mixtral")
                s["heads10"] = [h, np.nan, np.nan] if h is not None else None
                own = {"expert": S.CF_EXPERT[proto], "F": S.ko_phase4(proto, "cf", S.CF_EXPERT[proto]), "pattern": "A" if proto == "qwen3" else "A (tie L19-L21)"}
                wg = {"expert": S.WG_EXPERT[proto], "F": S.ko_phase4(proto, "cf", S.WG_EXPERT[proto])}
                rows.append({**base, "stratum": "all", **s, "own": own, "wg": wg})
                continue
            d = pairdata(level, proto)
            hk = heads_key(level, proto)
            if hk:
                d.attach_heads(ci_rows(hk))
            elif level == "ioi" and proto == "qwen3":
                pass
            cache[(level, proto)] = d
            s = S.summarise(d, nb=nb)
            if level == "ioi":
                h = S.ci_reference("ioi_qwen3") if proto == "qwen3" else None
                s["heads10"] = [h, np.nan, np.nan] if h is not None else None
                own = {"expert": "none (no expert pass)", "F": None, "pattern": ""}
                wg = {"expert": S.WG_EXPERT[proto], "F": S.ko_phase4(proto, "ioi", S.WG_EXPERT[proto])}
            elif level == "p3_opt":
                F = S.ko_phase4(proto, "wg_margin", S.WG_EXPERT[proto])
                own = {"expert": S.WG_EXPERT[proto], "F": F, "pattern": "A"}
                wg = {"expert": S.WG_EXPERT[proto], "F": F}
            else:
                e = reg.get(level, {}).get(proto, {})
                se = e.get("selected_expert") or {}
                k = S.ko_lookup(kt, level, proto)
                own = {"expert": se.get("name", "?"), "pattern": se.get("pattern", ""), "F": (k.get("own") or {}).get("F"),
                       "spec": se.get("spec")}
                wg = {"expert": S.WG_EXPERT[proto], "F": (k.get("wg") or {}).get("F")}
            rows.append({**base, "stratum": "all", **s, "unit_name": unit_name(level), "own": own, "wg": wg})
            if level in S.STRATA_ROWS:
                for lab, msk in (("names", d.names), ("objects", ~d.names)):
                    if msk.sum() >= 10 and (~msk).sum() >= 1:
                        ss = S.summarise(d, mask=msk, nb=nb)
                        rows.append({**base, "stratum": lab, **ss, "unit_name": unit_name(level), "own": None, "wg": None})
    # ordering key: the attention share's percentile rank within each model (0 = most MoE-heavy level of that model,
    # 1 = most attention-heavy), averaged over the models that have the level (a level present in one model only is
    # placed by that model's rank, not by its raw share, since the models' shares differ in scale)
    key, share_mean = {}, {}
    for proto in S.PROTOS:
        sh = {r["level"]: r["share"][0] for r in rows if r["proto"] == proto and r["stratum"] == "all" and r.get("share")}
        lv_sorted = sorted(sh, key=lambda k: sh[k])
        n = len(lv_sorted)
        for i, lv in enumerate(lv_sorted):
            key.setdefault(lv, []).append(i / (n - 1) if n > 1 else 0.0)
    for lv in list(key):
        key[lv] = float(np.mean(key[lv]))
        share_mean[lv] = float(np.mean([r["share"][0] for r in rows if r["level"] == lv and r["stratum"] == "all"]))
    order = sorted(key, key=lambda k: (key[k], share_mean[k]))
    return {"rows": rows, "key": key, "share_mean": share_mean, "order": order,
            "key_definition": "mean over models of the within-model percentile rank of the attention share (AUC+)"}, cache


def unit_name(level):
    if level.startswith("vt_") or level.startswith("cp_vt"):
        return "seed"
    if level == "cp_nat":
        return "twin"
    return "pair"


def part_contrasts(cache, nb):
    out = []
    for a, b, lab, kind in S.CONTRASTS:
        for proto in S.PROTOS:
            if (a, proto) not in cache or (b, proto) not in cache:
                continue
            log(f"contrast {a} - {b} / {proto}")
            e = {"a": a, "b": b, "label": lab, "kind": kind, "proto": proto, "model": S.MSHORT[proto]}
            r = S.paired(cache[(a, proto)], cache[(b, proto)], nb=nb)
            e["validation"] = r
            try:
                da = pairdata(a, proto, ("discovery", "validation"))
                db = pairdata(b, proto, ("discovery", "validation"))
                e["all"] = S.paired(da, db, nb=nb)
            except Exception as ex:  # noqa
                e["all"] = None
                log("all-split contrast failed:", ex)
            out.append(e)
    return out


def part_strata(nb):
    """Within-level association strata (pre-registered probe rule), names / objects, Phase-3 local association, and the
    per-pair probe-score regression; all splits and families of a level pooled (no selection involved)."""
    reg = S.registry()
    out = []
    specs = []
    for proto in S.PROTOS:
        # pre-neutral fill-in-the-blank runs (all association strata; kept as a record by p5-data)
        if os.path.exists(os.path.join(RESULTS, f"wino_ref_F1_{proto}", "str_sweep_rows.parquet")):
            specs.append(("f1pre", proto, f"wino_ref_F1_{proto}", ("main", "rep"), os.path.join(S.ROOT, f"data/wino_ref/pairs_p5_f1_{proto}.parquet")))
        if os.path.exists(os.path.join(RESULTS, f"wino_{proto}_str", "str_sweep_rows.parquet")):
            specs.append(("p3_opt", proto, f"wino_{proto}_str", ("main",), None))
        for c in ("dir_f1", "dir_opt", "cp_nat", "f1", "vt_e2a_orig", "vt_e2a_neu", "cp_vt_orig", "cp_vt_neu", "vt_a2e_orig", "f2e"):
            if S.base_ready(c, proto, reg):
                specs.append((c, proto, S.run_of(c, proto), ("main", "rep"), S.pairs_file(c, proto)))
    for lev, proto, run, fams, pf in specs:
        log(f"strata {lev}/{proto}")
        try:
            d = S.PairData(run, fams, ("discovery", "validation"), pf, proto)
        except Exception as ex:  # noqa
            log("  failed:", ex)
            continue
        if lev == "p3_opt":
            d.set_local_assoc_from_scan(f"wino_{proto}")
        e = {"level": lev, "proto": proto, "model": S.MSHORT[proto], "run": run, "n_pairs": d.n, "n_units": int(len(set(d.unit)))}
        if lev != "p3_opt" and len(set(d.stratum)) > 1:
            e["assoc"] = S.strata_compare(d, d.stratum, "neutral", nb=nb)
        if d.local_assoc is not None and d.local_assoc.any():
            lab = np.where(d.local_assoc, "local assoc", "not local assoc").astype(object)
            e["local"] = S.strata_compare(d, lab, "not local assoc", nb=nb)
        if d.names.any() and (~d.names).any():
            lab = np.where(d.names, "names", "objects").astype(object)
            e["names"] = S.strata_compare(d, lab, "objects", nb=nb)
        e["probe"] = S.probe_slope(d, nb=nb)
        e["score_source"] = "Phase-3 local-context baseline (d_local)" if lev == "p3_opt" else "association probes " + "+".join(S.R_PROBES[proto])
        out.append(e)
    return out


def confounds(lad):
    rows = [r for r in lad["rows"] if r["stratum"] == "all"]
    out = {}
    for proto in S.PROTOS:
        rr = [r for r in rows if r["proto"] == proto and r.get("T_median") is not None and np.isfinite(r.get("T_median") or np.nan)]
        if len(rr) < 4:
            continue
        sh = np.array([r["share"][0] for r in rr])
        e = {"n_levels": len(rr)}
        for k in ("T_median", "names_frac"):
            x = np.array([r[k] for r in rr], dtype=float)
            e[f"spearman_share_{k}"] = float(pd.Series(sh).corr(pd.Series(x), method="spearman"))
        x = np.array([r["drop"][0] for r in rr], dtype=float)
        e["spearman_share_drop"] = float(pd.Series(sh).corr(pd.Series(x), method="spearman"))
        ctx = np.array([r["ctx"] == "yes" for r in rr])
        e["share_ctx_yes_mean"] = float(sh[ctx].mean()) if ctx.any() else None
        e["share_ctx_no_mean"] = float(sh[~ctx].mean()) if (~ctx).any() else None
        a2e = np.array([r["direction"] == "attribute -> entity" for r in rr])
        e["share_a2e_mean"] = float(sh[a2e].mean()) if a2e.any() else None
        out[proto] = e
    # cross-model agreement of the level order
    sh = {}
    for r in rows:
        sh.setdefault(r["level"], {})[r["proto"]] = r["share"][0]
    both = [k for k, v in sh.items() if len(v) == 2]
    if len(both) >= 4:
        out["spearman_qwen3_mixtral"] = float(pd.Series([sh[k]["qwen3"] for k in both]).corr(pd.Series([sh[k]["mixtral_bos"] for k in both]), method="spearman"))
        out["n_levels_both"] = len(both)
    return out


# ------------------------------------------------------------------------------------------------------------------
def partners(lv):
    out = []
    for a, b, lab, kind in S.CONTRASTS:
        if lv == a:
            out.append(f"{b} ({kind})")
        elif lv == b:
            out.append(f"{a} ({kind})")
    return ", ".join(out)


def ladder_tables(lad, con, strata, conf):
    order = lad["order"]
    rank = {lv: i + 1 for i, lv in enumerate(order)}
    recs, recs_csv = [], []
    for lv in order:
        for strat in ("all", "objects", "names"):
            for proto in S.PROTOS:
                r = next((x for x in lad["rows"] if x["level"] == lv and x["stratum"] == strat and x["proto"] == proto), None)
                if r is None:
                    continue
                own, wg = r.get("own") or {}, r.get("wg") or {}
                lvname = S.LEVELS[lv]["label"] + ("" if strat == "all" else f" · {strat}")
                ownf = f"{own.get('expert', '')} {('(' + own['pattern'] + ')') if own.get('pattern') else ''}: {fci(own.get('F'), 3, True)}" if own else ""
                wgf = f"{wg.get('expert', '')}: {fci(wg.get('F'), 3, True)}" if wg else ""
                h = r.get("heads10")
                recs.append({"Rank": rank[lv] if strat == "all" else "", "Level": lvname, "Model": r["model"],
                             "Pairs (units)": f"{r.get('n_pairs')} ({r.get('n_units')})",
                             "Attention share (AUC+)": fci(r.get("share")), "A_direct / drop": fci(r.get("A_direct")),
                             "All-MoE M / drop": fci(r.get("M_all")), "Heads in first 10 mixed picks": fci(h, 1) if h else "n/a",
                             "Own expert: knockout F": ownf, "WG expert: knockout F": wgf,
                             "Mean drop": f"{r['drop'][0]:.2f}" if r.get("drop") else "", "T median": f"{r['T_median']:.0f}" if r.get("T_median") and np.isfinite(r["T_median"]) else "n/a",
                             "Names": f"{r['names_frac']:.2f}" if r.get("names_frac") is not None else "",
                             "Paired with (identical items)": partners(lv) if strat == "all" else ""})
                c = {"rank": rank[lv], "level": lv, "stratum": strat, "proto": proto, "direction": r["direction"], "format": r["fmt"],
                     "answer": r["answer"], "answer_in_context": r["ctx"], "items": r["items"], "n_pairs": r.get("n_pairs"), "n_units": r.get("n_units"),
                     "order_key": lad["key"][lv], "share_mean_models": lad.get("share_mean", {}).get(lv), "mean_drop": (r.get("drop") or [np.nan])[0], "T_median": r.get("T_median"), "names_frac": r.get("names_frac"),
                     "own_expert": own.get("expert"), "own_pattern": own.get("pattern"), "wg_expert": wg.get("expert")}
                for k in ("share", "A_direct", "M_all", "heads10", "auc_attn"):
                    v = r.get(k) or [np.nan] * 3
                    c[k], c[k + "_lo"], c[k + "_hi"] = v[0], v[1], v[2]
                for nm, src in (("own_F", own.get("F")), ("wg_F", wg.get("F"))):
                    v = src or [np.nan] * 3
                    c[nm], c[nm + "_lo"], c[nm + "_hi"] = v[0], v[1], v[2]
                recs_csv.append(c)
    md = save_table(pd.DataFrame(recs), "ext13syn_ladder",
                    "Task ladder (validation pairs; ordered by the attention share's within-model rank averaged over the two models; fractions of the drop; 95% cluster-bootstrap CIs; heads for CounterFact / IOI are Phase-4 point values)")
    pd.DataFrame(recs_csv).to_csv(os.path.join(TAB, "ext13syn_ladder.csv"), index=False)  # numeric CSV replaces the text one
    # contrasts
    cr = []
    for e in con:
        for split in ("validation", "all"):
            r = e.get(split)
            if not r:
                continue
            row = {"Contrast (a − b)": f"{e['a']} − {e['b']}", "Factor": e["kind"], "Model": e["model"], "Split": split,
                   "Shared items (units)": f"{r['n_items']} ({r['n_units']})" + (" *" if r["n_units"] < 8 else "")}
            for k, lab in (("share", "Δ attention share"), ("A_direct", "Δ A_direct"), ("M_all", "Δ all-MoE M"), ("heads10", "Δ heads in first 10")):
                if k in r and (split == "validation" or k != "heads10"):
                    row[lab] = f"{r[k]['a'][0]:.2f} vs {r[k]['b'][0]:.2f}: {fci(r[k]['diff'], 2, True)}"
                else:
                    row[lab] = ""
            cr.append(row)
    cmd = save_table(pd.DataFrame(cr), "ext13syn_ladder_contrasts",
                     "Paired contrasts on identical items (a − b; same unit resamples for both conditions; validation = the ladder's split, all = discovery + validation; * = fewer than 8 shared units, CI unreliable)")
    # strata
    sr = []
    for e in strata:
        for fam, ref in (("assoc", "neutral"), ("local", "not local assoc"), ("names", "objects")):
            if fam not in e:
                continue
            for lab, v in e[fam].items():
                d = v.get("diff_vs_ref", {})
                sr.append({"Level": e["level"], "Model": e["model"], "Strata": fam, "Stratum": lab, "Pairs (units)": f"{v['n_pairs']} ({v['n_units']})",
                           "Attention share": fci(v.get("share")), "A_direct": fci(v.get("A_direct")), "All-MoE M": fci(v.get("M_all")),
                           f"Δ share vs {ref}": fci(d.get("share"), 2, True) if d else "(reference)" if lab == ref else "",
                           f"Δ A_direct vs ref": fci(d.get("A_direct"), 2, True) if d else ""})
    smd = save_table(pd.DataFrame(sr), "ext13syn_ladder_strata",
                     "Within-level strata (all splits and families pooled): association probe strata, Phase-3 local association, names / objects")
    pr = []
    for e in strata:
        p = e.get("probe")
        if not p:
            continue
        pr.append({"Level": e["level"], "Model": e["model"], "Score": e["score_source"], "Pairs (units)": f"{p['n_pairs']} ({p['n_units']})",
                   "Score SD": f"{p['score_sd']:.2f}", "Share |s| ≥ 1": f"{p['frac_abs_ge1']:.2f}",
                   "Slope of A_direct share on s": fci(p["slope_signed"], 3, True), "r (signed)": fci(p["r_signed"], 2, True),
                   "Slope on |s|": fci(p["slope_abs"], 3, True), "r (|s|)": fci(p["r_abs"], 2, True)})
    pmd = save_table(pd.DataFrame(pr), "ext13syn_ladder_probe",
                     "Per-pair direct attention share regressed on the own-answer association score s (positive = the trigger alone points to the clean answer)")
    return md, cmd, smd, pmd


def ladder_figure(lad, con):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = lad["order"]
    rows = {(r["level"], r["proto"]): r for r in lad["rows"] if r["stratum"] == "all"}
    panels = [("share", "attention share (AUC+)"), ("A_direct", "direct attention / drop"), ("M_all", "all-MoE M / drop"), ("heads10", "heads in first 10 picks")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.42 * len(order) + 1.8), sharey=True)
    col = {"qwen3": "#1f6fb4", "mixtral_bos": "#d1612a"}
    mk = {"qwen3": "o", "mixtral_bos": "s"}
    for ax, (k, lab) in zip(axes, panels):
        for i, lv in enumerate(order):
            for j, proto in enumerate(S.PROTOS):
                r = rows.get((lv, proto))
                if r is None or not r.get(k) or r[k][0] is None or not np.isfinite(r[k][0]):
                    continue
                v = r[k]
                y = i + (-0.12 if j == 0 else 0.12)
                xerr = None if (v[1] is None or v[2] is None or not np.isfinite(v[1])) else [[v[0] - v[1]], [v[2] - v[0]]]
                ax.errorbar(v[0], y, xerr=xerr, fmt=mk[proto], color=col[proto], ms=5, lw=1, capsize=0,
                            label=S.MSHORT[proto] if (i == 0 or lv == order[0]) else None)
        ax.set_xlabel(lab)
        ax.grid(axis="x", alpha=0.3)
        if k in ("A_direct", "M_all"):
            ax.axvline(0, color="grey", lw=0.6)
            ax.axvline(1, color="grey", lw=0.6, ls=":")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([S.LEVELS[lv]["label"] for lv in order], fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    axes[0].legend(uniq.values(), uniq.keys(), fontsize=8, loc="lower right")
    fig.suptitle("Task ladder, ordered by the attention share (within-model rank, averaged over models; validation pairs, 95% cluster-bootstrap CIs)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext13syn_ladder.{ext}"), dpi=150)
    plt.close(fig)
    # paired contrasts forest plot
    items = [(e, sp) for e in con for sp in ("validation",) if e.get(sp)]
    if not items:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.38 * len(items) + 1.6), sharey=True)
    for ax, (k, lab) in zip(axes, panels[:3]):
        for i, (e, sp) in enumerate(items):
            r = e[sp]
            if k not in r:
                continue
            v = r[k]["diff"]
            if v[0] is None:
                continue
            xe = None if (v[1] is None or v[2] is None) else [[v[0] - v[1]], [v[2] - v[0]]]
            ax.errorbar(v[0], i, xerr=xe, fmt=mk[e["proto"]], color=col[e["proto"]], ms=5, lw=1, alpha=1.0 if r["n_units"] >= 8 else 0.4)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("Δ " + lab)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(items)))
    axes[0].set_yticklabels([f"{e['a']} − {e['b']} ({e['model']}, {e['validation']['n_items']} items)" for e, sp in items], fontsize=8)
    fig.suptitle("Paired contrasts on identical items (validation; cluster bootstrap over shared units; faint = fewer than 8 shared units)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext13syn_ladder_contrasts.{ext}"), dpi=150)
    plt.close(fig)


# ==================================================================================================================
# Task B: curves
# ==================================================================================================================
def part_curves(lad, nb):
    from moetrace import ext13syn_curves as C
    import ext13syn_curves_report as CR
    return CR.run(C, lad, nb, log)


# ==================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ladder,curves,text")
    ap.add_argument("--nb", type=int, default=S.NB)
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    os.makedirs(TAB, exist_ok=True), os.makedirs(FIG, exist_ok=True), os.makedirs(SEC, exist_ok=True)
    summ = json.load(open(SUMM)) if os.path.exists(SUMM) else {}
    if "ladder" in parts:
        lad, cache = part_ladder(args.nb)
        con = part_contrasts(cache, args.nb)
        strata = part_strata(args.nb)
        conf = confounds(lad)
        summ.update(clean({"ladder": lad, "contrasts": con, "strata": strata, "confounds": conf}))
        json.dump(summ, open(SUMM, "w"), indent=1)
        log("ladder computed")
    if "curves" in parts:
        cur = part_curves(summ.get("ladder"), args.nb)
        summ["curves"] = clean(cur)
        json.dump(summ, open(SUMM, "w"), indent=1)
        log("curves computed")
    if "text" in parts:
        import ext13syn_text as TX
        if "ladder" in summ:
            ladder_tables(summ["ladder"], summ["contrasts"], summ["strata"], summ["confounds"])
            ladder_figure(summ["ladder"], summ["contrasts"])
            TX.ladder_section(summ)
        if "curves" in summ:
            import ext13syn_curves_report as CR
            CR.tables_and_figures(summ)
            TX.curves_section(summ)
        log("tables, figures, sections written")
    summ["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(summ, open(SUMM, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
