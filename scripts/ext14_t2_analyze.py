"""Phase-6 tier-2 analysis (p6-analysis, CPU): per tier-2 cell the Phase-4/5 outputs, computed with the Phase-5 analysis
functions on the patched registries (scripts/ext14_t2_common.py):
  selection   two-stage selected expert (Spec, pattern), joint top, population top-10 (selection.json);
  add-back    scripts/ext13ab_analyze.analyse_task (ext8 machinery; unit-cluster bootstrap): all-MoE ceiling, greedy / oracle /
              random curves, k for 80 % of the ceiling, overshoot; random-equivalent marks (moetrace/ext14_plotutil);
  circuit     scripts/ext13ci_analyze.analyse1 (single heads at every layer, head DLA, attention mass) and analyse_group
              (mixed head + expert greedy: r(20), heads among the first 10 picks, k80);
  knockout    scripts/ext13ko_analyze.analyse (F of the selected / Phase-3 / CounterFact experts, same-layer controls,
              population sets vs frequency-matched random sets);
  writer      scripts/ext13wr_analyze.part_a (direct share D / T of the single-expert patch effect).
Writes results/p6_<cell>_<proto>_t2/t2_summary.json (full) and results/ext14_t2_summary.json (compact, all cells), tables
results/tables/ext14_t2_*.md|csv and figures results/figures/ext14_t2_*_addback.png|pdf (with the random-equivalent marks).

Usage: python scripts/ext14_t2_analyze.py [--cells e1_a2e/qwen3,...] [--no-figs]
"""
import argparse, json, os, sys, time, traceback
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, pandas as pd
import ext14_t2_common as T2
from moetrace.models import RESULTS
from moetrace.ext14_plotutil import random_equivalent, fmt_equiv

TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
SFX = T2.SUFFIX  # smoke runs: "_smoke" (tables / figures / summary carry the suffix)
SUMM = os.path.join(RESULTS, f"ext14_t2_summary{SFX}.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsafe(o):
    import ext12_analyze as E12
    return E12.jsafe(o)


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ------------------------------------------------------------------------------------------------------------------
def ab_part(cell, proto):
    AB = T2.patch_ab()
    import ext13ab_analyze as ABA
    reg = AB.registry(ready_only=False, check_rep=False)
    key = f"{cell}_{proto}"
    spec = reg.get(key)
    if spec is None or not os.path.exists(os.path.join(RESULTS, spec.out, "addback_state.pkl")):
        return None
    complete = AB.run_complete(spec.out)
    if not complete:
        return {"complete": False}
    s = ABA.analyse_task(spec, {})
    b = s["boot"]
    out = {"complete": True, "key": key, "K": s["K"], "n_eval_cases": s["n_eval_cases"], "n_units": s["n_units"], "mean_drop": s["mean_drop"],
           "ceiling": b.get("ceiling"), "restored_all_moe": b.get("restored_all_moe"), "direct": s.get("direct"), "label": s["label"],
           "tlabel": s["tlabel"], "cond": cell, "family": "main", "model": s["model"]}
    for o in ("greedy", "oracle", "rand", "dla", "pop"):
        if o in b:
            out[o] = {f: b[o][f] for f in ("k", "r_curve", "r_curve_lo", "r_curve_hi", "k80", "k80max", "r10", "r15", "max_r", "overshoot", "auc_log_k15") if f in b[o]}
    if "greedy" in b and "rand" in b:
        out["random_equiv_greedy"] = random_equivalent(b["greedy"]["k"], b["greedy"]["r_curve"], b["rand"]["k"], b["rand"]["r_curve"])
    if "oracle" in b and "rand" in b:
        out["random_equiv_oracle"] = random_equivalent(b["oracle"]["k"], b["oracle"]["r_curve"], b["rand"]["k"], b["rand"]["r_curve"])
    out["_full"] = s
    return out


def ci_part(cell, proto):
    CI = T2.patch_ci()
    import ext13ci_analyze as CA

    def addback_reference(cond, proto_, vrows, case_of, _orig=CA.addback_reference):
        return None  # the expert-only greedy comparison is read from this cell's add-back summary instead
    CA.addback_reference = addback_reference
    out = {}
    if CI.step1_complete(cell, proto):
        s1, _ = CA.analyse1(cell, proto)
        out["step1"] = {"n_rows": s1["n_rows"], "n_units": s1["n_units"], "mean_drop": s1["mean_drop"],
                        "top": [{k: h[k] for k in ("head", "layer", "h", "rescue_over_drop", "dla_over_drop", "share_of_layer_attn", "attn_label", "z") if k in h}
                                for h in s1["top"][:10]],
                        "z8": s1["z8"], "k_heads_pop_additive": s1["k_heads_pop_additive"], "experts": s1["experts"],
                        "attn": {k: v for k, v in (s1.get("attn") or {}).items() if k in ("classes", "rescue_weighted_mass_top10", "top10_labels")}}
        out["_full_step1"] = s1
    od2 = CI.step2_dir(proto, cell)
    if os.path.exists(os.path.join(od2, "circuit_state.pkl")):
        meta = json.load(open(os.path.join(od2, "run_meta.json")))
        if meta.get("complete"):
            res, info = CA.analyse_group(proto, cell)
            r = res[CI.key_of(cell, proto)]
            gm = r["curves"]["greedy_mix"]
            out["step2"] = {"complete": True, "n_rows": r["n_rows"], "ceilings": r["ceilings"],
                            "r20": gm.get("r20"), "r10": gm.get("r10"), "k_q": gm["k_q"], "heads_first10": gm["composition"]["10"]["mean_heads"],
                            "first_picks": gm.get("first_picks"), "recurring_first10": gm.get("recurring_first10"),
                            "greedy_mix": {k: gm[k] for k in ("k", "r", "lo", "hi")}, "greedy_head": {k: r["curves"]["greedy_head"][k] for k in ("k", "r", "lo", "hi")},
                            "pass_info": info}
            out["_full_step2"] = r
        else:
            out["step2"] = {"complete": False, "passes": meta.get("passes")}
    return out or None


def ko_part(cell, proto):
    K = T2.patch_ko()
    import ext13ko_analyze as KA
    e = K.entry(cell, proto)
    od = K.out_dir(e)
    if not os.path.exists(os.path.join(od, "plan_main.json")):
        return None
    meta = json.load(open(os.path.join(od, "run_meta.json"))) if os.path.exists(os.path.join(od, "run_meta.json")) else {}
    if not g(meta, "phases", "main", "complete"):
        return {"complete": False}
    s = KA.analyse(KA.Cond(cell, proto))
    tg = s["targets"]
    eff = s["effects"]

    def F(name):
        x = g(eff, f"tgt:{name}", "full") or next((v.get("full") for k, v in eff.items() if k.endswith(name) and "full" in v), None)
        return [x["F"], x["F_lo"], x["F_hi"]] if x else None
    out = {"complete": True, "targets": tg, "baseline": s["baseline"], "F": {t: F(t) for t in (tg.get("all") or [])},
           "same_layer": s.get("same_layer"), "sets": s.get("sets"), "final_vs_all": s.get("final_vs_all"), "selfrepair": s.get("selfrepair"),
           "gpu_s": s.get("gpu_s")}
    out["_full"] = s
    return out


def wr_part(cell, proto):
    WR = T2.patch_wr()
    K = T2.patch_ko()
    e = K.entry(cell, proto)
    if not WR.pass_done(e):
        return None
    import ext13wr_analyze as WA
    res, _ = WA.part_a(e, "")
    ev = res.get("experts_val") or {}
    out = {"routing_match": res.get("routing_match"), "n_cases": res.get("n_cases"), "band_share": res.get("band_share"), "targets": res.get("targets"),
           "experts_val": ev, "set_pop10_val": res.get("set_pop10_val")}
    out["_full"] = res
    return out


def analyse_cell(cell, proto):
    out = {"cell": cell, "proto": proto, "run": T2.t2_run(cell, proto), "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    csj = os.path.join(T2.t2_dir(cell, proto), "case_sets.json")
    ps = json.load(open(csj)).get("_pooled_split") if os.path.exists(csj) else None
    if ps:
        out["pooled_split"] = ps
        out["flag"] = f"pooled-split, n = {ps['n_evaluation_pairs']}"
    sel = T2.selection(cell, proto)
    if sel:
        ts = g(sel, "w6", "main", "two_stage") or {}
        out["selection"] = {"selected_expert": sel.get("selected_expert"), "pop_top10": sel.get("pop_top10"),
                            "layer": ts.get("layer"), "layer_val": ts.get("layer_val"), "layer_norm": ts.get("layer_norm"),
                            "coalitions": ts.get("coalitions"), "pattern": ts.get("pattern"), "equal_norm": ts.get("equal_norm"),
                            "joint_top": (g(sel, "w6", "main", "joint_top") or [])[:5], "fixed": g(sel, "w6", "main", "cf_experts"),
                            "interior": {k: (g(sel, "w6", "main", "interior") or {}).get(k) for k in ("layer", "selection", "pattern")}}
    for name, fn in (("addback", ab_part), ("circuit", ci_part), ("knockout", ko_part), ("writer", wr_part)):
        try:
            out[name] = fn(cell, proto)
        except Exception as ex:  # noqa: BLE001  (one failing part must not block the rest)
            out[name] = {"error": repr(ex), "trace": traceback.format_exc()[-1500:]}
            log(f"{cell}/{proto} {name}: ERROR {ex!r}")
    return out


def compact(o):
    if isinstance(o, dict):
        return {k: compact(v) for k, v in o.items() if not str(k).startswith("_")}
    return o


# ------------------------------------------------------------------------------------------------------------------
def md(df: pd.DataFrame, name: str, title: str, note: str = ""):
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name + ".md"), "w") as f:
        f.write(f"**{title}**\n\n")
        f.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for _, r in df.iterrows():
            f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
        if note:
            f.write(f"\n{note}\n")
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)


def f3(x, d=2, signed=False):
    if x is None:
        return "n/a"
    if isinstance(x, dict):
        x = [x.get("ratio", x.get("mean", x.get("v"))), x.get("lo", x.get("ci_lo")), x.get("hi", x.get("ci_hi"))]
    if isinstance(x, (list, tuple)):
        v = [np.nan if (y is None or isinstance(y, str)) else float(y) for y in list(x) + [None, None]][:3]
        if not np.isfinite(v[0]):
            return "n/a"
        s = f"{{:{'+' if signed else ''}.{d}f}}"
        return s.format(v[0]) + (f" [{s.format(v[1])}, {s.format(v[2])}]" if np.isfinite(v[1]) else "")
    return f"{float(x):{'+' if signed else ''}.{d}f}"


def fk(v):
    if v is None:
        return "n/a"
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if v is None:
        return "n/a"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "never" if not np.isfinite(v) else f"{int(round(v))}"


def ceil_ok(ab):
    """k80 of the all-MoE ceiling is meaningful only when the ceiling exceeds 0.1 of the drop (Phase-5 rule); otherwise
    k80 of each curve's own maximum (k80max) is reported."""
    c = ab.get("ceiling")
    try:
        return float(c[0]) > 0.1
    except (TypeError, ValueError, IndexError):
        return False


def tables(S):
    rows = []
    for key, s in S.items():
        sel = s.get("selection") or {}
        se = sel.get("selected_expert") or {}
        ab = s.get("addback") or {}
        ci2 = g(s, "circuit", "step2") or {}
        ko = s.get("knockout") or {}
        wr = s.get("writer") or {}
        name = se.get("name")
        Fsel = (ko.get("F") or {}).get(name) if name else None
        dsel = next((x for x in (wr.get("experts_val") or []) if isinstance(x, dict) and x.get("expert") == name), None) if name else None
        top10 = g(ko, "sets", "k10") or {}
        rows.append({"Cell": T2.XA.cell_label(s["cell"]) + (f" ({s['flag']})" if s.get("flag") else ""), "Model": T2.XA.LABEL.get(s["proto"], s["proto"]),
                     "Selected expert": name or "none", "Pattern": se.get("pattern", ""), "Spec [95% CI]": f3(se.get("spec"), 2, True),
                     "All-MoE ceiling": f3(ab.get("ceiling")) if ab.get("complete") else "pending",
                     "k80 greedy / oracle / random": (" / ".join(fk(g(ab, o, "k80")) for o in ("greedy", "oracle", "rand")) if ceil_ok(ab) else
                                                      "own-max k80: " + " / ".join(fk(g(ab, o, "k80max")) for o in ("greedy", "oracle", "rand"))) if ab.get("complete") else "pending",
                     "random k for greedy 1·5·10·15": fmt_equiv(ab["random_equiv_greedy"]) if ab.get("random_equiv_greedy") else "",
                     "Mixed r(20)": f3(ci2.get("r20")) if ci2.get("complete") else "pending",
                     "Heads in first 10": T2.XA.fnum(ci2.get("heads_first10"), ".1f", "") if ci2.get("complete") else "",
                     "KO F selected": f3(Fsel, 3, True) if Fsel else ("pending" if not ko.get("complete") else "n/a"),
                     "KO top-10 vs random": (f"{T2.XA.fnum(top10.get('pop'), '+.3f')} vs {T2.XA.fnum(top10.get('rand_mean'), '+.3f')}" if top10.get("pop") is not None else ""),
                     "Writer D/T selected": f3(dsel.get("share")) if isinstance(dsel, dict) and isinstance(dsel.get("share"), dict) else ""})
    if rows:
        md(pd.DataFrame(rows), "ext14_t2_overview" + SFX, "Phase 6 tier 2: selected expert, add-back, mixed head + expert greedy, knockout, writer (validation; cluster bootstrap)")
    eq = []
    for key, s in S.items():
        ab = s.get("addback") or {}
        for o in ("greedy", "oracle"):
            d = ab.get(f"random_equiv_{o}")
            if d:
                eq.append({"Cell": T2.XA.cell_label(s["cell"]) + (f" ({s['flag']})" if s.get("flag") else ""), "Model": T2.XA.LABEL.get(s["proto"], s["proto"]), "Curve": o,
                           **{f"k={k}": fk(v) if v is not None else "never" for k, v in d.items()}})
    if eq:
        md(pd.DataFrame(eq), "ext14_t2_random_equiv" + SFX, "Random-order k that reaches greedy's (oracle's) r at k = 1, 5, 10, 15 (P6-0 definition, interpolated in log k)",
           "'never' = the random order (which ends at the all-MoE ceiling) never reaches that value.")


def figures(S):
    import ext13ab_plot as ABP
    import matplotlib
    matplotlib.use("Agg")
    ABP.style()
    orig_save = ABP.save

    def save(fig, outdir, name):
        return orig_save(fig, outdir, name.replace("ext13ab_", "ext14_t2_ab_") + SFX)
    ABP.save = save
    runs = {}
    for key, s in S.items():
        ab = s.get("addback") or {}
        full = ab.get("_full")
        if full:
            runs[key] = jsafe(full)
    if runs:
        ABP.cond_figs({"runs": runs}, FIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None)
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    T2.patch_all()
    todo = []
    for proto, cells in T2.T2_CELLS.items():
        for c in cells:
            if args.cells and f"{c}/{proto}" not in args.cells.split(","):
                continue
            if os.path.exists(os.path.join(T2.t2_dir(c, proto), "selection.json")):
                todo.append((c, proto))
    S = {}
    for c, p in todo:
        log(f"analysing {c}/{p}")
        s = analyse_cell(c, p)
        with open(os.path.join(T2.t2_dir(c, p), "t2_summary.json"), "w") as f:
            json.dump(jsafe(s), f, indent=1)
        S[f"{c}/{p}"] = s
    if not S:
        log("no tier-2 cell with a selection yet")
        return
    # every tier-2 cell analysed so far (this run's cells fresh, the others from their t2_summary.json)
    S_all = {}
    for proto, cells in T2.T2_CELLS.items():
        for c in cells:
            k = f"{c}/{proto}"
            p = os.path.join(T2.t2_dir(c, proto), "t2_summary.json")
            if k in S:
                S_all[k] = json.loads(json.dumps(jsafe(S[k])))
            elif os.path.exists(p):
                S_all[k] = json.load(open(p))
    allc = {"cells": {k: compact(v) for k, v in S_all.items()}, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(SUMM, "w") as f:
        json.dump(allc, f, indent=1)
    tables(S_all)
    if not args.no_figs:
        try:
            figures(S_all)
        except Exception as ex:  # noqa: BLE001
            log(f"figures failed: {ex!r}\n{traceback.format_exc()}")
    log(f"wrote {SUMM} ({len(S)} cells)")


if __name__ == "__main__":
    main()
