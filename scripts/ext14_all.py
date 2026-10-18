"""Phase 6 (agent p6-analysis): rebuild every ext14 cross-cell output from disk (CPU only, idempotent).

Reads the registry data/dial/p6_conditions.json (p6-data), the tier-1 run dirs of every ready cell, the tier-2 summary
(results/ext14_t2_summary.json, scripts/ext14_t2_analyze.py) and the R-causal runs (results/p6_causal_*), and writes
  results/ext14_summary.json            every number keyed by cell / proto + the "predictions" block (IDs of
                                        data/dial/predictions.json: observed value, CI, verdict held / failed /
                                        insufficient data / pending);
  results/tables/ext14_*.md|csv         cells, ladder, Dial E, Dial A, Set C, R-metric, R-site, R-causal, hand-off,
                                        Stage 2 (twins, series), prediction values;
  results/figures/ext14_*.png|pdf       ladder, Dial E, Dial A dose-response, Set C, R-metric, hand-off, dense vs MoE curves and
                                        share arrows, capacity series;
  results/sections/ext14_{ladder,dial,dense,scale}.md.
Missing cells are skipped (the outputs fill in as cells reach BASE READY). Per-cell summaries are cached in the previous
summary and recomputed only when the run dir or the registry entry changed.

Usage: python scripts/ext14_all.py [--force] [--no-figs] [--no-sections]
"""
import argparse, fcntl, hashlib, json, os, sys, time, traceback
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, pandas as pd
from moetrace import ext14_analysis as X
from moetrace.models import RESULTS

ROOT = X.ROOT
TAB, FIG = X.TAB, X.FIG
SECT = os.path.join(RESULTS, "sections")
LOCK = os.path.join(ROOT, "logs", "ext14_analysis.lock")
SUMM = os.path.join(RESULTS, os.environ.get("EXT14_SUMMARY_NAME", "ext14_summary.json"))
SFX = os.environ.get("EXT14_OUT_SUFFIX", "")  # smoke runs write ext14_*<SFX> outputs


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsafe(o):
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
    if isinstance(o, (list, tuple)):
        return [jsafe(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsafe(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.str_):
        return str(o)
    return o


def nn(x):
    """JSON null -> nan inside CI triples."""
    if x is None:
        return None
    if isinstance(x, list):
        return [np.nan if v is None else v for v in x]
    return x


# ------------------------------------------------------------------------------------------------------------------
# compute
# ------------------------------------------------------------------------------------------------------------------
CACHE_VERSION = "2026-10-17e"  # bump whenever the per-cell analysis code changes (invalidates the cached cell summaries)


def signature(cell, proto, e):
    r = X.run_name(e)
    aw = ((e.get("anchor_runs") or {}).get("sweep_attention") or "")
    if aw:
        r = aw[len("results/"):] if aw.startswith("results/") else aw
    d = os.path.join(RESULTS, r) if r else None
    h = hashlib.md5((CACHE_VERSION + json.dumps(e, sort_keys=True, default=str)).encode())
    if d and os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p) and (f.endswith(".parquet") or f.endswith(".npz") or f == "case_sets.json"):
                st = os.stat(p)
                h.update(f"{f}:{st.st_mtime_ns}:{st.st_size}".encode())
    pf = X._abs(e.get("pair_file"))
    if pf and os.path.exists(pf):
        h.update(str(os.stat(pf).st_mtime_ns).encode())
    return h.hexdigest()


def compute(force=False):
    prev = json.load(open(SUMM)) if os.path.exists(SUMM) and not force else {}
    cache = X.Cache()
    reg = cache.reg
    S = {"updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "registry": X.REG_JSON, "cells": {}, "behaviour_only": {},
         "notes": {"share": "AUC+(attention) / (AUC+ attention + AUC+ MoE or MLP), single-layer final-token patches, validation pairs",
                   "ci": "cluster bootstrap over the cell's unit (WinoGrande twin / template seed / pair), 5,000 resamples, seed 0",
                   "paired": "contrasts on the common items with the same unit resamples (item = WinoGrande twin across levels, pair id across models)"}}
    sigs = {}
    for cell in sorted(reg):
        for proto, e in reg[cell].items():
            k = f"{cell}/{proto}"
            if e.get("behaviour_only"):
                S["behaviour_only"][k] = X.behaviour(e)
                continue
            if not cache.ready(cell, proto):
                continue
            sg = signature(cell, proto, e)
            sigs[k] = sg
            old = prev.get("cells", {}).get(k)
            if old and "error" not in old and prev.get("signatures", {}).get(k) == sg:
                S["cells"][k] = old
                continue
            log(f"cell {k}")
            try:
                S["cells"][k] = jsafe(X.cell_summary(cell, proto, reg))
            except Exception as ex:  # noqa: BLE001
                log(f"cell {k} FAILED: {ex!r}\n{traceback.format_exc()}")
                S["cells"][k] = {"error": repr(ex)}
    S["signatures"] = sigs
    protos = sorted({k.split("/")[1] for k in S["cells"]}, key=lambda p: X.ALL_PROTOS.index(p) if p in X.ALL_PROTOS else 99)
    S["protos"] = protos
    reg_protos = sorted({p for c in reg for p in reg[c]}, key=lambda p: X.ALL_PROTOS.index(p) if p in X.ALL_PROTOS else 99)
    for name, fn in (("ladder", X.ladder), ("dial_e", X.dial_e), ("dial_a", X.dial_a), ("set_c", X.set_c), ("r_metric", None), ("r_site", X.r_site)):
        S[name] = {}
        for p in (reg_protos if name == "set_c" else protos):
            try:
                if name == "r_metric":
                    st1 = [c for c in reg if p in reg[c] and not c.startswith("l_cf")] if p in X.STAGE1 else None
                    v = X.r_metric(cache, p, st1)
                else:
                    v = fn(cache, p)
            except Exception as ex:  # noqa: BLE001
                log(f"{name}/{p} FAILED: {ex!r}\n{traceback.format_exc()}")
                v = {"error": repr(ex)}
            if v:
                S[name][p] = jsafe(v)
    S["handoff"] = {p: X.handoff_table(S["cells"], p) for p in protos}
    # direction gap (a2e - e2a) per model and level on the models' own sets (independent resamples; E1 family: the plain
    # neutral cell, else the all-strata cell; E3: frame-matched P3 and the F1 anchor against the e2a E3 cell)
    S["direction_gap_models"] = {}
    for p in protos:
        ent = {}
        for lv, ca, cb in (("e1", "e1_a2e", "e1_e2a"), ("e1p", "e1p_a2e", "e1p_e2a"), ("e3p3", "e3p3_a2e", "e3_e2a"), ("e3", "e3_a2e", "e3_e2a"), ("e0", "e0_a2e", "e0_e2a")):
            for sfx_ in ("", "_all"):
                a_ = ca + sfx_ if lv in ("e1", "e1p") else ca
                b_ = cb + sfx_ if lv in ("e1", "e1p") else cb
                if cache.ready(a_, p) and cache.ready(b_, p):
                    try:
                        d_ = X.own_diff(cache, a_, p, b_, p)
                    except Exception as ex:  # noqa: BLE001
                        log(f"direction gap {p} {lv} FAILED: {ex!r}")
                        d_ = None
                    if d_:
                        pooled_ = bool((S["cells"].get(f"{a_}/{p}") or {}).get("pooled") or (S["cells"].get(f"{b_}/{p}") or {}).get("pooled"))
                        ent[lv] = {"gap": d_, "cells": [a_, b_], "pooled": pooled_}
                    break
        if ent:
            S["direction_gap_models"][p] = ent
    # Stage 2
    S["twins"] = {}
    for dense, moe in X.TWINS:
        try:
            v = X.twin_compare(cache, dense, moe)
        except Exception as ex:  # noqa: BLE001
            log(f"twin {dense}->{moe} FAILED: {ex!r}\n{traceback.format_exc()}")
            v = {"error": repr(ex)}
        if v:
            S["twins"][f"{dense}->{moe}"] = jsafe(v)
    S["series"] = {}
    for nm, ps in X.SERIES.items():
        try:
            v = X.series_compare(cache, ps)
        except Exception as ex:  # noqa: BLE001
            log(f"series {nm} FAILED: {ex!r}\n{traceback.format_exc()}")
            v = {"error": repr(ex)}
        if v and "cells" in v:
            if nm == "qwen3":
                for c, x in v["cells"].items():
                    sh = x["share"]
                    if "qwen3_4b" in sh and "qwen3_14b" in sh:
                        # 14B - 4B on the common items (joint resamples are not stored: CI from the two own CIs is not
                        # paired, so recompute paired on the two cells)
                        cv = x.get("variant", c)
                        d4, d14 = cache.data(cv, "qwen3_4b", item_mode="pair"), cache.data(cv, "qwen3_14b", item_mode="pair")
                        pp = X.paired(d14, d4) if d4 is not None and d14 is not None else None
                        if pp:
                            x["dense_trend"] = pp["share"]["diff"]
                rho = {}
                own30 = {c: (S["cells"].get(f"{c}/qwen3") or {}).get("validation", {}).get("share") for c in X.SET_A}
                for p in ("qwen3_4b", "qwen3_8b", "qwen3_14b"):
                    a, b = [], []
                    for c in X.SET_A:
                        s_ = (S["cells"].get(f"{c}/{p}") or {}).get("validation", {}).get("share")
                        if s_ and own30.get(c):
                            a.append(s_[0]); b.append(own30[c][0])
                    rho[p] = X.spearman(a, b) if len(a) >= 3 else None
                v["spearman_dense_vs_30b"] = rho
            S["series"][nm] = jsafe(v)
    # depth-normalised curves for every Set-A cell and model (figures)
    S["curves"] = {}
    for c in X.SET_A + ("e0_a2e", "e0_e2a"):
        for p in protos:
            k = f"{c}/{p}"
            cc = S["cells"].get(k, {}).get("validation", {})
            if cc.get("curve_attn"):
                L = len(cc["curve_attn"])
                S["curves"][k] = {"x": [round(i / L, 4) for i in range(L)], "attn": cc["curve_attn"], "mlp": cc["curve_moe"]}
    # behaviour of every scanned pair (E2 copy trap, Dial A statement vs prior, Set C statement vs event, ...)
    S["behaviour_rates"], S["beh_sigs"] = {}, {}
    for cell in sorted(reg):
        for proto, e in reg[cell].items():
            if cell.startswith(("e2_", "acon_", "aagr_", "e0_", "sc_", "e1p_", "e1_", "e3")):
                k = f"{cell}/{proto}"
                pf = X._abs(e.get("pair_file"))
                sg = f"{CACHE_VERSION}:{os.stat(pf).st_mtime_ns}:{os.stat(pf).st_size}" if pf and os.path.exists(pf) else None
                S["beh_sigs"][k] = sg
                if sg and prev.get("beh_sigs", {}).get(k) == sg and k in prev.get("behaviour_rates", {}):
                    S["behaviour_rates"][k] = prev["behaviour_rates"][k]
                    continue
                try:
                    b = X.behaviour_rates(cell, proto, e)
                except Exception as ex:  # noqa: BLE001
                    log(f"behaviour {cell}/{proto} FAILED: {ex!r}")
                    b = None
                if b:
                    S["behaviour_rates"][k] = jsafe(b)
    # Qwen2-57B shared vs routed (tier-2 shared-expert run, if present)
    S["qwen2_shared"] = shared_split(cache)
    # tier 2 and R-causal
    t2p = os.path.join(RESULTS, "ext14_t2_summary.json")
    S["tier2"] = json.load(open(t2p)).get("cells", {}) if os.path.exists(t2p) else {}
    S["r_causal"] = causal_summary()
    try:
        S["tier2_e3_ref"] = e3_tier2_reference()
    except Exception as ex:  # noqa: BLE001
        S["tier2_e3_ref"] = {"error": repr(ex)}
    cp = os.path.join(RESULTS, "ext14_circuit_summary.json")
    if os.path.exists(cp):
        try:
            S["circuit_predictions"] = json.load(open(cp)).get("predictions", {})
        except Exception:  # noqa: BLE001
            pass
    S = jsafe(S)
    S["predictions"] = jsafe(X.evaluate_predictions(denull(S)))
    return S


def denull(o):
    """JSON nulls inside numeric lists -> nan (for the prediction code that compares numbers)."""
    if isinstance(o, dict):
        return {k: denull(v) for k, v in o.items()}
    if isinstance(o, list):
        if o and all(v is None or isinstance(v, (int, float)) for v in o):
            return [np.nan if v is None else v for v in o]
        return [denull(v) for v in o]
    return o


def shared_split(cache):
    """Qwen2-57B-A14B (prediction D6): all-MoE vs routed-only vs shared-only joint patches of p6-data's T3
    (joint_shared_rows.parquet) per Set-A cell; D6 = the shared share of the MoE rescue in the knowledge cells (denoising)."""
    out = {"cells": {}}
    for cell in X.SET_A + ("e3p3_a2e",):
        try:
            v = X.shared_routed(cache, cell, "qwen2_57b")
        except Exception as ex:  # noqa: BLE001
            log(f"shared_routed {cell} FAILED: {ex!r}")
            v = None
        if v:
            out["cells"][cell] = v
    kn = {c: (v.get("denoise") or {}).get("shared_share") for c, v in out["cells"].items() if c in X.KNOWLEDGE_CELLS and (v.get("denoise") or {}).get("shared_share")}
    if kn:
        out["knowledge_shared_share"] = kn
    return out if out["cells"] else None


def e3_tier2_reference():
    """Tier 2 of E3 = the Phase-5 dir_f1 (a2e) / dir_opt (e2a) runs: selected expert, add-back, mixed circuit, knockout, writer
    (read from results/ext13{ab,ci,ko,wr}_summary.json and data/wino_ref/p5_conditions.json)."""
    def js(name):
        p = os.path.join(RESULTS, name)
        return json.load(open(p)) if os.path.exists(p) else {}
    ab, ci_, ko, wr = js("ext13ab_summary.json"), js("ext13ci_summary.json"), js("ext13ko_summary.json"), js("ext13wr_summary.json")
    reg = json.load(open(os.path.join(ROOT, "data/wino_ref/p5_conditions.json"))).get("conditions", {})
    out = {}
    for cell, cond in X.P5_REF.items():
        for proto in X.STAGE1:
            e = {"cond": cond, "proto": proto}
            se = ((reg.get(cond) or {}).get(proto) or {}).get("selected_expert") or {}
            e["selected_expert"] = {k: se.get(k) for k in ("name", "pattern", "spec", "rescue")}
            r = (ab.get("runs") or {}).get(f"{cond}_{proto}") or {}
            b = r.get("boot") or {}
            if b:
                e["addback"] = {"ceiling": b.get("ceiling"), "k80": {o: (b.get(o) or {}).get("k80") for o in ("greedy", "oracle", "rand")}}
            c = (ci_.get("step2") or {}).get(f"{cond}/{proto}") or {}
            gm = (c.get("curves") or {}).get("greedy_mix") or {}
            if gm:
                e["circuit"] = {"r20": gm.get("r20"), "heads_first10": ((gm.get("composition") or {}).get("10") or {}).get("mean_heads"), "k80": (gm.get("k_q") or {}).get("0.8")}
            k = ((ko.get("models") or {}).get(proto) or {}).get(cond) or {}
            if k:
                nm = (k.get("targets") or {}).get("selected")
                f = ((k.get("effects") or {}).get(f"tgt:{nm}") or {}).get("full") or {}
                t10 = (k.get("sets") or {}).get("k10") or {}
                e["knockout"] = {"selected": nm, "F": [f.get("F"), f.get("F_lo"), f.get("F_hi")] if f else None, "top10": t10.get("pop"), "top10_random": t10.get("rand_mean")}
            w = (((wr.get("models") or {}).get(proto) or {}).get(cond) or {}).get("A") or {}
            ev = next((x for x in (w.get("experts_val") or []) if x.get("expert") == e["selected_expert"].get("name")), None)
            if ev and isinstance(ev.get("share"), dict):
                e["writer_share"] = [ev["share"]["v"], ev["share"]["lo"], ev["share"]["hi"]]
            out[f"{cell}/{proto}"] = e
    return out


def causal_summary():
    p = os.path.join(RESULTS, "p6_causal_summary.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:  # noqa: BLE001
            return None
    return None


# ------------------------------------------------------------------------------------------------------------------
# tables
# ------------------------------------------------------------------------------------------------------------------
def md(df: pd.DataFrame, name: str, title: str, note: str = "") -> str:
    os.makedirs(TAB, exist_ok=True)
    esc = lambda v: str(v).replace("|", "\\|").replace("\n", " ")  # noqa: E731  (cell text may contain '|', e.g. |E1p - E1|)
    lines = [f"**{title}**", "", "| " + " | ".join(esc(c) for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(esc(v) for v in r.values) + " |" for _, r in df.iterrows()]
    if note:
        lines += ["", note]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(TAB, name + SFX + ".md"), "w") as f:
        f.write(txt)
    df.to_csv(os.path.join(TAB, name + SFX + ".csv"), index=False)
    return txt


F = lambda x, d=2, s=False: X.fr(nn(x), d, s)  # noqa: E731
N = X.fnum  # None / NaN safe number formatting


def lab(p):
    return X.SHORT.get(p, p)


def cell_order(c0):
    c = c0[:-4] if c0.endswith("_all") else c0
    order = ["e1_a2e", "e1p_a2e", "e2_a2e", "e3p3_a2e", "e3_a2e", "e0_a2e", "aagr_a2e", "acon_a2e", "e1_e2a", "e1p_e2a", "e2_e2a", "e3_e2a", "e0_e2a",
             "aagr_e2a", "acon_e2a", "l_ant", "l_prop", "l_cf", "l_bind", "l_ioi", "l_ind", "rsite_e1_a2e"]
    return (order.index(c) if c in order else 50, c0)


def fmt_k(v):
    v = nn(v)
    if v is None:
        return "n/a"
    v = v[0] if isinstance(v, list) else v
    if isinstance(v, str):
        return "never" if v in ("inf", "Infinity") else v
    return "never" if v is None or not np.isfinite(v) else f"{int(round(v))}"


def tables(S):
    """Every table block runs on its own: a failing block is logged and the others are still written."""
    T = {}
    def _blk0():  # all cells
        rows = []
        for k in sorted(S["cells"], key=lambda k: (X.ALL_PROTOS.index(k.split("/")[1]) if k.split("/")[1] in X.ALL_PROTOS else 99, cell_order(k.split("/")[0]))):
            c, p = k.split("/")
            s = S["cells"][k]
            v = s.get("validation") or {}
            if not v:
                continue
            ho = (s.get("grid") or {}).get("handoff") or {}
            rows.append({"Model": lab(p), "Cell": X.cell_label(c) + (" †" if s.get("pooled") else ""), "Pairs (units)": f"{v.get('n_pairs')} ({v.get('n_units')})",
                         "Attention share": F(v.get("share")), "share, norm. LD": F(v.get("share_nld")), "share, Δp": F(v.get("share_dp")),
                         "share without the last ⌈L/12⌉ layers": F(v.get("share_int")),
                         "AUC+ attn / drop": F(v.get("auc_attn")), "AUC+ MoE or MLP / drop": F(v.get("auc_moe")), "A_direct": F(v.get("A_direct")),
                         "All-MoE / all-MLP M": F(v.get("M_all")), "Peak attn": f"L{v['peak_attn'][0]} {N(v['peak_attn'][1])}" if v.get("peak_attn") else "",
                         "Peak MoE / MLP": f"L{v['peak_moe'][0]} {N(v['peak_moe'][1])}" if v.get("peak_moe") else "",
                         "Hand-off (swap ≥ 0.5 until → final ≥ 0.5 from)": f"L{N(ho['swap_last_ge_0.5'], '.0f')} → L{N(ho['final_first_ge_0.5'], '.0f')}"
                         if ho.get("swap_last_ge_0.5") and ho.get("final_first_ge_0.5") and ho["swap_last_ge_0.5"][0] is not None and ho["final_first_ge_0.5"][0] is not None else "",
                         "Top heads (val rescue / drop)": ", ".join(f"{h['name']} {N(h['val_norm'], '+.2f')}" for h in ((s.get("heads") or {}).get("top") or [])[:3]
                                                                     if h.get("val_norm") is not None),
                         "Clean top-1": F(v.get("clean_top1")), "Drop": N(v.get("drop")), "T median": N(v.get("T_median"), ".0f")})
        if rows:
            T["cells"] = md(pd.DataFrame(rows), "ext14_cells", "Phase 6 cells: attention share and secondary readouts (validation pairs; † = pooled cell, all pairs, no held-out split; fractions of the drop; 95% cluster-bootstrap CIs)",
                            "Share = AUC+ attention / (AUC+ attention + AUC+ MoE or MLP) of the single-layer final-token patch curves; normalised LD = each directed case's "
                            "rescue divided by its own drop before averaging; Δp = p(answer | patched) − p(answer | corrupted). A_direct = direct path of all attention "
                            "outputs (exact final norm); All-MoE M = every final-token MoE / MLP output patched at once.")
    def _blk1():  # ladder
        rows = []
        for p, L in (S.get("ladder") or {}).items():
            for r in sorted(L.get("rows", []), key=lambda r: (r["grade"], r["cell"])):
                rows.append({"Model": lab(p), "Grade": r["grade"], "Task": r["label"], "Pairs": r["n_pairs"], "Attention share": F(r["share"]),
                             "norm. LD": F(r.get("share_nld")), "Δp": F(r.get("share_dp")), "A_direct": F(r.get("A_direct")), "All-MoE M": F(r.get("M_all")),
                             "Cue distance (tokens)": N(r.get("cue_distance"), ".0f")})
            rows.append({"Model": lab(p), "Grade": "", "Task": f"Spearman(grade, share) = {N(L.get('spearman'))} "
                         f"(CI {F(L.get('spearman_ci'))}), partial given cue distance {N(L.get('partial_spearman_given_cue'))}; "
                         f"Spearman(cue distance, share) {N(L.get('spearman_cue'))}", "Pairs": "", "Attention share": "", "norm. LD": "", "Δp": "",
                         "A_direct": "", "All-MoE M": "", "Cue distance (tokens)": ""})
        if rows:
            T["ladder"] = md(pd.DataFrame(rows), "ext14_ladder", "Ladder v2: pre-registered grade vs measured attention share (validation; CIs: cluster bootstrap; Spearman CI over the cells' bootstrap draws)")
    def _blk2():  # Dial E
        rows, rows_p = [], []
        for p, D in (S.get("dial_e") or {}).items():
            for dr in X.DIRS:
                for lv in ("e1", "e1p", "e2", "e3p3", "e3", "e0"):
                    v = (D.get("levels", {}).get(dr) or {}).get(lv)
                    if not v:
                        continue
                    rows.append({"Model": lab(p), "Direction": X.DIR_LABEL[dr], "Level": X.LEVEL_LABEL[lv] + (" †" if v.get("pooled") else ""), "Pairs": v.get("n_pairs"), "Share": F(v.get("share")),
                                 "norm. LD": F(v.get("share_nld")), "Δp": F(v.get("share_dp")), "A_direct": F(v.get("A_direct")), "M_direct": F(v.get("M_direct")),
                                 "All-MoE M": F(v.get("M_all")), "Clean top-1": F(v.get("clean_top1"))})
            for key, pp in (D.get("paired") or {}).items():
                a, b = key.split("_")[0].split("-")
                dr = key.split("_")[1]
                rows_p.append({"Model": lab(p), "Direction": X.DIR_LABEL[dr], "Contrast": f"{a.upper()} − {b.upper()}".replace("P", "p"), "Common twins": pp["n_items"],
                               "Share a": F(pp["share"]["a"]), "Share b": F(pp["share"]["b"]), "Δ share": F(pp["share"]["diff"], 2, True),
                               "Δ A_direct": F((pp.get("A_direct") or {}).get("diff"), 2, True), "Δ M": F((pp.get("M_all") or {}).get("diff"), 2, True)})
            for lv, pp in (D.get("gap") or {}).items():
                rows_p.append({"Model": lab(p), "Direction": "a→e − e→a", "Contrast": f"direction gap at {lv.upper()}".replace("P", "p"), "Common twins": pp["n_items"],
                               "Share a": F(pp["share"]["a"]), "Share b": F(pp["share"]["b"]), "Δ share": F(pp["share"]["diff"], 2, True),
                               "Δ A_direct": F((pp.get("A_direct") or {}).get("diff"), 2, True), "Δ M": F((pp.get("M_all") or {}).get("diff"), 2, True)})
            seen = []
            for tag in ("h_test", "h_test_pw", "h_test_pw_f1", "h_test_4cell", "h_test_f1"):
                h = D.get(tag)
                if not h or any(h == x for x in seen):
                    continue
                seen.append(h)
                rows_p.append({"Model": lab(p), "Direction": "a→e − e→a",
                               "Contrast": f"{'PRIMARY ' if tag == 'h_test' else ''}gap(E1) − ½·gap(E3) with E3 a→e = {h.get('e3_a2e_cell')} ({h.get('method', '')}) "
                                           f"(H-type if > 0); gap(E1)/gap(E3) = {F(h['gap_ratio_e1_e3'])}; 'Share a' = gap(E1), 'Share b' = gap(E3)",
                               "Common twins": h["n_items"], "Share a": F(h["gap_e1"], 2, True), "Share b": F(h["gap_e3"], 2, True),
                               "Δ share": F(h["gap_e1_minus_half_e3"], 2, True), "Δ A_direct": "", "Δ M": ""})
        rows_s = []
        for p, D in (S.get("dial_e") or {}).items():
            for dr, ent in (D.get("e2_strata") or {}).items():
                for k, v in ent.items():
                    if "share" in v and isinstance(v.get("share"), list):
                        rows_s.append({"Model": lab(p), "Direction": X.DIR_LABEL[dr], "E2 subset": k.replace("_", " "), "Pairs / common twins": v.get("n_pairs"),
                                       "Share": F(v.get("share")), "Contrast": "", "Δ share": ""})
                    elif "share" in v:
                        rows_s.append({"Model": lab(p), "Direction": X.DIR_LABEL[dr], "E2 subset": "no degree adverb", "Pairs / common twins": v.get("n_items"),
                                       "Share": F(v["share"]["b"]), "Contrast": k.split("_no_")[0].upper().replace("P", "p"), "Δ share": F(v["share"]["diff"], 2, True)})
        if rows_s:
            T["dial_e2"] = md(pd.DataFrame(rows_s), "ext14_dial_e2_strata", "E2 (negation) without degree-adverb twins and by antonym type (coordinator QA: negating a "
                              "gradable adjective with a degree adverb does not entail the antonym)")
        if rows:
            T["dial_e"] = md(pd.DataFrame(rows), "ext14_dial_e", "Dial E: attention share per level and direction (own validation sets)")
        if rows_p:
            T["dial_e_paired"] = md(pd.DataFrame(rows_p), "ext14_dial_e_paired", "Dial E: paired contrasts on the common WinoGrande twins (same cluster resamples for both cells)")
    def _blk3():  # Dial A
        rows = []
        for p, D in (S.get("dial_a") or {}).items():
            for dr, cl in (D.get("classes") or {}).items():
                for c in ("E0", "agree", "neutral", "conflict"):
                    v = cl.get(c)
                    if not v:
                        continue
                    rows.append({"Model": lab(p), "Direction": X.DIR_LABEL[dr], "Prior class": c + (" †" if v.get("pooled") else ""), "Cell": v.get("cell"), "Pairs": v.get("n_pairs"), "Share": F(v.get("share")),
                                 "A_direct": F(v.get("A_direct")), "M_direct (toward stated answer)": F(v.get("M_direct"), 2, True), "DLA MoE": F(v.get("dla_moe"), 2, True),
                                 "All-MoE M": F(v.get("M_all")), "Prior support (mean)": N(v.get("prior_mean"), "+.2f", "")})
            for dr, ds in (D.get("dose") or {}).items():
                for b in ds["bins"]:
                    rows.append({"Model": lab(p), "Direction": X.DIR_LABEL[dr], "Prior class": f"bin {b['bin']} (prior {N(b['prior_mean'], '+.2f')})", "Cell": "aagr + acon",
                                 "Pairs": b["n_pairs"], "Share": F(b["share"]), "A_direct": "", "M_direct (toward stated answer)": "", "DLA MoE": "", "All-MoE M": "",
                                 "Prior support (mean)": f"Spearman over bins {N(ds.get('spearman'), '+.2f')}" if b["bin"] == 5 else ""})
        if rows:
            T["dial_a"] = md(pd.DataFrame(rows), "ext14_dial_a", "Dial A: attention share by prior class and five prior-support bins (validation)")
    def _blk4():  # Set C
        rows = []
        for p, D in (S.get("set_c") or {}).items():
            for c, b in (D.get("behaviour") or {}).items():
                v = (D.get("subsets") or {}).get(c, {})
                ref = next((k for k in v if k.startswith("vs_")), None)
                beh = (f"statement {b['statement_items']} / event {b['event_items']} of {b['scanned_items']} items (statement rate among margin items "
                       f"{N(b['statement_rate'])})" if b.get("statement_rate") is not None else
                       ", ".join(f"{k} {b[k]}" for k in ("n_pool", "n_margin", "top1_rate") if k in b))
                rows.append({"Model": lab(p), "Cell": X.cell_label(c), "Behaviour": beh,
                             "Pairs": v.get("n_pairs", ""), "Share": F(v.get("share")) if v else "", "Reference": ref[3:] if ref else "",
                             "Δ share vs reference": F(v[ref]["share"]["diff"], 2, True) if ref else ""})
        if rows:
            T["set_c"] = md(pd.DataFrame(rows), "ext14_set_c", "Set C (event + statement): behaviour and attention share per subset")
    def _blk5():  # behaviour-only cells (< 64 pairs overall) and pooled cells: behaviour
        rows = []
        for k, b in sorted((S.get("behaviour_only") or {}).items(), key=lambda kv: (kv[0].split("/")[1], cell_order(kv[0].split("/")[0]))):
            c, p = k.split("/")
            extra = {kk: vv for kk, vv in b.items() if any(w in kk for w in ("statement", "prior", "event", "follow", "wins"))}
            rows.append({"Model": lab(p), "Cell": X.cell_label(c), "Status": "behaviour only", "Scanned pairs": b.get("n_pool", ""),
                         "Margin pairs (both ways)": b.get("n_margin", ""), "Margin rate": N(b.get("pool_margin_rate_pool"), ".2f", ""),
                         "Pass rate by order": ", ".join(f"o{o} {N(r)}" for o, r in (b.get("order_pass_rates") or {}).items()),
                         "Top-1 rate": b.get("top1_rate", ""), "Split sizes (pairs)": json.dumps(b.get("split_sizes", {})),
                         "Which answer wins": json.dumps(extra) if extra else ""})
        for k, s_ in sorted(S["cells"].items()):
            if s_.get("pooled"):
                c, p = k.split("/")
                b = s_.get("registry") or {}
                rows.append({"Model": lab(p), "Cell": X.cell_label(c) + " †", "Status": "pooled (tier 1 on all pairs)", "Scanned pairs": b.get("n_pool", ""),
                             "Margin pairs (both ways)": b.get("n_margin", ""), "Margin rate": N(b.get("pool_margin_rate_pool"), ".2f", ""),
                             "Pass rate by order": ", ".join(f"o{o} {N(r)}" for o, r in (b.get("order_pass_rates") or {}).items()),
                             "Top-1 rate": b.get("top1_rate", ""), "Split sizes (pairs)": json.dumps(b.get("split_sizes", {})), "Which answer wins": ""})
        if rows:
            T["behaviour"] = md(pd.DataFrame(rows), "ext14_behaviour_only", "Cells below the reporting threshold: behaviour only (< 64 pairs overall) and pooled (< 64 validation pairs, ≥ 64 overall)",
                                "Margin = logit(answer) − logit(contrast) ≥ 1 in both directions of the pair (for Dial A conflict cells the answer is the STATED one, "
                                "so the margin rate is the rate at which the statement beats the prior in both directions).")
    def _blkB():  # behaviour rates of every scanned pair
        rows = []
        lab_other = {"e2": "copy trap (word stated with the other entity)", "acon": "prior", "aagr": "other answer", "sc": "event / other",
                     "e0": "other answer", "e1": "other answer", "e1p": "other answer", "e3": "other answer", "e3p3": "other answer"}
        for k, b in sorted((S.get("behaviour_rates") or {}).items(), key=lambda kv: (kv[0].split("/")[1], cell_order(kv[0].split("/")[0]))):
            c, p = k.split("/")
            lvl = c.split("_")[0]
            for sname, r in [("all", b["all"])] + sorted(b.get("strata", {}).items()):
                rows.append({"Model": lab(p), "Cell": X.cell_label(c), "Stratum": sname, "Pairs (items)": f"{r['n_pairs']} ({r['n_items']})",
                             "Target wins, margin both ways": F(r["t_marg"]), f"Other answer wins, margin both ways": F(r["o_marg"]),
                             "Target, sign both ways": F(r["t_sign"]), "Other, sign both ways": F(r["o_sign"]),
                             "Items: target in every order / other in every order": f"{r['items_target_all_orders']} / {r['items_other_all_orders']}",
                             "Other answer =": lab_other.get(lvl, "other answer")})
        comp = [r for r in rows if r["Cell"].startswith(("E2 ", "A-conflict")) and (r["Stratum"] == "all" or r["Stratum"].startswith(("degree_adv", "antonym_type")))]
        if comp:
            md(pd.DataFrame(comp), "ext14_behaviour_e2", "E2 (negation) and Dial-A conflict behaviour: target vs the other answer (copy trap / prior), all scanned pairs "
               "(twin-cluster bootstrap CIs)", "Full table with every cell and stratum: ext14_behaviour_rates.md.")
        if rows:
            T["beh"] = md(pd.DataFrame(rows), "ext14_behaviour_rates", "Behaviour of every scanned pair (twin-cluster bootstrap CIs, 1,000 resamples)",
                          "d = logit(target) − logit(other) on each prompt of the pair; 'margin both ways' = the reporting rule (≥ 1 in both directions). "
                          "E2: the other answer is the copy trap (the attribute word stated next to the other entity); Dial A conflict: the prior; "
                          "Set C conflict (base cell): the event's answer.")
    def _blk6():  # R-metric
        rows = []
        for p, R in (S.get("r_metric") or {}).items():
            for c, v in sorted((R.get("cells") or {}).items(), key=lambda kv: cell_order(kv[0])):
                rows.append({"Model": lab(p), "Cell": X.cell_label(c), "Pairs": v["n_pairs"], "Share (LD)": F(v["ld"]), "Share (norm. LD)": F(v.get("nld")), "Share (Δp)": F(v.get("dp"))})
            rows.append({"Model": lab(p), "Cell": f"Spearman of the cell ranking: LD vs norm. LD {N(R.get('spearman_ld_nld'))}, LD vs Δp {N(R.get('spearman_ld_dp'))}",
                         "Pairs": "", "Share (LD)": "", "Share (norm. LD)": "", "Share (Δp)": ""})
        if rows:
            T["r_metric"] = md(pd.DataFrame(rows), "ext14_r_metric", "R-metric: attention share under LD, normalised LD and Δp (validation)")
    def _blk7():  # R-site
        rows = []
        for p, R in (S.get("r_site") or {}).items():
            pp = R.get("paired")
            hr = ((R.get("rsite_grid") or {}).get("handoff") or {})
            he = ((R.get("e1_grid") or {}).get("handoff") or {})
            rows.append({"Model": lab(p), "Evidence-site share": F((R.get("rsite") or {}).get("share")), "Question-site share (E1 a→e)": F(pp["share"]["b"]) if pp else "",
                         "Δ (evidence − question), common twins": F(pp["share"]["diff"], 2, True) if pp else "", "Common twins": pp["n_items"] if pp else "",
                         "Hand-off evidence site": f"L{hr['swap_last_ge_0.5'][0]} → L{hr['final_first_ge_0.5'][0]}" if hr.get("swap_last_ge_0.5") and hr.get("final_first_ge_0.5") else "",
                         "Hand-off question site": f"L{he['swap_last_ge_0.5'][0]} → L{he['final_first_ge_0.5'][0]}" if he.get("swap_last_ge_0.5") and he.get("final_first_ge_0.5") else ""})
        if rows:
            T["r_site"] = md(pd.DataFrame(rows), "ext14_r_site", "R-site: corruption in the evidence (two entity mentions swapped) vs in the question (E1 a→e)")
    def _blk8():  # hand-off
        rows = []
        for p, H in (S.get("handoff") or {}).items():
            for h in sorted(H, key=lambda h: cell_order(h["cell"])):
                rows.append({"Model": lab(p), "Cell": X.cell_label(h["cell"]), "L": h["L"], "Swapped token ≥ 0.5 until": F(h.get("swap_last_ge_0.5"), 0),
                             "Final ≥ 0.1 from": F(h.get("final_first_ge_0.1"), 0), "Final ≥ 0.5 from": F(h.get("final_first_ge_0.5"), 0),
                             "Final ≥ 0.9 from": F(h.get("final_first_ge_0.9"), 0), "Crossing": F(h.get("crossing"), 0)})
        if rows:
            T["handoff"] = md(pd.DataFrame(rows), "ext14_handoff", "Hand-off layers from the position × layer residual grid (validation; CIs: cluster bootstrap of the layer index, 1,000 resamples)")
    def _blk9():  # Stage 2
        rows = []
        for tw, t in (S.get("twins") or {}).items():
            if not t or "cells" not in t:
                continue
            for c, v in sorted(t["cells"].items(), key=lambda kv: cell_order(kv[0])):
                pp = v["paired"]
                rows.append({"Twin (dense → MoE)": f"{lab(t['dense'])} → {lab(t['moe'])}", "Cell": X.cell_label(v.get("variant", c)) + (" †" if v.get("pooled") else ""),
                             "Dense, own set": F(v.get("dense_own")), "MoE, own set": F(v.get("moe_own")), "MoE − dense (own sets)": F(v.get("own_diff"), 2, True),
                             "Common pairs": pp["n_items"], "MoE − dense (paired)": F(pp["share"]["diff"], 2, True),
                             "paired, without the last ⌈L/12⌉ layers": F((pp.get("share_int") or {}).get("diff"), 2, True),
                             "Δ A_direct (paired)": F((pp.get("A_direct") or {}).get("diff"), 2, True), "Δ M, MoE vs MLP (paired)": F((pp.get("M_all") or {}).get("diff"), 2, True),
                             "Contrast used for D1–D4": f"{F(v.get('diff_used'), 2, True)} ({v.get('basis', '')})"})
            rows.append({"Twin (dense → MoE)": f"{lab(t['dense'])} → {lab(t['moe'])}", "Cell": f"D1: |Δ| ≤ 0.10 in {N(t.get('frac_within_0.10'), '.0%')} of {t.get('n_cells')} cells; "
                         f"D3: Δ ≥ +0.10 in {N(t.get('frac_moe_higher_0.10'), '.0%')}; D4: Spearman of the own-set cell order {N(t.get('spearman_cells'))}"})
        if rows:
            T["twins"] = md(pd.DataFrame(rows).fillna(""), "ext14_dense_twins", "Dense parent vs MoE child (validation; own sets and paired on the common pairs, which are identical prompts)",
                            "Contrast used for the D predictions: paired on the common pairs when there are ≥ 32, else the difference of the two own "
                            "validation sets (independent cluster bootstraps). E1-family cells use the all-association-strata variant when it exists "
                            "for both models (coordinator 2026-10-17 03:19Z). Dense models are tier 1 only (no expert tables, Spec, controls or expert "
                            "knockouts; the MLP output plays the MoE output's role). The depth-normalised interior variant drops the last ⌈L/12⌉ "
                            "layers of both models (48 → 4, 32 → 3, 28 → 3, 36 → 3, 40 → 4). † = pooled cell in at least one model.")
        for nm, se in (S.get("series") or {}).items():
            rows = []
            for c, v in sorted((se.get("cells") or {}).items(), key=lambda kv: cell_order(kv[0])):
                row = {"Cell": X.cell_label(v.get("variant", c)) + (" †" if v.get("pooled") else "")}
                for p in se["protos"]:
                    row[f"{lab(p)} (own set)"] = F((v.get("own") or {}).get(p)) if p in (v.get("own") or {}) else "–"
                row["Common pairs"] = v["n_items"]
                row["paired shares (common pairs) " + " → ".join(lab(p) for p in v["members"])] = " → ".join(N(v["share"][p]) for p in v["members"])
                row["last − first (paired)"] = F(v.get("last_minus_first"), 2, True)
                row["interior variant, paired"] = " → ".join(N(v["share_int"][p]) for p in v["members"] if p in (v.get("share_int") or {}))
                if v.get("dense_trend"):
                    row["14B − 4B (paired on their common pairs)"] = F(v["dense_trend"], 2, True)
                rows.append(row)
            if rows:
                T[f"series_{nm}"] = md(pd.DataFrame(rows).fillna(""), f"ext14_capacity_{nm}",
                                       f"Capacity series '{nm}' ({' → '.join(lab(p) for p in se['protos'])}): attention share on each model's own validation set "
                                       "and on the pairs common to every member present (validation)",
                                       "Own sets differ in items between models; the paired columns use the common pairs only (few for cells whose models "
                                       "select different twins). † = pooled cell in at least one model.")
    def _blk10():  # tier 2: E1 (Phase 6) next to E3 (Phase 5 dir_f1 / dir_opt)
        rows = []
        t2 = S.get("tier2") or {}
        ref = S.get("tier2_e3_ref") or {}
        for proto in X.STAGE1:
            for dr in X.DIRS:
                for cell in (f"e1_{dr}", f"e3_{dr}"):
                    k = f"{cell}/{proto}"
                    if cell.startswith("e1"):
                        s_ = t2.get(k)
                        if not s_:
                            continue
                        se = (s_.get("selection") or {}).get("selected_expert") or {}
                        ab = s_.get("addback") or {}
                        c2 = (s_.get("circuit") or {}).get("step2") or {}
                        ko = s_.get("knockout") or {}
                        wr = s_.get("writer") or {}
                        nm = se.get("name")
                        ev = next((x for x in (wr.get("experts_val") or []) if isinstance(x, dict) and x.get("expert") == nm), None)
                        row = {"Model": lab(proto), "Direction": X.DIR_LABEL[dr], "Level": "E1 (Phase 6)" + (f", {s_['flag']}" if s_.get("flag") else ""), "Selected expert": f"{nm} ({se.get('pattern')})" if nm else "none",
                               "Spec": F(se.get("spec"), 2, True), "All-MoE ceiling": F(ab.get("ceiling")) if ab.get("complete") else "pending",
                               "k80 greedy / random": ((f"{fmt_k((ab.get('greedy') or {}).get('k80'))} / {fmt_k((ab.get('rand') or {}).get('k80'))}"
                                                    if N(ab.get("ceiling")) != "n/a" and float(N(ab.get("ceiling"))) > 0.1 else
                                                    f"own-max k80 {fmt_k((ab.get('greedy') or {}).get('k80max'))} / {fmt_k((ab.get('rand') or {}).get('k80max'))}")
                                                   if ab.get("complete") else ""),
                               "Mixed r(20)": F([c2["r20"]["ratio"], c2["r20"]["lo"], c2["r20"]["hi"]]) if c2.get("r20") else "pending",
                               "Heads in first 10": N(c2.get("heads_first10"), ".1f", ""),
                               "KO F selected": F((ko.get("F") or {}).get(nm), 3, True) if ko.get("complete") and nm else "pending",
                               "Writer D/T": F([ev["share"]["v"], ev["share"]["lo"], ev["share"]["hi"]]) if ev and isinstance(ev.get("share"), dict) else ""}
                    else:
                        e = ref.get(k)
                        if not e:
                            continue
                        se = e.get("selected_expert") or {}
                        ab = e.get("addback") or {}
                        c2 = e.get("circuit") or {}
                        ko = e.get("knockout") or {}
                        row = {"Model": lab(proto), "Direction": X.DIR_LABEL[dr], "Level": f"E3 (Phase 5 {e['cond']})", "Selected expert": f"{se.get('name')} ({se.get('pattern')})",
                               "Spec": F(se.get("spec"), 2, True), "All-MoE ceiling": F(ab.get("ceiling")),
                               "k80 greedy / random": f"{fmt_k((ab.get('k80') or {}).get('greedy'))} / {fmt_k((ab.get('k80') or {}).get('rand'))}",
                               "Mixed r(20)": F([c2["r20"]["ratio"], c2["r20"]["lo"], c2["r20"]["hi"]]) if c2.get("r20") else "",
                               "Heads in first 10": N(c2.get("heads_first10"), ".1f", ""),
                               "KO F selected": F(ko.get("F"), 3, True) if ko.get("F") else "", "Writer D/T": F(e.get("writer_share"))}
                    rows.append(row)
        if rows:
            T["t2_e1_e3"] = md(pd.DataFrame(rows), "ext14_t2_e1_vs_e3", "Tier 2 at E1 (Phase 6) next to E3 (the Phase-5 dir_f1 / dir_opt runs on the Phase-5 item set; not paired)")
    def _blkG():  # direction gap per model
        rows = []
        for p, ent in (S.get("direction_gap_models") or {}).items():
            row = {"Model": lab(p)}
            for lv, nm in (("e1", "E1"), ("e1p", "E1p"), ("e3p3", "E3 (a→e in P3)"), ("e3", "E3 (a→e in F1)"), ("e0", "E0")):
                x = ent.get(lv)
                row[nm] = (F(x["gap"], 2, True) + (" †" if x.get("pooled") else "") + (" (all strata)" if x["cells"][0].endswith("_all") else "")) if x else ""
            rows.append(row)
        if rows:
            T["gap"] = md(pd.DataFrame(rows), "ext14_direction_gap_models", "Direction gap a→e − e→a of the attention share per model and level (each cell's own validation set; independent cluster bootstraps)",
                          "Own sets, so the two directions are not on identical twins (paired contrasts for Qwen3 / Mixtral in ext14_dial_e_paired.md). † = pooled cell.")

    def _blkQ():  # Qwen2-57B shared vs routed
        rows = []
        for c, v in sorted(((S.get("qwen2_shared") or {}).get("cells") or {}).items(), key=lambda kv: cell_order(kv[0])):
            for dr in ("denoise",):  # noise = denoise of the mirrored directed case (symmetric pairs), identical after pair averaging
                x = v.get(dr)
                if not x:
                    continue
                small = (x.get("all_moe_norm") or [0])[0] is not None and abs((x.get("all_moe_norm") or [0])[0]) < 0.1
                if small:
                    x = dict(x, shared_share=None, shared_over_all=None, routed_over_all=None)
                rows.append({"Cell": X.cell_label(c) + (" †" if v.get("pooled") else ""), "Pairs": v["n_pairs"],
                             "All-MoE / drop (routed + shared)": F(x["all_moe_norm"]), "Routed only / drop": F(x["routed_norm"]),
                             "Shared only / drop": F(x["shared_norm"]), "Shared / (shared + routed)": F(x["shared_share"]),
                             "Shared / all-MoE": F(x["shared_over_all"]), "Routed / all-MoE": F(x["routed_over_all"]),
                             "All − (shared + routed) / drop": F(x["additivity_gap_norm"], 2, True)})
        if rows:
            T["q2s"] = md(pd.DataFrame(rows), "ext14_qwen2_shared", "Qwen2-57B-A14B: every layer's whole MoE output vs only its routed experts vs only the "
                          "shared expert patched at the final token (joint patches, validation pairs; cluster bootstrap)",
                          "Denoising direction (the noising direction is identical after averaging the two directed cases of each symmetric pair). "
                          "Routed-only numbers cover the 64 routed experts only; the all-MoE column is the only 'all experts' value. Ratios are "
                          "n/a where the all-MoE effect is < 0.1 of the drop.")

    def _blk11():  # predictions
        rows = []
        for pid, r in (S.get("predictions") or {}).items():
            obs = []
            for m, v in r["per_model"].items():
                o = v.get("observed")
                if isinstance(o, list) and o and isinstance(o[0], (int, float)):
                    obs.append(f"{m}: {F(o, 2, False)}")
                elif isinstance(o, (int, float)):
                    obs.append(f"{m}: {N(o)}")
                elif isinstance(o, dict) and o and all(isinstance(vv, dict) for vv in o.values()):  # D2: {group: {cell: ci}}
                    obs.append(f"{m}: " + "; ".join(f"{g}: " + ", ".join(f"{X.cell_label(c)} {F(ci_, 2, True)}" for c, ci_ in (vv or {}).items()) for g, vv in o.items()))
                elif pid == "D5" and v.get("trend_14b_minus_4b"):
                    tr = v["trend_14b_minus_4b"]
                    obs.append(f"{m}: 14B − 4B per cell " + ", ".join(f"{X.cell_label(c)} {F(t_, 2, True)}" for c, t_ in tr.items())
                               + "; Spearman with 30B-A3B " + ", ".join(f"{lab(p_)} {N(r_)}" for p_, r_ in (v.get("spearman_vs_30b") or {}).items()))
                elif isinstance(o, dict):
                    obs.append(f"{m}: " + ", ".join(f"{k} {F(vv) if isinstance(vv, list) else (N(vv) if isinstance(vv, (int, float)) or vv is None else vv)}" for k, vv in o.items()
                                                   if not isinstance(vv, dict)))
                if v.get("decision"):
                    obs.append(f"{m}: {v['decision']}")
                if obs and obs[-1].startswith(f"{m}:"):
                    obs[-1] = obs[-1] + f" ({v['verdict']})"
                else:
                    obs.append(f"{m}: {v['verdict']}")
            rows.append({"ID": pid, "Cell": r["cell"], "Prediction": r["prediction"], "Observed": "; ".join(obs), "Verdict": r["verdict"],
                         "Verdict where testable": (f"{r['verdict_testable']} ({r['n_testable']} testable)" if r.get("verdict_testable") else "–")})
        if rows:
            T["pred"] = md(pd.DataFrame(rows), "ext14_prediction_values", "Pre-registered predictions: observed values (p6-analysis; the verdict table is the coordinator's ext14_prereg_check.md)")
    for name, fn in (('direction gap per model', _blkG), ('qwen2 shared', _blkQ), ('behaviour rates', _blkB), ('all cells', _blk0), ('ladder', _blk1), ('Dial E', _blk2), ('Dial A', _blk3), ('Set C', _blk4), ('behaviour-only cells (< 64 pairs overall) and pooled cells: behaviour', _blk5), ('R-metric', _blk6), ('R-site', _blk7), ('hand-off', _blk8), ('Stage 2', _blk9), ('tier 2: E1 (Phase 6) next to E3 (Phase 5 dir_f1 / dir_opt)', _blk10), ('predictions', _blk11)):
        try:
            fn()
        except Exception as ex:  # noqa: BLE001
            log(f"table block '{name}' FAILED: {ex!r}\n{traceback.format_exc()}")
    return T


# ------------------------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------------------------
COL = {"qwen3": "#2a78d6", "mixtral_bos": "#eb6834", "a2e": "#2a78d6", "e2a": "#eb6834", "attn": "#2a78d6", "mlp": "#eb6834", "dense": "#1baf7a"}
INK, INK2, GRID = "#1f2328", "#52514e", "#e4e4e0"


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 110})
    return plt


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}{SFX}.{ext}"), bbox_inches="tight", dpi=150 if ext == "png" else None)
    import matplotlib.pyplot as plt
    plt.close(fig)


def errbar(ax, x, v, color, marker="o", label=None, ms=6, lw=1.4):
    v = nn(v)
    if not v or v[0] is None or not np.isfinite(v[0]):
        return
    lo, hi = (v[1], v[2]) if len(v) >= 3 and np.isfinite(v[1]) else (v[0], v[0])
    ax.errorbar([x], [v[0]], yerr=[[v[0] - lo], [hi - v[0]]], fmt=marker, color=color, ms=ms, lw=lw, capsize=2.5, label=label, zorder=3)


def preds_band(pid):
    try:
        P = X.load_predictions()
        return X.band(P[pid]["prediction"])
    except Exception:  # noqa: BLE001
        return None


def fig_ladder(S, plt):
    L = S.get("ladder") or {}
    ps = [p for p in X.STAGE1 + X.STAGE2 if p in L and L[p].get("rows")]
    if not ps:
        return
    n = len(ps)
    nc = min(n, 3)
    nr = -(-n // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.9 * nr), squeeze=False, sharey=True)
    pb = {"qwen3": {"l_ant": "L1", "l_prop": "L3", "l_bind": "L5", "l_ind": "L7"}, "mixtral_bos": {"l_ant": "L2", "l_prop": "L4", "l_bind": "L6", "l_ind": "L8"}}
    for ax, p in zip(axes.ravel(), ps):
        rows = L[p]["rows"]
        for r in rows:
            jit = {"e3_e2a": -0.12, "l_cf": 0.12, "l_ioi": -0.12, "l_ind": 0.12}.get(r["cell"], 0.0)
            pid = pb.get(p, {}).get(r["cell"])
            if pid:
                b = preds_band(pid)
                if b:
                    lo, hi = (b[0] if b[0] is not None else 0.0), (b[1] if b[1] is not None else 1.0)
                    ax.add_patch(plt.Rectangle((r["grade"] + jit - 0.18, lo), 0.36, hi - lo, color=COL[p] if p in COL else INK2, alpha=0.12, lw=0, zorder=1))
            errbar(ax, r["grade"] + jit, r["share"], COL.get(p, INK))
            ax.annotate(r["label"].replace("WG ", ""), (r["grade"] + jit, r["share"][0]), xytext=(5, -3), textcoords="offset points", fontsize=7.2, color=INK2)
        ax.set_title(f"{X.LABEL.get(p, p)} — Spearman {N(L[p].get('spearman'))}", fontsize=9.5, loc="left")
        ax.set_xlim(0.5, 5.9)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("pre-registered grade (1 = answer only in the weights … 5 = verbatim copy)")
    for ax in axes[:, 0]:
        ax.set_ylabel("attention share (AUC+)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Ladder v2: predicted grade vs measured attention share (shaded: pre-registered bands; validation, 95% CI)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "ext14_ladder")


def fig_dial_e(S, plt):
    D = S.get("dial_e") or {}
    ps = [p for p in X.ALL_PROTOS if p in D and D[p].get("levels")]
    if not ps:
        return
    pb = {("qwen3", "a2e"): {"e1": "E1", "e3": "E5"}, ("mixtral_bos", "a2e"): {"e1": "E2", "e3": "E6"},
          ("qwen3", "e2a"): {"e1": "E7", "e3": "E11"}, ("mixtral_bos", "e2a"): {"e1": "E8", "e3": "E12"}}
    fig, axes = plt.subplots(1, len(ps), figsize=(4.8 * len(ps), 3.9), squeeze=False, sharey=True)
    lvls = list(X.DIAL_E_LEVELS)
    for ax, p in zip(axes[0], ps):
        for dr, off in (("a2e", -0.07), ("e2a", 0.07)):
            lv = D[p]["levels"].get(dr, {})
            seq = list(X.DIAL_LEVELS[dr]) if lv.get("e3p3") else lvls
            xs, ys = [], []
            if seq != lvls and lv.get("e3"):  # a2e: F1 anchor E3 as an open marker next to the frame-matched E3
                b = preds_band(pb.get((p, dr), {}).get("e3", ""))
                if b:
                    lo, hi = (b[0] if b[0] is not None else 0.0), (b[1] if b[1] is not None else 1.0)
                    ax.add_patch(plt.Rectangle((3 - 0.3 - 0.065, lo), 0.13, hi - lo, color=COL[dr], alpha=0.13, lw=0, zorder=1))
                errbar(ax, 3 - 0.3, lv["e3"]["share"], COL[dr], marker="D", ms=5)
                ax.annotate("F1", (3 - 0.3, lv["e3"]["share"][0]), xytext=(-12, 0), textcoords="offset points", fontsize=7, color=INK2, va="center")
            for i, l in enumerate(seq):
                v = lv.get(l)
                pid = pb.get((p, dr), {}).get(l) if l != "e3" or seq == lvls or dr == "e2a" else None
                if pid:
                    b = preds_band(pid)
                    if b:
                        lo, hi = (b[0] if b[0] is not None else 0.0), (b[1] if b[1] is not None else 1.0)
                        ax.add_patch(plt.Rectangle((i + off - 0.065, lo), 0.13, hi - lo, color=COL[dr], alpha=0.13, lw=0, zorder=1))
                if v:
                    errbar(ax, i + off, v["share"], COL[dr])
                    xs.append(i + off); ys.append(v["share"][0])
            if xs:
                ax.plot(xs, ys, color=COL[dr], lw=1.6, zorder=2)
                ax.annotate(X.DIR_LABEL[dr], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
            e0 = lv.get("e0")
            if e0:
                errbar(ax, len(lvls) + off, e0["share"], COL[dr], marker="s")
        ax.set_xticks(range(len(lvls) + 1))
        ax.set_xticklabels(["E1\nverbatim", "E1p\nparaphrase", "E2\nnegation", "E3\nevent", "E0\nprior only"], fontsize=8)
        ax.text(0.01, 0.01, "a→e: E1–E3 in frame P3 (◆ = E3 in the Phase-5 F1 frame)", transform=ax.transAxes, fontsize=7, color=INK2)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-0.4, len(lvls) + 0.9)
        ax.set_title(X.LABEL.get(p, p), fontsize=9.5, loc="left")
    axes[0][0].set_ylabel("attention share (AUC+)")
    fig.suptitle("Dial E on the same WinoGrande twins: attention share per level and direction (shaded: pre-registered bands; own validation sets)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "ext14_dial_e")


def fig_dial_a(S, plt):
    D = S.get("dial_a") or {}
    ps = [p for p in X.ALL_PROTOS if p in D and D[p].get("classes")]
    if not ps:
        return
    fig, axes = plt.subplots(2, len(ps), figsize=(4.8 * len(ps), 7.0), squeeze=False)
    for j, p in enumerate(ps):
        ax = axes[0][j]
        cls = ["E0", "agree", "neutral", "conflict"]
        for dr, off in (("a2e", -0.08), ("e2a", 0.08)):
            c = D[p]["classes"].get(dr, {})
            xs, ys = [], []
            for i, k in enumerate(cls):
                if c.get(k):
                    errbar(ax, i + off, c[k]["share"], COL[dr])
                    xs.append(i + off); ys.append(c[k]["share"][0])
            if xs:
                ax.plot(xs, ys, color=COL[dr], lw=1.4)
                ax.annotate(X.DIR_LABEL[dr], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
        ax.set_xticks(range(4)); ax.set_xticklabels(["E0 prior\nonly", "E1 agrees\nwith prior", "E1 neutral", "E1 conflicts\nwith prior"], fontsize=8)
        ax.set_ylim(0, 1.02); ax.set_xlim(-0.4, 4.0)
        ax.set_title(f"{X.LABEL.get(p, p)}: share by prior class", fontsize=9.5, loc="left")
        ax2 = axes[1][j]
        for dr in X.DIRS:
            ds = (D[p].get("dose") or {}).get(dr)
            if not ds:
                continue
            for b in ds["bins"]:
                errbar(ax2, b["prior_mean"], b["share"], COL[dr], ms=5)
            ax2.plot([b["prior_mean"] for b in ds["bins"]], [b["share"][0] for b in ds["bins"]], color=COL[dr], lw=1.4,
                     label=f"{X.DIR_LABEL[dr]} (Spearman {N(ds.get('spearman'), '+.2f')})")
        ax2.axvline(0, color=INK2, lw=0.8, ls=":")
        ax2.set_xlabel("prior support of the stated answer (probe log-odds; < 0 = conflict)")
        ax2.set_ylabel("attention share")
        ax2.set_ylim(0, 1.02)
        ax2.legend(fontsize=8, loc="best")
        ax2.set_title("dose–response (5 quantile bins, agree + conflict pooled)", fontsize=9, loc="left")
    axes[0][0].set_ylabel("attention share (AUC+)")
    fig.tight_layout()
    save(fig, "ext14_dial_a")


def fig_r_metric(S, plt):
    R = S.get("r_metric") or {}
    ps = [p for p in X.ALL_PROTOS if p in R and R[p].get("cells")]
    if not ps:
        return
    fig, axes = plt.subplots(1, len(ps), figsize=(4.2 * len(ps), 3.9), squeeze=False)
    for ax, p in zip(axes[0], ps):
        cs = R[p]["cells"]
        ld = [v["ld"][0] for v in cs.values()]
        for m, col, mk in (("nld", "#2a78d6", "o"), ("dp", "#eb6834", "s")):
            ys = [v[m][0] if v.get(m) else np.nan for v in cs.values()]
            ax.scatter(ld, ys, color=col, marker=mk, s=26, zorder=3, label={"nld": "normalised LD", "dp": "Δp(answer)"}[m])
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("attention share (LD)"); ax.set_ylabel("attention share (other metric)")
        ax.set_title(f"{X.LABEL.get(p, p)}: ρ nLD {N(R[p].get('spearman_ld_nld'))}, ρ Δp {N(R[p].get('spearman_ld_dp'))}", fontsize=9, loc="left")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "ext14_r_metric")


def fig_handoff(S, plt):
    H = S.get("handoff") or {}
    ps = [p for p in X.STAGE1 if H.get(p)]
    if not ps:
        return
    fig, axes = plt.subplots(1, len(ps), figsize=(5.4 * len(ps), 4.2), squeeze=False)
    for ax, p in zip(axes[0], ps):
        rows = sorted(H[p], key=lambda h: cell_order(h["cell"]))
        for i, h in enumerate(rows):
            a, b = nn(h.get("swap_last_ge_0.5")), nn(h.get("final_first_ge_0.5"))
            if a and b and np.isfinite(a[0]) and np.isfinite(b[0]):
                ax.plot([a[0], b[0]], [i, i], color=INK2, lw=1.0)
                ax.plot([a[0]], [i], "o", color=COL["e2a"], ms=5)
                ax.plot([b[0]], [i], "s", color=COL["a2e"], ms=5)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([X.cell_label(h["cell"]) for h in rows], fontsize=7.5)
        ax.set_xlabel("layer"); ax.set_xlim(0, (rows[0]["L"] if rows else 48))
        ax.set_title(f"{X.LABEL.get(p, p)}: ● swapped token ≥ 0.5 until · ■ final token ≥ 0.5 from", fontsize=9, loc="left")
    fig.tight_layout()
    save(fig, "ext14_handoff")


def fig_dense(S, plt):
    tw = S.get("twins") or {}
    for key, t in tw.items():
        if not t or "cells" not in t:
            continue
        cells = sorted(t["cells"], key=cell_order)
        n = len(cells)
        nc = min(5, n)
        nr = -(-n // nc)
        fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.7 * nr), squeeze=False, sharex=True)
        for ax, c in zip(axes.ravel(), cells):
            v = t["cells"][c]
            for nm, A, M, Lx, ls in (("dense", v["curve_dense_attn"], v["curve_dense_mlp"], v["L_dense"], "--"), ("MoE", v["curve_moe_attn"], v["curve_moe_moe"], v["L_moe"], "-")):
                if not A:
                    continue
                x = np.arange(len(A)) / len(A)  # l / L
                ax.plot(x, A, color=COL["attn"], ls=ls, lw=1.3)
                ax.plot(x, M, color=COL["mlp"], ls=ls, lw=1.3)
            pp = v["paired"]["share"]
            ax.set_title(f"{X.cell_label(c)}\nshare {N(pp['b'])} → {N(pp['a'])}", fontsize=8, loc="left")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], color=COL["attn"], lw=1.3), Line2D([], [], color=COL["mlp"], lw=1.3), Line2D([], [], color=INK2, ls="--"), Line2D([], [], color=INK2, ls="-")]
        fig.legend(hs, ["attention output", "MLP / MoE output", f"dense ({lab(t['dense'])})", f"MoE ({lab(t['moe'])})"], loc="upper center", ncol=4, fontsize=8.5,
                   bbox_to_anchor=(0.5, 1.02))
        fig.supxlabel("depth l / L", fontsize=9)
        fig.supylabel("rescue / drop (single-layer patch, final token)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save(fig, f"ext14_dense_curves_{t['moe']}")
    # arrows
    if tw:
        fig, axes = plt.subplots(1, len(tw), figsize=(5.2 * len(tw), 4.6), squeeze=False)
        for ax, (key, t) in zip(axes[0], tw.items()):
            if not t or "cells" not in t:
                continue
            cells = sorted(t["cells"], key=lambda c: t["cells"][c]["dense_own"][0])
            for i, c in enumerate(cells):
                a_, b_ = t["cells"][c]["moe_own"], t["cells"][c]["dense_own"]
                ax.annotate("", xy=(a_[0], i), xytext=(b_[0], i), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
                ax.plot([b_[0]], [i], "o", color=COL["dense"], ms=5)
                ax.plot([a_[0]], [i], "o", color=COL["mlp"], ms=5)
            ax.set_yticks(range(len(cells))); ax.set_yticklabels([X.cell_label(t["cells"][c].get("variant", c)) for c in cells], fontsize=7.5)
            ax.axvspan(0, 1, color="none")
            ax.set_xlim(0, 1)
            ax.set_xlabel("attention share")
            ax.set_title(f"{lab(t['dense'])} (aqua) → {lab(t['moe'])} (orange), own validation sets", fontsize=9, loc="left")
        fig.tight_layout()
        save(fig, "ext14_dense_arrows")


def fig_series(S, plt):
    se = S.get("series") or {}
    for nm, v in se.items():
        if not v or "cells" not in v:
            continue
        ps = v["protos"]
        cells = sorted(v["cells"], key=cell_order)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        xs = [np.log10(X.PARAMS_B.get(p, np.nan)) for p in ps]
        for c in cells:
            sh = v["cells"][c].get("own") or v["cells"][c]["share"]
            pts = [(np.log10(X.PARAMS_B[p]), sh[p][0]) for p in ps if p in sh and sh[p]]
            if len(pts) < 2:
                continue
            ax.plot([a for a, _ in pts], [b for _, b in pts], color=INK2, lw=1.0, marker="o", ms=4)
            ax.annotate(X.cell_label(c), pts[-1], xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK2, va="center")
        ax.set_xticks(xs); ax.set_xticklabels([lab(p) for p in ps], fontsize=8)
        ax.set_ylim(0, 1.02); ax.set_ylabel("attention share (each model's own validation set)")
        ax.set_title(f"Capacity series '{nm}' (x: total parameters, log scale)", fontsize=9.5, loc="left")
        fig.tight_layout()
        save(fig, f"ext14_capacity_{nm}")


def figures(S):
    plt = plt_setup()
    for fn in (fig_ladder, fig_dial_e, fig_dial_a, fig_r_metric, fig_handoff, fig_dense, fig_series):
        try:
            fn(S, plt)
        except Exception as ex:  # noqa: BLE001
            log(f"{fn.__name__} failed: {ex!r}\n{traceback.format_exc()}")


# ------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--no-sections", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        t0 = time.time()
        S = compute(args.force)
        with open(SUMM + ".tmp", "w") as f:
            json.dump(S, f, indent=1)
        os.replace(SUMM + ".tmp", SUMM)
        log(f"wrote {SUMM}: {len(S['cells'])} cells, protos {S['protos']}")
        S = denull(S)
        tables(S)
        if not args.no_figs:
            figures(S)
        if not args.no_sections:
            try:
                import ext14_analyze_sections as SEC
                SEC.write_all(S, SECT, SFX)
            except Exception as ex:  # noqa: BLE001
                log(f"sections failed: {ex!r}\n{traceback.format_exc()}")
        log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
