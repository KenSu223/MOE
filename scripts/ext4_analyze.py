"""Direction 4 analysis: per-category localisation on CodeFact and cross-category overlap, for every finished run.

Runs are results/codefact_<model>_<protocol>/ with scan_*.parquet (ext4_scan.py), case_sets.json + sweep_*.parquet
(ext4_select.py) and expert_rows.parquet (ext4_run_expert.py). Everything is post-processing through the public APIs of
moetrace.analysis and moetrace.ext1_analysis.

Usage: python scripts/ext4_analyze.py [--runs codefact_qwen3_raw,codefact_mixtral_nobos,codefact_qwen3_coder_raw,codefact_qwen3_coder_chat]
Writes results/tables/ext4_*.{md,csv}, results/figures/ext4_*.png, results/ext4_summary.json, results/sections/ext4_codefact.md.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import analysis as A
from moetrace import ext1_analysis as X
from moetrace.models import MODELS, RESULTS
from moetrace.report import md_table
from moetrace.ext4_data import CATEGORIES, CATEGORY_LABEL

TAB, FIG, SEC = (os.path.join(RESULTS, d) for d in ("tables", "figures", "sections"))
for d in (TAB, FIG, SEC):
    os.makedirs(d, exist_ok=True)
RUNS = {
    "codefact_qwen3_raw": dict(model="qwen3", label="Qwen3-30B-A3B-Base (tokenizer defaults)", short="qwen3",
                               factual=[(44, 69), (42, 115)], factual_note="L44E069 (paper), L42E115 (ext1 second locus)"),
    "codefact_mixtral_nobos": dict(model="mixtral", label="Mixtral-8x7B-v0.1 (no BOS, paper protocol)", short="mixtral",
                                   factual=[(19, 6), (19, 2), (18, 1)], factual_note="L19E006 (paper), L19E002 (BOS run), L18E001 (ext1 joint winner)"),
    "codefact_qwen3_coder_raw": dict(model="qwen3_coder", label="Qwen3-Coder-30B-A3B-Instruct (raw code prefix)", short="coder_raw",
                                     factual=[(44, 69), (42, 115)], factual_note="L44E069 / L42E115 (Qwen3-Base factual experts)"),
    "codefact_qwen3_coder_chat": dict(model="qwen3_coder", label="Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence)", short="coder_chat",
                                      factual=[(44, 69), (42, 115)], factual_note="L44E069 / L42E115 (Qwen3-Base factual experts)"),
}
SETS = list(CATEGORIES) + ["all"]
RECURRENCE = 64


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def ci(s):
    return "n/a" if s is None or s.get("n", 0) == 0 else f"{s['mean']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"


def ci3(m, lo, hi):
    return "n/a" if m is None or (isinstance(m, float) and np.isnan(m)) else f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if not isinstance(v, pd.DataFrame)}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else np.nan


# ---------------------------------------------------------------------------------------------------------------
def run_cfg(run: str) -> dict:
    """Config for a run: the registry entry, or a default built from scan_meta.json (ad-hoc / smoke runs)."""
    if run in RUNS:
        return RUNS[run]
    meta = json.load(open(os.path.join(RESULTS, run, "scan_meta.json")))
    base = {"qwen3": RUNS["codefact_qwen3_raw"], "qwen3_coder": RUNS["codefact_qwen3_coder_raw"], "mixtral": RUNS["codefact_mixtral_nobos"]}.get(meta["model"], RUNS["codefact_qwen3_raw"])
    cfg = dict(base, label=f"{MODELS[meta['model']]['label']} ({meta['protocol']}; run {run})", short=run.replace("codefact_", ""), model=meta["model"])
    RUNS[run] = cfg
    return cfg


def analyze_run(run: str) -> dict | None:
    cfg = run_cfg(run)
    od = os.path.join(RESULTS, run)
    if not os.path.exists(os.path.join(od, "case_sets.json")):
        log(f"{run}: no case_sets.json (scan/select not done)")
        return None
    md = A.load_model(run)
    nc = MODELS[cfg["model"]]["n_controls"]
    sets = md.sets
    scan = pd.read_parquet(os.path.join(od, "scan_cases.parquet"))
    meta = json.load(open(os.path.join(od, "scan_meta.json")))
    out = {"run": run, "label": cfg["label"], "model": cfg["model"], "protocol": meta["protocol"], "n_scanned": int(len(scan)),
           "scan_pass": {c: int(scan[(scan.category == c)].strict.sum()) for c in CATEGORIES}, "sets": {}, "has_expert": md.expert_rows is not None}
    names = [s for s in SETS if s in sets and not sets[s].get("skipped")]
    cache = X.ExpertCache(md) if md.expert_rows is not None else None
    n_layers = int(md.routing.layer.nunique())
    for s in names:
        disc, val = md.ids(s, "discovery"), md.ids(s, "validation")
        la = A.layer_analysis(md, s)
        L = la["L_star"]
        th = min(RECURRENCE, len(disc) // 2)
        d = {"n_disc": len(disc), "n_val": len(val), "partial": bool(sets[s].get("partial")), "n_pass": sets[s].get("n_pass"),
             "L_star": L, "layer_val": la["val_at_Lstar"], "layer_curve": la["curve"], "sharpness": la["sharpness"], "threshold": th,
             "disc_curve_top5": la["disc_curve_top5"]}
        if s == "all":
            d["composition"] = sets[s].get("composition")
        # sink diagnostic on the set's cases
        sc = scan.set_index("case_id").loc[disc + val]
        d["sink_final_is_max_L5"] = float(sc.final_is_max_L5.mean()) if "final_is_max_L5" in sc else None
        d["top1_clean_is_true"] = float(sc.top1_clean_is_true.mean())
        d["delta_clean_mean"] = float(sc.delta_clean.mean())
        d["drop_mean"] = float(sc["drop"].mean())
        d["n_tokens_mean"] = float(sc.n_tokens.mean())
        # interior selection: the paper's layer rule restricted to layers <= L-5 (the last MoE blocks act as a read-out on code)
        R = md.R
        interior_layers = [l for l in R.columns if l <= n_layers - 5]
        dm = R.loc[disc, interior_layers].mean(0)
        Li = int(dm.idxmax())
        d["L_interior"] = Li
        d["layer_val_interior"] = A.summarize(R.loc[val, Li].values)
        d["last_layer_val"] = A.summarize(R.loc[val, n_layers - 1].values, with_p=False)
        if cache is not None and L in cache.piv:
            ei, sti = X.select_at_layer(cache, Li, disc, th)
            if ei is not None:
                evi = A.evaluate_expert(md, Li, ei, val, nc)
                d.update(e_interior=ei, pair_interior=X.pair(Li, ei), disc_active_interior=int(sti[sti.expert == ei].disc_active.iloc[0]),
                         val_active_interior=evi["val_active"], rescue_interior=evi["rescue_all"], spec_interior=evi["spec_all"])
                rki = A.all_active_rank(md, Li, ei, val)
                d["rank_interior"] = {k: v for k, v in rki.items() if k != "per_case"}
            else:
                d.update(e_interior=None, pair_interior=None, max_activity_interior=int(sti.disc_active.max()) if len(sti) else 0)
            d["coalitions_interior"] = A.coalitions(md, Li, val)
            # joint search restricted to interior layers
            cand_all = X.joint_candidates(cache, disc, th)
            ci_ = cand_all[cand_all.layer <= n_layers - 5]
            if len(ci_):
                w = ci_.iloc[0]
                wi = X.evaluate_pair(md, cache, int(w.layer), int(w.expert), disc, val, nc)
                wi["rank_all_layers"] = int(w["rank"])
                d["joint_interior_winner"] = wi
                d["joint_interior_top"] = pd.DataFrame([dict(X.evaluate_pair(md, cache, int(r.layer), int(r.expert), disc, val, nc), rank=int(r["rank"]))
                                                        for _, r in ci_.head(10).iterrows()])
            else:
                d["joint_interior_winner"] = None
                d["joint_interior_top"] = pd.DataFrame()
            e, st = X.select_at_layer(cache, L, disc, th)
            d["n_candidates"] = int((st.disc_active >= th).sum())
            d["max_activity"] = int(st.disc_active.max()) if len(st) else 0
            if e is not None:
                ev = A.evaluate_expert(md, L, e, val, nc)
                d.update(e_star=e, pair=X.pair(L, e), disc_active=int(st[st.expert == e].disc_active.iloc[0]),
                         disc_allcase_mean=float(st[st.expert == e].disc_allcase_mean.iloc[0]), val_active=ev["val_active"],
                         rescue=ev["rescue_all"], spec=ev["spec_all"], control=ev["control_all"], rescue_active=ev["rescue_active"], spec_active=ev["spec_active"])
                rk = A.all_active_rank(md, L, e, val)
                d["rank"] = {k: v for k, v in rk.items() if k != "per_case"}
            else:
                d.update(e_star=None, pair=None)
            d["coalitions"] = A.coalitions(md, L, val)
            two = (L, e if e is not None else -1)
            js = X.joint_search(md, cache, s, nc, two, threshold=th, top_k=10)
            d["joint"] = {k: v for k, v in js.items() if k not in ("top", "candidates", "all_evaluated")}
            d["joint_top"] = js["top"]
            d["candidates"] = js["candidates"]
            d["all_evaluated"] = js["all_evaluated"]
            d["n_candidates_all_layers"] = int(len(js["candidates"]))
            d["candidate_layers"] = sorted(js["candidates"].layer.unique().tolist())
            # factual experts on this category
            fac = []
            for (fl, fe) in cfg["factual"]:
                if fl in cache.piv and fe in cache.piv[fl].columns:
                    r = X.evaluate_pair(md, cache, fl, fe, disc, val, nc)
                    m = js["candidates"][(js["candidates"].layer == fl) & (js["candidates"].expert == fe)]
                    r["recurrent"] = bool(len(m))
                    r["joint_rank"] = int(m["rank"].iloc[0]) if len(m) else None
                else:
                    r = dict(layer=fl, expert=fe, pair=X.pair(fl, fe), disc_active=0, val_active=0, val_rescue=0.0, val_spec=np.nan, recurrent=False, joint_rank=None)
                fac.append(r)
            d["factual"] = fac
            d["per_layer"] = X.per_layer_curve(md, cache, s, nc, threshold=th)
        out["sets"][s] = d
        msg = f"{run} {s}: n={len(disc)}+{len(val)} L*={L} block {ci(la['val_at_Lstar'])}"
        if "e_star" in d and d["e_star"] is not None:
            msg += f" e*={d['pair']} act {d['disc_active']}/{len(disc)} rescue {ci(d['rescue'])} spec {ci(d['spec'])} joint {d['joint']['winner']['pair'] if d['joint']['winner'] else None}"
        log(msg)
    out["n_layers"] = n_layers
    return out


# ---------------------------------------------------------------------------------------------------------------
def cross_category(out: dict) -> dict:
    """Jaccard overlaps of recurrent (layer, expert) pairs and of the top-10 joint pairs between categories; layer bands."""
    sets = {s: d for s, d in out["sets"].items() if "candidates" in d}
    cats = [s for s in sets]
    rec = {s: set(zip(sets[s]["candidates"].layer.astype(int), sets[s]["candidates"].expert.astype(int))) for s in cats}
    top = {s: set(zip(sets[s]["joint_top"].layer.astype(int), sets[s]["joint_top"].expert.astype(int))) if len(sets[s]["joint_top"]) else set() for s in cats}
    J_rec = pd.DataFrame([[jaccard(rec[a], rec[b]) for b in cats] for a in cats], index=cats, columns=cats)
    J_top = pd.DataFrame([[jaccard(top[a], top[b]) for b in cats] for a in cats], index=cats, columns=cats)
    # recurrent pairs at the selected layer only
    recL = {s: set(int(e) for e in sets[s]["candidates"][sets[s]["candidates"].layer == sets[s]["L_star"]].expert) for s in cats}
    common_all = set.intersection(*[rec[s] for s in cats if s != "all"]) if len([s for s in cats if s != "all"]) > 1 else set()
    counts = {}
    for s in cats:
        if s == "all":
            continue
        for p in rec[s]:
            counts[p] = counts.get(p, 0) + 1
    shared = sorted([(p, n) for p, n in counts.items() if n >= 2], key=lambda x: (-x[1], x[0]))
    nl = None
    for s in cats:
        nl = max(int(sets[s]["candidates"].layer.max()) + 1 if len(sets[s]["candidates"]) else 0, nl or 0)
    rec_int = {s: set(p for p in rec[s] if p[0] <= nl - 5) for s in cats}
    J_int = pd.DataFrame([[jaccard(rec_int[a], rec_int[b]) for b in cats] for a in cats], index=cats, columns=cats)
    top_int = {s: (set(zip(sets[s]["joint_interior_top"].layer.astype(int), sets[s]["joint_interior_top"].expert.astype(int))) if len(sets[s].get("joint_interior_top", [])) else set()) for s in cats}
    J_top_int = pd.DataFrame([[jaccard(top_int[a], top_int[b]) for b in cats] for a in cats], index=cats, columns=cats)
    return {"categories": cats, "jaccard_recurrent": J_rec, "jaccard_top10": J_top, "jaccard_recurrent_interior": J_int, "jaccard_top10_interior": J_top_int,
            "recurrent_pairs_interior": {s: sorted(rec_int[s]) for s in cats}, "top10_interior": {s: sorted(top_int[s]) for s in cats},
            "L_interior": {s: sets[s].get("L_interior") for s in cats}, "winner_interior": {s: (sets[s]["joint_interior_winner"]["pair"] if sets[s].get("joint_interior_winner") else None) for s in cats},
            "two_stage_interior": {s: sets[s].get("pair_interior") for s in cats},
            "recurrent_pairs": {s: sorted(rec[s]) for s in cats},
            "top10_pairs": {s: sorted(top[s]) for s in cats}, "recurrent_at_Lstar": {s: sorted(recL[s]) for s in cats},
            "n_recurrent": {s: len(rec[s]) for s in cats}, "common_to_all_categories": sorted(common_all), "shared_pairs": shared[:40],
            "L_star": {s: sets[s]["L_star"] for s in cats}, "winner": {s: (sets[s]["joint"]["winner"]["pair"] if sets[s]["joint"]["winner"] else None) for s in cats},
            "two_stage": {s: sets[s].get("pair") for s in cats}}


def n_last(out: dict) -> int:
    return int(out["n_layers"]) - 1


def subcategory_table(out: dict) -> str:
    """Calibration per sub-category (pass rates under the paper filter and the relative rule, top-1 agreement)."""
    run, short = out["run"], RUNS[out["run"]]["short"]
    scan = pd.read_parquet(os.path.join(RESULTS, run, "scan_cases.parquet"))
    rows = []
    for (cat, sub), g in scan.groupby(["category", "subcategory"]):
        if len(g) < 10:
            continue
        rows.append([cat, sub, len(g), f"{g.delta_clean.median():+.2f}", f"{g['drop'].median():+.2f}", f"{100 * g.strict.mean():.0f} %", f"{100 * g.relative25.mean():.0f} %",
                     f"{100 * g.top1_clean_is_true.mean():.0f} %", f"{100 * g.top1_clean_startswith_true.mean():.0f} %" if "top1_clean_startswith_true" in g else "n/a",
                     f"{g.n_tokens.mean():.0f}", f"{g.n_subject_tokens.mean():.1f}"])
    return md_table(["Category", "sub-category", "n", "median Δ_clean", "median drop", "paper filter pass", "relative-25 % pass", "top-1 = true", "top-1 starts with true",
                     "mean prefix tokens", "mean subject tokens"], rows, os.path.join(TAB, f"ext4_calibration_subcategories_{short}"),
                    f"{out['label']}: calibration per sub-category (sub-categories with >= 10 scanned items)")


def write_tables(out: dict, cc: dict | None):
    run, short = out["run"], RUNS[out["run"]]["short"]
    subcategory_table(out)
    rows = []
    for s, d in out["sets"].items():
        name = f"{s} {CATEGORY_LABEL.get(s, 'mixed')}" + (" (partial)" if d["partial"] else "")
        r = [name, f"{d['n_disc']}+{d['n_val']}" + (f" of {d['n_pass']} pass" if d["partial"] else ""), f"{100 * d['top1_clean_is_true']:.0f} %",
             f"{d['delta_clean_mean']:+.2f} / {d['drop_mean']:+.2f}", f"L{d['L_star']}", ci(d["layer_val"]), f"L{d['sharpness']['next_layer']} {d['sharpness']['next_rescue']:+.3f}"]
        if d.get("e_star") is not None:
            r += [d["pair"], f"{d['disc_active']}/{d['n_disc']} / {d['val_active']}/{d['n_val']}", ci(d["rescue"]), ci(d["spec"]),
                  f"{d['rank']['top1']}/{d['rank']['n']}" if d.get("rank") else "n/a",
                  f"{f3(d['coalitions']['coalition_clean']['mean'])} / {f3(d['coalitions']['coalition_union']['mean'])}",
                  (d["joint"]["winner"]["pair"] + (" (= two-stage)" if d["joint"]["same_as_two_stage"] else f" {ci3(d['joint']['winner']['val_rescue'], d['joint']['winner']['val_rescue_lo'], d['joint']['winner']['val_rescue_hi'])}, Spec {f3(d['joint']['winner']['val_spec'])}")) if d["joint"]["winner"] else "none",
                  str(d["n_candidates_all_layers"])]
        elif "coalitions" in d:
            r += [f"none (max activity {d.get('max_activity')}/{d['n_disc']} < {d['threshold']})", "", "", "", "",
                  f"{f3(d['coalitions']['coalition_clean']['mean'])} / {f3(d['coalitions']['coalition_union']['mean'])}",
                  (d["joint"]["winner"]["pair"] + f" {ci3(d['joint']['winner']['val_rescue'], d['joint']['winner']['val_rescue_lo'], d['joint']['winner']['val_rescue_hi'])}, Spec {f3(d['joint']['winner']['val_spec'])}") if d["joint"]["winner"] else "none",
                  str(d["n_candidates_all_layers"])]
        else:
            r += ["(expert pass pending)"] + [""] * 7
        rows.append(r)
    hdr = ["Set", "n (disc+val)", "top-1 = true", "mean Δ_clean / drop", "L*", "block rescue (val)", "2nd layer", "e* (two-stage)", "active disc / val",
           "expert rescue (val)", "Spec (active-random)", "rank-1 among active", "coalition clean / union", "joint winner (all layers)", "recurrent pairs (all layers)"]
    t = md_table(hdr, rows, os.path.join(TAB, f"ext4_summary_{short}"), f"{out['label']}: per-category localisation on CodeFact (validation split; recurrence threshold {RECURRENCE} of 128)")
    tabs = {"summary": t}
    # interior selection (layers <= L-5)
    rows = []
    for s, d in out["sets"].items():
        if "L_interior" not in d:
            continue
        r = [s, f"L{n_last(out)}: {ci(d['last_layer_val'])}", f"L{d['L_interior']}", ci(d["layer_val_interior"])]
        if d.get("e_interior") is not None:
            r += [d["pair_interior"], f"{d['disc_active_interior']}/{d['n_disc']} / {d['val_active_interior']}/{d['n_val']}", ci(d["rescue_interior"]), ci(d["spec_interior"]),
                  f"{d['rank_interior']['top1']}/{d['rank_interior']['n']}",
                  f"{f3(d['coalitions_interior']['coalition_clean']['mean'])} / {f3(d['coalitions_interior']['coalition_union']['mean'])}"]
        elif "coalitions_interior" in d:
            r += [f"none (max activity {d.get('max_activity_interior')}/{d['n_disc']})", "", "", "", "",
                  f"{f3(d['coalitions_interior']['coalition_clean']['mean'])} / {f3(d['coalitions_interior']['coalition_union']['mean'])}"]
        else:
            r += ["(expert pass pending)"] + [""] * 5
        w = d.get("joint_interior_winner")
        r.append((w["pair"] + f" {ci3(w['val_rescue'], w['val_rescue_lo'], w['val_rescue_hi'])}, Spec {ci3(w['val_spec'], w['val_spec_lo'], w['val_spec_hi'])} (rank {w['rank_all_layers']} over all layers)") if w else ("none" if "coalitions_interior" in d else ""))
        rows.append(r)
    if rows:
        tabs["interior"] = md_table(["Set", "last-layer block rescue (val)", "L* interior (≤ L−5)", "block rescue (val)", "e* interior", "active disc / val", "expert rescue (val)",
                                     "Spec (active-random)", "rank-1 among active", "coalition clean / union", "joint winner (interior layers)"], rows,
                                    os.path.join(TAB, f"ext4_interior_{short}"), f"{out['label']}: the same selection restricted to interior layers (excluding the last 4 MoE blocks, whose patch acts as a read-out on code)")
    # factual experts
    rows = []
    for s, d in out["sets"].items():
        for r in d.get("factual", []):
            rows.append([s, r["pair"], f"{r['disc_active']}/{d['n_disc']}", "yes" if r["recurrent"] else "no", r["joint_rank"] if r["joint_rank"] else "-",
                         f"{r['val_active']}/{d['n_val']}", ci3(r["val_rescue"], r.get("val_rescue_lo", np.nan), r.get("val_rescue_hi", np.nan)) if r["val_active"] else "0 (never active)",
                         ci3(r["val_spec"], r.get("val_spec_lo", np.nan), r.get("val_spec_hi", np.nan)) if r["val_active"] else "n/a"])
    if rows:
        tabs["factual"] = md_table(["Set", "factual expert", "disc active", "recurrent (≥ threshold)", "joint rank", "val active", "val rescue", "val Spec"], rows,
                                   os.path.join(TAB, f"ext4_factual_experts_{short}"), f"{out['label']}: the factual-recall experts ({RUNS[run]['factual_note']}) on the code categories")
    # joint top-10 per set
    rows = []
    for s, d in out["sets"].items():
        if "joint_top" not in d:
            continue
        for _, r in d["joint_top"].iterrows():
            rows.append([s, int(r["rank"]), r["pair"], f"{int(r.disc_active)}/{d['n_disc']}", f3(r.disc_allcase_mean), f"{int(r.val_active)}/{d['n_val']}",
                         ci3(r.val_rescue, r.val_rescue_lo, r.val_rescue_hi), ci3(r.val_spec, r.val_spec_lo, r.val_spec_hi), "yes" if r.is_two_stage else ""])
    if rows:
        tabs["top10"] = md_table(["Set", "rank", "pair", "disc active", "disc all-case mean", "val active", "val rescue", "val Spec", "two-stage"], rows,
                                 os.path.join(TAB, f"ext4_joint_top10_{short}"), f"{out['label']}: top-10 recurrent (layer, expert) pairs per category by discovery all-case rescue")
    # cross-category
    if cc is not None:
        cats = cc["categories"]
        rows = [[a] + [f"{cc['jaccard_recurrent'].loc[a, b]:.2f}" for b in cats] for a in cats]
        tabs["jaccard_rec"] = md_table(["recurrent pairs: Jaccard"] + cats, rows, os.path.join(TAB, f"ext4_jaccard_recurrent_{short}"),
                                       f"{out['label']}: Jaccard overlap of the recurrent (layer, expert) pairs (all layers, discovery activity ≥ threshold) between categories")
        rows = [[a] + [f"{cc['jaccard_top10'].loc[a, b]:.2f}" for b in cats] for a in cats]
        tabs["jaccard_top"] = md_table(["top-10 joint pairs: Jaccard"] + cats, rows, os.path.join(TAB, f"ext4_jaccard_top10_{short}"),
                                       f"{out['label']}: Jaccard overlap of the top-10 joint (layer, expert) pairs between categories")
        rows = [[a] + [f"{cc['jaccard_recurrent_interior'].loc[a, b]:.2f}" for b in cats] for a in cats]
        tabs["jaccard_rec_int"] = md_table(["recurrent pairs, interior layers: Jaccard"] + cats, rows, os.path.join(TAB, f"ext4_jaccard_recurrent_interior_{short}"),
                                           f"{out['label']}: Jaccard overlap of the recurrent (layer, expert) pairs restricted to interior layers (≤ L−5)")
        rows = [[a] + [f"{cc['jaccard_top10_interior'].loc[a, b]:.2f}" for b in cats] for a in cats]
        tabs["jaccard_top_int"] = md_table(["top-10 interior joint pairs: Jaccard"] + cats, rows, os.path.join(TAB, f"ext4_jaccard_top10_interior_{short}"),
                                           f"{out['label']}: Jaccard overlap of the top-10 joint pairs restricted to interior layers")
        rows = [[s, f"L{cc['L_star'][s]}", cc["two_stage"][s] or "none", cc["winner"][s] or "none", f"L{cc['L_interior'][s]}", cc["two_stage_interior"][s] or "none",
                 cc["winner_interior"][s] or "none", cc["n_recurrent"][s], len(cc["recurrent_pairs_interior"][s]),
                 ", ".join(f"E{e:03d}" for e in cc["recurrent_at_Lstar"][s]) or "-"] for s in cats]
        tabs["bands"] = md_table(["Set", "L* (paper rule)", "two-stage e*", "joint winner (all layers)", "L* interior", "two-stage e* interior", "joint winner (interior)",
                                  "recurrent pairs (all layers)", "recurrent pairs (interior)", "recurrent experts at L*"], rows,
                                 os.path.join(TAB, f"ext4_layer_bands_{short}"), f"{out['label']}: selected layers, experts and recurrent sets per category")
        # pairs recurrent in >= 2 categories, interior layers
        counts = {}
        for s in cats:
            if s == "all":
                continue
            for p in cc["recurrent_pairs_interior"][s]:
                counts[tuple(p)] = counts.get(tuple(p), 0) + 1
        shared_int = sorted([(p, n) for p, n in counts.items() if n >= 2], key=lambda x: (-x[1], x[0]))[:40]
        rows = [[X.pair(*p), n, ", ".join(s for s in cats if s != "all" and tuple(p) in set(map(tuple, cc["recurrent_pairs_interior"][s])))] for p, n in shared_int]
        if rows:
            tabs["shared_int"] = md_table(["pair (interior)", "n categories", "categories"], rows, os.path.join(TAB, f"ext4_shared_recurrent_interior_{short}"),
                                          f"{out['label']}: interior (layer, expert) pairs recurrent in ≥ 2 categories")
        rows = [[X.pair(*p), n, ", ".join(s for s in cats if s != "all" and p in set(cc["recurrent_pairs"][s]))] for p, n in cc["shared_pairs"]]
        if rows:
            tabs["shared"] = md_table(["pair", "n categories", "categories"], rows, os.path.join(TAB, f"ext4_shared_recurrent_{short}"),
                                      f"{out['label']}: (layer, expert) pairs recurrent in ≥ 2 categories")
    # per-layer curves csv
    for s, d in out["sets"].items():
        if "per_layer" in d:
            d["per_layer"].to_csv(os.path.join(TAB, f"ext4_layer_curve_{short}_{s}.csv"), index=False)
    return tabs


# ---------------------------------------------------------------------------------------------------------------
def figure_curves(out: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
    COL = {"S1": "#2a78d6", "S2": "#3d9bd1", "S3": "#5cc0c0", "R1": "#eb6834", "R2": "#d9483b", "R3": "#b0648f", "all": "#52514e"}
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    names = list(out["sets"].keys())
    fig, axes = plt.subplots(1, len(names), figsize=(2.9 * len(names), 3.3), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], names):
        d = out["sets"][s]
        cur = pd.DataFrame(d["layer_curve"])
        c = COL.get(s, INK2)
        ax.fill_between(cur.layer, cur.ci_lo, cur.ci_hi, color=c, alpha=0.15, lw=0)
        ax.plot(cur.layer, cur.val_mean, color=c, lw=1.5, label="MoE-block rescue")
        if "per_layer" in d:
            pl = d["per_layer"]
            ax.plot(pl.layer, pl.val_rescue, color=INK, lw=1.0, marker="o", ms=2.0, label="best recurrent expert")
            ax.plot(pl.layer, pl.val_spec, color=INK, lw=0.8, ls="--", alpha=0.7, label="its Spec")
        ax.axvline(d["L_star"], color=INK2, lw=0.6, ls=":")
        if "L_interior" in d:
            ax.axvline(d["L_interior"], color=c, lw=0.6, ls="--")
        ax.axhline(0, color=INK2, lw=0.5)
        ttl = f"{s} {CATEGORY_LABEL.get(s, 'mixed')}\nn={d['n_disc']}+{d['n_val']}{' partial' if d['partial'] else ''}, L*={d['L_star']}"
        if d.get("e_star") is not None:
            ttl += f", {d['pair']}"
        ax.set_title(ttl, fontsize=8, loc="left")
        ax.set_xlabel("MoE layer")
        ax.grid(axis="y", color=GRID, lw=0.5)
    axes[0][0].set_ylabel("validation rescue (Δ patched − Δ noised)")
    h, l = axes[0][0].get_legend_handles_labels()
    axes[0][0].legend(h, l, frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(f"{out['label']}: layer curves per CodeFact category (validation means, 95 % bootstrap band for the block)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_overlap(cc: dict, label: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = cc["categories"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, key, ttl in ((axes[0], "jaccard_recurrent", "recurrent (layer, expert) pairs, all layers"), (axes[1], "jaccard_top10", "top-10 joint pairs")):
        M = cc[key].values.astype(float)
        im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(cats)))
        ax.set_yticks(range(len(cats)))
        ax.set_xticklabels(cats)
        ax.set_yticklabels(cats)
        for i in range(len(cats)):
            for j in range(len(cats)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if M[i, j] > 0.6 else "#0b0b0b")
        ax.set_title(f"Jaccard: {ttl}", fontsize=9, loc="left")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle(f"{label}: expert overlap between CodeFact categories", fontsize=9, x=0.01, ha="left")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def write_section(summary: dict, path: str):
    """Assemble the markdown section from the run summaries and the calibration/build tables on disk."""
    P = []
    P.append("## Direction 4: expert-aware tracing on code (CodeFact)\n")
    P.append("**Question.** Does the paper's expert-aware causal tracing transfer from factual recall to code, where the 'fact' is syntactic "
             "(matching bracket, block keyword) or semantic (variable, API, constant recall)? Do the categories localise to the same layers and experts, "
             "and do the factual-recall experts (Qwen3 L44E069 / L42E115, Mixtral L19E006 / L19E002 / L18E001) play any role?\n")
    P.append("**Dataset (Phase A, `data/codefact/`).** CounterFact-style next-token counterfactuals built from Python with `ast`/`tokenize` "
             "(`scripts/ext4_build_codefact.py`; items in `items.jsonl`, yields and licences in `build_stats.md`, 20 eyeballed items per category in "
             "`samples.md`). Sources: HumanEval canonical solutions (MIT), MBPP (CC-BY-4.0) and a seed-0 sample of CodeSearchNet Python functions "
             "(The Stack is gated on the Hub and no token is available; CodeSearchNet was collected from repositories whose licences permit "
             "redistribution, per-function repo and URL are recorded). Docstrings are removed, CRLF normalised. Each item = (prefix, true next token, "
             "foil next token of the same category, subject span). Categories: **S1** closing bracket (`)` `]` `}` vs another closer; subject = the "
             "matching opener), **S2** block keyword (`else`/`elif`/`except`/`finally` vs a sibling keyword; subject = the `if`/`for`/`while`/`try` "
             "head), **S3** keyword completion (` in` after `for x` vs `,`; `:` after an `if`/`elif`/`while` header vs ` and`; subject = the `for`/`if` "
             "token), **R1** variable recall (a name bound earlier, foil = the most recently bound other name; subject = the definition site; the name "
             "occurred at most twice before), **R2** attribute/API recall (`.append` after `x = []`, `re.search` after `import re`; foil = another "
             "attribute of the same type/module; subject = the literal / module name), **R3** constant recall (a string or single-digit literal that "
             "occurred earlier, foil = another earlier literal; subject = the first occurrence). True and foil must each be ONE token as a continuation "
             "of the prefix in the model tokenizer (`moetrace/ext4_data.py`: the boundary backs off over up to 4 punctuation/whitespace characters when "
             "the tokenizer merges, e.g. Qwen's `.append`, ` else`); items whose true token occurs in the last 3 prefix tokens are excluded (copying). "
             "Prefixes are capped at 160 tokens. Noise, thresholds, split (128/128, seed 0), recurrence (64/128), active-random controls (3 Qwen3, "
             "1 Mixtral), bootstrap CIs: exactly as in the paper protocol; the only change to the pipeline is the case loader.\n")
    bs = os.path.join(RESULTS, "..", "data", "codefact", "build_stats.md")
    if os.path.exists(bs):
        txt = open(bs).read()
        try:
            yl = txt.split("## Yields per category\n")[1].split("\nRejects:")[0]
            P.append("**Yields per category (candidates -> single-token items per tokenizer):**\n" + yl.strip() + "\n")
        except IndexError:
            pass
    # calibration
    P.append("### Threshold calibration (Qwen3-30B-A3B-Base, all scanned items)\n")
    P.append("One GPU scan per model/protocol runs, for every item, the clean and subject-noised prefill rows AND the MoE-block patch at every layer "
             "(`scripts/ext4_scan.py`, chunks of up to 1,024 items sorted by length), so the filter and the layer sweep are the same pass. "
             "The paper's absolute filter (Δ_clean ≥ 1.0, drop ≥ 0.5) is primary; per-category pass rates are results in their own right. "
             "The relative rule (drop ≥ 25 % of Δ_clean, Δ_clean ≥ 1) is reported in the appendix table only.\n")
    for run in ("codefact_qwen3_raw", "codefact_mixtral_nobos", "codefact_qwen3_coder_raw", "codefact_qwen3_coder_chat"):
        cal = os.path.join(TAB, f"ext4_calibration_{run}.md")
        if os.path.exists(cal):
            P.append(open(cal).read())
    if os.path.exists(os.path.join(FIG, "ext4_calibration.png")):
        P.append("![CodeFact calibration](figures/ext4_calibration.png)\n")
    P.append("**Reading the calibration.** Syntax items have large clean margins (the model is nearly always right, top-1 = true in the great "
             "majority) but the subject noise often does not move them: the answer is redundantly determined by the rest of the context "
             "(a `)` after `foo(bar` is predicted from `foo` being a call even when the `(` embedding is destroyed; `else` is predicted from the "
             "dedent and the block content). A low pass rate under the paper's filter is therefore the expected signature of a *non-recall* category, "
             "not a construction error (the 20 eyeballed items per category in `data/codefact/samples.md` have the right subject). Recall categories "
             "carry their information in one place (the definition site) and are noise-sensitive like CounterFact.\n")
    # per-run results
    for run, out in summary["runs"].items():
        if out is None:
            P.append(f"### {RUNS[run]['label']}\n\n_Pending: run not available yet ({run})._\n")
            continue
        short = RUNS[run]["short"]
        P.append(f"### {out['label']}\n")
        P.append(f"Scanned {out['n_scanned']} items; passing the paper filter per category: " + ", ".join(f"{c} {n}" for c, n in out["scan_pass"].items()) + ". "
                 "Sets with fewer than 256 passing items use all passing items (marked partial; recurrence threshold = half the discovery split).\n")
        t = os.path.join(TAB, f"ext4_summary_{short}.md")
        if os.path.exists(t):
            P.append(open(t).read())
        if os.path.exists(os.path.join(FIG, f"ext4_curves_{short}.png")):
            P.append(f"![layer curves {short}](figures/ext4_curves_{short}.png)\n")
        for name in ("interior", "factual_experts", "layer_bands", "jaccard_recurrent", "jaccard_recurrent_interior", "jaccard_top10", "jaccard_top10_interior", "shared_recurrent", "shared_recurrent_interior"):
            t = os.path.join(TAB, f"ext4_{name}_{short}.md")
            if os.path.exists(t):
                P.append(open(t).read())
        if os.path.exists(os.path.join(FIG, f"ext4_overlap_{short}.png")):
            P.append(f"![expert overlap {short}](figures/ext4_overlap_{short}.png)\n")
        if out.get("verdict"):
            P.append(out["verdict"] + "\n")
    if summary.get("cross_runs"):
        P.append("### Summary across models and protocols\n")
        P.append(summary["cross_runs"] + "\n")
    nar = os.path.join(SEC, "ext4_codefact_narrative.md")
    if os.path.exists(nar):
        P.append(open(nar).read().strip() + "\n")
    if summary.get("coder_vs_base"):
        P.append("### Coder-Instruct vs Base on the same items\n")
        P.append(summary["coder_vs_base"] + "\n")
    if summary.get("overall"):
        P.append("### Verdict\n")
        P.append(summary["overall"] + "\n")
    P.append("### Appendix: relative threshold rule and quantiles\n")
    for run in ("codefact_qwen3_raw", "codefact_mixtral_nobos", "codefact_qwen3_coder_raw", "codefact_qwen3_coder_chat"):
        q = os.path.join(TAB, f"ext4_calibration_quantiles_{run}.md")
        if os.path.exists(q):
            P.append(open(q).read())
    for run, out in summary["runs"].items():
        if out is None:
            continue
        q = os.path.join(TAB, f"ext4_calibration_subcategories_{RUNS[run]['short']}.md")
        if os.path.exists(q):
            P.append(open(q).read())
    P.append("The relative rule (drop ≥ 25 % of Δ_clean) admits the syntax items whose absolute drop is small relative to a very large margin only "
             "when the drop is also a quarter of that margin, so it is stricter than the paper's rule for high-margin items and looser for low-margin "
             "ones; the pass rates above show where the two rules disagree. It is not used for any selection in this section.\n")
    P.append("**Deviations and open questions.** (1) The Stack replaced by CodeSearchNet (gated dataset, no token). (2) Tokenizer merges force a boundary "
             "back-off for many items (all S2 and R2 items under Qwen: ` else`, `.append` are single tokens), so the 'true token' sometimes contains the "
             "preceding punctuation; the contrast between true and foil is unchanged. (3) In Qwen's tokenizer an opener often merges with the following "
             "identifier (`(bar`), so the S1 subject token carries the first argument too. (4) R3 integer items (single digits) are weak by construction; "
             "string items are the informative part. (5) Per-category sets share one expert pass (union of cases), so a category's cases can appear in the "
             "mixed `all` set. (6) Coder-Instruct runs depend on the ext2 download (marked pending if absent).\n")
    P.append(f"_Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by scripts/ext4_analyze.py._\n")
    open(path, "w").write("\n".join(P))


def verdict_for_run(out: dict, cc: dict | None) -> str:
    L = []
    S = [s for s in CATEGORIES if s in out["sets"] and out["sets"][s].get("e_star") is not None]
    for s in CATEGORIES:
        if s not in out["sets"]:
            L.append(f"- **{s}** ({CATEGORY_LABEL[s]}): fewer than 64 items pass the paper filter ({out['scan_pass'].get(s)}), not run.")
            continue
        d = out["sets"][s]
        line = f"- **{s}** ({CATEGORY_LABEL[s]}, n={d['n_disc']}+{d['n_val']}{', partial' if d['partial'] else ''}): L*={d['L_star']}, block rescue {ci(d['layer_val'])}"
        if "L_interior" in d:
            line += f"; interior L*={d['L_interior']} {ci(d['layer_val_interior'])}"
            if d.get("e_interior") is not None:
                line += f" -> {d['pair_interior']} rescue {ci(d['rescue_interior'])} Spec {ci(d['spec_interior'])}"
            elif "coalitions_interior" in d:
                line += " -> no recurrent expert"
            wi = d.get("joint_interior_winner")
            if wi:
                line += f", interior joint winner {wi['pair']} rescue {f3(wi['val_rescue'])} Spec {f3(wi['val_spec'])}"
        if d.get("e_star") is not None:
            line += (f"; two-stage expert {d['pair']} (active {d['disc_active']}/{d['n_disc']} disc), rescue {ci(d['rescue'])}, Spec {ci(d['spec'])}, "
                     f"rank-1 among active in {d['rank']['top1']}/{d['rank']['n']}; coalition (clean top-k) {f3(d['coalitions']['coalition_clean']['mean'])}")
            w = d["joint"]["winner"]
            if w:
                line += f"; joint winner {w['pair']}" + (" (same)" if d["joint"]["same_as_two_stage"] else f" rescue {f3(w['val_rescue'])} Spec {f3(w['val_spec'])}")
        elif "coalitions" in d:
            line += f"; no expert reaches the recurrence threshold at L* (max activity {d.get('max_activity')}/{d['n_disc']}); coalition {f3(d['coalitions']['coalition_clean']['mean'])}"
            w = d["joint"]["winner"]
            if w:
                line += f"; joint winner over all layers {w['pair']} rescue {f3(w['val_rescue'])} Spec {f3(w['val_spec'])}"
        L.append(line + ".")
    if cc is not None:
        cats = [c for c in cc["categories"] if c != "all"]
        J = cc["jaccard_recurrent"]
        offd = [J.loc[a, b] for i, a in enumerate(cats) for b in cats[i + 1:]]
        Jt = cc["jaccard_top10"]
        offt = [Jt.loc[a, b] for i, a in enumerate(cats) for b in cats[i + 1:]]
        sy = [c for c in cats if c.startswith("S")]
        rc = [c for c in cats if c.startswith("R")]
        within_s = [J.loc[a, b] for i, a in enumerate(sy) for b in sy[i + 1:]]
        within_r = [J.loc[a, b] for i, a in enumerate(rc) for b in rc[i + 1:]]
        across = [J.loc[a, b] for a in sy for b in rc]
        L.append(f"- **Cross-category overlap**: mean pairwise Jaccard of the recurrent (layer, expert) sets {np.nanmean(offd):.2f} "
                 f"(within syntax {np.nanmean(within_s) if within_s else float('nan'):.2f}, within recall {np.nanmean(within_r) if within_r else float('nan'):.2f}, "
                 f"syntax-recall {np.nanmean(across) if across else float('nan'):.2f}); of the top-10 joint pairs {np.nanmean(offt):.2f}. "
                 f"Selected layers: " + ", ".join(f"{s} L{cc['L_star'][s]}" for s in cats) + f"; pairs recurrent in every category: {len(cc['common_to_all_categories'])}.")
    if out["sets"] and any("factual" in d for d in out["sets"].values()):
        fl = []
        for s, d in out["sets"].items():
            for r in d.get("factual", []):
                if r["recurrent"] or (r["val_active"] and r["val_rescue"] > 0.1):
                    fl.append(f"{r['pair']} on {s} (active {r['disc_active']}/{d['n_disc']}, rescue {f3(r['val_rescue'])}, Spec {f3(r['val_spec'])})")
        L.append("- **Factual-recall experts on code**: " + ("; ".join(fl) if fl else "none of them is recurrent or rescues > 0.1 on any code category") + ".")
    return "\n".join(L)


def cross_run_table(summary: dict) -> str | None:
    """Categories x runs: L*, two-stage expert, its validation rescue / Spec, interior selection."""
    runs = [r for r, o in summary["runs"].items() if o is not None]
    if not runs:
        return None
    rows = []
    for s in SETS:
        r = [f"{s} {CATEGORY_LABEL.get(s, 'mixed')}"]
        for run in runs:
            o = summary["runs"][run]
            if s not in o["sets"]:
                r.append("not run (< 64 pass)")
                continue
            d = o["sets"][s]
            cell = f"L{d['L_star']} {f3(d['layer_val']['mean'])}"
            if d.get("e_star") is not None:
                cell += f" -> {d['pair']} {f3(d['rescue']['mean'])} / Spec {f3(d['spec']['mean'])}"
            elif "coalitions" in d:
                cell += " -> no recurrent expert"
            if d.get("L_interior") is not None:
                cell += f"; interior L{d['L_interior']} {f3(d['layer_val_interior']['mean'])}"
                if d.get("e_interior") is not None:
                    cell += f" -> {d['pair_interior']} {f3(d['rescue_interior']['mean'])} / Spec {f3(d['spec_interior']['mean'])}"
            if d["partial"]:
                cell += f" (partial n={d['n_disc']}+{d['n_val']})"
            r.append(cell)
        rows.append(r)
    hdr = ["Set"] + [RUNS[r]["label"] for r in runs]
    return md_table(hdr, rows, os.path.join(TAB, "ext4_cross_runs"),
                    "All runs: layer L* and MoE-block validation rescue -> two-stage expert with validation rescue / active-random Spec; then the same restricted to interior layers (≤ L−5)")


def coder_vs_base(summary: dict) -> str | None:
    base = summary["runs"].get("codefact_qwen3_raw")
    outs = {k: summary["runs"].get(k) for k in ("codefact_qwen3_coder_raw", "codefact_qwen3_coder_chat")}
    if base is None or not any(outs.values()):
        return None
    rows = []
    for s in SETS:
        if s not in base["sets"]:
            continue
        b = base["sets"][s]
        r = [s, f"L{b['L_star']} {ci(b['layer_val'])}", b.get("pair") or "none", ci(b.get("rescue")) if b.get("rescue") else "n/a", ci(b.get("spec")) if b.get("spec") else "n/a"]
        for k, o in outs.items():
            if o is None or s not in o["sets"]:
                r += ["pending" if o is None else "not run"] * 4
                continue
            d = o["sets"][s]
            r += [f"L{d['L_star']} {ci(d['layer_val'])}", d.get("pair") or "none", ci(d.get("rescue")) if d.get("rescue") else "n/a", ci(d.get("spec")) if d.get("spec") else "n/a"]
        rows.append(r)
    hdr = ["Set", "Base L* / block", "Base e*", "Base rescue", "Base Spec", "Coder raw L* / block", "Coder raw e*", "Coder raw rescue", "Coder raw Spec",
           "Coder chat L* / block", "Coder chat e*", "Coder chat rescue", "Coder chat Spec"]
    t = md_table(hdr, rows, os.path.join(TAB, "ext4_coder_vs_base"), "Qwen3-Coder-30B-A3B-Instruct vs Qwen3-30B-A3B-Base per category (each model on its own passing items)")
    # shared items: items passing in both
    txt = [t]
    for k, o in outs.items():
        if o is None:
            continue
        sb = pd.read_parquet(os.path.join(RESULTS, "codefact_qwen3_raw", "scan_cases.parquet")).set_index("case_id")
        so = pd.read_parquet(os.path.join(RESULTS, k, "scan_cases.parquet")).set_index("case_id")
        common = sb.index.intersection(so.index)
        both = (sb.loc[common].strict & so.loc[common].strict)
        rows = []
        for c in CATEGORIES:
            m = sb.loc[common].category == c
            rows.append([c, int(m.sum()), int(sb.loc[common][m].strict.sum()), int(so.loc[common][m].strict.sum()), int(both[m].sum()),
                         f"{np.corrcoef(sb.loc[common][m].delta_clean, so.loc[common][m].delta_clean)[0, 1]:.2f}" if m.sum() > 2 else "n/a",
                         f"{np.corrcoef(sb.loc[common][m]['drop'], so.loc[common][m]['drop'])[0, 1]:.2f}" if m.sum() > 2 else "n/a"])
        txt.append(md_table(["Category", "items scanned by both", "pass Base", f"pass {RUNS[k]['short']}", "pass both", "corr Δ_clean", "corr drop"], rows,
                            os.path.join(TAB, f"ext4_shared_items_{RUNS[k]['short']}"), f"Items scanned by both Base and {RUNS[k]['label']}: filter agreement"))
        # same pairs?
        same = []
        for s in SETS:
            if s in base["sets"] and s in o["sets"]:
                bp, op = base["sets"][s].get("pair"), o["sets"][s].get("pair")
                same.append(f"{s}: Base {bp or 'none'} vs Coder {op or 'none'}" + (" (same expert)" if bp and bp == op else ""))
        txt.append("Selected experts per category, " + RUNS[k]["label"] + ": " + "; ".join(same) + ".\n")
    return "\n".join(txt)


def cached_analyze(run: str, reuse: bool) -> dict | None:
    """analyze_run with a pickle cache (results/<run>/ext4_analysis.pkl), reused when newer than expert_rows.parquet."""
    import pickle
    od = os.path.join(RESULTS, run)
    pk = os.path.join(od, "ext4_analysis.pkl")
    er = os.path.join(od, "expert_rows.parquet")
    if reuse and os.path.exists(pk) and os.path.exists(er) and os.path.getmtime(pk) > os.path.getmtime(er):
        log(f"{run}: reusing cached analysis {pk}")
        with open(pk, "rb") as f:
            return pickle.load(f)
    out = analyze_run(run)
    if out is not None and out["has_expert"]:
        with open(pk, "wb") as f:
            pickle.dump(out, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUNS.keys()))
    ap.add_argument("--reuse", action="store_true", help="reuse cached per-run analyses when newer than expert_rows.parquet")
    args = ap.parse_args()
    summary = {"runs": {}, "cross": {}}
    for run in args.runs.split(","):
        out = cached_analyze(run, args.reuse)
        if out is None:
            summary["runs"][run] = None
            continue
        cc = cross_category(out) if out["has_expert"] else None
        write_tables(out, cc)
        short = RUNS[run]["short"]
        figure_curves(out, os.path.join(FIG, f"ext4_curves_{short}.png"))
        if cc is not None:
            figure_overlap(cc, out["label"], os.path.join(FIG, f"ext4_overlap_{short}.png"))
            summary["cross"][run] = jsonable({k: v for k, v in cc.items() if not isinstance(v, pd.DataFrame)})
            summary["cross"][run]["jaccard_recurrent"] = cc["jaccard_recurrent"].round(3).to_dict()
            summary["cross"][run]["jaccard_top10"] = cc["jaccard_top10"].round(3).to_dict()
        out["verdict"] = verdict_for_run(out, cc)
        summary["runs"][run] = out
    summary["coder_vs_base"] = coder_vs_base(summary)
    summary["cross_runs"] = cross_run_table(summary)
    # overall verdict
    q = summary["runs"].get("codefact_qwen3_raw")
    m = summary["runs"].get("codefact_mixtral_nobos")
    lines = []
    for o in (q, m):
        if o is None:
            continue
        S = {s: o["sets"][s] for s in o["sets"] if s != "all"}
        sy = [s for s in S if s.startswith("S")]
        rc = [s for s in S if s.startswith("R")]
        def mean_of(keys, f):
            v = [f(S[s]) for s in keys if f(S[s]) is not None]
            return float(np.mean(v)) if v else float("nan")
        blk_s = mean_of(sy, lambda d: d["layer_val"]["mean"])
        blk_r = mean_of(rc, lambda d: d["layer_val"]["mean"])
        ex_s = mean_of(sy, lambda d: d["rescue"]["mean"] if d.get("rescue") else None)
        ex_r = mean_of(rc, lambda d: d["rescue"]["mean"] if d.get("rescue") else None)
        sp_s = mean_of(sy, lambda d: d["spec"]["mean"] if d.get("spec") else None)
        sp_r = mean_of(rc, lambda d: d["spec"]["mean"] if d.get("spec") else None)
        pr_s = np.mean([o["scan_pass"][c] for c in CATEGORIES if c.startswith("S")])
        pr_r = np.mean([o["scan_pass"][c] for c in CATEGORIES if c.startswith("R")])
        lines.append(f"- {o['label']}: pass counts syntax vs recall {pr_s:.0f} vs {pr_r:.0f} of the scanned items per category; mean block rescue at L* "
                     f"{blk_s:+.3f} (S) vs {blk_r:+.3f} (R); mean two-stage expert rescue {ex_s:+.3f} (S) vs {ex_r:+.3f} (R); mean Spec {sp_s:+.3f} (S) vs {sp_r:+.3f} (R); "
                     f"layers " + ", ".join(f"{s} L{S[s]['L_star']}" for s in S) + ".")
    summary["overall"] = "\n".join(lines) if lines else None
    with open(os.path.join(RESULTS, "ext4_summary.json"), "w") as f:
        json.dump(jsonable({k: v for k, v in summary.items()}), f, indent=1, default=str)
    write_section(summary, os.path.join(SEC, "ext4_codefact.md"))
    log("section written:", os.path.join(SEC, "ext4_codefact.md"))


if __name__ == "__main__":
    main()
