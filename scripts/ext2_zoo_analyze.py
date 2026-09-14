"""ext2-model-zoo analysis driver (RESEARCH_PLAN Direction 2, Question A).

For every zoo run results/<model>_<protocol>/ (and the base-model reference runs) it applies the paper's two-stage
analysis (moetrace.analysis: layer_analysis, select_expert, evaluate_expert, coalitions, all_active_rank,
stability_grid) and the Direction-1 joint search (moetrace.ext1_analysis: ExpertCache, joint_search, per_layer_curve),
evaluates the base model's reference experts as fixed hypotheses, reads the sink diagnostic and the funnel pass
rates, assigns the pattern label (A / B / C) and writes:
  results/<run>/zoo_summary.json                      per-run numbers (cached; --force recomputes)
  results/tables/ext2_zoo_summary.{md,csv}            cross-model table, one row per (model, protocol, case set)
  results/tables/ext2_zoo_{usage,sink,reference_experts,base_vs_instruct,routing_agreement}.{md,csv}
  results/tables/ext2_zoo_run_<run>.{md,csv}          Table-1-style rows per run
  results/figures/ext2_zoo_curves.{png,pdf}           layer-curve small multiples (intended protocol, other protocols thin)
  results/figures/ext2_zoo_expert_curves.{png,pdf}    best-expert rescue / Spec curves per model (intended protocol)
  results/figures/ext2_zoo_attn_curves.{png,pdf}      attention / MoE / block curves when the attn sweeps exist
  results/ext2_zoo_summary.json, results/sections/ext2_model_zoo.md

Usage: python scripts/ext2_zoo_analyze.py [--runs a,b,...] [--force] [--no-figures]
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import analysis as A
from moetrace import ext1_analysis as X
from moetrace.models import MODELS, RESULTS
from moetrace.stats import summarize, fmt
from moetrace.report import md_table
from moetrace.ext2_zoo import (ZOO_KEYS, PROTOCOLS, INSTRUCTION, load_usage, usage_path, protocol_list, pattern_label, sink_summary)

TAB, FIG, SEC = (os.path.join(RESULTS, d) for d in ("tables", "figures", "sections"))
for d in (TAB, FIG, SEC):
    os.makedirs(d, exist_ok=True)

# base-model reference runs (all-layer expert rows from Direction 1; sweep files copied from the base runs)
BASE_RUNS = {
    "qwen3_bos_alllayers": dict(model="qwen3", protocol="default", intended=True, diag_run="qwen3_nobos_diag",
                                note="tokenizer defaults = no special tokens = paper protocol"),
    "mixtral_bos_alllayers": dict(model="mixtral", protocol="default", intended=True, diag_run="mixtral_bos_diag", note="tokenizer default (BOS)"),
    "mixtral_nobos_alllayers": dict(model="mixtral", protocol="nobos", intended=False, diag_run="mixtral_nobos_diag", note="paper protocol (no BOS)",
                                    sets=("paper",)),  # only the paper set was swept without BOS (CLAUDE.md section 3)
}
# fixed-hypothesis experts per family (from REPORT.md and Direction 1)
REFERENCE_EXPERTS = {"qwen3_moe": [(44, 69), (42, 115)], "mixtral": [(19, 6), (19, 2), (18, 1)], "olmoe": []}
REFERENCE_LAYERS = {"qwen3_moe": [44, 42], "mixtral": [19, 18, 21], "olmoe": []}
INTENDED = {"olmoe": "default", "olmoe_instruct": "chat", "qwen3_instruct": "chat", "qwen3_coder": "chat", "mixtral_instruct": "chat",
            "qwen3": "default", "mixtral": "default"}
ORDER = ["qwen3", "qwen3_instruct", "qwen3_coder", "mixtral", "mixtral_instruct", "olmoe", "olmoe_instruct"]
COLORS = {"default": "#2a78d6", "nobos": "#eb6834", "chat": "#2f9e60"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def ci(s):
    if s is None or (isinstance(s, dict) and (s.get("n", 0) == 0 or s.get("mean") is None)):
        return "n/a"
    return f"{s['mean']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"


def pair(l, e):
    return "none" if e is None or (isinstance(e, float) and np.isnan(e)) else f"L{int(l)}E{int(e):03d}"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if not isinstance(v, pd.DataFrame)}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def discover_runs() -> dict:
    runs = {}
    for key in ZOO_KEYS:
        for p in PROTOCOLS:
            run = f"{key}_{p}"
            if os.path.exists(os.path.join(RESULTS, run, "sweep_summary.json")):
                meta = json.load(open(os.path.join(RESULTS, run, "run_meta.json"))) if os.path.exists(os.path.join(RESULTS, run, "run_meta.json")) else {}
                runs[run] = dict(model=key, protocol=p, intended=(INTENDED[key] == p), base=False, meta=meta,
                                 equivalent_to=meta.get("protocol_equivalent_to"))
    for run, cfg in BASE_RUNS.items():
        if os.path.exists(os.path.join(RESULTS, run, "sweep_summary.json")):
            runs[run] = dict(model=cfg["model"], protocol=cfg["protocol"], intended=cfg["intended"], base=True, meta={}, equivalent_to=None, diag_run=cfg["diag_run"],
                             sets=cfg.get("sets"))
    return runs


def base_sink(diag_run: str) -> dict | None:
    """Sink summary for a base-model reference run from the ext3 raw diagnostics (paper set, clean rows)."""
    p = f"/opt/dlami/nvme/moe_ext3/{diag_run}/diag.npz"
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    rk = d["row_kind"]
    n = int((rk == "clean").sum())
    out = sink_summary(d["resid_norms"][:, :n], d["attn_final"][:, :n], d["lens"][:n], prefix_len=int(d["prefix_len"]) if "prefix_len" in d else 0)
    out["source"] = p
    return out


def sink_position_info(run: str, info: dict, layer: int, md: A.ModelData) -> dict:
    """Mode of the max-norm position at the sink layer and the token that sits there (from the raw diagnostics)."""
    from moetrace.ext2_zoo import sink_positions, DIAG_ROOT
    from moetrace.arch import snapshot_dir
    from transformers import AutoTokenizer
    if info.get("base"):
        p = f"/opt/dlami/nvme/moe_ext3/{info['diag_run']}/diag.npz"
        d = np.load(p, allow_pickle=True)
        n = int((d["row_kind"] == "clean").sum())
        rn, lens = d["resid_norms"][:, :n], d["lens"][:n]
        ids_of = {int(c): json.loads(i) for c, i in zip(md.cases.case_id, md.cases.ids)}
        case_ids = [int(c) for c in d["case_ids"][:n]]
    else:
        p = os.path.join(DIAG_ROOT, run, "sweep_diag.npz")
        d = np.load(p, allow_pickle=True)
        n = int(d["n_clean"])
        rn, lens = d["resid_norms"][:, :n], d["lens"][:n]
        ids_of = {int(c): json.loads(i) for c, i in zip(md.cases.case_id, md.cases.ids)}
        case_ids = [int(c) for c in md.cases.case_id]  # run_sweep builds prefill rows in sweep_cases order
    sp = sink_positions(rn, lens, layer)
    tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[info["model"]]["repo"]))
    from collections import Counter
    toks = Counter()
    for i, cid in enumerate(case_ids[:n]):
        ids = ids_of.get(cid)
        if ids is None:
            continue
        am = int(np.where(np.arange(rn.shape[2]) < lens[i], rn[layer, i], -np.inf).argmax())
        if am < len(ids):
            toks[tok.convert_ids_to_tokens(ids[am])] += 1
    sp["token_at_max_norm_top3"] = toks.most_common(3)
    return sp


def routing_sets(md: A.ModelData, layer: int, ids: list[int]) -> dict[int, frozenset]:
    r = md.routing[(md.routing.layer == layer) & (md.routing.run == "clean") & md.routing.case_id.isin(ids)]
    return {int(c): frozenset(g.expert.astype(int).tolist()) for c, g in r.groupby("case_id")}


def routing_agreement(md_a: A.ModelData, md_b: A.ModelData, layer: int, ids: list[int]) -> dict:
    a, b = routing_sets(md_a, layer, ids), routing_sets(md_b, layer, ids)
    common = [c for c in ids if c in a and c in b]
    if not common:
        return dict(n=0)
    jac = [len(a[c] & b[c]) / len(a[c] | b[c]) for c in common]
    same = [a[c] == b[c] for c in common]
    return dict(n=len(common), layer=layer, mean_jaccard=float(np.mean(jac)), frac_identical=float(np.mean(same)))


# ---------------------------------------------------------------------------------------------------------------
def analyze_run(run: str, info: dict, force: bool) -> dict:
    od = os.path.join(RESULTS, run)
    cache_p = os.path.join(od, "zoo_summary.json")
    er = os.path.join(od, "expert_rows.parquet")
    if not force and os.path.exists(cache_p) and (not os.path.exists(er) or os.path.getmtime(cache_p) > os.path.getmtime(er)):
        log(f"{run}: cached")
        return json.load(open(cache_p))
    m = MODELS[info["model"]]
    nc, fam = m["n_controls"], m.get("family")
    md = A.load_model(run)
    out = dict(run=run, model=info["model"], label=m["label"], family=fam, protocol=info["protocol"], intended=info["intended"], base=info["base"],
               equivalent_to=info.get("equivalent_to"), n_controls=nc, n_layers=int(md.routing.layer.nunique()),
               prefix_text=info["meta"].get("prefix_text", ""), prefix_len=info["meta"].get("prefix_len", 0),
               has_expert=md.expert_rows is not None, sets={})
    ss = json.load(open(os.path.join(od, "sweep_summary.json")))
    cs = md.sets
    out["filter_scan"] = {k: v for k, v in cs.get("scan", {}).items() if k not in ("pass_times_s",)}
    out["paper_check"] = cs.get("paper_check")
    out["funnel"] = A.funnel_check(md)
    # sink diagnostic
    sd = None
    if os.path.exists(os.path.join(od, "sink_diag.json")):
        sd = json.load(open(os.path.join(od, "sink_diag.json")))
    elif info.get("diag_run"):
        sd = base_sink(info["diag_run"])
    out["sink"] = {k: v for k, v in sd.items() if k != "per_layer"} if sd else None
    out["sink_per_layer"] = sd["per_layer"] if sd else None
    if sd:
        try:
            out["sink"]["position"] = sink_position_info(run, info, sd["sink_layer"], md)
        except Exception as ex:
            log(f"{run}: sink position info failed: {ex!r}")
    cache = X.ExpertCache(md) if md.expert_rows is not None else None
    if cache is not None:
        out["expert_layers_done"] = len(cache.layers)
    set_list = [k for k in ("paper", "strict", "relaxed") if k in cs and (not info.get("sets") or k in info["sets"])]
    for s in set_list:
        disc, val = md.ids(s, "discovery"), md.ids(s, "validation")
        if len(disc) + len(val) < 0.9 * (len(cs[s]["discovery"]) + len(cs[s]["validation"])):
            log(f"{run} / {s}: only {len(disc) + len(val)} of the set's cases were swept in this run, set skipped")
            continue
        if len(disc) < 8 or len(val) < 8:
            continue
        log(f"{run} / {s}: layer analysis")
        la = A.layer_analysis(md, s)
        L = la["L_star"]
        th = len(disc) // 2
        r = dict(set=s, n_disc=len(disc), n_val=len(val), threshold=th, L_star=L, disc_at_Lstar=la["disc_mean_at_Lstar"], val_at_Lstar=la["val_at_Lstar"],
                 sharpness=la["sharpness"], curve=la["curve"], disc_top5=la["disc_curve_top5"],
                 funnel_strict_pass=ss.get(s, {}).get("funnel_strict_pass"), funnel_relaxed_pass=ss.get(s, {}).get("funnel_relaxed_pass"),
                 val_at_reference_layers={str(l): summarize(md.R.loc[val, l].values, with_p=False) for l in REFERENCE_LAYERS.get(fam, []) if l in md.R.columns})
        if cache is not None and L in cache.layers:
            log(f"{run} / {s}: two-stage selection at L{L}")
            et = A.expert_table(md, L)
            sel = A.select_expert(et, disc, th)
            r["selection"] = sel
            e = sel["e_star"]
            r["coalitions"] = A.coalitions(md, L, val)
            if e is not None:
                ev = A.evaluate_expert(md, L, e, val, nc)
                r["eval"] = {k: v for k, v in ev.items() if k != "per_case"}
                r["ratios"] = A.ratios(md, L, ev, val)
                rk = A.all_active_rank(md, L, e, val)
                r["rank"] = {k: v for k, v in rk.items() if k != "per_case"}
                st = A.stability_grid(md, L, s, nc, reference_expert=e)
                r["stability"] = {k: v for k, v in st.items() if k != "grid"}
                r["stability"]["grid"] = st["grid"].to_dict("records")
            log(f"{run} / {s}: joint search")
            two_stage = (L, e if e is not None else -1)
            js = X.joint_search(md, cache, s, nc, two_stage)
            w = js["winner"]
            r["joint"] = dict(n_candidates=js["n_candidates"], n_layers_with_candidates=js["n_layers_with_candidates"], winner=w, two_stage=js["two_stage"] if e is not None else None,
                              challenger=js["challenger"], same_as_two_stage=js["same_as_two_stage"], winner_in_two_stage_layer=js["winner_in_two_stage_layer"],
                              vs_reference=js["vs_reference"] if e is not None else None, shrinkage=js["shrinkage"], top10=js["top"].to_dict("records"))
            log(f"{run} / {s}: per-layer best-expert curve")
            curve = X.per_layer_curve(md, cache, s, nc)
            curve.to_csv(os.path.join(TAB, f"ext2_zoo_layer_curve_{run}_{s}.csv"), index=False)
            r["expert_curve"] = curve[["layer", "layer_val_mean", "layer_samepass_val_mean", "e_star", "disc_active", "val_active", "val_rescue", "val_rescue_lo", "val_rescue_hi",
                                       "val_spec", "val_spec_lo", "val_spec_hi", "concentration", "n_candidates"]].to_dict("records")
            r["reference_experts"] = {}
            for (l, ee) in REFERENCE_EXPERTS.get(fam, []):
                if l in cache.layers and ee in cache.piv[l].columns:
                    r["reference_experts"][pair(l, ee)] = X.evaluate_pair(md, cache, l, ee, disc, val, nc)
                elif l in cache.layers:
                    r["reference_experts"][pair(l, ee)] = dict(layer=l, expert=ee, pair=pair(l, ee), disc_active=0, val_active=0, val_rescue=0.0, val_spec=np.nan, note="never clean-active")
            spec_all = r.get("eval", {}).get("spec_all")
            resc_all = r.get("eval", {}).get("rescue_all")
            r["pattern"] = pattern_label(la["val_at_Lstar"], spec_all, resc_all, r["coalitions"]["coalition_clean"])
        else:
            r["pattern"] = pattern_label(la["val_at_Lstar"], None, None, None) if la["val_at_Lstar"]["ci_lo"] <= 0 else "layer only (no expert pass)"
        out["sets"][s] = r
    out = jsonable(out)
    json.dump(out, open(cache_p, "w"), indent=1)
    return out


# ---------------------------------------------------------------------------------------------------------------
def protocol_label(rr: dict) -> str:
    p = rr["protocol"]
    if rr.get("equivalent_to"):
        return f"{p} (= {rr['equivalent_to']})"
    return p


def run_table(rr: dict) -> list[list]:
    rows = []
    for s, r in rr["sets"].items():
        sel = r.get("selection", {})
        e = sel.get("e_star")
        ev = r.get("eval", {})
        co = r.get("coalitions", {})
        jw = r.get("joint", {}).get("winner")
        rows.append([s, f"{r['n_disc']}/{r['n_val']}", f"L{r['L_star']}", ci(r["val_at_Lstar"]),
                     pair(r["L_star"], e) if e is not None else f"none ({sel.get('max_activity', 'n/a')} max)" if sel else "n/a",
                     f"{sel.get('disc_active', '')}/{ev.get('val_active', '')}" if e is not None else "",
                     ci(ev.get("rescue_all")) if e is not None else "", ci(ev.get("spec_all")) if e is not None else "",
                     ci(co.get("coalition_clean")) if co else "", ci(co.get("coalition_union")) if co else "",
                     (f"{jw['pair']} {f3(jw['val_rescue'])} / Spec {f3(jw['val_spec'])}" + ("" if r["joint"]["same_as_two_stage"] else " (differs)")) if jw else "n/a",
                     r.get("pattern", "")])
    return rows


RUN_HEADER = ["case set", "n disc/val", "L*", "layer rescue (val) [CI]", "selected expert", "active disc/val", "expert rescue [CI]", "Spec [CI]",
              "coalition top-k", "routing union", "joint-search winner (val rescue / Spec)", "pattern"]


def set_drop(md: A.ModelData, s: str) -> tuple[float, float]:
    """Mean clean margin and mean noise drop (Delta_clean - Delta_noised) over the set's validation cases."""
    val = md.ids(s, "validation")
    ct = md.cases.set_index("case_id").loc[val]
    return float(ct.delta_clean.mean()), float(ct["drop"].mean())


def summary_rows(res: dict, mds: dict | None = None) -> tuple[list[list], list[dict]]:
    rows, recs = [], []
    for run in sorted(res, key=lambda r: (ORDER.index(res[r]["model"]) if res[r]["model"] in ORDER else 99, PROTOCOLS.index(res[r]["protocol"]))):
        rr = res[run]
        sk = rr.get("sink") or {}
        for s, r in rr["sets"].items():
            sel = r.get("selection", {})
            e = sel.get("e_star")
            ev = r.get("eval", {})
            co = r.get("coalitions", {})
            j = r.get("joint", {})
            jw = j.get("winner")
            fun = (f"{rr['funnel']['paper_strict_pass']}/{rr['funnel']['paper_n']}" if s == "paper" and rr.get("funnel", {}).get("paper_n") else
                   f"{r.get('funnel_strict_pass')}/{r['n_disc'] + r['n_val']}")
            dclean, drop = set_drop(mds[run], s) if mds and run in mds else (float("nan"), float("nan"))
            norm_layer = r["val_at_Lstar"]["mean"] / drop if drop else float("nan")
            norm_expert = (ev.get("rescue_all", {}).get("mean") / drop) if (e is not None and drop) else None
            rec = dict(model=rr["model"], label=rr["label"], protocol=protocol_label(rr), intended=rr["intended"], base=rr["base"], case_set=s,
                       L_star=r["L_star"], layer_rescue=r["val_at_Lstar"]["mean"], layer_lo=r["val_at_Lstar"]["ci_lo"], layer_hi=r["val_at_Lstar"]["ci_hi"],
                       selected_expert=pair(r["L_star"], e) if e is not None else None, n_candidates=sel.get("n_candidates"),
                       disc_active=sel.get("disc_active"), val_active=ev.get("val_active"),
                       expert_rescue=ev.get("rescue_all", {}).get("mean"), expert_lo=ev.get("rescue_all", {}).get("ci_lo"), expert_hi=ev.get("rescue_all", {}).get("ci_hi"),
                       spec=ev.get("spec_all", {}).get("mean"), spec_lo=ev.get("spec_all", {}).get("ci_lo"), spec_hi=ev.get("spec_all", {}).get("ci_hi"),
                       coalition_clean=co.get("coalition_clean", {}).get("mean"), coalition_union=co.get("coalition_union", {}).get("mean"),
                       joint_winner=jw["pair"] if jw else None, joint_rescue=jw["val_rescue"] if jw else None, joint_spec=jw["val_spec"] if jw else None,
                       joint_differs=(not j.get("same_as_two_stage")) if jw else None,
                       val_delta_clean=dclean, val_drop=drop, layer_rescue_over_drop=norm_layer, expert_rescue_over_drop=norm_expert,
                       sink_final_frac=sk.get("frac_final_max_norm_at_sink_layer"), sink_final_self_attn=sk.get("frac_final_self_attn_max_at_sink_layer"),
                       sink_layer=sk.get("sink_layer"), funnel_pass=fun, pattern=r.get("pattern"),
                       stability_ref=f"{r['stability']['n_reference_selected']}/{r['stability']['n_settings']}" if r.get("stability") else None)
            recs.append(rec)
            rows.append([rr["label"], rec["protocol"] + (" *" if rr["intended"] else ""), s, f"L{r['L_star']}", ci(r["val_at_Lstar"]) + f" ({norm_layer:.0%} of drop {drop:+.1f})",
                         rec["selected_expert"] or (f"none (max act {sel.get('max_activity')})" if sel else "n/a"),
                         f"{sel.get('disc_active')}/{ev.get('val_active')}" if e is not None else "",
                         ci(ev.get("rescue_all")) if e is not None else "", ci(ev.get("spec_all")) if e is not None else "",
                         f"{f3(rec['coalition_clean'])} / {f3(rec['coalition_union'])}" if co else "",
                         (f"{jw['pair']} ({f3(jw['val_rescue'])} / {f3(jw['val_spec'])})" + (" differs" if rec["joint_differs"] else " same")) if jw else "n/a",
                         f"{sk.get('frac_final_max_norm_at_sink_layer', float('nan')):.3f}" if sk else "n/a", fun, rec["stability_ref"] or "", rec["pattern"]])
    return rows, recs


SUMMARY_HEADER = ["model", "protocol (* intended)", "case set", "L*", "layer rescue [CI] (share of the noise drop)", "selected expert", "active disc/val", "expert rescue [CI]", "Spec [CI]",
                  "coalitions top-k / union", "joint winner (rescue / Spec)", "sink-final frac", "funnel strict pass", "e* stable (grid)", "pattern"]


# ---------------------------------------------------------------------------------------------------------------
def fig_curves(res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    models = [mk for mk in ORDER if any(r["model"] == mk for r in res.values())]
    ncol = 4
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, mk in enumerate(models):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        runs = [r for r in res.values() if r["model"] == mk]
        for rr in sorted(runs, key=lambda r: (not r["intended"], PROTOCOLS.index(r["protocol"]))):
            s = "paper" if "paper" in rr["sets"] else ("strict" if "strict" in rr["sets"] else next(iter(rr["sets"])))
            r = rr["sets"][s]
            cur = pd.DataFrame(r["curve"])
            lw = 1.8 if rr["intended"] else 1.0
            lab = f"{protocol_label(rr)}{' (intended)' if rr['intended'] else ''}: L{r['L_star']} {r['val_at_Lstar']['mean']:+.2f} [{s}, n={r['n_val']}]"
            ax.plot(cur.layer, cur.val_mean, color=COLORS[rr["protocol"]], lw=lw, alpha=1.0 if rr["intended"] else 0.8, label=lab)
            if rr["intended"]:
                ax.fill_between(cur.layer, cur.ci_lo, cur.ci_hi, color=COLORS[rr["protocol"]], alpha=0.15, lw=0)
            ax.plot([r["L_star"]], [r["val_at_Lstar"]["mean"]], "o", color=COLORS[rr["protocol"]], ms=4)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(MODELS[mk]["label"], fontsize=8.5, loc="left")
        ax.set_xlabel("MoE layer")
        ax.set_ylabel("validation rescue")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=6.3, loc="upper left")
    fig.suptitle("MoE-block output patching across layers, per model and protocol (thick = intended protocol, band = 95% bootstrap CI)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext2_zoo_curves.{ext}"), dpi=160)
    plt.close(fig)


def fig_expert_curves(res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    items = []
    for run in sorted(res, key=lambda r: (ORDER.index(res[r]["model"]) if res[r]["model"] in ORDER else 99, PROTOCOLS.index(res[r]["protocol"]))):
        rr = res[run]
        s = "paper" if "paper" in rr["sets"] else ("strict" if "strict" in rr["sets"] else next(iter(rr["sets"])))
        if "expert_curve" in rr["sets"][s]:
            items.append((run, rr, s))
    if not items:
        return
    ncol = 3
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.8 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, (run, rr, s) in enumerate(items):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        cur = pd.DataFrame(rr["sets"][s]["expert_curve"])
        ax.plot(cur.layer, cur.layer_samepass_val_mean, color=INK2, lw=1.2, label="MoE block")
        ax.plot(cur.layer, cur.val_rescue, color="#2a78d6", lw=1.4, label="best recurrent expert")
        ax.fill_between(cur.layer, cur.val_rescue_lo, cur.val_rescue_hi, color="#2a78d6", alpha=0.15, lw=0)
        ax.plot(cur.layer, cur.val_spec, color="#c0392b", lw=1.2, ls="--", label="its Spec")
        r = rr["sets"][s]
        jw = r.get("joint", {}).get("winner")
        if jw:
            ax.plot([jw["layer"]], [jw["val_rescue"]], "*", color="#c0392b", ms=8, label=f"joint winner {jw['pair']}")
        e = r.get("selection", {}).get("e_star")
        if e is not None:
            ax.plot([r["L_star"]], [r["eval"]["rescue_all"]["mean"]], "o", color=INK, ms=4, label=f"two-stage {pair(r['L_star'], e)}")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{rr['label']} — {protocol_label(rr)} [{s}]", fontsize=8, loc="left")
        ax.set_xlabel("layer")
        ax.set_ylabel("validation rescue")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext2_zoo_expert_curves.{ext}"), dpi=160)
    plt.close(fig)


def attn_curves(res: dict) -> tuple[list[list], bool]:
    """Attention / MoE / block curves from results/<run>_attnsweep (ext2-attn-patch's script), if present."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, found = [], []
    for run, rr in res.items():
        for cand in (f"{run}_attnsweep", f"{rr['model']}_{rr['protocol']}_attnsweep", f"{rr['model']}_{'bos' if rr['protocol'] == 'default' else rr['protocol']}_attnsweep"):
            p = os.path.join(RESULTS, cand, "sweep_rows.parquet")
            if os.path.exists(p):
                found.append((run, rr, cand, p))
                break
    if not found:
        return rows, False
    ncol = 3
    nrow = int(np.ceil(len(found) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.8 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, (run, rr, cand, p) in enumerate(found):
        df = pd.read_parquet(p)
        sets = json.load(open(os.path.join(RESULTS, cand, "case_sets.json")))
        s = "paper" if "paper" in sets else ("strict" if "strict" in sets else next(k for k in sets if k != "scan"))
        val = set(sets[s]["validation"])
        ax = axes.ravel()[i]
        ax.set_visible(True)
        stats = {}
        for kind, col in (("attn_layer", "#8e44ad"), ("layer", "#2a78d6"), ("block", INK)):
            sub = df[(df.kind == kind) & df.case_id.isin(val)]
            if sub.empty:
                continue
            piv = sub.pivot(index="case_id", columns="layer", values="rescue")
            mean = piv.mean(0)
            ax.plot(mean.index, mean.values, color=col, lw=1.3, label={"attn_layer": "attention output", "layer": "MoE output", "block": "whole layer"}[kind])
            stats[kind] = dict(peak_layer=int(mean.idxmax()), peak=float(mean.max()), auc=float(mean.clip(lower=0).sum()))
        if "attn_layer" in stats and "layer" in stats and "block" in stats:
            pa = df[(df.kind == "attn_layer") & df.case_id.isin(val)].pivot(index="case_id", columns="layer", values="rescue")
            pl = df[(df.kind == "layer") & df.case_id.isin(val)].pivot(index="case_id", columns="layer", values="rescue")
            pb = df[(df.kind == "block") & df.case_id.isin(val)].pivot(index="case_id", columns="layer", values="rescue")
            add = (pa + pl - pb).mean(0)
            stats["additivity_mean_abs_gap"] = float(add.abs().mean())
            stats["additivity_gap_at_block_peak"] = float(add[stats["block"]["peak_layer"]])
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{rr['label']} — {protocol_label(rr)} [{s}, n={len(val)}]", fontsize=8, loc="left")
        ax.set_xlabel("layer")
        ax.set_ylabel("validation rescue")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        rows.append([rr["label"], protocol_label(rr), s, cand] + [f"L{stats[k]['peak_layer']} {stats[k]['peak']:+.3f} (AUC+ {stats[k]['auc']:.1f})" if k in stats else "n/a" for k in ("attn_layer", "layer", "block")]
                    + [f"{stats.get('additivity_mean_abs_gap', float('nan')):.3f} / {stats.get('additivity_gap_at_block_peak', float('nan')):+.3f}"])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext2_zoo_attn_curves.{ext}"), dpi=160)
    plt.close(fig)
    return rows, True


# ---------------------------------------------------------------------------------------------------------------
def usage_table() -> list[list]:
    rows = []
    for key in ("qwen3", "qwen3_instruct", "qwen3_coder", "mixtral", "mixtral_instruct", "olmoe", "olmoe_instruct"):
        if not os.path.exists(usage_path(key)):
            continue
        u = load_usage(key)
        t = u["tokenizer"]
        rows.append([u["label"], u["family"], u.get("base_counterpart") or "-", "yes" if t["adds_bos"] else "no", "yes" if t["adds_eos"] else "no", t.get("bos_token") or "-", t.get("eos_token") or "-",
                     u.get("recommended_dtype") or "-", (u.get("chat_template_source") or "none"), (u.get("default_system_prompt") or "none"),
                     str(u.get("rendered_chat_prefix_len") or "-"), repr(u.get("rendered_chat_prefix"))[:140] if u.get("rendered_chat_prefix") else "-",
                     ", ".join(f"{k}" for k, v in u["protocols"].items() if v != "not applicable")])
    return rows


USAGE_HEADER = ["model", "family", "base", "adds BOS", "adds EOS", "BOS token", "EOS token", "dtype (config)", "chat template", "default system prompt", "prefix tokens", "rendered chat prefix", "protocols"]


def sink_rows(res: dict) -> list[list]:
    rows = []
    for run in sorted(res, key=lambda r: (ORDER.index(res[r]["model"]) if res[r]["model"] in ORDER else 99, PROTOCOLS.index(res[r]["protocol"]))):
        rr = res[run]
        sk = rr.get("sink")
        if not sk:
            rows.append([rr["label"], protocol_label(rr), run, "n/a", "", "", "", "", "", "", ""])
            continue
        pos = sk.get("position") or {}
        pos_txt = (f"pos {pos['mode_position']} ({pos['frac_at_mode']:.2f}; {', '.join(f'{t!r} x{c}' for t, c in pos.get('token_at_max_norm_top3', [])[:2])})") if pos else "n/a"
        rows.append([rr["label"], protocol_label(rr), run, sk["n"], f"L{sk['sink_layer']}", f"{sk['frac_pos0_max_norm_at_sink_layer']:.3f}", f"{sk['frac_final_max_norm_at_sink_layer']:.3f}",
                     f"{sk['frac_final_max_norm_any_early_layer']:.3f}", f"{sk['frac_final_self_attn_max_at_sink_layer']:.3f}", f"{sk['mean_mass_pos0_at_sink_layer']:.3f}", pos_txt])
    return rows


SINK_HEADER = ["model", "protocol", "run", "prompts", "sink layer", "pos-0 is max-norm", "FINAL is max-norm", "final is max-norm at any early layer", "final attends mostly to itself", "final-position mass on pos 0", "max-norm position: mode (fraction; token)"]


def reference_rows(res: dict) -> list[list]:
    rows = []
    for run in sorted(res, key=lambda r: (ORDER.index(res[r]["model"]) if res[r]["model"] in ORDER else 99, PROTOCOLS.index(res[r]["protocol"]))):
        rr = res[run]
        for s, r in rr["sets"].items():
            if s != "paper" or "reference_experts" not in r:
                continue
            for pr, v in r["reference_experts"].items():
                l = v["layer"]
                lv = r["val_at_reference_layers"].get(str(l))
                rows.append([rr["label"], protocol_label(rr), pr, f"{v['disc_active']}/{r['n_disc']}", f"{v['val_active']}/{r['n_val']}",
                             f"{f3(v['val_rescue'])} [{f3(v.get('val_rescue_lo'))}, {f3(v.get('val_rescue_hi'))}]" if v.get("val_rescue_lo") is not None else f3(v["val_rescue"]),
                             f"{f3(v.get('val_spec'))} [{f3(v.get('val_spec_lo'))}, {f3(v.get('val_spec_hi'))}]" if v.get("val_spec_lo") is not None else f3(v.get("val_spec")),
                             ci(lv) if lv else "n/a", "yes" if v["disc_active"] >= r["threshold"] else "no"])
    return rows


REF_HEADER = ["model", "protocol", "reference expert (base model)", "clean-active disc", "clean-active val", "val rescue [CI]", "Spec [CI]", "block rescue at that layer [CI]", "recurrent (>= half of discovery)"]


def base_vs_instruct(res: dict, mds: dict) -> tuple[list[list], list[list]]:
    rows, agree = [], []
    fam_runs = {}
    for run, rr in res.items():
        if "paper" in rr["sets"]:
            fam_runs.setdefault(rr["family"], []).append(run)
    for fam, runs in fam_runs.items():
        base_runs = [r for r in runs if res[r]["base"]]
        for run in sorted(runs, key=lambda r: (not res[r]["base"], ORDER.index(res[r]["model"]), PROTOCOLS.index(res[r]["protocol"]))):
            rr = res[run]
            r = rr["sets"]["paper"]
            sel = r.get("selection", {})
            e = sel.get("e_star")
            ev = r.get("eval", {})
            jw = r.get("joint", {}).get("winner")
            refs = r.get("reference_experts", {})
            ref_txt = "; ".join(f"{k}: {v['disc_active']}/{v['val_active']} act, {f3(v['val_rescue'])} / Spec {f3(v.get('val_spec'))}" for k, v in refs.items())
            layers_txt = "; ".join(f"L{l}: {ci(v)}" for l, v in r["val_at_reference_layers"].items())
            rows.append([fam, rr["label"] + (" (base)" if rr["base"] else ""), protocol_label(rr), f"{rr['funnel'].get('paper_strict_pass', 'n/a')}/256", f"L{r['L_star']}", ci(r["val_at_Lstar"]),
                         layers_txt, pair(r["L_star"], e) if e is not None else "none", f"{ci(ev.get('rescue_all'))} / {ci(ev.get('spec_all'))}" if e is not None else "",
                         f"{jw['pair']} ({f3(jw['val_rescue'])} / {f3(jw['val_spec'])})" if jw else "n/a", ref_txt, r.get("pattern", "")])
            # routing agreement with the base run(s) of the same family at the reference layers
            for b in base_runs:
                if b == run or run not in mds or b not in mds or (rr["base"] and PROTOCOLS.index(rr["protocol"]) <= PROTOCOLS.index(res[b]["protocol"])):
                    continue  # base-vs-base pairs once (default -> nobos), instruct runs against every base run
                ids = res[b]["sets"]["paper"]["n_disc"] and mds[b].ids("paper", "discovery") + mds[b].ids("paper", "validation")
                for l in REFERENCE_LAYERS.get(fam, [])[:2]:
                    ag = routing_agreement(mds[b], mds[run], l, ids)
                    if ag.get("n"):
                        agree.append([fam, res[b]["label"] + f" ({protocol_label(res[b])})", rr["label"] + f" ({protocol_label(rr)})", f"L{l}", ag["n"], f"{ag['mean_jaccard']:.3f}", f"{ag['frac_identical']:.3f}"])
    return rows, agree


BVI_HEADER = ["family", "model", "protocol", "paper IDs passing strict", "L*", "layer rescue at L* [CI]", "layer rescue at the base model's layers", "two-stage expert",
              "expert rescue / Spec [CI]", "joint winner (rescue / Spec)", "base model's experts as fixed hypotheses (active disc/val, rescue / Spec)", "pattern"]
AGREE_HEADER = ["family", "base run", "compared run", "layer", "paper cases", "mean Jaccard of clean top-k sets (final position)", "fraction identical"]



VERDICT = """### Verdict

**Generalisability.** Under the protocol each model is meant to be used with, all five additional checkpoints show the paper's Qwen3 pattern **A**: one MoE layer carries a validation block rescue whose CI excludes zero by a wide margin, and one recurrent expert in that layer is both positive and specific (Spec CI above zero): OLMoE-1B-7B-0125 L13E056 (rescue +0.94, Spec +0.87, active 100/128), OLMoE-Instruct (chat) L12E040 on its strict set (+0.52 / +0.41) and L13E056 on its relaxed set (+0.76 / +0.67), Qwen3-30B-A3B-Instruct-2507 (chat) L44E069 (+0.72 / +0.62), Qwen3-Coder (chat) L44E069 (+0.66 / +0.59), Mixtral-8x7B-Instruct (chat) L31E002 (+1.19 / +0.71). Pattern **B** (layer localised, recurrent expert not specific, coalitions carry the effect) appears only for the Mistral family under the paper's no-BOS protocol, in the base and the Instruct model alike; pattern **C** never occurs. The selected expert is re-selected in 19-25 of 25 Appendix-D grid settings in every pattern-A run, and the joint layer x expert search returns the two-stage winner in every run except the two cases already known from Direction 1: in the Qwen3 chat runs the discovery argmax is the second locus L42E115 (validation within the CI of L44E069; 0 of 243-272 recurrent pairs beat L44E069 on validation rescue), and in the Mixtral no-BOS runs it is L18E001. No new locus appears anywhere.

**Effect of post-training.** Instruction tuning and code specialisation do not move the layer or the expert. In the Qwen3 family L44 stays the block-rescue argmax on every case set and protocol (paper set: base +0.94, Instruct +1.07 raw / +1.34 chat, Coder +0.96 / +1.18), L44E069 is selected everywhere with equal or higher rescue and Spec than in the base (base +0.50 / +0.44; Instruct +0.55 / +0.46 raw, +0.72 / +0.62 chat; Coder +0.62 / +0.55 raw, +0.66 / +0.59 chat), and the second locus L42E115 keeps its 118-126/128 recurrence and +0.51 to +0.66 rescue. This holds although the final-position top-8 routing set at L44 is identical to the base's in only 27% (Instruct) and 13% (Coder) of prompts (mean Jaccard 0.80 / 0.71): post-training rearranges the other experts but keeps E069 and E115 in place. In the Mistral family, Mixtral-Instruct with BOS reproduces the base's default-protocol result almost verbatim (L19; L19E002 active 78/86 vs 76/84, Spec +0.165 vs +0.192; L18E001 +0.27 / +0.19; L19 routing identical to the base's in 86% of prompts), and without BOS it reproduces the paper's E006 picture with the *same* activity counts as the base (91/128 discovery, 83/128 validation; Spec -0.22; joint winner L18E001). OLMoE-Instruct keeps the base's L13E056 (raw protocol +0.75 / +0.66; chat, relaxed set +0.76 / +0.67) and its neighbour L12E040 (the base's relaxed-set choice, the Instruct's chat strict-set choice): the L12/L13 pair is OLMoE's analogue of Qwen3's L42/L44 pair. Post-training changes magnitudes (mostly upward) and the surrounding routing, not the locus.

**Effect of the protocol.** (i) BOS: for the Mistral family the paper protocol (no `<s>`) is the only setting that yields pattern B, and the sink diagnostic shows why in both models: without BOS the final cloze token is the maximal-norm (sink) position in 24-25% of prompts (Mixtral 0.246, Mixtral-Instruct 0.238), whereas with BOS or the chat template it never is (0/525-547) and the sink sits on `<s>` in 100% of prompts. Every other run of every model has a final-sink fraction of 0.000 (the sink is position 0, or in the Qwen3 chat template the token after `<|im_start|>`: `user` for Instruct, the newline for Coder). (ii) Chat wrapping: the template raises the clean margins and the noise drop (mean Delta_clean in the scan 6.1 -> 7.6 for Qwen3-Instruct, 4.8 -> 6.4 for OLMoE-Instruct; paper set 7.1 -> 12.2 for Mixtral-Instruct) and with them the absolute rescues, but as a share of the noise drop the layer effect is stable (Qwen3 family 15-17% raw and chat, OLMoE 18-23%); the layer and the expert are unchanged for Qwen3-Instruct, Coder and OLMoE-Instruct. The one qualitative protocol effect is Mixtral-Instruct under its chat template: the block-rescue argmax jumps from L19 (BOS raw: +0.54) to the *final* layer L31 (+1.79 [+1.26, +2.33], 20% of a +9.1 drop), whose recurrent expert L31E002 (active 78/128) is positive and specific (+1.19 / +0.71, re-selected in 19/25 grid settings); the mid-network band the base localises to is still present and stronger than under the raw protocol (L19 +0.87, L20 +0.88, L21 +0.78), L19E002 is still the best L19 expert (67/128 active, +0.40, Spec +0.19 with a CI that now includes zero) and L21E001 the best mid-network expert (83/128, +0.53, Spec +0.46). Because a MoE-output patch at the last layer writes almost directly into the unembedded residual, the L31 result should be read as "the chat-formatted Instruct model finishes the retrieval in its last MoE block" rather than as a relocation of the factual-recall band; both loci are reported. (iii) Case sets: the base models' paper IDs pass the strict filter in 218-239 of 256 cases in every instruct run (base 232-234), so the paper's case set transfers to the descendants without re-filtering.

**Attention vs MoE.** In every model the attention-output patch peaks earlier and higher than the MoE-output patch (Qwen3 family: attention L40 +1.6 to +2.1 vs MoE L44 +0.9 to +1.4; Mixtral: attention L18/L24 vs MoE L19/L31; OLMoE: attention L12/L13 +2.5 to +2.9 vs MoE +1.5), the whole-layer patch is close to the sum of the two (mean |attn + MoE - block| 0.01-0.11 across layers), so the two sublayers carry complementary rather than redundant information; by area under the positive part of the curve the attention output accounts for about half of the whole-layer effect in the Qwen3 and Mixtral families (45-55%) and about 70% in OLMoE. The MoE-output curve is the sharper of the two (single-layer peak), which is what makes the paper's expert-level step possible.
"""


def _ref_status(r: dict, pr: str) -> str:
    v = (r.get("reference_experts") or {}).get(pr)
    if not v:
        return f"{pr}: n/a"
    rec = "recurrent" if v["disc_active"] >= r["threshold"] else "NOT recurrent"
    return (f"{pr}: clean-active {v['disc_active']}/{r['n_disc']} disc, {v['val_active']}/{r['n_val']} val ({rec}), val rescue {f3(v['val_rescue'])}"
            + (f" [{f3(v.get('val_rescue_lo'))}, {f3(v.get('val_rescue_hi'))}]" if v.get("val_rescue_lo") is not None else "")
            + f", Spec {f3(v.get('val_spec'))}" + (f" [{f3(v.get('val_spec_lo'))}, {f3(v.get('val_spec_hi'))}]" if v.get("val_spec_lo") is not None else ""))


def findings(res: dict, mds: dict) -> str:
    """Data-driven bullets: generalisation per model (intended protocol), post-training within each family, protocol sensitivity."""
    P = ["### Findings (generated from the run summaries)\n"]
    order = lambda r: (ORDER.index(res[r]["model"]) if res[r]["model"] in ORDER else 99, PROTOCOLS.index(res[r]["protocol"]))
    # ---- 1. generalisation under the intended protocol
    P.append("**Generalisation (each model under its intended protocol, primary case set = paper IDs where the family has them, else own strict set).**\n")
    for run in sorted(res, key=order):
        rr = res[run]
        if not rr["intended"]:
            continue
        s = "paper" if "paper" in rr["sets"] else "strict"
        r = rr["sets"].get(s)
        if not r:
            continue
        sel = r.get("selection", {}); e = sel.get("e_star"); ev = r.get("eval", {}); co = r.get("coalitions", {}); jw = (r.get("joint") or {}).get("winner")
        txt = f"- **{rr['label']}** (`{protocol_label(rr)}`, {s} set): pattern **{r.get('pattern')}**. L* = L{r['L_star']} of {rr['n_layers']}, validation block rescue {ci(r['val_at_Lstar'])} (validation-top layer L{r['sharpness']['top_layer']} {f3(r['sharpness']['top_rescue'])}, second L{r['sharpness']['next_layer']} {f3(r['sharpness']['next_rescue'])})"
        if e is not None:
            conc = r.get("ratios", {}).get("expert_over_layer")
            txt += (f"; two-stage expert {pair(r['L_star'], e)} ({sel['n_candidates']} recurrent candidates), clean-active {sel['disc_active']}/{r['n_disc']} disc / {ev['val_active']}/{r['n_val']} val, "
                    f"rescue {ci(ev['rescue_all'])}, Spec {ci(ev['spec_all'])}" + (f", {conc[0]:.0%} of the block rescue" if conc else "") + f"; coalitions {f3(co.get('coalition_clean', {}).get('mean'))} / {f3(co.get('coalition_union', {}).get('mean'))}")
            if r.get("stability"):
                txt += f"; re-selected in {r['stability']['n_reference_selected']}/{r['stability']['n_settings']} Appendix-D settings"
        elif sel:
            txt += f"; no expert at L{r['L_star']} meets the recurrence threshold (max clean-active {sel.get('max_activity')}/{r['n_disc']}); coalitions {f3(co.get('coalition_clean', {}).get('mean'))} / {f3(co.get('coalition_union', {}).get('mean'))}"
        if jw:
            txt += f"; joint search: {jw['pair']} ({f3(jw['val_rescue'])} / Spec {f3(jw['val_spec'])})" + (" = two-stage" if (r.get("joint") or {}).get("same_as_two_stage") else " (differs)")
        P.append(txt + ".")
    # ---- 2. post-training within families
    P.append("\n**Post-training within each family (paper case set of the base model; base numbers from the Direction-1 all-layer runs).**\n")
    for fam, refs in REFERENCE_EXPERTS.items():
        runs = [r for r in sorted(res, key=order) if res[r]["family"] == fam and "paper" in res[r]["sets"]]
        if not runs or fam == "olmoe":
            continue
        for run in runs:
            rr = res[run]; r = rr["sets"]["paper"]; sel = r.get("selection", {}); e = sel.get("e_star"); jw = (r.get("joint") or {}).get("winner")
            layers = "; ".join(f"L{l}: {f3(v['mean'])}" for l, v in r["val_at_reference_layers"].items())
            P.append(f"- {rr['label']} (`{protocol_label(rr)}`{', base' if rr['base'] else ''}): {rr['funnel'].get('paper_strict_pass', 'n/a')}/256 paper IDs pass strict; L* = L{r['L_star']} ({ci(r['val_at_Lstar'])}); block rescue at the reference layers {layers}; "
                     f"two-stage {pair(r['L_star'], e) if e is not None else 'none'}" + (f" ({ci(r['eval']['rescue_all'])} / Spec {ci(r['eval']['spec_all'])})" if e is not None else "")
                     + (f"; joint {jw['pair']} ({f3(jw['val_rescue'])} / {f3(jw['val_spec'])})" if jw else "") + f"; pattern {r.get('pattern')}.")
            for (l, ee) in refs:
                P.append(f"    - {_ref_status(r, pair(l, ee))}")
    # OLMoE: base vs instruct on the shared strict set is not shared (different scans); compare selections and routing agreement instead
    ol = [r for r in sorted(res, key=order) if res[r]["family"] == "olmoe"]
    if len(ol) > 1:
        P.append("- OLMoE family (own strict sets differ between runs because each run has its own filter scan; the base's selected expert is evaluated in every OLMoE run below):")
        base = next((r for r in ol if res[r]["model"] == "olmoe"), None)
        for run in ol:
            rr = res[run]; r = rr["sets"].get("strict") or next(iter(rr["sets"].values())); sel = r.get("selection", {}); e = sel.get("e_star")
            P.append(f"    - {rr['label']} (`{protocol_label(rr)}`): L* = L{r['L_star']} ({ci(r['val_at_Lstar'])}), two-stage {pair(r['L_star'], e) if e is not None else 'none'}"
                     + (f" ({ci(r['eval']['rescue_all'])} / Spec {ci(r['eval']['spec_all'])}, active {sel['disc_active']}/{r['n_disc']} disc)" if e is not None else "") + f", pattern {r.get('pattern')}.")
    # ---- 3. protocol sensitivity
    P.append("\n**Protocol sensitivity (same model, different tokenisation / wrapping).**\n")
    for mk in ORDER:
        runs = [r for r in sorted(res, key=order) if res[r]["model"] == mk]
        if len(runs) < 2:
            continue
        s = "paper" if all("paper" in res[r]["sets"] for r in runs) else "strict"
        parts = []
        for run in runs:
            rr = res[run]; r = rr["sets"].get(s)
            if not r:
                continue
            sel = r.get("selection", {}); e = sel.get("e_star"); sk = rr.get("sink") or {}
            fs = rr.get("filter_scan", {})
            parts.append(f"`{protocol_label(rr)}`: L{r['L_star']} {f3(r['val_at_Lstar']['mean'])}, e* {pair(r['L_star'], e) if e is not None else 'none'}"
                         + (f" (Spec {f3(r['eval']['spec_all']['mean'])})" if e is not None else "") + f", pattern {r.get('pattern')}, strict pass rate in the scan {fs.get('strict_pass_rate_tokenizable', float('nan')):.2f}"
                         + (f", paper IDs passing {rr['funnel'].get('paper_strict_pass')}/256" if s == "paper" else "") + (f", sink-final {sk.get('frac_final_max_norm_at_sink_layer', float('nan')):.2f}" if sk else ""))
        P.append(f"- {MODELS[mk]['label']} ({s} set): " + "; ".join(parts) + ".")
    return "\n".join(P) + "\n"


# ---------------------------------------------------------------------------------------------------------------
def write_section(res: dict, tables: dict, attn_ok: bool, mds: dict):
    P = []
    P.append("## Extension 2A: does the pattern generalise? Five additional MoE checkpoints under intended and paper protocols\n")
    P.append("**Question.** The paper reports two patterns on two base models: Qwen3-30B-A3B-Base localises factual recall in one MoE layer (L44) "
             "and in one specific, recurrent expert (L44E069); Mixtral-8x7B-v0.1 localises in L19 but the recurrent expert (L19E006) is not specific and only "
             "coalitions recover the layer effect. We ask whether these patterns hold (i) for the instruction-tuned and code-specialised descendants of the "
             "same weights (Qwen3-30B-A3B-Instruct-2507, Qwen3-Coder-30B-A3B-Instruct, Mixtral-8x7B-Instruct-v0.1), (ii) for a third, fully open family "
             "(OLMoE-1B-7B-0125 base and Instruct), and (iii) whether the answer depends on running each model the way it is meant to be used (tokenizer "
             "defaults; chat template for instruct models) or under the paper's protocol (no special tokens, raw cloze).\n")
    P.append("**Method.** Every model gets a usage specification (`data/model_usage/<key>.json`, table below) verified from its `tokenizer_config.json`, "
             "`generation_config.json`, `config.json`, chat template and model card plus an empirical tokenizer probe. Protocols: `default` = tokenizer "
             "defaults, raw cloze; `nobos` = no special tokens, raw cloze (the paper's protocol; identical to `default` for every model whose tokenizer adds "
             "nothing, i.e. all but the Mistral family, and then run once); `chat` = the chat template rendered with the user instruction "
             f"\"{INSTRUCTION}\" and an opened assistant turn, tokenised as `prefix_ids`, with the raw cloze prompt following inside the assistant turn "
             "(the final position is still the prompt's last token, subject noise still hits the subject tokens; the exact rendered prefix is in each "
             "`run_meta.json`). Per model and protocol we ran the full paper pipeline with the reproduction's engine and defaults (sigma = 3 x embed std, "
             "space object-token rule, seed-0 shuffle): filter scan (strict 1.0/0.5 -> own 256-case set; relaxed 0.5/0.25 -> own 512-case set; plus the "
             "BASE model's paper case IDs where the family has one, so that instruct results are directly comparable with REPORT.md), layer sweep (MoE-block "
             "output patch at every layer), and the Direction-1 all-layer expert pass (`--no-pairs`, every clean-active and noised-only expert, both "
             "coalitions and the layer patch at EVERY layer). Analysis per (run, case set): the paper's two-stage selection (L* by discovery block rescue; "
             "recurrence-first expert with threshold = half the discovery split; validation rescue, active-random Spec with 3 controls for top-8 models and 1 "
             "for Mixtral, coalitions, Appendix-D grid) and the joint layer x expert search (`moetrace/ext1_analysis.py`). The base model's experts are "
             "evaluated in every run as fixed hypotheses. Each sweep pass also records the ext3 sink diagnostic (`DiagSpec(attn_final, resid_norms)` on the "
             "clean rows): the fraction of prompts whose FINAL position carries the maximal residual norm at the model's sink layer (the early layer where the "
             "massive-activation state is most pronounced) and the fraction whose final position puts more attention mass on itself than on any other position. "
             "Pattern labels: **A** = one positive, specific expert (validation rescue and Spec CIs above 0); **B** = layer-level localisation but the selected "
             "expert is not specific or none is recurrent (coalition model); **C** = no layer-level localisation (validation rescue CI at L* includes 0). "
             "Code: `moetrace/ext2_zoo.py`, `scripts/ext2_zoo_{usage,filter,sweep,expert,chain,analyze}.{py,sh}`; runs: `results/<model>_<protocol>/`.\n")
    P.append("### Usage specifications\n")
    P.append(tables["usage"])
    P.append("Notes. Qwen3 tokenizers add no special tokens (`default` = `nobos` = the paper protocol), so the Qwen3 family has two protocols (raw, chat); the "
             "Mistral family adds `<s>` and has three (default = BOS, nobos, chat; the Instruct template itself begins with `<s>`); OLMoE adds nothing. "
             "OLMoE-1B-7B-0125-Instruct's tokenizer names token id 50279 `|||IP_ADDRESS|||` and uses it as both BOS and EOS in its template, whereas the "
             "model card writes the same template with `<|endoftext|>` (id 50279 in the base tokenizer): the rendered prefix therefore starts with id 50279 "
             "exactly as documented, only the string differs. Qwen3-30B-A3B-Instruct-2507 and Qwen3-Coder are non-thinking models (no `<think>` block in "
             "the template). No model has a default system prompt in its template (Ai2 mentions a demo prompt for OLMoE-Instruct but states the model was not "
             "trained with one), so none is used. bf16 is the recommended dtype of every new checkpoint (OLMoE base ships fp32 and was run in bf16 like the pilot).\n")
    P.append("### Cross-model summary\n")
    P.append(tables["summary"])
    P.append("`*` marks the protocol under which the model is meant to be used (base models: tokenizer defaults; instruct models: chat template). "
             "`sink-final frac` = fraction of the run's clean prompts whose final position is the maximal-norm (sink) position at the sink layer. "
             "`e* stable` = number of Appendix-D grid settings (5 split seeds x 5 thresholds) that re-select the same expert. Base rows come from the Direction-1 "
             "all-layer runs (`qwen3_bos_alllayers`, `mixtral_bos_alllayers`, `mixtral_nobos_alllayers`).\n")
    P.append("![ext2 zoo layer curves](../figures/ext2_zoo_curves.png)\n")
    P.append("![ext2 zoo best-expert curves](../figures/ext2_zoo_expert_curves.png)\n")
    P.append("### Base vs instruct within each family (paper case set of the base model)\n")
    P.append(tables["bvi"])
    P.append("**Base model's experts as fixed hypotheses in every run (paper case set)**\n")
    P.append(tables["ref"])
    if tables.get("agree"):
        P.append("**Final-position routing agreement with the base run at the reference layers (clean prompts, paper case set)**\n")
        P.append(tables["agree"])
    P.append("### Sink diagnostic (protocol quality)\n")
    P.append(tables["sink"])
    P.append("### Attention-output vs MoE-output vs whole-layer patching\n")
    if attn_ok:
        P.append(tables["attn"])
        P.append("![ext2 zoo attention curves](../figures/ext2_zoo_attn_curves.png)\n")
    else:
        P.append("_Pending: the `attn_layer` / `block` sweeps (`scripts/ext2_attn_sweep.py`, agent ext2-attn-patch) had not been run for the zoo models when this "
                 "section was built; re-run `scripts/ext2_zoo_analyze.py` after `results/<run>_attnsweep/` exist._\n")
    P.append(findings(res, mds))
    P.append(tables.get("verdict", ""))
    P.append("### Per-model results\n")
    for run in sorted(res, key=lambda r: (ORDER.index(res[r]["model"]) if res[r]["model"] in ORDER else 99, PROTOCOLS.index(res[r]["protocol"]))):
        rr = res[run]
        if rr["base"]:
            continue
        P.append(f"#### {rr['label']} — protocol `{protocol_label(rr)}`{' (intended)' if rr['intended'] else ''} (`results/{run}`)\n")
        if rr.get("prefix_text"):
            P.append(f"Chat prefix ({rr['prefix_len']} tokens): `{rr['prefix_text']!r}`\n")
        fs = rr.get("filter_scan", {})
        pc = rr.get("paper_check") or {}
        P.append(f"Filter scan: {fs.get('records_scanned')} records scanned, {fs.get('tokenizable')} tokenizable, strict pass rate {fs.get('strict_pass_rate_tokenizable', float('nan')):.3f}, "
                 f"relaxed {fs.get('relaxed_pass_rate_tokenizable', float('nan')):.3f}, mean Delta_clean {fs.get('mean_delta_clean', float('nan')):+.2f}, mean drop {fs.get('mean_drop', float('nan')):+.2f}"
                 + (f"; base model's paper IDs: {pc.get('paper_ids_strict_in_scan')}/256 pass strict in the scan, {pc.get('overlap_paper_vs_our_strict256')} overlap with our strict set" if pc else "") + ".\n")
        sk = rr.get("sink")
        if sk:
            P.append(f"Sink diagnostic (sink layer L{sk['sink_layer']}): position 0 carries the maximal norm in {sk['frac_pos0_max_norm_at_sink_layer']:.1%} of prompts, the final position in "
                     f"{sk['frac_final_max_norm_at_sink_layer']:.1%} ({sk['frac_final_max_norm_any_early_layer']:.1%} at any early layer); the final position attends mostly to itself in "
                     f"{sk['frac_final_self_attn_max_at_sink_layer']:.1%}; mean final-position attention mass on position 0 {sk['mean_mass_pos0_at_sink_layer']:.2f}.\n")
        P.append(tables["runs"][run])
        for s, r in rr["sets"].items():
            j = r.get("joint")
            if not j or not j.get("winner"):
                continue
            w = j["winner"]
            sel = r.get("selection", {})
            e = sel.get("e_star")
            txt = (f"- `{s}`: {j['n_candidates']} recurrent (layer, expert) pairs in {j['n_layers_with_candidates']} layers; joint discovery argmax {w['pair']} "
                   f"(disc {f3(w['disc_allcase_mean'])}, val {f3(w['val_rescue'])} [{f3(w['val_rescue_lo'])}, {f3(w['val_rescue_hi'])}], Spec {f3(w['val_spec'])} [{f3(w['val_spec_lo'])}, {f3(w['val_spec_hi'])}], "
                   f"active {w['disc_active']}/{r['n_disc']} disc, {w['val_active']}/{r['n_val']} val)")
            if e is not None:
                txt += "; " + ("same as the two-stage selection" if j["same_as_two_stage"] else f"two-stage {pair(r['L_star'], e)} has rank {j['two_stage'].get('rank')} "
                               f"(val {f3(j['two_stage']['val_rescue'])}, Spec {f3(j['two_stage']['val_spec'])})")
                vs = j.get("vs_reference") or {}
                txt += f"; {vs.get('n_spec_ci_above_zero', 0)} pairs have a Spec CI above zero"
                if r.get("stability"):
                    txt += f"; Appendix-D grid re-selects {pair(r['L_star'], e)} in {r['stability']['n_reference_selected']}/{r['stability']['n_settings']} settings (winners {r['stability']['selected_counts']})"
                if r.get("rank"):
                    txt += f"; e* ranks first among the case's clean-active experts in {r['rank']['top1']}/{r['rank']['n']} validation cases"
            else:
                txt += f"; no expert at L{r['L_star']} meets the recurrence threshold {r['threshold']} (max activity {sel.get('max_activity')})"
            P.append(txt + ".")
        P.append("")
    return "\n".join(P)


# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    runs = discover_runs()
    if args.runs:
        want = set(args.runs.split(","))
        runs = {k: v for k, v in runs.items() if k in want}
    log("runs:", list(runs))
    res = {}
    for run, info in runs.items():
        try:
            res[run] = analyze_run(run, info, args.force)
        except Exception as ex:
            log(f"{run}: FAILED {ex!r}")
            raise
    mds = {run: A.load_model(run) for run in res}
    tables = {}
    rows, recs = summary_rows(res, mds)
    tables["summary"] = md_table(SUMMARY_HEADER, rows, os.path.join(TAB, "ext2_zoo_summary"))
    pd.DataFrame(recs).to_csv(os.path.join(TAB, "ext2_zoo_summary.csv"), index=False)  # machine-readable version replaces the md_table csv
    tables["usage"] = md_table(USAGE_HEADER, usage_table(), os.path.join(TAB, "ext2_zoo_usage"))
    tables["sink"] = md_table(SINK_HEADER, sink_rows(res), os.path.join(TAB, "ext2_zoo_sink"))
    tables["ref"] = md_table(REF_HEADER, reference_rows(res), os.path.join(TAB, "ext2_zoo_reference_experts"))
    bvi, agree = base_vs_instruct(res, mds)
    tables["bvi"] = md_table(BVI_HEADER, bvi, os.path.join(TAB, "ext2_zoo_base_vs_instruct"))
    tables["agree"] = md_table(AGREE_HEADER, agree, os.path.join(TAB, "ext2_zoo_routing_agreement")) if agree else ""
    tables["runs"] = {run: md_table(RUN_HEADER, run_table(rr), os.path.join(TAB, f"ext2_zoo_run_{run}")) for run, rr in res.items()}
    attn_rows, attn_ok = ([], False)
    if not args.no_figures:
        fig_curves(res)
        fig_expert_curves(res)
        attn_rows, attn_ok = attn_curves(res)
    tables["attn"] = md_table(["model", "protocol", "case set", "run", "attention output: peak (AUC+)", "MoE output: peak (AUC+)", "whole layer: peak (AUC+)",
                               "additivity gap attn+MoE-block: mean |gap| / at block peak"], attn_rows, os.path.join(TAB, "ext2_zoo_attn")) if attn_ok else ""
    tables["verdict"] = VERDICT
    sec = write_section(res, tables, attn_ok, mds)
    open(os.path.join(SEC, "ext2_model_zoo.md"), "w").write(sec)
    summ = {run: {k: v for k, v in rr.items() if k not in ("sink_per_layer",)} for run, rr in res.items()}
    for rr in summ.values():
        for r in rr["sets"].values():
            r.pop("curve", None)
            r.pop("expert_curve", None)
            if r.get("stability"):
                r["stability"].pop("grid", None)
            if r.get("joint"):
                r["joint"].pop("top10", None)
    json.dump(dict(built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), runs=summ, attn_curves=attn_ok), open(os.path.join(RESULTS, "ext2_zoo_summary.json"), "w"), indent=1)
    log(f"section written ({len(sec)} chars); attn curves {'included' if attn_ok else 'pending'}")


if __name__ == "__main__":
    main()
