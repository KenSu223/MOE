"""ext12 analysis (CPU only): 4a add-back replication / CounterFact fold swap, 4b role swap vs option swap on identical
items, 4c Mixtral without BOS on WinoGrande. Missing inputs are skipped (rerun when the GPU chains have finished).

4a / 4c add-back: every ext12 run (moetrace/ext12_complete.TASKS) is analysed with ext8's own `analyse` (scripts/
ext8_addback_analyze.py, imported; only its task loader is replaced) and, for CIs of the curve summaries (AUC over log k
full and k <= 15, k80, overshoot, greedy - static, DLA - oracle, answer restored), with a unit-cluster bootstrap of the
per-case curve matrices (5,000 resamples, seed 0, as ext8's ratio_cols; units = CounterFact cases / WinoGrande pairs).
Comparisons with the Phase-3 runs (results/<ext8 run>, read-only): replication / fold swap vs main = independent samples
(difference of independent bootstrap draws, main seed 0, other seed 1); Mixtral no BOS vs BOS = identical pairs (paired
bootstrap, same resamples).
4b: twin-level paired comparison (units = WinoGrande twins; both role pairs of a twin pooled), plus the clean-prompt-matched
variant (role d = 0 vs the option-swap directed case with the same clean prompt and the same r, r').
4c: the ext7-wino analysis functions (scripts/ext7_wino_analyze.py: descriptors, w2, w6, joint) on results/
wino_mixtral_nobos_str, paired no-BOS - BOS differences on the identical main pairs, sink-carrying final-token strata.

Usage: python scripts/ext12_analyze.py [--parts 4a,4b,4c]
Outputs results/ext12_complete_summary.json, results/tables/ext12_*.{md,csv}, results/figures/ext12_*.{png,pdf}
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, pandas as pd
from moetrace import ext8_addback as X
from moetrace import ext12_complete as C
from moetrace import ext7_pairs as P
from moetrace.models import RESULTS

TAB, FIG = os.path.join(RESULTS, "tables"), os.path.join(RESULTS, "figures")
SUMM = os.path.join(RESULTS, "ext12_complete_summary.json")
ORD = ("oracle", "pop", "layerwise", "dla", "vnorm", "weight", "rand")
NB = 5000
KG = 15


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsafe(o):
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
    if isinstance(o, (list, tuple)):
        return [jsafe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else (None if np.isnan(o) else ("inf" if o > 0 else "-inf"))
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsafe(o.tolist())
    return o


def md_table(header, rows, name, caption=None):
    lines = ([f"**{caption}**", ""] if caption else []) + ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    txt = "\n".join(lines) + "\n"
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name + ".md"), "w") as f:
        f.write(txt)
    pd.DataFrame(rows, columns=header).to_csv(os.path.join(TAB, name + ".csv"), index=False)
    return txt


def pct(x, lo=2.5, hi=97.5, method="linear"):
    """Percentile CI (linear interpolation as ext8 / moetrace.stats; method='inverted_cdf' for grid-valued k, which may be inf)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return [float("nan"), float("nan")]
    return [float(np.percentile(x, lo, method=method)), float(np.percentile(x, hi, method=method))]


# =================================================================================================================
# 4a / 4c add-back
# =================================================================================================================
def addback_mats(t, R):
    """Per evaluation case: drop (donor mean of the pass-0 reference drop), all-MoE ceiling rescue, all-MoE Delta, static
    curves per ordering (case x k rescue, all_active endpoint at k = K), greedy curves (case x 1..15), units."""
    rows, st = R["rows"], R["state"]
    cases = t.cases.set_index("case_id")
    val = [int(c) for c in cases.index[cases.split == "validation"]]
    dref = rows.drop_duplicates("row_id").set_index("row_id").drop_ref
    rcase = t.rows.set_index("row_id").case_id
    drop_case = dref.groupby(rcase.loc[dref.index].values).mean().reindex(val)
    a0 = rows[(rows.fam == "a0") & (rows.order == "all_moe") & (rows.dir == "d") & rows.case_id.isin(val)]
    ceil = a0.groupby("case_id").rescue.mean().reindex(val)
    allmoe_delta = a0.groupby("case_id").delta.mean().reindex(val)
    d = rows[(rows.fam == "a1") & (rows.dir == "d") & rows.case_id.isin(val)].copy()
    aa = d[d.order == "all_active"].groupby("case_id").rescue.mean().reindex(val)
    d = d[d.order != "all_active"]
    d["order2"] = np.where(d.order.str.startswith("rand"), "rand", d.order)
    d = d.groupby(["case_id", "row_id", "order2", "k"], as_index=False).rescue.mean()
    mats = {}
    for o in ORD:
        M = d[d.order2 == o].groupby(["case_id", "k"]).rescue.mean().unstack("k").reindex(val)
        if M.empty:
            continue
        M[t.K] = aa
        M = M[sorted(M.columns)]
        assert not M.isna().any().any(), (o, int(M.isna().sum().sum()))
        mats[o] = M
    gr = []
    for rid, g in st.greedy.items():
        rr = list(g.resc) + [g.resc[-1]] * (X.GREEDY_STEPS - len(g.resc)) if g.resc else [np.nan] * X.GREEDY_STEPS
        gr.append({"case_id": int(t.rows.case_id.iloc[rid]), **{k + 1: rr[k] for k in range(X.GREEDY_STEPS)}})
    gm = pd.DataFrame(gr).groupby("case_id").mean().reindex(val) if gr else None
    unit = cases.unit.reindex(val)
    return dict(val=val, unit=unit, drop=drop_case, ceil=ceil, allmoe_delta=allmoe_delta, mats=mats, greedy=gm, K=t.K,
                n_greedy_done=int(sum(g.done for g in st.greedy.values())), n_greedy=len(st.greedy))


def unit_weights(units: np.ndarray, seed: int, n_boot: int = NB):
    """(unit -> column index, W [n_boot + 1, n_units]); row 0 = the point estimate (all weights 1)."""
    u = np.unique(units)
    W = np.ones((n_boot + 1, len(u)), dtype=np.float64)
    if n_boot:
        uu, Wb = X.boot_weights(units, n_boot, seed)
        assert (uu == u).all()
        W[1:] = Wb
    return {x: i for i, x in enumerate(u)}, W


def curve_boot(m, W, upos):
    """Bootstrap distributions (row 0 = point estimate) of the add-back summaries."""
    units = m["unit"].to_numpy()
    Mu = np.zeros((W.shape[1], len(units)))
    Mu[[upos[x] for x in units], np.arange(len(units))] = 1.0
    WU = W @ Mu  # [B, cases] case weights
    sd = WU @ m["drop"].to_numpy()
    out = {"ceiling": (WU @ m["ceil"].to_numpy()) / sd}
    out["restored_all_moe"] = (WU @ (m["allmoe_delta"].to_numpy() > 0).astype(float)) / WU.sum(1)
    ks15 = np.arange(1, KG + 1)
    for o, M in m["mats"].items():
        ks = np.array(M.columns, dtype=float)
        r = (WU @ M.to_numpy()) / sd[:, None]
        x = np.log(ks)
        e = {"k": ks, "r": r, "auc_log": np.trapezoid(r, x, axis=1) / x.max()}
        r15 = np.stack([np.interp(np.log(ks15), x, ri) for ri in r])
        e["auc_log_k15"] = np.trapezoid(r15, np.log(ks15), axis=1) / np.log(KG)
        e["r10"] = r[:, list(ks).index(10)]
        thr = 0.8 * out["ceiling"]
        hit = r >= thr[:, None]
        e["k80"] = np.where(hit.any(1), ks[np.argmax(hit, axis=1)], np.inf)
        e["max_r"] = r.max(1)
        e["overshoot"] = e["max_r"] - out["ceiling"]
        out[o] = e
    if m["greedy"] is not None:
        G = m["greedy"].to_numpy()
        ok = ~np.isnan(G).any(1)
        WG = WU[:, ok]
        rg = (WG @ G[ok]) / (WG @ m["drop"].to_numpy()[ok])[:, None]
        e = {"k": ks15.astype(float), "r": rg, "auc_log_k15": np.trapezoid(rg, np.log(ks15), axis=1) / np.log(KG), "r10": rg[:, 9],
             "r15": rg[:, 14]}
        thr = 0.8 * out["ceiling"]
        hit = rg >= thr[:, None]
        e["k80"] = np.where(hit.any(1), ks15[np.argmax(hit, axis=1)].astype(float), np.inf)
        out["greedy"] = e
        if "oracle" in out:
            out["greedy_minus_oracle_r10"] = e["r10"] - out["oracle"]["r10"]
            best = np.max(np.stack([out[o]["r10"] for o in ORD if o in out and o != "rand"]), axis=0)
            out["greedy_minus_best_static_r10"] = e["r10"] - best
            out["greedy_minus_oracle_auc15"] = e["auc_log_k15"] - out["oracle"]["auc_log_k15"]
    if "dla" in out and "oracle" in out:
        out["dla_minus_oracle_auc"] = out["dla"]["auc_log"] - out["oracle"]["auc_log"]
        out["dla_minus_oracle_auc15"] = out["dla"]["auc_log_k15"] - out["oracle"]["auc_log_k15"]
    if "layerwise" in out and "pop" in out:
        out["layerwise_minus_pop_auc"] = out["layerwise"]["auc_log"] - out["pop"]["auc_log"]
    return out


SCALARS = [("ceiling", None), ("restored_all_moe", None), ("greedy_minus_oracle_r10", None), ("greedy_minus_best_static_r10", None),
           ("greedy_minus_oracle_auc15", None), ("dla_minus_oracle_auc", None), ("dla_minus_oracle_auc15", None), ("layerwise_minus_pop_auc", None)]
PER_ORD = ("auc_log", "auc_log_k15", "r10", "k80", "max_r", "overshoot")


def summarise_boot(b):
    """Point estimate + percentile CI of every scalar of curve_boot."""
    s = {}
    for k, _ in SCALARS:
        if k in b:
            s[k] = [float(b[k][0])] + pct(b[k][1:])
    for o in list(ORD) + ["greedy"]:
        if o not in b:
            continue
        e = b[o]
        s[o] = {f: [float(e[f][0])] + pct(e[f][1:], method="inverted_cdf" if f == "k80" else "linear") for f in PER_ORD + ("r15",) if f in e}
        s[o]["k"] = [float(x) for x in e["k"]]
        s[o]["r_curve"] = [float(x) for x in e["r"][0]]
        s[o]["r_curve_lo"] = [float(x) for x in np.percentile(e["r"][1:], 2.5, axis=0)]
        s[o]["r_curve_hi"] = [float(x) for x in np.percentile(e["r"][1:], 97.5, axis=0)]
    return s


def diff_boot(a, b):
    """a - b for matching scalars of two curve_boot outputs (rows aligned: paired if both used the same W, otherwise
    independent draws). Point = a[0] - b[0]."""
    out = {}
    for k, _ in SCALARS:
        if k in a and k in b:
            d = a[k] - b[k]
            out[k] = [float(d[0])] + pct(d[1:])
    for o in list(ORD) + ["greedy"]:
        if o in a and o in b:
            out[o] = {}
            for f in PER_ORD + ("r15",):
                if f in a[o] and f in b[o]:
                    d = a[o][f] - b[o][f]
                    if f == "k80":  # differences of grid k: report point and CI only when finite
                        d = np.where(np.isfinite(a[o][f]) & np.isfinite(b[o][f]), d, np.nan)
                    out[o][f] = [float(d[0])] + pct(d[1:], method="inverted_cdf" if f == "k80" else "linear")
    return out


def main_cfg(key):
    import ext8_addback_analyze as A8
    return A8.RUNS[key]


def load_main_task(key):
    cfg = main_cfg(key)
    if cfg["task"] == "cf":
        return X.load_cf(cfg["model"], cfg["src"])
    return X.load_wino(cfg["model"], cfg["src"], os.path.join(C.ROOT, cfg["pairs"]), os.path.join(C.ROOT, cfg["case_sets"]))


def run_complete(out):
    p = os.path.join(RESULTS, out, "run_meta.json")
    return os.path.exists(p) and json.load(open(p)).get("complete")


def part_addback(S, partial=False):
    import ext8_addback_analyze as A8
    res, boots, mats_cache = {}, {}, {}
    main_needed = set()
    for key, cfg in C.TASKS.items():
        if not run_complete(cfg["out"]) and not (partial and os.path.exists(os.path.join(RESULTS, cfg["out"], "addback_state.pkl"))):
            log(f"4a/4c: {key} not complete, skipped")
            continue
        log(f"4a/4c: analysing {key}")
        A8.load_task = lambda _cfg, _k=key: C.load_task(_k)
        acfg = dict(model=cfg["model"], task="cf" if cfg["kind"] == "cf_swap" else "wino", src=cfg["src"], out=cfg["out"], label=cfg["label"],
                    tlabel=cfg["tlabel"])
        r, _, t = A8.analyse(key, acfg)
        R = X.load_out(cfg["out"])
        m = addback_mats(t, R)
        upos, W = unit_weights(m["unit"].to_numpy(), seed=0)
        b = curve_boot(m, W, upos)
        res[key] = {"ext8_analyse": r, "boot": summarise_boot(b), "n_eval_cases": len(m["val"]), "n_units": len(upos),
                    "greedy_done": f"{m['n_greedy_done']}/{m['n_greedy']}", "mean_drop": float(m["drop"].mean())}
        # sanity: bootstrap point estimates == ext8 analyse
        if "oracle" in r["a1"]:
            res[key]["check_auc_oracle_ext8_vs_boot"] = [r["a1"]["oracle"]["auc_log"], float(b["oracle"]["auc_log"][0])]
        boots[key] = (b, m)
        main_needed.add(cfg["main_key"])
    # main runs (Phase 3, read-only) for comparison
    mains = {}
    for mk in sorted(main_needed):
        cfg = main_cfg(mk)
        log(f"4a/4c: bootstrapping the Phase-3 run {mk}")
        t = load_main_task(mk)
        R = X.load_out(cfg["out"])
        m = addback_mats(t, R)
        upos0, W0 = unit_weights(m["unit"].to_numpy(), seed=0)
        b0 = curve_boot(m, W0, upos0)
        upos1, W1 = unit_weights(m["unit"].to_numpy(), seed=1)
        mains[mk] = {"m": m, "b0": b0, "upos0": upos0, "W0": W0}
        res.setdefault("_main", {})[mk] = {"boot": summarise_boot(b0), "n_eval_cases": len(m["val"])}
    # comparisons
    comp = {}
    for key, (b, m) in boots.items():
        cfg = C.TASKS[key]
        mk = cfg["main_key"]
        if mk not in mains:
            continue
        mm = mains[mk]
        if key == "wino_mixtral_nobos":  # identical pairs: paired bootstrap with the main run's resamples
            same = sorted(m["val"]) == sorted(mm["m"]["val"])
            bp = curve_boot(m, mm["W0"], mm["upos0"]) if same else None
            comp[key] = {"vs": mk, "paired": bool(same), "diff": diff_boot(bp, mm["b0"]) if same else None}
        else:  # independent samples: other run's draws with seed 1
            upos1, W1 = unit_weights(m["unit"].to_numpy(), seed=1)
            b1 = curve_boot(m, W1, upos1)
            comp[key] = {"vs": mk, "paired": False, "diff": diff_boot(b1, mm["b0"])}
    S["addback"] = {"runs": res, "comparisons": comp}
    return res, comp, boots, mains


def num(x):
    if isinstance(x, str):
        return float("inf") if x in ("inf", "Infinity") else float(x)
    return float("nan") if x is None else float(x)


def fk(x):
    return "never" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{int(x)}"


def f3(v, d=3, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def fci(x, d=3, sign=False):
    if x is None:
        return "n/a"
    return f"{f3(x[0], d, sign)} [{f3(x[1], d, sign)}, {f3(x[2], d, sign)}]"


def fkci(x):
    if x is None:
        return "n/a"
    return f"{fk(x[0])} [{fk(x[1])}, {fk(x[2])}]"


def tables_addback(S):
    A = S.get("addback")
    if not A:
        return
    res, comp = A["runs"], A["comparisons"]
    mains = res.get("_main", {})
    order = [k for k in C.TASKS if k in res]
    rows = []
    seen_main = set()
    for key in order:
        cfg = C.TASKS[key]
        mk = cfg["main_key"]
        for which, bs, lab in (("main", mains.get(mk, {}).get("boot"), "Phase 3 (main)"), ("new", res[key]["boot"], cfg["tlabel"])):
            if bs is None or (which == "main" and mk in seen_main):
                continue
            if which == "main":
                seen_main.add(mk)
            g = bs.get("greedy", {})
            e8 = (json.load(open(os.path.join(RESULTS, "ext8_addback_summary.json")))["runs"][mk] if which == "main" else res[key]["ext8_analyse"])
            ckr = e8["a1"]["oracle"]["case_k_restored"]
            rows.append([cfg["label"], "CounterFact" if cfg["kind"] == "cf_swap" else "WinoGrande", lab,
                         (mains[mk]["n_eval_cases"] if which == "main" else res[key]["n_eval_cases"]),
                         fci(bs["ceiling"]), fci(bs["restored_all_moe"], 2),
                         f"{fkci(bs['oracle']['k80'])} / {fkci(g.get('k80'))} / {fkci(bs['rand']['k80'])}",
                         f"{fci(bs['oracle']['auc_log'])} / {fci(bs['dla']['auc_log'])} / {fci(bs['pop']['auc_log'])} / {fci(bs['layerwise']['auc_log'])}",
                         f"{fci(g.get('auc_log_k15'))} / {fci(bs['oracle']['auc_log_k15'])} / {fci(bs['dla']['auc_log_k15'])}",
                         fci(bs.get("greedy_minus_oracle_r10"), sign=True), fci(bs.get("dla_minus_oracle_auc"), sign=True),
                         fci(bs["oracle"]["overshoot"], sign=True), f"{fk(num(ckr['median']))}; {num(ckr['frac_finite']):.2f}"])
    md_table(["Model", "Task", "Run", "Eval. cases", "All-MoE ceiling M", "Answer restored by all MoE", "k80 of ceiling oracle / greedy / random",
              "AUC log k oracle / DLA / pop / layer-wise", "AUC k ≤ 15 greedy / oracle / DLA", "greedy − oracle r(10)", "DLA − oracle AUC",
              "oracle overshoot (max r − M)", "case k answer restored (oracle; median; frac. of cases)"], rows, "ext12_4a_addback",
             "4a / 4c add-back: Phase-3 run vs replication split / fold swap / no BOS (fraction of the drop; unit bootstrap 95% CI)")
    rows = []
    for key in order:
        c = comp.get(key)
        if not c or not c.get("diff"):
            continue
        d = c["diff"]
        cfg = C.TASKS[key]
        rows.append([cfg["label"], cfg["tlabel"], "paired (identical pairs)" if c["paired"] else "independent samples",
                     fci(d.get("ceiling"), sign=True), fci(d.get("restored_all_moe"), 2, sign=True),
                     fci(d["oracle"].get("auc_log"), sign=True), fci(d.get("greedy", {}).get("auc_log_k15"), sign=True),
                     fci(d["dla"].get("auc_log"), sign=True), fci(d.get("greedy_minus_oracle_r10"), sign=True),
                     fci(d.get("dla_minus_oracle_auc"), sign=True), fci(d["oracle"].get("overshoot"), sign=True),
                     fci(d["oracle"].get("k80"), 0, sign=True), fci(d.get("greedy", {}).get("k80"), 0, sign=True)])
    md_table(["Model", "Comparison (new − Phase 3)", "Bootstrap", "Δ ceiling", "Δ restored", "Δ AUC oracle", "Δ AUC k≤15 greedy", "Δ AUC DLA",
              "Δ (greedy − oracle r(10))", "Δ (DLA − oracle AUC)", "Δ overshoot", "Δ k80 oracle", "Δ k80 greedy"], rows, "ext12_4a_differences",
             "4a / 4c: differences new run − Phase-3 run (95% bootstrap CI; k80 differences only where both are finite)")
    # full ext8-style ordering table for the new runs
    rows = []
    for key in order:
        cfg = C.TASKS[key]
        bs = res[key]["boot"]
        for o in list(ORD) + ["greedy"]:
            if o not in bs:
                continue
            e = bs[o]
            rows.append([cfg["label"], cfg["tlabel"], o, f3(e["r_curve"][0]), fci(e.get("r10")), fci(e.get("auc_log")) if "auc_log" in e else "–",
                         fci(e.get("auc_log_k15")), fkci(e.get("k80")), fci(e.get("max_r")) if "max_r" in e else "–"])
    md_table(["Model", "Run", "Ordering", "r(1)", "r(10)", "AUC log k (1..K)", "AUC log k (1..15)", "k80 of ceiling", "max r"], rows,
             "ext12_4a_orderings", "4a / 4c add-back orderings (validation / evaluation cases; fraction of the drop; 95% CI)")


def fig_addback(S, boots, mains):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = S.get("addback")
    if not A:
        return
    keys = [k for k in C.TASKS if k in A["runs"]]
    if not keys:
        return
    col = {"oracle": "#1f77b4", "dla": "#9467bd", "pop": "#ff7f0e", "layerwise": "#2ca02c", "rand": "#7f7f7f", "greedy": "#d62728"}
    nc = min(3, len(keys))
    nr = (len(keys) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(4.8 * nc, 3.9 * nr), squeeze=False)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    for ax, key in zip(axes.ravel(), keys):
        cfg = C.TASKS[key]
        new = A["runs"][key]["boot"]
        old = A["runs"].get("_main", {}).get(cfg["main_key"], {}).get("boot")
        for o in ("oracle", "dla", "pop", "layerwise", "rand", "greedy"):
            if o in new:
                ax.plot(new[o]["k"], new[o]["r_curve"], color=col[o], lw=1.6, label=f"{o} (new)")
                ax.fill_between(new[o]["k"], new[o]["r_curve_lo"], new[o]["r_curve_hi"], color=col[o], alpha=0.12)
            if old and o in old:
                ax.plot(old[o]["k"], old[o]["r_curve"], color=col[o], lw=1.0, ls="--")
        ax.axhline(new["ceiling"][0], color="k", lw=0.8, label=f"all MoE (new) {new['ceiling'][0]:.2f}")
        if old:
            ax.axhline(old["ceiling"][0], color="k", lw=0.8, ls="--", label=f"all MoE (Phase 3) {old['ceiling'][0]:.2f}")
        ax.set_xscale("log")
        ax.set_xlabel("k experts patched back")
        ax.set_ylabel("r(k) = rescue / drop")
        ax.set_title(f"{cfg['label']}\n{cfg['tlabel']} (solid) vs Phase 3 (dashed)", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext12_4a_curves.{ext}"), dpi=150)
    plt.close(fig)


# =================================================================================================================
# 4b role swap vs option swap on identical items
# =================================================================================================================
ROLE = {"qwen3": dict(role="wino_role_qwen3_str", opt="wino_roleitems_qwen3_str", label="Qwen3-30B-A3B-Base"),
        "mixtral_bos": dict(role="wino_role_mixtral_bos_str", opt="wino_roleitems_mixtral_bos_str", label="Mixtral-8x7B, BOS")}


def twin_mats(run: P.PairRun, case_twin: dict, ids: list[int], extra: dict):
    """Twin-level (mean over the twin's directed cases) matrices / vectors."""
    tw = pd.Series({c: case_twin[c] for c in ids})
    out = {"attn": run.mat("attn_layer", ids).groupby(tw).mean(), "moe": run.mat("layer", ids).groupby(tw).mean(),
           "block": run.mat("block", ids).groupby(tw).mean(), "drop": run.drop.loc[ids].groupby(tw).mean()}
    for k, s in extra.items():
        out[k] = s.loc[ids].groupby(tw).mean()
    out["n_cases"] = tw.groupby(tw).size()
    return out


def share_from(Wt, A, M):
    """AUC+ share for weight rows Wt [B, n] over twin matrices A, M [n, L] (weighted means of the layer curves)."""
    s = Wt.sum(1, keepdims=True)
    a = np.clip((Wt @ A) / s, 0, None).sum(1)
    m = np.clip((Wt @ M) / s, 0, None).sum(1)
    return a / (a + m), a, m


def role_option_compare(proto, splits, matched=False):
    cfg = ROLE[proto]
    lk = pd.read_parquet(os.path.join(RESULTS, "wino_roleitems_inputs", f"link_{proto}.parquet"))
    cs = json.load(open(os.path.join(RESULTS, "wino_roleitems_inputs", f"case_sets_{proto}.json")))
    inv = {v: k for k, v in cs["pair_idx_of"].items()}
    rr, orun = P.PairRun(cfg["role"]), P.PairRun(cfg["opt"])
    lk = lk[lk.split.isin(splits)]
    if matched:
        lk = lk[lk.d == 0]
        role_ids = lk.role_case_id.astype(int).tolist()
        opt_ids = lk.opt_case_same_clean.astype(int).tolist()
    else:
        role_ids = lk.role_case_id.astype(int).tolist()
        opt_ids = [c for s in splits for c in cs["directed"].get(s, [])]
    role_twin = dict(zip(lk.role_case_id.astype(int), lk.twin_id))
    opt_twin = {c: inv[c // 2] for c in opt_ids}
    twins = sorted(set(role_twin[c] for c in role_ids))
    assert twins == sorted(set(opt_twin[c] for c in opt_ids)), "role / option twin sets differ"

    def extras(run_name, ids):
        ex = {}
        dp = os.path.join(RESULTS, run_name, "direct_split.parquet")
        if os.path.exists(dp):
            ds = pd.read_parquet(dp).set_index("case_id")
            ds = ds.loc[ids]
            ex["a_dir"] = ds.d_attn_direct - ds.d_corrupt_fp32
            ex["m_dir"] = ds.d_moe_direct - ds.d_corrupt_fp32
            ex["drop32"] = ds.d_clean_fp32 - ds.d_corrupt_fp32
        jp = os.path.join(RESULTS, run_name, "joint_rows.parquet")
        if os.path.exists(jp):
            j = pd.read_parquet(jp)
            j = j[(j.span == "all") & (j.direction == "denoise") & j.case_id.isin(ids)]
            ex["M_all_moe"] = j[j.kind == "all_moe"].set_index("case_id").effect.reindex(ids)
            ex["A_all_attn"] = j[j.kind == "all_attn"].set_index("case_id").effect.reindex(ids)
            ex["drop_joint"] = j[j.kind == "all_moe"].set_index("case_id")["drop"].reindex(ids)
        return ex

    Tr = twin_mats(rr, role_twin, role_ids, extras(cfg["role"], role_ids))
    To = twin_mats(orun, opt_twin, opt_ids, extras(cfg["opt"], opt_ids))
    for T in (Tr, To):
        for k, v in T.items():
            T[k] = v.reindex(twins)
    n = len(twins)
    idx = C.boot_idx(n, NB, 0)
    Wt = np.zeros((NB + 1, n))
    Wt[0] = 1.0
    np.add.at(Wt, (np.repeat(np.arange(1, NB + 1), n), idx.ravel()), 1.0)
    out = {"n_twins": n, "n_role_cases": len(role_ids), "n_opt_cases": len(opt_ids), "splits": list(splits), "matched": matched}
    sr, ar, mr = share_from(Wt, Tr["attn"].to_numpy(), Tr["moe"].to_numpy())
    so, ao, mo = share_from(Wt, To["attn"].to_numpy(), To["moe"].to_numpy())
    dr_r = Wt @ Tr["drop"].to_numpy() / Wt.sum(1)
    dr_o = Wt @ To["drop"].to_numpy() / Wt.sum(1)

    def pc(x):
        return [float(x[0])] + pct(x[1:])
    out["share_role"], out["share_option"], out["share_diff"] = pc(sr), pc(so), pc(sr - so)
    out["auc_attn_norm_role"], out["auc_attn_norm_option"] = pc(ar / dr_r), pc(ao / dr_o)
    out["auc_moe_norm_role"], out["auc_moe_norm_option"] = pc(mr / dr_r), pc(mo / dr_o)
    out["auc_attn_norm_diff"], out["auc_moe_norm_diff"] = pc(ar / dr_r - ao / dr_o), pc(mr / dr_r - mo / dr_o)
    out["drop_role"], out["drop_option"] = pc(dr_r), pc(dr_o)
    for num, den, name in (("a_dir", "drop32", "A_direct"), ("m_dir", "drop32", "M_direct"), ("M_all_moe", "drop_joint", "M_all_moe"),
                           ("A_all_attn", "drop_joint", "A_all_attn_sanity")):
        if num in Tr and num in To:
            vr = (Wt @ Tr[num].to_numpy()) / (Wt @ Tr[den].to_numpy())
            vo = (Wt @ To[num].to_numpy()) / (Wt @ To[den].to_numpy())
            out[f"{name}_role"], out[f"{name}_option"], out[f"{name}_diff"] = pc(vr), pc(vo), pc(vr - vo)
    # peaks of the twin-weighted normalised mean curves (descriptive, no selection) and the role-swap attention peak layer
    for tag, T in (("role", Tr), ("option", To)):
        d = float(T["drop"].mean())
        for k in ("attn", "moe", "block"):
            c = T[k].mean(axis=0) / d
            out[f"peak_{k}_{tag}"] = [int(c.idxmax()), float(c.max())]
        out[f"curve_attn_{tag}"] = [float(x) for x in (T["attn"].mean(axis=0) / d)]
        out[f"curve_moe_{tag}"] = [float(x) for x in (T["moe"].mean(axis=0) / d)]
    # Phase-3 pair-weighted role share on exactly these role pairs (the Phase-3 definition) for reference
    rid = [c for c in role_ids]
    out["share_role_pairweighted"] = P.attention_share(rr, rid)
    out["share_option_pairweighted"] = P.attention_share(orun, opt_ids)
    return out


def renaming_check(proto):
    """Structural check: the role-swapped corrupted prompt of role pair <twin>|X equals the option-swapped corrupted prompt
    of the same clean prompt with the two names exchanged everywhere (text level), and per-case agreement of the two
    corruptions on the clean-prompt-matched directed cases (Pearson r over case x layer of the rescue, r of the drops)."""
    import re
    lk = pd.read_parquet(os.path.join(RESULTS, "wino_roleitems_inputs", f"link_{proto}.parquet"))
    lk = lk[lk.d == 0]
    rp = pd.read_parquet(os.path.join(C.ROOT, f"data/wino_role/pairs_{proto}.parquet")).set_index("pair_id")
    op = pd.read_parquet(os.path.join(C.ROOT, f"data/wino_str/pairs_train_xl_{proto}.parquet")).set_index("pair_id")
    ok = 0
    for r in lk.itertuples():
        a, o = rp.loc[r.role_pair_id], op.loc[r.twin_id]
        n1, n2 = o.ans_a, o.ans_b
        t = re.sub(rf"\b({re.escape(n1)}|{re.escape(n2)})\b", lambda m: n2 if m.group(0) == n1 else n1, a.prompt_b)
        ok += t == (o.prompt_b if r.role == "A" else o.prompt_a)
    cfg = ROLE[proto]
    rr, oo = P.PairRun(cfg["role"]), P.PairRun(cfg["opt"])
    ri, oi = lk.role_case_id.astype(int).tolist(), lk.opt_case_same_clean.astype(int).tolist()
    out = {"n_matched_cases": len(lk), "renamed_role_corrupt_equals_option_corrupt": int(ok)}
    for k in ("layer", "attn_layer", "block"):
        A_, B_ = rr.mat(k, ri).to_numpy(), oo.mat(k, oi).to_numpy()
        out[f"r_case_layer_{k}"] = float(np.corrcoef(A_.ravel(), B_.ravel())[0, 1])
        out[f"mean_abs_diff_{k}"] = float(np.abs(A_ - B_).mean())
        out[f"mean_abs_{k}"] = float(np.abs(A_).mean())
    dr, do = rr.drop.loc[ri].to_numpy(), oo.drop.loc[oi].to_numpy()
    out["r_drop"] = float(np.corrcoef(dr, do)[0, 1])
    return out


def part_role(S):
    out = {}
    for proto, cfg in ROLE.items():
        if not os.path.exists(os.path.join(RESULTS, cfg["opt"], "str_sweep_rows.parquet")):
            log(f"4b: {cfg['opt']} missing, skipped")
            continue
        cs = json.load(open(os.path.join(RESULTS, "wino_roleitems_inputs", f"case_sets_{proto}.json")))
        all_splits = [s for s in ("discovery", "validation", "rep_discovery", "rep_validation") if s in cs["directed"]]
        e = {"all": role_option_compare(proto, all_splits), "validation": role_option_compare(proto, ["validation"]),
             "all_matched": role_option_compare(proto, all_splits, matched=True)}
        e["renaming"] = renaming_check(proto)
        # Phase-3 option-swap reference (776-pair main validation, ext7-wino) for context
        e["phase3_option_main_validation_share"] = json.load(open(os.path.join(RESULTS, "ext7_wino_summary.json"))).get(
            "wino_qwen3_str" if proto == "qwen3" else "wino_mixtral_bos_str", {}).get("w2", {}).get("share", {}).get("main")
        out[proto] = e
        log(f"4b {proto}: share role {e['all']['share_role'][0]:.3f} vs option {e['all']['share_option'][0]:.3f} "
            f"(diff {e['all']['share_diff'][0]:+.3f} [{e['all']['share_diff'][1]:+.3f}, {e['all']['share_diff'][2]:+.3f}]), twins {e['all']['n_twins']}")
    S["role_vs_option"] = out
    return out


def tables_role(S):
    R = S.get("role_vs_option")
    if not R:
        return
    rows = []
    for proto, e in R.items():
        for sub, lab in (("all", "all role items (4 splits)" if proto == "qwen3" else "all role items (2 splits)"), ("validation", "validation split"),
                         ("all_matched", "all, clean-prompt matched (role d = 0)")):
            x = e[sub]
            rows.append([ROLE[proto]["label"], lab, x["n_twins"], f"{x['n_role_cases']} / {x['n_opt_cases']}",
                         f"{fci(x['drop_role'], 2)} / {fci(x['drop_option'], 2)}",
                         fci(x["share_role"]), fci(x["share_option"]), fci(x["share_diff"], sign=True),
                         f"{fci(x.get('A_direct_role'))} / {fci(x.get('A_direct_option'))}", fci(x.get("A_direct_diff"), sign=True),
                         f"{fci(x.get('M_all_moe_role'))} / {fci(x.get('M_all_moe_option'))}", fci(x.get("M_all_moe_diff"), sign=True)])
    md_table(["Model", "Items", "Twins", "Directed cases role / option", "Drop role / option", "Attention share role", "Attention share option",
              "Δ share (role − option)", "Direct-path attention A_dir role / option", "Δ A_dir", "All-MoE M role / option", "Δ M"], rows,
             "ext12_4b_role_vs_option",
             "4b: role swap vs option swap on identical WinoGrande items (twin-level, twin bootstrap 95% CI; attention share = AUC+(attention) / "
             "(AUC+(attention) + AUC+(MoE)) of the twin-weighted final-position single-layer curves)")
    rows = []
    for proto, e in R.items():
        x = e["all"]
        rows.append([ROLE[proto]["label"], f"L{x['peak_attn_role'][0]} {x['peak_attn_role'][1]:.3f} / L{x['peak_attn_option'][0]} {x['peak_attn_option'][1]:.3f}",
                     f"L{x['peak_moe_role'][0]} {x['peak_moe_role'][1]:.3f} / L{x['peak_moe_option'][0]} {x['peak_moe_option'][1]:.3f}",
                     f"L{x['peak_block_role'][0]} {x['peak_block_role'][1]:.3f} / L{x['peak_block_option'][0]} {x['peak_block_option'][1]:.3f}",
                     f"{fci(x['auc_attn_norm_role'])} / {fci(x['auc_attn_norm_option'])}", fci(x["auc_attn_norm_diff"], sign=True),
                     f"{fci(x['auc_moe_norm_role'])} / {fci(x['auc_moe_norm_option'])}", fci(x["auc_moe_norm_diff"], sign=True),
                     f"{x['share_role_pairweighted']['share']:.3f} / {x['share_option_pairweighted']['share']:.3f}"])
    md_table(["Model", "Attention peak role / option (L, / drop)", "MoE peak role / option", "Block peak role / option",
              "AUC+ attention / drop role / option", "Δ", "AUC+ MoE / drop role / option", "Δ", "Share, pair-weighted (Phase-3 definition) role / option"],
             rows, "ext12_4b_peaks", "4b: final-position single-layer curves on identical items (all role items; descriptive peaks of the twin-weighted means)")


def fig_role(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = S.get("role_vs_option")
    if not R:
        return
    fig, axes = plt.subplots(1, len(R), figsize=(5.5 * len(R), 3.8), squeeze=False)
    for ax, (proto, e) in zip(axes[0], R.items()):
        x = e["all"]
        L = len(x["curve_attn_role"])
        for k, c in (("attn", "#d62728"), ("moe", "#1f77b4")):
            ax.plot(range(L), x[f"curve_{k}_role"], color=c, lw=1.6, label=f"{'attention' if k == 'attn' else 'MoE'}, role swap")
            ax.plot(range(L), x[f"curve_{k}_option"], color=c, lw=1.2, ls="--", label=f"{'attention' if k == 'attn' else 'MoE'}, option swap")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("layer (final-position patch)")
        ax.set_ylabel("rescue / drop")
        ax.set_title(f"{ROLE[proto]['label']}: {x['n_twins']} identical twins\nshare role {x['share_role'][0]:.2f} vs option {x['share_option'][0]:.2f}",
                     fontsize=8)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext12_4b_role_vs_option.{ext}"), dpi=150)
    plt.close(fig)


# =================================================================================================================
# 4c Mixtral without BOS on WinoGrande
# =================================================================================================================
NOBOS_CFG = {"run": "wino_mixtral_nobos_str", "model": "mixtral", "label": "Mixtral-8x7B, no BOS", "scan": "wino_mixtral_nobos",
             "cf": "mixtral_nobos_str", "cf_experts": [(20, 0), (19, 6), (21, 6), (19, 2), (18, 1), (21, 1)]}
BOS_RUN = "wino_mixtral_bos_str"


def paired_pair_stats(nb: P.PairRun, bs: P.PairRun, ids: list[int]):
    """Paired (identical pairs) no-BOS - BOS differences: attention share, AUC+ / drop, peaks."""
    out = {}
    pairs = sorted({c // 2 for c in ids})
    n = len(pairs)
    idx = C.boot_idx(n, NB, 0)
    Wt = np.zeros((NB + 1, n))
    Wt[0] = 1.0
    np.add.at(Wt, (np.repeat(np.arange(1, NB + 1), n), idx.ravel()), 1.0)
    res = {}
    for tag, run in (("nobos", nb), ("bos", bs)):
        A = P.matrix_pairs(run.mat("attn_layer", ids)).reindex(pairs).to_numpy()
        M = P.matrix_pairs(run.mat("layer", ids)).reindex(pairs).to_numpy()
        d = P.pair_means(run.drop.loc[ids]).reindex(pairs).to_numpy()
        sh, a, m = share_from(Wt, A, M)
        dd = Wt @ d / Wt.sum(1)
        res[tag] = (sh, a / dd, m / dd, dd)
    for i, name in enumerate(("share", "auc_attn_norm", "auc_moe_norm", "drop")):
        a, b = res["nobos"][i], res["bos"][i]
        out[name] = {"nobos": [float(a[0])] + pct(a[1:]), "bos": [float(b[0])] + pct(b[1:]), "diff": [float(a[0] - b[0])] + pct((a - b)[1:])}
    return out, Wt, pairs


def paired_ratio(nb_num, nb_den, bs_num, bs_den, Wt, pairs):
    def pm(s):
        return P.pair_means(s).reindex(pairs).to_numpy()
    a = (Wt @ pm(nb_num)) / (Wt @ pm(nb_den))
    b = (Wt @ pm(bs_num)) / (Wt @ pm(bs_den))
    return {"nobos": [float(a[0])] + pct(a[1:]), "bos": [float(b[0])] + pct(b[1:]), "diff": [float(a[0] - b[0])] + pct((a - b)[1:])}


def part_nobos(S):
    if not os.path.exists(os.path.join(RESULTS, NOBOS_CFG["run"], "str_sweep_rows.parquet")):
        log("4c: no sweep yet, skipped")
        return None
    import ext7_wino_analyze as W
    run = W.load_run(NOBOS_CFG)
    out = {"label": NOBOS_CFG["label"], "descriptors": W.descriptors(run, NOBOS_CFG)}
    w2 = W.w2(run, NOBOS_CFG)
    w2["curves"] = {k: v.to_dict("records") for k, v in w2["curves"].items()}
    w2["additivity"] = {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in w2["additivity"].items()}
    out["w2"] = w2
    meta = json.load(open(os.path.join(run.dir, "run_meta.json")))
    if meta.get("expert", {}).get("complete"):
        out["w6"] = W.w6(run, NOBOS_CFG, w2)
    if meta.get("joint", {}).get("complete") and meta.get("direct_split", {}).get("completed_utc"):
        out["w4"] = W.joint(run, NOBOS_CFG)
    # paired no-BOS - BOS on the identical main pairs (validation, and all 256 main pairs)
    bs = P.PairRun(BOS_RUN)
    comp = {}
    for sub, ids in (("main_validation", run.ids("main", "validation")), ("main_all", run.ids("main", "discovery") + run.ids("main", "validation"))):
        st, Wt, pairs = paired_pair_stats(run, bs, ids)
        e = dict(st)
        for k in ("layer", "attn_layer", "block"):
            cn = (P.matrix_pairs(run.mat(k, ids)).mean(axis=0) / P.pair_means(run.drop.loc[ids]).mean())
            cb = (P.matrix_pairs(bs.mat(k, ids)).mean(axis=0) / P.pair_means(bs.drop.loc[ids]).mean())
            e[f"peak_{k}"] = {"nobos": [int(cn.idxmax()), float(cn.max())], "bos": [int(cb.idxmax()), float(cb.max())]}
            e[f"curve_{k}"] = {"nobos": [float(x) for x in cn], "bos": [float(x) for x in cb]}
        dn, db = os.path.join(run.dir, "direct_split.parquet"), os.path.join(bs.dir, "direct_split.parquet")
        if meta.get("direct_split", {}).get("completed_utc") and os.path.exists(db):
            xn, xb = pd.read_parquet(dn).set_index("case_id").loc[ids], pd.read_parquet(db).set_index("case_id").loc[ids]
            for name, f in (("A_direct", lambda x: x.d_attn_direct - x.d_corrupt_fp32), ("M_direct", lambda x: x.d_moe_direct - x.d_corrupt_fp32)):
                e[name] = paired_ratio(f(xn), xn.d_clean_fp32 - xn.d_corrupt_fp32, f(xb), xb.d_clean_fp32 - xb.d_corrupt_fp32, Wt, pairs)
        jn, jb = os.path.join(run.dir, "joint_rows.parquet"), os.path.join(bs.dir, "joint_rows.parquet")
        if meta.get("joint", {}).get("complete") and os.path.exists(jb):
            def jm(p):
                j = pd.read_parquet(p)
                j = j[(j.span == "all") & (j.direction == "denoise") & (j.kind == "all_moe")].set_index("case_id")
                return j.effect.loc[ids], j["drop"].loc[ids]
            (en, dn_), (eb, db_) = jm(jn), jm(jb)
            e["M_all_moe"] = paired_ratio(en, dn_, eb, db_, Wt, pairs)
        comp[sub] = e
    out["vs_bos"] = comp
    # ---- sink-carrying final tokens (Direction 3 flag: final position = max residual norm at L5)
    sp = os.path.join(run.dir, "sink_flags.parquet")
    if meta.get("sink_flags", {}).get("completed_utc"):
        sf = pd.read_parquet(sp)
        out["sink"] = sink_analysis(run, sf, out.get("w6"))
    S["nobos"] = out
    return out


def sink_analysis(run, sf, w6):
    res = {}
    for proto in ("nobos", "bos"):
        x = sf[sf.proto == proto]
        res[proto] = {"n_prompts": int(len(x)), "final_is_max_L5": int(x.final_is_max_L5.sum()), "frac": float(x.final_is_max_L5.mean()),
                      "final_is_max_any": int(x.final_is_max_any.sum()), "pos0_is_max_L5": int(x.pos0_is_max_L5.sum()),
                      "pos0_mass_L1": float(x.pos0_mass_L1.mean()), "self_mass_L1": float(x.self_mass_L1.mean())}
        fl = x[x.final_is_max_L5]
        if len(fl):
            res[proto]["final_tokens_flagged"] = fl.final_tok.value_counts().head(10).to_dict()
            try:
                from transformers import AutoTokenizer
                tk = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
                res[proto]["final_tokens_flagged_str"] = {tk.convert_ids_to_tokens([int(k)])[0]: int(v) for k, v in fl.final_tok.value_counts().head(10).items()}
            except Exception as ex:  # tokenizer not available: ids only
                res[proto]["final_tokens_flagged_str"] = {"error": repr(ex)}
        res[proto]["max_norm_position_L5_top5"] = {int(k): int(v) for k, v in x.argmax_pos_L5.value_counts().head(5).items()}
        res[proto]["max_norm_position_L5_rel_median"] = float((x.argmax_pos_L5 / (x["T"] - 1)).median())
    nb = sf[sf.proto == "nobos"].set_index(["pair_idx", "prompt"])
    bo = sf[sf.proto == "bos"].set_index(["pair_idx", "prompt"])
    # routing of the flagged (no-BOS) prompts at L19 / L20 vs the same prompts with BOS
    for l in (19, 20):
        col = f"route_L{l}"
        fl = nb[nb.final_is_max_L5]
        if len(fl):
            from collections import Counter
            cn = Counter(e for s in fl[col] for e in json.loads(s))
            cb = Counter(e for s in bo.loc[fl.index, col] for e in json.loads(s))
            res[f"routing_L{l}_flagged"] = {"nobos": dict(cn.most_common(8)), "bos_same_prompts": dict(cb.most_common(8)), "n": int(len(fl))}
            oth = nb[~nb.final_is_max_L5]
            co = Counter(e for s in oth[col] for e in json.loads(s))
            res[f"routing_L{l}_other"] = {"nobos": dict(co.most_common(8)), "n": int(len(oth))}
    # final-position routing agreement no BOS vs BOS (same prompt text), per layer: identical top-k set
    L = len([c for c in sf.columns if c.startswith("route_L")])
    common = nb.index.intersection(bo.index)
    agree = [float((nb.loc[common, f"route_L{l}"].values == bo.loc[common, f"route_L{l}"].values).mean()) for l in range(L)]
    res["routing_set_agreement_by_layer"] = agree
    res["routing_set_agreement_mean"] = float(np.mean(agree))
    for l in (19, 20):
        fl_i = nb.index[nb.final_is_max_L5]
        ot_i = nb.index[~nb.final_is_max_L5]
        res[f"routing_agreement_L{l}"] = {"all": agree[l],
                                          "flagged": float((nb.loc[fl_i, f"route_L{l}"].values == bo.loc[fl_i, f"route_L{l}"].values).mean()) if len(fl_i) else None,
                                          "other": float((nb.loc[ot_i, f"route_L{l}"].values == bo.loc[ot_i, f"route_L{l}"].values).mean()) if len(ot_i) else None}
    # per directed case: clean prompt flagged? (d = 0: clean = a; d = 1: clean = b)
    cases = run.cases
    flag = {}
    for c in cases.index:
        pi, d = divmod(int(c), 2)
        cl, co = ("a", "b") if d == 0 else ("b", "a")
        if (pi, cl) in nb.index:
            flag[int(c)] = (bool(nb.loc[(pi, cl), "final_is_max_L5"]), bool(nb.loc[(pi, co), "final_is_max_L5"]))
    fl_df = pd.DataFrame([(c, a, b) for c, (a, b) in flag.items()], columns=["case_id", "clean_sink", "corrupt_sink"]).set_index("case_id")
    res["directed_cases"] = {"n": int(len(fl_df)), "clean_sink": int(fl_df.clean_sink.sum()), "corrupt_sink": int(fl_df.corrupt_sink.sum()),
                             "either": int((fl_df.clean_sink | fl_df.corrupt_sink).sum())}
    # strata: W2 MoE / attention at the main peaks and the selected expert, flagged (clean or corrupted prompt) vs other
    ids_all = run.ids("main", "discovery") + run.ids("main", "validation")
    pk = {}
    try:
        import ext7_wino_analyze as W
        pk = {p["kind"]: p["L_disc"] for p in W.P.sweep_peaks(run, "main", ["layer", "attn_layer", "block"])}
    except Exception:
        pass
    sel = None
    if w6 and w6.get("main", {}).get("two_stage", {}).get("eval"):
        ev = w6["main"]["two_stage"]["eval"]
        sel = (w6["main"]["two_stage"]["layer"], ev["expert"])
    st = []
    md = P.model_data(run.run) if sel else None
    for lab, msk in (("clean or corrupted prompt sink-carrying", lambda c: flag.get(c, (False, False))[0] or flag.get(c, (False, False))[1]),
                     ("neither prompt sink-carrying", lambda c: not (flag.get(c, (False, False))[0] or flag.get(c, (False, False))[1]))):
        ids = [c for c in ids_all if msk(c)]
        row = {"stratum": lab, "n_directed": len(ids), "n_pairs": len({c // 2 for c in ids})}
        if len(ids) >= 10:
            row["drop"] = float(run.drop.loc[ids].mean())
            for k, l in pk.items():
                row[f"norm_{k}_L{l}"] = P.ratio(run.mat(k, ids)[l], run.drop.loc[ids])
            if len({c // 2 for c in ids}) >= 5:
                row["share"] = P.attention_share(run, ids)
            if sel:
                from moetrace import analysis as A
                ev = A.evaluate_expert(md, sel[0], sel[1], ids, 1)
                pc = ev["per_case"].set_index("case_id")
                row["sel_expert"] = f"L{sel[0]}E{sel[1]:03d}"
                row["sel_active"] = int(pc.active.sum())
                row["sel_rescue"] = P.summ(pc.rescue, with_p=False)
                row["sel_spec"] = P.summ(pc.spec.dropna(), with_p=False) if pc.spec.notna().sum() >= 4 else None
        st.append(row)
    res["strata"] = st
    return res


def tables_nobos(S):
    N = S.get("nobos")
    if not N:
        return
    ext7 = json.load(open(os.path.join(RESULTS, "ext7_wino_summary.json"))).get(BOS_RUN, {})
    rows = []
    for tag, s in (("no BOS (ext12)", N), ("BOS (Phase 3)", ext7)):
        w2 = s.get("w2", {})
        pks = {p["kind"]: p for p in w2.get("peaks", {}).get("main", [])}
        sh = w2.get("share", {}).get("main", {})
        rep = w2.get("share", {}).get("rep", {})
        cells = [f"Mixtral-8x7B, {tag}"]
        for k in ("layer", "attn_layer", "block"):
            p = pks.get(k)
            cells.append(f"L{p['L_disc']}: {p['norm_at_L_disc'][0]:.3f} [{p['norm_at_L_disc'][1]:.3f}, {p['norm_at_L_disc'][2]:.3f}]" if p else "n/a")
        cells.append(f"{sh.get('share', float('nan')):.3f} [{sh.get('share_lo', float('nan')):.3f}, {sh.get('share_hi', float('nan')):.3f}]" if sh else "n/a")
        cells.append(f"{rep.get('share', float('nan')):.3f}" if rep else "n/a")
        d = s.get("descriptors", {}).get("main", {})
        cells.append(f"{d.get('drop', {}).get('mean', float('nan')):+.2f}" if d else "n/a")
        rows.append(cells)
    md_table(["Run", "MoE peak (disc. L, val. / drop)", "Attention peak", "Block peak", "Attention share (main val.)", "Share (rep val.)", "Mean drop (main)"],
             rows, "ext12_4c_w2", "4c W2: final-position single-layer patches, Mixtral without vs with BOS on the identical 776-pair case set "
                                  "(main 128/128 pairs; pair bootstrap 95% CI)")
    rows = []
    for sub in ("main_validation", "main_all"):
        e = N.get("vs_bos", {}).get(sub)
        if not e:
            continue
        for name, lab in (("share", "attention share"), ("auc_attn_norm", "AUC+ attention / drop"), ("auc_moe_norm", "AUC+ MoE / drop"),
                          ("A_direct", "direct-path attention A_dir"), ("M_direct", "direct-path MoE M_dir"), ("M_all_moe", "all-MoE M (denoise)"),
                          ("drop", "mean drop (logits)")):
            if name in e:
                x = e[name]
                rows.append([sub.replace("_", " "), lab, fci(x["nobos"]), fci(x["bos"]), fci(x["diff"], sign=True)])
    md_table(["Pairs", "Quantity", "no BOS", "BOS", "no BOS − BOS (paired)"], rows, "ext12_4c_vs_bos",
             "4c: Mixtral no BOS vs BOS on identical pairs (paired pair bootstrap, 95% CI)")
    rows = []
    for tag, w6 in (("no BOS", N.get("w6") or {}), ("BOS (Phase 3)", ext7.get("w6") or {})):
        for f in ("main", "rep"):
            ent = w6.get(f)
            if not ent:
                continue
            for rule in ("two_stage", "interior"):
                r = ent.get(rule)
                if not r or (rule == "interior" and ent.get("two_stage") and r["layer"] == ent["two_stage"]["layer"]):
                    continue
                ev = r.get("eval")
                eq = r.get("equal_norm", {}) or {}
                es = r["selection"].get("e_star")
                rows.append([tag, f, rule.replace("_", "-"), f"L{r['layer']}", f"L{r['layer']}E{int(es):03d}" if es is not None else "none",
                             r.get("pattern"), fci([ev["rescue"]["mean"], ev["rescue"]["ci_lo"], ev["rescue"]["ci_hi"]], sign=True) if ev else "n/a",
                             fci([ev["spec"]["mean"], ev["spec"]["ci_lo"], ev["spec"]["ci_hi"]], sign=True) if ev and ev["spec"].get("n") else "n/a",
                             fci([eq["spec_eq"]["mean"], eq["spec_eq"]["ci_lo"], eq["spec_eq"]["ci_hi"]], sign=True) if eq.get("spec_eq") else "n/a",
                             f"{ev['val_active']}" if ev else "n/a",
                             ", ".join(f"L{j['layer']}E{j['expert']:03d} ({j['disc_active']})" for j in ent.get("joint_top", [])[:3])])
    md_table(["Protocol", "Family", "Rule", "Layer", "Selected expert", "Pattern", "Val. rescue", "Spec", "Equal-norm Spec", "Val. active (of 256)",
              "Joint top-3 (disc. active)"], rows, "ext12_4c_w6",
             "4c W6: two-stage expert selection, Mixtral without vs with BOS (recurrence gate 128 of 256 discovery directed cases; pair bootstrap 95% CI; "
             "the interior-layer rule (≤ L−5) selects the same layer in every row and is not repeated)")
    w6 = N.get("w6") or {}
    fx = w6.get("main", {}).get("cf_experts", [])
    if fx:
        rows = [[f"L{x['layer']}E{x['expert']:03d}", x["disc_active"], x["val_active"], fci([x["rescue"]["mean"], x["rescue"]["ci_lo"], x["rescue"]["ci_hi"]], sign=True),
                 fci([x["spec"]["mean"], x["spec"]["ci_lo"], x["spec"]["ci_hi"]], sign=True) if x["spec"].get("n") else "n/a"] for x in fx]
        md_table(["Expert (fixed hypothesis)", "Disc. active (of 256)", "Val. active", "Val. rescue", "Spec"], rows, "ext12_4c_fixed_experts",
                 "4c: fixed experts evaluated without BOS (WinoGrande BOS selection L20E000, the Direction-3 sink expert L19E006, CounterFact experts; "
                 "main validation; Spec of an expert that is rarely clean-active is negative by construction, its rescue being 0 where it is not routed)")
    sk = N.get("sink")
    if sk:
        rows = [[p, sk[p]["n_prompts"], f"{sk[p]['final_is_max_L5']} ({sk[p]['frac']:.3f})", sk[p]["final_is_max_any"], sk[p]["pos0_is_max_L5"],
                 f"{sk[p]['pos0_mass_L1']:.2f}"] for p in ("nobos", "bos")]
        md_table(["Protocol", "Prompts", "Final position = max norm at L5", "… at any early layer", "Position 0 = max norm at L5", "Mass on position 0 at L1"],
                 rows, "ext12_4c_sink", "4c: sink-carrying final tokens (Direction-3 flag) on the WinoGrande prompts (main + rep pairs, both prompts)")


def fig_nobos(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = S.get("nobos")
    if not N or "vs_bos" not in N:
        return
    e = N["vs_bos"]["main_validation"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, c, lab in (("layer", "#1f77b4", "MoE"), ("attn_layer", "#d62728", "attention"), ("block", "#2ca02c", "block")):
        ax.plot(e[f"curve_{k}"]["nobos"], color=c, lw=1.6, label=f"{lab}, no BOS")
        ax.plot(e[f"curve_{k}"]["bos"], color=c, lw=1.0, ls="--", label=f"{lab}, BOS")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("layer (final-position patch)")
    ax.set_ylabel("rescue / drop (main validation)")
    ax.set_title(f"Mixtral WinoGrande STR, identical pairs: attention share no BOS {e['share']['nobos'][0]:.2f} vs BOS {e['share']['bos'][0]:.2f}", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext12_4c_w2_nobos_vs_bos.{ext}"), dpi=150)
    plt.close(fig)


# =================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="4a,4b,4c")
    ap.add_argument("--partial", action="store_true", help="also analyse add-back runs that are not complete")
    ap.add_argument("--outdir", default=None, help="write tables / figures / summary here instead of results/ (testing)")
    args = ap.parse_args()
    global TAB, FIG, SUMM
    if args.outdir:
        TAB = FIG = args.outdir
        SUMM = os.path.join(args.outdir, "ext12_complete_summary.json")
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    S = json.load(open(SUMM)) if os.path.exists(SUMM) else {}
    parts = args.parts.split(",")
    if "4a" in parts or "4c" in parts:
        res, comp, boots, mains = part_addback(S, args.partial)
        tables_addback(S)
        fig_addback(S, boots, mains)
    if "4b" in parts:
        part_role(S)
        tables_role(S)
        fig_role(S)
    if "4c" in parts:
        part_nobos(S)
        tables_nobos(S)
        fig_nobos(S)
    S["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(SUMM, "w") as f:
        json.dump(jsafe(S), f, indent=1, default=str)
    log(f"wrote {SUMM}")


if __name__ == "__main__":
    main()
