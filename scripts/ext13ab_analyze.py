"""ext13ab (Phase 5) analysis, CPU only: add-back / deletion curves of every Phase-5 condition.

Per task (moetrace/ext13ab.registry; runs results/p5_<cond>_<proto>[_rep]_ab):
  * ext8's own `analyse` (scripts/ext8_addback_analyze.py, imported; only its task loader is replaced): A0 ceilings and
    the attention / MoE decomposition, A1 / A3 curve summaries, greedy;
  * unit-cluster bootstrap (5,000 resamples, seed 0; units = pairs) of the add-back curve summaries with ext12's code
    (scripts/ext12_analyze.py: ceiling, answer restored, AUC over log k full and k <= 15, k80, max r, overshoot =
    max r - ceiling, greedy - static, DLA - oracle) and the same for the deletion curves (noise_oracle, oracle, pop,
    layerwise, dla, rand; k80 of the deletion ceiling on the curve's own cases);
  * seed-clustered bootstrap for template conditions (units = seed templates; many pairs share a template);
  * names / objects strata (>= 20 pairs per stratum);
  * the direct-path split of p5-data's run (results/p5_<cond>_<proto>/direct_split.parquet; validation cases of the
    family; fallback: this run's addback_direct.parquet) for the attention-vs-MoE bars.
Comparisons: Phase 3 (WinoGrande option swap, CounterFact; ext8 runs, read-only) and the copy / direction contrasts on
identical items (paired bootstrap over the common units when >= 20 are shared, otherwise independent samples).

Usage: python scripts/ext13ab_analyze.py [--tasks key,key] [--partial] [--smoke] [--outdir DIR]
Outputs results/ext13ab_summary.json, results/tables/ext13ab_*.{md,csv} (figures: scripts/ext13ab_plot.py)
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, pandas as pd
from moetrace import ext8_addback as X
from moetrace import ext13ab as AB
from moetrace import ext7_pairs as P
from moetrace.models import RESULTS
import ext12_analyze as E12
import ext8_addback_analyze as A8

TAB = os.path.join(RESULTS, "tables")
SUMM = os.path.join(RESULTS, "ext13ab_summary.json")
NB = E12.NB
DEL_ORD = ("noise_oracle", "oracle", "pop", "layerwise", "dla", "rand")
MIN_UNITS = 20
P3 = {"wino_qwen3": ("qwen3", "WinoGrande option swap (Phase 3)"), "wino_mixtral": ("mixtral", "WinoGrande option swap (Phase 3)"),
      "cf_qwen3": ("qwen3", "CounterFact (Phase 3)"), "cf_mixtral": ("mixtral", "CounterFact (Phase 3)")}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------------------------
# matrices
# ------------------------------------------------------------------------------------------------------------------
def deletion_mats(t, R, val):
    """Per validation case: deletion ceiling (all MoE outputs corrupted, damage), damage curves per A3 ordering
    (case x k, the all-clean-active deletion endpoint at k = K; noise_oracle only on its noise rows)."""
    rows = R["rows"]
    a0 = rows[(rows.fam == "a0") & (rows.order == "all_moe") & (rows.dir == "n") & rows.case_id.isin(val)]
    ceil = a0.groupby("case_id").damage.mean().reindex(val)
    flip = a0.groupby("case_id").delta.mean().reindex(val)
    d = rows[(rows.fam == "a3") & (rows.dir == "n") & rows.case_id.isin(val)].copy()
    aa = d[d.order == "all_active"].groupby("case_id").damage.mean().reindex(val)
    d = d[d.order != "all_active"]
    d["order2"] = np.where(d.order.str.startswith("rand"), "rand", d.order)
    d = d.groupby(["case_id", "row_id", "order2", "k"], as_index=False).damage.mean()
    mats = {}
    for o in DEL_ORD:
        M = d[d.order2 == o].groupby(["case_id", "k"]).damage.mean().unstack("k")
        if M.empty:
            continue
        M = M.reindex(val)
        M[t.K] = aa
        M = M[sorted(M.columns)]
        M = M[~M.isna().any(axis=1)]
        mats[o] = M
    return dict(ceil=ceil, flip=flip, mats=mats)


def deletion_boot(m, dm, W, upos):
    """Bootstrap distributions (row 0 = point) of the deletion summaries; each ordering on its own case set."""
    units = m["unit"].to_numpy()
    Mu = np.zeros((W.shape[1], len(units)))
    Mu[[upos[x] for x in units], np.arange(len(units))] = 1.0
    WU = W @ Mu
    drop = m["drop"].to_numpy()
    out = {"ceiling": (WU @ dm["ceil"].to_numpy()) / (WU @ drop),
           "flipped_all_moe": (WU @ (dm["flip"].to_numpy() < 0).astype(float)) / WU.sum(1)}
    pos = {c: i for i, c in enumerate(m["val"])}
    for o, M in dm["mats"].items():
        sel = np.array([pos[c] for c in M.index])
        Ws = WU[:, sel]
        sd = Ws @ drop[sel]
        ks = np.array(M.columns, dtype=float)
        r = (Ws @ M.to_numpy()) / sd[:, None]
        ceil = (Ws @ dm["ceil"].to_numpy()[sel]) / sd
        x = np.log(ks)
        e = {"k": ks, "r": r, "ceiling": ceil, "auc_log": np.trapezoid(r, x, axis=1) / x.max(), "n_cases": len(sel)}
        ks15 = np.arange(1, E12.KG + 1)
        r15 = np.stack([np.interp(np.log(ks15), x, ri) for ri in r])
        e["auc_log_k15"] = np.trapezoid(r15, np.log(ks15), axis=1) / np.log(E12.KG)
        e["r10"] = r[:, list(ks).index(10)]
        hit = r >= 0.8 * ceil[:, None]
        e["k80"] = np.where(hit.any(1), ks[np.argmax(hit, axis=1)], np.inf)
        e["max_r"] = r.max(1)
        e["overshoot"] = e["max_r"] - ceil
        hit = r >= 0.8 * e["max_r"][:, None]
        e["k80max"] = np.where((e["max_r"] > 0) & hit.any(1), ks[np.argmax(hit, axis=1)], np.nan)
        out[o] = e
    return out


def summarise_del(b):
    s = {k: [float(b[k][0])] + E12.pct(b[k][1:]) for k in ("ceiling", "flipped_all_moe")}
    for o in DEL_ORD:
        if o not in b:
            continue
        e = b[o]
        s[o] = {f: [float(e[f][0])] + E12.pct(e[f][1:], method="inverted_cdf" if f in ("k80", "k80max") else "linear")
                for f in ("ceiling", "auc_log", "auc_log_k15", "r10", "k80", "k80max", "max_r", "overshoot")}
        s[o]["k"] = [float(x) for x in e["k"]]
        s[o]["n_cases"] = int(e["n_cases"])
        s[o]["r_curve"] = [float(x) for x in e["r"][0]]
        s[o]["r_curve_lo"] = [float(x) for x in np.percentile(e["r"][1:], 2.5, axis=0)]
        s[o]["r_curve_hi"] = [float(x) for x in np.percentile(e["r"][1:], 97.5, axis=0)]
    return s


def sub_m(m, cases, unit=None):
    """Restrict ext12 add-back matrices to a case subset (order kept); optionally replace the units."""
    cases = [c for c in m["val"] if c in set(cases)]
    o = dict(m)
    o["val"] = cases
    o["unit"] = (unit if unit is not None else m["unit"]).reindex(cases)
    for k in ("drop", "ceil", "allmoe_delta"):
        o[k] = m[k].reindex(cases)
    o["mats"] = {k: v.reindex(cases) for k, v in m["mats"].items()}
    o["greedy"] = m["greedy"].reindex(cases) if m["greedy"] is not None else None
    return o


def sub_dm(dm, cases):
    s = set(cases)
    return dict(ceil=dm["ceil"].reindex(cases), flip=dm["flip"].reindex(cases),
                mats={k: v[v.index.isin(s)] for k, v in dm["mats"].items()})


def boot_all(m, dm, seed=0):
    upos, W = E12.unit_weights(m["unit"].to_numpy(), seed=seed)
    b = add_k80max(E12.curve_boot(m, W, upos))
    db = deletion_boot(m, dm, W, upos) if dm is not None else None
    return b, db, upos, W


def compact(s):
    """Scalars of an add-back summary (no curves)."""
    out = {k: v for k, v in s.items() if not isinstance(v, dict)}
    for o in ("oracle", "greedy", "rand", "dla", "pop", "layerwise"):
        if o in s:
            out[o] = {f: s[o][f] for f in ("auc_log", "auc_log_k15", "r10", "r15", "k80", "k80max", "max_r", "overshoot") if f in s[o]}
    return out


CEIL_MIN = 0.1  # k80 of the ceiling is reported only when the all-MoE ceiling exceeds this fraction of the drop


def add_k80max(b):
    """k80max = smallest grid k with r(k) >= 0.8 x the curve's own maximum (defined when the maximum is > 0); also for
    tasks whose all-MoE ceiling is ~0 or negative, where k80 of the ceiling is meaningless."""
    for o, e in b.items():
        if isinstance(e, dict) and "r" in e and np.ndim(e["r"]) == 2:
            r, ks = e["r"], np.asarray(e["k"], dtype=float)
            mx = r.max(1)
            hit = r >= 0.8 * mx[:, None]
            e["k80max"] = np.where((mx > 0) & hit.any(1), ks[np.argmax(hit, axis=1)], np.nan)
    return b


def summ_ab(b):
    s = E12.summarise_boot(b)
    for o, e in b.items():
        if isinstance(e, dict) and "k80max" in e and o in s:
            s[o]["k80max"] = [float(e["k80max"][0])] + E12.pct(e["k80max"][1:], method="inverted_cdf")
    s["ceiling_ok"] = bool(s["ceiling"][0] > CEIL_MIN)
    return s


def diff_ab(ba, bb):
    d = E12.diff_boot(ba, bb)
    for o in list(E12.ORD) + ["greedy"]:
        if o in ba and o in bb and "k80max" in ba[o] and "k80max" in bb[o]:
            x = ba[o]["k80max"] - bb[o]["k80max"]
            d.setdefault(o, {})["k80max"] = [float(x[0])] + E12.pct(x[1:], method="inverted_cdf")
    return d


# ------------------------------------------------------------------------------------------------------------------
# keys for paired contrasts
# ------------------------------------------------------------------------------------------------------------------
def base_id(pid: str) -> str:
    """WinoGrande twin id without condition-specific suffixes (copy-prefix attribute order, etc.)."""
    s = str(pid)
    for sep in ("|", "#", "::"):
        if sep in s:
            s = s.split(sep)[0]
    return s


ITEM_COLS = ("item_key", "base_item", "item_id", "fill_id", "seed_fill", "base_pair_id")


def case_keys(spec, t):
    """Per validation case: (item key for pairing across conditions, cluster for resampling, names flag)."""
    pt = AB.pair_table(spec, t).set_index("case_id")
    tmpl = spec.cond.startswith("vt_") or spec.cond.startswith("cp_vt")
    nm = [c for c in ("name_x", "name_y") if c in pt.columns]
    if tmpl and "seed" in pt.columns and len(nm) == 2:  # template item = seed template x name pair (both copy-prefix orders share it)
        item = pt["seed"].astype(str) + "|" + pt.name_x.astype(str) + "|" + pt.name_y.astype(str)
    else:  # natural twins: WinoGrande twin id (copy-prefix order suffix stripped)
        item = pt.pair_id.map(base_id)
    clus = pt["cluster"].astype(str) if tmpl else pt.pair_id.map(base_id)
    names = pt["names"].astype(bool) if "names" in pt.columns else None
    return item, clus, names, pt


def p3_keys(mk, t):
    cfg = E12.main_cfg(mk)
    if cfg["task"] == "cf":
        return None
    cs = json.load(open(os.path.join(AB.ROOT, cfg["case_sets"])))
    pairs = P.load_pairs(os.path.join(AB.ROOT, cfg["pairs"]), cs)
    pid = {int(c): pairs.loc[int(u)].pair_id for c, u in zip(t.cases.case_id, t.cases.unit)}
    s = pd.Series(pid)
    nm = pd.Series({int(c): bool(pairs.loc[int(u)].names) for c, u in zip(t.cases.case_id, t.cases.unit)})
    return s.map(base_id), s.map(base_id), nm


# ------------------------------------------------------------------------------------------------------------------
# direct split (p5-data's run)
# ------------------------------------------------------------------------------------------------------------------
def direct_bars(spec, t, mine):
    val = set(int(c) for c in t.cases.case_id[t.cases.split == "validation"])
    p = os.path.join(RESULTS, spec.src, "direct_split.parquet")
    src = "p5-data direct_split.parquet"
    dd = pd.read_parquet(p) if os.path.exists(p) else None
    if dd is None or not val <= set(dd.case_id.astype(int)):
        dd, src = mine, "this run's addback_direct.parquet (first pass)"
    if dd is None:
        return None
    dd = dd[dd.case_id.isin(val)].copy()
    dd["dr32"] = dd.d_clean_fp32 - dd.d_corrupt_fp32
    dd["a_dir"] = dd.d_attn_direct - dd.d_corrupt_fp32
    dd["m_dir"] = dd.d_moe_direct - dd.d_corrupt_fp32
    g = dd.groupby("case_id")[["dr32", "a_dir", "m_dir"]].mean()
    unit = t.cases.set_index("case_id").unit
    rc = X.ratio_cols(g[["a_dir", "m_dir"]], g.dr32, unit)
    out = {k: [float(v.ratio), float(v.lo), float(v.hi)] for k, v in rc.iterrows()}
    out["source"] = src
    out["n_cases"] = int(len(g))
    if mine is not None and src.startswith("p5-data"):
        mm = mine[mine.case_id.isin(val)]
        mm = mm.assign(dr32=mm.d_clean_fp32 - mm.d_corrupt_fp32, a_dir=mm.d_attn_direct - mm.d_corrupt_fp32)
        out["check_this_run_a_dir"] = float(mm.a_dir.sum() / mm.dr32.sum())
    return out


def strata_boot(m, dm, strata):
    st = {}
    for lab, cs in strata.items():
        mm = sub_m(m, cs)
        nu = int(mm["unit"].nunique())
        st[lab] = {"n_cases": len(mm["val"]), "n_units": nu}
        if nu >= MIN_UNITS:
            bs, dbs, _, _ = boot_all(mm, sub_dm(dm, mm["val"]))
            st[lab].update(compact(summ_ab(bs)))
            st[lab]["del_ceiling"] = summarise_del(dbs)["ceiling"]
    return st


# ------------------------------------------------------------------------------------------------------------------
def analyse_task(spec, keep):
    log(f"analysing {spec.key}")
    A8.load_task = lambda _cfg, _s=spec: AB.load_task(_s)
    cfg = dict(model=spec.model, task="wino", src=spec.src, out=spec.out, label=spec.label, tlabel=spec.tlabel)
    r, rows, t = A8.analyse(spec.key, cfg)
    R = X.load_out(spec.out)
    m = E12.addback_mats(t, R)
    dm = deletion_mats(t, R, m["val"])
    b, db, upos, W = boot_all(m, dm)
    out = {"key": spec.key, "cond": spec.cond, "proto": spec.proto, "model": spec.model, "family": spec.family, "label": spec.label,
           "tlabel": spec.tlabel, "src": spec.src, "out": spec.out, "K": t.K, "n_eval_cases": len(m["val"]),
           "n_units": len(upos), "mean_drop": float(m["drop"].mean()), "greedy_done": f"{m['n_greedy_done']}/{m['n_greedy']}",
           "ext8": r, "boot": summ_ab(b), "del": summarise_del(db),
           "check_auc_oracle_ext8_vs_boot": [r["a1"]["oracle"]["auc_log"], float(b["oracle"]["auc_log"][0])] if "oracle" in r["a1"] else None}
    item, clus, names, pt = case_keys(spec, t)
    # seed / template clustering
    if spec.cluster_col:
        mc = sub_m(m, m["val"], clus)
        bc, dbc, uc, _ = boot_all(mc, dm)
        out["boot_seed"] = compact(summ_ab(bc))
        out["del_seed"] = {k: v for k, v in summarise_del(dbc).items() if k in ("ceiling", "noise_oracle", "rand")}
        out["n_clusters"] = len(uc)
    # strata: names / objects; association probe neutral / non-neutral (where the case set mixes strata)
    strata = {}
    if names is not None:
        strata["names"] = [c for c in m["val"] if bool(names.get(c, False))]
        strata["objects"] = [c for c in m["val"] if not bool(names.get(c, True))]
    if "stratum_assoc" in pt.columns and pt.loc[m["val"], "stratum_assoc"].nunique() > 1:
        sa = pt.stratum_assoc.astype(str)
        strata["assoc neutral"] = [c for c in m["val"] if sa.get(c) == "neutral"]
        strata["assoc not neutral"] = [c for c in m["val"] if sa.get(c) != "neutral"]
    out["strata"] = strata_boot(m, dm, strata)
    out["direct"] = direct_bars(spec, t, R.get("direct"))
    keep[spec.key] = dict(m=m, dm=dm, item=item.reindex(m["val"]), clus=clus.reindex(m["val"]), names=names, model=spec.model)
    return out


def analyse_p3(mk, keep):
    log(f"Phase-3 reference {mk}")
    cfg = E12.main_cfg(mk)
    t = E12.load_main_task(mk)
    R = X.load_out(cfg["out"])
    m = E12.addback_mats(t, R)
    dm = deletion_mats(t, R, m["val"])
    b, db, upos, W = boot_all(m, dm)
    model, lab = P3[mk]
    out = {"key": mk, "cond": "P3_" + mk.split("_")[0], "model": model, "label": AB.MODEL_LABEL[model], "tlabel": lab, "K": t.K,
           "n_eval_cases": len(m["val"]), "n_units": len(upos), "mean_drop": float(m["drop"].mean()),
           "boot": summ_ab(b), "del": summarise_del(db)}
    e8 = json.load(open(os.path.join(RESULTS, "ext8_addback_summary.json")))["runs"][mk]
    out["ext8"] = {"a0": e8["a0"]}
    dv = e8["a0"]["validation"].get("direct")
    if dv:
        out["direct"] = {k: [dv[k]["ratio"], dv[k]["lo"], dv[k]["hi"]] for k in ("a_dir", "m_dir")}
        out["direct"]["source"] = "ext8 summary (Phase 3)"
    ks = p3_keys(mk, t)
    if ks is not None:
        item, clus, names = ks
        keep[mk] = dict(m=m, dm=dm, item=item.reindex(m["val"]), clus=clus.reindex(m["val"]), names=names, model=model)
        out["strata"] = strata_boot(m, dm, {"names": [c for c in m["val"] if bool(names.get(c, False))],
                                            "objects": [c for c in m["val"] if not bool(names.get(c, True))]})
    else:
        keep[mk] = dict(m=m, dm=dm, item=None, clus=None, names=None, model=model)
    return out


# ------------------------------------------------------------------------------------------------------------------
# contrasts
# ------------------------------------------------------------------------------------------------------------------
def contrast_specs():
    """(a, b, label, kind): a - b. Keys are condition names (protocol appended per model) or Phase-3 keys 'P3wino'."""
    return [
        ("cp_nat", "P3wino", "copy availability, natural twins (C2 - C1): copy prefix - Phase-3 option swap", "copy"),
        ("cp_vt_orig", "vt_e2a_orig", "copy availability, templates orig verb (C2 - C1)", "copy"),
        ("cp_vt_neu", "vt_e2a_neu", "copy availability, templates neutral verb (C2 - C1)", "copy"),
        ("dir_f1", "cp_nat", "direction with copy fixed, natural twins (C3 - C2): F1 - copy prefix", "copy_dir"),
        ("vt_a2e_orig", "cp_vt_orig", "direction with copy fixed, templates orig (C3 - C2)", "copy_dir"),
        ("vt_a2e_neu", "cp_vt_neu", "direction with copy fixed, templates neutral (C3 - C2)", "copy_dir"),
        ("dir_f1", "dir_opt", "direction on identical twins: F1 (attr->entity) - option swap (entity->attr)", "direction"),
        ("vt_a2e_orig", "vt_e2a_orig", "direction on identical templates, orig verb", "direction"),
        ("vt_a2e_neu", "vt_e2a_neu", "direction on identical templates, neutral verb", "direction"),
        ("dir_opt", "P3wino", "sanity: option swap on the F1-margin twins - Phase-3 option swap", "sanity"),
        ("vt_e2a_neu", "vt_e2a_orig", "verb control ent->attr: neutral - orig", "verb"),
        ("vt_a2e_neu", "vt_a2e_orig", "verb control attr->entity: neutral - orig", "verb"),
        ("f1", "P3wino", "F1 (neutral main set) - Phase-3 option swap", "other"),
        ("f2e", "f1", "F2E natural sentence - F1", "other"),
    ]


def resolve(name, model, keep):
    if name == "P3wino":
        return f"wino_{model}" if f"wino_{model}" in keep else None
    if name == "F1pre_smoke":
        k = f"F1pre_{AB.MODEL_PROTO[model]}_smoke"
        return k if k in keep else None
    k = f"{name}_{AB.MODEL_PROTO[model]}"
    return k if k in keep else None


def contrast(ka, kb, keep):
    A, B = keep[ka], keep[kb]
    tmpl = any(k.startswith(("vt_", "cp_vt")) for k in (ka, kb))
    out = {"a": ka, "b": kb}
    common = set()
    if A["item"] is not None and B["item"] is not None:
        common = set(A["item"].values) & set(B["item"].values)
    ca = [c for c in A["m"]["val"] if A["item"] is not None and A["item"].get(c) in common]
    cb = [c for c in B["m"]["val"] if B["item"] is not None and B["item"].get(c) in common]
    ua = set(A["clus"].reindex(ca).values) if ca else set()
    ub = set(B["clus"].reindex(cb).values) if cb else set()
    n_common_units = len(ua & ub)
    out.update(n_common_items=len(common), n_cases_a=len(ca), n_cases_b=len(cb), n_common_units=n_common_units,
               cluster="seed template" if tmpl else "twin")
    if n_common_units >= MIN_UNITS:
        cu = ua & ub
        ca = [c for c in ca if A["clus"].get(c) in cu]
        cb = [c for c in cb if B["clus"].get(c) in cu]
        ma, mb = sub_m(A["m"], ca, A["clus"]), sub_m(B["m"], cb, B["clus"])
        allu = np.array(sorted(cu))
        upos, W = E12.unit_weights(allu, seed=0)
        ba, bb = add_k80max(E12.curve_boot(ma, W, upos)), add_k80max(E12.curve_boot(mb, W, upos))
        da = deletion_boot(ma, sub_dm(A["dm"], ca), W, upos)
        dbb = deletion_boot(mb, sub_dm(B["dm"], cb), W, upos)
        out["paired"] = True
        out["a_vals"], out["b_vals"] = compact(summ_ab(ba)), compact(summ_ab(bb))
        out["diff"] = diff_ab(ba, bb)
        out["del_diff"] = {"ceiling": _d(da["ceiling"], dbb["ceiling"])}
        for o in ("noise_oracle", "rand"):
            if o in da and o in dbb:
                out["del_diff"][o] = {f: _d(da[o][f], dbb[o][f], f == "k80") for f in ("auc_log", "k80", "overshoot")}
    else:  # independent samples (full validation sets; a seed 0, b seed 1)
        ma, mb = A["m"], B["m"]
        if tmpl:
            ma, mb = sub_m(ma, ma["val"], A["clus"]), sub_m(mb, mb["val"], B["clus"])
        ba, _, _, _ = boot_all(ma, None, seed=0)
        bb, _, _, _ = boot_all(mb, None, seed=1)
        out["paired"] = False
        out["a_vals"], out["b_vals"] = compact(summ_ab(ba)), compact(summ_ab(bb))
        out["diff"] = diff_ab(ba, bb)
    return out


def _d(a, b, k80=False):
    d = a - b
    if k80:
        d = np.where(np.isfinite(a) & np.isfinite(b), d, np.nan)
    return [float(d[0])] + E12.pct(d[1:], method="inverted_cdf" if k80 else "linear")


# ------------------------------------------------------------------------------------------------------------------
# tables
# ------------------------------------------------------------------------------------------------------------------
fci, fkci, fk, f3 = E12.fci, E12.fkci, E12.fk, E12.f3


def tables(S):
    os.makedirs(TAB, exist_ok=True)
    runs = S["runs"]
    order = sorted(runs, key=lambda k: (AB.PLAN_ORDER.index(runs[k]["cond"]) if runs[k]["cond"] in AB.PLAN_ORDER else 99,
                                       runs[k]["family"] != "main", runs[k]["model"] != "qwen3"))
    rows = []
    for key in list(S.get("phase3", {})) + order:
        r = S["phase3"][key] if key in S.get("phase3", {}) else runs[key]
        b, dl = r["boot"], r["del"]
        g = b.get("greedy", {})
        dr = r.get("direct") or {}
        rows.append([r["label"], r["tlabel"], r["n_eval_cases"], r["n_units"], f"{r['mean_drop']:.2f}",
                     fci(b["ceiling"]), fci(dl["ceiling"]), f3(r.get("ext8", {}).get("a0", {}).get("validation", {}).get("d", {}).get("int_moe", {}).get("ratio")),
                     (f"{f3(dr['a_dir'][0])} / {f3(dr['m_dir'][0])}" if dr.get("a_dir") else ""),
                     (f"{fkci(g.get('k80'))} / {fkci(b['oracle']['k80'])} / {fkci(b['rand']['k80'])}" if b["ceiling"][0] > CEIL_MIN
                      else f"n/a (M ≤ {CEIL_MIN})"),
                     f"{fkci(g.get('k80max'))} / {fkci(b['oracle'].get('k80max'))} / {fkci(b['rand'].get('k80max'))}",
                     fci(b["oracle"]["overshoot"], sign=True), fci(g.get("r15")),
                     f"{fci(g.get('auc_log_k15'))} / {fci(b['oracle']['auc_log_k15'])}",
                     fci(b.get("greedy_minus_best_static_r10"), sign=True), fci(b.get("dla_minus_oracle_auc"), sign=True),
                     fci(b["restored_all_moe"], 2),
                     f"{fkci(dl.get('noise_oracle', {}).get('k80'))} / {fkci(dl.get('rand', {}).get('k80'))}"])
    E12.md_table(["Model", "Condition", "Eval. cases", "Units", "Mean drop", "All-MoE ceiling (add-back)", "All-MoE deletion",
                  "MoE ≤ L−5", "Direct attn / MoE", "k80 of ceiling greedy / oracle / random", "k80 of own max greedy / oracle / random", "Oracle overshoot (max r − M)", "Greedy r(15)",
                  "AUC k ≤ 15 greedy / oracle", "Greedy − best static r(10)", "DLA − oracle AUC", "Answer restored (all MoE)",
                  "Deletion k80 oracle / random"], rows, "ext13ab_conditions",
                 "Phase 5 add-back / deletion per condition (validation directed cases; fraction of the drop; pair bootstrap 95% CI)")
    # orderings (add-back) and deletion orderings
    rows = []
    for key in order:
        r = runs[key]
        for o in list(E12.ORD) + ["greedy"]:
            e = r["boot"].get(o)
            if not e:
                continue
            rows.append([r["label"], r["tlabel"], o, f3(e["r_curve"][0]), fci(e.get("r10")), fci(e.get("auc_log")) if "auc_log" in e else "–",
                         fci(e.get("auc_log_k15")), fkci(e.get("k80")) if r["boot"]["ceiling"][0] > CEIL_MIN else "n/a", fkci(e.get("k80max")),
                         fci(e.get("max_r")) if "max_r" in e else "–",
                         fci(e.get("overshoot"), sign=True) if "overshoot" in e else "–"])
    E12.md_table(["Model", "Condition", "Ordering", "r(1)", "r(10)", "AUC log k (1..K)", "AUC log k (1..15)", "k80 of ceiling", "k80 of own max", "max r", "overshoot"],
                 rows, "ext13ab_orderings", "Add-back orderings per condition (validation; 95% CI)")
    rows = []
    for key in order:
        r = runs[key]
        for o in DEL_ORD:
            e = r["del"].get(o)
            if not e:
                continue
            rows.append([r["label"], r["tlabel"], o, e["n_cases"], fci(e["ceiling"]), f3(e["r_curve"][0]), fci(e["r10"]), fci(e["auc_log"]),
                         fkci(e["k80"]) if e["ceiling"][0] > CEIL_MIN else "n/a", fkci(e["k80max"]), fci(e["max_r"]), fci(e["overshoot"], sign=True)])
    E12.md_table(["Model", "Condition", "Ordering", "Cases", "Deletion ceiling (same cases)", "d(1)", "d(10)", "AUC log k", "k80 of ceiling",
                  "k80 of own max", "max damage", "overshoot"], rows, "ext13ab_deletion", "Deletion curves per condition (damage / drop; validation; 95% CI)")
    # seed-clustered and strata
    rows = []
    for key in order:
        r = runs[key]
        if "boot_seed" not in r:
            continue
        a, s = r["boot"], r["boot_seed"]
        rows.append([r["label"], r["tlabel"], r["n_units"], r.get("n_clusters"), fci(a["ceiling"]), fci(s["ceiling"]),
                     f"{fkci(a['greedy']['k80'])} / {fkci(s['greedy']['k80'])}", f"{fkci(a['oracle']['k80'])} / {fkci(s['oracle']['k80'])}",
                     f"{fci(a['oracle']['overshoot'], sign=True)} / {fci(s['oracle']['overshoot'], sign=True)}",
                     f"{fci(a['greedy']['auc_log_k15'])} / {fci(s['greedy']['auc_log_k15'])}"])
    if rows:
        E12.md_table(["Model", "Condition", "Pairs", "Seed templates", "Ceiling (pair CI)", "Ceiling (seed CI)", "k80 greedy (pair / seed CI)",
                      "k80 oracle (pair / seed CI)", "Overshoot (pair / seed CI)", "Greedy AUC k ≤ 15 (pair / seed CI)"], rows, "ext13ab_seed_ci",
                     "Template conditions: pair-clustered vs seed-clustered bootstrap CIs (same point estimates)")
    rows = []
    for key in list(S.get("phase3", {})) + order:
        r = S["phase3"][key] if key in S.get("phase3", {}) else runs[key]
        for lab, e in (r.get("strata") or {}).items():
            if "ceiling" not in e:
                rows.append([r["label"], r["tlabel"], lab, e["n_cases"], e["n_units"], "–", "–", "–", "–", "–"])
                continue
            rows.append([r["label"], r["tlabel"], lab, e["n_cases"], e["n_units"], fci(e["ceiling"]), fci(e["del_ceiling"]),
                         f"{fkci(e['greedy']['k80'])} / {fkci(e['oracle']['k80'])}", fci(e["oracle"]["overshoot"], sign=True),
                         fci(e["greedy"]["auc_log_k15"])])
    E12.md_table(["Model", "Condition", "Stratum", "Cases", "Pairs", "Ceiling", "Deletion ceiling", "k80 greedy / oracle", "Oracle overshoot",
                  "Greedy AUC k ≤ 15"], rows, "ext13ab_strata", f"Strata: names vs objects, association-probe neutral vs not (CIs only with >= {MIN_UNITS} pairs)")
    # contrasts
    rows = []
    for c in S.get("contrasts", []):
        d = c["diff"]
        rows.append([AB.MODEL_LABEL[c["model"]], c["label"], c["a"], c["b"],
                     ("paired" if c["paired"] else "independent") + f" ({c['n_common_units']} common {c['cluster']} units)",
                     f"{f3(c['a_vals']['ceiling'][0])} / {f3(c['b_vals']['ceiling'][0])}", fci(d.get("ceiling"), sign=True),
                     fci(d.get("oracle", {}).get("overshoot"), sign=True),
                     (fci(d.get("greedy", {}).get("k80"), 0, sign=True) if min(c["a_vals"]["ceiling"][0], c["b_vals"]["ceiling"][0]) > CEIL_MIN else "n/a"),
                     (fci(d.get("oracle", {}).get("k80"), 0, sign=True) if min(c["a_vals"]["ceiling"][0], c["b_vals"]["ceiling"][0]) > CEIL_MIN else "n/a"),
                     fci(d.get("greedy", {}).get("k80max"), 0, sign=True), fci(d.get("greedy", {}).get("auc_log_k15"), sign=True),
                     fci(d.get("oracle", {}).get("auc_log"), sign=True), fci(d.get("restored_all_moe"), 2, sign=True),
                     fci(c.get("del_diff", {}).get("ceiling"), sign=True) if c.get("del_diff") else "–"])
    if rows:
        E12.md_table(["Model", "Contrast", "a", "b", "Bootstrap", "Ceiling a / b", "Δ ceiling", "Δ oracle overshoot", "Δ k80 greedy", "Δ k80 oracle",
                      "Δ k80 of own max greedy", "Δ greedy AUC k ≤ 15", "Δ oracle AUC", "Δ answer restored", "Δ deletion ceiling"], rows, "ext13ab_contrasts",
                     "Copy / direction / verb contrasts a − b (paired = restricted to common items, same resamples; 95% CI; k80 only where both finite)")


def jdump(S, path):
    with open(path, "w") as f:
        json.dump(E12.jsafe(S), f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-phase3", action="store_true")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    global TAB, SUMM
    if args.outdir:
        TAB = args.outdir
        os.makedirs(TAB, exist_ok=True)
        SUMM = os.path.join(args.outdir, "ext13ab_summary.json")
    E12.TAB = TAB
    if args.smoke:
        specs = {s.key: s for s in (AB.smoke_spec("qwen3"), AB.smoke_spec("mixtral"))}
    else:
        specs = AB.registry(ready_only=False)
    if args.tasks:
        specs = {k: v for k, v in specs.items() if k in args.tasks.split(",")}
    S = {"runs": {}, "phase3": {}, "contrasts": [], "notes": {
        "units": "pairs (both directed cases of a pair together); template conditions also seed-clustered",
        "deletion_k80": "k80 of the deletion ceiling computed on the curve's own cases (noise_oracle: its noise rows)",
        "contrasts": f"paired when >= {MIN_UNITS} common units (restricted to items in both conditions' validation sets), else independent"}}
    keep = {}
    for key, spec in specs.items():
        od = os.path.join(RESULTS, spec.out)
        if not os.path.exists(os.path.join(od, "addback_state.pkl")):
            continue
        if not AB.run_complete(spec.out) and not args.partial:
            log(f"{key}: not complete (use --partial)")
            continue
        S["runs"][key] = analyse_task(spec, keep)
    models = sorted({r["model"] for r in S["runs"].values()})
    if not args.no_phase3:
        for mk, (model, _) in P3.items():
            if model in models:
                S["phase3"][mk] = analyse_p3(mk, keep)
    for model in models:
        cspecs = contrast_specs() + ([("F1pre_smoke", "P3wino", "smoke test of the contrast code", "smoke")] if args.smoke else [])
        for a, b, lab, kind in cspecs:
            ka, kb = resolve(a, model, keep), resolve(b, model, keep)
            if ka and kb:
                log(f"contrast {ka} - {kb}")
                c = contrast(ka, kb, keep)
                c.update(model=model, label=lab, kind=kind)
                S["contrasts"].append(c)
    S["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jdump(S, SUMM)
    tables(S)
    log(f"wrote {SUMM}")


if __name__ == "__main__":
    main()
