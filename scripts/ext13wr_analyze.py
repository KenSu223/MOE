"""ext13wr analysis (CPU): writer vs computer per Phase-5 condition (Part A) and vocabulary projections (Part B).

Part A: total T (p5-data's single-expert STR patch rescue) vs direct D (DLA of delta_e from the writer pass, final norm
frozen at the corrupted twin) vs indirect I = T - D; per layer (sum over clean-active experts), per expert (selected,
WinoGrande, CounterFact experts, population top-10; validation cases), the top-10 as a set, T-vs-D agreement by band,
ranking agreement. Statistics = moetrace.ext11_writer (cluster bootstrap over the condition's units).
Part B: vec_rows / vec_top of the writer pass: rank of the right answer token over the vocabulary and within the
condition's own readout class, class z-scores (condition's attribute words, its entities, Phase-3 trigger vocabulary, IOI
names, CounterFact objects), top-50 shares, most promoted tokens, exact vs frozen final norm.

Writes results/ext13wr_summary<suffix>.json, results/tables/ext13wr_{A,B}_*<suffix>.md|csv,
results/figures/ext13wr_*<suffix>.png.

    python scripts/ext13wr_analyze.py [--protos qwen3,mixtral_bos] [--conds ...] [--suffix _smoke] [--no-figs]
"""
import argparse, collections, json, os, sys, time, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
import numpy as np, pandas as pd
from moetrace import ext13ko as K, ext13wr as WR
from moetrace import ext11_writer as W
from moetrace.models import MODELS, RESULTS

TABLES = os.path.join(RESULTS, "tables")
FIGS = os.path.join(RESULTS, "figures")
CLAB = {"f1": "F1 a→e", "dir_f1": "776 a→e", "dir_opt": "776 e→a", "f2e": "F2E a→e", "vt_e2a_orig": "tmpl e→a orig",
        "vt_e2a_neu": "tmpl e→a neu", "vt_a2e_orig": "tmpl a→e orig", "vt_a2e_neu": "tmpl a→e neu",
        "cp_vt_orig": "copy tmpl orig", "cp_vt_neu": "copy tmpl neu", "cp_nat": "copy 776", "f1pre": "F1 pre (smoke)"}
CLS = ("attr_own", "ent_own", "wg_trigger", "names", "cf_any")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def v3(d, k="v"):
    return None if not d or d.get(k) is None or not np.isfinite(d.get(k)) else round(float(d[k]), 4)


def ci(d, digits=3):
    if not d or d.get("v") is None or not np.isfinite(d.get("v", np.nan)):
        return "n/a"
    return f"{d['v']:+.{digits}f} [{d['lo']:+.{digits}f}, {d['hi']:+.{digits}f}]"


def part_a(e, sfx):
    m = WR.pair_table(e, sfx)
    n_all = len(m)
    match = WR.match_rate(m)
    m = m[m.routed_wr].copy()
    from moetrace.arch import load_spec
    spec, _ = load_spec(MODELS[e["model"]]["repo"])
    L = spec.n_layers
    kod = K.out_dir(e, sfx)
    tg = K.targets(e, kod)
    pop = [(l, x) for l, x, _ in K.pop_ranking(e, kod)[:10]]
    exps = list(dict.fromkeys(tg["all"] + pop))
    res = {"n_pairs_rows": int(n_all), "routing_match": round(match, 4), "n_cases": int(m.case_id.nunique()), "L": L,
           "splits": m.groupby("split").case_id.nunique().to_dict(), "targets": {k: (K.le(*v) if isinstance(v, tuple) else
           [K.le(*x) for x in v] if isinstance(v, list) else None) for k, v in tg.items()}, "pop_top10": [K.le(*x) for x in pop]}
    sp_main = ("discovery", "validation")
    have = set(m.split.unique())
    val = ("validation",) if "validation" in have else tuple(have)
    res["layers"] = W.layer_profile(m, L, splits=tuple(s for s in m.split.unique()))
    res["agreement"] = W.band_agreement(m, L, splits=tuple(s for s in m.split.unique()))
    res["experts_val"] = W.expert_stats(m, exps, splits=val)
    res["experts_all"] = W.expert_stats(m, exps, splits=tuple(m.split.unique()))
    if any(s.startswith("rep_") for s in have):
        res["experts_rep"] = W.expert_stats(m, exps, splits=tuple(s for s in have if s.startswith("rep_")))
    res["set_pop10_val"] = W.set_stats(m, pop, splits=val)
    try:
        res["ranking_val"] = W.ranking_agreement(m, splits=val)
    except Exception as ex:  # noqa: BLE001
        res["ranking_val"] = {"error": str(ex)}
    # share by depth band of the summed layer writes
    lay = res["layers"]
    res["band_share"] = {}
    for nm, (lo, hi) in {"first half": (0, L // 2), "L/2..L-9": (L // 2, L - 8), "L-8..L-5": (L - 8, L - 4), "last 4": (L - 4, L)}.items():
        T = sum(x["T_frac"]["v"] for x in lay if lo <= x["layer"] < hi)
        D = sum(x["D_frac"]["v"] for x in lay if lo <= x["layer"] < hi)
        res["band_share"][nm] = {"T_frac": round(T, 4), "D_frac": round(D, 4), "share": round(D / T, 4) if abs(T) > 1e-9 else None}
    # cross-check vs p5-addback's DLA when available
    ab = os.path.join(RESULTS, f"p5_{e['cond']}_{e['proto']}_ab", "addback_dla.parquet")
    if not os.path.exists(ab):
        ab = os.path.join(RESULTS, f"p5_{e['proto']}_ab", e["cond"], "addback_dla.parquet")
    if os.path.exists(ab):
        try:
            x = pd.read_parquet(ab)
            key = "case_id" if "case_id" in x.columns else "row_id"
            j = m.merge(x, left_on=[key if key == "case_id" else "row_id", "layer", "expert"], right_on=[key, "layer", "expert"])
            res["xcheck_addback_dla"] = {"file": ab, "n": int(len(j)), "r": float(np.corrcoef(j["D"], j["dla"])[0, 1]),
                                         "mean_abs_diff": float((j["D"] - j["dla"]).abs().mean())}
        except Exception as ex:  # noqa: BLE001
            res["xcheck_addback_dla"] = {"file": ab, "error": str(ex)}
    return res, m


def part_b(e, sfx, tok, piece):
    od = WR.out_dir(e, sfx)
    vr = pd.read_parquet(os.path.join(od, "vec_rows.parquet"))
    vt = pd.read_parquet(os.path.join(od, "vec_top.parquet"))
    pr = pd.read_parquet(os.path.join(od, "prompts.parquet")).set_index("item")
    cls = {k: set(v) for k, v in json.load(open(os.path.join(od, "classes.json"))).items()}
    from moetrace.arch import load_spec
    Vn = int(load_spec(MODELS[e["model"]]["repo"])[0].vocab)
    vr["vec"] = np.arange(len(vr))
    vr["name"] = [K.le(a, b) for a, b in zip(vr.layer, vr.expert)]
    vr["split"] = pr.split.reindex(vr.prompt.values).values
    vr["unit"] = pr.unit.reindex(vr.prompt.values).values
    out = {"n_vectors": int(len(vr)), "class_sizes": {k: len(v) for k, v in cls.items()}, "V": Vn, "experts": {}}
    for nm, s0 in vr.groupby("name"):
        rec = {}
        for kind in ("delta", "clean"):
            s = s0[s0.kind == kind]
            if len(s) < 5:
                continue
            st = {"n": int(len(s)), "n_units": int(s.unit.nunique()), "median_rank_true": float(s.rank_true.median()),
                  "frac_true_top10": float((s.rank_true <= 10).mean()), "frac_true_top100": float((s.rank_true <= 100).mean()),
                  "median_rank_low_foil": float(s.rank_low_foil.median()),
                  "median_rank_true_in_class": float(s.rank_true_in_class.median()) if s.rank_true_in_class.notna().any() else None,
                  "n_class": float(s.n_class.median()) if s.n_class.notna().any() else None,
                  "mean_val_true_minus_foil": float((s.val_true - s.val_foil).mean()),
                  "exact_over_frozen": float(s.exact.sum() / s.frozen.sum()) if abs(s.frozen.sum()) > 1e-9 else None,
                  "mean_abs_err": float((s.exact - s.frozen).abs().mean())}
            u, Wm = W.boot_matrix(s.unit.values, 1000)
            for k in CLS:
                col = f"clsz_{k}"
                if col in s:
                    S = W.unit_sums(s.assign(one=1.0), [col, "one"], u)
                    st[f"z_{k}"] = W.ratio_ci(S[:, 0], S[:, 1], Wm)
                    st[f"top50_{k}"] = float(s[f"top50_{k}"].mean())
                    st[f"base_{k}"] = len(cls.get(k, ())) / Vn
            tp = vt.iloc[s.vec.values]
            ct = collections.Counter(int(i) for ids in tp.top_ids for i in ids[:10])
            cb = collections.Counter(int(i) for ids in tp.bot_ids for i in ids[:10])
            st["top_promoted"] = [(piece(i), round(c / len(s), 3)) for i, c in ct.most_common(12)]
            st["top_suppressed"] = [(piece(i), round(c / len(s), 3)) for i, c in cb.most_common(8)]
            rec[kind] = st
        out["experts"][nm] = rec
    d = vr[vr.kind == "delta"]
    if len(d):
        out["norm_pooled"] = {"n": int(len(d)), "exact_over_frozen": float(d.exact.sum() / d.frozen.sum()),
                              "r": float(np.corrcoef(d.exact, d.frozen)[0, 1]) if len(d) > 3 else None,
                              "mean_abs_err": float((d.exact - d.frozen).abs().mean())}
    return out


def table(df, name, title, note=""):
    os.makedirs(TABLES, exist_ok=True)
    df.to_csv(os.path.join(TABLES, name + ".csv"), index=False)
    with open(os.path.join(TABLES, name + ".md"), "w") as f:
        f.write(f"**{title}**\n\n" + W.md_table(df) + ("\n\n" + note if note else "") + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protos", default="qwen3,mixtral_bos")
    ap.add_argument("--conds", default=",".join(K.CONDITIONS))
    ap.add_argument("--suffix", default="")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    sfx = args.suffix
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    S = {"updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": {}}
    rows_a, rows_l, rows_b = [], [], []
    for proto in args.protos.split(","):
        model = K.PROTO[proto]["model"]
        tok = None
        for cond in args.conds.split(","):
            try:
                e = K.entry(cond, proto)
            except AssertionError:
                continue
            if not WR.pass_done(e, sfx):
                continue
            if tok is None:
                tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[model]["repo"]))
                piece = (lambda i: tok.convert_tokens_to_string([tok.convert_ids_to_tokens(int(i))])) if model == "qwen3" else \
                    (lambda i: tok.convert_ids_to_tokens(int(i)).replace("▁", " "))
            log(f"{cond}/{proto}")
            rec = {"direction": e["direction"]}
            try:
                a, m = part_a(e, sfx)
                rec["A"] = a
            except FileNotFoundError as ex:
                log(f"  Part A skipped: {ex}")
                a = None
            rec["B"] = part_b(e, sfx, tok, piece)
            S["models"].setdefault(proto, {})[cond] = rec
            lab = K.PROTO[proto]["label"]
            if a:
                tg = a["targets"]
                for x in a["experts_val"]:
                    role = []
                    if x["expert"] == tg["selected"]:
                        role.append("selected")
                    if x["expert"] == tg["wg"]:
                        role.append("WG")
                    if x["expert"] in tg["cf"]:
                        role.append("CF")
                    if x["expert"] in a["pop_top10"]:
                        role.append(f"pop{a['pop_top10'].index(x['expert']) + 1}")
                    rows_a.append({"model": lab, "condition": cond, "expert": x["expert"], "role": "+".join(role),
                                   "active cases": f"{x['n_active']}/{x['n_cases']}", "total T / drop": ci(x.get("T_frac")),
                                   "direct D / drop": ci(x.get("D_frac")), "direct share D/T": ci(x.get("share"), 2),
                                   "all-case T / drop": v3(x.get("T_allcase_frac"))})
                sp = a["set_pop10_val"]
                rows_a.append({"model": lab, "condition": cond, "expert": "top-10 set", "role": "pop", "active cases": "",
                               "total T / drop": ci(sp["T_frac"]), "direct D / drop": ci(sp["D_frac"]),
                               "direct share D/T": ci(sp["share"], 2), "all-case T / drop": None})
                bs = a["band_share"]
                rows_l.append({"model": lab, "condition": cond, "routing match": a["routing_match"],
                               **{f"{k}: T / D / share": f"{v['T_frac']:+.3f} / {v['D_frac']:+.3f} / {v['share'] if v['share'] is None else round(v['share'], 2)}"
                                  for k, v in bs.items()},
                               "r(D, T) all": round(a["agreement"]["all"]["corr"]["r"], 3) if np.isfinite(a["agreement"]["all"]["corr"]["r"]) else None})
            b = rec["B"]
            tgts = K.targets(e, K.out_dir(e, sfx))
            for nm, r in b["experts"].items():
                for kind, st in r.items():
                    if nm not in [K.le(*x) for x in tgts["all"]] and kind == "clean":
                        continue
                    rows_b.append({"model": lab, "condition": cond, "expert": nm, "vectors": kind, "n": st["n"],
                                   "median rank of r (of V)": f"{st['median_rank_true']:.0f}",
                                   "r in top-10 / top-100 (%)": f"{100 * st['frac_true_top10']:.0f} / {100 * st['frac_true_top100']:.0f}",
                                   "median rank in own class (size)": f"{st['median_rank_true_in_class']:.0f} ({st['n_class']:.0f})" if st["median_rank_true_in_class"] is not None else "",
                                   **{f"z {k}": (f"{st[f'z_{k}']['v']:+.2f}" if f"z_{k}" in st and np.isfinite(st[f'z_{k}']['v']) else "") for k in CLS},
                                   "top-50 attr / ent (%)": f"{100 * st.get('top50_attr_own', np.nan):.1f} / {100 * st.get('top50_ent_own', np.nan):.1f}",
                                   "most promoted": "; ".join(f"{repr(t_)} {100 * c:.0f}%" for t_, c in st["top_promoted"][:6])})
    sp = os.path.join(RESULTS, f"ext13wr_summary{sfx}.json")
    json.dump(S, open(sp, "w"), indent=1, default=W._json_default)
    if rows_a:
        table(pd.DataFrame(rows_a), f"ext13wr_A_experts{sfx}",
              "Writer vs computer per condition (validation cases): total T (single-expert STR patch rescue), direct D (DLA of δ_e, "
              "final norm frozen at the corrupted twin), as fractions of the drop over the cases where the expert is clean-active; "
              "direct share D/T [95% CI]; all-case T = pop convention")
        table(pd.DataFrame(rows_l), f"ext13wr_A_bands{sfx}",
              "Summed single-expert writes by depth band (all cases): T / drop, D / drop, direct share D/T; routing match = share of "
              "p5-data's clean-active (case, expert) pairs also routed in the writer pass")
    if rows_b:
        table(pd.DataFrame(rows_b), f"ext13wr_B_vocab{sfx}",
              "Vocabulary projections of the expert writes at the final position (δ_e: frozen at the corrupted twin; c_e: own "
              "prompt): rank of the right answer token r over the vocabulary and within the condition's readout class; class z "
              "= (class mean − vocabulary mean) / SD of the projection; attr_own / ent_own = the condition's trigger words / "
              "entities; wg_trigger = Phase-3 trigger vocabulary; names = IOI names; cf_any = CounterFact objects")
    if not args.no_figs and S["models"]:
        figures(S, sfx)
    log(f"wrote {sp}")
    # sections (the chains call ext13ko_analyze.py, then this script, after every finished condition)
    import subprocess
    for scr, summ in (("ext13ko_text.py", f"ext13ko_summary{sfx}.json"), ("ext13wr_text.py", f"ext13wr_summary{sfx}.json")):
        if os.path.exists(os.path.join(RESULTS, summ)):
            r = subprocess.run([sys.executable, os.path.join("/home/ubuntu/MOE/scripts", scr)] + (["--suffix", sfx] if sfx else []),
                               capture_output=True, text=True)
            log(f"{scr}: rc={r.returncode} {r.stdout.strip()[-200:]} {r.stderr.strip()[-500:]}")


def figures(S, sfx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGS, exist_ok=True)
    for proto, cd in S["models"].items():
        for cond, rec in cd.items():
            a = rec.get("A")
            if not a:
                continue
            fig, ax = plt.subplots(figsize=(7, 3.2))
            Ls = [x["layer"] for x in a["layers"]]
            ax.bar(np.array(Ls) - 0.2, [x["T_frac"]["v"] for x in a["layers"]], width=0.4, color="#4472c4", label="total (single patches)")
            ax.bar(np.array(Ls) + 0.2, [x["D_frac"]["v"] for x in a["layers"]], width=0.4, color="#ed7d31", label="direct (DLA)")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("layer")
            ax.set_ylabel("sum over experts / drop")
            ax.legend(fontsize=7)
            ax.set_title(f"{K.PROTO[proto]['label']} — {CLAB.get(cond, cond)}: writer vs computer by layer", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(FIGS, f"ext13wr_layers_{proto}_{cond}{sfx}.png"), dpi=130)
            plt.close(fig)
    # overview: direct share and class z of the selected and WinoGrande experts across conditions
    ms = list(S["models"].items())
    fig, axes = plt.subplots(len(ms), 2, figsize=(13, 3.6 * len(ms)), squeeze=False)
    for row, (proto, cd) in enumerate(ms):
        conds = [c for c in list(K.CONDITIONS) + ["f1pre"] if c in cd]
        wg = K.le(*K.PROTO[proto]["wg"])
        ax = axes[row, 0]
        for si, (nm, col) in enumerate((("selected", "#3b6fb6"), ("wg", "#2e9e5b"), ("top10", "#7a4fa3"))):
            ys, lo, hi = [], [], []
            for c in conds:
                a = cd[c].get("A")
                d = None
                if a:
                    if nm == "top10":
                        d = a["set_pop10_val"]["D_frac"]
                    else:
                        t = a["targets"]["selected"] if nm == "selected" else wg
                        d = next((x.get("D_frac") for x in a["experts_val"] if x["expert"] == t), None)
                ok = d is not None and d.get("v") is not None and np.isfinite(d["v"])
                ys.append(d["v"] if ok else np.nan)
                lo.append(d["v"] - d["lo"] if ok else 0)
                hi.append(d["hi"] - d["v"] if ok else 0)
            ax.bar(np.arange(len(conds)) + (si - 1) * 0.27, ys, width=0.27, color=col, yerr=[lo, hi], capsize=2,
                   error_kw={"lw": 0.6}, label={"selected": "own selected expert", "wg": f"WinoGrande expert {wg}", "top10": "own top-10 set"}[nm])
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels([CLAB.get(c, c) for c in conds], rotation=30, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("direct write D / drop (active cases)")
        ax.set_title(f"{K.PROTO[proto]['label']}: direct write (validation)", fontsize=9)
        ax.legend(fontsize=6)
        ax = axes[row, 1]
        for si, (k, col) in enumerate((("attr_own", "#c8553d"), ("ent_own", "#3b6fb6"), ("names", "#999999"))):
            ys = []
            for c in conds:
                st = cd[c]["B"]["experts"].get(wg, {}).get("delta", {})
                z = st.get(f"z_{k}")
                ys.append(z["v"] if z else np.nan)
            ax.bar(np.arange(len(conds)) + (si - 1) * 0.27, ys, width=0.27, color=col, label=k)
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels([CLAB.get(c, c) for c in conds], rotation=30, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("class z of δ_e projection")
        ax.set_title(f"{wg} δ_e: attribute vs entity vocabulary", fontsize=9)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"ext13wr_overview{sfx}.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
