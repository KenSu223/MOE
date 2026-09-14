"""Direction 4: turn a CodeFact scan (scripts/ext4_scan.py) into case sets, calibration tables and the sweep files
that moetrace.analysis / ext1_analysis expect.

Per category: items passing the paper's absolute filter (Delta_clean >= 1.0, drop >= 0.5; primary) are shuffled with
random.Random(0); the first 256 form the category set, split 128/128 (moetrace.ext4_data.split_ids, seed 0). With
fewer than 256 passing items the set is all passing items (>= --min-cases), split in half, and flagged `partial`. A
mixed set `all` (256, stratified over categories from the category sets) is added. The relative rule (drop >= 25 % of
Delta_clean, Delta_clean >= 1.0) is reported in the calibration tables only (appendix), not used for selection.

Usage: python scripts/ext4_select.py <run_name> [--n-cases 256] [--min-cases 64] [--figure]
Writes results/<run>/case_sets.json, sweep_cases.parquet, sweep_rows.parquet, sweep_routing.parquet, sweep_summary.json,
results/tables/ext4_calibration_<run>.{md,csv}, ext4_calibration_quantiles_<run>.{md,csv}, and (with --figure)
results/figures/ext4_calibration_<run>.png.
"""
import argparse, json, os, sys, time, random
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import RESULTS
from moetrace.ext4_data import CATEGORIES, CATEGORY_LABEL, split_ids
from moetrace.report import md_table

TAB, FIG = os.path.join(RESULTS, "tables"), os.path.join(RESULTS, "figures")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_sets(ct: pd.DataFrame, n_cases: int, min_cases: int, seed: int = 0) -> dict:
    sets = {}
    for cat in CATEGORIES:
        p = ct[(ct.category == cat) & ct.strict].case_id.tolist()
        rng = random.Random(seed)
        rng.shuffle(p)
        if len(p) < min_cases:
            sets[cat] = {"discovery": [], "validation": [], "n_pass": len(p), "partial": True, "skipped": True}
            continue
        chosen = p[:n_cases]
        d, v = split_ids(chosen, seed)
        sets[cat] = {"discovery": d, "validation": v, "n_pass": len(p), "partial": len(p) < n_cases, "skipped": False}
    # mixed set: stratified over the category sets
    pools = {c: sorted(sets[c]["discovery"] + sets[c]["validation"]) for c in CATEGORIES if not sets[c]["skipped"]}
    for c in pools:
        random.Random(seed + 1).shuffle(pools[c])
    mixed, i = [], 0
    while len(mixed) < n_cases and any(pools.values()):
        for c in CATEGORIES:
            if c in pools and pools[c] and len(mixed) < n_cases:
                mixed.append(pools[c].pop())
    d, v = split_ids(mixed, seed)
    sets["all"] = {"discovery": d, "validation": v, "n_pass": int(ct.strict.sum()), "partial": len(mixed) < n_cases, "skipped": len(mixed) < min_cases,
                   "composition": {c: int(sum(ct.set_index('case_id').loc[mixed].category == c)) for c in CATEGORIES}}
    return sets


def calibration_tables(ct: pd.DataFrame, run: str, label: str) -> tuple[str, str]:
    rows = []
    for cat in list(CATEGORIES) + ["ALL"]:
        g = ct if cat == "ALL" else ct[ct.category == cat]
        if not len(g):
            continue
        n = len(g)
        rows.append([f"{cat} {'' if cat == 'ALL' else CATEGORY_LABEL[cat]}".strip(), n,
                     f"{g.delta_clean.median():+.2f}", f"{g['drop'].median():+.2f}",
                     f"{int(g.strict.sum())} ({100 * g.strict.mean():.0f} %)", f"{int(g.relaxed.sum())} ({100 * g.relaxed.mean():.0f} %)",
                     f"{int(g.relative25.sum())} ({100 * g.relative25.mean():.0f} %)", f"{int(g.relative50.sum())} ({100 * g.relative50.mean():.0f} %)",
                     f"{100 * g.top1_clean_is_true.mean():.0f} % / {100 * g.top1_clean_startswith_true.mean():.0f} %" if "top1_clean_startswith_true" in g else f"{100 * g.top1_clean_is_true.mean():.0f} %",
                     f"{100 * g.top1_noised_is_true.mean():.0f} % / {100 * g.top1_noised_startswith_true.mean():.0f} %" if "top1_noised_startswith_true" in g else f"{100 * g.top1_noised_is_true.mean():.0f} %",
                     f"{100 * (g.delta_clean >= 1.0).mean():.0f} %", f"{100 * (g['drop'] >= 0.5).mean():.0f} %",
                     f"{100 * g.final_is_max_L5.mean():.1f} %" if "final_is_max_L5" in g else "n/a"])
    hdr = ["Category", "n scanned", "median Δ_clean", "median drop", "paper filter (Δ≥1, drop≥0.5)", "relaxed (Δ≥0.5, drop≥0.25)",
           "relative (drop ≥ 25 % Δ_clean, Δ≥1)", "relative 50 %", "top-1 = true / starts with true (clean)", "top-1 = true / starts with true (noised)", "Δ_clean ≥ 1", "drop ≥ 0.5",
           "final token carries max norm (L5)"]
    t1 = md_table(hdr, rows, os.path.join(TAB, f"ext4_calibration_{run}"),
                  f"{label}: per-category calibration of the paper's filter on CodeFact (clean vs subject-noised Δ = logit(true) − logit(foil))")
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    rows = []
    for cat in CATEGORIES:
        g = ct[ct.category == cat]
        if not len(g):
            continue
        rows.append([cat, len(g), "Δ_clean"] + [f"{v:+.2f}" for v in g.delta_clean.quantile(qs)] + [f"{g.delta_clean.mean():+.2f}"])
        rows.append([cat, len(g), "drop"] + [f"{v:+.2f}" for v in g["drop"].quantile(qs)] + [f"{g['drop'].mean():+.2f}"])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = (g["drop"] / g.delta_clean)[g.delta_clean >= 1.0]
        rows.append([cat, int(len(rel)), "drop / Δ_clean (Δ_clean ≥ 1)"] + [f"{v:+.2f}" for v in rel.quantile(qs)] + [f"{rel.mean():+.2f}"])
    t2 = md_table(["Category", "n", "quantity", "5 %", "25 %", "median", "75 %", "95 %", "mean"], rows,
                  os.path.join(TAB, f"ext4_calibration_quantiles_{run}"), f"{label}: quantiles of Δ_clean and of the subject-noise drop per category")
    return t1, t2


def calibration_figure(ct: pd.DataFrame, path: str, label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
    COL = {"S1": "#2a78d6", "S2": "#3d9bd1", "S3": "#5cc0c0", "R1": "#eb6834", "R2": "#d9483b", "R3": "#b0648f"}
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.4), sharex="row")
    xb = np.linspace(-5, 25, 31)
    db = np.linspace(-5, 20, 26)
    for j, cat in enumerate(CATEGORIES):
        g = ct[ct.category == cat]
        ax = axes[0][j]
        ax.hist(np.clip(g.delta_clean, xb[0], xb[-1]), bins=xb, color=COL[cat], alpha=0.85)
        ax.axvline(1.0, color=INK, lw=0.8, ls="--")
        ax.set_title(f"{cat} {CATEGORY_LABEL[cat]} (n={len(g)})", fontsize=8.5, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax = axes[1][j]
        ax.hist(np.clip(g["drop"], db[0], db[-1]), bins=db, color=COL[cat], alpha=0.85)
        ax.axvline(0.5, color=INK, lw=0.8, ls="--")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_xlabel("drop = Δ_clean − Δ_noised")
        ax.text(0.98, 0.92, f"pass {100 * g.strict.mean():.0f} %\nrel25 {100 * g.relative25.mean():.0f} %", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    axes[0][0].set_ylabel("items (Δ_clean; dashed: 1.0)")
    axes[1][0].set_ylabel("items (drop; dashed: 0.5)")
    fig.suptitle(f"{label}: CodeFact calibration, clean margin and subject-noise drop per category (values clipped to the axis range)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n-cases", type=int, default=256)
    ap.add_argument("--min-cases", type=int, default=64)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    od = os.path.join(RESULTS, args.run)
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    ct = pd.read_parquet(os.path.join(od, "scan_cases.parquet"))
    rows = pd.read_parquet(os.path.join(od, "scan_rows.parquet"))
    routing = pd.read_parquet(os.path.join(od, "scan_routing.parquet"))
    meta = json.load(open(os.path.join(od, "scan_meta.json")))
    label = args.label or f"{meta['model']} ({meta['protocol']})"
    log(f"{args.run}: {len(ct)} scanned items; strict pass per category: {ct[ct.strict].category.value_counts().to_dict()}")
    sets = build_sets(ct, args.n_cases, args.min_cases)
    sets["scan"] = {"n_scanned": int(len(ct)), "per_category": ct.category.value_counts().to_dict(),
                    "strict_pass": ct[ct.strict].category.value_counts().to_dict(), "relaxed_pass": ct[ct.relaxed].category.value_counts().to_dict(),
                    "relative25_pass": ct[ct.relative25].category.value_counts().to_dict(), "thresholds": {"primary": "paper absolute (1.0, 0.5)"},
                    "sigma": meta["sigma"], "embed_std": meta["embed_std"], "protocol": meta["protocol"], "model": meta["model"]}
    with open(os.path.join(od, "case_sets.json"), "w") as f:
        json.dump(sets, f, indent=1)
    names = [s for s in list(CATEGORIES) + ["all"] if not sets[s]["skipped"]]
    union = []
    for s in names:
        for c in sets[s]["discovery"] + sets[s]["validation"]:
            if c not in union:
                union.append(c)
    log(f"sets: {[(s, len(sets[s]['discovery']), len(sets[s]['validation']), 'partial' if sets[s]['partial'] else '') for s in names]}; union {len(union)}")
    # sweep files restricted to the union (analysis.load_model API)
    sc = ct[ct.case_id.isin(union)].copy()
    for s in list(CATEGORIES) + ["all"]:
        d, v = set(sets[s]["discovery"]), set(sets[s]["validation"])
        sc[f"in_{s}"] = sc.case_id.isin(d | v)
        sc[f"split_{s}"] = sc.case_id.map(lambda c: "discovery" if c in d else ("validation" if c in v else ""))
    sc = sc.set_index("case_id").loc[union].reset_index()
    sc.to_parquet(os.path.join(od, "sweep_cases.parquet"), index=False)
    rows[rows.case_id.isin(union)].to_parquet(os.path.join(od, "sweep_rows.parquet"), index=False)
    routing[routing.case_id.isin(union)].to_parquet(os.path.join(od, "sweep_routing.parquet"), index=False)
    R = rows[(rows.kind == "layer") & rows.case_id.isin(union)].pivot(index="case_id", columns="layer", values="rescue")
    summ = {"n_cases": len(union), "sets": names}
    for s in names:
        dsc, val = sets[s]["discovery"], sets[s]["validation"]
        md_, mv = R.loc[dsc].mean(0), R.loc[val].mean(0)
        lstar = int(md_.idxmax())
        summ[s] = {"n_disc": len(dsc), "n_val": len(val), "L_star_discovery": lstar, "disc_mean_at_Lstar": float(md_[lstar]),
                   "val_mean_at_Lstar": float(mv[lstar]), "val_argmax": int(mv.idxmax()), "val_max": float(mv.max()),
                   "val_curve": [round(float(x), 3) for x in mv.values], "partial": sets[s]["partial"]}
        log(f"set {s}: n={len(dsc)}+{len(val)} L*={lstar} disc={md_[lstar]:+.3f} val@L*={mv[lstar]:+.3f} val argmax L{int(mv.idxmax())} {mv.max():+.3f}")
    with open(os.path.join(od, "sweep_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    t1, t2 = calibration_tables(ct, args.run, label)
    print(t1)
    if args.figure:
        calibration_figure(ct, os.path.join(FIG, f"ext4_calibration_{args.run}.png"), label)
    log("done")


if __name__ == "__main__":
    main()
