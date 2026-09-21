"""Phase 2 / F1.3 driver (ext5-analysis): per-case minimal expert sets and pairwise interactions from the exhaustive
subset passes produced by the ext5-engine agent.

Input schema (agreed in RESEARCH_PLAN.md "Phase 2 execution plan"): results/<run>_subsets/subset_rows.parquet with columns
case_id, layer, experts (sorted comma-joined expert ids), n_experts, rescue; one row per non-empty subset of the case's
clean-active experts at the layer. The per-case block rescue comes from the subset run's own expert_rows.parquet (kind
`layer`) when present, else from the matching all-layer pass (bf16 noise only).

Usage: python scripts/ext5_rank_subsets.py [--dirs results/qwen3_subsets,results/mixtral_nobos_subsets] [--no-rebuild]
       python scripts/ext5_rank_subsets.py --synthetic qwen3_bos_alllayers:44 --out /tmp/x   (schema test on fabricated data)
Writes results/tables/ext5_rank_subsets_*.{md,csv}, results/figures/ext5_rank_subsets_<short>.{png,pdf},
results/ext5_subsets_summary.json and the fragment results/tables/ext5_rank_subsets_section.md, then rebuilds
results/sections/ext5_f1_rankings.md via scripts/ext5_rank_analyze.py --section-only.
"""
import argparse, itertools, json, os, subprocess, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import analysis as A
from moetrace import ext5_rank as R
from moetrace.models import RESULTS
from moetrace.report import md_table
from moetrace.stats import summarize

TAB, FIG = (os.path.join(RESULTS, d) for d in ("tables", "figures"))
SUMMARY = os.path.join(RESULTS, "ext5_subsets_summary.json")
# subset run directory name -> (all-layer run for the block fallback / case sets, short name, label, singletons of interest)
CANDIDATES = {
    "qwen3_subsets": ("qwen3_bos_alllayers", "qwen3", "Qwen3-30B-A3B-Base (tokenizer defaults)", {44: [69], 42: [115]}),
    "qwen3_bos_subsets": ("qwen3_bos_alllayers", "qwen3", "Qwen3-30B-A3B-Base (tokenizer defaults)", {44: [69], 42: [115]}),
    "mixtral_nobos_subsets": ("mixtral_nobos_alllayers", "mixtral_nobos", "Mixtral-8x7B-v0.1 (no BOS, paper protocol)", {19: [6, 2], 18: [1]}),
    "mixtral_bos_subsets": ("mixtral_bos_alllayers", "mixtral_bos", "Mixtral-8x7B-v0.1 (BOS, tokenizer default)", {19: [2], 18: [1]}),
    "mixtral_subsets": ("mixtral_bos_alllayers", "mixtral_bos", "Mixtral-8x7B-v0.1 (BOS, tokenizer default)", {19: [2], 18: [1]}),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def cis(s):
    return "n/a" if s["n"] == 0 else f"{s['mean']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"


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


# ---------------------------------------------------------------------------------------------------------------
def make_synthetic(run: str, layer: int, out_dir: str, seed: int = 0, noise: float = 0.15) -> str:
    """Fabricate subset_rows for a schema test: singles = the real single-expert rescues, the full set = the real
    coalition_clean rescue, intermediate subsets = additive sum + N(0, noise) shrunk towards the coalition."""
    md = A.load_model(run)
    e = md.expert_rows
    et = e[(e.layer == layer) & (e.kind == "expert") & e.clean_active.astype(bool)]
    co = e[(e.layer == layer) & (e.kind == "coalition_clean")].set_index("case_id").rescue
    rng = np.random.default_rng(seed)
    rows = []
    for cid, g in et.groupby("case_id"):
        singles = dict(zip(g.expert.astype(int), g.rescue.astype(float)))
        ex = sorted(singles)
        k = len(ex)
        full = float(co[cid])
        total_dev = full - sum(singles.values())
        for r in range(1, k + 1):
            for sub in itertools.combinations(ex, r):
                s = sum(singles[x] for x in sub)
                if r == k:
                    val = full
                elif r == 1:
                    val = s
                else:
                    val = s + total_dev * (r * (r - 1)) / (k * (k - 1)) + rng.normal(0, noise)
                rows.append((int(cid), layer, ",".join(str(x) for x in sub), r, float(val)))
    df = pd.DataFrame(rows, columns=["case_id", "layer", "experts", "n_experts", "rescue"])
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "subset_rows.parquet"), index=False)
    return os.path.join(out_dir, "subset_rows.parquet")


# ---------------------------------------------------------------------------------------------------------------
def analyze_dir(sub_dir: str, base_run: str, short: str, label: str, singletons: dict) -> dict:
    sub = R.load_subsets(os.path.join(sub_dir, "subset_rows.parquet"))
    md = A.load_model(base_run)
    split = {c: "discovery" for c in md.sets["paper"]["discovery"]}
    split.update({c: "validation" for c in md.sets["paper"]["validation"]})
    layers = sorted(int(l) for l in sub.layer.unique())
    out = dict(dir=sub_dir, base_run=base_run, short=short, label=label, layers=layers, n_rows=int(len(sub)), n_cases=int(sub.case_id.nunique()), per_layer={})
    size_rows, single_rows, inter_rows, add_rows, decomp_rows = [], [], [], [], []
    figdata = {}
    for L in layers:
        s = sub[sub.layer == L]
        ids = sorted(s.case_id.unique())
        block = R.block_reference(sub_dir, md, L, ids, sub)
        ref_rows = sub.attrs.get("reference")
        k = int(s.n_experts.max())
        # sanity: exhaustive?
        n_per_case = s.groupby("case_id").size()
        exhaustive = bool((n_per_case == 2 ** k - 1).all())
        mins = R.per_case_minimal_sets(s, block, L)
        mins["split"] = mins.case_id.map(split)
        mins.to_csv(os.path.join(TAB, f"ext5_rank_subsets_percase_{short}_L{L}.csv"), index=False)
        # exact full set vs additive vs block (fingerprint against the all-layer pass)
        full_vs_block = summarize((mins.full_rescue - mins.block).values, with_p=False)
        full_vs_coal = None
        if ref_rows is not None and (ref_rows.kind == "coalition_clean").any():
            cc = ref_rows[(ref_rows.kind == "coalition_clean") & (ref_rows.layer == L)].set_index("case_id").rescue.reindex(mins.case_id).values
            full_vs_coal = summarize(mins.full_rescue.values - cc, with_p=False)
        res = dict(k=k, exhaustive=exhaustive, n_cases=int(len(ids)), n_eligible=int(len(mins)), n_block_le_0=int(len(ids) - len(mins)),
                   full_minus_block=full_vs_block, full_minus_coalition_clean=full_vs_coal, mean_full_over_block=float(mins.full_rescue.mean() / mins.block.mean()))
        for t in R.TARGETS:
            tag = f"{int(round(100 * t))}"
            sz = mins[f"min_size_{tag}"]
            reached = sz.notna()
            hist = {int(i): int((sz == i).sum()) for i in range(1, k + 1)}
            add_sz = mins[f"add_size_{tag}"]
            both = reached & add_sz.notna()
            same_size = (sz[both] == add_sz[both])
            same_set = (mins.loc[both, f"min_set_{tag}"] == mins.loc[both, f"add_set_{tag}"])
            res[tag] = dict(n_reached=int(reached.sum()), hist=hist, median=float(sz.median()) if reached.any() else None, mean=float(sz.mean()) if reached.any() else None,
                            frac_size1=float((sz == 1).sum() / len(mins)), frac_size_le2=float((sz <= 2).sum() / len(mins)), n_never=int((~reached).sum()),
                            additive_n_reached=int(add_sz.notna().sum()), additive_mean=float(add_sz.mean()) if add_sz.notna().any() else None,
                            additive_same_size=int(same_size.sum()), additive_same_set=int(same_set.sum()), n_both=int(both.sum()),
                            additive_set_reaches_exactly=int(mins.loc[add_sz.notna(), f"add_set_reaches_{tag}"].sum()), additive_larger=int((add_sz[both] > sz[both]).sum()), additive_smaller=int((add_sz[both] < sz[both]).sum()))
            size_rows.append([f"L{L}", f"{int(round(100 * t))}%", f"{len(mins)}/{len(ids)}", " / ".join(str(hist[i]) for i in range(1, k + 1)) + f" / never {int((~reached).sum())}",
                              f"{sz.median():.0f} / {sz.mean():.2f}" if reached.any() else "n/a", f"{100 * (sz == 1).sum() / len(mins):.0f}% / {100 * (sz <= 2).sum() / len(mins):.0f}%",
                              f"{add_sz.mean():.2f}" if add_sz.notna().any() else "n/a", f"{int(same_size.sum())} / {int(same_set.sum())} / {int(both.sum())}",
                              f"{int(mins.loc[add_sz.notna(), f'add_set_reaches_{tag}'].sum())}/{int(add_sz.notna().sum())}"])
        for e in singletons.get(L, []):
            ss = R.singleton_sufficiency(mins, s, L, e)
            res.setdefault("singletons", {})[str(e)] = ss
            single_rows.append([f"L{L}E{e:03d}", f"{ss['n_active']}/{ss['n_eligible']}"] + [f"{ss[f'suffices_{tag}']} ({100 * ss[f'suffices_{tag}'] / max(ss['n_active'], 1):.0f}% of active)" for tag in ("50", "80", "90")]
                               + [f"{ss[f'is_min_set_{tag}']}" for tag in ("50", "80", "90")])
        # interactions
        Mi, Cn, long = R.interaction_matrix(s, L)
        Mi.to_csv(os.path.join(TAB, f"ext5_rank_subsets_interaction_matrix_{short}_L{L}.csv")); Cn.to_csv(os.path.join(TAB, f"ext5_rank_subsets_interaction_counts_{short}_L{L}.csv"))
        long.to_parquet(os.path.join(TAB, f"ext5_rank_subsets_interaction_long_{short}_L{L}.parquet"), index=False)
        isum = summarize(long.interaction.values, with_p=True) if len(long) else summarize([])
        res["interaction_all_pairs"] = isum
        res["frac_pairs_negative"] = float((long.interaction < 0).mean()) if len(long) else np.nan
        # per expert pair with enough co-occurrences: mean interaction, sorted
        if len(long):
            g = long.groupby(["a", "b"]).agg(n=("interaction", "size"), mean_i=("interaction", "mean"), mean_ra=("r_a", "mean"), mean_rb=("r_b", "mean"), mean_rab=("r_ab", "mean")).reset_index()
            g = g[g.n >= max(5, len(ids) // 25)].sort_values("mean_i")
            g.to_csv(os.path.join(TAB, f"ext5_rank_subsets_interaction_pairs_{short}_L{L}.csv"), index=False)
            res["n_pairs_evaluated"] = int(len(g))
            res["most_redundant"] = g.head(5).to_dict("records"); res["most_synergistic"] = g.tail(5).iloc[::-1].to_dict("records")
            shown = pd.concat([g.head(5), g.tail(5).iloc[::-1]]).drop_duplicates(["a", "b"]) if len(g) > 10 else g
            for _, r in shown.iterrows():
                inter_rows.append([f"L{L}", f"E{int(r.a):03d} + E{int(r.b):03d}", int(r.n), f3(r.mean_ra), f3(r.mean_rb), f3(r.mean_rab), f3(r.mean_i), "redundant" if r.mean_i < 0 else "synergistic"])
            # interactions involving the singletons of interest
            for e in singletons.get(L, []):
                ge = long[(long.a == e) | (long.b == e)]
                res.setdefault("interaction_with", {})[str(e)] = summarize(ge.interaction.values, with_p=True) if len(ge) else summarize([])
        dec = R.nonadditivity_decomposition(s, L)
        dec.to_csv(os.path.join(TAB, f"ext5_rank_subsets_decomposition_{short}_L{L}.csv"), index=False)
        if len(dec):
            res["decomposition"] = dict(total=summarize(dec.total_nonadd.values, with_p=True), pairwise=summarize(dec.pairwise_sum.values, with_p=True),
                                        higher=summarize(dec.higher_order.values, with_p=True), r_total_pairwise=float(np.corrcoef(dec.total_nonadd, dec.pairwise_sum)[0, 1]) if len(dec) > 2 else np.nan,
                                        mean_abs_total=float(dec.total_nonadd.abs().mean()), mean_abs_higher=float(dec.higher_order.abs().mean()))
            decomp_rows.append([f"L{L}", k, cis(res["decomposition"]["total"]), f"{dec.total_nonadd.abs().mean():.3f}", cis(res["decomposition"]["pairwise"]), cis(res["decomposition"]["higher"]),
                                f"{dec.higher_order.abs().mean():.3f}", f"{res['decomposition']['r_total_pairwise']:.2f}"])
        add_rows.append([f"L{L}", f"{len(mins)}/{len(ids)}", cis(full_vs_block), f"{100 * res['mean_full_over_block']:.0f}%", cis(isum), f"{100 * res['frac_pairs_negative']:.0f}%" if len(long) else "n/a"])
        out["per_layer"][str(L)] = res
        figdata[L] = dict(mins=mins, long=long, k=k)
    md_table(["Layer", "Target (x own block)", "Eligible cases (block > 0)", "Smallest exact subset size: 1 / 2 / ... / k / never", "Median / mean size", "Size 1 / <= 2 (share of eligible)",
              "Additive prediction: mean size", "Additive = exact: size / set / n compared", "Additive set reaches target when patched exactly"],
             size_rows, os.path.join(TAB, f"ext5_rank_subsets_sizes_{short}"), caption=f"{label}: per-case minimal expert sets from exact subset patches (smallest subset of the case's clean-active experts whose joint patch reaches the target fraction of that case's block rescue)")
    if single_rows:
        md_table(["Expert", "Active among eligible", "Alone reaches 50%", "Alone reaches 80%", "Alone reaches 90%", "Is the 50% minimal set", "Is the 80% minimal set", "Is the 90% minimal set"],
                 single_rows, os.path.join(TAB, f"ext5_rank_subsets_singletons_{short}"), caption=f"{label}: singleton sufficiency of the experts of interest")
    md_table(["Layer", "Pair", "n cases", "rescue(a)", "rescue(b)", "rescue(a,b)", "Interaction", "Type"],
             inter_rows, os.path.join(TAB, f"ext5_rank_subsets_interactions_{short}"), caption=f"{label}: the five most redundant and five most synergistic expert pairs per layer (interaction = rescue(a,b) - rescue(a) - rescue(b), mean over cases where both are active; pairs with >= 5 co-occurrences)")
    md_table(["Layer", "Eligible", "Exact full set - block [95% CI]", "Full set / block", "Mean pairwise interaction, all pairs [95% CI]", "Pairs with negative interaction"],
             add_rows, os.path.join(TAB, f"ext5_rank_subsets_overview_{short}"), caption=f"{label}: subset-pass overview")
    if decomp_rows:
        md_table(["Layer", "k", "rescue(full) - sum singles [95% CI]", "mean |.|", "Sum of pairwise interactions [95% CI]", "Higher-order remainder [95% CI]", "mean |remainder|", "r(total, pairwise)"],
                 decomp_rows, os.path.join(TAB, f"ext5_rank_subsets_decomposition_{short}"), caption=f"{label}: decomposition of the per-case non-additivity into second-order (pairwise) and higher-order terms")
    figure(short, label, figdata, os.path.join(FIG, f"ext5_rank_subsets_{short}"))
    return out


def figure(short, label, figdata, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = "#2a78d6" if short.startswith("qwen") else "#eb6834"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    layers = sorted(figdata)
    fig, axes = plt.subplots(2, len(layers), figsize=(4.6 * len(layers), 6.6), squeeze=False)
    for j, L in enumerate(layers):
        d = figdata[L]
        ax = axes[0][j]
        k = d["k"]
        xs = np.arange(1, k + 2)
        for t, a in zip((0.5, 0.8, 0.9), (0.35, 0.7, 1.0)):
            sz = d["mins"][f"min_size_{int(round(100 * t))}"]
            counts = [int((sz == i).sum()) for i in range(1, k + 1)] + [int(sz.isna().sum())]
            ax.bar(xs + (a - 0.7) * 0.28, counts, width=0.26, color=C, alpha=a, label=f"{int(100 * t)}% of own block")
        ax.set_xticks(xs); ax.set_xticklabels([str(i) for i in range(1, k + 1)] + ["never"])
        ax.set_xlabel("smallest exact subset size"); ax.set_ylabel("cases")
        ax.set_title(f"L{L}: per-case minimal set size ({len(d['mins'])} cases with block > 0)", loc="left", fontsize=9)
        ax.legend(frameon=False, fontsize=7.5); ax.grid(axis="y", color=GRID, lw=0.6)
        ax = axes[1][j]
        if len(d["long"]):
            ax.hist(d["long"].interaction.values, bins=40, color=C, alpha=0.8)
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_xlabel("pairwise interaction rescue(a,b) - rescue(a) - rescue(b)"); ax.set_ylabel("expert pairs x cases")
            ax.set_title(f"L{L}: interactions (mean {d['long'].interaction.mean():+.3f}, {100 * (d['long'].interaction < 0).mean():.0f}% negative)", loc="left", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6)
    fig.suptitle(f"{label}: exact per-case minimal sets and pairwise interactions (subset passes)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path + ".png", dpi=170); fig.savefig(path + ".pdf"); plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def write_fragment(summary: dict, path: str):
    L = ["### F1.3 per-case minimal sets and interactions (exact subset patches)\n"]
    L.append("**Method.** For every case and layer of the subset passes (`results/<run>_subsets/subset_rows.parquet`, ext5-engine's `coalition_set` kind), all "
             "2^k - 1 subsets of the case's k clean-active experts are patched jointly. The per-case minimal set at target t is the smallest subset whose "
             "exact rescue reaches t x the case's own block rescue (ties broken by the higher rescue; cases with block <= 0 excluded); the additive prediction "
             "sorts the single-expert rescues and accumulates until the target. Pairwise interaction at S = {} is rescue({a,b}) - rescue({a}) - rescue({b}); "
             "negative = redundant (the two restore the same thing), positive = synergistic. The per-case non-additivity rescue(full) - sum(singles) is "
             "decomposed into the sum of all pairwise interactions and a higher-order remainder.\n")
    for key, o in summary["runs"].items():
        L.append(f"#### {o['label']} (`{os.path.relpath(o['dir'], '/home/ubuntu/MOE') if os.path.isabs(o['dir']) else o['dir']}`)\n")
        for Ls, r in o["per_layer"].items():
            fb = r["full_minus_block"]
            fc = r.get("full_minus_coalition_clean")
            L.append(f"- L{Ls}: k = {r['k']} experts per case, {r['n_cases']} cases, {'exhaustive' if r['exhaustive'] else 'NOT exhaustive'} ({2 ** r['k'] - 1} subsets each); {r['n_eligible']} cases with block > 0. "
                     f"Exact full set minus block: {cis(fb)}" + (f"; minus the same pass's coalition_clean row: {cis(fc)} (identity check)" if fc else "") + ". Smallest exact subset for 80% of the case's block: "
                     f"median {r['80']['median']}, size 1 in {100 * r['80']['frac_size1']:.0f}% and <= 2 in {100 * r['80']['frac_size_le2']:.0f}% of eligible cases, never in {r['80']['n_never']}; "
                     f"the additive prediction has the same size in {r['80']['additive_same_size']}/{r['80']['n_both']} and the same set in {r['80']['additive_same_set']}/{r['80']['n_both']} cases, "
                     f"and reaches the target when patched exactly in {r['80']['additive_set_reaches_exactly']}/{r['80']['additive_n_reached']}. "
                     f"Mean pairwise interaction {cis(r['interaction_all_pairs'])}, {100 * r['frac_pairs_negative']:.0f}% of pair-cases negative.")
            for e, ss in r.get("singletons", {}).items():
                L.append(f"  - {ss['pair']} alone: active in {ss['n_active']}/{ss['n_eligible']} eligible cases; reaches 50/80/90% of the case's block in {ss['suffices_50']}/{ss['suffices_80']}/{ss['suffices_90']} "
                         f"of them; is the exact minimal 80% set in {ss['is_min_set_80']} cases. Its mean interaction with co-active experts: {cis(r.get('interaction_with', {}).get(e, summarize([])))}.")
            if r.get("decomposition"):
                d = r["decomposition"]
                L.append(f"  - Non-additivity: rescue(full) - sum(singles) {cis(d['total'])} (mean |.| {d['mean_abs_total']:.3f}); pairwise sum {cis(d['pairwise'])}; higher-order remainder {cis(d['higher'])} "
                         f"(mean |.| {d['mean_abs_higher']:.3f}); r(total, pairwise) = {d['r_total_pairwise']:.2f}.")
        L.append("")
        for t in ("sizes", "singletons", "interactions", "decomposition"):
            p = os.path.join(TAB, f"ext5_rank_subsets_{t}_{o['short']}.md")
            if os.path.exists(p):
                L.append(open(p).read())
        L.append(f"![ext5 subsets {o['short']}](../figures/ext5_rank_subsets_{o['short']}.png)\n")
        L.append(f"Figure E5-F1.3-{o['short']}: top, distribution of the smallest exact subset reaching 50/80/90% of the case's own block rescue ('never' = not even the full clean set); bottom, all pairwise interactions.\n")
    if summary.get("reading"):
        L.append(summary["reading"] + "\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="", help="comma-separated subset run directories (default: every results/*_subsets that exists)")
    ap.add_argument("--synthetic", default="", help="<alllayer_run>:<layer> to fabricate a test subset_rows into --out")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-rebuild", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="test mode: do not write the summary/fragment into results/")
    args = ap.parse_args()
    if args.synthetic:
        global TAB, FIG
        TAB = FIG = args.out  # keep test outputs out of results/
        run, layer = args.synthetic.split(":")
        p = make_synthetic(run, int(layer), args.out)
        log("synthetic subset_rows written to", p)
        dirs = [args.out]
    elif args.dirs:
        dirs = args.dirs.split(",")
    else:
        dirs = [os.path.join(RESULTS, d) for d in CANDIDATES if os.path.exists(os.path.join(RESULTS, d, "subset_rows.parquet"))]
    if not dirs:
        log("no subset_rows.parquet found in", ", ".join(f"results/{d}" for d in CANDIDATES)); return
    summary = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": {}}
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        base_run, short, label, singletons = CANDIDATES.get(name, (None, None, None, {}))
        if base_run is None:  # synthetic / unknown dir: infer from the --synthetic argument or the name
            if args.synthetic:
                base_run = args.synthetic.split(":")[0]
                cfg = R.RUNS[base_run]; short = cfg["short"] + "_synthetic"; label = cfg["label"] + " [SYNTHETIC TEST DATA]"
                singletons = {l: [e] for l, e in (cfg["two_stage"], cfg["second"])}
            else:
                log("unknown subset dir", d, "skipped"); continue
        log("analysing", d)
        summary["runs"][name] = analyze_dir(d, base_run, short, label, singletons)
    try:
        from ext5_rank_text import F13_READING  # optional interpretive paragraph, added after the data were seen
        summary["reading"] = F13_READING
    except Exception:
        summary["reading"] = ""
    if args.no_write or args.synthetic:
        log("test mode: summary not written"); print(json.dumps(jsonable(summary), indent=1)[:3000]); return
    with open(SUMMARY, "w") as f:
        json.dump(jsonable(summary), f, indent=1)
    write_fragment(summary, os.path.join(TAB, "ext5_rank_subsets_section.md"))
    if not args.no_rebuild:
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext5_rank_analyze.py"), "--section-only"], check=True)
    log("done")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
