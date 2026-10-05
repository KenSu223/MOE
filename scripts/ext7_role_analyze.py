"""ext7-controls W7 analysis (CPU): WinoGrande role swap (second STR site) vs the option swap of agent ext7-wino.

Runs: results/wino_role_<proto>_str (W2 sweep; W4 joint rows when present), results/wino_role_<proto>_grid (W3 grid,
first 64 validation pairs); option swap: results/wino_<proto>_str (ext7-wino's W2 run on data/wino_str/case_sets.json).
Fixed hypotheses = the option swap's discovery peak layers (MoE, attention, block), evaluated on the role-swap validation
pairs; for Mixtral (BOS) the role-swap pool is below 256 pairs, so all its pairs form one validation set and the
role-swap peaks are descriptive (validation argmax) while the fixed hypotheses carry the inference.

Outputs results/tables/ext7_controls_role_{w2,fixed,grid}.{md,csv}, ext7_controls_role_curves.csv,
results/figures/ext7_controls_role_{curves,grid}.png, key "role" of results/ext7_controls_summary.json
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from moetrace import ext7_controls as C
from moetrace import ext7_pairs as P
from moetrace.models import RESULTS

TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
LAB = {"qwen3": "Qwen3-30B-A3B-Base", "mixtral_bos": "Mixtral-8x7B, BOS"}
KL = {"attn_layer": "attention", "layer": "MoE", "block": "block"}


def ci(s, d=3):
    return f"{s['mean']:+.{d}f} [{s['ci_lo']:+.{d}f}, {s['ci_hi']:+.{d}f}]"


def rci(r, d=3):
    return f"{r[0]:.{d}f} [{r[1]:.{d}f}, {r[2]:.{d}f}]"


def has(run, f):
    return os.path.exists(os.path.join(RESULTS, run, f))


def role_grid(run: str) -> pd.DataFrame:
    """Per (position class, kind, layer): pair-mean rescue summed over the class's positions / pair-mean drop. Classes
    from scripts/ext7_role_grid.py: mention1 / mention2 (the two exchanged first mentions, in text order), filled (the
    option at the blank), final."""
    g = pd.read_parquet(os.path.join(RESULTS, run, "str_grid_w1_rows.parquet"))
    drop = g.groupby("case_id").apply(lambda x: float((x.delta_clean - x.delta_corrupt).iloc[0]))
    dpair = drop.groupby(np.asarray(drop.index) // 2).mean().mean()
    agg = g.groupby(["case_id", "cls", "kind", "layer"]).rescue.sum().reset_index()
    agg["pair"] = agg.case_id // 2
    pm = agg.groupby(["pair", "cls", "kind", "layer"]).rescue.mean().reset_index()
    out = pm.groupby(["cls", "kind", "layer"]).rescue.agg(["mean", "std", "count"]).reset_index()
    out["norm"] = out["mean"] / dpair
    out["mean_drop"] = dpair
    return out


def main():
    os.makedirs(TAB, exist_ok=True)
    summ, w2rows, fixrows, curves, gridrows = {}, [], [], [], []
    for proto in C.PROTOS:
        run, opt = f"wino_role_{proto}_str", f"wino_{proto}_str"
        if not (has(run, "str_sweep_rows.parquet") and has(run, "sweep_cases.parquet")):
            continue
        if not json.load(open(os.path.join(RESULTS, run, "run_meta.json"))).get("sweep", {}).get("complete"):
            continue
        so = C.pair_run_summary(opt, "main") if has(opt, "str_sweep_rows.parquet") else None
        fixed = {}
        if so:
            by_layer = {}
            for k in ("layer", "attn_layer", "block"):
                by_layer.setdefault(int(so["peaks"][k]["L_sel"]), []).append(KL[k])
            for l, nm in sorted(by_layer.items()):
                for k in ("layer", "attn_layer", "block"):
                    fixed[f"L{l} = option-swap {' / '.join(nm)} peak|{k}"] = (k, l)
        s = C.pair_run_summary(run, "main", fixed)
        fams = P.load_families(run)
        if "rep" in fams and fams["rep"].get("discovery"):
            s["replication"] = C.pair_run_summary(run, "rep", fixed)
        s["option_swap"] = so
        summ[proto] = s
        for name, (ss, src) in {"role swap": (s, "main"), "role swap (replication)": (s.get("replication"), "rep"),
                                "option swap (ext7-wino run)": (so, "main")}.items():
            if ss is None:
                continue
            for k, pk in ss["peaks"].items():
                w2rows.append({"model": LAB[proto], "corruption": name, "component": KL[k], "L_sel": pk["L_sel"], "selected_on": pk["selected_on"],
                               "val": ci(pk["val_at_L_sel"]), "norm": rci(pk["norm_at_L_sel"]), "L_val": pk["L_val"], "auc_pos": pk["auc_pos"],
                               "drop": ss["drop"]["mean"], "delta_clean": ss["delta_clean"]["mean"], "delta_corrupt": ss["delta_corrupt"]["mean"],
                               "n_disc_pairs": ss["n_disc_pairs"], "n_val_pairs": ss["n_val_pairs"],
                               "share": rci((ss["share"]["share"], ss["share"]["share_lo"], ss["share"]["share_hi"]))})
        for nm, fx in s.get("fixed", {}).items():
            fixrows.append({"model": LAB[proto], "hypothesis": nm.split("|")[0], "kind": KL[fx["kind"]], "layer": fx["layer"], "val": ci(fx["val"]),
                            "p": fx["val"].get("p"), "norm": rci(fx["norm"])})
        cv = C.pair_run_curves(run, "main", "validation")
        cv.insert(0, "model", LAB[proto]); cv.insert(1, "corruption", "role swap")
        curves.append(cv)
        if so:
            co = C.pair_run_curves(opt, "main", "validation")
            co.insert(0, "model", LAB[proto]); co.insert(1, "corruption", "option swap")
            curves.append(co)
        grun = f"wino_role_{proto}_grid"
        if has(grun, "str_grid_w1_rows.parquet") and json.load(open(os.path.join(RESULTS, grun, "run_meta.json"))).get("grid_w1", {}).get("complete"):
            gs = role_grid(grun)
            gs.insert(0, "model", LAB[proto])
            gridrows.append(gs)
            pk = gs.loc[gs.groupby(["cls", "kind"])["norm"].idxmax()]
            s["grid_peaks"] = pk[["cls", "kind", "layer", "norm", "mean"]].to_dict("records")
            s["grid_layer_sums"] = gs.groupby(["cls", "kind"]).norm.sum().round(3).reset_index().to_dict("records")
        if has(run, "joint_rows.parquet"):
            s["joint"] = C.joint_summary(run, "main")
        if has(run, "direct_split.parquet"):
            s["direct"] = C.direct_summary(run, "main")
    if w2rows:
        W = pd.DataFrame(w2rows)
        W.to_csv(os.path.join(TAB, "ext7_controls_role_w2.csv"), index=False)
        L = ["| Model | Corruption | Component | L* (selected on) | Val. rescue at L* [95% CI] | / drop | Val. argmax | AUC+ | Mean drop (Δ clean / Δ corrupt) | Pairs disc/val | Attention share (AUC+) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in W.itertuples():
            sel = "disc." if r.selected_on == "discovery" else "val., descriptive"
            L.append(f"| {r.model} | {r.corruption} | {r.component} | L{r.L_sel} ({sel}) | {r.val} | {r.norm} | L{r.L_val} | {r.auc_pos:.2f} | "
                     f"{r.drop:+.2f} ({r.delta_clean:+.2f} / {r.delta_corrupt:+.2f}) | {r.n_disc_pairs}/{r.n_val_pairs} | {r.share} |")
        open(os.path.join(TAB, "ext7_controls_role_w2.md"), "w").write("\n".join(L) + "\n")
    if fixrows:
        F = pd.DataFrame(fixrows)
        F.to_csv(os.path.join(TAB, "ext7_controls_role_fixed.csv"), index=False)
        L = ["| Model | Fixed layer (option-swap discovery peak) | Patched | Role-swap val. rescue [95% CI] | sign-flip p | / drop |", "|---|---|---|---|---|---|"]
        for r in F.itertuples():
            L.append(f"| {r.model} | {r.hypothesis} | {r.kind} | {r.val} | {r.p:.4f} | {r.norm} |")
        open(os.path.join(TAB, "ext7_controls_role_fixed.md"), "w").write("\n".join(L) + "\n")
    if curves:
        CV = pd.concat(curves, ignore_index=True)
        CV.to_csv(os.path.join(TAB, "ext7_controls_role_curves.csv"), index=False)
        models = list(dict.fromkeys(CV.model))
        fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 3.8), squeeze=False)
        col = {"attn_layer": "#1f77b4", "layer": "#ff7f0e", "block": "#2ca02c"}
        for ax, m in zip(axes[0], models):
            for k in ("attn_layer", "layer", "block"):
                for corr, ls in (("role swap", "-"), ("option swap", "--")):
                    x = CV[(CV.model == m) & (CV.corruption == corr) & (CV.kind == k)]
                    if x.empty:
                        continue
                    ax.plot(x.layer, x.norm, ls, color=col[k], lw=1.6 if ls == "-" else 1.1, label=f"{KL[k]} ({corr})")
                    if ls == "-":
                        ax.fill_between(x.layer, x.norm_lo, x.norm_hi, color=col[k], alpha=0.15, lw=0)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_title(f"{m}: WinoGrande, final position")
            ax.set_xlabel("layer"); ax.set_ylabel("rescue / mean drop (val.)")
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "ext7_controls_role_curves.png"), dpi=130)
    if gridrows:
        G = pd.concat(gridrows, ignore_index=True)
        G.to_csv(os.path.join(TAB, "ext7_controls_role_grid.csv"), index=False)
        L = ["| Model | Position class | Kind | Peak layer | Peak / drop | Layer sum / drop |", "|---|---|---|---|---|---|"]
        order = ["mention1", "mention2", "filled", "final"]
        for m, g in G.groupby("model", sort=False):
            for cl in [c for c in order if c in set(g.cls)]:
                for k in ("layer", "attn_layer"):
                    gg = g[(g.cls == cl) & (g.kind == k)]
                    if gg.empty:
                        continue
                    j = gg.norm.idxmax()
                    L.append(f"| {m} | {cl} | {KL[k]} | L{int(gg.loc[j, 'layer'])} | {gg.loc[j, 'norm']:+.3f} | {gg.norm.sum():+.3f} |")
        open(os.path.join(TAB, "ext7_controls_role_grid.md"), "w").write("\n".join(L) + "\n")
        models = list(dict.fromkeys(G.model))
        fig, axes = plt.subplots(len(models), 2, figsize=(10, 2.8 * len(models)), squeeze=False)
        for i, m in enumerate(models):
            for j, k in enumerate(("layer", "attn_layer")):
                g = G[(G.model == m) & (G.kind == k)]
                piv = g.pivot(index="cls", columns="layer", values="norm").reindex([c for c in order if c in set(g.cls)])
                ax = axes[i, j]
                v = np.nanmax(np.abs(piv.values)) if piv.size else 1
                im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
                ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7)
                ax.set_title(f"{m}, role swap: {KL[k]} at p (rescue / drop)", fontsize=8)
                ax.set_xlabel("layer", fontsize=7)
                fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "ext7_controls_role_grid.png"), dpi=120)
    s0 = json.load(open(C.SUMMARY)) if os.path.exists(C.SUMMARY) else {}
    s0["role"] = summ
    s0["role_data"] = {p: {"funnel": json.load(open(os.path.join(C.ROLE_DIR, f"funnel_{p}.json"))),
                           "scan": json.load(open(os.path.join(RESULTS, f"wino_role_{p}", "scan_summary.json"))) if has(f"wino_role_{p}", "scan_summary.json") else None,
                           "case_sets": {k: len(v) for k, v in json.load(open(os.path.join(C.ROLE_DIR, f"case_sets_{p}.json")))["pairs"].items()}
                           if os.path.exists(os.path.join(C.ROLE_DIR, f"case_sets_{p}.json")) else None} for p in C.PROTOS}
    json.dump(s0, open(C.SUMMARY, "w"), indent=1, default=float)
    for f in ("w2", "fixed", "grid"):
        p = os.path.join(TAB, f"ext7_controls_role_{f}.md")
        if os.path.exists(p):
            print(open(p).read())


if __name__ == "__main__":
    main()
