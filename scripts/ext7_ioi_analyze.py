"""ext7-controls W8 analysis (CPU): IOI under STR on Qwen3 and Mixtral (BOS), two corruption sites.

Runs (ext7 generic pair runner, moetrace/ext7_pairs.py): results/ioi_<proto>_<corr> (W2 final-position sweep, W5 heads,
W4 joint decomposition when present), results/ioi_<proto>_<corr>_grid (W3 grid on the first 64 validation items),
results/ioi_<proto>/attn_names.npz (final-position attention of every head on IO / S1 / S2). corr = s2io ((i) S2 -> IO,
both directions) or s1io ((ii) S1 and IO -> other names, d = 0 only).

Outputs results/tables/ext7_controls_ioi_{w2,heads,grid}.{md,csv}, ext7_controls_ioi_curves.csv,
results/figures/ext7_controls_ioi_{curves,grid}.png, key "ioi" of results/ext7_controls_summary.json
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
CORR = {"s2io": "IOI (i) S2 -> IO", "s1io": "IOI (ii) S1, IO -> other names"}
KL = {"attn_layer": "attention", "layer": "MoE", "block": "block"}


def ci(s, d=3):
    return f"{s['mean']:+.{d}f} [{s['ci_lo']:+.{d}f}, {s['ci_hi']:+.{d}f}]"


def rci(r, d=3):
    return f"{r[0]:.{d}f} [{r[1]:.{d}f}, {r[2]:.{d}f}]"


def has(run, f):
    return os.path.exists(os.path.join(RESULTS, run, f))


def grid_summary(run: str) -> pd.DataFrame:
    """Per (position class, kind, layer): pair-mean rescue summed over the class's positions / pair-mean drop
    (directed cases of the grid set; classes from scripts/ext7_ioi_grid.py: S1, IO, S2, after_S2, final)."""
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
    summ, w2rows, curves, headrows, gridrows = {}, [], [], [], []
    for proto in C.PROTOS:
        names = np.load(os.path.join(RESULTS, f"ioi_{proto}", "attn_names.npz")) if has(f"ioi_{proto}", "attn_names.npz") else None
        for corr in ("s2io", "s1io"):
            run = f"ioi_{proto}_{corr}"
            if not (has(run, "str_sweep_rows.parquet") and has(run, "sweep_cases.parquet")):
                continue
            if not json.load(open(os.path.join(RESULTS, run, "run_meta.json"))).get("sweep", {}).get("complete"):
                continue
            key = f"{proto}:{corr}"
            s = C.pair_run_summary(run, "main")
            fams = P.load_families(run)
            if "rep" in fams and fams["rep"].get("discovery"):
                s["replication"] = C.pair_run_summary(run, "rep")
            summ[key] = s
            for fam, ss in (("main", s), ("rep", s.get("replication"))):
                if ss is None:
                    continue
                for k, pk in ss["peaks"].items():
                    w2rows.append({"model": LAB[proto], "corruption": CORR[corr], "family": fam, "component": KL[k], "L_sel": pk["L_sel"],
                                   "val": ci(pk["val_at_L_sel"]), "norm": rci(pk["norm_at_L_sel"]), "L_val": pk["L_val"],
                                   "auc_pos": pk["auc_pos"], "drop": ss["drop"]["mean"], "delta_clean": ss["delta_clean"]["mean"],
                                   "delta_corrupt": ss["delta_corrupt"]["mean"], "n_val_pairs": ss["n_val_pairs"],
                                   "share": rci((ss["share"]["share"], ss["share"]["share_lo"], ss["share"]["share_hi"]))})
            cv = C.pair_run_curves(run, "main", "validation")
            cv.insert(0, "model", LAB[proto]); cv.insert(1, "corruption", CORR[corr])
            curves.append(cv)
            # W5 heads
            if has(run, "head_rows.parquet"):
                hr = P.HeadRun(run)
                disc, val = fams["main"]["discovery"], fams["main"]["validation"]
                ht = P.head_table(hr, disc, val)
                ht.insert(0, "model", LAB[proto]); ht.insert(1, "corruption", CORR[corr])
                lay = {l: P.head_layer_summary(hr, l, disc, val) for l in hr.layers}
                # attention of the heads on IO / S1 / S2 (validation items, d = 0 view: clean prompt and the corrupted prompt)
                if names is not None:
                    pidx = list(names["pair_idx"])
                    vi = [pidx.index(c // 2) for c in val if c % 2 == 0 and (c // 2) in pidx]
                    M = names["mass"].astype(np.float32)[:, vi].mean(1)  # [3, L, H, 4]
                    kc = 1 if corr == "s2io" else 2
                    for col, (k, t) in {"clean_IO": (0, 0), "clean_S1": (0, 1), "clean_S2": (0, 2), "corr_IO": (kc, 0), "corr_S1": (kc, 1),
                                        "corr_S2": (kc, 2)}.items():
                        ht[col] = [float(M[k, int(l), int(h), t]) for l, h in zip(ht.layer, ht["head"])]
                # final-position attention mass on the final token itself and on position 0 (runner npz, head layers only)
                v0 = [c for c in val if c % 2 == 0]
                fin, p0 = {}, {}
                for l in hr.layers:
                    oc, _on = P.attention_mass(hr, l, v0)
                    for h in range(oc.shape[1]):
                        fin[(l, h)] = float(oc[:, h, P.HEAD_CLASSES.index("final")].mean())
                        p0[(l, h)] = float(oc[:, h, P.HEAD_CLASSES.index("pos0")].mean())
                ht["clean_final"] = [fin.get((int(l), int(h)), np.nan) for l, h in zip(ht.layer, ht["head"])]
                ht["clean_pos0"] = [p0.get((int(l), int(h)), np.nan) for l, h in zip(ht.layer, ht["head"])]
                headrows.append(ht)
                s["heads"] = {"layers": hr.layers, "per_layer": lay, "detected_2sd": ht[ht.detected_2sd][["layer", "head", "val_mean", "z_val", "z_disc", "spec"] +
                              [c for c in ("clean_IO", "clean_S1", "clean_S2", "corr_IO", "corr_S1", "clean_final", "clean_pos0") if c in ht.columns]].to_dict("records"),
                              "top5": ht.head(5)[["layer", "head", "val_mean", "val_lo", "val_hi", "z_val", "share_attn_layer"] +
                                                 [c for c in ("clean_IO", "clean_S1", "clean_S2", "corr_IO", "corr_S1", "clean_final", "clean_pos0") if c in ht.columns]].to_dict("records")}
                if names is not None:  # name-mover-like heads over ALL layers by attention alone (clean IO mass >= 0.3)
                    M0 = names["mass"].astype(np.float32).mean(1)
                    io = M0[0, :, :, 0]
                    top = np.dstack(np.unravel_index(np.argsort(-io.ravel())[:8], io.shape))[0]
                    s["heads"]["attn_to_IO_top8_all_layers"] = [{"layer": int(l), "head": int(h), "clean_IO": float(io[l, h]),
                                                                 "clean_S1": float(M0[0, l, h, 1]), "clean_S2": float(M0[0, l, h, 2]),
                                                                 "s2io_IO": float(M0[1, l, h, 0]), "s2io_S1": float(M0[1, l, h, 1])} for l, h in top]
            # W3 grid
            grun = f"{run}_grid"
            if has(grun, "str_grid_w1_rows.parquet") and json.load(open(os.path.join(RESULTS, grun, "run_meta.json"))).get("grid_w1", {}).get("complete"):
                gs = grid_summary(grun)
                gs.insert(0, "model", LAB[proto]); gs.insert(1, "corruption", CORR[corr])
                gridrows.append(gs)
                pk = gs.loc[gs.groupby(["cls", "kind"])["norm"].idxmax()]
                s["grid_peaks"] = pk[["cls", "kind", "layer", "norm", "mean"]].to_dict("records")
                s["grid_layer_sums"] = gs[gs.norm > -9].groupby(["cls", "kind"]).norm.sum().round(3).reset_index().to_dict("records")
            # W4 joint
            if has(run, "joint_rows.parquet"):
                s["joint"] = C.joint_summary(run, "main")
            if has(run, "direct_split.parquet"):
                s["direct"] = C.direct_summary(run, "main")
    if w2rows:
        W = pd.DataFrame(w2rows)
        W.to_csv(os.path.join(TAB, "ext7_controls_ioi_w2.csv"), index=False)
        L = ["| Model | Corruption | Family | Component | L* | Val. rescue at L* [95% CI] | / drop | Val. argmax | AUC+ | Mean drop (Δ clean / Δ corrupt) | Attention share (AUC+) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in W.itertuples():
            L.append(f"| {r.model} | {r.corruption} | {r.family} | {r.component} | L{r.L_sel} | {r.val} | {r.norm} | L{r.L_val} | {r.auc_pos:.2f} | "
                     f"{r.drop:+.2f} ({r.delta_clean:+.2f} / {r.delta_corrupt:+.2f}) | {r.share} |")
        open(os.path.join(TAB, "ext7_controls_ioi_w2.md"), "w").write("\n".join(L) + "\n")
    if curves:
        CV = pd.concat(curves, ignore_index=True)
        CV.to_csv(os.path.join(TAB, "ext7_controls_ioi_curves.csv"), index=False)
        combos = [(p, c) for p in C.PROTOS for c in ("s2io", "s1io") if ((CV.model == LAB[p]) & (CV.corruption == CORR[c])).any()]
        fig, axes = plt.subplots(1, len(combos), figsize=(4.6 * len(combos), 3.6), squeeze=False)
        col = {"attn_layer": "#1f77b4", "layer": "#ff7f0e", "block": "#2ca02c"}
        for ax, (p, c) in zip(axes[0], combos):
            for k in ("attn_layer", "layer", "block"):
                x = CV[(CV.model == LAB[p]) & (CV.corruption == CORR[c]) & (CV.kind == k)]
                ax.plot(x.layer, x.norm, color=col[k], label=KL[k])
                ax.fill_between(x.layer, x.norm_lo, x.norm_hi, color=col[k], alpha=0.15, lw=0)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_title(f"{LAB[p]}\n{CORR[c]}", fontsize=9)
            ax.set_xlabel("layer"); ax.set_ylabel("rescue / mean drop (val.)")
            ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "ext7_controls_ioi_curves.png"), dpi=130)
    if headrows:
        H = pd.concat(headrows, ignore_index=True)
        H.to_csv(os.path.join(TAB, "ext7_controls_ioi_heads.csv"), index=False)
        L = ["| Model | Corruption | Head | Val. rescue [95% CI] | z (val / disc) | Spec | Share of attn layer | Clean attention IO / S1 / S2 / final / pos0 | Corrupted attention IO / S1 |",
             "|---|---|---|---|---|---|---|---|---|"]
        for (m, c), g in H.groupby(["model", "corruption"], sort=False):
            g = g[(g.detected_2sd) | (g.index.isin(g.head(5).index))].head(10)
            for r in g.itertuples():
                att = (f"{r.clean_IO:.2f} / {r.clean_S1:.2f} / {r.clean_S2:.2f} / {r.clean_final:.2f} / {r.clean_pos0:.2f} | {r.corr_IO:.2f} / {r.corr_S1:.2f}"
                       if hasattr(r, "clean_IO") else "n/a | n/a")
                L.append(f"| {m} | {c} | L{r.layer}H{r.head}{' (2SD)' if r.detected_2sd else ''} | {r.val_mean:+.3f} [{r.val_lo:+.3f}, {r.val_hi:+.3f}] | "
                         f"{r.z_val:+.1f} / {r.z_disc:+.1f} | {r.spec:+.3f} | {r.share_attn_layer:.2f} | {att} |")
        open(os.path.join(TAB, "ext7_controls_ioi_heads.md"), "w").write("\n".join(L) + "\n")
    if gridrows:
        G = pd.concat(gridrows, ignore_index=True)
        G.to_csv(os.path.join(TAB, "ext7_controls_ioi_grid.csv"), index=False)
        L = ["| Model | Corruption | Position | Kind | Peak layer | Peak / drop | Layer sum / drop |", "|---|---|---|---|---|---|---|"]
        for (m, c), g in G.groupby(["model", "corruption"], sort=False):
            for (cl, k), gg in g.groupby(["cls", "kind"]):
                j = gg.norm.idxmax()
                L.append(f"| {m} | {c} | {cl} | {KL.get(k, k)} | L{int(gg.loc[j, 'layer'])} | {gg.loc[j, 'norm']:+.3f} | {gg.norm.sum():+.3f} |")
        open(os.path.join(TAB, "ext7_controls_ioi_grid.md"), "w").write("\n".join(L) + "\n")
        combos = list(G.groupby(["model", "corruption"], sort=False).groups)
        fig, axes = plt.subplots(len(combos), 2, figsize=(10, 2.6 * len(combos)), squeeze=False)
        for i, (m, c) in enumerate(combos):
            for j, k in enumerate(("layer", "attn_layer")):
                g = G[(G.model == m) & (G.corruption == c) & (G.kind == k)]
                piv = g.pivot(index="cls", columns="layer", values="norm").reindex([x for x in ("S1", "IO", "S2", "after_S2", "final") if x in set(g.cls)])
                ax = axes[i, j]
                v = np.nanmax(np.abs(piv.values)) if piv.size else 1
                im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
                ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7)
                ax.set_title(f"{m}, {c}: {KL[k]} at p (rescue / drop)", fontsize=8)
                ax.set_xlabel("layer", fontsize=7)
                fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "ext7_controls_ioi_grid.png"), dpi=120)
    s0 = json.load(open(C.SUMMARY)) if os.path.exists(C.SUMMARY) else {}
    s0["ioi"] = summ
    s0["ioi_data"] = json.load(open(os.path.join(C.IOI_DIR, "pool.json"))) if os.path.exists(os.path.join(C.IOI_DIR, "pool.json")) else None
    json.dump(s0, open(C.SUMMARY, "w"), indent=1, default=float)
    for f in ("w2", "heads", "grid"):
        p = os.path.join(TAB, f"ext7_controls_ioi_{f}.md")
        if os.path.exists(p):
            print(open(p).read())


if __name__ == "__main__":
    main()
