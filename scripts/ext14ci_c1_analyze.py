"""ext14 C1 analysis (CPU; p6-circuit): neurons inside the selected experts, Mistral-7B dense analogue, upcycling check.

Reads results/p6ci_c1_<task>/ (scripts/ext14ci_c1_run.py) and results/p6ci_upcycle/upcycle.json (scripts/ext14ci_upcycle.py);
writes results/tables/ext14ci_c1_*.md, results/tables/ext14ci_upcycle_*.md, results/figures/ext14ci_c1_*.png and the
'c1' / 'upcycle' keys of results/ext14ci_summary.json. Every part is skipped when its inputs are missing.

Usage: python scripts/ext14ci_c1_analyze.py [--smoke]
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext14_circuit as C
from moetrace import ext14_circuit_an as AN
from moetrace.models import RESULTS

TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
RK_LABEL = {"dla_case": "DLA (per case)", "mag_case": "magnitude (per case)", "dla_pop": "DLA (fixed set, discovery)",
            "rand": "random"}


def fmt_k(k, ci=None):
    if k is None or (isinstance(k, float) and np.isnan(k)):
        return "never"
    s = f"{k:.0f}" if k >= 10 else f"{k:.1f}"
    if ci and ci[0] is not None:
        s += f" [{ci[0]:.0f}, {ci[1]:.0f}]"
    return s


def expert_runs(smoke=False):
    out = []
    for task, cfg in C.C1_TASKS.items():
        for (l, e) in cfg["experts"]:
            nm = C.ename(l, e)
            if os.path.exists(os.path.join(C.task_dir(task), f".done_patch_{nm}")):
                out.append((task, nm))
    return out


def curves_figure(sums, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(sums)
    if not n:
        return
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows), squeeze=False)
    colors = {"dla_case": "#1f5fa8", "dla_pop": "#6aa0d8", "mag_case": "#d07a1f", "rand": "#8a8a8a"}
    for ax, s in zip(axs.flat, sums):
        for rk in ("dla_case", "dla_pop", "mag_case", "rand"):
            c = s["curves"][rk]
            ks = np.array(c["k"], float)
            ax.plot(ks, c["r"], color=colors[rk], lw=1.6 if rk == "dla_case" else 1.2, label=RK_LABEL[rk],
                    ls="--" if rk == "rand" else "-")
            ax.fill_between(ks, c["r_lo"], c["r_hi"], color=colors[rk], alpha=0.12, lw=0)
        ax.axhline(0.8, color="k", lw=0.6, ls=":")
        ax.axhline(1.0, color="k", lw=0.6)
        ax.set_xscale("log")
        top = max(max(s["curves"][rk]["r"]) for rk in ("dla_case", "dla_pop", "mag_case"))
        ax.set_ylim(-0.3, min(4.0, max(1.6, 1.05 * top)))
        ttl = f"{s['expert']} ({s['task'].split('_', 1)[1]})"
        ax.set_title(f"{s['model_label']}\n{ttl}", fontsize=8.5)
        ax.set_xlabel("neurons patched (k)", fontsize=8)
        ax.set_ylabel("fraction of expert rescue", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def mapping(tm, nm_m, td, nm_d, split="validation"):
    """Per-neuron population DLA vectors of a Mixtral expert and the Mistral MLP of the same layer on the same items,
    compared under the identity neuron mapping (the upcycling check finds best match = same index)."""
    from scipy.stats import spearmanr
    res = {}
    vecs = {}
    for task, nm in ((tm, nm_m), (td, nm_d)):
        try:
            f = C.load_full(task, nm)
        except FileNotFoundError:
            return None
        R = pd.read_parquet(os.path.join(C.task_dir(task), "c1_rows.parquet"))
        cfg = C.C1_TASKS[task]
        R = R[R.split == split]
        if cfg.get("margin") is not None:
            ok = (R.delta_clean >= 1) & (R.delta_corrupt <= -1)
            if cfg["items"][0] == "wino":
                ok = ok & ok.groupby(R.unit).transform("all")
            R = R[ok]
        pos = {int(r): j for j, r in enumerate(f["row_id"])}
        D = f["dla"][[pos[int(r)] for r in R.row_id]].astype(np.float32)
        cm = pd.DataFrame(D).groupby(R.case_id.to_numpy()).mean()
        vecs[task] = (cm, R)
    (cm_m, _), (cm_d, _) = vecs[tm], vecs[td]
    common = cm_m.index.intersection(cm_d.index)
    a, b = cm_m.loc[common].mean(0).to_numpy(), cm_d.loc[common].mean(0).to_numpy()
    res["n_cases_common"] = int(len(common))
    res["spearman"] = float(spearmanr(a, b).correlation)
    res["pearson"] = float(np.corrcoef(a, b)[0, 1])
    I = len(a)
    for k in (10, 50, 200):
        ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
        res[f"top{k}_overlap"] = len(ta & tb)
        res[f"top{k}_expected"] = round(k * k / I, 2)
    # rank of the Mixtral expert's top-10 neurons in the Mistral MLP ranking (identity map)
    rb = np.empty(I, int)
    rb[np.argsort(-b)] = np.arange(1, I + 1)
    res["mixtral_top10_rank_in_mistral"] = [int(rb[i]) for i in np.argsort(-a)[:10]]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        C.DIR_SUFFIX = "_smoke"
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    sums = []
    for task, nm in expert_runs(a.smoke):
        er = AN.ExpertRun(task, nm)
        s = er.summary(n_boot=300 if a.smoke else AN.N_BOOT)
        s["model"] = C.C1_TASKS[task]["model"]
        s["model_label"] = C.C1_TASKS[task]["label"]
        sums.append(s)
        print(task, nm, "k80", {rk: s["curves"][rk]["k80"] for rk in s["curves"]}, "direct", round(s["direct_share"], 3))
    suffix = "_smoke" if a.smoke else ""
    if sums:
        lines = ["| model, task | expert | neurons | cases (active rows) | expert rescue T [95% CI] | T / drop | direct share (sum DLA / T) "
                 "| k50 DLA | k80 DLA [95% CI] | k80 fixed set | k80 magnitude | k80 random | per-case k80 median (IQR; n) |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in sums:
            c = s["curves"]
            pc = s["per_case_k80"]
            lines.append(f"| {s['model_label']} | {s['expert']} | {s['inter']} | {s['n_cases']} ({s['n_active_rows']}) | "
                         f"{s['T_mean']:+.3f} [{s['T_ci'][0]:+.3f}, {s['T_ci'][1]:+.3f}] | {s['T_over_drop']:.3f} | {s['direct_share']:.2f} | "
                         f"{fmt_k(c['dla_case']['k50'])} | {fmt_k(c['dla_case']['k80'], c['dla_case']['k80_ci'])} | "
                         f"{fmt_k(c['dla_pop']['k80'])} | {fmt_k(c['mag_case']['k80'])} | {fmt_k(c['rand']['k80'])} | "
                         f"{fmt_k(pc['median'])} ({fmt_k(pc['q25'])}-{fmt_k(pc['q75'])}; {pc['n']}) |")
        lines += ["", "k50 / k80 = the number of neurons whose joint patch (set to the clean values inside the corrupted run, final "
                  "position, downstream recomputed) restores 50 / 80 % of the expert's own single-expert patch (ratio of case means, "
                  "validation split; log-k interpolation; 'never' = not reached before all neurons). DLA = the row's own neuron DLA "
                  "ranking; fixed set = one neuron order from the discovery-split mean DLA; magnitude = |w a change| x |W_down column|.",
                  "Per-case k80 on cases with T >= 0.25 logits."]
        open(os.path.join(TAB, f"ext14ci_c1_neurons{suffix}.md"), "w").write("\n".join(lines) + "\n")
        # agreement + overlap table
        L2 = ["| expert | task | sum DLA vs expert patch r | subset DLA vs patch r (slope) | single-neuron r (slope; sign agree; n) | "
              "top-10 Jaccard across cases (random) | top-50 Jaccard | neurons in >= half of cases' top-10 | pos. DLA mass in top-10 / 50 neurons |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s in sums:
            g, o = s["agreement"], s["overlap"]
            k10, k50 = o.get("k10", {}), o.get("k50", {})
            pm = o.get("pos_mass_top", {})
            L2.append(f"| {s['expert']} | {s['task']} | {_f(g['expert_r'])} | {_f(g['subset_r'])} ({_f(g['subset_slope'])}) | "
                      f"{_f(g['single_r'])} ({_f(g['single_slope'])}; {_f(g['single_sign_agree'])}; {g['single_n']}) | "
                      f"{_f(k10.get('jaccard_mean'))} ({_f(k10.get('rand_jaccard'), 4)}) | {_f(k50.get('jaccard_mean'))} | "
                      f"{k10.get('n_in_half', '')} | {_f(pm.get('10'))} / {_f(pm.get('50'))} |")
        open(os.path.join(TAB, f"ext14ci_c1_agreement{suffix}.md"), "w").write("\n".join(L2) + "\n")
        curves_figure(sums, os.path.join(FIG, f"ext14ci_c1_curves{suffix}.png"))
    # exact single-neuron oracle (Qwen3 experts)
    oracles = {}
    for task, cfg in C.C1_TASKS.items():
        for (l, e) in cfg["experts"]:
            o = AN.oracle_summary(task, C.ename(l, e))
            if o:
                oracles[f"{task}|{C.ename(l, e)}"] = o
    if oracles:
        L3 = ["| task | expert | rows (cases) | k80 oracle | k80 DLA | k80 magnitude | k80 fixed set | k80 random | single-neuron DLA vs exact r (slope) | "
              "sum of singles / expert patch | neurons for 80 % of summed abs. single effects (median) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, o in oracles.items():
            t_, nm_ = key.split("|")
            c = o["curves"]
            L3.append(f"| {t_} | {nm_} | {o['n_rows']} ({o['n_cases']}) | {fmt_k(c['oracle']['k80'])} | {fmt_k(c['dla_case']['k80'])} | "
                      f"{fmt_k(c['mag_case']['k80'])} | {fmt_k(c['dla_pop']['k80'])} | {fmt_k(c['rand0']['k80'])} | {_f(o['single_r_dla_vs_exact'])} "
                      f"({_f(o['single_slope'])}) | {_f(o['additivity_sum_singles_over_ref'])} | {o['abs_mass_k80_median']} |")
        open(os.path.join(TAB, f"ext14ci_c1_oracle{suffix}.md"), "w").write("\n".join(L3) + "\n")
    # dense sweeps
    dense = {}
    for task in ("mistral7b_cf", "mistral7b_wino"):
        if os.path.exists(os.path.join(C.task_dir(task), ".done_score")):
            dense[task] = AN.dense_sweep(task)
    # neuron mapping Mixtral expert <-> Mistral MLP (identity map)
    maps = {}
    pairs = [("mixtral_wino", "L20E000", "mistral7b_wino", "L20E000"), ("mixtral_cf", "L19E002", "mistral7b_cf", "L19E000"),
             ("mixtral_cf", "L21E001", "mistral7b_cf", "L21E000"), ("mixtral_cf", "L18E001", "mistral7b_cf", "L18E000")]
    for tm, nm_m, td, nm_d in pairs:
        if all(os.path.exists(os.path.join(C.task_dir(t), ".done_score")) for t in (tm, td)):
            r = mapping(tm, nm_m, td, nm_d)
            if r:
                maps[f"{nm_m}~{nm_d}"] = r
    up = None
    upp = os.path.join(RESULTS, "p6ci_upcycle", "upcycle.json")
    if os.path.exists(upp) and not a.smoke:
        up = json.load(open(upp))
        at = pd.DataFrame(up["attn"])
        lines = ["| layer | q cos | k cos | v cos | o cos | input norm cos | post-attn norm cos | q rel. diff |", "|---|---|---|---|---|---|---|---|"]
        for r in at.itertuples():
            lines.append(f"| {r.layer} | {r.q_cos:.3f} | {r.k_cos:.3f} | {r.v_cos:.3f} | {r.o_cos:.3f} | {r.ln1_cos:.3f} | {r.ln2_cos:.3f} | {r.q_reldiff:.3f} |")
        g = up["globals"]
        lines += ["", "Globals: " + ", ".join(f"{k} cos {v['cos']:.3f} (rel. diff {v['reldiff']:.3f})" for k, v in g.items())]
        open(os.path.join(TAB, "ext14ci_upcycle_attn.md"), "w").write("\n".join(lines) + "\n")
        ff = pd.DataFrame(up["ffn"])
        lines = ["| layer | expert | gate row cos (median) | up row cos | down col cos | best match = same index | mutual | "
                 "matched cos median (shuffled control) |", "|---|---|---|---|---|---|---|---|"]
        for r in ff.itertuples():
            idr = getattr(r, "identity_rate", np.nan)
            lines.append(f"| {r.layer} | {r.expert} | {r.gate_median:.3f} | {r.up_median:.3f} | {r.down_median:.3f} | "
                         + (f"{idr:.3f} | {r.mutual_rate:.3f} | {r.match_cos_median:.3f} ({r.shuffled_match_cos_median:.3f}) |"
                            if not (isinstance(idr, float) and np.isnan(idr)) else "| | |"))
        open(os.path.join(TAB, "ext14ci_upcycle_ffn.md"), "w").write("\n".join(lines) + "\n")
        up = {"attn_cos_median": {k: float(at[f"{k}_cos"].median()) for k in ("q", "k", "v", "o", "ln1", "ln2")},
              "attn_cos_min": {k: float(at[f"{k}_cos"].min()) for k in ("q", "k", "v", "o")},
              "globals": g,
              "ffn_gate_median_by_layer": ff.groupby("layer").gate_median.median().round(3).to_dict(),
              "match": ff.dropna(subset=["identity_rate"])[["layer", "expert", "identity_rate", "mutual_rate", "match_cos_median",
                                                              "shuffled_match_cos_median"]].to_dict("records")
              if "identity_rate" in ff.columns else []}
    if not a.smoke:
        C.update_summary("c1", {"experts": sums, "dense_sweep": dense, "mapping": maps, "oracle": oracles})
        if up:
            C.update_summary("upcycle", up)
    else:
        json.dump({"experts": sums, "dense": dense, "mapping": maps}, open(os.path.join(RESULTS, "ext14ci_summary_smoke.json"), "w"),
                  indent=1, default=C._json_default)
    print("analysis done:", len(sums), "experts;", "dense", list(dense), "maps", list(maps))


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


if __name__ == "__main__":
    main()
