"""Phase-6 R-causal analysis (p6-analysis, CPU): does knocking out the top mover heads / top experts move the attention
share, beyond matched random knockouts? (predictions R3, R4; runner scripts/ext14_causal_run.py)

Per cell and model, every finished condition (none, heads_top3, heads_r0-2, experts_top3, experts_r0-2) is read as a
pair run (results/p6_causal_<cell>_<proto>/<cond>, validation pairs) and all conditions are summarised on the SAME
cluster resamples (unit = WinoGrande twin, 5,000 resamples; moetrace.ext14_analysis.joint), so every contrast is paired:
  rel_attn = AUC+(attention)_ko / AUC+(attention)_none - 1      (raw logit AUC+, the drop changes with the knockout)
  rel_moe  = AUC+(MoE)_ko / AUC+(MoE)_none - 1
  d_share  = share_ko - share_none;  rel_margin = clean margin_ko / clean margin_none - 1;  d_A_direct, d_M_all
  vs random = top-3 value minus the mean of the three matched random draws.
R3 (heads): rel_attn(top) - rel_attn(random) <= -0.20 and d_share(top) - d_share(random) <= -0.05, both with the CI's
upper end < 0. R4 (experts): rel_moe(top) - rel_moe(random) <= -0.20 with the CI's upper end < 0.
Writes results/p6_causal_summary.json and results/tables/ext14_r_causal.md|csv.

Usage: python scripts/ext14_causal_analyze.py
"""
import json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, pandas as pd
from moetrace import ext14_analysis as XA
from moetrace.models import RESULTS
import ext14_causal_run as CR

OUT = os.path.join(RESULTS, f"p6_causal_summary{CR.SUFFIX}.json")


def complete(cell, proto, cond):
    return all(CR.done(cell, proto, cond, s) for s in CR.STAGES)


def analyse(cell, proto):
    plan = json.load(open(os.path.join(CR.cdir(cell, proto), "plan.json")))
    e = XA.entry_of(cell, proto)
    conds = [c for c in CR.CONDS if complete(cell, proto, c)]
    if "none" not in conds or len(conds) < 2:
        return None
    ds = {c: XA.CellData(cell, proto, entry=e, run=CR.run_name(cell, proto, c)) for c in conds}
    j = XA.joint([ds[c] for c in conds])
    st = dict(zip(conds, j["stats"]))
    b = st["none"]
    out = {"n_items": j["n_items"], "n_units": j["n_units"], "plan": {k: plan[k] for k in ("heads", "experts")}, "conds": {}}
    der = {}
    for c in conds:
        s = st[c]
        dv = {"rel_attn": s["auc_attn_raw"] / b["auc_attn_raw"] - 1, "rel_moe": s["auc_moe_raw"] / b["auc_moe_raw"] - 1, "d_share": s["share"] - b["share"],
              "rel_margin": s["margin_clean"] / b["margin_clean"] - 1 if "margin_clean" in s else None,
              "d_A_direct": s["A_direct"] - b["A_direct"] if "A_direct" in s and "A_direct" in b else None,
              "d_M_all": s["M_all"] - b["M_all"] if "M_all" in s and "M_all" in b else None}
        der[c] = dv
        out["conds"][c] = {"mask": plan["masks"][c], **{k: XA.ci(v) for k, v in s.items() if k in ("share", "auc_attn", "auc_moe", "auc_attn_raw", "auc_moe_raw", "A_direct",
                                                                                                   "M_all", "margin_clean", "drop", "clean_top1")},
                           **{k: XA.ci(v) for k, v in dv.items() if v is not None and c != "none"}}
    tests = {}
    for kind, pid in (("heads", "R3"), ("experts", "R4")):
        top = f"{kind}_top3"
        rnd = [f"{kind}_r{i}" for i in range(3) if f"{kind}_r{i}" in der]
        if top not in der or not rnd:
            continue
        res = {"targets": [f"L{l}{'H' if kind == 'heads' else 'E'}{x if kind == 'heads' else f'{x:03d}'}" for l, x in plan[kind]["top3"]], "n_random": len(rnd)}
        for k in ("rel_attn", "rel_moe", "d_share", "rel_margin", "d_A_direct", "d_M_all"):
            if der[top].get(k) is None:
                continue
            rm = np.mean([der[r][k] for r in rnd], axis=0)
            res[k] = XA.ci(der[top][k])
            res[f"rand_{k}"] = XA.ci(rm)
            res[f"{k}_vs_rand"] = XA.ci(der[top][k] - rm)
        # names used by the section writer
        res["d_auc_attn_rel"], res["rand_d_auc_attn_rel"] = res.get("rel_attn"), res.get("rand_rel_attn")
        res["d_auc_moe_rel"], res["rand_d_auc_moe_rel"] = res.get("rel_moe"), res.get("rand_rel_moe")
        res["rand_d_share"], res["d_margin_rel"] = res.get("rand_d_share"), res.get("rel_margin")
        out[kind] = res
        if pid == "R3":
            a, s_ = res.get("rel_attn_vs_rand"), res.get("d_share_vs_rand")
            ok = bool(a and s_ and a[0] <= -0.20 and a[2] < 0 and s_[0] <= -0.05 and s_[2] < 0)
            tests[f"R3:{cell}/{proto}"] = {"pred": "R3", "verdict": "held" if ok else "failed", "observed": {"rel_attn_vs_rand": a, "d_share_vs_rand": s_},
                                           "targets": res["targets"]}
        else:
            a = res.get("rel_moe_vs_rand")
            ok = bool(a and a[0] <= -0.20 and a[2] < 0)
            tests[f"R4:{cell}/{proto}"] = {"pred": "R4", "verdict": "held" if ok else "failed", "observed": {"rel_moe_vs_rand": a}, "targets": res["targets"]}
    out["complete"] = len(conds) == len(CR.CONDS)
    return out, tests


def main():
    S = {"cells": {}, "tests": {}, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "definition": __doc__.split("Writes")[0].strip()}
    rows = []
    for cell in CR.CELLS:
        for proto in CR.PROTOS:
            if not os.path.exists(os.path.join(CR.cdir(cell, proto), "plan.json")):
                continue
            r = analyse(cell, proto)
            if not r:
                continue
            out, tests = r
            S["cells"][f"{cell}/{proto}"] = out
            S["tests"].update(tests)
            for kind in ("heads", "experts"):
                x = out.get(kind)
                if not x:
                    continue
                F = lambda v, d=2: XA.fr(v, d, True)  # noqa: E731
                rows.append({"Cell": XA.cell_label(cell), "Model": XA.SHORT.get(proto, proto), "Knockout": f"top-3 {kind}: {', '.join(x['targets'])}",
                             "Attention AUC+ change (top / random)": f"{F(x.get('rel_attn'))} / {F(x.get('rand_rel_attn'))}",
                             "MoE AUC+ change (top / random)": f"{F(x.get('rel_moe'))} / {F(x.get('rand_rel_moe'))}",
                             "Share change (top / random)": f"{F(x.get('d_share'))} / {F(x.get('rand_d_share'))}",
                             "Top − random: attention / MoE / share": f"{F(x.get('rel_attn_vs_rand'))} / {F(x.get('rel_moe_vs_rand'))} / {F(x.get('d_share_vs_rand'))}",
                             "Clean margin change (top / random)": f"{F(x.get('rel_margin'))} / {F(x.get('rand_rel_margin'))}",
                             "Verdict": (S["tests"].get(f"{'R3' if kind == 'heads' else 'R4'}:{cell}/{proto}") or {}).get("verdict", "")})
    with open(OUT, "w") as f:
        json.dump(XA_jsafe(S), f, indent=1)
    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(XA.TAB, exist_ok=True)
        with open(os.path.join(XA.TAB, f"ext14_r_causal{CR.SUFFIX}.md"), "w") as f:
            f.write("**R-causal: zero-mode knockout of the top-3 mover heads (discovery single-head rescue) or the top-3 experts (discovery "
                    "all-case single-expert rescue) in both prefill rows and every spawned row, then the T2 sweep and the T3 all-MoE joint "
                    "re-run; relative changes vs the unmasked baseline in the same code path; random = mean of 3 matched draws (heads: same "
                    "layers; experts: same layer, nearest final-position routing frequency); paired cluster bootstrap over twins (validation pairs)**\n\n")
            f.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
            for _, r in df.iterrows():
                f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
        df.to_csv(os.path.join(XA.TAB, f"ext14_r_causal{CR.SUFFIX}.csv"), index=False)
    try:
        figure(S)
    except Exception as ex:  # noqa: BLE001
        print(f"figure failed: {ex!r}")
    print(f"wrote {OUT}: {len(S['cells'])} cells, {len(S['tests'])} tests")


def figure(S):
    """Per cell: relative change of attention AUC+, MoE AUC+ and the share change for the top-3 knockout vs the mean of the
    matched random knockouts (heads panel, experts panel)."""
    if not S["cells"]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(S["cells"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.9 + 0.75 * len(keys) * 1.6), squeeze=False)
    metr = (("rel_attn", "attention AUC+ (rel.)", "#2a78d6"), ("rel_moe", "MoE AUC+ (rel.)", "#eb6834"), ("d_share", "share (abs.)", "#1baf7a"))
    for ax, kind in zip(axes[0], ("heads", "experts")):
        y = 0
        ticks, labs = [], []
        for k in keys:
            x = S["cells"][k].get(kind)
            if not x:
                continue
            for j, (m, lab_, col) in enumerate(metr):
                v, r = x.get(m), x.get(f"rand_{m}")
                yy = y + j * 0.25
                if v:
                    ax.errorbar([v[0]], [yy], xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", color=col, ms=5, capsize=2, label=lab_ if y == 0 else None)
                if r:
                    ax.plot([r[0]], [yy], marker="|", color=col, ms=12, mew=2)
            ticks.append(y + 0.25)
            labs.append(f"{XA.cell_label(k.split('/')[0])}\n{XA.SHORT.get(k.split('/')[1], k)}")
            y += 1
        ax.axvline(0, color="#52514e", lw=0.8)
        ax.invert_yaxis()
        ax.set_yticks(ticks)
        ax.set_yticklabels(labs, fontsize=8)
        ax.set_title(f"top-3 {kind} knocked out (● top-3 with 95% CI, | mean of 3 matched random)", fontsize=9, loc="left")
        ax.set_xlabel("change vs no knockout")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(XA.FIG, f"ext14_r_causal{CR.SUFFIX}.{ext}"), bbox_inches="tight", dpi=150 if ext == "png" else None)
    plt.close(fig)


def XA_jsafe(o):
    import ext14_all as A
    return A.jsafe(o)


if __name__ == "__main__":
    main()
