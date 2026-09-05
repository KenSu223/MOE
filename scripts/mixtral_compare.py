"""Compare Mixtral runs (BOS default, no-BOS, paper_like token rule) at L19 and at each run's selected layer.
Usage: python scripts/mixtral_compare.py [run_name ...]   (default: mixtral mixtral_nobos mixtral_alt)
Writes results/mixtral_compare.json and prints a markdown summary."""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import analysis as A
from moetrace.models import MODELS, RESULTS

PAPER_E = 6
runs = sys.argv[1:] or ["mixtral", "mixtral_nobos", "mixtral_alt"]
out = {}
for name in runs:
    if not os.path.exists(os.path.join(RESULTS, name, "sweep_rows.parquet")):
        print(f"[skip] {name}: no sweep_rows"); continue
    md = A.load_model(name)
    la = A.layer_analysis(md, "paper")
    disc, val = md.ids("paper", "discovery"), md.ids("paper", "validation")
    R = md.R
    rec = {"run": name, "L_star": la["L_star"], "val_at_Lstar": la["val_at_Lstar"]["mean"], "val_at_Lstar_ci": [la["val_at_Lstar"]["ci_lo"], la["val_at_Lstar"]["ci_hi"]],
           "val_at_L19": float(R.loc[val, 19].mean()), "disc_at_L19": float(R.loc[disc, 19].mean()),
           "sharpness": la["sharpness"], "disc_top5": la["disc_curve_top5"],
           "paper_ids_passing_strict": f"{int(md.cases.strict.sum())}/{len(md.cases)}",
           "mean_delta_clean": float(md.cases.delta_clean.mean()) if "delta_clean" in md.cases else None,
           "mean_drop": float((md.cases.delta_clean - md.cases.delta_noised).mean()) if "delta_noised" in md.cases else None}
    if md.expert_rows is not None:
        for L in sorted({19, la["L_star"]} & set(md.expert_rows.layer)):
            et = A.expert_table(md, L)
            # per-expert activity and all-case means
            pe = []
            for e in range(8):
                sub = et[et.expert == e]
                d_act = int(sub.case_id.isin(disc).sum()); v_act = int(sub.case_id.isin(val).sum())
                d_mean = float(sub[sub.case_id.isin(disc)].rescue.sum() / len(disc)); v_mean = float(sub[sub.case_id.isin(val)].rescue.sum() / len(val))
                pe.append({"expert": e, "disc_active": d_act, "val_active": v_act, "disc_allcase_mean": round(d_mean, 3), "val_allcase_mean": round(v_mean, 3)})
            sel = A.select_expert(et, disc, len(disc) // 2)
            evs = {}
            for e in sorted({sel["e_star"], PAPER_E} - {None}):
                ev = A.evaluate_expert(md, L, e, val, MODELS["mixtral"]["n_controls"])
                evs[f"E{e:03d}"] = {"val_active": ev["val_active"], "rescue_all": ev["rescue_all"]["mean"], "rescue_ci": [ev["rescue_all"]["ci_lo"], ev["rescue_all"]["ci_hi"]],
                                    "spec_all": ev["spec_all"]["mean"], "spec_ci": [ev["spec_all"]["ci_lo"], ev["spec_all"]["ci_hi"]],
                                    "control_all": ev["control_all"]["mean"], "rescue_active": ev["rescue_active"]["mean"], "spec_active": ev["spec_active"]["mean"]}
            co = A.coalitions(md, L, val)
            rec[f"L{L}"] = {"per_expert": pe, "selection": {k: v for k, v in sel.items() if k != "table"}, "eval": evs,
                            "coalition_clean": co["coalition_clean"]["mean"], "coalition_union": co["coalition_union"]["mean"]}
    out[name] = rec
json.dump(out, open(os.path.join(RESULTS, "mixtral_compare.json"), "w"), indent=1, default=str)
# markdown summary
print("| run | paper IDs strict | L* | val@L* | val@L19 | next layer | e* (cand.) | E002 disc/val act | E002 rescue / spec | E006 disc/val act | E006 rescue / spec | coalition top2 / union |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, r in out.items():
    l19 = r.get("L19", {})
    pe = {p["expert"]: p for p in l19.get("per_expert", [])}
    ev = l19.get("eval", {})
    def es(e):
        x = ev.get(f"E{e:03d}"); return f"{x['rescue_all']:+.3f} / {x['spec_all']:+.3f}" if x else "n/a"
    sel = l19.get("selection", {})
    print(f"| {name} | {r['paper_ids_passing_strict']} | L{r['L_star']} | {r['val_at_Lstar']:+.3f} | {r['val_at_L19']:+.3f} | L{r['sharpness']['next_layer']} {r['sharpness']['next_rescue']:+.3f} | "
          f"E{sel.get('e_star', -1):03d} ({sel.get('n_candidates', '?')}) | {pe.get(2, {}).get('disc_active', '?')}/{pe.get(2, {}).get('val_active', '?')} | {es(2)} | "
          f"{pe.get(6, {}).get('disc_active', '?')}/{pe.get(6, {}).get('val_active', '?')} | {es(6)} | {l19.get('coalition_clean', float('nan')):+.3f} / {l19.get('coalition_union', float('nan')):+.3f} |")
print("paper: E006 disc/val active 91/83; L19 val rescue +0.457; E006 rescue +0.099, spec -0.175; coalitions +0.461 / +0.490")
