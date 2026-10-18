"""Phase-6 tier 2 (p6-analysis): two-stage expert selection on a tier-2 cell (CPU).

Reads the expert singles of results/p6_<cell>_<proto>_t2/ (scripts/ext7_wino_expert.py --out <t2 run>) and runs the
Phase-3 W6 analysis unchanged (scripts/ext7_wino_analyze.w6: layer = discovery argmax of the final-position MoE curve,
recurrence gate = half of the discovery directed cases, Spec / equal-norm control / joint search / fixed experts), then
writes selection.json: selected_expert (scripts/ext13_p5_analyze.selected), pop_top10 (discovery all-case single-expert
rescue, ext8 population ranking) and the compact W6 summary.

Usage: python scripts/ext14_t2_select.py <cell> <proto>
"""
import json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import ext14_t2_common as T2
from moetrace import ext7_pairs as P

FIXED = {"qwen3": [(41, 117), (44, 69), (42, 115), (37, 32)], "mixtral": [(20, 0), (19, 2), (21, 1), (18, 1), (19, 6), (20, 1)],
         "qwen2_57b": []}


def main():
    cell, proto = sys.argv[1], sys.argv[2]
    import ext7_wino_analyze as W
    import ext13_p5_analyze as A5
    run = T2.t2_run(cell, proto)
    model = T2.T2_MODEL[proto]
    cfg = {"run": run, "model": model, "label": T2.XA.LABEL.get(proto, proto), "cf_experts": FIXED.get(model, [])}
    pr = P.PairRun(run)
    W.STRATA = [(c, lab) for c, lab in W.STRATA if c in pr.cases.columns]
    t0 = time.time()
    w6 = W.w6(pr, cfg, {})
    if w6 is None:
        raise SystemExit(f"{run}: no str_expert_rows.parquet")
    sel = A5.selected({"w6": w6})
    pop = A5.pop_top10(cfg, pr)
    out = {"cell": cell, "proto": proto, "run": run, "selected_expert": sel, "pop_top10": pop,
           "w6": A5.clean({f: v for f, v in w6.items()}), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "two-stage selection (ext7_wino_analyze.w6) on the tier-2 expert singles; pop_top10 = discovery all-case single-expert rescue"}
    with open(os.path.join(T2.t2_dir(cell, proto), "selection.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    s = sel or {}
    print(f"{cell}/{proto}: selected {s.get('name')} (layer L{s.get('layer')}, pattern {s.get('pattern')}, spec {s.get('spec')}); "
          f"pop top-3 {pop[:3] if pop else None} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
