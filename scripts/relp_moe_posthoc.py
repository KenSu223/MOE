#!/usr/bin/env python3
"""Phase 7 post-hoc analyses (2026-11-08/09, from the follow-up questions; CPU only, existing results, no new GPU run).

1. picks: exact single-unit rescue (discovery and test means, % of the paired difference) of RelP's top-1 expert-position
   and of the expert chosen by Lu's two-stage rule (final position, recurrence gate).
2. lu_floor: Lu's two-stage rule re-run with the earliest layers excluded (floor 0 = original, 1, 3). In simple / within_rc the
   subject IS the final token, so the final-position rule otherwise lands on the subject's layer-0 MoE output (token identity;
   Zhang & Nanda 2024). Pure re-selection from the recorded patch grids.
3. consistency: mean Jaccard of the top-10 experts between the four tasks, RelP (whole experts) vs exact patching (any
   position / final position / layers >= 3), plus the top-3 per task.
4. coverage: patch-grid positions and unit counts per task.
5. top20_routing: routing status at the two endpoints of RelP's top-20 expert-positions on the test pairs (both / clean only /
   corrupt only / neither): how often a held-out denoise patch injects, removes or replaces a contribution (no renormalisation).
Writes results/relp_sva_moe/posthoc.json and results/tables/relp_moe_posthoc_*.md.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import relp_moe_analyze as A  # noqa: E402
from moetrace.analysis import evaluate_expert, expert_table, layer_analysis, select_expert  # noqa: E402

RES, TAB = ROOT / "results/relp_sva_moe", ROOT / "results/tables"
TASKS = ("simple", "nounpp", "rc", "within_rc")


def grid(m, t):
    b = RES / m / "paper" / t
    cols = ["split", "row", "direction", "kind", "layer", "pos", "expert", "effect", "active_A", "active_B"]
    return pd.concat([pd.read_parquet(p, columns=cols) for p in sorted((b / "patch_full").glob("part_*.parquet"))])


def unit_mean(d, split, kind, l, p, e=-1):
    x = d[(d.split == split) & (d.direction == "denoise") & (d.kind == kind)]
    n = x.row.nunique()
    return float(x[(x.layer == l) & (x.pos == p) & (x.expert == e)].effect.sum() / n)


def lu_with_floor(df, rows, T, model, floor):
    md = A.lu_modeldata(df, rows, T - 1)
    md.sweep_rows = md.sweep_rows[md.sweep_rows.layer >= floor]
    md.expert_rows = md.expert_rows[md.expert_rows.layer >= floor]
    la = layer_analysis(md, "sva")
    L = la["L_star"]
    disc = md.sets["sva"]["discovery"]
    e = select_expert(expert_table(md, L), disc, len(disc) // 2)["e_star"]
    te = evaluate_expert(md, L, e, md.sets["sva"]["test"], A.N_CONTROLS[model])["rescue_all"]["mean"] if e is not None else None
    return {"layer": L, "expert": e, "test_rescue": te}


def main():
    s = json.loads((RES / "summary.json").read_text())
    out = {"picks": {}, "lu_floor": {}, "consistency": {}, "coverage": {}, "top20_routing": {}}
    for m in ("qwen3", "mixtral"):
        relp_sets, ex_sets = {}, {"any": {}, "final": {}, "layers>=3": {}}
        route = {"both": 0, "clean_only": 0, "corrupt_only": 0, "neither": 0}
        for t in TASKS:
            key = f"{m}/paper/{t}"
            b = RES / m / "paper" / t
            d = grid(m, t)
            T = int(d.pos.max()) + 1
            gap = s["relp"][key]["discovery"]["paired_diff"]
            # 1. picks
            r1 = s["relp"][key]["top_expert_position"][0]
            lu = s["patch"][key]["lu"]
            L, E = lu["L_star"], lu["selection"]["e_star"]
            out["picks"][key] = {
                "gap": gap,
                "relp": {"unit": [r1["layer"], r1["pos"], r1["expert"]],
                         "disc": unit_mean(d, "discovery", "expert", r1["layer"], r1["pos"], r1["expert"]),
                         "test": unit_mean(d, "test", "expert", r1["layer"], r1["pos"], r1["expert"])},
                "lu": {"unit": [L, T - 1, E], "disc": unit_mean(d, "discovery", "expert", L, T - 1, E),
                       "test": unit_mean(d, "test", "expert", L, T - 1, E)}}
            # 2. Lu with layer floors
            df = A.load_patch(b)
            rows = {sp: A.manifest_rows(m, "paper", t, sp) for sp in ("discovery", "validation", "test")}
            out["lu_floor"][key] = {str(f): lu_with_floor(df, rows, T, m, f) for f in (0, 1, 3)}
            out["lu_floor"][key]["relp_top1"] = [r1["layer"], r1["expert"]]
            out["lu_floor"][key]["gap"] = gap
            # 3. consistency inputs
            relp_sets[t] = [tuple(c["coordinate"]) for c in json.loads((b / "score_zero/expert_all_positions_ranking.json").read_text())[:10]]
            x = d[(d.split == "discovery") & (d.direction == "denoise") & (d.kind == "expert")]
            g = (x.groupby(["layer", "pos", "expert"]).effect.sum() / x.row.nunique()).reset_index()
            ex_sets["any"][t] = [(r.layer, r.expert) for r in g.nlargest(10, "effect").itertuples()]
            ex_sets["final"][t] = [(r.layer, r.expert) for r in g[g.pos == T - 1].nlargest(10, "effect").itertuples()]
            ex_sets["layers>=3"][t] = [(r.layer, r.expert) for r in g[g.layer >= 3].nlargest(10, "effect").itertuples()]
            # 4. coverage
            plan = json.loads((b / "patch_full/plan.json").read_text())
            r0 = rows["discovery"][0]
            out["coverage"][key] = {"tokens": r0["tokens_input"], "changed": r0["changed_positions"],
                                    "patched_positions": plan["positions"], "pairs": plan["rows"], "units": plan["units"]}
            # 5. routing status of RelP's top-20 expert-positions on the test pairs
            top = [tuple(c["coordinate"]) for c in json.loads((b / "score_zero/expert_position_ranking.json").read_text())[:20]]
            xt = d[(d.split == "test") & (d.direction == "denoise") & (d.kind == "expert")].set_index(["row", "layer", "pos", "expert"])
            for r in xt.index.get_level_values(0).unique():
                for (l, p, e) in top:
                    if (r, l, p, e) in xt.index:
                        a_, b_ = xt.loc[(r, l, p, e), ["active_A", "active_B"]]
                        route["both" if a_ and b_ else "clean_only" if a_ else "corrupt_only"] += 1
                    else:
                        route["neither"] += 1
        jac = lambda D: [len(set(D[a]) & set(D[b])) / len(set(D[a]) | set(D[b])) for a, b in itertools.combinations(TASKS, 2)]
        out["consistency"][m] = {"relp_whole_experts": {"mean_jaccard": float(np.mean(jac(relp_sets))),
                                                        "top3": {t: relp_sets[t][:3] for t in TASKS}}}
        for k, D in ex_sets.items():
            out["consistency"][m][f"exact_{k}"] = {"mean_jaccard": float(np.mean(jac(D))), "top3": {t: D[t][:3] for t in TASKS}}
        n = sum(route.values())
        out["top20_routing"][m] = {k: v / n for k, v in route.items()} | {"n": n}
    (RES / "posthoc.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    write_tables(out)
    print("wrote", RES / "posthoc.json")


def write_tables(o):
    pct = lambda v, g: "" if v is None else f"{v:.2f} ({v / g:.0%})"
    lines = ["| model/task | paired difference | RelP top-1 (layer, position, expert) | exact rescue disc / test | Lu rule (layer, expert) | exact rescue disc / test |",
             "|---|---|---|---|---|---|"]
    for k, v in o["picks"].items():
        r, lu, g = v["relp"], v["lu"], v["gap"]
        lines.append(f"| {k} | {g:.2f} | L{r['unit'][0]} p{r['unit'][1]} E{r['unit'][2]} | {pct(r['disc'], g)} / {pct(r['test'], g)} | "
                     f"L{lu['unit'][0]} E{lu['unit'][2]} | {pct(lu['disc'], g)} / {pct(lu['test'], g)} |")
    (TAB / "relp_moe_posthoc_picks.md").write_text("\n".join(lines) + "\n")
    lines = ["| model/task | Lu rule, all layers (test rescue) | layers >= 1 | layers >= 3 | RelP top-1 |", "|---|---|---|---|---|"]
    for k, v in o["lu_floor"].items():
        g = v["gap"]
        cell = lambda x: f"L{x['layer']}E{x['expert']} {pct(x['test_rescue'], g)}"
        lines.append(f"| {k} | {cell(v['0'])} | {cell(v['1'])} | {cell(v['3'])} | L{v['relp_top1'][0]}E{v['relp_top1'][1]} |")
    (TAB / "relp_moe_posthoc_lu_floor.md").write_text("\n".join(lines) + "\n")
    lines = ["| model | list | mean Jaccard of top-10 between tasks | top-3 per task (simple; nounpp; rc; within_rc) |", "|---|---|---|---|"]
    for m, d in o["consistency"].items():
        for k, v in d.items():
            top = "; ".join(",".join(f"L{l}E{e}" for l, e in v["top3"][t]) for t in TASKS)
            lines.append(f"| {m} | {k} | {v['mean_jaccard']:.2f} | {top} |")
    (TAB / "relp_moe_posthoc_consistency.md").write_text("\n".join(lines) + "\n")
    lines = ["| model/task | tokens (A) | patched positions | pairs | patched units |", "|---|---|---|---|---|"]
    for k, v in o["coverage"].items():
        toks = [v["tokens"][i] for i in v["patched_positions"]]
        lines.append(f"| {k} | {' '.join(v['tokens'])} | {', '.join(toks)} | {v['pairs']} | {v['units']:,} |")
    lines += ["", "| model | RelP top-20 expert-positions on test pairs: routed at both | clean only (injected, no renormalisation) | corrupt only (removed) | neither (no-op) |",
              "|---|---|---|---|---|"]
    for m, v in o["top20_routing"].items():
        lines.append(f"| {m} | {v['both']:.0%} | {v['clean_only']:.0%} | {v['corrupt_only']:.0%} | {v['neither']:.0%} |")
    (TAB / "relp_moe_posthoc_coverage.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
