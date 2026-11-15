#!/usr/bin/env python3
"""Phase 8 S2.6 selection (docs/relp_moe_phase8_plan.md): the router-aware variant used in the add-back curves, chosen on
discovery pairs only: highest Pearson r between the variant's endpoint-B discovery means and the exact denoise discovery
means over all expert-position units, averaged over the four SVA tasks; one choice per model.
Writes results/relp_sva_moe/p8/selected_rw.json (all variants' r per task, for the record)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import relp_moe_p8_data as D  # noqa: E402

CANDIDATES = ["relp_rw", "relp_rw3", "relp_soft05", "relp_soft1", "relp_soft2"]
REFERENCE = ["relp", "grad", "grad_rw", "relp_nohalf", "relp_nonorm", "relp_noattn", "relp_nogate"]


def main():
    out = {}
    for model in ("qwen3", "mixtral"):
        per_task = {}
        for task in D.TASKS:
            d = D.task_dir(model, task)
            ex = D.exact_means(d)
            n = int(ex.n_pairs.iloc[0])
            sc = D.load_sparse(d / "p8_rules/scores_B.npz")
            vs = [c for c in CANDIDATES + REFERENCE if c in sc.columns]
            mu = sc.groupby(["layer", "pos", "expert"])[vs].sum() / n
            j = ex.set_index(["layer", "pos", "expert"])[["mean"]].join(mu, how="outer").fillna(0.0)
            per_task[task] = {v: float(np.corrcoef(j["mean"], j[v])[0, 1]) for v in vs}
        mean_r = {v: float(np.mean([per_task[t][v] for t in D.TASKS])) for v in CANDIDATES + REFERENCE}
        best = max(CANDIDATES, key=lambda v: mean_r[v])
        out[model] = {"variant": best, "mean_r": mean_r, "per_task": per_task,
                      "rule": "max over CANDIDATES of mean_task Pearson(endpoint-B discovery means, exact denoise discovery means), expert-position units"}
        print(model, "->", best, {v: round(mean_r[v], 4) for v in CANDIDATES + REFERENCE})
    p = ROOT / "results/relp_sva_moe/p8"
    p.mkdir(parents=True, exist_ok=True)
    (p / "selected_rw.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
