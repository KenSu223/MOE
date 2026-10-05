"""ext8: AUC over log k restricted to k = 1..15 for every ordering, so adaptive greedy (15 steps) is comparable.

The static orderings are evaluated on the grid of results/tables/ext8_a1_curves.csv (k = 1..10, 12, 16, ...); r(15) is
interpolated linearly in log k between k = 12 and 16, as `auc_log_k15_oracle_interp` in ext8_addback_analyze.py does.
Greedy = results/ext8_addback_summary.json runs/<run>/greedy/auc_log_k15 (k = 1..15 evaluated exactly).

Usage: python scripts/ext8_partial_auc.py
Outputs results/tables/ext8_a1_partial_auc_k15.{csv,md}
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace import ext8_addback as X

R = "/home/ubuntu/MOE/results"
RUNS = ("cf_qwen3", "cf_mixtral", "wino_qwen3", "wino_mixtral")
ORDS = ("oracle", "dla", "pop", "layerwise", "vnorm", "weight", "rand")
KMAX = 15


def main():
    cur = pd.read_csv(f"{R}/tables/ext8_a1_curves.csv")
    summ = json.load(open(f"{R}/ext8_addback_summary.json"))["runs"]
    gk = np.arange(1, KMAX + 1)
    rows = []
    for run in RUNS:
        row = {"run": run, "greedy": summ[run]["greedy"]["auc_log_k15"]}
        for o in ORDS:
            c = cur[(cur.run == run) & (cur.donors == "mean") & (cur.ordering == o)].sort_values("k")
            row[o] = X.auc_log(gk, np.interp(np.log(gk), np.log(c.k.values), c.r.values))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{R}/tables/ext8_a1_partial_auc_k15.csv", index=False)
    cols = ["greedy", *ORDS]
    md = ["| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for r in rows:
        md.append(f"| {r['run']} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    open(f"{R}/tables/ext8_a1_partial_auc_k15.md", "w").write(
        "AUC of r(k) over log k for k = 1..15 (validation, donor mean; fraction of the drop; static orderings "
        "interpolated at k = 15)\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
