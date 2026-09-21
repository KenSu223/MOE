"""ext5 F4: pick the layers for the subject-site expert pass from a subject sweep.

Rule (documented default): the top --n layers of the DISCOVERY-split mean rescue of the MoE-output patch (`layer`) at the
last subject token, plus the final-token hypothesis layers given with --fixed (layer:expert pairs). Writes
results/<run>/expert_layers.json and prints the comma-separated layer list (consumed by the chain).

Usage: python scripts/ext5_subject_select.py <run> [--n 3] [--fixed 44:69,42:115] [--set paper]
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import pandas as pd
from moetrace.models import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--fixed", default="")
    ap.add_argument("--set", default="paper")
    args = ap.parse_args()
    od = os.path.join(RESULTS, args.run)
    df = pd.read_parquet(os.path.join(od, "sweep_rows.parquet"))
    sets = json.load(open(os.path.join(od, "case_sets.json")))
    R = df[df.kind == "layer"].pivot(index="case_id", columns="layer", values="rescue")
    disc = [c for c in sets[args.set]["discovery"] if c in R.index]
    md = R.loc[disc].mean(0).sort_values(ascending=False)
    peak_layers = [int(l) for l in md.index[: args.n]]
    fixed = [(int(a), int(b)) for a, b in (x.split(":") for x in args.fixed.split(",") if x)]
    layers = sorted(set(peak_layers) | {l for l, _ in fixed})
    out = {"run": args.run, "set": args.set, "rule": f"top-{args.n} discovery-mean layers of the MoE-output patch at the last subject token + fixed hypothesis layers",
           "peak_layers": peak_layers, "peak_disc_means": [round(float(md[l]), 4) for l in peak_layers], "fixed": fixed, "layers": layers,
           "disc_top8": [(int(l), round(float(v), 4)) for l, v in md.head(8).items()]}
    with open(os.path.join(od, "expert_layers.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(",".join(map(str, layers)))


if __name__ == "__main__":
    main()
