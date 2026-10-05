"""ext7-controls W8 / W3 (GPU): position x layer grid of IOI at the positions S1, IO, S2 and every position from S2 to
the final token (S2 -> IO corruption: only S2 and later can differ; S1/IO -> other names: S1 and IO are the STR site).
Same executor and row schema as agent ext7-wino's scripts/ext7_wino_grid.py (moetrace.ext5_subject, window 1, kinds
layer and attn_layer); the tokens between IO/S1 and S2 (", went to the store.") are skipped to bound the cost.
Positions before the first STR position are exactly zero under STR and are not run.

Usage: python scripts/ext7_ioi_grid.py --model qwen3|mixtral --pairs data/ioi/pairs_<corr>_<proto>.parquet
           --case-sets data/ioi/case_sets_<corr>_grid.json --out ioi_<proto>_<corr>_grid [--kinds layer,attn_layer]
Outputs results/<out>/str_grid_w1_rows.parquet (+ cls: S1, IO, S2, after_S2, final), str_grid_w1_prefill.parquet
"""
import argparse, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace import ext7_controls as C


def positions(r, x):
    s1, io, s2 = int(r.s1_pos), int(r.io_pos), int(r.s2_pos)
    first = min(x.str_pos)
    out = [(p, lab) for p, lab in ((s1, "S1"), (io, "IO")) if p >= first]
    out += [(s2, "S2")] + [(p, "after_S2") for p in range(s2 + 1, x.T - 1)] + [(x.T - 1, "final")]
    return out


def main():
    ap = argparse.ArgumentParser()
    C.grid_args(ap)
    args = ap.parse_args()
    args.kinds = args.kinds.split(",")
    args.positions_doc = "S1 and IO (when at or after the first STR position), S2, every position after S2, final"
    C.run_grid_positions(args, positions, log=lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True))


if __name__ == "__main__":
    main()
