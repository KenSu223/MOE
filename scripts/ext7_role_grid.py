"""ext7-controls W7 / W3 (GPU): position x layer grid of the WinoGrande role swap at the STR site and the answer-relevant
positions only: the two exchanged first mentions (mention1, mention2, all their tokens), the filled option at the blank
(filled) and the final position (final; equals the W2 sweep at window 1). Same executor and row schema as agent
ext7-wino's scripts/ext7_wino_grid.py (moetrace.ext5_subject, window 1, kinds layer and attn_layer); positions between
and after are skipped to bound the cost (they carry no STR difference and were not requested).

Usage: python scripts/ext7_role_grid.py --model qwen3|mixtral --pairs data/wino_role/pairs_<proto>.parquet
           --case-sets data/wino_role/case_sets_<proto>_grid.json --out wino_role_<proto>_grid [--kinds layer,attn_layer]
Outputs results/<out>/str_grid_w1_rows.parquet (+ cls), str_grid_w1_prefill.parquet, case_sets.json, run_meta.json
"""
import argparse, json, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace import ext7_controls as C


def positions(r, x):
    m = json.loads(r.mention_pos)
    o = json.loads(r.opt_pos)
    out = [(p, "mention1") for p in m[0]] + [(p, "mention2") for p in m[1]] + [(p, "filled") for p in o] + [(x.T - 1, "final")]
    return out


def main():
    ap = argparse.ArgumentParser()
    C.grid_args(ap)
    args = ap.parse_args()
    args.kinds = args.kinds.split(",")
    args.positions_doc = "mention1, mention2 (exchanged first mentions), filled (option at the blank), final"
    C.run_grid_positions(args, positions, log=lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True))


if __name__ == "__main__":
    main()
