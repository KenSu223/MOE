"""ext7-controls W7 step 3 (CPU): case sets of the WinoGrande role-swap STR pairs (contract format of
data/wino_str/case_sets.json).

Pool = role pairs that pass the margin both ways in the model's competence scan (results/wino_role_<proto>/
scan_pairs.parquet). pair_idx = row index of the pair in data/wino_role/pairs_<proto>.parquet (sorted by pair_id).
random.Random(0) shuffle over WinoGrande twins (both role pairs of a twin stay in one split) -> 128 discovery / 128
validation pairs (+ replication as far as the pool allows) when >= 256 pairs pass; otherwise all passing pairs form a
fixed-hypothesis validation set (no discovery split). Directed ids 2 * pair_idx + d, both directions.

Usage: python scripts/ext7_role_casesets.py [--protos qwen3,mixtral_bos]
Outputs data/wino_role/case_sets_<proto>.json and case_sets_<proto>_grid.json (W3 position x layer grid: the first 64
validation pairs, key "validation" only, to bound the GPU cost)
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import pandas as pd
from moetrace import ext7_controls as C
from moetrace.models import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protos", default="qwen3,mixtral_bos")
    args = ap.parse_args()
    for proto in args.protos.split(","):
        pairs = pd.read_parquet(os.path.join(C.ROLE_DIR, f"pairs_{proto}.parquet"))
        sc = pd.read_parquet(os.path.join(RESULTS, f"wino_role_{proto}", "scan_pairs.parquet"))
        assert (sc.pair_id.values == pairs.pair_id.values).all()
        ids = pairs.pair_id.tolist()
        pool = [i for i, ok in enumerate(sc.margin.values) if ok]
        groups = {i: pairs.twin_id.iloc[i] for i in pool}
        cs = C.case_sets(ids, pool, seed=0, groups=groups, both=True,
                         note=f"W7 role-swap pairs of {proto}: margin both ways (Delta_clean >= 1, Delta_swapped <= -1); seed-0 "
                              "shuffle over WinoGrande twins; directed id = 2 * pair_idx + d (d = 0: clean = prompt_a, corrupted = "
                              "role-swapped prompt_b, true = trig_a)")
        cs["pool_twins"] = int(pairs.twin_id.iloc[pool].nunique())
        cs["top1_both_in_pool"] = int(sc.top1_both.values[pool].sum())
        json.dump(cs, open(os.path.join(C.ROLE_DIR, f"case_sets_{proto}.json"), "w"), indent=1)
        g = cs["pairs"]["validation"][:64]
        json.dump({"note": "W3 grid subset: the first 64 validation pairs of case_sets_%s.json" % proto, "pair_idx_of": cs["pair_idx_of"],
                   "pairs": {"validation": g}, "directed": {"validation": C.directed(g, True)}},
                  open(os.path.join(C.ROLE_DIR, f"case_sets_{proto}_grid.json"), "w"), indent=1)
        print(proto, cs["mode"], {k: len(v) for k, v in cs["pairs"].items()}, "pool", len(pool), "twins", cs["pool_twins"])


if __name__ == "__main__":
    main()
