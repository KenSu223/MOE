"""ext13: Phase-5 case sets for the attribute -> entity WinoGrande formats (F1, F2R, F2E), per model (CPU).

The two models share too few usable pairs (F1 31, F2 65 in the first scans), so every model gets its own sets (Phase 3
used one shared set). Pool = the format's balanced pairs from results/wino_ref_<proto>/scan_pairs_<scan fmt>.parquet:
margin (Delta_A >= 1, Delta_B <= -1) & main (content swap) & not associative (RESEARCH_PLAN Phase 5, literature rule);
F2E = the F2 scan restricted to pairs with no word after the blank (the cut at the blank loses nothing).
pair_idx = index of the HIT id in the sorted list of all train_xl twins (rule W1, as data/wino_str/case_sets.json), so
directed case ids (2 * pair_idx + d) are stable across tokenizers and formats.
Splits: seed-0 shuffle (random.Random(0)) of the pool sorted by pair_idx -> discovery n, validation n, then
rep_discovery / rep_validation (n each, or half of the remainder each when the pool is smaller; omitted below 32 each).
--n 0 = half of the pool each for discovery / validation and no replication (small pools, user decision pending for F2).
Other pools are listed under "pools" (not run by default): connective margin pairs, associative margin pairs.

Usage: python scripts/ext13_casesets.py <qwen3|mixtral_bos> --fmt F1|F2R|F2E [--n 128]
Writes data/wino_ref/case_sets_<fmt>_<proto>.json (format of data/wino_str/case_sets.json)
"""
import argparse
import json
import random
import re
import sys

import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace.ext7_wino import load_twins
from moetrace.models import RESULTS

D = "/home/ubuntu/MOE/data/wino_ref"


def blank_at_end(sent: str) -> bool:
    return not re.search(r"[A-Za-z]", sent[sent.index("_") + 1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto", choices=["qwen3", "mixtral_bos"])
    ap.add_argument("--fmt", required=True, choices=["F1", "F2R", "F2E"])
    ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()
    scan_fmt = "F2" if a.fmt == "F2E" else a.fmt
    p = pd.read_parquet(f"{RESULTS}/wino_ref_{a.proto}/scan_pairs_{scan_fmt}.parquet")
    if a.fmt == "F2E":
        p = p[[blank_at_end(s) for s in p.sent_a]]
    twins, _ = load_twins("train_xl")
    pidx = {t.pair_id: i for i, t in enumerate(twins)}
    p = p.assign(pair_idx=p.pair_id.map(pidx)).sort_values("pair_idx")
    pool = list(p[p.balanced].pair_idx)
    rng = random.Random(0)
    order = list(pool)
    rng.shuffle(order)
    n = a.n if a.n else len(order) // 2
    sp = {"discovery": order[:n], "validation": order[n:2 * n]}
    rest = order[2 * n:]
    if a.n:
        r = min(n, len(rest) // 2)
        if r >= 32:
            sp["rep_discovery"], sp["rep_validation"] = rest[:r], rest[r:2 * r]
    used = {i for v in sp.values() for i in v}
    byidx = p.set_index("pair_idx")
    out = {
        "note": f"ext13 {a.fmt} case sets, {a.proto}: pool = balanced pairs (margin & content swap & not associative) of "
                f"results/wino_ref_{a.proto}/scan_pairs_{scan_fmt}.parquet" + (" with no word after the blank" if a.fmt == "F2E" else "")
                + "; seed-0 shuffle; per-model sets (scripts/ext13_casesets.py).",
        "pair_idx_of": {byidx.loc[i, "pair_id"]: int(i) for i in sorted(set(p.pair_idx))},
        "pairs": {k: [int(i) for i in v] for k, v in sp.items()},
        "directed": {k: [2 * int(i) + d for i in v for d in (0, 1)] for k, v in sp.items()},
        "pools": {"balanced": [int(i) for i in pool],
                  "hard": [int(i) for i in p[p.balanced & (p.assoc_cat == "hard")].pair_idx],
                  "connective_margin": [int(i) for i in p[p.margin & ~p.main].pair_idx],
                  "associative_margin": [int(i) for i in p[p.margin & p.main & (p.assoc_cat == "associative")].pair_idx]},
        "strata": {k: {"pairs": len(v), "names": int(byidx.loc[v, "names"].sum()),
                       "hard": int((byidx.loc[v, "assoc_cat"] == "hard").sum())} for k, v in sp.items()},
        "unused_balanced": len(pool) - len(used),
    }
    path = f"{D}/case_sets_{a.fmt}_{a.proto}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(path, json.dumps(out["strata"]), "pools", {k: len(v) for k, v in out["pools"].items()})


if __name__ == "__main__":
    main()
