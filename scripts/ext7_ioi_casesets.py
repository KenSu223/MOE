"""ext7-controls W8 step 3 (CPU): IOI case sets, identical items for Qwen3 and Mixtral (BOS) and for both corruptions.

Pool = items that pass corruption (i)'s margin both ways (Delta_clean >= 1, Delta_s2io <= -1) under BOTH models
(results/ioi_{qwen3,mixtral_bos}/scan_pairs.parquet). random.Random(0) shuffle -> 128 discovery / 128 validation /
128 replication discovery / 128 replication validation items. Corruption (i) S2 -> IO: both directions (directed ids
2i, 2i + 1). Corruption (ii) S1, IO -> random names: the same items (their clean margin holds by construction), d = 0
only (the corrupted prompt has no answer in {IO, S}). pair_idx = item index (pair_id ioiNNNN).

Usage: python scripts/ext7_ioi_casesets.py
Outputs data/ioi/case_sets_s2io.json, data/ioi/case_sets_s1io.json, data/ioi/pool.json, and
data/ioi/case_sets_{s2io,s1io}_grid.json (W3 grid: the first 64 validation items, key "validation" only)
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import pandas as pd
from moetrace import ext7_controls as C
from moetrace.models import RESULTS


def main():
    it = pd.read_parquet(os.path.join(C.IOI_DIR, "items.parquet"))
    sc = {p: pd.read_parquet(os.path.join(RESULTS, f"ioi_{p}", "scan_pairs.parquet")) for p in C.PROTOS}
    for p, s in sc.items():
        assert (s.pair_id.values == it.pair_id.values).all()
    m_i = (sc["qwen3"].margin_s2io & sc["mixtral_bos"].margin_s2io).values
    m_ii = (sc["qwen3"].margin_clean & sc["mixtral_bos"].margin_clean).values
    pool = [i for i, ok in enumerate(m_i) if ok]
    ids = it.pair_id.tolist()
    note = ("W8 IOI items passing corruption (i) margin both ways under Qwen3 AND Mixtral BOS; seed-0 shuffle; pair_idx = item "
            "index; directed id = 2 * pair_idx + d")
    cs1 = C.case_sets(ids, pool, seed=0, both=True, note=note + " (corruption (i) S2 -> IO, both directions)")
    cs2 = C.case_sets(ids, pool, seed=0, both=False, note=note + " (corruption (ii) S1 and IO -> two other names, d = 0 only)")
    assert cs1["pairs"] == cs2["pairs"]
    json.dump(cs1, open(os.path.join(C.IOI_DIR, "case_sets_s2io.json"), "w"), indent=1)
    json.dump(cs2, open(os.path.join(C.IOI_DIR, "case_sets_s1io.json"), "w"), indent=1)
    g = cs1["pairs"]["validation"][:64]
    for name, both in (("s2io", True), ("s1io", False)):
        json.dump({"note": f"W3 grid subset: the first 64 validation items of case_sets_{name}.json", "pair_idx_of": cs1["pair_idx_of"],
                   "pairs": {"validation": g}, "directed": {"validation": C.directed(g, both)}},
                  open(os.path.join(C.IOI_DIR, f"case_sets_{name}_grid.json"), "w"), indent=1)
    info = {"n_items": len(it), "pool_s2io_both_models": int(m_i.sum()), "pool_clean_margin_both_models": int(m_ii.sum()),
            "per_model": {p: {"margin_clean": int(s.margin_clean.sum()), "margin_s2io": int(s.margin_s2io.sum()),
                              "top1_both_s2io": int(s.top1_both_s2io.sum())} for p, s in sc.items()},
            "sets": {k: len(v) for k, v in cs1["pairs"].items()}, "mode": cs1["mode"],
            "template_type_in_sets": {k: it.template_type.iloc[v].value_counts().to_dict() for k, v in cs1["pairs"].items() if v}}
    json.dump(info, open(os.path.join(C.IOI_DIR, "pool.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
