"""ext12 4b: option-swap case sets on EXACTLY the role-swap items (CPU).

The W7 role-swap pairs (data/wino_role/pairs_<proto>.parquet, case sets data/wino_role/case_sets_<proto>.json) were built
from each model's WinoGrande option-swap margin pool: role pair <twin>|A = (A, swap(A)) with r = trig of A, r' = trig of B;
<twin>|B = (B, swap(B)). The option-swap pair of the same twin is (A, B) in data/wino_str/pairs_train_xl_<proto>.parquet
(pair_id = twin id). This script writes, per model,
  results/wino_roleitems_inputs/case_sets_<proto>.json  option-swap directed ids of every twin that has a role pair in a
      role split (the twin keeps the role split's name: discovery / validation / rep_discovery / rep_validation), in the
      format of data/wino_str/case_sets.json (pair_idx_of of that file, "directed" {split: ids});
  results/wino_roleitems_inputs/link_<proto>.parquet  one row per role directed case: role case id, twin, option pair idx,
      the option directed case with the SAME clean prompt and the same (r, r') (role d = 0 of <twin>|A <-> option d = 0,
      role d = 0 of <twin>|B <-> option d = 1; role d = 1 has a role-swapped clean prompt: no option analogue), split.
Checks: the option pair is in the model's option-swap margin pool (results/wino_<proto>/scan_pairs.parquet margin) and the
clean prompt ids / trigger ids of linked cases are identical.

Usage: python scripts/ext12_roleitems_casesets.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace import ext7_pairs as P

ROOT = "/home/ubuntu/MOE"
OUT = os.path.join(ROOT, "results", "wino_roleitems_inputs")
SPLITS = ("discovery", "validation", "rep_discovery", "rep_validation")


def jl(x):
    return json.loads(x) if isinstance(x, str) else [int(v) for v in x]


def main():
    os.makedirs(OUT, exist_ok=True)
    wcs = json.load(open(os.path.join(ROOT, "data/wino_str/case_sets.json")))
    pio = wcs["pair_idx_of"]
    summary = {}
    for proto in ("qwen3", "mixtral_bos"):
        rcs = json.load(open(os.path.join(ROOT, f"data/wino_role/case_sets_{proto}.json")))
        rp = pd.read_parquet(os.path.join(ROOT, f"data/wino_role/pairs_{proto}.parquet")).set_index("pair_id", drop=False)
        op = pd.read_parquet(os.path.join(ROOT, f"data/wino_str/pairs_train_xl_{proto}.parquet")).set_index("pair_id", drop=False)
        scan = pd.read_parquet(os.path.join(ROOT, f"results/wino_{proto}/scan_pairs.parquet")).set_index("pair_id")
        rinv = {v: k for k, v in rcs["pair_idx_of"].items()}
        links, directed, twins_of = [], {}, {}
        for split in SPLITS:
            ids = rcs["directed"].get(split, [])
            if not ids:
                continue
            tw = []
            for c in ids:
                ri, d = divmod(int(c), 2)
                rpid = rinv[ri]
                r = rp.loc[rpid]
                twin = r.twin_id
                assert twin in op.index and twin in pio, (proto, twin)
                assert bool(scan.loc[twin, "margin"]), (proto, twin, "option pair not in the margin pool")
                oi = int(pio[twin])
                o = op.loc[twin]
                # same clean prompt and same (r, r') for role d = 0
                role_clean = jl(r.ids_a)
                if r.role == "A":
                    od, o_clean, o_true, o_foil = 0, jl(o.ids_a), int(o.trig_a), int(o.trig_b)
                else:
                    od, o_clean, o_true, o_foil = 1, jl(o.ids_b), int(o.trig_b), int(o.trig_a)
                assert role_clean == o_clean and int(r.trig_a) == o_true and int(r.trig_b) == o_foil, (proto, rpid)
                links.append(dict(role_case_id=int(c), role_pair_idx=ri, role_pair_id=rpid, role=r.role, d=d, twin_id=twin,
                                  opt_pair_idx=oi, opt_case_same_clean=(2 * oi + od) if d == 0 else -1, split=split))
                if twin not in tw:
                    tw.append(twin)
                twins_of.setdefault(twin, split)
                assert twins_of[twin] == split, (proto, twin, "twin in two role splits")
            directed[split] = [2 * int(pio[t]) + dd for t in tw for dd in (0, 1)]
        cs = {"note": __doc__ + f"\nModel/protocol: {proto}.", "pair_idx_of": pio, "directed": directed,
              "source": {"role_case_sets": f"data/wino_role/case_sets_{proto}.json", "role_pairs": f"data/wino_role/pairs_{proto}.parquet",
                         "option_pairs": f"data/wino_str/pairs_train_xl_{proto}.parquet"}}
        json.dump(cs, open(os.path.join(OUT, f"case_sets_{proto}.json"), "w"), indent=1)
        lk = pd.DataFrame(links)
        lk.to_parquet(os.path.join(OUT, f"link_{proto}.parquet"), index=False)
        # sanity: the option directed ids resolve in the option pair file
        pairs = P.load_pairs(os.path.join(ROOT, f"data/wino_str/pairs_train_xl_{proto}.parquet"), cs)
        P.directed_cases(pairs, [c for v in directed.values() for c in v])
        summary[proto] = {s: {"role_pairs": len(rcs["directed"][s]) // 2, "twins": len(v) // 2,
                              "twins_with_both_roles": int(lk[lk.split == s].groupby("twin_id").role.nunique().eq(2).sum())}
                          for s, v in directed.items()}
        print(proto, summary[proto])
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
