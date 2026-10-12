"""p5-atp step 1 (CPU): task data for the attribution-patching pilot on OLMoE.

  python scripts/ext13atp_build.py ioi-pairs     IOI S2 -> IO pairs re-tokenised for OLMoE (results/p5_atp_data/pairs_ioi_s2io_olmoe.parquet)
  python scripts/ext13atp_build.py wino-sets     WinoGrande case sets: 64 / 64 pairs from the Phase-3 discovery / validation
                                                 pairs that pass the OLMoE margin scan (results/wino_olmoe/scan_pairs.parquet)
  python scripts/ext13atp_build.py ioi-sets      IOI case sets from the OLMoE scan written by scripts/ext13atp_engine.py scan-ioi
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import pandas as pd
from moetrace import ext13atp as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["ioi-pairs", "wino-sets", "ioi-sets"])
    args = ap.parse_args()
    os.makedirs(A.DATA, exist_ok=True)
    if args.what == "ioi-pairs":
        from moetrace.ext7_controls import load_tok
        tok = load_tok("olmoe")
        p, info = A.build_ioi_pairs(tok)
        p.to_parquet(os.path.join(A.DATA, "pairs_ioi_s2io_olmoe.parquet"), index=False)
        json.dump(info, open(os.path.join(A.DATA, "build_ioi.json"), "w"), indent=1)
        print(json.dumps(info), p.n_tokens.describe().to_dict())
    elif args.what == "wino-sets":
        s = pd.read_parquet(os.path.join(A.RESULTS, "wino_olmoe", "scan_pairs.parquet"))
        elig = set(s[s.margin].pair_id)
        cs3 = json.load(open(os.path.join(A.ROOT, "data", "wino_str", "case_sets.json")))
        cs = A.make_case_sets(cs3, elig, note="p5-atp: 64/64 pairs of the Phase-3 WinoGrande discovery / validation pairs that pass "
                                              "the OLMoE margin (results/wino_olmoe/scan_pairs.parquet margin), random.Random(13)")
        json.dump(cs, open(A.TASKS["wino"]["case_sets"], "w"), indent=1)
        print({k: len(v) for k, v in cs["directed"].items()}, "eligible", len(elig))
    else:
        sc = pd.read_parquet(os.path.join(A.DATA, "scan_ioi_olmoe.parquet"))
        elig = set(sc[sc.margin].pair_id)
        cs3 = json.load(open(os.path.join(A.ROOT, "data", "ioi", "case_sets_s2io.json")))
        cs = A.make_case_sets(cs3, elig, note="p5-atp: 64/64 pairs of the Phase-3 IOI (s2io) discovery / validation pairs that pass "
                                              "the OLMoE margin (results/p5_atp_data/scan_ioi_olmoe.parquet), random.Random(13)")
        json.dump(cs, open(A.TASKS["ioi"]["case_sets"], "w"), indent=1)
        print({k: len(v) for k, v in cs["directed"].items()}, "eligible", len(elig))


if __name__ == "__main__":
    main()
