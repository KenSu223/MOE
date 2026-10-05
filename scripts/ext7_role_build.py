"""ext7-controls W7 step 1 (CPU): WinoGrande role-swap STR pairs (second corruption site, Zhang & Nanda Z7).

Input: a model's WinoGrande margin pool (results/wino_<proto>/scan_pairs.parquet, column margin; built by agent
ext7-wino with scripts/ext7_wino_scan.py) restricted to name pairs (column names: both options are capitalised names).
Rules (moetrace/ext7_controls.py): R1 each option mentioned exactly once before the blank, R2 no mention after the
blank, R3 token symmetry (same length, the two mention spans at the same token positions, identical tokens elsewhere),
R4 the same single-token continuation ids for both triggers after both prompts (= the WinoGrande pair's trig ids),
R5 the filled option is not the final token. Two role pairs per twin: role A (clean = prompt A, r = trig_a) and role B
(clean = prompt B, r = trig_b); in the contract each is stored with its own A = the clean prompt.

Usage: python scripts/ext7_role_build.py [--protos qwen3,mixtral_bos]
Outputs data/wino_role/pairs_<proto>.parquet, data/wino_role/funnel_<proto>.json
"""
import argparse, collections, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import pandas as pd
from moetrace import ext7_controls as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protos", default="qwen3,mixtral_bos")
    args = ap.parse_args()
    os.makedirs(C.ROLE_DIR, exist_ok=True)
    for proto in args.protos.split(","):
        key, st = C.PROTOS[proto]
        tok = C.load_tok(key)
        sp = pd.read_parquet(os.path.join(C.RESULTS, f"wino_{proto}", "scan_pairs.parquet"))
        pool = sp[sp.margin & sp.names].reset_index(drop=True)
        funnel = collections.Counter({"wino_pairs": len(sp), "margin": int(sp.margin.sum()), "margin_names": len(pool)})
        rows, twins_ok = [], set()
        for r in pool.itertuples():
            for role in ("A", "B"):
                d, why = C.build_role_pair(r, tok, st, role)
                funnel[f"{role}:{why}"] += 1
                if d is not None:
                    rows.append(d)
                    twins_ok.add(r.pair_id)
        df = pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)
        df.to_parquet(os.path.join(C.ROLE_DIR, f"pairs_{proto}.parquet"), index=False)
        funnel["role_pairs"] = len(df)
        funnel["twins_with_a_role_pair"] = len(twins_ok)
        json.dump(dict(funnel), open(os.path.join(C.ROLE_DIR, f"funnel_{proto}.json"), "w"), indent=1)
        print(proto, dict(funnel))
        for r in df.head(3).itertuples():
            print("  ", repr(r.prompt_a), "->", r.word_a, "|", repr(r.prompt_b), "->", r.word_b, "str_pos", r.str_pos)


if __name__ == "__main__":
    main()
