"""ext13 Phase 5 scans (GPU, prefill rows only; agent p5-data).

  --part probes  every row of data/wino_ref/p5_probe_specs_<proto>.parquet (probe v2 for every key, v1 where no earlier
                 scan has it); Delta = logit(ans_a) - logit(ans_b) on the probe prompt.
                 -> results/wino_ref_<proto>/p5_probe_rows.parquet, p5_probe_values.parquet (key, v1_a, v1_b, v2_a, v2_b;
                 probe_x = mean over the two candidate orders; v1 from the earlier scans where known)
  --part conds   clean rows of both twins of every pair in data/wino_ref/p5build_<cond>_<proto>.parquet
                 (cond in vt_a2e, cp_vt, cp_nat, mem): d_clean_a, d_clean_b (Delta = logit(trig_a) - logit(trig_b)), top-1,
                 margin (d_a >= 1 and d_b <= -1), correct_both -> results/wino_ref_<proto>/scan_pairs_p5_<cond>.parquet
Usage: python scripts/ext13_p5_scan.py <qwen3|mixtral_bos> --part probes|conds [--conds vt_a2e,cp_vt,cp_nat,mem]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace.models import MODELS, RESULTS

PROTOS = {"qwen3": ("qwen3", True), "mixtral_bos": ("mixtral", True)}
D = "/home/ubuntu/MOE/data/wino_ref"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def chunks_by_tokens(lens, max_tokens):
    order = sorted(range(len(lens)), key=lambda j: lens[j])
    out, cur, cur_max = [], [], 0
    for j in order:
        T = lens[j]
        if cur and max(cur_max, T) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, cur_max = [], 0
        cur.append(j)
        cur_max = max(cur_max, T)
    if cur:
        out.append(cur)
    return out


def run_rows(eng, specs, max_tokens):
    ch = chunks_by_tokens([len(s.ids) for s in specs], max_tokens)
    delta = np.full(len(specs), np.nan)
    top1 = np.full(len(specs), -1, dtype=np.int64)
    for ci, c in enumerate(ch):
        res = eng.run([specs[j] for j in c], [], record_routing=False, log=log if ci == 0 else None)
        delta[c], top1[c] = res.delta, res.top1
        log(f"  pass {ci + 1}/{len(ch)} ({len(c)} rows, {res.extra['total_s']:.0f}s)")
    return delta, top1, len(ch)


def write_meta(od, key, meta):
    rm = os.path.join(od, "run_meta.json")
    allm = json.load(open(rm)) if os.path.exists(rm) else {}
    allm[key] = meta
    json.dump(allm, open(rm, "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto", choices=list(PROTOS))
    ap.add_argument("--part", required=True, choices=["probes", "conds"])
    ap.add_argument("--conds", default="vt_a2e,cp_vt,cp_nat,mem")
    ap.add_argument("--max-tokens", type=int, default=140000)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--meta-key", default=None, help="run_meta.json key (default p5_scan_<part>)")
    a = ap.parse_args()
    key, st = PROTOS[a.proto]
    m = MODELS[key]
    od = os.path.join(RESULTS, f"wino_ref_{a.proto}")
    os.makedirs(od, exist_ok=True)
    mkey = a.meta_key or f"p5_scan_{a.part}"
    meta = {"model": key, "repo": m["repo"], "proto": a.proto, "special_tokens": st, "part": a.part, "agent": "p5-data",
            "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    write_meta(od, mkey, meta)
    from moetrace.engine import Engine, PrefillSpec
    eng = Engine(m["repo"])
    if a.part == "probes":
        sp = pd.read_parquet(f"{D}/p5_probe_specs_{a.proto}.parquet")
        specs = [PrefillSpec(json.loads(r.ids), int(r.ea), int(r.eb)) for r in sp.itertuples()]
        log(f"probes: {len(sp)} rows, {sp.key.nunique()} keys")
        d, t1, n = run_rows(eng, specs, a.max_tokens)
        sp = sp.drop(columns=["ids"]).assign(delta=d, top1=t1)
        sp.to_parquet(os.path.join(od, "p5_probe_rows.parquet"), index=False)
        sp["trig"] = sp.slot.str[0]
        g = sp.groupby(["key", "version", "trig"]).delta.mean().unstack(["version", "trig"])
        g.columns = [f"{v}_{t}" for v, t in g.columns]
        kd = pd.read_parquet(f"{D}/p5_probe_keys_{a.proto}.parquet").set_index("key")
        out = kd.join(g, how="left")
        if "v1_a" not in out:
            out["v1_a"], out["v1_b"] = np.nan, np.nan
        known = out.v1_spec == "known"
        out.loc[known, "v1_a"] = out.loc[known, "v1_known_a"]
        out.loc[known, "v1_b"] = out.loc[known, "v1_known_b"]
        out.reset_index().to_parquet(os.path.join(od, "p5_probe_values.parquet"), index=False)
        meta.update(n_rows=len(sp), n_keys=int(len(out)), n_passes=n)
        log("probe values written", out[["v1_a", "v1_b", "v2_a", "v2_b"]].describe().round(3).to_dict())
    else:
        tot = {}
        for cond in a.conds.split(","):
            p = pd.read_parquet(f"{D}/p5build_{cond}_{a.proto}.parquet")
            specs = []
            for r in p.itertuples():
                specs.append(PrefillSpec(json.loads(r.ids_a), int(r.trig_a), int(r.trig_b)))
                specs.append(PrefillSpec(json.loads(r.ids_b), int(r.trig_a), int(r.trig_b)))
            log(f"{cond}: {len(p)} pairs, {len(specs)} rows")
            d, t1, n = run_rows(eng, specs, a.max_tokens)
            q = p.drop(columns=[c for c in ("ids_a", "ids_b", "probe_json") if c in p.columns]).copy()
            q["d_clean_a"], q["d_clean_b"] = d[0::2], d[1::2]
            q["top1_a"], q["top1_b"] = t1[0::2], t1[1::2]
            q["drop_str"] = q.d_clean_a - q.d_clean_b
            q["margin"] = (q.d_clean_a >= a.margin) & (q.d_clean_b <= -a.margin)
            q["correct_both"] = (q.d_clean_a > 0) & (q.d_clean_b < 0)
            q["top1_both"] = (q.top1_a == q.trig_a) & (q.top1_b == q.trig_b)
            q.to_parquet(os.path.join(od, f"scan_pairs_p5_{cond}.parquet"), index=False)
            gcol = "variant" if "variant" in q.columns else None
            summ = (q.groupby(gcol)[["margin", "correct_both", "top1_both"]].mean().round(3).to_dict() if gcol else
                    q[["margin", "correct_both", "top1_both"]].mean().round(3).to_dict())
            tot[cond] = {"pairs": int(len(q)), "margin": int(q.margin.sum()), "passes": n, "rates": summ}
            log(cond, json.dumps(tot[cond]))
        meta.update(conds=tot)
    meta.update(complete=True, completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_meta(od, mkey, meta)


if __name__ == "__main__":
    main()
