"""Pass 4: noise-scale sensitivity (Appendix E). Fixed layer L* and expert e*; sigma multipliers {1, 2, 4} (3.0 comes
from the main passes). Rows: clean prefill (shared), noised_sigma prefill, layer patch at L*, expert patch e* at L*,
plus coalition rows for completeness.

Usage: python scripts/run_noise.py <model_key> --layer 44 --expert 69 --set paper [--sigmas 1,2,4]
Outputs results/<model>/noise_rows.parquet
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS
from moetrace.protocol import out_dir, load_case_sets, cases_by_id
from moetrace.engine import Engine, PrefillSpec, SpawnSpec
from moetrace.noise import noise_draw


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--expert", type=int, required=True)
    ap.add_argument("--set", default="paper")
    ap.add_argument("--sigmas", default="1,2,4")
    args = ap.parse_args()
    sigmas = [float(x) for x in args.sigmas.split(",")]
    m = MODELS[args.model]
    od = out_dir(args.model)
    sets = load_case_sets(args.model)
    ids = sets[args.set]["discovery"] + sets[args.set]["validation"]
    cases, rej = cases_by_id(args.model, ids)
    ids = [c for c in ids if c in cases]
    eng = Engine(m["repo"])
    Hd = eng.hidden
    n = len(ids)
    L = args.layer
    pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ids]
    spawns, tags = [], []
    for si, sm in enumerate(sigmas):
        sigma = sm * eng.embed_std
        off = n * (1 + si)
        pre += [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id, cases[c].subject_pos,
                            noise_draw(c, len(cases[c].subject_pos), Hd, sigma)) for c in ids]
        for i, c in enumerate(ids):
            spawns.append(SpawnSpec(L, off + i, i, "layer")); tags.append((c, sm, "layer", -1))
            spawns.append(SpawnSpec(L, off + i, i, "expert", expert=args.expert)); tags.append((c, sm, "expert", args.expert))
            spawns.append(SpawnSpec(L, off + i, i, "coalition_clean")); tags.append((c, sm, "coalition_clean", -1))
            spawns.append(SpawnSpec(L, off + i, i, "coalition_union")); tags.append((c, sm, "coalition_union", -1))
    log(f"{args.model}: {n} cases x sigmas {sigmas}: {len(pre)} prefill rows, {len(spawns)} spawn rows")
    t0 = time.time()
    res = eng.run(pre, spawns, record_routing=True, log=log)
    log(f"pass time {time.time() - t0:.1f}s")
    d = res.delta
    rows = []
    for i, c in enumerate(ids):
        rows.append(dict(case_id=c, sigma_mult=0.0, kind="clean", expert=-1, delta=float(d[i]), delta_clean=float(d[i]),
                         delta_noised=np.nan, rescue=np.nan, clean_active=np.nan))
    for si, sm in enumerate(sigmas):
        off = n * (1 + si)
        for i, c in enumerate(ids):
            rows.append(dict(case_id=c, sigma_mult=sm, kind="noised", expert=-1, delta=float(d[off + i]), delta_clean=float(d[i]),
                             delta_noised=float(d[off + i]), rescue=np.nan, clean_active=np.nan))
    for j, (c, sm, kind, e) in enumerate(tags):
        i = ids.index(c)
        si = sigmas.index(sm)
        off = n * (1 + si)
        ca = bool(e in res.route_idx[L, i].tolist()) if e >= 0 else np.nan
        rows.append(dict(case_id=c, sigma_mult=sm, kind=kind, expert=e, delta=float(res.sp_delta[j]), delta_clean=float(d[i]),
                         delta_noised=float(d[off + i]), rescue=float(res.sp_delta[j] - d[off + i]), clean_active=ca,
                         vnorm=float(res.sp_vnorm[j])))
    df = pd.DataFrame(rows)
    df["layer"] = L
    df["set"] = args.set
    df.to_parquet(os.path.join(od, "noise_rows.parquet"), index=False)
    val = set(sets[args.set]["validation"])
    for sm in sigmas:
        sub = df[(df.sigma_mult == sm) & df.case_id.isin(val)]
        e_ = sub[sub.kind == "expert"]
        nz = sub[sub.kind == "noised"]
        log(f"sigma {sm}: drop {float((nz.delta_clean - nz.delta_noised).mean()):+.3f}  expert rescue {e_.rescue.mean():+.3f} "
            f"pos {float((e_.rescue > 0).mean()):.2f} active {int(e_.clean_active.sum())}/{len(e_)}  layer rescue {sub[sub.kind=='layer'].rescue.mean():+.3f}")
    log("done")


if __name__ == "__main__":
    main()
