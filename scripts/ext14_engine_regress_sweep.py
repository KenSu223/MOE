"""ext14 (p6-engine) regression: re-run the first sweep pass (chunk 0) of a stored Phase-3/5 STR sweep run with the
current engine and compare every row with the stored str_sweep_rows.parquet / str_sweep_routing.parquet bit for bit.

The pass is rebuilt exactly as scripts/ext7_wino_sweep.py built it (same prefill rows, same spawn order, same layer chunk,
record_routing on, metrics as stored, wf_chunk 2048), so an unchanged engine reproduces the stored rows exactly. Nothing is
written into the stored run directory.

Usage: python scripts/ext14_engine_regress_sweep.py --run p5_f2e_qwen3 [--out results/verify_ext14_regress_sweep_<run>.json]
"""
import argparse, json, os, sys, time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "moetrace" not in sys.modules:
    sys.path.insert(0, TREE)
import numpy as np
import pandas as pd

MAIN = "/home/ubuntu/MOE"
METRICS = ("logp_true", "logp_foil", "p_true", "p_foil", "rank_true", "kl_to_clean")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from moetrace.models import MODELS
    from moetrace import ext7_pairs as P
    from moetrace.engine import Engine, PrefillSpec, SpawnSpec
    rd = os.path.join(MAIN, "results", args.run)
    meta = json.load(open(os.path.join(rd, "run_meta.json")))["sweep"]
    cs = json.load(open(os.path.join(MAIN, meta["case_sets"])))
    stored_fam = {k: v for k, v in json.load(open(os.path.join(rd, "case_sets.json"))).items() if not k.startswith("_")}
    fam = P.resolve_sets(cs, None, None)
    assert {f: {s: len(v) for s, v in sp.items()} for f, sp in fam.items()} == meta["families"], "families differ from run_meta"
    assert all(stored_fam[f] == sp for f, sp in fam.items())
    pairs = P.load_pairs(os.path.join(MAIN, meta["pairs"]), cs)
    ids = P.all_ids(fam)
    dc = P.directed_cases(pairs, ids)
    n = len(ids)
    kinds = meta["kinds"]
    ch = meta["layer_chunks"][0]
    metrics = bool(meta["metrics"])
    eng = Engine(MODELS[meta["model"]]["repo"])
    pre = [PrefillSpec(dc[c].clean_ids, dc[c].true_id, dc[c].foil_id) for c in ids]
    pre += [PrefillSpec(dc[c].corrupt_ids, dc[c].true_id, dc[c].foil_id, clean_ref=i) for i, c in enumerate(ids)]
    spawns, tags = [], []
    for i, c in enumerate(ids):
        for l in ch:
            for k in kinds:
                spawns.append(SpawnSpec(l, n + i, i, k)); tags.append((i, l, k))
    log(f"{args.run}: {len(pre)} prefill rows, {len(spawns)} spawns (chunk 0: layers {ch[0]}..{ch[-1]})")
    t0 = time.time()
    res = eng.run(pre, spawns, record_routing=True, metrics=metrics, wf_chunk=2048)
    log(f"pass {time.time() - t0:.1f}s")
    old = pd.read_parquet(os.path.join(rd, "str_sweep_rows.parquet"))
    old = old[old.chunk == 0]
    d = res.delta
    new_rows = []
    for i, c in enumerate(ids):
        for slot, j in ((-1, i), (0, n + i)):
            r = dict(case_id=c, kind="clean" if slot < 0 else "corrupt", layer=-1, delta=float(d[j]), logit_true=float(res.logit_true[j]),
                     logit_foil=float(res.logit_foil[j]), top1=int(res.top1[j]), vnorm=np.nan)
            if metrics:
                r.update({k: (int(res.metrics_prefill[k][j]) if k == "rank_true" else float(res.metrics_prefill[k][j])) for k in METRICS})
            new_rows.append(r)
    sd = res.sp_delta
    for jj, (i, l, k) in enumerate(tags):
        r = dict(case_id=ids[i], kind=k, layer=l, delta=float(sd[jj]), logit_true=float(res.sp_logit_true[jj]),
                 logit_foil=float(res.sp_logit_foil[jj]), top1=-1, vnorm=float(res.sp_vnorm[jj]))
        if metrics:
            r.update({k_: (int(res.metrics_spawn[k_][jj]) if k_ == "rank_true" else float(res.metrics_spawn[k_][jj])) for k_ in METRICS})
        new_rows.append(r)
    new = pd.DataFrame(new_rows)
    key = ["case_id", "kind", "layer"]
    m = old.merge(new, on=key, suffixes=("_old", "_new"), how="outer", indicator=True)
    rep = {"run": args.run, "model": meta["model"], "n_rows_old": int(len(old)), "n_rows_new": int(len(new)),
           "unmatched": int((m["_merge"] != "both").sum()), "engine_file": sys.modules["moetrace.engine"].__file__,
           "pass_s": time.time() - t0, "fields": {}}
    cols = ["delta", "logit_true", "logit_foil", "top1", "vnorm"] + (list(METRICS) if metrics else [])
    ok = rep["unmatched"] == 0
    for c in cols:
        a, b = m[c + "_old"].to_numpy(), m[c + "_new"].to_numpy()
        same = (a == b) | (pd.isna(a) & pd.isna(b))
        rep["fields"][c] = {"n_differ": int((~same).sum()), "maxabs": float(np.nanmax(np.abs(a.astype(float) - b.astype(float)))) if (~same).any() else 0.0}
        ok &= bool(same.all())
    ro = pd.read_parquet(os.path.join(rd, "str_sweep_routing.parquet"))
    L, k_ = eng.spec.n_layers, eng.spec.top_k
    ri, rw, rc = res.route_idx, res.route_w, res.route_cnorm
    row_case = np.array(ids + ids)
    row_slot = np.array([-1] * n + [0] * n)
    Ls, Bs, Ks = np.meshgrid(np.arange(L), np.arange(2 * n), np.arange(k_), indexing="ij")
    rn = pd.DataFrame(dict(case_id=row_case[Bs.ravel()], slot=row_slot[Bs.ravel()].astype(np.int8), layer=Ls.ravel().astype(np.int16),
                           kslot=Ks.ravel().astype(np.int8), expert=ri.ravel().astype(np.int16), weight=rw.ravel(), cnorm=rc.ravel()))
    mr = ro.merge(rn, on=["case_id", "slot", "layer", "kslot"], suffixes=("_old", "_new"))
    rep["routing"] = {"n": int(len(mr)), "n_old": int(len(ro)),
                      **{c: int((mr[c + "_old"] != mr[c + "_new"]).sum()) for c in ("expert", "weight", "cnorm")}}
    ok &= len(mr) == len(ro) and all(rep["routing"][c] == 0 for c in ("expert", "weight", "cnorm"))
    rep["identical"] = bool(ok)
    out = args.out or os.path.join("results", f"verify_ext14_regress_sweep_{args.run}.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    log(json.dumps(rep))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
