"""ext7 W2: final-position layer sweep on STR pairs (WinoGrande twins; generic pair files, see moetrace/ext7_pairs.py).

Per directed case (id = 2 * pair_idx + d): a clean prefill row and a corrupted prefill row (the other prompt of the
pair, a plain prefill row: STR, no noise). Spawns (parent = corrupted row, clean = clean row) at EVERY layer for each kind
in --kinds (default layer = MoE output, attn_layer = attention output, block = both; resid optional). Routing at the
final position of every prefill row is recorded (first pass) for the expert pass. Metrics on (full-softmax log p / p /
rank of true and foil, KL to the clean row). Layers are chunked so that a pass has at most --max-rows spawn rows.

Usage: python scripts/ext7_wino_sweep.py --model qwen3 --pairs data/wino_str/pairs_train_xl_qwen3.parquet
           --case-sets data/wino_str/case_sets.json --out wino_qwen3_str [--own-pool qwen3] [--splits ...]
           [--kinds layer,attn_layer,block] [--max-rows 60000] [--dry-run]
Outputs results/<out>/: str_sweep_rows.parquet (kinds clean, corrupt, <kinds>; slot -1 clean / 0 corrupted),
        str_sweep_routing.parquet (slot -1 clean, 0 corrupted), sweep_cases.parquet, case_sets.json (families),
        str_sweep_summary.json, run_meta.json ("sweep")
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import MODELS
from moetrace import ext7_pairs as P

METRICS = ("logp_true", "logp_foil", "p_true", "p_foil", "rank_true", "kl_to_clean")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mcols(md, j):
    if md is None:
        return {}
    return {k: (int(md[k][j]) if k == "rank_true" else float(md[k][j])) for k in METRICS}


def layer_chunks(L, per_layer, max_rows):
    chunks, cur = [], []
    for l in range(L):
        if cur and (len(cur) + 1) * per_layer > max_rows:
            chunks.append(cur); cur = []
        cur.append(l)
    chunks.append(cur)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    P.add_common_args(ap)
    ap.add_argument("--kinds", default="layer,attn_layer,block")
    ap.add_argument("--max-rows", type=int, default=60000)
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--wf-chunk", type=int, default=2048, help="wavefront attention row chunk (memory)")
    args = ap.parse_args()
    kinds = args.kinds.split(",")
    m = MODELS[args.model]
    cs, pairs, fam, ids, dc = P.setup_run(args, "sweep")
    od = P.run_dir(args.out)
    n = len(ids)
    from moetrace.arch import load_spec
    L = load_spec(m["repo"])[0].n_layers
    chunks = layer_chunks(L, n * len(kinds), args.max_rows)
    log(f"{args.model} -> {args.out}: {n} directed cases ({len(set(c // 2 for c in ids))} pairs), families "
        f"{ {f: {s: len(v) for s, v in sp.items()} for f, sp in fam.items()} }, kinds {kinds}, {L} layers, {len(chunks)} passes "
        f"({n * len(kinds) * L} spawn rows), T max {max(x.T for x in dc.values())}")
    if args.dry_run:
        return
    import torch
    from moetrace.engine import Engine, PrefillSpec, SpawnSpec
    metrics = not args.no_metrics
    meta = {"model": args.model, "repo": m["repo"], "pairs": args.pairs, "case_sets": args.case_sets, "families": {f: {s: len(v) for s, v in sp.items()} for f, sp in fam.items()},
            "kinds": kinds, "n_cases": n, "max_rows": args.max_rows, "layer_chunks": chunks, "metrics": metrics, "agent": args.agent,
            "protocol": "STR pairs: clean row + corrupted row (other prompt of the pair) per directed case; spawns parent = corrupted, clean = clean",
            "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    P.write_meta(args.out, "sweep", meta)
    eng = Engine(m["repo"])
    assert eng.spec.n_layers == L
    pre = [PrefillSpec(dc[c].clean_ids, dc[c].true_id, dc[c].foil_id) for c in ids]
    pre += [PrefillSpec(dc[c].corrupt_ids, dc[c].true_id, dc[c].foil_id, clean_ref=i) for i, c in enumerate(ids)]
    rows, pass_times = [], []
    for ci, ch in enumerate(chunks):
        spawns, tags = [], []
        for i, c in enumerate(ids):
            for l in ch:
                for k in kinds:
                    spawns.append(SpawnSpec(l, n + i, i, k)); tags.append((i, l, k))
        log(f"pass {ci + 1}/{len(chunks)}: layers {ch[0]}..{ch[-1]}, {len(pre)} prefill rows, {len(spawns)} spawn rows")
        res = eng.run(pre, spawns, record_routing=(ci == 0), log=log if ci == 0 else None, metrics=metrics, wf_chunk=args.wf_chunk)
        pass_times.append(res.extra["total_s"])
        d = res.delta
        mp, ms = (res.metrics_prefill, res.metrics_spawn) if metrics else (None, None)
        if ci == 0:
            for i, c in enumerate(ids):
                for slot, j in ((-1, i), (0, n + i)):
                    rows.append(dict(case_id=c, slot=slot, kind="clean" if slot < 0 else "corrupt", layer=-1, delta=float(d[j]),
                                     logit_true=float(res.logit_true[j]), logit_foil=float(res.logit_foil[j]), top1=int(res.top1[j]),
                                     rescue=np.nan, delta_corrupt=float(d[n + i]), delta_clean=float(d[i]), vnorm=np.nan, chunk=ci, **mcols(mp, j)))
            k_ = eng.spec.top_k
            ri, rw, rc = res.route_idx, res.route_w, res.route_cnorm  # [L, 2n, k]
            row_case = np.array(ids + ids)
            row_slot = np.array([-1] * n + [0] * n)
            Ls, Bs, Ks = np.meshgrid(np.arange(L), np.arange(2 * n), np.arange(k_), indexing="ij")
            pd.DataFrame(dict(case_id=row_case[Bs.ravel()], slot=row_slot[Bs.ravel()].astype(np.int8), layer=Ls.ravel().astype(np.int16),
                              kslot=Ks.ravel().astype(np.int8), expert=ri.ravel().astype(np.int16), weight=rw.ravel(), cnorm=rc.ravel())
                         ).to_parquet(os.path.join(od, "str_sweep_routing.parquet"), index=False)
        sd = res.sp_delta
        for jj, (i, l, k) in enumerate(tags):
            rows.append(dict(case_id=ids[i], slot=0, kind=k, layer=l, delta=float(sd[jj]), logit_true=float(res.sp_logit_true[jj]),
                             logit_foil=float(res.sp_logit_foil[jj]), top1=-1, rescue=float(sd[jj] - d[n + i]), delta_corrupt=float(d[n + i]),
                             delta_clean=float(d[i]), vnorm=float(res.sp_vnorm[jj]), chunk=ci, **mcols(ms, jj)))
        pd.DataFrame(rows).to_parquet(os.path.join(od, "str_sweep_rows.parquet"), index=False)
        del res
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    ct = P.case_table(dc, pairs, fam)
    cl = df[df.kind == "clean"].set_index("case_id")
    co = df[df.kind == "corrupt"].set_index("case_id")
    ct["delta_clean"] = ct.case_id.map(cl.delta)
    ct["delta_corrupt"] = ct.case_id.map(co.delta)
    ct["drop"] = ct.delta_clean - ct.delta_corrupt
    ct["delta_noised"] = ct.delta_corrupt  # alias for moetrace.analysis (the corrupted run)
    ct["top1_clean"] = ct.case_id.map(cl.top1)
    ct["top1_corrupt"] = ct.case_id.map(co.top1)
    if metrics:
        for k in ("p_true", "p_foil", "rank_true"):
            ct[f"{k}_clean"] = ct.case_id.map(cl[k])
            ct[f"{k}_corrupt"] = ct.case_id.map(co[k])
    ct.to_parquet(os.path.join(od, "sweep_cases.parquet"), index=False)
    summ = {"n_cases": n, "pass_times_s": pass_times, "mean_delta_clean": float(ct.delta_clean.mean()),
            "mean_delta_corrupt": float(ct.delta_corrupt.mean()), "mean_drop": float(ct["drop"].mean()), "families": {}}
    for f, sp in fam.items():
        ent = {}
        for k in kinds:
            R = df[df.kind == k].pivot(index="case_id", columns="layer", values="rescue")
            for s, v in sp.items():
                mv = R.loc[v].mean(0)
                ent[f"{k}_{s}_argmax"] = int(mv.idxmax())
                ent[f"{k}_{s}_max"] = round(float(mv.max()), 3)
                ent[f"{k}_{s}_top5"] = [(int(l), round(float(x), 3)) for l, x in mv.sort_values(ascending=False).head(5).items()]
        summ["families"][f] = ent
        log(f"family {f}: " + "; ".join(f"{k} disc L{ent.get(f'{k}_discovery_argmax', ent.get(f'{k}_all_argmax'))}" for k in kinds))
    with open(os.path.join(od, "str_sweep_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    log(f"Δclean {summ['mean_delta_clean']:+.2f} Δcorrupt {summ['mean_delta_corrupt']:+.2f} drop {summ['mean_drop']:+.2f}; "
        + json.dumps(summ["families"].get("main", {}), default=str)[:600])
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True, "pass_times_s": pass_times, "n_rows": len(df)})
    P.write_meta(args.out, "sweep", meta)


if __name__ == "__main__":
    main()
