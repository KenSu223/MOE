"""Pass 2: layer sweep. For the union of all case sets (paper IDs, our strict 256, our relaxed 512): clean + noised
prefill rows and a MoE-block output patch (layer patch) at every layer. Records per-layer final-position routing
(clean and noised top-k indices, weights, contribution norms).

Usage: python scripts/run_sweep.py <model_key>
Outputs results/<model>/sweep_rows.parquet, sweep_routing.parquet, sweep_cases.parquet
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS
from moetrace.protocol import out_dir, load_case_sets, set_names, cases_by_id, membership_table
from moetrace.engine import Engine, PrefillSpec, SpawnSpec
from moetrace.noise import noise_draw


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sigma-mult", type=float, default=3.0)
    ap.add_argument("--token-rule", default="space", help="space (default) | paper_like")
    ap.add_argument("--out", default=None, help="results subdir name (default: model key)")
    ap.add_argument("--no-special-tokens", action="store_true", help="tokenise without BOS/special tokens")
    ap.add_argument("--sets", default=None, help="comma list; default all available")
    args = ap.parse_args()
    m = MODELS[args.model]
    od = out_dir(args.out or args.model)
    sets = load_case_sets(args.out or args.model)
    names = args.sets.split(",") if args.sets else set_names(sets)
    all_ids = []
    for s in names:
        for cid in sets[s]["discovery"] + sets[s]["validation"]:
            if cid not in all_ids:
                all_ids.append(cid)
    log(f"{args.model}: sets={names} union={len(all_ids)} cases")
    cases, rej = cases_by_id(args.model, all_ids, token_rule=args.token_rule, special_tokens=not args.no_special_tokens)
    if rej:
        log(f"WARNING: {len(rej)} case ids not tokenizable with our single-token rule: {rej}")
    ids = [c for c in all_ids if c in cases]
    eng = Engine(m["repo"])
    sigma = args.sigma_mult * eng.embed_std
    Hd, L = eng.hidden, eng.spec.n_layers
    n = len(ids)
    pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ids]
    pre += [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id, cases[c].subject_pos,
                        noise_draw(c, len(cases[c].subject_pos), Hd, sigma)) for c in ids]
    spawns, tags = [], []
    for i, c in enumerate(ids):
        for l in range(L):
            spawns.append(SpawnSpec(l, n + i, i, "layer"))
            tags.append((c, l))
    log(f"running pass: {len(pre)} prefill rows, {len(spawns)} spawn rows")
    t0 = time.time()
    res = eng.run(pre, spawns, record_routing=True, log=log)
    log(f"pass time {time.time() - t0:.1f}s")
    d = res.delta
    dfull = res.logit_true_full - res.logit_foil_full
    rows = []
    for i, c in enumerate(ids):
        for kind, j in (("clean", i), ("noised", n + i)):
            rows.append(dict(case_id=c, kind=kind, layer=-1, sigma_mult=args.sigma_mult, logit_true=float(res.logit_true[j]),
                             logit_foil=float(res.logit_foil[j]), delta=float(d[j]), delta_full=float(dfull[j]),
                             top1=int(res.top1[j]), rescue=np.nan))
    sd = res.sp_delta
    idx_of = {c: i for i, c in enumerate(ids)}
    for j, (c, l) in enumerate(tags):
        i = idx_of[c]
        rows.append(dict(case_id=c, kind="layer", layer=l, sigma_mult=args.sigma_mult, logit_true=float(res.sp_logit_true[j]),
                         logit_foil=float(res.sp_logit_foil[j]), delta=float(sd[j]), delta_full=np.nan, top1=-1,
                         rescue=float(sd[j] - d[n + i])))
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(od, "sweep_rows.parquet"), index=False)
    # routing tables
    k = eng.spec.top_k
    rr = []
    for run, off in (("clean", 0), ("noised", n)):
        ri = res.route_idx[:, off : off + n]  # [L, n, k]
        rw = res.route_w[:, off : off + n]
        rc = res.route_cnorm[:, off : off + n]
        Ls, Ns, Ks = np.meshgrid(np.arange(L), np.arange(n), np.arange(k), indexing="ij")
        rr.append(pd.DataFrame(dict(case_id=np.array(ids)[Ns.ravel()], run=run, layer=Ls.ravel().astype(np.int16),
                                    slot=Ks.ravel().astype(np.int8), expert=ri.ravel().astype(np.int16),
                                    weight=rw.ravel(), cnorm=rc.ravel())))
    pd.concat(rr).to_parquet(os.path.join(od, "sweep_routing.parquet"), index=False)
    # case table
    memb = membership_table(sets, ids)
    meta = pd.DataFrame([cases[c].to_row() for c in ids])
    ct = meta.merge(memb, on="case_id")
    dc = df[df.kind == "clean"].set_index("case_id").delta
    dn = df[df.kind == "noised"].set_index("case_id").delta
    ct["delta_clean"] = ct.case_id.map(dc)
    ct["delta_noised"] = ct.case_id.map(dn)
    ct["drop"] = ct.delta_clean - ct.delta_noised
    ct["strict"] = (ct.delta_clean >= 1.0) & (ct["drop"] >= 0.5)
    ct["relaxed"] = (ct.delta_clean >= 0.5) & (ct["drop"] >= 0.25)
    ct.to_parquet(os.path.join(od, "sweep_cases.parquet"), index=False)
    # quick summary
    R = df[df.kind == "layer"].pivot(index="case_id", columns="layer", values="rescue")
    summ = {"n_cases": n, "pass_time_s": res.extra["total_s"], "T": res.extra["T"], "rejected": rej,
            "layer_times": [(int(l), round(a, 3), round(b, 3)) for l, a, b in res.layer_times]}
    for s in names:
        dsc = [c for c in sets[s]["discovery"] if c in R.index]
        val = [c for c in sets[s]["validation"] if c in R.index]
        md = R.loc[dsc].mean(0)
        mv = R.loc[val].mean(0)
        lstar = int(md.idxmax())
        summ[s] = {"n_disc": len(dsc), "n_val": len(val), "L_star_discovery": lstar, "disc_mean_at_Lstar": float(md[lstar]),
                   "val_mean_at_Lstar": float(mv[lstar]), "val_argmax": int(mv.idxmax()), "val_max": float(mv.max()),
                   "val_curve": [round(float(x), 3) for x in mv.values],
                   "funnel_strict_pass": int(ct[ct.case_id.isin(dsc + val)].strict.sum()),
                   "funnel_relaxed_pass": int(ct[ct.case_id.isin(dsc + val)].relaxed.sum())}
        log(f"set {s}: L*={lstar} disc={md[lstar]:+.3f} val@L*={mv[lstar]:+.3f} val argmax L{int(mv.idxmax())} {mv.max():+.3f} "
            f"funnel strict {summ[s]['funnel_strict_pass']}/{len(dsc)+len(val)}")
    with open(os.path.join(od, "sweep_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
