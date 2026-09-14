"""ext2 attention-vs-MoE layer sweep (RESEARCH_PLAN Direction 2, Question B). Modelled on run_sweep.py.

For every case of the chosen case set(s): clean + noised prefill rows and, at EVERY layer, one spawn per kind in
--kinds (default attn_layer, layer, block, resid; see moetrace/engine.py for their definitions). One pass per run.

Usage: python scripts/ext2_attn_sweep.py <model_key> [--out <run>] [--base-run <run with case_sets.json>]
                                         [--sets paper] [--kinds attn_layer,layer,block,resid] [--no-special-tokens]
Outputs results/<run>/: sweep_rows.parquet (kinds clean, noised, <kinds>; extra column vnorm = fp32 norm of the
patched vector), sweep_routing.parquet, sweep_cases.parquet, sweep_summary.json (per kind), run_meta.json.
"""
import argparse, json, os, shutil, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import MODELS, RESULTS
from moetrace.protocol import out_dir, load_case_sets, set_names, cases_by_id, membership_table
from moetrace.engine import Engine, PrefillSpec, SpawnSpec, KINDS
from moetrace.noise import noise_draw

DEFAULT_KINDS = ["attn_layer", "layer", "block", "resid"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sigma-mult", type=float, default=3.0)
    ap.add_argument("--token-rule", default="space", help="space (default) | paper_like")
    ap.add_argument("--out", default=None, help="results subdir (default <model>_<bos|nobos>_attnsweep)")
    ap.add_argument("--base-run", default=None, help="run dir whose case_sets.json is used (default: model key)")
    ap.add_argument("--no-special-tokens", action="store_true", help="tokenise without BOS/special tokens")
    ap.add_argument("--sets", default="paper", help="comma list of case sets (default paper)")
    ap.add_argument("--kinds", default=",".join(DEFAULT_KINDS))
    ap.add_argument("--layers", default=None, help="comma list / a-b range (default all)")
    ap.add_argument("--agent", default="ext2-attn-patch")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    m = MODELS[args.model]
    special = not args.no_special_tokens
    run = args.out or f"{args.model}_{'bos' if special else 'nobos'}_attnsweep"
    base = args.base_run or args.model
    kinds = [k for k in args.kinds.split(",") if k]
    for k in kinds:
        assert k in KINDS, k
    od = out_dir(run)
    if not os.path.exists(os.path.join(od, "case_sets.json")):
        shutil.copy(os.path.join(RESULTS, base, "case_sets.json"), os.path.join(od, "case_sets.json"))
    sets = load_case_sets(run)
    names = [s for s in args.sets.split(",") if s in sets]
    all_ids = []
    for s in names:
        for cid in sets[s]["discovery"] + sets[s]["validation"]:
            if cid not in all_ids:
                all_ids.append(cid)
    log(f"{args.model} -> {run}: sets={names} union={len(all_ids)} cases, kinds={kinds}, special_tokens={special}")
    cases, rej = cases_by_id(args.model, all_ids, token_rule=args.token_rule, special_tokens=special)
    if rej:
        log(f"WARNING: {len(rej)} case ids not tokenizable: {rej}")
    ids = [c for c in all_ids if c in cases]
    meta = {"model": args.model, "repo": m["repo"], "base_run": base, "special_tokens": special, "token_rule": args.token_rule,
            "sigma_mult": args.sigma_mult, "case_sets": names, "kinds": kinds, "agent": args.agent,
            "experiment": "ext2 attention-vs-MoE attribution sweep (RESEARCH_PLAN Direction 2, Question B)",
            "command": "python " + " ".join(sys.argv), "created_utc": utc(), "n_cases": len(ids), "rejected": rej, "complete": False}
    if args.dry_run:
        print(json.dumps(meta, indent=1))
        return
    eng = Engine(m["repo"])
    sigma = args.sigma_mult * eng.embed_std
    Hd, L = eng.hidden, eng.spec.n_layers
    if args.layers:
        layers = []
        for part in args.layers.split(","):
            if "-" in part:
                a, b = part.split("-")
                layers += list(range(int(a), int(b) + 1))
            else:
                layers.append(int(part))
    else:
        layers = list(range(L))
    meta["layers"] = f"{layers[0]}-{layers[-1]} ({len(layers)} of {L})"
    n = len(ids)
    pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ids]
    pre += [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id, cases[c].subject_pos,
                        noise_draw(c, len(cases[c].subject_pos), Hd, sigma)) for c in ids]
    spawns, tags = [], []
    for i, c in enumerate(ids):
        for l in layers:
            for kind in kinds:
                spawns.append(SpawnSpec(l, n + i, i, kind))
                tags.append((c, l, kind))
    log(f"running pass: {len(pre)} prefill rows, {len(spawns)} spawn rows")
    with open(os.path.join(od, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
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
                             top1=int(res.top1[j]), rescue=np.nan, vnorm=np.nan))
    sd = res.sp_delta
    idx_of = {c: i for i, c in enumerate(ids)}
    for j, (c, l, kind) in enumerate(tags):
        i = idx_of[c]
        rows.append(dict(case_id=c, kind=kind, layer=l, sigma_mult=args.sigma_mult, logit_true=float(res.sp_logit_true[j]),
                         logit_foil=float(res.sp_logit_foil[j]), delta=float(sd[j]), delta_full=np.nan, top1=-1,
                         rescue=float(sd[j] - d[n + i]), vnorm=float(res.sp_vnorm[j])))
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(od, "sweep_rows.parquet"), index=False)
    # routing tables (clean / noised prefill rows, every layer)
    k = eng.spec.top_k
    rr = []
    for run_, off in (("clean", 0), ("noised", n)):
        ri = res.route_idx[:, off : off + n]
        rw = res.route_w[:, off : off + n]
        rc = res.route_cnorm[:, off : off + n]
        Ls, Ns, Ks = np.meshgrid(np.arange(L), np.arange(n), np.arange(k), indexing="ij")
        rr.append(pd.DataFrame(dict(case_id=np.array(ids)[Ns.ravel()], run=run_, layer=Ls.ravel().astype(np.int16),
                                    slot=Ks.ravel().astype(np.int8), expert=ri.ravel().astype(np.int16),
                                    weight=rw.ravel(), cnorm=rc.ravel())))
    pd.concat(rr).to_parquet(os.path.join(od, "sweep_routing.parquet"), index=False)
    # case table
    memb = membership_table(sets, ids)
    ct = pd.DataFrame([cases[c].to_row() for c in ids]).merge(memb, on="case_id")
    dc = df[df.kind == "clean"].set_index("case_id").delta
    dn = df[df.kind == "noised"].set_index("case_id").delta
    ct["delta_clean"] = ct.case_id.map(dc)
    ct["delta_noised"] = ct.case_id.map(dn)
    ct["drop"] = ct.delta_clean - ct.delta_noised
    ct["strict"] = (ct.delta_clean >= 1.0) & (ct["drop"] >= 0.5)
    ct["relaxed"] = (ct.delta_clean >= 0.5) & (ct["drop"] >= 0.25)
    ct.to_parquet(os.path.join(od, "sweep_cases.parquet"), index=False)
    # summary per set and kind
    summ = {"n_cases": n, "pass_time_s": res.extra["total_s"], "T": res.extra["T"], "rejected": rej, "kinds": kinds,
            "layer_times": [(int(l), round(a, 3), round(b, 3)) for l, a, b in res.layer_times]}
    for s in names:
        summ[s] = {}
        for kind in kinds:
            R = df[df.kind == kind].pivot(index="case_id", columns="layer", values="rescue")
            dsc = [c for c in sets[s]["discovery"] if c in R.index]
            val = [c for c in sets[s]["validation"] if c in R.index]
            md_, mv = R.loc[dsc].mean(0), R.loc[val].mean(0)
            lstar = int(md_.idxmax())
            summ[s][kind] = {"n_disc": len(dsc), "n_val": len(val), "L_star_discovery": lstar, "disc_mean_at_Lstar": float(md_[lstar]),
                             "val_mean_at_Lstar": float(mv[lstar]), "val_argmax": int(mv.idxmax()), "val_max": float(mv.max()),
                             "val_curve": [round(float(x), 3) for x in mv.values]}
            log(f"set {s} kind {kind:10s}: L*={lstar:2d} disc={md_[lstar]:+.3f} val@L*={mv[lstar]:+.3f} "
                f"val argmax L{int(mv.idxmax())} {mv.max():+.3f}")
    with open(os.path.join(od, "sweep_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    meta.update({"completed_utc": utc(), "pass_time_s": res.extra["total_s"], "n_rows": len(df), "prefill_rows": len(pre),
                 "spawn_rows": len(spawns), "rows_by_kind": {k_: int(v) for k_, v in df.kind.value_counts().items()},
                 "complete": True, "outputs": ["sweep_rows.parquet", "sweep_routing.parquet", "sweep_cases.parquet", "sweep_summary.json"]})
    with open(os.path.join(od, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
