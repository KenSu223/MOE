"""ext5 F2: per-head attention patching at the final position (RESEARCH_PLAN Phase 2, F2). Modelled on ext2_attn_sweep.py.

For every case of the case set(s) and every requested layer: one `attn_head` spawn per head (v_h = W_o[:, h] (H_h_clean -
H_h_noised) added to the noised attention output before the MoE of that layer), plus `attn_layer`, `layer` and `block`
reference rows. Prefill diagnostics: the final position's attention distribution over positions at the requested layers
(DiagSpec.attn_final, saved for the clean and noised rows). One engine pass per run.

Usage: python scripts/ext5_heads_sweep.py <model_key> --out <run> --layers 40,43,44 [--base-run <run with case_sets.json>]
                                          [--sets paper] [--no-special-tokens] [--token-rule space]
Outputs results/<run>/:
  head_rows.parquet      case_id, layer, head (-1 for reference kinds), kind (attn_head | attn_layer | layer | block),
                         logit_true, logit_foil, delta, rescue, vnorm, delta_clean, delta_noised, sigma_mult
  head_prefill.parquet   case_id, delta_clean, delta_noised, top1_clean, top1_noised
  sweep_cases.parquet    case table (ids, subject_pos as JSON, set membership, drop, funnel flags)
  sweep_routing.parquet  clean/noised final-position routing at every layer (as in run_sweep.py)
  head_attn_final.npz    attn [n_layers, 2n, n_heads, T] fp16 (rows: clean 0..n-1, noised n..2n-1), layers, case_ids, lens
  head_summary.json, run_meta.json
"""
import argparse, json, os, shutil, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import MODELS, RESULTS
from moetrace.protocol import out_dir, load_case_sets, cases_by_id, membership_table
from moetrace.engine import Engine, PrefillSpec, SpawnSpec, DiagSpec
from moetrace.noise import noise_draw

REF_KINDS = ("attn_layer", "layer", "block")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", required=True, help="comma list of layers")
    ap.add_argument("--base-run", default=None, help="run dir whose case_sets.json is used (default: model key)")
    ap.add_argument("--sets", default="paper")
    ap.add_argument("--sigma-mult", type=float, default=3.0)
    ap.add_argument("--token-rule", default="space")
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--agent", default="ext5-engine")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    m = MODELS[args.model]
    special = not args.no_special_tokens
    layers = sorted(set(int(x) for x in args.layers.split(",")))
    base = args.base_run or args.model
    od = out_dir(args.out)
    if not os.path.exists(os.path.join(od, "case_sets.json")):
        shutil.copy(os.path.join(RESULTS, base, "case_sets.json"), os.path.join(od, "case_sets.json"))
    sets = load_case_sets(args.out)
    names = [s for s in args.sets.split(",") if s in sets]
    all_ids = []
    for s in names:
        for cid in sets[s]["discovery"] + sets[s]["validation"]:
            if cid not in all_ids:
                all_ids.append(cid)
    cases, rej = cases_by_id(args.model, all_ids, token_rule=args.token_rule, special_tokens=special)
    if rej:
        log(f"WARNING: {len(rej)} case ids not tokenizable: {rej}")
    ids = [c for c in all_ids if c in cases]
    n = len(ids)
    meta = {"model": args.model, "repo": m["repo"], "base_run": base, "special_tokens": special, "token_rule": args.token_rule,
            "sigma_mult": args.sigma_mult, "case_sets": names, "layers": layers, "kinds": ["attn_head"] + list(REF_KINDS),
            "agent": args.agent, "experiment": "ext5 F2 per-head attention patching at the final position",
            "command": "python " + " ".join(sys.argv), "created_utc": utc(), "n_cases": n, "rejected": rej, "complete": False}
    log(f"{args.model} -> {args.out}: {n} cases, layers {layers}, special_tokens={special}")
    if args.dry_run:
        print(json.dumps(meta, indent=1))
        return
    eng = Engine(m["repo"])
    nH, Hd, L = eng.spec.n_heads, eng.hidden, eng.spec.n_layers
    sigma = args.sigma_mult * eng.embed_std
    pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ids]
    pre += [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id, cases[c].subject_pos,
                        noise_draw(c, len(cases[c].subject_pos), Hd, sigma)) for c in ids]
    spawns, tags = [], []
    for i, c in enumerate(ids):
        for l in layers:
            for h in range(nH):
                spawns.append(SpawnSpec(l, n + i, i, "attn_head", expert=h)); tags.append((c, l, "attn_head", h))
            for kind in REF_KINDS:
                spawns.append(SpawnSpec(l, n + i, i, kind)); tags.append((c, l, kind, -1))
    meta["spawn_rows"] = len(spawns)
    with open(os.path.join(od, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log(f"running pass: {len(pre)} prefill rows, {len(spawns)} spawn rows ({nH} heads x {len(layers)} layers + {len(REF_KINDS)} reference kinds)")
    t0 = time.time()
    res = eng.run(pre, spawns, record_routing=True, log=log, diag=DiagSpec(attn_final=True))
    log(f"pass time {time.time() - t0:.1f}s")
    d = res.delta
    sd = res.sp_delta
    idx_of = {c: i for i, c in enumerate(ids)}
    rows = []
    for j, (c, l, kind, h) in enumerate(tags):
        i = idx_of[c]
        rows.append(dict(case_id=c, layer=l, head=h, kind=kind, logit_true=float(res.sp_logit_true[j]), logit_foil=float(res.sp_logit_foil[j]),
                         delta=float(sd[j]), rescue=float(sd[j] - d[n + i]), vnorm=float(res.sp_vnorm[j]),
                         delta_clean=float(d[i]), delta_noised=float(d[n + i]), sigma_mult=args.sigma_mult))
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(od, "head_rows.parquet"), index=False)
    pf = pd.DataFrame(dict(case_id=ids, delta_clean=d[:n], delta_noised=d[n:], top1_clean=res.top1[:n], top1_noised=res.top1[n:]))
    pf.to_parquet(os.path.join(od, "head_prefill.parquet"), index=False)
    # routing tables and case table (same layout as run_sweep.py)
    k = eng.spec.top_k
    rr = []
    for run_, off in (("clean", 0), ("noised", n)):
        ri, rw, rc = res.route_idx[:, off : off + n], res.route_w[:, off : off + n], res.route_cnorm[:, off : off + n]
        Ls, Ns, Ks = np.meshgrid(np.arange(L), np.arange(n), np.arange(k), indexing="ij")
        rr.append(pd.DataFrame(dict(case_id=np.array(ids)[Ns.ravel()], run=run_, layer=Ls.ravel().astype(np.int16),
                                    slot=Ks.ravel().astype(np.int8), expert=ri.ravel().astype(np.int16), weight=rw.ravel(), cnorm=rc.ravel())))
    pd.concat(rr).to_parquet(os.path.join(od, "sweep_routing.parquet"), index=False)
    ct = pd.DataFrame([cases[c].to_row() for c in ids]).merge(membership_table(sets, ids), on="case_id")
    ct["delta_clean"] = d[:n]
    ct["delta_noised"] = d[n:]
    ct["drop"] = ct.delta_clean - ct.delta_noised
    ct["strict"] = (ct.delta_clean >= 1.0) & (ct["drop"] >= 0.5)
    ct["relaxed"] = (ct.delta_clean >= 0.5) & (ct["drop"] >= 0.25)
    ct.to_parquet(os.path.join(od, "sweep_cases.parquet"), index=False)
    attn = res.extra["diag"]["attn_final"]  # [L, 2n, nH, T] fp16
    np.savez_compressed(os.path.join(od, "head_attn_final.npz"), attn=attn[layers], layers=np.array(layers), case_ids=np.array(ids),
                        lens=res.lens, T=res.extra["T"])
    # summary: per layer, validation mean rescue of the reference kinds and the top heads
    summ = {"n_cases": n, "pass_time_s": res.extra["total_s"], "T": res.extra["T"], "n_heads": nH, "layers": layers, "rejected": rej,
            "layer_times": [(int(l), round(a, 3), round(b, 3)) for l, a, b in res.layer_times]}
    for s in names:
        val = [c for c in sets[s]["validation"] if c in idx_of]
        dsc = [c for c in sets[s]["discovery"] if c in idx_of]
        summ[s] = {}
        for l in layers:
            sub = df[(df.layer == l) & df.case_id.isin(val)]
            ent = {kind: float(sub[sub.kind == kind].rescue.mean()) for kind in REF_KINDS}
            hm = sub[sub.kind == "attn_head"].groupby("head").rescue.mean().sort_values(ascending=False)
            ent["head_sum"] = float(hm.sum())
            ent["top5_heads_val"] = {int(h): round(float(v), 3) for h, v in hm.head(5).items()}
            hd = df[(df.layer == l) & df.case_id.isin(dsc) & (df.kind == "attn_head")].groupby("head").rescue.mean().sort_values(ascending=False)
            ent["top5_heads_disc"] = {int(h): round(float(v), 3) for h, v in hd.head(5).items()}
            summ[s][f"L{l}"] = ent
            log(f"set {s} L{l}: attn_layer {ent['attn_layer']:+.3f} layer {ent['layer']:+.3f} block {ent['block']:+.3f} "
                f"sum heads {ent['head_sum']:+.3f}; top heads (val) {ent['top5_heads_val']}")
    with open(os.path.join(od, "head_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    meta.update({"completed_utc": utc(), "pass_time_s": res.extra["total_s"], "n_rows": len(df), "prefill_rows": len(pre),
                 "rows_by_kind": {k_: int(v) for k_, v in df.kind.value_counts().items()}, "complete": True,
                 "outputs": ["head_rows.parquet", "head_prefill.parquet", "sweep_cases.parquet", "sweep_routing.parquet",
                             "head_attn_final.npz", "head_summary.json"],
                 "head_rows_columns": df.columns.tolist()})
    with open(os.path.join(od, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
