"""ext5 F4 expert-level pass at the LAST SUBJECT TOKEN (modelled on scripts/run_expert.py, no equal-norm pairs).

For every case of the subject run's case table and every requested layer, suffix rows patched at p = last subject token:
kinds layer, coalition_clean, coalition_union, block, attn_layer, resid (references); `expert` for every clean-active
expert at p (routing from the subject sweep's sweep_routing.parquet, recorded AT p); `expert_noised_only` for experts
routed only in the noised run; and `expert_fixed` for every --fixed layer:expert hypothesis (patched for EVERY case,
active or not: delta_e = 0 when the expert is routed in neither run).

Usage: python scripts/ext5_subject_expert.py <model_key> --out <subject run> --layers 40,42,44 [--fixed 44:69,42:115]
                                             [--no-special-tokens] [--layer-chunks 1]
Output: results/<run>/expert_rows.parquet (run_expert.py schema + pos), expert_prefill_L*.parquet, expert_meta.json.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS
from moetrace.protocol import out_dir, cases_by_id
from moetrace.noise import noise_draw
from moetrace.ext5_subject import SubjectEngine, SubjectSpawn, subject_prefill_rows, last_subject_pos

REF_KINDS = ("layer", "coalition_clean", "coalition_union", "block", "attn_layer", "resid")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_spawns(ids, cases, layers, rt, offs, fixed):
    spawns, tags = [], []
    fx = {}
    for l, e in fixed:
        fx.setdefault(l, []).append(e)
    for i, c in enumerate(ids):
        p = last_subject_pos(cases[c])
        par, cln = offs["noised"] + i, offs["clean"] + i
        for l in layers:
            ce, ne = rt[(c, "clean", l)], rt[(c, "noised", l)]
            clean_active = sorted(ce)
            noised_only = sorted(set(ne) - set(ce))
            for kind in REF_KINDS:
                spawns.append(SubjectSpawn(l, par, cln, kind, p)); tags.append((c, l, kind, -1))
            for e in clean_active:
                spawns.append(SubjectSpawn(l, par, cln, "expert", p, expert=e)); tags.append((c, l, "expert", e))
            for e in noised_only:
                spawns.append(SubjectSpawn(l, par, cln, "expert", p, expert=e)); tags.append((c, l, "expert_noised_only", e))
            for e in fx.get(l, []):
                spawns.append(SubjectSpawn(l, par, cln, "expert", p, expert=e)); tags.append((c, l, "expert_fixed", e))
    return spawns, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--fixed", default="", help="layer:expert pairs patched for every case, e.g. 44:69,42:115")
    ap.add_argument("--sigma-mult", type=float, default=3.0)
    ap.add_argument("--token-rule", default="space")
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--layer-chunks", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    layers = sorted(set(int(x) for x in args.layers.split(",")))
    fixed = [(int(a), int(b)) for a, b in (x.split(":") for x in args.fixed.split(",") if x)]
    m = MODELS[args.model]
    od = out_dir(args.out)
    ct = pd.read_parquet(os.path.join(od, "sweep_cases.parquet"))
    ids = ct.case_id.tolist()
    routing = pd.read_parquet(os.path.join(od, "sweep_routing.parquet"))
    routing = routing[routing.layer.isin(layers)]
    cases, rej = cases_by_id(args.model, ids, token_rule=args.token_rule, special_tokens=not args.no_special_tokens)
    assert not rej, rej
    n = len(ids)
    rt = {}
    for (cid, run, layer), g in routing.groupby(["case_id", "run", "layer"]):
        rt[(int(cid), run, int(layer))] = dict(zip(g.expert.astype(int).tolist(), g.weight.tolist()))
    chunks = [[int(x) for x in a] for a in np.array_split(np.array(layers), max(1, min(args.layer_chunks, len(layers))))]
    offs = {"clean": 0, "noised": n}
    if args.dry_run:
        for ch in chunks:
            sp, _ = build_spawns(ids, cases, ch, rt, offs, fixed)
            log(f"[dry-run] layers {ch}: {len(sp)} suffix rows")
        return
    meta = {"model": args.model, "repo": m["repo"], "run": args.out, "layers": layers, "fixed": fixed, "special_tokens": not args.no_special_tokens,
            "patch_position": "last subject token", "command": "python " + " ".join(sys.argv), "created_utc": utc(), "complete": False}
    with open(os.path.join(od, "expert_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    eng = SubjectEngine(m["repo"])
    sigma = args.sigma_mult * eng.embed_std
    pre, offs = subject_prefill_rows(cases, ids, sigma, eng.hidden, last_subject_pos, noise_draw, extra_noise_rows=False)
    path = os.path.join(od, "expert_rows.parquet")
    all_df = None
    for ci, ch in enumerate(chunks):
        spawns, tags = build_spawns(ids, cases, ch, rt, offs, fixed)
        log(f"chunk {ci + 1}/{len(chunks)}: layers {ch}, {len(pre)} prefill rows, {len(spawns)} suffix rows")
        t0 = time.time()
        res = eng.run_subject(pre, spawns, log=log)
        log(f"pass time {time.time() - t0:.1f}s")
        d, sd = res.delta, res.sp_delta
        pos_of = {c: i for i, c in enumerate(ids)}
        rows = []
        for j, (c, l, kind, e) in enumerate(tags):
            i = pos_of[c]
            ce, ne = rt[(c, "clean", l)], rt[(c, "noised", l)]
            rows.append(dict(case_id=c, layer=l, kind=kind, expert=e, partner=-1, alpha=1.0, norm_e=float(res.sp_norm_e[j]), norm_partner=0.0,
                             vnorm=float(res.sp_vnorm[j]), logit_true=float(res.sp_logit_true[j]), logit_foil=float(res.sp_logit_foil[j]),
                             delta=float(sd[j]), delta_clean=float(d[i]), delta_noised=float(d[n + i]), rescue=float(sd[j] - d[n + i]),
                             clean_active=bool(e in ce), noised_active=bool(e in ne), clean_weight=float(ce.get(e, np.nan)),
                             noised_weight=float(ne.get(e, np.nan)), n_clean_active=len(ce), sigma_mult=args.sigma_mult,
                             pos=last_subject_pos(cases[c])))
        df = pd.DataFrame(rows)
        pd.DataFrame(dict(case_id=ids, delta_clean=d[:n], delta_noised=d[n:])).to_parquet(
            os.path.join(od, f"expert_prefill_L{'_'.join(map(str, ch))}.parquet"), index=False)
        if os.path.exists(path):
            old = pd.read_parquet(path)
            df = pd.concat([old[~old.layer.isin(ch)], df], ignore_index=True)
        df.to_parquet(path, index=False)
        all_df = df
        for l in ch:
            e_rows = df[(df.layer == l) & (df.kind == "expert")]
            act = e_rows.groupby("expert").size()
            allcase = e_rows.groupby("expert").rescue.sum() / n
            top = allcase.sort_values(ascending=False).head(5)
            log(f"L{l}: {len(act)} distinct clean-active experts at p; max activity {act.max()}/{n}; top5 all-case mean: "
                + ", ".join(f"E{int(e):03d} {v:+.3f} ({act[e]})" for e, v in top.items()))
        del res
        torch.cuda.empty_cache()
    meta.update({"completed_utc": utc(), "n_rows": int(len(all_df)), "rows_by_kind": {k: int(v) for k, v in all_df.kind.value_counts().items()},
                 "complete": True})
    with open(os.path.join(od, "expert_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
