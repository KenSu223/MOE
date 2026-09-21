"""ext5 F4 layer sweep at the LAST SUBJECT TOKEN (RESEARCH_PLAN Phase 2, F4; modelled on scripts/ext2_attn_sweep.py).

For every case of the chosen case set(s): prefill rows clean, noised (whole subject span), noised_lastonly (the same
draw restricted to the last subject token) and noised_exceptlast (the rest of the span), all recording position
p = last subject token; and, at every layer, one suffix wavefront row per kind in --kinds patched at p (see
moetrace/ext5_subject.py). The layer list is split into --layer-chunks passes (auto: padded suffix token-rows per pass
<= --token-budget); every pass re-runs the prefill rows and each row's rescue is measured against the noised delta of its
own pass (as run_expert.py does).

Usage: python scripts/ext5_subject_sweep.py <model_key> --out <run> [--base-run <run with case_sets.json>]
                                            [--sets paper] [--kinds attn_layer,layer,resid] [--no-special-tokens]
Outputs results/<run>/: sweep_rows.parquet (kinds clean, noised, noised_lastonly, noised_exceptlast, <kinds>; columns as
ext2 plus pos, pass_chunk), sweep_routing.parquet (routing AT p, + pos), sweep_cases.parquet (+ pos, delta_noised_lastonly,
delta_noised_exceptlast, drop_lastonly, drop_exceptlast), sweep_summary.json, run_meta.json.
"""
import argparse, json, math, os, shutil, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS, RESULTS
from moetrace.protocol import out_dir, load_case_sets, cases_by_id, membership_table
from moetrace.noise import noise_draw
from moetrace.ext5_subject import SubjectEngine, SubjectSpawn, KINDS, subject_prefill_rows, last_subject_pos

DEFAULT_KINDS = ["attn_layer", "layer", "resid"]
PREFILL_KINDS = ["clean", "noised", "noised_lastonly", "noised_exceptlast"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_layers(spec: str, L: int) -> list[int]:
    if not spec:
        return list(range(L))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sigma-mult", type=float, default=3.0)
    ap.add_argument("--token-rule", default="space")
    ap.add_argument("--out", required=True, help="results subdir, e.g. qwen3_bos_subject")
    ap.add_argument("--base-run", default=None, help="run dir whose case_sets.json is used (default: model key)")
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--sets", default="paper")
    ap.add_argument("--kinds", default=",".join(DEFAULT_KINDS))
    ap.add_argument("--layers", default=None)
    ap.add_argument("--layer-chunks", type=int, default=0, help="0 = auto from --token-budget")
    ap.add_argument("--token-budget", type=int, default=150_000, help="max padded suffix token-rows per pass")
    ap.add_argument("--agent", default="ext5-subject")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    m = MODELS[args.model]
    special = not args.no_special_tokens
    run, base = args.out, args.base_run or args.model
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
    cases, rej = cases_by_id(args.model, all_ids, token_rule=args.token_rule, special_tokens=special)
    if rej:
        log(f"WARNING: {len(rej)} case ids not tokenizable: {rej}")
    ids = [c for c in all_ids if c in cases]
    n = len(ids)
    from moetrace.arch import load_spec
    spec, _ = load_spec(m["repo"])
    L = spec.n_layers
    layers = parse_layers(args.layers, L)
    S_of = {c: len(cases[c].ids) - last_subject_pos(cases[c]) for c in ids}
    Smax = max(S_of.values())
    n_spawn = n * len(layers) * len(kinds)
    n_chunks = args.layer_chunks or max(1, math.ceil(n_spawn * Smax / args.token_budget))
    chunks = [[int(x) for x in a] for a in np.array_split(np.array(layers), min(n_chunks, len(layers)))]
    meta = {"model": args.model, "repo": m["repo"], "base_run": base, "special_tokens": special, "token_rule": args.token_rule,
            "sigma_mult": args.sigma_mult, "case_sets": names, "kinds": kinds, "agent": args.agent, "patch_position": "last subject token",
            "experiment": "ext5 F4: causal tracing at the last subject token (suffix wavefront rows)",
            "command": "python " + " ".join(sys.argv), "created_utc": utc(), "n_cases": n, "rejected": rej, "complete": False,
            "layers": f"{layers[0]}-{layers[-1]} ({len(layers)} of {L})", "layer_chunks": chunks, "Smax": Smax,
            "suffix_tokens_per_layer_kind": int(sum(S_of.values())), "padded_token_rows_total": int(n_spawn * Smax)}
    log(f"{args.model} -> {run}: sets={names} n={n} kinds={kinds} special={special} layers={len(layers)} chunks={len(chunks)} "
        f"Smax={Smax} spawn rows={n_spawn} suffix tokens={meta['suffix_tokens_per_layer_kind'] * len(layers) * len(kinds)}")
    if args.dry_run:
        print(json.dumps(meta, indent=1))
        return
    with open(os.path.join(od, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    eng = SubjectEngine(m["repo"])
    sigma = args.sigma_mult * eng.embed_std
    Hd = eng.hidden
    pre, offs = subject_prefill_rows(cases, ids, sigma, Hd, last_subject_pos, noise_draw)
    rows, routing_df, pass_times = [], None, []
    for ci, ch in enumerate(chunks):
        spawns, tags = [], []
        for i, c in enumerate(ids):
            p = last_subject_pos(cases[c])
            for l in ch:
                for kind in kinds:
                    spawns.append(SubjectSpawn(l, offs["noised"] + i, offs["clean"] + i, kind, p))
                    tags.append((c, l, kind, p))
        log(f"chunk {ci + 1}/{len(chunks)}: layers {ch[0]}..{ch[-1]}, {len(pre)} prefill rows, {len(spawns)} suffix rows")
        t0 = time.time()
        res = eng.run_subject(pre, spawns, log=log)
        pass_times.append(res.extra["total_s"])
        log(f"pass time {time.time() - t0:.1f}s")
        d = res.delta
        dfull = res.logit_true_full - res.logit_foil_full
        if ci == 0:
            for i, c in enumerate(ids):
                p = last_subject_pos(cases[c])
                for kind in PREFILL_KINDS:
                    j = offs[kind] + i
                    rows.append(dict(case_id=c, kind=kind, layer=-1, pos=p, sigma_mult=args.sigma_mult, logit_true=float(res.logit_true[j]),
                                     logit_foil=float(res.logit_foil[j]), delta=float(d[j]), delta_full=float(dfull[j]), top1=int(res.top1[j]),
                                     rescue=np.nan, vnorm=np.nan, pass_chunk=ci))
            k = eng.spec.top_k
            rr = []
            for run_, off in (("clean", offs["clean"]), ("noised", offs["noised"])):
                ri, rw, rc = res.route_idx[:, off : off + n], res.route_w[:, off : off + n], res.route_cnorm[:, off : off + n]
                Ls, Ns, Ks = np.meshgrid(np.arange(L), np.arange(n), np.arange(k), indexing="ij")
                rr.append(pd.DataFrame(dict(case_id=np.array(ids)[Ns.ravel()], run=run_, layer=Ls.ravel().astype(np.int16),
                                            slot=Ks.ravel().astype(np.int8), expert=ri.ravel().astype(np.int16), weight=rw.ravel(),
                                            cnorm=rc.ravel(), pos=np.array([last_subject_pos(cases[c]) for c in ids])[Ns.ravel()])))
            routing_df = pd.concat(rr)
        sd = res.sp_delta
        idx_of = {c: i for i, c in enumerate(ids)}
        for j, (c, l, kind, p) in enumerate(tags):
            i = idx_of[c]
            rows.append(dict(case_id=c, kind=kind, layer=l, pos=p, sigma_mult=args.sigma_mult, logit_true=float(res.sp_logit_true[j]),
                             logit_foil=float(res.sp_logit_foil[j]), delta=float(sd[j]), delta_full=np.nan, top1=-1,
                             rescue=float(sd[j] - d[offs["noised"] + i]), vnorm=float(res.sp_vnorm[j]), pass_chunk=ci))
        df = pd.DataFrame(rows)
        df.to_parquet(os.path.join(od, "sweep_rows.parquet"), index=False)  # saved after every pass
        del res
        torch.cuda.empty_cache()
    routing_df.to_parquet(os.path.join(od, "sweep_routing.parquet"), index=False)
    memb = membership_table(sets, ids)
    ct = pd.DataFrame([cases[c].to_row() for c in ids]).merge(memb, on="case_id")
    ct["pos"] = ct.case_id.map({c: last_subject_pos(cases[c]) for c in ids})
    for kind in PREFILL_KINDS:
        ct["delta_" + kind] = ct.case_id.map(df[df.kind == kind].set_index("case_id").delta)
    ct["drop"] = ct.delta_clean - ct.delta_noised
    ct["drop_lastonly"] = ct.delta_clean - ct.delta_noised_lastonly
    ct["drop_exceptlast"] = ct.delta_clean - ct.delta_noised_exceptlast
    ct["strict"] = (ct.delta_clean >= 1.0) & (ct["drop"] >= 0.5)
    ct["relaxed"] = (ct.delta_clean >= 0.5) & (ct["drop"] >= 0.25)
    ct.to_parquet(os.path.join(od, "sweep_cases.parquet"), index=False)
    summ = {"n_cases": n, "pass_times_s": pass_times, "rejected": rej, "kinds": kinds, "Smax": Smax,
            "drop_mean": float(ct["drop"].mean()), "drop_lastonly_mean": float(ct.drop_lastonly.mean()),
            "drop_exceptlast_mean": float(ct.drop_exceptlast.mean())}
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
                             "disc_curve": [round(float(x), 3) for x in md_.values], "val_curve": [round(float(x), 3) for x in mv.values]}
            log(f"set {s} kind {kind:10s}: L*={lstar:2d} disc={md_[lstar]:+.3f} val@L*={mv[lstar]:+.3f} val argmax L{int(mv.idxmax())} {mv.max():+.3f}")
    log(f"drop: full {summ['drop_mean']:+.3f} last-only {summ['drop_lastonly_mean']:+.3f} except-last {summ['drop_exceptlast_mean']:+.3f}")
    with open(os.path.join(od, "sweep_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    meta.update({"completed_utc": utc(), "pass_times_s": pass_times, "n_rows": len(df), "prefill_rows": len(pre),
                 "spawn_rows": int(n_spawn), "rows_by_kind": {k_: int(v) for k_, v in df.kind.value_counts().items()}, "complete": True,
                 "outputs": ["sweep_rows.parquet", "sweep_routing.parquet", "sweep_cases.parquet", "sweep_summary.json"]})
    with open(os.path.join(od, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
