"""ext5 F1.3: exhaustive joint patches of every non-empty subset of each case's clean-active experts at a layer.

For every case and requested layer: all 2^k - 1 subsets S of the clean top-k set (k = 8 for Qwen3 -> 255 subsets, k = 2 for
Mixtral -> 3) as `coalition_set` rows (v = sum_{e in S} c_e_clean - c_e_noised), plus `layer`, `coalition_clean` and `block`
reference rows. One engine pass per layer (Qwen3: 256 x 258 = 66k wavefront rows per pass). The clean/noised sets come
from the base run's sweep_routing.parquet (copied into the run dir, as the expert passes do); the pass records its own
routing and the agreement is logged and stored in run_meta.json.

Usage: python scripts/ext5_subsets_run.py <model_key> --out <run> --layers 44,42 --base-run qwen3_metrics [--sets paper]
                                          [--no-special-tokens] [--token-rule space]
Outputs results/<run>/:
  subset_rows.parquet     case_id, layer, kind (coalition_set | layer | coalition_clean | block), experts (sorted, comma-joined;
                          '' for reference kinds), n_experts (0 for reference kinds), logit_true, logit_foil, delta, rescue,
                          vnorm, delta_clean, delta_noised, n_clean_active, sigma_mult      (merged over layers)
  subset_prefill_L<l>.parquet  case_id, delta_clean, delta_noised of that layer's pass
  run_meta.json; inputs copied from the base run: case_sets.json, sweep_cases.parquet, sweep_routing.parquet
"""
import argparse, itertools, json, os, shutil, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS, RESULTS
from moetrace.protocol import out_dir, load_case_sets, cases_by_id
from moetrace.noise import noise_draw

REF_KINDS = ("layer", "coalition_clean", "block")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def subsets(items):
    items = sorted(items)
    for r in range(1, len(items) + 1):
        for comb in itertools.combinations(items, r):
            yield comb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--base-run", required=True, help="run dir with case_sets.json, sweep_cases.parquet, sweep_routing.parquet")
    ap.add_argument("--sets", default="paper")
    ap.add_argument("--sigma-mult", type=float, default=3.0)
    ap.add_argument("--token-rule", default="space")
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--agent", default="ext5-engine")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    m = MODELS[args.model]
    special = not args.no_special_tokens
    layers = [int(x) for x in args.layers.split(",")]
    od = out_dir(args.out)
    copied = []
    for fn in ("case_sets.json", "sweep_cases.parquet", "sweep_routing.parquet"):
        if not os.path.exists(os.path.join(od, fn)):
            shutil.copy(os.path.join(RESULTS, args.base_run, fn), os.path.join(od, fn))
            copied.append(fn)
    sets = load_case_sets(args.out)
    names = [s for s in args.sets.split(",") if s in sets]
    all_ids = []
    for s in names:
        for cid in sets[s]["discovery"] + sets[s]["validation"]:
            if cid not in all_ids:
                all_ids.append(cid)
    cases, rej = cases_by_id(args.model, all_ids, token_rule=args.token_rule, special_tokens=special)
    assert not rej, rej
    ids = [c for c in all_ids if c in cases]
    n = len(ids)
    routing = pd.read_parquet(os.path.join(od, "sweep_routing.parquet"))
    routing = routing[routing.layer.isin(layers) & routing.case_id.isin(ids)]
    rt = {}
    for (cid, run, layer), g in routing.groupby(["case_id", "run", "layer"]):
        rt[(int(cid), run, int(layer))] = sorted(g.expert.astype(int).tolist())
    meta_path = os.path.join(od, "run_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update({"model": args.model, "repo": m["repo"], "base_run": args.base_run, "special_tokens": special, "token_rule": args.token_rule,
                 "sigma_mult": args.sigma_mult, "case_sets": names, "layers": sorted(set(layers) | set(meta.get("layers", []))),
                 "agent": args.agent, "experiment": "ext5 F1.3 exhaustive clean-active expert subsets (coalition_set)",
                 "command": "python " + " ".join(sys.argv), "created_utc": meta.get("created_utc", utc()), "n_cases": n,
                 "inputs_copied_from_base": sorted(set(meta.get("inputs_copied_from_base", [])) | set(copied)), "complete": False})
    from moetrace.engine import Engine, PrefillSpec, SpawnSpec
    if not args.dry_run:
        eng = Engine(m["repo"])
        sigma = args.sigma_mult * eng.embed_std
        Hd = eng.hidden
        pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ids]
        pre += [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id, cases[c].subject_pos,
                            noise_draw(c, len(cases[c].subject_pos), Hd, sigma)) for c in ids]
    path = os.path.join(od, "subset_rows.parquet")
    for l in layers:
        spawns, tags = [], []
        for i, c in enumerate(ids):
            cs = rt[(c, "clean", l)]
            for S in subsets(cs):
                spawns.append(SpawnSpec(l, n + i, i, "coalition_set", experts=tuple(S))); tags.append((c, "coalition_set", S))
            for kind in REF_KINDS:
                spawns.append(SpawnSpec(l, n + i, i, kind)); tags.append((c, kind, ()))
        log(f"{args.model} -> {args.out} L{l}: {n} cases, {len(spawns)} spawn rows")
        if args.dry_run:
            continue
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
        t0 = time.time()
        res = eng.run(pre, spawns, record_routing=True, log=log)
        log(f"pass time {time.time() - t0:.1f}s")
        d, sd = res.delta, res.sp_delta
        agree = int(sum(sorted(res.route_idx[l, i].tolist()) == rt[(ids[i], "clean", l)] for i in range(n)))
        log(f"L{l}: in-pass clean routing equals the base run's for {agree}/{n} cases")
        rows = []
        idx_of = {c: i for i, c in enumerate(ids)}
        for j, (c, kind, S) in enumerate(tags):
            i = idx_of[c]
            rows.append(dict(case_id=c, layer=l, kind=kind, experts=",".join(map(str, S)), n_experts=len(S),
                             logit_true=float(res.sp_logit_true[j]), logit_foil=float(res.sp_logit_foil[j]), delta=float(sd[j]),
                             rescue=float(sd[j] - d[n + i]), vnorm=float(res.sp_vnorm[j]), delta_clean=float(d[i]), delta_noised=float(d[n + i]),
                             n_clean_active=len(rt[(c, "clean", l)]), sigma_mult=args.sigma_mult))
        df = pd.DataFrame(rows)
        if os.path.exists(path):
            old = pd.read_parquet(path)
            df = pd.concat([old[old.layer != l], df], ignore_index=True)
        df.to_parquet(path, index=False)
        pd.DataFrame(dict(case_id=ids, delta_clean=d[:n], delta_noised=d[n:])).to_parquet(os.path.join(od, f"subset_prefill_L{l}.parquet"), index=False)
        meta.setdefault("passes", {})[f"L{l}"] = {"spawn_rows": len(spawns), "pass_time_s": res.extra["total_s"], "T": res.extra["T"],
                                                  "routing_agreement_with_base": f"{agree}/{n}", "completed_utc": utc()}
        # quick summary on the validation split: how often the best single expert / the full clean set reach 80% of block
        for s in names:
            val = [c for c in sets[s]["validation"] if c in idx_of]
            sub = df[(df.layer == l) & df.case_id.isin(val)]
            blk = sub[sub.kind == "block"].set_index("case_id").rescue
            coal = sub[sub.kind == "coalition_clean"].set_index("case_id").rescue
            lay = sub[sub.kind == "layer"].set_index("case_id").rescue
            singles = sub[(sub.kind == "coalition_set") & (sub.n_experts == 1)]
            best1 = singles.groupby("case_id").rescue.max()
            log(f"L{l} set {s} (val, n={len(val)}): block {blk.mean():+.3f} layer {lay.mean():+.3f} coalition_clean {coal.mean():+.3f} "
                f"best single {best1.mean():+.3f}; best single >= 0.8*block in {int((best1 >= 0.8 * blk.loc[best1.index]).sum())}/{len(best1)} cases")
        log(f"L{l}: wrote {len(rows)} rows; {path} holds {len(df)} rows over layers {sorted(df.layer.unique().tolist())}")
        del res
        torch.cuda.empty_cache()
    if not args.dry_run:
        meta.update({"completed_utc": utc(), "complete": True, "n_rows_total": int(len(pd.read_parquet(path))),
                     "outputs": ["subset_rows.parquet", "subset_prefill_L<l>.parquet"],
                     "subset_rows_columns": ["case_id", "layer", "kind", "experts", "n_experts", "logit_true", "logit_foil", "delta", "rescue",
                                             "vnorm", "delta_clean", "delta_noised", "n_clean_active", "sigma_mult"]})
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
