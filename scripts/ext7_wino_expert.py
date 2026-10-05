"""ext7 W6: expert-level tracing at the final position on STR pairs (all layers), in the schema of
results/qwen3_str/str_expert_rows.parquet (ext6) with case_id = directed id and slot = 0.

Per directed case and requested layer (parent = corrupted row, clean = clean row): the layer patch (same-pass
reference), the clean top-k and routing-union coalitions, one expert patch delta_e = c_e(clean) - c_e(corrupted) for
EVERY clean-active expert, and (kind expert_corrupt_only) for experts routed only in the corrupted run; equal-norm rows
(kind expert_scaled) for every ordered pair of clean-active experts at the --pairs-layers. Clean-/corrupt-active sets
come from str_sweep_routing.parquet of the same run (ext7_wino_sweep.py must have run).

--pairs-layers auto = the discovery argmax of the final-position MoE-layer curve (family 'main', or the first family
with a discovery split). --only-scaled computes only the expert_scaled rows of the --pairs-layers and replaces those
rows in the existing file (other rows untouched). --families restricts the directed cases (default: all families of the
run's case_sets.json that have a sweep).

Usage: python scripts/ext7_wino_expert.py --model qwen3 --pairs <parquet> --case-sets <json> --out wino_qwen3_str
           [--layers all|a-b|l1,l2] [--pairs-layers auto|l1,l2] [--only-scaled] [--families main,rep] [--max-spawn 60000]
Outputs results/<out>/str_expert_rows.parquet (rows of the processed layers / kinds replaced), str_expert_prefill.parquet
"""
import argparse, itertools, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import MODELS
from moetrace import ext7_pairs as P


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def parse_layers(s, L):
    if s == "all":
        return list(range(L))
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return sorted(set(int(x) for x in s.split(",")))


def auto_layer(od, fam):
    sw = pd.read_parquet(os.path.join(od, "str_sweep_rows.parquet"), columns=["case_id", "kind", "layer", "rescue"])
    R = sw[sw.kind == "layer"].pivot(index="case_id", columns="layer", values="rescue")
    f = "main" if "main" in fam else next(k for k, v in fam.items() if "discovery" in v)
    return int(R.loc[fam[f]["discovery"]].mean(0).idxmax())


def build(ids, n, layers, rt_clean, rt_cor, pairs_layers, only_scaled, corrupt_only):
    from moetrace.engine import SpawnSpec
    spawns, tags = [], []
    for i, c in enumerate(ids):
        for l in layers:
            ce, ne = rt_clean[(c, l)], rt_cor[(c, l)]
            ca = sorted(ce)
            if not only_scaled:
                for kind in ("layer", "coalition_clean", "coalition_union"):
                    spawns.append(SpawnSpec(l, n + i, i, kind)); tags.append((i, l, kind, -1, -1))
                for e in ca:
                    spawns.append(SpawnSpec(l, n + i, i, "expert", expert=e)); tags.append((i, l, "expert", e, -1))
                if corrupt_only:
                    for e in sorted(set(ne) - set(ce)):
                        spawns.append(SpawnSpec(l, n + i, i, "expert", expert=e)); tags.append((i, l, "expert_corrupt_only", e, -1))
            if l in pairs_layers:
                for a, b in itertools.permutations(ca, 2):
                    spawns.append(SpawnSpec(l, n + i, i, "expert_scaled", expert=a, partner=b)); tags.append((i, l, "expert_scaled", a, b))
    return spawns, tags


def main():
    ap = argparse.ArgumentParser()
    P.add_common_args(ap)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--pairs-layers", default="", help="'auto' or comma list (equal-norm expert_scaled rows)")
    ap.add_argument("--only-scaled", action="store_true")
    ap.add_argument("--families", default=None, help="comma list of families of the run (default: all)")
    ap.add_argument("--no-corrupt-only", action="store_true")
    ap.add_argument("--max-spawn", type=int, default=60000)
    ap.add_argument("--wf-chunk", type=int, default=2048, help="wavefront attention row chunk (memory)")
    args = ap.parse_args()
    m = MODELS[args.model]
    from moetrace.arch import load_spec
    L = load_spec(m["repo"])[0].n_layers
    od = P.run_dir(args.out)
    cs = json.load(open(args.case_sets))
    pairs = P.load_pairs(args.pairs, cs)
    fam_all = P.load_families(args.out)
    fam = {k: v for k, v in fam_all.items() if not args.families or k in args.families.split(",")}
    ids = P.all_ids(fam)
    dc = P.directed_cases(pairs, ids)
    n = len(ids)
    layers = parse_layers(args.layers, L)
    if args.pairs_layers == "auto":
        pairs_layers = [auto_layer(od, fam_all)]
    else:
        pairs_layers = [int(x) for x in args.pairs_layers.split(",") if x != ""]
    if args.only_scaled:
        layers = sorted(pairs_layers)
    rt = pd.read_parquet(os.path.join(od, "str_sweep_routing.parquet"))
    rt = rt[rt.layer.isin(layers) & rt.case_id.isin(ids)]
    rt_clean, rt_cor = {}, {}
    for (c, s, l), g in rt.groupby(["case_id", "slot", "layer"]):
        dct = dict(zip(g.expert.astype(int).tolist(), g.weight.tolist()))
        (rt_clean if s < 0 else rt_cor)[(int(c), int(l))] = dct
    corrupt_only = not args.no_corrupt_only
    per = {l: len(build(ids, n, [l], rt_clean, rt_cor, pairs_layers, args.only_scaled, corrupt_only)[0]) for l in layers}
    chunks, cur, tot = [], [], 0
    for l in layers:
        if cur and tot + per[l] > args.max_spawn:
            chunks.append(cur); cur, tot = [], 0
        cur.append(l); tot += per[l]
    chunks.append(cur)
    log(f"{args.model} -> {args.out}: {n} directed cases, families {list(fam)}, layers {layers[0]}..{layers[-1]} ({len(layers)}), "
        f"pairs layers {pairs_layers}, only_scaled={args.only_scaled}, {sum(per.values())} spawn rows in {len(chunks)} passes")
    if args.dry_run:
        for ch in chunks:
            log(f"[dry-run] layers {ch}: {sum(per[l] for l in ch)} spawn rows")
        return
    import torch
    from moetrace.engine import Engine, PrefillSpec
    key = "expert_scaled" if args.only_scaled else "expert"
    meta = {"model": args.model, "repo": m["repo"], "families": list(fam), "n_cases": n, "layers": layers, "pairs_layers": pairs_layers,
            "only_scaled": args.only_scaled, "corrupt_only": corrupt_only, "max_spawn": args.max_spawn, "chunks": chunks, "agent": args.agent,
            "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    P.write_meta(args.out, key, meta)
    eng = Engine(m["repo"])
    pre = [PrefillSpec(dc[c].clean_ids, dc[c].true_id, dc[c].foil_id) for c in ids]
    pre += [PrefillSpec(dc[c].corrupt_ids, dc[c].true_id, dc[c].foil_id) for c in ids]
    path = os.path.join(od, "str_expert_rows.parquet")
    pass_times = []
    for ci, ch in enumerate(chunks):
        spawns, tags = build(ids, n, ch, rt_clean, rt_cor, pairs_layers, args.only_scaled, corrupt_only)
        log(f"pass {ci + 1}/{len(chunks)}: layers {ch}, {len(pre)} prefill rows, {len(spawns)} spawn rows")
        res = eng.run(pre, spawns, record_routing=False, log=log if ci == 0 else None, wf_chunk=args.wf_chunk)
        pass_times.append(res.extra["total_s"])
        d, sd = res.delta, res.sp_delta
        rows = []
        for jj, (i, l, kind, e, pa) in enumerate(tags):
            c = ids[i]
            ce, ne = rt_clean[(c, l)], rt_cor[(c, l)]
            rows.append(dict(case_id=c, slot=0, layer=l, kind=kind, expert=e, partner=pa, alpha=float(res.sp_alpha[jj]),
                             norm_e=float(res.sp_norm_e[jj]), norm_partner=float(res.sp_norm_partner[jj]), vnorm=float(res.sp_vnorm[jj]),
                             delta=float(sd[jj]), delta_clean=float(d[i]), delta_corrupt=float(d[n + i]), rescue=float(sd[jj] - d[n + i]),
                             clean_active=bool(e in ce), corrupt_active=bool(e in ne), clean_weight=float(ce.get(e, np.nan)),
                             corrupt_weight=float(ne.get(e, np.nan)), n_clean_active=len(ce)))
        df = pd.DataFrame(rows)
        if os.path.exists(path):
            old = pd.read_parquet(path)
            if args.only_scaled:
                drop = old.layer.isin(ch) & (old.kind == "expert_scaled") & old.case_id.isin(ids)
            else:
                drop = old.layer.isin(ch) & old.case_id.isin(ids) & ((old.kind != "expert_scaled") | old.layer.isin(pairs_layers))
            df = pd.concat([old[~drop], df], ignore_index=True)
        df.to_parquet(path, index=False)
        pd.DataFrame(dict(case_id=ids + ids, slot=[-1] * n + [0] * n, delta=d)).to_parquet(os.path.join(od, "str_expert_prefill.parquet"), index=False)
        if not args.only_scaled and "main" in fam:
            new = df[df.layer.isin(ch) & (df.kind == "expert") & df.case_id.isin(fam["main"]["discovery"])]
            dsc = fam["main"]["discovery"]
            for l in ch:
                sub = new[new.layer == l]
                act = sub.groupby("expert").size()
                allc = sub.groupby("expert").rescue.sum() / len(dsc)
                cand = act[act >= len(dsc) // 2].index
                if len(cand):
                    b = allc.loc[cand].idxmax()
                    log(f"  L{l}: {len(cand)} recurrent (>= {len(dsc) // 2}/{len(dsc)}); best E{int(b):03d} all-case {allc[b]:+.3f} active {act[b]}")
        log(f"pass {ci + 1}: {res.extra['total_s']:.0f}s; {path} now {len(df)} rows")
        del res
        torch.cuda.empty_cache()
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True, "pass_times_s": pass_times})
    P.write_meta(args.out, key, meta)


if __name__ == "__main__":
    main()
