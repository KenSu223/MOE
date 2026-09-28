"""ext6 STR pass 3: expert-level tracing with symmetric token replacement.

For every retained case, every selected donor and every requested layer (parent = donor row, clean = clean row):
the layer patch (same-pass reference), clean top-k and routing-union coalitions, an expert patch
delta_e = c_e(clean) - c_e(donor) for EVERY clean-active expert, and (tagged expert_corrupt_only) for experts routed
only in the donor run; optionally equal-norm rows for every ordered pair of clean-active experts (--pairs).
Clean-active sets come from the clean run, so every donor of a case carries the same expert rows and per-case values
are donor means (moetrace.ext6_str.model_data).

Usage: python scripts/ext6_str_expert.py <model> --out <run> --layers 44,42 [--no-special-tokens] [--pairs]
       [--layer-chunks N] [--dry-run]
Outputs results/<run>/str_expert_rows.parquet (donor level; rows of the given layers replaced), str_expert_prefill_L*.parquet
"""
import argparse, itertools, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS
from moetrace.protocol import cases_by_id
from moetrace import ext6_str as S


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build(dn, pos, n, layers, rt_clean, rt_cor, pairs):
    from moetrace.engine import SpawnSpec
    spawns, tags = [], []
    for j, r in enumerate(dn.itertuples()):
        c, s = int(r.case_id), int(r.slot)
        i = pos[c]
        for l in layers:
            ce, ne = rt_clean[(c, l)], rt_cor[(c, s, l)]
            ca = sorted(ce)
            for kind in ("layer", "coalition_clean", "coalition_union"):
                spawns.append(SpawnSpec(l, n + j, i, kind)); tags.append((j, l, kind, -1, -1))
            for e in ca:
                spawns.append(SpawnSpec(l, n + j, i, "expert", expert=e)); tags.append((j, l, "expert", e, -1))
            for e in sorted(set(ne) - set(ce)):
                spawns.append(SpawnSpec(l, n + j, i, "expert", expert=e)); tags.append((j, l, "expert_corrupt_only", e, -1))
            if pairs:
                for a, b in itertools.permutations(ca, 2):
                    spawns.append(SpawnSpec(l, n + j, i, "expert_scaled", expert=a, partner=b)); tags.append((j, l, "expert_scaled", a, b))
    return spawns, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--layer-chunks", type=int, default=1)
    ap.add_argument("--max-spawn", type=int, default=0, help="auto-chunk layers to at most this many spawn rows per pass (overrides --layer-chunks)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    layers = sorted(set(int(x) for x in args.layers.split(",")))
    m = MODELS[args.model]
    st = not args.no_special_tokens
    od = S.run_dir(args.out)
    sets = json.load(open(os.path.join(od, "case_sets.json")))
    ids = sets["paper"]["discovery"] + sets["paper"]["validation"]
    pos = {c: i for i, c in enumerate(ids)}
    n = len(ids)
    dn = S.load_donors(args.out)
    dn = dn[dn.case_id.isin(ids)].reset_index(drop=True)
    rt = pd.read_parquet(os.path.join(od, "str_sweep_routing.parquet"))
    rt = rt[rt.layer.isin(layers)]
    rt_clean, rt_cor = {}, {}
    for (c, s, l), g in rt.groupby(["case_id", "slot", "layer"]):
        dct = dict(zip(g.expert.astype(int).tolist(), g.weight.tolist()))
        if s < 0:
            rt_clean[(int(c), int(l))] = dct
        else:
            rt_cor[(int(c), int(s), int(l))] = dct
    chunks = [[int(x) for x in a] for a in np.array_split(np.array(layers), max(1, min(args.layer_chunks, len(layers))))]
    if args.max_spawn:  # contiguous layer groups with at most max_spawn wavefront rows each
        per = {l: len(build(dn, pos, n, [l], rt_clean, rt_cor, args.pairs)[0]) for l in layers}
        chunks, cur, tot = [], [], 0
        for l in layers:
            if cur and tot + per[l] > args.max_spawn:
                chunks.append(cur); cur, tot = [], 0
            cur.append(l); tot += per[l]
        chunks.append(cur)
    if args.dry_run:
        for ch in chunks:
            sp, _ = build(dn, pos, n, ch, rt_clean, rt_cor, args.pairs)
            log(f"[dry-run] layers {ch}: {n + len(dn)} prefill rows, {len(sp)} spawn rows")
        return
    from moetrace.engine import Engine, PrefillSpec
    cases, rej = cases_by_id(args.model, ids, special_tokens=st)
    assert not rej, rej
    meta = {"model": args.model, "repo": m["repo"], "special_tokens": st, "layers": layers, "pairs": args.pairs,
            "layer_chunks": args.layer_chunks, "command": "python " + " ".join(sys.argv),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    S.write_meta(args.out, f"expert_L{'_'.join(map(str, layers)) if len(layers) <= 6 else f'{layers[0]}-{layers[-1]}'}", meta)
    eng = Engine(m["repo"])
    pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ids]
    pre += [PrefillSpec(r.ids_list, cases[r.case_id].true_id, cases[r.case_id].foil_id) for r in dn.itertuples()]
    path = os.path.join(od, "str_expert_rows.parquet")
    pass_times = []
    for ci, ch in enumerate(chunks):
        spawns, tags = build(dn, pos, n, ch, rt_clean, rt_cor, args.pairs)
        log(f"chunk {ci + 1}/{len(chunks)}: layers {ch}, {len(pre)} prefill rows, {len(spawns)} spawn rows")
        res = eng.run(pre, spawns, record_routing=False, log=log)
        pass_times.append(res.extra["total_s"])
        d, sd = res.delta, res.sp_delta
        rows = []
        for jj, (j, l, kind, e, p) in enumerate(tags):
            r = dn.iloc[j]
            c, s = int(r.case_id), int(r.slot)
            ce, ne = rt_clean[(c, l)], rt_cor[(c, s, l)]
            rows.append(dict(case_id=c, slot=s, layer=l, kind=kind, expert=e, partner=p, alpha=float(res.sp_alpha[jj]),
                             norm_e=float(res.sp_norm_e[jj]), norm_partner=float(res.sp_norm_partner[jj]), vnorm=float(res.sp_vnorm[jj]),
                             delta=float(sd[jj]), delta_clean=float(d[pos[c]]), delta_corrupt=float(d[n + j]),
                             rescue=float(sd[jj] - d[n + j]), clean_active=bool(e in ce), corrupt_active=bool(e in ne),
                             clean_weight=float(ce.get(e, np.nan)), corrupt_weight=float(ne.get(e, np.nan)), n_clean_active=len(ce)))
        df = pd.DataFrame(rows)
        if os.path.exists(path):
            old = pd.read_parquet(path)
            df = pd.concat([old[~old.layer.isin(ch)], df], ignore_index=True)
        df.to_parquet(path, index=False)
        tag = "_".join(map(str, ch)) if len(ch) <= 4 else f"{ch[0]}-{ch[-1]}"
        pd.DataFrame(dict(case_id=ids + dn.case_id.tolist(), slot=[-1] * n + dn.slot.astype(int).tolist(), delta=d)).to_parquet(
            os.path.join(od, f"str_expert_prefill_L{tag}.parquet"), index=False)
        # log recurrence-first selection on the donor-mean rows
        new = df[df.layer.isin(ch) & (df.kind == "expert")]
        agg = new.groupby(["case_id", "layer", "expert"]).rescue.mean().reset_index()
        dsc = sets["paper"]["discovery"]
        for l in ch:
            sub = agg[(agg.layer == l) & agg.case_id.isin(dsc)]
            act = sub.groupby("expert").size()
            allc = sub.groupby("expert").rescue.sum() / len(dsc)
            cand = act[act >= len(dsc) // 2].index
            if len(cand):
                b = allc.loc[cand].idxmax()
                log(f"L{l}: {len(cand)} recurrent candidates (>= {len(dsc) // 2}/{len(dsc)}); best E{int(b):03d} all-case {allc[b]:+.3f} "
                    f"active {act[b]}; top5 {allc.loc[cand].sort_values(ascending=False).head(5).round(3).to_dict()}")
            else:
                log(f"L{l}: no recurrent candidate (max activity {act.max() if len(act) else 0})")
        log(f"chunk {ci + 1}: {path} now {len(df)} rows over layers {sorted(df.layer.unique().tolist())}")
        del res
        torch.cuda.empty_cache()
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True, "pass_times_s": pass_times})
    S.write_meta(args.out, f"expert_L{'_'.join(map(str, layers)) if len(layers) <= 6 else f'{layers[0]}-{layers[-1]}'}", meta)


if __name__ == "__main__":
    main()
