"""Pass 3: expert-level tracing at the selected layer(s).

For every case of the union set and every requested layer: clean + noised prefill rows; expert patches for EVERY
clean-active expert (and, tagged separately, for noised-only-active experts, whose literal delta_e = -c_e^noised);
equal-norm rows for every ordered pair (a, b) of clean-active experts (unless --no-pairs); coalition rows (clean top-k,
routing union); and the layer patch (for a same-pass reference).

Usage: python scripts/run_expert.py <model_key> --layers 44[,42]            (one pass)
       python scripts/run_expert.py <model_key> --layers 0,1,...,47 --no-pairs --layer-chunks 4
The layer list is split into --layer-chunks contiguous ranges, one engine pass each (bounds the number of wavefront
rows resident on the GPU; every chunk re-runs the prefill rows, and each row's rescue is measured against the noised
delta of its own pass). Rows are merged into results/<out>/expert_rows.parquet after every chunk (rows for the given
layers are replaced, other layers kept). --dry-run builds the job and prints row counts without touching the GPU.
"""
import argparse, itertools, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS
from moetrace.protocol import out_dir, load_case_sets, set_names, cases_by_id
from moetrace.noise import noise_draw


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def chunk_layers(layers: list[int], n_chunks: int) -> list[list[int]]:
    n_chunks = max(1, min(n_chunks, len(layers)))
    return [[int(x) for x in a] for a in np.array_split(np.array(layers), n_chunks)]


def build_spawns(ids, layers, rt, n, no_pairs):
    """SpawnSpecs and their tags (case_id, layer, kind, expert, partner) for the given layers."""
    from moetrace.engine import SpawnSpec
    spawns, tags = [], []
    for i, c in enumerate(ids):
        for l in layers:
            ce = rt[(c, "clean", l)]
            ne = rt[(c, "noised", l)]
            clean_active = sorted(ce)
            noised_only = sorted(set(ne) - set(ce))
            spawns.append(SpawnSpec(l, n + i, i, "layer")); tags.append((c, l, "layer", -1, -1))
            spawns.append(SpawnSpec(l, n + i, i, "coalition_clean")); tags.append((c, l, "coalition_clean", -1, -1))
            spawns.append(SpawnSpec(l, n + i, i, "coalition_union")); tags.append((c, l, "coalition_union", -1, -1))
            for e in clean_active:
                spawns.append(SpawnSpec(l, n + i, i, "expert", expert=e)); tags.append((c, l, "expert", e, -1))
            for e in noised_only:
                spawns.append(SpawnSpec(l, n + i, i, "expert", expert=e)); tags.append((c, l, "expert_noised_only", e, -1))
            if not no_pairs:
                for a, b in itertools.permutations(clean_active, 2):
                    spawns.append(SpawnSpec(l, n + i, i, "expert_scaled", expert=a, partner=b)); tags.append((c, l, "expert_scaled", a, b))
    return spawns, tags


def rows_from_result(res, tags, ids, rt, sigma_mult):
    d = res.delta
    sd = res.sp_delta
    n = len(ids)
    pos = {c: i for i, c in enumerate(ids)}
    rows = []
    for j, (c, l, kind, e, p) in enumerate(tags):
        i = pos[c]
        ce = rt[(c, "clean", l)]
        ne = rt[(c, "noised", l)]
        rows.append(dict(case_id=c, layer=l, kind=kind, expert=e, partner=p, alpha=float(res.sp_alpha[j]),
                         norm_e=float(res.sp_norm_e[j]), norm_partner=float(res.sp_norm_partner[j]), vnorm=float(res.sp_vnorm[j]),
                         logit_true=float(res.sp_logit_true[j]), logit_foil=float(res.sp_logit_foil[j]), delta=float(sd[j]),
                         delta_clean=float(d[i]), delta_noised=float(d[n + i]), rescue=float(sd[j] - d[n + i]),
                         clean_active=bool(e in ce), noised_active=bool(e in ne),
                         clean_weight=float(ce.get(e, np.nan)), noised_weight=float(ne.get(e, np.nan)),
                         n_clean_active=len(ce), sigma_mult=sigma_mult))
    return pd.DataFrame(rows)


def layer_tag(layers: list[int]) -> str:
    return "_".join(map(str, layers)) if len(layers) <= 4 else f"{layers[0]}-{layers[-1]}"


def merge_rows(path: str, df: pd.DataFrame, layers: list[int]) -> pd.DataFrame:
    if os.path.exists(path):
        old = pd.read_parquet(path)
        old = old[~old.layer.isin(layers)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(path, index=False)
    return df


def log_selection_summary(df: pd.DataFrame, sets: dict, layers: list[int]):
    """Per layer and set: recurrence-first candidates and the best all-case mean rescue on discovery (logging only)."""
    for l in layers:
        e_rows = df[(df.layer == l) & (df.kind == "expert")]
        for s in set_names(sets):
            dsc = set(sets[s]["discovery"])
            sub = e_rows[e_rows.case_id.isin(dsc)]
            act = sub.groupby("expert").size()
            allcase = sub.groupby("expert").rescue.sum() / len(dsc)  # zero rows for non-active cases
            cand = act[act >= len(dsc) // 2].index
            if len(cand):
                best = allcase.loc[cand].idxmax()
                log(f"L{l} set {s}: {len(cand)} candidates (>= {len(dsc)//2} active); best E{int(best):03d} all-case mean {allcase[best]:+.3f} "
                    f"active {act[best]}/{len(dsc)}; top5: {allcase.loc[cand].sort_values(ascending=False).head(5).round(3).to_dict()}")
            else:
                log(f"L{l} set {s}: no candidate meets the recurrence threshold; max activity {act.max() if len(act) else 0}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--sigma-mult", type=float, default=3.0)
    ap.add_argument("--token-rule", default="space", help="space (default) | paper_like")
    ap.add_argument("--out", default=None, help="results subdir name (default: model key)")
    ap.add_argument("--no-special-tokens", action="store_true", help="tokenise without BOS/special tokens")
    ap.add_argument("--no-pairs", action="store_true")
    ap.add_argument("--layer-chunks", type=int, default=1, help="split the layer list into this many passes")
    ap.add_argument("--dry-run", action="store_true", help="build the job, print row counts, do not run the engine")
    args = ap.parse_args()
    layers = sorted(set(int(x) for x in args.layers.split(",")))
    m = MODELS[args.model]
    od = out_dir(args.out or args.model)
    sets = load_case_sets(args.out or args.model)
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
    chunks = chunk_layers(layers, args.layer_chunks)
    if args.dry_run:
        for ch in chunks:
            spawns, tags = build_spawns(ids, ch, rt, n, args.no_pairs)
            log(f"[dry-run] {args.model}: {n} cases, layers {ch[0]}..{ch[-1]} ({len(ch)}), {2 * n} prefill rows, {len(spawns)} spawn rows")
        return
    from moetrace.engine import Engine, PrefillSpec
    eng = Engine(m["repo"])
    sigma = args.sigma_mult * eng.embed_std
    Hd = eng.hidden
    pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ids]
    pre += [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id, cases[c].subject_pos,
                        noise_draw(c, len(cases[c].subject_pos), Hd, sigma)) for c in ids]
    path = os.path.join(od, "expert_rows.parquet")
    for ci, ch in enumerate(chunks):
        spawns, tags = build_spawns(ids, ch, rt, n, args.no_pairs)
        log(f"{args.model}: chunk {ci + 1}/{len(chunks)}: {n} cases, layers {ch[0]}..{ch[-1]} ({len(ch)}), {len(pre)} prefill rows, {len(spawns)} spawn rows")
        t0 = time.time()
        res = eng.run(pre, spawns, record_routing=False, log=log)
        log(f"pass time {time.time() - t0:.1f}s")
        df = rows_from_result(res, tags, ids, rt, args.sigma_mult)
        d = res.delta
        pf = pd.DataFrame(dict(case_id=ids, delta_clean=d[:n], delta_noised=d[n:]))
        pf.to_parquet(os.path.join(od, f"expert_prefill_L{layer_tag(ch)}.parquet"), index=False)
        all_df = merge_rows(path, df, ch)
        log_selection_summary(df, sets, ch)
        log(f"chunk {ci + 1}: wrote {len(df)} new rows; {path} now holds {len(all_df)} rows over layers {sorted(all_df.layer.unique().tolist())}")
        del res, df
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
