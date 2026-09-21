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


METRICS = ("logp_true", "logp_foil", "p_true", "p_foil", "rank_true", "kl_to_clean")


def _mcols(md, j: int) -> dict:
    """ext5: the six metric columns of row j (empty when metrics are off)."""
    if md is None:
        return {}
    return {k: (int(md[k][j]) if k == "rank_true" else float(md[k][j])) for k in METRICS}


def rows_from_result(res, tags, ids, rt, sigma_mult, metrics: bool = False):
    d = res.delta
    sd = res.sp_delta
    n = len(ids)
    pos = {c: i for i, c in enumerate(ids)}
    mp, ms = (res.metrics_prefill, res.metrics_spawn) if metrics else (None, None)
    rows = []
    for j, (c, l, kind, e, p) in enumerate(tags):
        i = pos[c]
        ce = rt[(c, "clean", l)]
        ne = rt[(c, "noised", l)]
        extra = _mcols(ms, j)
        if mp is not None:  # the noised prefill row's values of the same pass (for probability-scale rescue)
            extra.update(p_true_noised=float(mp["p_true"][n + i]), logp_true_noised=float(mp["logp_true"][n + i]))
        rows.append(dict(case_id=c, layer=l, kind=kind, expert=e, partner=p, alpha=float(res.sp_alpha[j]),
                         norm_e=float(res.sp_norm_e[j]), norm_partner=float(res.sp_norm_partner[j]), vnorm=float(res.sp_vnorm[j]),
                         logit_true=float(res.sp_logit_true[j]), logit_foil=float(res.sp_logit_foil[j]), delta=float(sd[j]),
                         delta_clean=float(d[i]), delta_noised=float(d[n + i]), rescue=float(sd[j] - d[n + i]),
                         clean_active=bool(e in ce), noised_active=bool(e in ne),
                         clean_weight=float(ce.get(e, np.nan)), noised_weight=float(ne.get(e, np.nan)),
                         n_clean_active=len(ce), sigma_mult=sigma_mult, **extra))
    return pd.DataFrame(rows)


def _json_safe(o):
    """Keep only JSON-native values (numpy scalars/arrays converted; DataFrames and other objects dropped as their type name)."""
    import numpy as _np
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return f"<dropped {type(o).__name__}>"


def _write_meta(od: str, upd: dict):
    """Merge upd into results/<run>/run_meta.json (ext5; only written when --agent or --metrics is given).
    A corrupt existing file is replaced; values are sanitised to JSON-native types."""
    p = os.path.join(od, "run_meta.json")
    cur = {}
    if os.path.exists(p):
        try:
            cur = json.load(open(p))
        except Exception:
            cur = {}
    cur.update(_json_safe(upd))
    with open(p, "w") as f:
        json.dump(cur, f, indent=1, default=str)


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
    ap.add_argument("--metrics", action="store_true",
                    help="ext5 (F5): full-vocabulary metrics on every row (+ p_true_noised, logp_true_noised of the same pass)")
    ap.add_argument("--agent", default=None, help="agent name recorded in run_meta.json (written when --agent or --metrics is given)")
    args = ap.parse_args()
    layers = sorted(set(int(x) for x in args.layers.split(",")))
    m = MODELS[args.model]
    od = out_dir(args.out or args.model)
    write_meta = bool(args.agent) or args.metrics
    meta = {"model": args.model, "repo": m["repo"], "special_tokens": not args.no_special_tokens, "token_rule": args.token_rule,
            "sigma_mult": args.sigma_mult, "layers": layers, "pairs": not args.no_pairs, "layer_chunks": args.layer_chunks,
            "metrics": args.metrics, "agent": args.agent, "command": "python " + " ".join(sys.argv),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
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
    if write_meta:
        _write_meta(od, {"expert": meta})
    pass_times = []
    for ci, ch in enumerate(chunks):
        spawns, tags = build_spawns(ids, ch, rt, n, args.no_pairs)
        log(f"{args.model}: chunk {ci + 1}/{len(chunks)}: {n} cases, layers {ch[0]}..{ch[-1]} ({len(ch)}), {len(pre)} prefill rows, {len(spawns)} spawn rows")
        t0 = time.time()
        res = eng.run(pre, spawns, record_routing=False, log=log, metrics=args.metrics)
        log(f"pass time {time.time() - t0:.1f}s" + (f" (metrics {res.extra.get('metrics_s', 0):.1f}s)" if args.metrics else ""))
        pass_times.append(res.extra["total_s"])
        df = rows_from_result(res, tags, ids, rt, args.sigma_mult, metrics=args.metrics)
        d = res.delta
        pf = pd.DataFrame(dict(case_id=ids, delta_clean=d[:n], delta_noised=d[n:]))
        if args.metrics:  # the prefill rows' full-vocabulary metrics of this pass (clean_ref of noised row n+i is row i)
            for k in METRICS:
                pf[f"{k}_clean"] = res.metrics_prefill[k][:n]
                pf[f"{k}_noised"] = res.metrics_prefill[k][n:]
        pf.to_parquet(os.path.join(od, f"expert_prefill_L{layer_tag(ch)}.parquet"), index=False)
        all_df = merge_rows(path, df, ch)
        log_selection_summary(df, sets, ch)
        log(f"chunk {ci + 1}: wrote {len(df)} new rows; {path} now holds {len(all_df)} rows over layers {sorted(all_df.layer.unique().tolist())}")
        del res, df
        torch.cuda.empty_cache()
    if write_meta:
        meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_cases": n, "pass_times_s": pass_times,
                     "n_rows_total": len(all_df), "columns": all_df.columns.tolist(), "complete": True})
        _write_meta(od, {"expert": meta})


if __name__ == "__main__":
    main()
