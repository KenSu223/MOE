"""Pass 3: expert-level tracing at the selected layer(s).

For every case of the union set and every requested layer: clean + noised prefill rows; expert patches for EVERY
clean-active expert (and, tagged separately, for noised-only-active experts, whose literal delta_e = -c_e^noised);
equal-norm rows for every ordered pair (a, b) of clean-active experts; coalition rows (clean top-k, routing union);
and the layer patch (for a same-pass reference).

Usage: python scripts/run_expert.py <model_key> --layers 44[,42]
Outputs results/<model>/expert_rows.parquet (appends/replaces rows for the given layers)
"""
import argparse, itertools, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS
from moetrace.protocol import out_dir, load_case_sets, set_names, cases_by_id
from moetrace.engine import Engine, PrefillSpec, SpawnSpec
from moetrace.noise import noise_draw


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--sigma-mult", type=float, default=3.0)
    ap.add_argument("--token-rule", default="space", help="space (default) | paper_like")
    ap.add_argument("--out", default=None, help="results subdir name (default: model key)")
    ap.add_argument("--no-special-tokens", action="store_true", help="tokenise without BOS/special tokens")
    ap.add_argument("--no-pairs", action="store_true")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    m = MODELS[args.model]
    od = out_dir(args.out or args.model)
    sets = load_case_sets(args.out or args.model)
    ct = pd.read_parquet(os.path.join(od, "sweep_cases.parquet"))
    ids = ct.case_id.tolist()
    routing = pd.read_parquet(os.path.join(od, "sweep_routing.parquet"))
    routing = routing[routing.layer.isin(layers)]
    cases, rej = cases_by_id(args.model, ids, token_rule=args.token_rule, special_tokens=not args.no_special_tokens)
    assert not rej, rej
    eng = Engine(m["repo"])
    sigma = args.sigma_mult * eng.embed_std
    Hd = eng.hidden
    n = len(ids)
    pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ids]
    pre += [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id, cases[c].subject_pos,
                        noise_draw(c, len(cases[c].subject_pos), Hd, sigma)) for c in ids]
    spawns, tags = [], []
    rt = {}
    for (cid, run, layer), g in routing.groupby(["case_id", "run", "layer"]):
        rt[(cid, run, layer)] = dict(zip(g.expert.astype(int).tolist(), g.weight.tolist()))
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
            if not args.no_pairs:
                for a, b in itertools.permutations(clean_active, 2):
                    spawns.append(SpawnSpec(l, n + i, i, "expert_scaled", expert=a, partner=b)); tags.append((c, l, "expert_scaled", a, b))
    log(f"{args.model}: {n} cases, layers {layers}, {len(pre)} prefill rows, {len(spawns)} spawn rows")
    t0 = time.time()
    res = eng.run(pre, spawns, record_routing=False, log=log)
    log(f"pass time {time.time() - t0:.1f}s")
    d = res.delta
    sd = res.sp_delta
    rows = []
    for j, (c, l, kind, e, p) in enumerate(tags):
        i = ids.index(c)
        ce = rt[(c, "clean", l)]
        ne = rt[(c, "noised", l)]
        rows.append(dict(case_id=c, layer=l, kind=kind, expert=e, partner=p, alpha=float(res.sp_alpha[j]),
                         norm_e=float(res.sp_norm_e[j]), norm_partner=float(res.sp_norm_partner[j]), vnorm=float(res.sp_vnorm[j]),
                         logit_true=float(res.sp_logit_true[j]), logit_foil=float(res.sp_logit_foil[j]), delta=float(sd[j]),
                         delta_clean=float(d[i]), delta_noised=float(d[n + i]), rescue=float(sd[j] - d[n + i]),
                         clean_active=bool(e in ce), noised_active=bool(e in ne),
                         clean_weight=float(ce.get(e, np.nan)), noised_weight=float(ne.get(e, np.nan)),
                         n_clean_active=len(ce), sigma_mult=args.sigma_mult))
    df = pd.DataFrame(rows)
    # prefill deltas from this pass (for consistency checks against the sweep pass)
    pf = pd.DataFrame(dict(case_id=ids, delta_clean=d[:n], delta_noised=d[n:]))
    path = os.path.join(od, "expert_rows.parquet")
    if os.path.exists(path):
        old = pd.read_parquet(path)
        old = old[~old.layer.isin(layers)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(path, index=False)
    pf.to_parquet(os.path.join(od, f"expert_prefill_L{'_'.join(map(str, layers))}.parquet"), index=False)
    # summary per layer: candidate experts and mean rescue (all sets' discovery, for logging only)
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
                log(f"L{l} set {s}: no candidate meets the recurrence threshold; max activity {act.max()}")
    log(f"wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
