"""ext6 STR pass 2: layer sweep with symmetric token replacement.

Prefill rows: the clean prompt of every retained case + every selected donor prompt (the corrupted runs, no noise).
Spawns: the paper's MoE-block output patch (kind 'layer') at every layer for every donor row (parent = donor row,
clean = the case's clean row). Every row's rescue is measured against its own donor row's Delta in the same pass.
Records final-position routing of all prefill rows (first chunk) for the expert pass.

Usage: python scripts/ext6_str_sweep.py <model> --out <run> [--no-special-tokens] [--layer-chunks N] [--no-metrics]
Outputs results/<run>/str_sweep_rows.parquet (donor level), str_sweep_routing.parquet, sweep_cases.parquet,
        str_sweep_summary.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS
from moetrace.protocol import cases_by_id, membership_table
from moetrace.engine import Engine, PrefillSpec, SpawnSpec
from moetrace import ext6_str as S

METRICS = ("logp_true", "logp_foil", "p_true", "p_foil", "rank_true", "kl_to_clean")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mcols(md, j):
    if md is None:
        return {}
    return {k: (int(md[k][j]) if k == "rank_true" else float(md[k][j])) for k in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--layer-chunks", type=int, default=2)
    ap.add_argument("--no-metrics", action="store_true")
    args = ap.parse_args()
    m = MODELS[args.model]
    st = not args.no_special_tokens
    metrics = not args.no_metrics
    od = S.run_dir(args.out)
    sets = json.load(open(os.path.join(od, "case_sets.json")))
    ids = sets["paper"]["discovery"] + sets["paper"]["validation"]
    cases, rej = cases_by_id(args.model, ids, special_tokens=st)
    assert not rej, rej
    dn = S.load_donors(args.out)
    dn = dn[dn.case_id.isin(ids)].reset_index(drop=True)
    n, nd = len(ids), len(dn)
    pos = {c: i for i, c in enumerate(ids)}
    meta = {"model": args.model, "repo": m["repo"], "special_tokens": st, "layer_chunks": args.layer_chunks, "metrics": metrics,
            "n_cases": n, "n_donor_rows": nd, "command": "python " + " ".join(sys.argv),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    S.write_meta(args.out, "sweep", meta)
    eng = Engine(m["repo"])
    L = eng.spec.n_layers
    pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ids]
    for r in dn.itertuples():
        assert len(r.ids_list) == len(cases[r.case_id].ids)
        pre.append(PrefillSpec(r.ids_list, cases[r.case_id].true_id, cases[r.case_id].foil_id, clean_ref=pos[r.case_id]))
    chunks = [[int(x) for x in a] for a in np.array_split(np.arange(L), max(1, args.layer_chunks))]
    rows, routing, pass_times = [], None, []
    for ci, ch in enumerate(chunks):
        spawns, tags = [], []
        for j, r in enumerate(dn.itertuples()):
            for l in ch:
                spawns.append(SpawnSpec(l, n + j, pos[r.case_id], "layer"))
                tags.append((j, l))
        log(f"chunk {ci + 1}/{len(chunks)}: layers {ch[0]}..{ch[-1]}, {len(pre)} prefill rows, {len(spawns)} spawn rows")
        res = eng.run(pre, spawns, record_routing=(ci == 0), log=log, metrics=metrics)
        pass_times.append(res.extra["total_s"])
        d = res.delta
        mp, ms = (res.metrics_prefill, res.metrics_spawn) if metrics else (None, None)
        if ci == 0:
            for i, c in enumerate(ids):
                rows.append(dict(case_id=c, slot=-1, kind="clean", layer=-1, delta=float(d[i]), logit_true=float(res.logit_true[i]),
                                 logit_foil=float(res.logit_foil[i]), top1=int(res.top1[i]), rescue=np.nan, delta_corrupt=np.nan,
                                 chunk=ci, **mcols(mp, i)))
            for j, r in enumerate(dn.itertuples()):
                rows.append(dict(case_id=r.case_id, slot=int(r.slot), kind="corrupt", layer=-1, delta=float(d[n + j]),
                                 logit_true=float(res.logit_true[n + j]), logit_foil=float(res.logit_foil[n + j]), top1=int(res.top1[n + j]),
                                 rescue=np.nan, delta_corrupt=float(d[n + j]), chunk=ci, **mcols(mp, n + j)))
            k = eng.spec.top_k
            ri, rw, rc = res.route_idx, res.route_w, res.route_cnorm  # [L, B, k]
            row_case = np.array(ids + dn.case_id.tolist())
            row_slot = np.array([-1] * n + dn.slot.astype(int).tolist())
            Ls, Bs, Ks = np.meshgrid(np.arange(L), np.arange(n + nd), np.arange(k), indexing="ij")
            routing = pd.DataFrame(dict(case_id=row_case[Bs.ravel()], slot=row_slot[Bs.ravel()].astype(np.int8),
                                        layer=Ls.ravel().astype(np.int16), kslot=Ks.ravel().astype(np.int8),
                                        expert=ri.ravel().astype(np.int16), weight=rw.ravel(), cnorm=rc.ravel()))
            routing.to_parquet(os.path.join(od, "str_sweep_routing.parquet"), index=False)
        sd = res.sp_delta
        for jj, (j, l) in enumerate(tags):
            r = dn.iloc[j]
            rows.append(dict(case_id=int(r.case_id), slot=int(r.slot), kind="layer", layer=l, delta=float(sd[jj]),
                             logit_true=float(res.sp_logit_true[jj]), logit_foil=float(res.sp_logit_foil[jj]), top1=-1,
                             rescue=float(sd[jj] - d[n + j]), delta_corrupt=float(d[n + j]), chunk=ci, **mcols(ms, jj)))
        pd.DataFrame(rows).to_parquet(os.path.join(od, "str_sweep_rows.parquet"), index=False)
        del res
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    # case table (per-case clean delta, donor-mean corrupt delta, drop)
    cor = df[df.kind == "corrupt"]
    memb = membership_table({"paper": sets["paper"]}, ids)
    ct = pd.DataFrame([cases[c].to_row() for c in ids]).merge(memb, on="case_id")
    ct["delta_clean"] = ct.case_id.map(df[df.kind == "clean"].set_index("case_id").delta)
    ct["delta_corrupt"] = ct.case_id.map(cor.groupby("case_id").delta.mean())
    ct["delta_corrupt_first"] = ct.case_id.map(cor[cor.slot == 0].set_index("case_id").delta)
    ct["delta_corrupt_sd"] = ct.case_id.map(cor.groupby("case_id").delta.std())
    ct["n_donors"] = ct.case_id.map(cor.groupby("case_id").size())
    ct["drop"] = ct.delta_clean - ct.delta_corrupt
    ct["delta_noised"] = ct.delta_corrupt  # alias for moetrace.analysis (the corrupted run)
    ct.to_parquet(os.path.join(od, "sweep_cases.parquet"), index=False)
    # quick summary on the donor-mean layer curve
    R = df[df.kind == "layer"].groupby(["case_id", "layer"]).rescue.mean().unstack()
    dsc, val = sets["paper"]["discovery"], sets["paper"]["validation"]
    mdc, mvl = R.loc[dsc].mean(0), R.loc[val].mean(0)
    lstar = int(mdc.idxmax())
    summ = {"n_cases": n, "n_donor_rows": nd, "L_star_discovery": lstar, "disc_at_Lstar": float(mdc[lstar]), "val_at_Lstar": float(mvl[lstar]),
            "val_argmax": int(mvl.idxmax()), "val_curve": [round(float(x), 3) for x in mvl.values],
            "disc_top5": [(int(l), round(float(v), 3)) for l, v in mdc.sort_values(ascending=False).head(5).items()],
            "mean_delta_clean": float(ct.delta_clean.mean()), "mean_delta_corrupt": float(ct.delta_corrupt.mean()),
            "mean_drop": float(ct["drop"].mean()), "pass_times_s": pass_times}
    with open(os.path.join(od, "str_sweep_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    log(f"L*={lstar} disc {mdc[lstar]:+.3f} val@L* {mvl[lstar]:+.3f} (val argmax L{int(mvl.idxmax())}); disc top5 {summ['disc_top5']}; "
        f"Δclean {summ['mean_delta_clean']:+.2f} Δcorrupt {summ['mean_delta_corrupt']:+.2f} drop {summ['mean_drop']:+.2f}")
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True, "pass_times_s": pass_times, "n_rows": len(df)})
    S.write_meta(args.out, "sweep", meta)


if __name__ == "__main__":
    main()
