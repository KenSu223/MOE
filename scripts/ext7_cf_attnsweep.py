"""ext7-controls task 1 (GPU): CounterFact STR attention / block sweep at the final position (Phase 3, W2 for the
three-task comparison).

Same cases, donors and prefill rows as the Direction-6 STR layer sweep (scripts/ext6_str_sweep.py): the clean prompt of
every retained case + every selected donor prompt (corrupted run, no noise). Spawns at every layer for every donor row
(parent = donor row, clean = the case's clean row), kinds
    attn_layer  attention-sublayer output at the final position := clean (the MoE of that layer recomputes)
    layer       MoE output := clean (the paper's patch; repeated here so that all three kinds share one pass)
    block       attention + MoE output := clean
Rescue = Delta_patched - Delta(donor row), measured against the donor row of the same pass. Donor level rows; per-case
values are donor means (primary) or the first donor (sensitivity), see moetrace.ext7_controls.

Usage: python scripts/ext7_cf_attnsweep.py <model> --str-run <ext6 run> --out <run> [--layer-chunks N]
Outputs results/<run>/str_attn_rows.parquet (donor level), case_sets.json and sweep_cases.parquet (copied from the STR
run), run_meta.json, str_attn_summary.json
"""
import argparse, json, os, shutil, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS, RESULTS
from moetrace.protocol import cases_by_id
from moetrace.engine import Engine, PrefillSpec, SpawnSpec
from moetrace import ext6_str as S

KINDS = ("attn_layer", "layer", "block")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--str-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer-chunks", type=int, default=4)
    ap.add_argument("--kinds", default=",".join(KINDS))
    args = ap.parse_args()
    kinds = [k for k in args.kinds.split(",") if k]
    m = MODELS[args.model]
    sd_run = os.path.join(RESULTS, args.str_run)
    od = S.run_dir(args.out)
    for f in ("case_sets.json", "sweep_cases.parquet"):
        shutil.copy(os.path.join(sd_run, f), os.path.join(od, f))
    st = json.load(open(os.path.join(sd_run, "run_meta.json")))["sweep"]["special_tokens"]
    sets = json.load(open(os.path.join(od, "case_sets.json")))
    ids = sets["paper"]["discovery"] + sets["paper"]["validation"]
    cases, rej = cases_by_id(args.model, ids, special_tokens=st)
    assert not rej, rej
    dn = S.load_donors(args.str_run)
    dn = dn[dn.case_id.isin(ids)].reset_index(drop=True)
    n, nd = len(ids), len(dn)
    pos = {c: i for i, c in enumerate(ids)}
    meta = {"model": args.model, "repo": m["repo"], "special_tokens": st, "str_run": args.str_run, "kinds": kinds,
            "layer_chunks": args.layer_chunks, "n_cases": n, "n_donor_rows": nd,
            "case_sets": "paper discovery/validation of the STR run (cases with >= 1 selected donor)",
            "experiment": "ext7-controls task 1: CounterFact STR attention / MoE / block sweep at the final position",
            "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "complete": False}
    S.write_meta(args.out, "attnsweep", meta)
    eng = Engine(m["repo"])
    L = eng.spec.n_layers
    pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ids]
    for r in dn.itertuples():
        assert len(r.ids_list) == len(cases[r.case_id].ids)
        pre.append(PrefillSpec(r.ids_list, cases[r.case_id].true_id, cases[r.case_id].foil_id, clean_ref=pos[r.case_id]))
    chunks = [[int(x) for x in a] for a in np.array_split(np.arange(L), max(1, args.layer_chunks))]
    rows, pass_times = [], []
    path = os.path.join(od, "str_attn_rows.parquet")
    for ci, ch in enumerate(chunks):
        spawns, tags = [], []
        for j, r in enumerate(dn.itertuples()):
            for l in ch:
                for k in kinds:
                    spawns.append(SpawnSpec(l, n + j, pos[r.case_id], k))
                    tags.append((j, l, k))
        log(f"chunk {ci + 1}/{len(chunks)}: layers {ch[0]}..{ch[-1]}, {len(pre)} prefill rows, {len(spawns)} spawn rows")
        res = eng.run(pre, spawns, record_routing=False, log=log)
        pass_times.append(res.extra["total_s"])
        d = res.delta
        if ci == 0:
            for i, c in enumerate(ids):
                rows.append(dict(case_id=c, slot=-1, kind="clean", layer=-1, delta=float(d[i]), rescue=np.nan,
                                 delta_corrupt=np.nan, vnorm=np.nan, chunk=ci))
        for j, r in enumerate(dn.itertuples()):  # donor rows in every chunk (same-pass reference)
            rows.append(dict(case_id=int(r.case_id), slot=int(r.slot), kind="corrupt", layer=-1, delta=float(d[n + j]),
                             rescue=np.nan, delta_corrupt=float(d[n + j]), vnorm=np.nan, chunk=ci))
        sdl = res.sp_delta
        for jj, (j, l, k) in enumerate(tags):
            r = dn.iloc[j]
            rows.append(dict(case_id=int(r.case_id), slot=int(r.slot), kind=k, layer=l, delta=float(sdl[jj]),
                             rescue=float(sdl[jj] - d[n + j]), delta_corrupt=float(d[n + j]), vnorm=float(res.sp_vnorm[jj]),
                             chunk=ci))
        pd.DataFrame(rows).to_parquet(path, index=False)
        del res
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    dsc, val = sets["paper"]["discovery"], sets["paper"]["validation"]
    summ = {"n_cases": n, "n_donor_rows": nd, "pass_times_s": pass_times}
    for k in kinds:
        R = df[df.kind == k].groupby(["case_id", "layer"]).rescue.mean().unstack()
        mdc, mvl = R.loc[dsc].mean(0), R.loc[val].mean(0)
        lstar = int(mdc.idxmax())
        summ[k] = {"L_star_discovery": lstar, "disc_at_Lstar": float(mdc[lstar]), "val_at_Lstar": float(mvl[lstar]),
                   "val_argmax": int(mvl.idxmax()), "val_max": float(mvl.max()), "val_curve": [round(float(x), 3) for x in mvl.values]}
        log(f"{k:10s}: L*={lstar} disc {mdc[lstar]:+.3f} val@L* {mvl[lstar]:+.3f} val argmax L{int(mvl.idxmax())} {mvl.max():+.3f}")
    json.dump(summ, open(os.path.join(od, "str_attn_summary.json"), "w"), indent=1)
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True,
                 "pass_times_s": pass_times, "n_rows": len(df), "outputs": ["str_attn_rows.parquet", "str_attn_summary.json"]})
    S.write_meta(args.out, "attnsweep", meta)


if __name__ == "__main__":
    main()
