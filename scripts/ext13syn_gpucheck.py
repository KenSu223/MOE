"""ext13syn GPU check (agent p5-synth, Task B / P5-4): add back every expert except the negative ones, and the late-MoE
reaction test, on Qwen3 add-back tasks (default f1_qwen3, dir_f1_qwen3, dir_opt_qwen3; validation rows).

Negative set Neg = (layer, expert) pairs whose all-case population single-expert rescue on the DISCOVERY cases is < 0
(moetrace/ext13syn_curves.neg_sets; cross-fitted, so the validation rows' noise does not select the set). Late = the last
four layers (L-4..L-1), early = the rest (the A0 'int_moe' split). Variants per validation row (denoising: parent =
corrupted row, source = clean row; engine kind multi, read-only use of moetrace/engine.py; all values same pass):
  all_moe            every MoE output := clean (layer steps at all layers)                       [= ceiling M]
  int_moe            MoE outputs := clean at layers < L-4 only                                    [= M(<= L-5)]
  noneg              every expert := clean except Neg (coalition_set over all experts minus Neg; Neg experts keep the
                     row's own, recomputed output)
  noneg_late         as noneg, Neg excluded only in the last four layers
  noneg_early        as noneg, Neg excluded only below L-4
  negonly            only the Neg experts := clean
  noneg_row          every expert := clean except the row's own negative singles (noisy per-row signs)
  lateonly           MoE := clean in the last four layers only
  intattn            attention outputs := clean at layers < L-4 (MoE recomputed everywhere)
  intattn_latemoe    intattn + MoE := clean in the last four layers
  intblock           attention and MoE := clean at layers < L-4
  intblock_latemoe   intblock + MoE := clean in the last four layers
Reaction test: late-MoE mismatch (recomputed minus clean late MoE) given the upstream restored by MoE only
(int_moe - all_moe), by attention only (intattn - intattn_latemoe) and by both (intblock - intblock_latemoe).

Usage: bash scripts/gpu_queue.sh ext13syn-gpucheck -- python scripts/ext13syn_gpucheck.py [--tasks f1_qwen3,...] [--dry-run]
Outputs results/ext13syn_gpucheck/rows_<task>.parquet, run_meta.json.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
import numpy as np, pandas as pd
from moetrace import ext13ab as AB
from moetrace import ext13syn_curves as C
from moetrace.models import MODELS, RESULTS
from moetrace.arch import load_spec

OD = os.path.join(RESULTS, "ext13syn_gpucheck")
VARIANTS = ("all_moe", "int_moe", "noneg", "noneg_late", "noneg_early", "negonly", "noneg_row", "lateonly",
            "intattn", "intattn_latemoe", "intblock", "intblock_latemoe")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def steps_for(variant, L, E, neg_by_layer, rowneg_by_layer):
    late = set(range(L - 4, L))
    allx = set(range(E))

    def cs(l, excl):
        s = allx - set(excl)
        if len(s) == E:
            return (l, "layer", None)
        if not s:
            return None
        return (l, "coalition_set", tuple(sorted(s)))
    st = []
    for l in range(L):
        if variant == "all_moe":
            st.append((l, "layer", None))
        elif variant == "int_moe":
            if l not in late:
                st.append((l, "layer", None))
        elif variant == "noneg":
            st.append(cs(l, neg_by_layer.get(l, ())))
        elif variant == "noneg_late":
            st.append(cs(l, neg_by_layer.get(l, ())) if l in late else (l, "layer", None))
        elif variant == "noneg_early":
            st.append((l, "layer", None) if l in late else cs(l, neg_by_layer.get(l, ())))
        elif variant == "negonly":
            if neg_by_layer.get(l):
                st.append((l, "coalition_set", tuple(sorted(neg_by_layer[l]))))
        elif variant == "noneg_row":
            st.append(cs(l, rowneg_by_layer.get(l, ())))
        elif variant == "lateonly":
            if l in late:
                st.append((l, "layer", None))
        elif variant == "intattn":
            if l not in late:
                st.append((l, "attn_layer", None))
        elif variant == "intattn_latemoe":
            st.append((l, "layer", None) if l in late else (l, "attn_layer", None))
        elif variant == "intblock":
            if l not in late:
                st.append((l, "block", None))
        elif variant == "intblock_latemoe":
            st.append((l, "layer", None) if l in late else (l, "block", None))
        else:
            raise ValueError(variant)
    return tuple(s for s in st if s is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="f1_qwen3,dir_f1_qwen3,dir_opt_qwen3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(OD, exist_ok=True)
    reg = AB.registry()
    eng = None
    meta = {"agent": "p5-synth", "command": "python " + " ".join(sys.argv), "variants": VARIANTS,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tasks": {}}
    for key in args.tasks.split(","):
        spec = reg[key]
        t_full = AB.load_task(spec)
        neg, pos = C.neg_sets(t_full)
        val = t_full.cases.case_id[t_full.cases.split == "validation"].astype(int).tolist()
        t = AB.subset_task(t_full, val)
        sp, _ = load_spec(MODELS[spec.model]["repo"])
        L, E = t.L, sp.n_experts
        nbl = {}
        for l, e in neg:
            nbl.setdefault(l, set()).add(e)
        nC, R = t.nC, len(t.rows)
        items = []
        for r in t.rows.itertuples():
            rid = int(r.row_id)
            ci = t.case_index[int(r.case_id)]
            rneg = {}
            for (l, e), v in t.single[rid].items():
                if v < 0:
                    rneg.setdefault(l, set()).add(e)
            for v in VARIANTS:
                st = steps_for(v, L, E, nbl, rneg)
                items.append((v, rid, int(r.case_id), ci, st))
        n_neg_late = sum(1 for l, e in neg if l >= L - 4)
        log(f"{key}: {nC} cases, {R} rows, {len(items)} spawns; Neg {len(neg)} pairs ({n_neg_late} in the last 4 layers), Pos {len(pos)}")
        meta["tasks"][key] = {"n_cases": nC, "n_rows": R, "n_spawns": len(items), "n_neg": len(neg), "n_neg_late": n_neg_late, "n_pos": len(pos),
                              "neg_late": sorted(f"L{l}E{e:03d}" for l, e in neg if l >= L - 4)}
        if args.dry_run:
            continue
        import torch
        from moetrace import engine as EN
        if eng is None:
            eng = EN.Engine(MODELS[spec.model]["repo"])
        pre = [EN.PrefillSpec(list(c.clean_ids), int(c.true_id), int(c.foil_id), clean_ref=i) for i, c in enumerate(t.cases.itertuples())]
        for r in t.rows.itertuples():
            c = t.cases.iloc[t.case_index[int(r.case_id)]]
            pre.append(EN.PrefillSpec(list(r.corrupt_ids), int(c.true_id), int(c.foil_id), clean_ref=t.case_index[int(r.case_id)]))
        spawns = [EN.SpawnSpec(st[0][0], nC + rid, ci, "multi", steps=st, kl_ref=ci) for (v, rid, cid, ci, st) in items]
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        res = eng.run(pre, spawns, record_routing=False, log=log)
        d = np.asarray(res.delta)
        dclean = d[:nC]
        dcor = d[nC:]
        df = pd.DataFrame({"variant": [it[0] for it in items], "row_id": [it[1] for it in items], "case_id": [it[2] for it in items],
                           "delta": np.asarray(res.sp_delta, dtype=float)})
        df["delta_clean"] = [float(dclean[it[3]]) for it in items]
        df["delta_corrupt"] = [float(dcor[it[1]]) for it in items]
        df["rescue"] = df.delta - df.delta_corrupt
        df["drop"] = df.delta_clean - df.delta_corrupt
        df.to_parquet(os.path.join(OD, f"rows_{key}.parquet"), index=False)
        g = df.groupby("variant").rescue.sum() / df.groupby("variant")["drop"].sum()
        meta["tasks"][key].update({"pass_s": time.time() - t0, "peak_GB": torch.cuda.max_memory_allocated() / 2**30,
                                   "r": {k: float(v) for k, v in g.items()}})
        log(f"{key}: done in {time.time() - t0:.0f}s; " + ", ".join(f"{k} {v:+.3f}" for k, v in g.items()))
        json.dump(meta, open(os.path.join(OD, "run_meta.json"), "w"), indent=1)
        del res
        torch.cuda.empty_cache()
    meta["complete"] = not args.dry_run
    json.dump(meta, open(os.path.join(OD, "run_meta.json" if not args.dry_run else "run_meta_dryrun.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
