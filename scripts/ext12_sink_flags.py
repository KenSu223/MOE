"""ext12 4c: sink-carrying final-token flags (Direction 3) for the WinoGrande STR prompts, Mixtral without and with BOS.

Direction 3 found that without <s> Mixtral forms no position-0 attention sink; the massive-norm "sink state" lands on the
first trigger token of the prompt, sometimes the final one, and a final position that carries the sink state is routed
like <s> (E006 at L19). Flag used there (results/sections/ext3_bos_mechanism.md, Experiment 1b): the final position has the
maximal residual norm over all positions at layer 5. Here one prefill-only pass over prompts A and B of every pair of the
chosen case-set splits, under both protocols (pairs_train_xl_mixtral_nobos / _bos: identical text, the BOS file prepends
<s>), with DiagSpec(resid_norms, attn_final) and the final-position routing of every layer.

Per prompt (row of sink_flags.parquet): pair_idx, prompt (a | b), proto (nobos | bos), T, final token id,
  argmax_pos_L5 / final_is_max_L5 (Direction-3 flag), final_is_max_any (any layer 1 .. L/2 - 1), pos0_is_max_L5,
  norm_ratio_L5 = final norm / median norm of the other positions, self_mass_L1 / pos0_mass_L1 (head-mean final-position
  attention on itself / on position 0 at layer 1), top-k experts at the final position for every layer (route_L<l>, JSON).
Also writes sink_diag.json (moetrace.ext2_zoo.sink_summary per protocol) next to it.

Usage: python scripts/ext12_sink_flags.py [--splits discovery,validation,rep_discovery,rep_validation] [--out wino_mixtral_nobos_str]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext7_pairs as P
from moetrace.models import MODELS, RESULTS

ROOT = "/home/ubuntu/MOE"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="discovery,validation,rep_discovery,rep_validation")
    ap.add_argument("--case-sets", default="data/wino_str/case_sets.json")
    ap.add_argument("--out", default="wino_mixtral_nobos_str")
    ap.add_argument("--sink-layer", type=int, default=5)
    args = ap.parse_args()
    cs = json.load(open(os.path.join(ROOT, args.case_sets)))
    fam = P.resolve_sets(cs, args.splits.split(","))
    ids = P.all_ids(fam)
    pidx = sorted({c // 2 for c in ids})
    pairs = {proto: P.load_pairs(os.path.join(ROOT, f"data/wino_str/pairs_train_xl_mixtral_{proto}.parquet"), cs) for proto in ("nobos", "bos")}
    pre_meta = []
    from moetrace.engine import DiagSpec, Engine, PrefillSpec
    pre = []
    for proto in ("nobos", "bos"):
        p = pairs[proto]
        for pi in pidx:
            r = p.loc[pi]
            for which, ids_ in (("a", r.ids_a_l), ("b", r.ids_b_l)):
                pre.append(PrefillSpec(list(ids_), int(r.trig_a), int(r.trig_b)))
                pre_meta.append((pi, which, proto, len(ids_), int(ids_[-1])))
    log(f"{len(pre)} prefill rows ({len(pidx)} pairs x 2 prompts x 2 protocols)")
    od = P.run_dir(args.out)
    t0 = time.time()
    eng = Engine(MODELS["mixtral"]["repo"])
    L = eng.spec.n_layers
    res = eng.run(pre, [], record_routing=True, diag=DiagSpec(resid_norms=True, attn_final=True), log=log)
    dg = res.extra["diag"]
    rn, af = dg["resid_norms"], dg["attn_final"]  # [L, B, T] fp32, [L, B, nH, T] fp16
    lens = np.asarray(res.lens)
    ri = res.route_idx  # [L, B, k]
    sl = args.sink_layer
    half = L // 2
    rows = []
    for j, (pi, which, proto, T, last) in enumerate(pre_meta):
        T = int(lens[j])
        nr = rn[:, j, :T].astype(np.float64)
        am = nr.argmax(1)  # [L]
        fin = T - 1
        others = np.delete(nr[sl], fin)
        att1 = af[1, j, :, :T].astype(np.float32).mean(0)
        row = dict(pair_idx=int(pi), prompt=which, proto=proto, T=T, final_tok=last, argmax_pos_L5=int(am[sl]),
                   final_is_max_L5=bool(am[sl] == fin), pos0_is_max_L5=bool(am[sl] == 0),
                   final_is_max_any=bool((am[1:half] == fin).any()), norm_ratio_L5=float(nr[sl, fin] / max(np.median(others), 1e-6)),
                   final_norm_L5=float(nr[sl, fin]), self_mass_L1=float(att1[fin]), pos0_mass_L1=float(att1[0]),
                   delta=float(res.delta[j]))
        for l in range(L):
            row[f"route_L{l}"] = json.dumps(sorted(int(e) for e in ri[l, j]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(od, "sink_flags.parquet"), index=False)
    from moetrace.ext2_zoo import sink_summary
    summ = {}
    for proto in ("nobos", "bos"):
        m = (df.proto == proto).to_numpy()
        jj = np.nonzero(m)[0]
        Tm = int(lens[jj].max())
        s = sink_summary(rn[:, jj, :Tm], af[:, jj, :, :Tm], lens[jj])
        s["frac_final_is_max_L5"] = float(df[m].final_is_max_L5.mean())
        s["n_final_is_max_L5"] = int(df[m].final_is_max_L5.sum())
        summ[proto] = s
    json.dump(summ, open(os.path.join(od, "sink_diag.json"), "w"), indent=1, default=float)
    P.write_meta(args.out, "sink_flags", {"model": "mixtral", "splits": args.splits, "n_prompts": len(df), "sink_layer": sl,
                                         "seconds": time.time() - t0, "pass_s": res.extra.get("total_s"), "agent": "ext12-complete",
                                         "command": "python " + " ".join(sys.argv), "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    for proto in ("nobos", "bos"):
        x = df[df.proto == proto]
        log(f"{proto}: final position = max norm at L{sl} in {int(x.final_is_max_L5.sum())}/{len(x)} prompts "
            f"(any early layer {int(x.final_is_max_any.sum())}); pos0 max {int(x.pos0_is_max_L5.sum())}; pos0 mass L1 {x.pos0_mass_L1.mean():.2f}")


if __name__ == "__main__":
    main()
