"""ext14 C3 (p6-circuit): per-expert DLA (prefill only, no expert singles) on the SELECTION cases of a C3 task whose run has no
expert singles (Mixtral ss_e1 = Phase-6 e1_e2a, tier 2 cancelled). Same quantity as scripts/ext14ci_c3_country.py dla:
(c_e(clean) - c_e(corrupt)) . gamma (W_U[true] - W_U[foil]) / rms(h_final corrupted), per routed expert at every layer.
Writes results/p6ci_c3_dla_<task>_<proto>/expert_dla.parquet (+ .done_dla).
Usage: python scripts/ext14ci_c3_dla.py --proto mixtral_bos --task ss_e1
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import pandas as pd
from moetrace import ext14_circuit_c3 as C3
from moetrace import ext7_pairs as P
from moetrace.models import MODELS, RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", required=True)
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    from ext14ci_c3_existing import spec, item_ok, ab
    from moetrace.engine import Engine, PrefillSpec, DiagSpec
    sp = spec(a.task, a.proto)
    pairs_df = pd.read_parquet(ab(sp["pairs"]))
    cs = json.load(open(ab(sp["cs"])))
    inv = {int(v): k for k, v in cs["pair_idx_of"].items()}
    ok = item_ok(pairs_df, sp["sel"])
    ids = [2 * int(i) + d for i in cs["pairs"]["validation"] if inv.get(int(i)) in ok for d in (0, 1)]
    pairs = P.load_pairs(ab(sp["pairs"]), cs)
    dc = P.directed_cases(pairs, ids)
    eng = Engine(MODELS[C3.PROTO_MODEL[a.proto]]["repo"])
    out = []
    for s0 in range(0, len(ids), 512):
        chunk = ids[s0:s0 + 512]
        pre = []
        for c in chunk:
            x = dc[c]
            pre.append(PrefillSpec(list(x.clean_ids), int(x.true_id), int(x.foil_id)))
            pre.append(PrefillSpec(list(x.corrupt_ids), int(x.true_id), int(x.foil_id)))
        t0 = time.time()
        res = eng.run(pre, [], record_routing=True, diag=DiagSpec(contrib_dla=True))
        dg = res.extra["diag"]
        for j, c in enumerate(chunk):
            ca, co = 2 * j, 2 * j + 1
            rms = float(dg["final_rms"][co])
            for l in range(res.route_idx.shape[0]):
                ec = dict(zip(res.route_idx[l, ca].tolist(), dg["contrib_dla"][l, ca].tolist()))
                eo = dict(zip(res.route_idx[l, co].tolist(), dg["contrib_dla"][l, co].tolist()))
                for e in set(ec) | set(eo):
                    out.append((c, l, int(e), (ec.get(e, 0.0) - eo.get(e, 0.0)) / rms, float(res.delta[ca]), float(res.delta[co])))
        print(time.strftime("%H:%M:%S"), f"pass {len(pre)} rows {time.time() - t0:.0f}s", flush=True)
    od = os.path.join(RESULTS, f"p6ci_c3_dla_{a.task}_{a.proto}")
    os.makedirs(od, exist_ok=True)
    pd.DataFrame(out, columns=["case_id", "layer", "expert", "dla", "delta_clean", "delta_corrupt"]).to_parquet(os.path.join(od, "expert_dla.parquet"))
    open(os.path.join(od, ".done_dla"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")


if __name__ == "__main__":
    main()
