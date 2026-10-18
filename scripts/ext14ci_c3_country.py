"""ext14 C3 COUNTRY tasks (p6-circuit; NEED HEADS cf_country/<proto> posted 2026-10-17): CounterFact STR pair sets of the
country relations, head singles at fixed CounterFact attention layers, per-expert DLA as the expert score.

  merge   (CPU) results/p6ci_c3_data/cf_{P17,P27,P495,P103}_<proto>.parquet (scripts/ext14ci_c3_cf_build.py) ->
          pairs_cfc_<proto>.parquet (column c3_task: co_p17 = P17, co_nat = P27 + P495, ct_lang = P103) and
          case_sets_cfc_<proto>.json: "discovery" = SELECTION directed cases (the relation files' discovery pairs; heads + DLA run
          on these only), "evaluation" = the relation files' validation pairs (not run here; the C3 validation uses them)
  dla     (GPU) one prefill pass over the selection cases (clean + corrupted rows, DiagSpec.contrib_dla): per routed expert
          direct effect (c_e(clean) - c_e(corrupt)) . gamma (W_U[true] - W_U[foil]) / rms(h_final corrupted)
          -> results/p6ci_c3_cfc_<proto>/expert_dla.parquet (case_id, layer, expert, dla, delta_clean, delta_corrupt)
Head singles: scripts/ext7_wino_heads.py --layers HEAD_LAYERS[proto] --families main on the merged files (run by the chain).

Usage: python scripts/ext14ci_c3_country.py merge|dla --proto qwen3|mixtral_bos
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext14_circuit_c3 as C3
from moetrace.models import MODELS, RESULTS

# top CounterFact attention layers (qwen3_str_attnsweep / mixtral_bos_str_attnsweep) + the layers of the other concepts' heads
HEAD_LAYERS = {"qwen3": [28, 38, 39, 40, 42, 43], "mixtral_bos": [13, 15, 16, 18, 19, 24]}
GROUPS = {"co_p17": ("P17",), "co_nat": ("P27", "P495"), "ct_lang": ("P103",)}


def merge(proto):
    frames = []
    for task, rels in GROUPS.items():
        for rel in rels:
            p = os.path.join(C3.DATA, f"cf_{rel}_{proto}.parquet")
            c = os.path.join(C3.DATA, f"case_sets_cf_{rel}_{proto}.json")
            if not os.path.exists(p):
                continue
            df = pd.read_parquet(p)
            cs = json.load(open(c))
            inv = {int(v): k for k, v in cs["pair_idx_of"].items()}
            sp = {pid: s for s in ("discovery", "validation") for pid in (inv[int(i)] for i in cs["pairs"][s])}
            df["c3_task"] = task
            df["c3_split"] = df.pair_id.map(sp)
            frames.append(df)
    P = pd.concat(frames, ignore_index=True)
    P = P.sort_values(["c3_split", "c3_task", "pair_id"]).reset_index(drop=True)
    idx = {p: i for i, p in enumerate(P.pair_id)}
    sel = [idx[p] for p in P.pair_id[P.c3_split == "discovery"]]
    ev = [idx[p] for p in P.pair_id[P.c3_split == "validation"]]
    js = {"note": f"p6-circuit C3 COUNTRY CounterFact pairs for {proto}: 'discovery' = selection cases (head singles, expert DLA), "
                  "'evaluation' = held-out evaluation cases (C3 knockout / add-back / edges)",
          "pair_idx_of": idx, "pairs": {"discovery": sel, "evaluation": ev},
          "directed": {"discovery": [2 * i + d for i in sel for d in (0, 1)], "evaluation": [2 * i + d for i in ev for d in (0, 1)]}}
    P.to_parquet(os.path.join(C3.DATA, f"pairs_cfc_{proto}.parquet"), index=False)
    json.dump(js, open(os.path.join(C3.DATA, f"case_sets_cfc_{proto}.json"), "w"), indent=1)
    print(P.groupby(["c3_task", "c3_split"]).size().to_dict())


def dla(proto):
    from moetrace import ext7_pairs as P
    from moetrace.engine import Engine, PrefillSpec, DiagSpec
    cs = json.load(open(os.path.join(C3.DATA, f"case_sets_cfc_{proto}.json")))
    pairs = P.load_pairs(os.path.join(C3.DATA, f"pairs_cfc_{proto}.parquet"), cs)
    ids = [int(c) for c in cs["directed"]["discovery"]]
    dc = P.directed_cases(pairs, ids)
    eng = Engine(MODELS[C3.PROTO_MODEL[proto]]["repo"])
    out = []
    for s0 in range(0, len(ids), 1024):
        chunk = ids[s0:s0 + 1024]
        pre = []
        for c in chunk:
            x = dc[c]
            pre.append(PrefillSpec(list(x.clean_ids), int(x.true_id), int(x.foil_id)))
            pre.append(PrefillSpec(list(x.corrupt_ids), int(x.true_id), int(x.foil_id)))
        t0 = time.time()
        res = eng.run(pre, [], record_routing=True, diag=DiagSpec(contrib_dla=True))
        dg = res.extra["diag"]
        L = res.route_idx.shape[0]
        for j, c in enumerate(chunk):
            a, b = 2 * j, 2 * j + 1
            rms = float(dg["final_rms"][b])
            for l in range(L):
                ec = dict(zip(res.route_idx[l, a].tolist(), dg["contrib_dla"][l, a].tolist()))
                eo = dict(zip(res.route_idx[l, b].tolist(), dg["contrib_dla"][l, b].tolist()))
                for e in set(ec) | set(eo):
                    out.append((c, l, int(e), (ec.get(e, 0.0) - eo.get(e, 0.0)) / rms, float(res.delta[a]), float(res.delta[b])))
        print(time.strftime("%H:%M:%S"), f"dla pass {len(pre)} rows {time.time() - t0:.0f}s", flush=True)
    od = os.path.join(RESULTS, f"p6ci_c3_cfc_{proto}")
    os.makedirs(od, exist_ok=True)
    pd.DataFrame(out, columns=["case_id", "layer", "expert", "dla", "delta_clean", "delta_corrupt"]).to_parquet(os.path.join(od, "expert_dla.parquet"))
    open(os.path.join(od, ".done_dla"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["merge", "dla"])
    ap.add_argument("--proto", required=True)
    a = ap.parse_args()
    merge(a.proto) if a.stage == "merge" else dla(a.proto)
