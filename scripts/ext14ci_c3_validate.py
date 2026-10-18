"""ext14 C3 validation runs (GPU; p6-circuit) for one proto, from results/p6ci_c3_data/circuits_<proto>.json:

  base     clean + corrupted rows of every task's validation directed cases (unmasked)
  ko       knockout of a concept circuit's part (all | heads | experts; draw 0 = the circuit, draws 1..3 = matched random sets)
           in the CLEAN row: heads PrefillSpec.head_mask (zero), experts route_mask (reroute, all positions)
  addback  the same sets patched to their clean values inside the corrupted row: one multi spawn with a heads_experts step
           per layer (denoising, final position)
  edge_*   for every pair A, B of a circuit with layer(A) < layer(B), on the circuit's own tasks: single patches of A and B
           and the joint patch (multi: A's step, then B's step) -> edge A -> B = rescue(A) + rescue(B) - rescue(A u B)
Rows are packed into passes of <= --max-rows prefill rows / --max-spawn spawns; one engine pass per pack.
Writes results/p6ci_c3_data/validate_<proto>.parquet.

Usage: python scripts/ext14ci_c3_validate.py --proto qwen3|mixtral_bos [--tasks a,b] [--max-rows 2048] [--max-spawn 20000]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext14_circuit_c3 as C3
from moetrace import ext14_circuit as C
from moetrace import ext7_pairs as P
from moetrace.models import MODELS, RESULTS

log = C.log


def parse(c):
    if "H" in c:
        l, h = c[1:].split("H")
        return "head", int(l), int(h)
    l, e = C.parse_ename(c)
    return "expert", l, e


def steps_of(comps):
    by = {}
    for c in comps:
        k, l, i = parse(c)
        by.setdefault(l, ([], []))[0 if k == "head" else 1].append(i)
    return tuple((l, "heads_experts", (tuple(sorted(h)), tuple(sorted(e)))) for l, (h, e) in sorted(by.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--max-rows", type=int, default=2048)
    ap.add_argument("--max-spawn", type=int, default=20000)
    a = ap.parse_args()
    from moetrace.engine import Engine, PrefillSpec, SpawnSpec
    js = json.load(open(os.path.join(C3.DATA, f"circuits_{a.proto}.json")))
    model = C3.PROTO_MODEL[a.proto]
    circuits = {k: v for k, v in js["circuits"].items() if not v.get("insufficient") and (v["heads"] or v["experts"])}
    tasks = a.tasks.split(",") if a.tasks else js["tasks"]
    concept_of = js["concept_of"]
    sets = []  # (concept, part, draw, comps)
    for conc, cc in circuits.items():
        draws = [{"heads": cc["heads"], "experts": cc["experts"]}] + cc["random"]
        for d, s in enumerate(draws):
            for part in ("all", "heads", "experts"):
                comps = (s["heads"] if part != "experts" else []) + (s["experts"] if part != "heads" else [])
                if comps:
                    sets.append((conc, part, d, comps))
    log(f"{a.proto}: {len(circuits)} circuits, {len(sets)} sets, tasks {tasks}")
    eng = Engine(MODELS[model]["repo"])
    out_rows = []
    outp = os.path.join(C3.DATA, f"validate_{a.proto}" + (f"__{a.tasks.replace(',', '+')}" if a.tasks else "") + ".parquet")
    for t in tasks:
        pp, cp = C3.pair_paths(t, a.proto)
        cs = json.load(open(cp))
        pairs = P.load_pairs(pp, cs)
        val = [int(c) for c in cs["directed"]["validation"]]
        dc = P.directed_cases(pairs, val)
        # ---- knockout (clean rows only) + base rows ----
        pre, tags = [], []
        for c in val:
            x = dc[c]
            pre.append(PrefillSpec(list(x.clean_ids), int(x.true_id), int(x.foil_id)))
            tags.append(("base", c, None, None, None))
            for conc, part, d, comps in sets:
                hm = tuple((l, i) for k, l, i in map(parse, comps) if k == "head")
                rm = tuple((l, i) for k, l, i in map(parse, comps) if k == "expert")
                pre.append(PrefillSpec(list(x.clean_ids), int(x.true_id), int(x.foil_id), head_mask=hm, route_mask=rm))
                tags.append(("ko", c, conc, part, d))
        t0 = time.time()
        for s0 in range(0, len(pre), a.max_rows):
            res = eng.run(pre[s0:s0 + a.max_rows], [], record_routing=False)
            for j, dl in enumerate(res.delta):
                kind, c, conc, part, d = tags[s0 + j]
                out_rows.append(dict(task=t, kind=kind, case_id=c, concept=conc, part=part, draw=d, delta=float(dl)))
        log(f"{t}: knockout {len(pre)} rows ({time.time() - t0:.0f}s)")
        # ---- add-back + edges (clean + corrupted rows, spawns) ----
        spawn_plan = []  # (case, kind, concept, part, draw, a, b, steps)
        for c in val:
            for conc, part, d, comps in sets:
                spawn_plan.append((c, "addback", conc, part, d, None, None, steps_of(comps)))
            for conc, cc in circuits.items():
                if concept_of.get(t) != conc:
                    continue
                comps = cc["heads"] + cc["experts"]
                for A in comps:
                    spawn_plan.append((c, "edge_single", conc, None, None, A, None, steps_of([A])))
                for A in comps:
                    for B in comps:
                        if parse(A)[1] < parse(B)[1]:
                            spawn_plan.append((c, "edge_pair", conc, None, None, A, B, steps_of([A, B])))
        per_case = {}
        for i, sp in enumerate(spawn_plan):
            per_case.setdefault(sp[0], []).append(i)
        packs, cur, n = [], [], 0
        for c in val:
            k = len(per_case.get(c, []))
            if cur and n + k > a.max_spawn:
                packs.append(cur)
                cur, n = [], 0
            cur.append(c)
            n += k
        if cur:
            packs.append(cur)
        t0 = time.time()
        for pk in packs:
            pre, rowof = [], {}
            for c in pk:
                x = dc[c]
                rowof[c] = (len(pre), len(pre) + 1)
                pre.append(PrefillSpec(list(x.clean_ids), int(x.true_id), int(x.foil_id)))
                pre.append(PrefillSpec(list(x.corrupt_ids), int(x.true_id), int(x.foil_id)))
            spawns, meta = [], []
            for c in pk:
                ci, oi = rowof[c]
                for i in per_case.get(c, []):
                    _, kind, conc, part, d, A, B, st = spawn_plan[i]
                    spawns.append(SpawnSpec(st[0][0], oi, ci, "multi", steps=st))
                    meta.append((c, kind, conc, part, d, A, B))
            res = eng.run(pre, spawns, record_routing=False)
            par = np.array([s.parent for s in spawns])
            for j, (c, kind, conc, part, d, A, B) in enumerate(meta):
                ci, oi = rowof[c]
                out_rows.append(dict(task=t, kind=kind, case_id=c, concept=conc, part=part, draw=d, a=A, b=B,
                                     delta=float(res.sp_delta[j]), delta_clean=float(res.delta[ci]), delta_corrupt=float(res.delta[oi]),
                                     rescue=float(res.sp_delta[j] - res.delta[oi])))
        log(f"{t}: add-back / edges {len(spawn_plan)} spawns in {len(packs)} passes ({time.time() - t0:.0f}s)")
        pd.DataFrame(out_rows).to_parquet(outp)
    pd.DataFrame(out_rows).to_parquet(outp)
    log("done")


if __name__ == "__main__":
    main()
