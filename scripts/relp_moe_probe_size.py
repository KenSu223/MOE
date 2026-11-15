#!/usr/bin/env python3
"""Phase 7 S4f probe: how large is a unit's write difference relative to the residual stream? A first-order method (RelP)
can only be trusted when the patch moves the residual a small relative distance. For the first discovery rows of a task,
at the subject position and the final position, per layer: ||o_A - o_B|| / ||h_B|| (whole MoE output, h_B = the B run's
residual after the layer) and, for given experts, ||c_e(A) - c_e(B)|| / ||h_B|| (expert contribution; 0 if not routed).
Writes JSON. Development probe (not a protocol step); uses the verified engine, no interventions.
Usage: python scripts/relp_moe_probe_size.py <model_dir> <manifest_dir> <task> <out.json> [--rows 64] [--experts L:E ...]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("manifest_dir")
    p.add_argument("task")
    p.add_argument("out")
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--experts", nargs="*", default=[])
    a = p.parse_args()
    import torch
    from moetrace import relp_sva as S
    from moetrace.relp_moe_stream import Group, StreamRelP, forward_pass
    import relp_sva_moe_run as R
    _, rows = S.read_split(a.manifest_dir, a.task, "discovery")
    rows = rows[:a.rows]
    n = len(rows)
    m = StreamRelP(a.model_dir, "cuda", torch.bfloat16)
    A, Bb = R.ids_tensor(rows), R.ids_tensor(rows, True)
    T = A.shape[1]
    subj = min(min(r["changed_positions"]) for r in rows)
    want = {(int(x.split(":")[0]), int(x.split(":")[1])) for x in a.experts}
    res = {"task": a.task, "rows": n, "subject_pos": subj, "final_pos": T - 1, "layers": []}
    g = Group("g", torch.cat([A, Bb]), [r["target_id"] for r in rows] * 2, [r["foil_id"] for r in rows] * 2, pairs=True,
              keep=set())

    def on_layer(l, w, st):
        rec = st["layer_out"]["g"]
        out = rec.out.float()                                      # [2n, T, H]
        h = st["groups"]["g"].h.float()                            # residual after this layer (all rows)
        row = {"layer": l}
        for name, t in (("subject", subj), ("final", T - 1)):
            dif = (out[:n, t] - out[n:, t]).norm(dim=-1)
            hn = h[n:, t].norm(dim=-1)
            row[f"moe_out_rel_{name}"] = float((dif / hn).mean())
            row[f"h_norm_{name}"] = float(hn.mean())
        topi = rec.topi.view(2 * n, T, -1)
        c = rec.c.view(2 * n, T, topi.shape[-1], -1).float()
        for (L, E) in want:
            if L != l:
                continue
            for name, t in (("subject", subj), ("final", T - 1)):
                def ce(rows_):
                    hit = topi[rows_, t] == E
                    return (c[rows_, t] * hit[..., None]).sum(1)
                d = (ce(slice(0, n)) - ce(slice(n, 2 * n))).norm(dim=-1)
                row[f"expert_{L}_{E}_rel_{name}"] = float((d / h[n:, t].norm(dim=-1)).mean())
                row[f"expert_{L}_{E}_routed_A_{name}"] = float((topi[:n, t] == E).any(-1).float().mean())
                row[f"expert_{L}_{E}_routed_B_{name}"] = float((topi[n:, t] == E).any(-1).float().mean())
        res["layers"].append(row)
    forward_pass(m, [g], on_layer=on_layer)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
