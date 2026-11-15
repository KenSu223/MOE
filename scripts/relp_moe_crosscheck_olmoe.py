#!/usr/bin/env python3
"""Node-by-node cross-check of the streamed RelP engine against the guide's reference implementation on the real
OLMoE-1B-7B-0125 checkpoint (guide section 9.2: a sparse implementation must equal the explicit all-expert FModel).

Reference: moetrace.relp_sva.score_pair / evaluate_case with ext14_circuit_relp.FModel (all experts, one global autograd
graph). Candidate: moetrace.relp_moe_stream (layer streaming, reverse-streamed backward, sparse experts), driven by the
same functions as scripts/relp_sva_moe_run.py. Both in the same dtype; fp32 = logic check (FModel on the CPU because
fp32 OLMoE does not fit the GPU), bf16 = numerics report (FModel on the GPU).

Usage: python scripts/relp_moe_crosscheck_olmoe.py --dtype float32 --out results/relp_moe_dev/crosscheck_olmoe_fp32.json
"""
import argparse
import json
import sys
import time
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/opt/dlami/nvme/hf/hub/models--allenai--OLMoE-1B-7B-0125/snapshots/"
                                          "9b0c1aa87e34a20052389dce1f0cf01da783f654")
    p.add_argument("--manifest", default=str(ROOT / "data/relp_sva/olmoe_smoke_v1"))
    p.add_argument("--task", default="nounpp")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--n-score", type=int, default=6)
    p.add_argument("--n-eval", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import torch
    from moetrace import relp_sva as S
    from moetrace.ext14_circuit_relp import FModel
    from moetrace.relp_moe_stream import StreamRelP
    import relp_sva_moe_run as R
    dt = getattr(torch, a.dtype)
    torch.set_num_threads(16)
    _, disc = S.read_split(a.manifest, a.task, "discovery")
    _, val = S.read_split(a.manifest, a.task, "validation")
    _, cal = S.read_split(a.manifest, a.task, "calibration")
    disc, val = disc[:a.n_score], val[:a.n_eval]
    out = {"dtype": a.dtype, "task": a.task, "manifest": a.manifest, "n_score": len(disc), "n_eval": len(val)}
    t0 = time.time()
    m = StreamRelP(a.model_dir, "cuda", dt)
    tot_s, sup_s, cases_s = R.score(m, disc, "paired", "non_special", 300)
    out["stream_score_seconds"] = time.time() - t0
    qmean_s, counts_s = R.calibrate_q(m, cal)
    # reference
    fm_dev = "cpu" if a.dtype == "float32" else "cuda"
    t0 = time.time()
    fm = FModel(a.model_dir, device=fm_dev, dtype=dt)
    total = support = None
    margins = []
    for row in disc:
        sc, sp, rec = S.score_pair(fm, row, "paired", "non_special")
        total = sc.double() if total is None else total + sc.double()
        support = sp if support is None else support | sp
        margins.append((rec["margin_A"], rec["margin_B"]))
    out["fmodel_score_seconds"] = time.time() - t0
    qmean_r, counts_r = S.calibrate_q(fm, cal)
    tot_s = tot_s.double()
    d = (tot_s - total)
    out["score"] = {
        "support_equal": bool(torch.equal(sup_s, support)),
        "support_nodes": int(support.sum()),
        "max_abs_diff": float(d.abs().max()), "max_abs_ref": float(total.abs().max()),
        "rel_l2": float(d.norm() / total.norm()),
        "expert_position_rel_l2": float((tot_s.sum(-1) - total.sum(-1)).norm() / total.sum(-1).norm()),
        "corr": float(torch.corrcoef(torch.stack([tot_s.flatten(), total.flatten()]))[0, 1]),
        "margins_max_abs_diff": max(max(abs(c["margin_A"] - mr[0]), abs(c["margin_B"] - mr[1]))
                                    for c, mr in zip(cases_s, margins)),
        "calibration_counts_equal": bool(torch.equal(counts_s.double(), counts_r.double())),
        "calibration_qmean_max_abs_diff": float((qmean_s - qmean_r).abs().max()),
    }
    for unit_name in ("neuron", "expert_position"):
        u_s = tot_s if unit_name == "neuron" else tot_s.sum(-1)
        u_r = total if unit_name == "neuron" else total.sum(-1)
        e_r = support if unit_name == "neuron" else support.any(-1)
        k = 1000 if unit_name == "neuron" else 50
        i_s = S.rank_indices(u_s, e_r, k)
        i_r = S.rank_indices(u_r, e_r, k)
        out["score"][f"{unit_name}_top{k}_overlap"] = len(set(i_s.tolist()) & set(i_r.tolist())) / k
        out["score"][f"{unit_name}_top10_identical_order"] = i_s[:10].tolist() == i_r[:10].tolist()
    # interventions: the same fixed lists (from the reference ranking) on held-out rows, all four kinds + empty
    mean = total / len(disc)
    Label = namedtuple("Label", "ranking seed budget")
    res_eval = {}
    for unit, budgets in (("neuron", (1, 5, 50)), ("expert_position", (1, 2, 5))):
        values = mean if unit == "neuron" else mean.sum(-1)
        eligible = support if unit == "neuron" else support.any(-1)
        order = S.rank_indices(values, eligible, max(budgets))
        for baseline, qm in (("zero", None), ("mean_q_routed", qmean_r)):
            masks_unit, masks_full = {}, {}
            for n in budgets:
                chosen = torch.zeros_like(eligible)
                chosen.reshape(-1)[order[:n]] = True
                rnd = S.random_matched(chosen, eligible, torch.Generator().manual_seed(101 + n))
                for lab, x in ((Label("relp", 0, n), chosen), (Label("random", 101, n), rnd)):
                    masks_unit[lab] = x
                    masks_full[lab] = x if unit == "neuron" else x[..., None].expand_as(mean)
            rec_s = R.evaluate(m, val, [{"masks": masks_unit, "unit": unit, "baseline": baseline, "qmean": qm}],
                               "non_special", 1040)[0]
            rec_r = []
            for row in val:
                rec_r += S.evaluate_case(fm, row, masks_full, baseline, "non_special", qm)
            diffs = [abs(x["margin"] - y["margin"]) for x, y in zip(rec_s, rec_r)]
            keys_ok = all((x["row_id"], x["ranking"], x["budget"], x["intervention"]) ==
                          (y["row_id"], y["ranking"], y["budget"], y["intervention"]) for x, y in zip(rec_s, rec_r))
            route_ok = sum(x["topk_changed_layer_positions"] == y["topk_changed_layer_positions"] for x, y in zip(rec_s, rec_r))
            res_eval[f"{unit}/{baseline}"] = {
                "records": len(rec_s), "keys_identical": keys_ok and len(rec_s) == len(rec_r),
                "margin_max_abs_diff": max(diffs), "margin_abs_mean": sum(abs(y["margin"]) for y in rec_r) / len(rec_r),
                "empty_max_abs_diff": max(abs(x["margin_empty"] - y["margin_empty"]) for x, y in zip(rec_s, rec_r)),
                "route_change_counts_equal": f"{route_ok}/{len(rec_s)}"}
    out["evaluate"] = res_eval
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
