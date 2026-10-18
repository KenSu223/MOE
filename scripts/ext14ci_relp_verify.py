"""ext14 C2 (p6-circuit): checks of the functional forward (moetrace/ext14_circuit_relp.FModel) against transformers.

For a few STR prompts of the pilot tasks: Delta from the HF model (bf16 logits, eager attention) vs FModel with relp off
and on (the RelP rules change only the backward, so the forward must agree); the final-position MLP / MoE output of every
layer vs HF forward hooks; the neuron activations of a dense MLP (input of down_proj) vs a hook; an exact patch identity
(all neurons of a layer at a position set to the clean values == MLP-output patch). Writes
results/p6ci_relp_verify_<model>.json.

Usage: python scripts/ext14ci_relp_verify.py <olmoe|mistral7b> <task>
"""
import json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, torch
from moetrace.models import RESULTS
from moetrace.ext14_circuit_relp import FModel, log
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
from ext14ci_relp_run import load_cases


def main():
    model, task = sys.argv[1], sys.argv[2]
    cases = load_cases(model, task, 6)
    fm = FModel(model)
    dev = fm.dev
    hooks, store = [], {}
    for l, lay in enumerate(fm.model.model.layers):
        hooks.append(lay.mlp.register_forward_hook(lambda m, a, o, l=l: store.__setitem__(("mlp", l), o.detach())))
        if not fm.moe:
            hooks.append(lay.mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: store.__setitem__(("act", l), a[0].detach())))
    res = {"model": model, "task": task, "cases": []}
    for x in cases:
        for side in ("clean", "corrupt"):
            ids = torch.tensor(x[side], device=dev)[None]
            with torch.no_grad():
                lg = fm.model(ids).logits[0, -1]
                d_hf = float(lg[x["true"]].float() - lg[x["foil"]].float())
                r0 = fm.forward(ids, [x["true"]], [x["foil"]], record=True)
                r1 = fm.forward(ids, [x["true"]], [x["foil"]], record=True, relp=True)
            mlp_diff = max(float((r0["out"][l][0, -1].float() - store[("mlp", l)][0, -1].float()).abs().max()) for l in range(fm.L))
            mlp_rel = max(float((r0["out"][l][0, -1].float() - store[("mlp", l)][0, -1].float()).norm()
                                / store[("mlp", l)][0, -1].float().norm()) for l in range(fm.L))
            per_layer = [float((r0["out"][l][0, -1].float() - store[("mlp", l)][0, -1].float()).norm()
                               / store[("mlp", l)][0, -1].float().norm()) for l in range(fm.L)]
            rec = dict(case_id=x["case_id"], side=side, delta_hf_bf16=d_hf, delta_f=float(r0["delta"][0]),
                       mlp_rel_by_layer=[round(v, 4) for v in per_layer],
                       delta_f_relp=float(r1["delta"][0]), mlp_out_maxabs=mlp_diff, mlp_out_maxrel=mlp_rel)
            if not fm.moe:
                rec["act_maxrel"] = max(float((r0["q"][l][0].float() - store[("act", l)][0].float()).norm()
                                              / store[("act", l)][0].float().norm()) for l in range(fm.L))
            res["cases"].append(rec)
        # identity: all neurons of layer l at the final position := clean  ==  MLP-output patch at (l, T-1)
        T = len(x["clean"])
        tc = torch.tensor(x["clean"], device=dev)[None]
        to = torch.tensor(x["corrupt"], device=dev)[None]
        with torch.no_grad():
            rc = fm.forward(tc, [x["true"]], [x["foil"]], record=True)
            ids_l = []
            for l in (fm.L // 2, fm.L - 2):
                mk = torch.zeros_like(rc["q"][l], dtype=torch.bool)
                mk[0, T - 1] = True
                a = fm.forward(to, [x["true"]], [x["foil"]], interv={l: (mk, rc["q"][l])})
                bm = torch.zeros(1, T, dtype=torch.bool, device=dev)
                bm[0, T - 1] = True
                b = fm.forward(to, [x["true"]], [x["foil"]], out_patch={l: (bm, rc["out"][l])})
                ids_l.append(abs(float(a["delta"][0] - b["delta"][0])))
        res["cases"][-1]["neurons_all_vs_outpatch_maxabs"] = max(ids_l)
        # RelP backward runs and is finite
        _, g = fm.node_grads(to, [x["true"]], [x["foil"]], relp=True,
                             route=[(ti, tw) for ti, tw in rc["route"]] if fm.moe else None)
        res["cases"][-1]["relp_grad_finite"] = bool(all(torch.isfinite(t).all() for t in g))
    for h in hooks:
        h.remove()
    cs = res["cases"]
    res["summary"] = {
        "max_abs_delta_f_vs_hf": max(abs(c["delta_f"] - c["delta_hf_bf16"]) for c in cs),
        "max_abs_delta_relp_vs_plain": max(abs(c["delta_f_relp"] - c["delta_f"]) for c in cs),
        "max_mlp_out_rel": max(c["mlp_out_maxrel"] for c in cs),
        "max_act_rel": max(c.get("act_maxrel", 0.0) for c in cs),
        "max_neurons_all_vs_outpatch": max(c.get("neurons_all_vs_outpatch_maxabs", 0.0) for c in cs),
        "grads_finite": all(c.get("relp_grad_finite", True) for c in cs)}
    log(json.dumps(res["summary"]))
    json.dump(res, open(os.path.join(RESULTS, f"p6ci_relp_verify_{model}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
