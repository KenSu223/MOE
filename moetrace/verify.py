"""Verification of the streaming engine against transformers (reference) plus internal invariances.

HF side (small model fully on GPU): clean/noised forwards via inputs_embeds, MoE-block output patching via a forward
hook on the MoE block, and expert-contribution patching implemented literally (ablation difference under the original
routing by zeroing one expert's routing weight at the final position; delta_e added to the noised block output).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from .arch import snapshot_dir
from .data import Case, load_records, prepare_case, shuffled_order
from .engine import Engine, PrefillSpec, SpawnSpec
from .noise import noise_draw


@dataclass
class HFRef:
    case_ids: list[int]
    logits_clean: np.ndarray  # [n, V] fp32
    logits_noised: np.ndarray
    delta_clean: np.ndarray
    delta_noised: np.ndarray
    layer_patch: dict  # (case_idx, layer) -> delta_patched
    expert_patch: dict  # (case_idx, layer, expert) -> delta_patched
    routing: dict  # (case_idx, layer) -> (clean top-k list, noised top-k list)
    embed_std: float


def _hf_load(repo: str, attn: str = "eager"):
    from transformers import AutoModelForCausalLM
    snap = snapshot_dir(repo)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn)
    model.eval()
    return model


def _moe_module(model, l: int):
    layer = model.model.layers[l]
    return getattr(layer, "mlp", None) if hasattr(layer, "mlp") else layer.block_sparse_moe


@torch.no_grad()
def hf_reference(repo: str, cases: list[Case], layers_for_patch: list[int], n_patch_cases: int = 10,
                 n_expert_cases: int = 5, sigma_mult: float = 3.0, log=print) -> HFRef:
    model = _hf_load(repo)
    dev = "cuda"
    emb_w = model.get_input_embeddings().weight
    embed_std = float(emb_w.float().std().item())
    sigma = sigma_mult * embed_std
    L = model.config.num_hidden_layers
    ids_l = [torch.tensor([c.ids], device=dev) for c in cases]

    def embeds(i, noised: bool):
        e = model.get_input_embeddings()(ids_l[i])  # [1, T, H] bf16
        if noised:
            c = cases[i]
            eps = noise_draw(c.case_id, len(c.subject_pos), e.shape[-1], sigma).to(dev)
            e = e.clone()
            pos = torch.tensor(c.subject_pos, device=dev)
            e[0, pos] = (e[0, pos].float() + eps).to(e.dtype)
        return e

    def fwd(e):
        return model(inputs_embeds=e).logits[0, -1].float()

    lc, ln = [], []
    for i in range(len(cases)):
        lc.append(fwd(embeds(i, False)).cpu().numpy())
        ln.append(fwd(embeds(i, True)).cpu().numpy())
    lc, ln = np.stack(lc), np.stack(ln)
    ti = np.array([c.true_id for c in cases])
    fi = np.array([c.foil_id for c in cases])
    ar = np.arange(len(cases))
    d_clean = lc[ar, ti] - lc[ar, fi]
    d_noised = ln[ar, ti] - ln[ar, fi]
    log(f"HF forwards done for {len(cases)} cases")

    # --- MoE-block output patching via hooks
    store = {}
    layer_patch = {}
    routing = {}

    def make_record_hook(l):
        def hook(mod, args, out):
            store[("out", l)] = (out[0] if isinstance(out, tuple) else out)[0, -1].detach().clone()
        return hook

    def make_replace_hook(l):
        def hook(mod, args, out):
            o = out[0] if isinstance(out, tuple) else out
            o = o.clone()
            o[0, -1] = store[("clean_out", l)]
            return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        return hook

    def make_route_hook(l, key):
        def hook(mod, args, out):
            # out: (router_logits, router_scores, router_indices)
            routing.setdefault(key, {})[l] = out[2][-1].tolist()
        return hook

    for i in range(min(n_patch_cases, len(cases))):
        handles = [_moe_module(model, l).register_forward_hook(make_record_hook(l)) for l in layers_for_patch]
        fwd(embeds(i, False))
        for h in handles:
            h.remove()
        for l in layers_for_patch:
            store[("clean_out", l)] = store[("out", l)]
        for l in layers_for_patch:
            h = _moe_module(model, l).register_forward_hook(make_replace_hook(l))
            lg = fwd(embeds(i, True))
            h.remove()
            layer_patch[(i, l)] = float(lg[ti[i]] - lg[fi[i]])
    log(f"HF layer patches done: {len(layer_patch)}")

    # --- expert-contribution patching, literal ablation difference at the final position
    expert_patch = {}
    for i in range(min(n_expert_cases, len(cases))):
        for l in layers_for_patch:
            moe = _moe_module(model, l)
            gate = moe.gate
            exps = moe.experts
            # routing at the final position for the clean and noised runs
            rh = [gate.register_forward_hook(make_route_hook(l, "c"))]
            fwd(embeds(i, False))
            rh[0].remove()
            rh = [gate.register_forward_hook(make_route_hook(l, "n"))]
            fwd(embeds(i, True))
            rh[0].remove()
            clean_set = routing["c"][l]
            noised_set = routing["n"][l]
            # helper: MoE-block final-position output with expert e suppressed (or none)
            def block_out(noised: bool, suppress: int):
                def pre(mod, args):
                    hs, idx, wts = args
                    wts = wts.clone()
                    m = idx[-1] == suppress
                    wts[-1][m] = 0
                    return (hs, idx, wts)
                hp = exps.register_forward_pre_hook(pre) if suppress >= 0 else None
                hr = moe.register_forward_hook(make_record_hook(l))
                fwd(embeds(i, noised))
                hr.remove()
                if hp is not None:
                    hp.remove()
                return store[("out", l)].float().clone()
            out_c = block_out(False, -1)
            out_n = block_out(True, -1)
            for e in sorted(set(clean_set) | set(noised_set)):
                c_e_clean = out_c - block_out(False, e)  # zero if e not clean-routed (suppression is a no-op)
                c_e_noised = out_n - block_out(True, e)
                delta_e = c_e_clean - c_e_noised
                def add_hook(mod, args, out):
                    o = out[0] if isinstance(out, tuple) else out
                    o = o.clone()
                    o[0, -1] = (o[0, -1].float() + delta_e).to(o.dtype)
                    return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
                h = moe.register_forward_hook(add_hook)
                lg = fwd(embeds(i, True))
                h.remove()
                expert_patch[(i, l, int(e))] = float(lg[ti[i]] - lg[fi[i]])
            routing[(i, l)] = (clean_set, noised_set)
    routing = {k: v for k, v in routing.items() if isinstance(k, tuple)}
    log(f"HF expert patches done: {len(expert_patch)}")
    del model
    torch.cuda.empty_cache()
    return HFRef([c.case_id for c in cases], lc, ln, d_clean, d_noised, layer_patch, expert_patch, routing, embed_std)


def pick_cases(repo: str, n: int, seed: int = 0) -> list[Case]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_dir(repo))
    recs = load_records()
    out = []
    for i in shuffled_order(len(recs), seed):
        c, why = prepare_case(recs[i], tok)
        if c is not None:
            out.append(c)
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def engine_check(repo: str, cases: list[Case], ref: HFRef, layers_for_patch: list[int], log=print) -> dict:
    eng = Engine(repo)
    L = eng.spec.n_layers
    Hd = eng.hidden
    sigma = 3.0 * eng.embed_std
    report = {"embed_std_engine": eng.embed_std, "embed_std_hf": ref.embed_std}
    n = len(cases)
    prefill = []
    for c in cases:
        prefill.append(PrefillSpec(c.ids, c.true_id, c.foil_id))
    for c in cases:
        eps = noise_draw(c.case_id, len(c.subject_pos), Hd, sigma)
        prefill.append(PrefillSpec(c.ids, c.true_id, c.foil_id, list(c.subject_pos), eps))
    spawns = []
    tags = []
    # (a) layer patches at every layer for every case
    for i in range(n):
        for l in range(L):
            spawns.append(SpawnSpec(l, n + i, i, "layer"))
            tags.append(("layer", i, l, -1))
    # (b) zero interventions on the noised parent (must reproduce the noised prefill logits)
    for i in range(n):
        for l in [0, L // 2, L - 1]:
            spawns.append(SpawnSpec(l, n + i, i, "zero"))
            tags.append(("zero_noised", i, l, -1))
    # (c) zero interventions on the clean run as parent (identity)
    for i in range(n):
        for l in [0, L // 2, L - 1]:
            spawns.append(SpawnSpec(l, i, i, "zero"))
            tags.append(("zero_clean", i, l, -1))
    # (d) expert patches matching the HF expert patches
    for (i, l, e) in ref.expert_patch:
        spawns.append(SpawnSpec(l, n + i, i, "expert", expert=e))
        tags.append(("expert", i, l, e))
    # (e) union coalition vs layer patch (should agree to bf16 rounding)
    for i in range(min(n, 10)):
        for l in layers_for_patch:
            spawns.append(SpawnSpec(l, n + i, i, "coalition_union"))
            tags.append(("union", i, l, -1))
            spawns.append(SpawnSpec(l, n + i, i, "coalition_clean"))
            tags.append(("coal_clean", i, l, -1))
    t0 = time.time()
    res = eng.run(prefill, spawns, record_routing=True, log=log)
    report["engine_pass_s"] = time.time() - t0
    d = res.delta
    d_full = res.logit_true_full - res.logit_foil_full
    dc, dn = d[:n], d[n:]
    report["delta_clean_maxdiff_vs_hf"] = float(np.abs(dc - ref.delta_clean).max())
    report["delta_noised_maxdiff_vs_hf"] = float(np.abs(dn - ref.delta_noised).max())
    report["delta_clean_frac_within_0.1"] = float((np.abs(dc - ref.delta_clean) < 0.1).mean())
    report["delta_noised_frac_within_0.1"] = float((np.abs(dn - ref.delta_noised) < 0.1).mean())
    report["delta_bmm_vs_fullmatmul_maxdiff"] = float(np.abs(d - d_full).max())
    report["delta_bmm_vs_fullmatmul_frac_equal"] = float((d == d_full).mean())
    # top-1 agreement and logit agreement at true/foil
    top1_hf = ref.logits_clean.argmax(-1)
    report["top1_agreement_clean"] = float((res.top1[:n] == top1_hf).mean())
    ti = np.array([c.true_id for c in cases])
    report["logit_true_clean_maxdiff_vs_hf"] = float(np.abs(res.logit_true_full[:n] - ref.logits_clean[np.arange(n), ti]).max())
    sd = res.sp_delta
    tags_arr = np.array([t[0] for t in tags])
    # zero on noised parent == noised prefill
    z = np.array([abs(sd[j] - dn[tags[j][1]]) for j in np.nonzero(tags_arr == "zero_noised")[0]])
    report["zero_on_noised_maxdiff"] = float(z.max())
    z = np.array([abs(sd[j] - dc[tags[j][1]]) for j in np.nonzero(tags_arr == "zero_clean")[0]])
    report["zero_on_clean_maxdiff"] = float(z.max())
    # layer patches vs HF hooks
    diffs = []
    for j in np.nonzero(tags_arr == "layer")[0]:
        _, i, l, _ = tags[j]
        if (i, l) in ref.layer_patch:
            diffs.append(sd[j] - ref.layer_patch[(i, l)])
    diffs = np.array(diffs)
    report["layer_patch_vs_hf_n"] = int(len(diffs))
    report["layer_patch_vs_hf_maxdiff"] = float(np.abs(diffs).max())
    report["layer_patch_vs_hf_frac_within_0.1"] = float((np.abs(diffs) < 0.1).mean())
    # rescue agreement (Delta_patched - Delta_noised) vs HF rescue
    resc = []
    for j in np.nonzero(tags_arr == "layer")[0]:
        _, i, l, _ = tags[j]
        if (i, l) in ref.layer_patch:
            resc.append((sd[j] - dn[i]) - (ref.layer_patch[(i, l)] - ref.delta_noised[i]))
    report["layer_rescue_vs_hf_maxdiff"] = float(np.abs(np.array(resc)).max())
    # expert patches vs HF
    diffs = []
    for j in np.nonzero(tags_arr == "expert")[0]:
        _, i, l, e = tags[j]
        diffs.append(sd[j] - ref.expert_patch[(i, l, e)])
    diffs = np.array(diffs)
    report["expert_patch_vs_hf_n"] = int(len(diffs))
    report["expert_patch_vs_hf_maxdiff"] = float(np.abs(diffs).max()) if len(diffs) else None
    report["expert_patch_vs_hf_frac_within_0.1"] = float((np.abs(diffs) < 0.1).mean()) if len(diffs) else None
    # routing agreement
    agree, tot = 0, 0
    for (i, l), (cs, ns) in ref.routing.items():
        tot += 2
        agree += int(sorted(res.route_idx[l, i].tolist()) == sorted(cs))
        agree += int(sorted(res.route_idx[l, n + i].tolist()) == sorted(ns))
    report["routing_set_agreement"] = f"{agree}/{tot}"
    # union coalition vs layer patch
    du, dl = [], []
    for j in np.nonzero(tags_arr == "union")[0]:
        _, i, l, _ = tags[j]
        jl = [jj for jj in np.nonzero(tags_arr == "layer")[0] if tags[jj][1] == i and tags[jj][2] == l][0]
        du.append(sd[j]); dl.append(sd[jl])
    du, dl = np.array(du), np.array(dl)
    report["union_vs_layer_maxdiff"] = float(np.abs(du - dl).max())
    report["union_vs_layer_frac_equal"] = float((du == dl).mean())
    report["layer_times_first_pass"] = [(int(l), round(a, 3), round(b, 3)) for l, a, b in res.layer_times]
    report["pass_total_s"] = res.extra["total_s"]
    # mean layer rescue curve on these cases (sanity: should be positive somewhere)
    curve = np.zeros(L)
    for j in np.nonzero(tags_arr == "layer")[0]:
        _, i, l, _ = tags[j]
        curve[l] += (sd[j] - dn[i]) / n
    report["mean_layer_rescue_curve"] = [round(float(x), 3) for x in curve]
    return report
