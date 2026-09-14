"""ext3: verify the engine's new diagnostics and position-offset option against transformers on OLMoE-1B-7B-0125.

Checks, on a few CounterFact prompts (bf16, eager attention on both sides):
  attn_final           final-position attention probabilities per head/layer vs output_attentions=True
  resid_norms          per-position residual norms after every layer vs output_hidden_states=True
  router_logits_final  router logits at the final position vs a hook on mlp.gate
  route_all_layers     top-k routing of every token vs the gate hook
  token_logprobs       next-token log-probs vs log_softmax of the HF logits
  pos_offset           engine rows with pos_offset=1 vs HF with position_ids = arange + 1 (final-position logits)
  prefix_ids           prepare_case shifts subject positions and keeps object ids
Writes results/verify_ext3_diag_olmoe.json.  Usage: python scripts/ext3_verify_diag.py [n_cases]
"""
import json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, torch
from moetrace.verify import pick_cases, _hf_load
from moetrace.engine import Engine, PrefillSpec, DiagSpec
from moetrace.data import load_records, prepare_case
from moetrace.arch import snapshot_dir

REPO = "allenai/OLMoE-1B-7B-0125"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    cases = pick_cases(REPO, n)
    log(f"{len(cases)} cases; T = {[len(c.ids) for c in cases]}")
    rep = {"n_cases": n, "prompts": [c.prompt for c in cases]}

    # ---- prefix_ids unit check (tokenizer only)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_dir(REPO))
    recs = load_records()
    c0, _ = prepare_case(recs[cases[0].case_id], tok)
    c1, _ = prepare_case(recs[cases[0].case_id], tok, prefix_ids=[50279, 187])
    assert c1.ids == [50279, 187] + c0.ids and c1.subject_pos == [p + 2 for p in c0.subject_pos]
    assert (c1.true_id, c1.foil_id) == (c0.true_id, c0.foil_id)
    rep["prefix_ids_check"] = "ok"

    # ---- engine pass: rows 0..n-1 default positions, rows n..2n-1 with pos_offset=1
    eng = Engine(REPO)
    L = eng.spec.n_layers
    pre = [PrefillSpec(c.ids, c.true_id, c.foil_id) for c in cases]
    pre += [PrefillSpec(c.ids, c.true_id, c.foil_id, pos_offset=1) for c in cases]
    diag = DiagSpec(attn_final=True, resid_norms=True, router_logits_final=True, resid_final=True, token_logprobs=True,
                    route_all_layers=(0, L // 2, L - 1))
    res = eng.run(pre, [], record_routing=True, log=log, diag=diag)
    dg = res.extra["diag"]
    rep["use_offsets"] = res.extra["use_offsets"]

    # ---- sink transplant identity check: donor rows = [X] + prompt (X = '\n', id 187); transplant rows = prompt with
    # RoPE positions starting at 1 and the donor's position-0 K/V as an extra slot. With value scale 1 the content
    # tokens see exactly what they see in the donor row, so final logits, routing and layer-patch rescues must agree
    # to bf16 noise (prefill AND wavefront paths). Key-only (value scale 0) rows must run and differ.
    from moetrace.engine import SpawnSpec
    from moetrace.noise import noise_draw
    sigma = 3.0 * eng.embed_std
    Hd = eng.hidden
    X = 187
    pre2 = [PrefillSpec([X] + c.ids, c.true_id, c.foil_id) for c in cases]  # 0..n-1 donors (clean)
    pre2 += [PrefillSpec([X] + c.ids, c.true_id, c.foil_id, [p + 1 for p in c.subject_pos], noise_draw(c.case_id, len(c.subject_pos), Hd, sigma)) for c in cases]  # n..2n-1 donors noised
    pre2 += [PrefillSpec(c.ids, c.true_id, c.foil_id, pos_offset=1, sink_donor=i, sink_vscale=1.0) for i, c in enumerate(cases)]  # 2n..3n-1 transplant clean
    pre2 += [PrefillSpec(c.ids, c.true_id, c.foil_id, c.subject_pos, noise_draw(c.case_id, len(c.subject_pos), Hd, sigma), pos_offset=1, sink_donor=i, sink_vscale=1.0) for i, c in enumerate(cases)]  # 3n..4n-1 transplant noised
    pre2 += [PrefillSpec(c.ids, c.true_id, c.foil_id, pos_offset=1, sink_donor=i, sink_vscale=0.0) for i, c in enumerate(cases)]  # 4n..5n-1 key-only clean
    pre2 += [PrefillSpec(c.ids, c.true_id, c.foil_id, pos_offset=1) for i, c in enumerate(cases)]  # 5n..6n-1 no sink, shifted (= plain prompt)
    sp2, tg2 = [], []
    for i in range(n):
        for l in range(L):
            sp2.append(SpawnSpec(l, n + i, i, "layer")); tg2.append(("donor", i, l))
            sp2.append(SpawnSpec(l, 3 * n + i, 2 * n + i, "layer")); tg2.append(("transplant", i, l))
    res2 = eng.run(pre2, sp2, record_routing=True, log=log, diag=DiagSpec(attn_final=True))
    d2 = res2.delta
    rep["sink_identity_delta_maxdiff_clean"] = float(np.abs(d2[2 * n : 3 * n] - d2[:n]).max())
    rep["sink_identity_delta_maxdiff_noised"] = float(np.abs(d2[3 * n : 4 * n] - d2[n : 2 * n]).max())
    ra = res2.route_idx
    rep["sink_identity_routing_set_agreement_clean"] = float(np.mean([sorted(ra[l, 2 * n + i].tolist()) == sorted(ra[l, i].tolist()) for l in range(L) for i in range(n)]))
    sd2 = res2.sp_delta
    tg2a = np.array([t[0] for t in tg2])
    rd = sd2[tg2a == "donor"] - d2[n : 2 * n].repeat(L)
    rt = sd2[tg2a == "transplant"] - d2[3 * n : 4 * n].repeat(L)
    rep["sink_identity_layer_rescue_maxdiff"] = float(np.abs(rd - rt).max())
    rep["sink_identity_layer_rescue_mean_absdiff"] = float(np.abs(rd - rt).mean())
    rep["sink_keyonly_delta_mean_absdiff_vs_donor"] = float(np.abs(d2[4 * n : 5 * n] - d2[:n]).mean())
    rep["sink_keyonly_delta_mean_absdiff_vs_plain"] = float(np.abs(d2[4 * n : 5 * n] - d2[5 * n : 6 * n]).mean())
    rep["shift1_identity_delta_maxdiff_vs_default"] = float(np.abs(d2[5 * n : 6 * n] - res.delta[:n]).max())
    att2 = res2.extra["diag"]
    rep["sink_slot_mass_transplant_vs_donor_pos0_maxdiff"] = float(np.abs(att2["attn_sink"][:, 2 * n : 3 * n].astype(np.float32) - att2["attn_final"][:, :n, :, 0].astype(np.float32)).max())
    rep["sink_slot_mass_keyonly_mean"] = float(att2["attn_sink"][:, 4 * n : 5 * n].astype(np.float32).mean())
    rep["sink_slot_mass_transplant_mean"] = float(att2["attn_sink"][:, 2 * n : 3 * n].astype(np.float32).mean())
    del eng
    torch.cuda.empty_cache()

    # ---- HF reference
    model = _hf_load(REPO)
    gate_logits, gate_idx = {}, {}

    def mk(l):
        def hook(mod, args, out):
            gate_logits[l] = out[0].detach().float().cpu().numpy()
            gate_idx[l] = out[2].detach().cpu().numpy()
        return hook

    handles = [model.model.layers[l].mlp.gate.register_forward_hook(mk(l)) for l in range(L)]
    attn_diff, norm_diff_l1, norm_diff_l0, rl_diff, lp_diff, route_agree, route_tot, off_diff, off_vs_noff = [], [], [], [], [], 0, 0, [], []
    final_norm_diff = []
    with torch.no_grad():
        for i, c in enumerate(cases):
            ids = torch.tensor([c.ids], device="cuda")
            T = len(c.ids)
            out = model(input_ids=ids, output_attentions=True, output_hidden_states=True)
            # attention of the final position
            for l in range(L):
                a_hf = out.attentions[l][0, :, -1, :].float().cpu().numpy()  # [nH, T]
                a_en = dg["attn_final"][l, i, :, :T].astype(np.float32)
                attn_diff.append(float(np.abs(a_hf - a_en).max()))
            # residual norms: hidden_states[j] convention test (j = l+1 -> output of layer l)
            hs = [h[0].float().norm(dim=-1).cpu().numpy() for h in out.hidden_states]
            for l in range(L - 1):
                en = dg["resid_norms"][l, i, :T]
                norm_diff_l1.append(float((np.abs(hs[l + 1] - en) / np.maximum(en, 1e-6)).max()))
                norm_diff_l0.append(float((np.abs(hs[l] - en) / np.maximum(en, 1e-6)).max()))
            # router logits at the final position and all-token routing at the recorded layers
            for l in range(L):
                rl_diff.append(float(np.abs(gate_logits[l][-1] - dg["router_logits_final"][l, i]).max()))
            for l, (ti, tv, rl) in dg["route_all"].items():
                for t in range(T):
                    route_tot += 1
                    route_agree += int(sorted(ti[i, t].tolist()) == sorted(gate_idx[l][t].tolist()))
            # token log-probs
            logp = torch.log_softmax(out.logits[0].float(), dim=-1)
            g = logp[torch.arange(T - 1), ids[0, 1:]].cpu().numpy()
            lp_diff.append(float(np.abs(g - dg["token_logprobs"][i, : T - 1]).max()))
            d_hf = float(out.logits[0, -1, c.true_id] - out.logits[0, -1, c.foil_id])
            # position offset: HF with position_ids shifted by one
            out1 = model(input_ids=ids, position_ids=torch.arange(1, T + 1, device="cuda")[None])
            d1_hf = float(out1.logits[0, -1, c.true_id] - out1.logits[0, -1, c.foil_id])
            off_diff.append(float(abs(d1_hf - res.delta[n + i])))
            off_vs_noff.append(float(abs(d1_hf - d_hf)))
            final_norm_diff.append(float(abs(res.delta[i] - d_hf)))
    for h in handles:
        h.remove()
    rep.update({
        "attn_final_maxdiff": max(attn_diff), "attn_final_mean_maxdiff": float(np.mean(attn_diff)),
        "resid_norm_max_reldiff_hidden_states[l+1]": max(norm_diff_l1), "resid_norm_max_reldiff_hidden_states[l]": max(norm_diff_l0),
        "router_logits_final_maxdiff": max(rl_diff), "router_logits_final_mean_maxdiff": float(np.mean(rl_diff)),
        "route_all_set_agreement": f"{route_agree}/{route_tot}",
        "token_logprobs_maxdiff": max(lp_diff),
        "delta_default_maxdiff_vs_hf": max(final_norm_diff),
        "delta_pos_offset1_maxdiff_vs_hf_position_ids": max(off_diff), "delta_pos_offset1_mean_diff": float(np.mean(off_diff)),
        "hf_delta_change_from_offset_mean_abs": float(np.mean(off_vs_noff)),
        "engine_delta_change_from_offset_mean_abs": float(np.mean(np.abs(res.delta[n:] - res.delta[:n]))),
        "resid_final_shape": list(dg["resid_final"].shape), "attn_final_shape": list(dg["attn_final"].shape),
        "logprob_true_matches_final_logprob": bool(np.allclose(dg["logprob_true"][:n] - dg["logprob_foil"][:n], res.delta[:n], atol=0.1)),
    })
    os.makedirs("/home/ubuntu/MOE/results", exist_ok=True)
    with open("/home/ubuntu/MOE/results/verify_ext3_diag_olmoe.json", "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in rep.items():
        if k != "prompts":
            log(f"{k}: {v}")


if __name__ == "__main__":
    main()
