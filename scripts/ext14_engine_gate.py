"""ext14 (p6-engine) pass / fail gate on verification JSONs of scripts/ext14_engine_verify.py.

Three precision regimes (chosen per JSON from its dtype and units):
  tiny32  fp32 tiny random checkpoints: exact to fp32 precision (|dDelta| <= 1e-4, correlations >= 0.999).
  real32  fp32 runs on the real checkpoints (engine fp32 vs transformers fp32): implementation exactness on real weights;
          fp32 summation-order differences only (|dDelta| <= 0.05 max, <= 0.01 mean per group, rescue r >= 0.999).
  bf16    bf16 runs: deviations of the order of transformers' own bf16 noise. Absolute defaults (max |dDelta| <= 1.0, mean
          per group <= 0.15, rescue r >= 0.99, groups with effects >= 0.3: r >= 0.95). With --noise <chain JSON> the bf16
          full-mode thresholds are scaled by the model's measured noise floor (transformers bf16 vs transformers fp32 on
          the same prompts, chain mode): mean per group <= max(0.15, 2.5 x floor), max <= max(1.0, 12 x floor); the overall
          rescue correlation must then be >= 0.98.
Per-layer mode: update relerr vs transformers <= 0.05, or, when the fp32 reference layer was run, engine-vs-fp32 relerr <=
  1.5 x transformers(bf16)-vs-fp32 + 0.01 and engine-vs-fp32 routing agreement >= transformers-vs-fp32 - 0.1 (flat routers
  make bf16 routing ties common; the noise floor decides). Engine identities, sanity prompts (answer rank <= 5 in >= 3).
Chain mode: engine fp32 vs transformers fp32 (when it ran): |dDelta| max <= 0.05, routing agreement mean >= 0.98; engine bf16
  vs transformers fp32 within 1.5 x the transformers bf16-vs-fp32 deviation + 0.05 (|dDelta| mean) and routing agreement
  >= transformers bf16-vs-fp32 - 0.05.
Usage: python scripts/ext14_engine_gate.py <json> [<json> ...] [--fp32] [--noise <chain json>]
Exit 0 = PASS (prints PASS), 1 = FAIL (prints the failing checks).
"""
import json, sys


def regime(r, fp32_flag):
    if r.get("dtype") == "fp32":
        return "tiny32" if r.get("units", {}).get("kind") == "tiny_random" else "real32"
    return "tiny32" if fp32_flag else "bf16"


def check_full(r, reg, fails, notes, floor=None):
    T = {"tiny32": dict(tol=1e-4, mean=1e-5, corr=0.999, gcorr=0.999, geff=0.05, ident=1e-4, diag=1e-4, metric=1e-4),
         "real32": dict(tol=0.05, mean=0.01, corr=0.999, gcorr=0.99, geff=0.05, ident=0.05, diag=1e-3, metric=0.05),
         "bf16": dict(tol=1.0, mean=0.15, corr=0.99, gcorr=0.95, geff=0.3, ident=0.5, diag=None, metric=0.5)}[reg]
    if reg == "bf16" and floor is not None:
        T = dict(T, tol=max(1.0, 12 * floor), mean=max(0.15, 2.5 * floor), corr=0.98, geff=max(0.3, 5 * floor),
                 ident=max(0.5, 6 * floor), metric=max(0.5, 6 * floor))
        notes.append(f"bf16 thresholds scaled by the transformers bf16-vs-fp32 noise floor {floor:.3f}: "
                     f"max {T['tol']:.2f}, mean {T['mean']:.2f}, overall r >= {T['corr']}")
    c = r["spawn_rescue_corr_all"]
    notes.append(f"[{reg}] rescue corr {c:.5f}, spawn max |dDelta| {r['spawn_delta_maxabs_all']:.3g}"
                 + (f", competent units r {r['spawn_rescue_corr_competent']:.5f}" if r.get("spawn_rescue_corr_competent") else ""))
    if c is None or c < T["corr"]:
        if reg == "bf16" and r.get("spawn_rescue_corr_competent") and r["spawn_rescue_corr_competent"] >= T["corr"]:
            notes.append(f"overall rescue corr {c:.4f} < {T['corr']}, but {r['spawn_rescue_corr_competent']:.4f} on the competent units "
                         f"({len(r['competent_units'])} with transformers clean Delta >= 1)")
        else:
            fails.append(f"rescue corr {c}")
    if r["spawn_delta_maxabs_all"] > T["tol"]:
        fails.append(f"spawn max |dDelta| {r['spawn_delta_maxabs_all']}")
    pm = max(v["maxabs"] for v in r["prefill_delta_vs_hf"].values())
    notes.append(f"prefill max |dDelta| {pm:.3g}")
    if pm > T["tol"]:
        fails.append(f"prefill max |dDelta| {pm}")
    groups = r["spawn_delta_vs_hf"]
    for need in ("hmz_layer", "hmm_layer", "hmz_multi", "hmz_attn"):
        if need not in groups:
            fails.append(f"missing group {need}")
    if r["family"] not in ("mistral", "qwen2", "qwen3"):
        for need in ("rmr_layer", "rmr_attn", "rmr_multi", "rmz_layer"):
            if need not in groups:
                fails.append(f"missing group {need}")
    if r.get("shared_inter"):
        for need in ("rms_layer", "rms_attn", "rms_multi", "rms_allmlp", "rms_sharedmasked", "rms_sharedfree", "den_multi_shared"):
            if need not in groups:
                fails.append(f"missing group {need}")
        if not r.get("shared_reroute_raises"):
            fails.append("route_mask reroute on the shared expert does not raise")
        if r.get("route_mask_effect_on_delta_hf", {}).get("Rs", 0) <= 0:
            fails.append("shared-expert knockout without effect")
    for g, v in groups.items():
        if v["maxabs"] > T["tol"]:
            fails.append(f"group {g} max |dDelta| {v['maxabs']:.3g}")
        if v["meanabs"] > T["mean"]:
            fails.append(f"group {g} mean |dDelta| {v['meanabs']:.3g}")
        if v.get("rescue_corr") is not None and v["rescue_hf_meanabs"] >= T["geff"] and v["n"] >= 8 and v["rescue_corr"] < T["gcorr"]:
            fails.append(f"group {g} rescue corr {v['rescue_corr']:.3f}")
    ide = r["identities"]
    if ide["neurons_linearity_relerr_max"] > 1e-4:
        fails.append(f"neuron linearity {ide['neurons_linearity_relerr_max']}")
    for k in ("neurons_all_vs_expert_ddelta", "attn_head_on_masked_head_vs_parent", "zero_vs_parent",
              "coalition_union_or_dense_clean_vs_layer_ddelta"):
        if ide[k]["maxabs"] > T["ident"]:
            fails.append(f"identity {k} {ide[k]['maxabs']:.3g}")
    # diagnostics: exact in fp32; in bf16 the engine and transformers runs differ by accumulated bf16 noise (residual
    # stream ~1 %), amplified in head means (cancellation) and SiLU(g) * u activations
    lim = {"diag_resid_layers_relerr": ("maxabs", 0.05), "diag_attn_heads_mean_relerr": ("meanabs", 0.05),
           "diag_expert_neurons_act_relerr": ("meanabs", 0.15)}
    for k, (stat, t3) in lim.items():
        if T["diag"] is not None:
            stat, t3 = "maxabs", T["diag"]
        elif floor is not None:
            t3 = max(t3, t3 * floor / 0.05)  # scaled with the measured noise floor (0.05 = OLMoE / Mistral-like)
        if r[k].get("n", 0) and r[k][stat] > t3:
            fails.append(f"{k} {r[k][stat]:.3g}")
    if r.get("diag_expert_neurons_routed_mismatch", 0) > 0:
        notes.append(f"expert_neurons routed flag differs from transformers in {r['diag_expert_neurons_routed_mismatch']} rows (bf16 routing)")
    for k in ("logp_true", "kl_to_clean"):
        if r["spawn_metrics_vs_hf"][k]["maxabs"] > T["metric"]:
            fails.append(f"metric {k} {r['spawn_metrics_vs_hf'][k]['maxabs']:.3g}")
    if min(r["head_mask_effect_on_delta_hf"].values()) <= 0:
        fails.append("head masks without effect")


def check_pl(r, reg, fails, notes, floor=None):
    pl = r["per_layer"]
    t = 1e-4 if reg == "tiny32" else (0.005 if reg == "real32" else 0.05)
    for l, v in pl["layers"].items():
        ur = v["update_relerr"]["maxabs"]
        s = f"layer {l}: update relerr max {ur:.3g}"
        if v.get("routing_set_agreement") is not None:
            s += f", routing {v['routing_set_agreement']:.3f}"
        if "engine_vs_fp32_update_relerr" in v:
            e32, h32 = v["engine_vs_fp32_update_relerr"]["maxabs"], v["hf_vs_fp32_update_relerr"]["maxabs"]
            s += f"; vs fp32: engine {e32:.3g} / transformers {h32:.3g}"
            if v.get("engine_vs_fp32_routing_agreement") is not None:
                ra_e, ra_h = v["engine_vs_fp32_routing_agreement"], v["hf_vs_fp32_routing_agreement"]
                s += f", routing vs fp32: engine {ra_e:.3f} / transformers {ra_h:.3f}"
                if ra_e < ra_h - 0.1:
                    fails.append(f"layer {l} routing vs fp32 {ra_e:.3f} < transformers {ra_h:.3f} - 0.1")
            if e32 > 1.5 * h32 + 0.01:
                fails.append(f"layer {l} update relerr vs fp32 {e32:.3g} > 1.5 x transformers {h32:.3g} + 0.01")
            notes.append(s)
            continue
        notes.append(s)
        if ur > t and v.get("update_relerr_routing_equal"):  # MoE: judge on positions whose routed set agrees
            notes.append(f"layer {l}: update relerr on positions with equal routing {v['update_relerr_routing_equal']['maxabs']:.3g}")
            ur = v["update_relerr_routing_equal"]["maxabs"]
        if ur > t:
            fails.append(f"layer {l} update relerr {ur}")
        if v.get("routing_set_agreement") is not None and v["routing_set_agreement"] < 0.9:
            fails.append(f"layer {l} routing agreement {v['routing_set_agreement']}")
    ide = pl["identities"]
    t2 = 1e-4 if reg == "tiny32" else 0.5
    for k in ("union_vs_layer", "setall_vs_layer", "neurons_all_vs_expert", "shared_expert_vs_set", "shared_neurons_all_vs_expert"):
        if k in ide and ide[k]["maxabs"] > t2:
            fails.append(f"identity {k} {ide[k]['maxabs']:.3g}")
    if ide["neurons_linearity_relerr_max"] > 1e-4:
        fails.append(f"neuron linearity {ide['neurons_linearity_relerr_max']}")
    san = pl["behaviour"]["sanity"]
    if san:
        good = sum(1 for x in san if x["rank"] <= 5)
        notes.append(f"sanity {good}/{len(san)} answers in the top 5: " + ", ".join(f"{x['answer'].strip()}#{x['rank']}" for x in san))
        if good < min(3, len(san)):
            fails.append(f"sanity {good}/{len(san)}")
    notes.append(f"clean Delta > 0 in {ide['clean_delta_pos_frac']:.2f} of the units, mean drop {ide['drop_mean']:.2f}")


def check_chain(r, reg, fails, notes, floor=None):
    ch = r["chain"]
    cmp_ = ch["compare"]
    for k, v in cmp_.items():
        notes.append(f"chain {k}: |dDelta| mean {v['delta_absdiff']['meanabs']:.3g} max {v['delta_absdiff']['maxabs']:.3g}, "
                     f"rank equal {v['rank_equal_frac']:.2f}"
                     + (f", routing agreement mean {v['routing_agreement_mean']:.3f} min {v['routing_agreement_min']:.3f}"
                        if "routing_agreement_mean" in v else ""))
    for tag, run in ch["runs"].items():
        if "error" in run:
            notes.append(f"chain {tag}: {run['error'][:80]}")
        else:
            notes.append(f"chain {tag} sanity ranks {run['sanity_rank']}")
    e32 = cmp_.get("eng_fp32__hf_fp32")
    if e32 is not None:
        if e32["delta_absdiff"]["maxabs"] > 0.05:
            fails.append(f"engine fp32 vs transformers fp32 |dDelta| {e32['delta_absdiff']['maxabs']:.3g}")
        if "routing_agreement_mean" in e32 and e32["routing_agreement_mean"] < 0.98:
            fails.append(f"engine fp32 vs transformers fp32 routing agreement {e32['routing_agreement_mean']:.3f}")
    eb, hb = cmp_.get("eng_bf16__hf_fp32"), cmp_.get("hf_bf16__hf_fp32")
    if eb is not None and hb is not None:
        if eb["delta_absdiff"]["meanabs"] > 1.5 * hb["delta_absdiff"]["meanabs"] + 0.05:
            fails.append(f"engine bf16 vs fp32 |dDelta| mean {eb['delta_absdiff']['meanabs']:.3g} > 1.5 x transformers "
                         f"{hb['delta_absdiff']['meanabs']:.3g} + 0.05")
        if "routing_agreement_mean" in eb and eb["routing_agreement_mean"] < hb["routing_agreement_mean"] - 0.05:
            fails.append(f"engine bf16 routing vs fp32 {eb['routing_agreement_mean']:.3f} < transformers "
                         f"{hb['routing_agreement_mean']:.3f} - 0.05")
    else:
        fails.append("chain comparisons missing")


def check_vsfp32(r, reg, fails, notes, floor=None):
    """engine (model protocol) vs engine fp32 on a T2-like plan: reported (the model's own bf16 sensitivity, not an
    engine-vs-transformers check); fails only if the protocol run is not even rank-consistent (r < 0.9)."""
    v = r["vs_fp32"]
    notes.append(f"engine {v['protocol']} vs engine fp32: rescue r {v['rescue_corr_all']:.4f}, |d rescue| mean "
                 f"{v['rescue_absdiff']['meanabs']:.3g}, attention share {v['attn_share_mean_curve']['proto']:.3f} vs "
                 f"{v['attn_share_mean_curve']['fp32']:.3f}" + (f", final routing agreement {v['final_routing_agreement_mean']:.3f}"
                                                                 if "final_routing_agreement_mean" in v else ""))
    if v["rescue_corr_all"] is None or v["rescue_corr_all"] < 0.9:
        fails.append(f"engine protocol vs fp32 rescue corr {v['rescue_corr_all']}")


def noise_floor(path):
    r = json.load(open(path))
    return r["chain"]["compare"]["hf_bf16__hf_fp32"]["delta_absdiff"]["meanabs"]


def main():
    argv = sys.argv[1:]
    floor = None
    if "--noise" in argv:
        i = argv.index("--noise")
        floor = noise_floor(argv[i + 1])
        del argv[i:i + 2]
    fp32 = "--fp32" in argv
    args = [a for a in argv if not a.startswith("--")]
    fails, notes = [], []
    for p in args:
        r = json.load(open(p))
        reg = regime(r, fp32)
        if "vs_fp32" in r:
            check_vsfp32(r, reg, fails, notes, floor)
        elif "verdict" in r:  # scripts/ext14_engine_noise_compare.py
            notes.append(f"noise compare: corr vs fp32 transformers bf16 {r['corr']['hf_bf16_vs_fp32']:.4f} / engine bf16 "
                         f"{r['corr']['eng_bf16_vs_fp32']:.4f}; mean |d| {r['mean_absdiff']['hf_bf16_vs_fp32']:.3g} / "
                         f"{r['mean_absdiff']['eng_bf16_vs_fp32']:.3g}; engine fp32 vs transformers fp32 r "
                         f"{r['corr']['eng_fp32_vs_hf_fp32']:.6f} -> {r['verdict']}")
            if r["verdict"] != "noise":
                fails.append("noise compare verdict: engine")
        elif "chain" in r:
            check_chain(r, reg, fails, notes, floor)
        elif "per_layer" in r:
            check_pl(r, reg, fails, notes, floor)
        else:
            check_full(r, reg, fails, notes, floor)
    for n in notes:
        print("  " + n)
    if fails:
        print("FAIL: " + "; ".join(fails))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
