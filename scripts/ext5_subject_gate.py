"""ext5 F4 gate: the subject-site verification on OLMoE must sit within the bf16 noise floor before the big models run.

Floors: the `layer` kind at p against HF hooks (same cases and layers) calibrates per-case and mean-curve deviations
(as in ext2_attn_gate.py); the final-token consistency check must agree with the stored ext2 rows within the engine's
own bf16 noise (zero_on_noised_maxdiff of verify_olmoe.json is 0.41). Exit code 1 stops the chain."""
import json, sys
R = "/home/ubuntu/MOE/results/"
try:
    d = json.load(open(R + "verify_ext5_subject_olmoe.json"))
except FileNotFoundError:
    print("verify_ext5_subject_olmoe.json missing\nGATE FAIL")
    sys.exit(1)
ok = True
rows = []
floor, floor_mean, floor_curve = d["layer_vs_hf_maxdiff"], d["layer_vs_hf_meanabsdiff"], d["layer_mean_curve_maxdiff_vs_hf"]
# noised prefill runs are chaotic per case (verify_olmoe.json: noised 0.61 vs clean 0.38 max vs HF); 1.5 = 2.5x that floor
checks = [("delta_clean_maxdiff_vs_hf", "<=", 0.7), ("delta_noised_maxdiff_vs_hf", "<=", 1.5),
          ("delta_noised_lastonly_maxdiff_vs_hf", "<=", 1.5), ("delta_noised_exceptlast_maxdiff_vs_hf", "<=", 1.5),
          ("prefill_rec_pos_vs_final_maxdiff", "<=", 0.0),
          ("layer_vs_hf_maxdiff", "<=", 1.5), ("layer_vs_hf_meanabsdiff", "<=", 0.3), ("layer_rescue_corr_vs_hf", ">=", 0.95)]
for k in ("attn_layer", "resid", "block"):
    checks += [(f"{k}_vs_hf_maxdiff", "<=", max(0.7, 1.5 * floor)), (f"{k}_vs_hf_meanabsdiff", "<=", 1.5 * floor_mean),
               (f"{k}_mean_curve_maxdiff_vs_hf", "<=", max(0.1, 1.5 * floor_curve))]
    # per-case rescue correlation is noise-limited when the HF rescues spread little beyond the bf16 floor (OLMoE
    # attn_layer at p: HF std 0.54 vs per-case engine-HF difference std ~0.2). With independent noise of variance s_d^2/2 on
    # each side the attainable correlation is r_exp = 1 - s_d^2 / (2 s_hf^2); require corr >= min(0.95, r_exp - 0.03).
    s_hf, s_d = d.get(f"{k}_hf_rescue_std"), d.get(f"{k}_rescue_diff_std")
    r_exp = 1.0 - s_d ** 2 / (2 * s_hf ** 2) if (s_hf and s_d) else 1.0
    checks.append((f"{k}_rescue_corr_vs_hf", ">=", round(min(0.95, r_exp - 0.03), 3)))
checks += [("zero_on_noised_maxdiff", "<=", 0.5)] + [(f"identity_{k}_maxdiff", "<=", 0.5) for k in ("attn_layer", "layer", "resid", "block")]
rows.append(f"{'bf16 floor (layer_vs_hf_maxdiff / meanabs / curve)':52s} {floor:.4f} / {floor_mean:.4f} / {floor_curve:.4f}")
for k, op, th in checks:
    v = d.get(k)
    good = v is not None and ((v >= th) if op == ">=" else (v <= th))
    ok &= good
    rows.append(f"{k:52s} {v!s:>10}  {op} {th:.3f}" + ("" if good else "  <-- FAIL"))
# (e) same prefill batch through run_subject (p = T-1) and Engine.run: must agree within the engine's own bf16 floor
e = d["final_token_consistency_in_process"]
good = e["prefill_delta_maxdiff"] == 0.0 and e["delta_maxdiff"] <= 0.6 and e["delta_meanabsdiff"] <= 0.1
ok &= good
rows.append(f"in-process consistency (run_subject vs Engine.run) n={e['n']}: prefill maxdiff {e['prefill_delta_maxdiff']:.4f} (== 0), "
            f"spawn delta maxdiff {e['delta_maxdiff']:.4f} (<= 0.6) meanabs {e['delta_meanabsdiff']:.4f} (<= 0.1) frac_equal {e['delta_frac_equal']:.2f} "
            f"vnorm maxdiff {e['vnorm_maxdiff']:.4f}" + ("" if good else "  <-- FAIL"))
# (d) vs the stored ext2 rows (different batch composition -> batch-dependent bf16 noise, ~0.15 mean): curves and ranking
for kind, c in d["final_token_consistency_vs_olmoe_attnsweep"].items():
    good = c["mean_curve_maxdiff"] <= 0.15 and c["rescue_corr"] >= 0.95 and c["delta_meanabsdiff"] <= 1.5 * floor_mean
    ok &= good
    rows.append(f"stored-rows consistency {kind:10s} n={c['n']} delta meanabs {c['delta_meanabsdiff']:.4f} (<= {1.5 * floor_mean:.3f}) "
                f"rescue corr {c['rescue_corr']:.3f} (>= 0.95) mean-curve maxdiff {c['mean_curve_maxdiff']:.4f} (<= 0.15)" + ("" if good else "  <-- FAIL"))
rows.append(f"vnorm_vs_hf: {json.dumps(d['vnorm_vs_hf'])}")
print("\n".join(rows))
print("GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
