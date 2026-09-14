"""ext3 gate: the pilot verification after the engine changes must match the backup within bf16 noise, and the
diagnostics/sink-transplant verification must pass. Exit code 1 stops the GPU chain."""
import json, sys
a = json.load(open("/home/ubuntu/MOE/results/verify_olmoe_before_ext3.json"))
b = json.load(open("/home/ubuntu/MOE/results/verify_olmoe.json"))
d = json.load(open("/home/ubuntu/MOE/results/verify_ext3_diag_olmoe.json"))
ok = True
rows = []
for k in sorted(set(a) | set(b)):
    if k in ("layer_times_first_pass", "mean_layer_rescue_curve", "engine_pass_s", "pass_total_s"):
        continue
    va, vb = a.get(k), b.get(k)
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        diff = abs(va - vb)
        tol = 0.1 if "maxdiff" in k else 0.05
        flag = "" if diff <= tol else "  <-- MISMATCH"
        ok &= diff <= tol
        rows.append(f"{k:45s} before {va:10.4f}  after {vb:10.4f}  diff {diff:.4f}{flag}")
    else:
        flag = "" if va == vb else "  <-- differs"
        rows.append(f"{k:45s} before {va!s:>10}  after {vb!s:>10}{flag}")
curve_a, curve_b = a["mean_layer_rescue_curve"], b["mean_layer_rescue_curve"]
cd = max(abs(x - y) for x, y in zip(curve_a, curve_b))
rows.append(f"{'mean_layer_rescue_curve max |diff|':45s} {cd:.4f}" + ("" if cd <= 0.05 else "  <-- MISMATCH"))
ok &= cd <= 0.05
checks = {"sink_identity_routing_set_agreement_clean": (">=", 0.9), "sink_identity_delta_maxdiff_clean": ("<=", 0.7),
          "sink_identity_layer_rescue_mean_absdiff": ("<=", 0.1), "shift1_identity_delta_maxdiff_vs_default": ("<=", 0.7),
          "attn_final_mean_maxdiff": ("<=", 0.05), "router_logits_final_mean_maxdiff": ("<=", 0.1),
          "resid_norm_max_reldiff_hidden_states[l+1]": ("<=", 0.05), "token_logprobs_maxdiff": ("<=", 0.5),
          "delta_pos_offset1_maxdiff_vs_hf_position_ids": ("<=", 0.7), "sink_slot_mass_transplant_vs_donor_pos0_maxdiff": ("<=", 0.05)}
for k, (op, th) in checks.items():
    v = d.get(k)
    good = v is not None and ((v >= th) if op == ">=" else (v <= th))
    ok &= good
    rows.append(f"{k:45s} {v!s:>10}  {op} {th}" + ("" if good else "  <-- FAIL"))
print("\n".join(rows))
print("GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
