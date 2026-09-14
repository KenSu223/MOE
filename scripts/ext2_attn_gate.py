"""ext2 gate: (1) verify_olmoe.json after the engine change must be IDENTICAL to the backup on every non-timing metric
(the pass is deterministic; existing kinds' numerics are untouched), (2) the new-kinds verification on OLMoE must be
within the bf16 noise floor established by the existing `layer` kind. Exit code 1 stops the GPU chain."""
import json, sys
R = "/home/ubuntu/MOE/results/"
a = json.load(open(R + "verify_olmoe_before_ext2.json"))
b = json.load(open(R + "verify_olmoe.json"))
ok = True
rows = []
TIMING = ("layer_times_first_pass", "engine_pass_s", "pass_total_s")
n_ident = 0
for k in sorted(set(a) | set(b)):
    if k in TIMING:
        continue
    va, vb = a.get(k), b.get(k)
    same = va == vb
    n_ident += same
    ok &= same
    rows.append(f"{k:45s} before {va!s:>22}  after {vb!s:>22}" + ("" if same else "  <-- NOT IDENTICAL"))
rows.append(f"verify_olmoe identity: {n_ident}/{len(rows)} non-timing metrics bit-identical")
try:
    d = json.load(open(R + "verify_ext2_attn_olmoe.json"))
except FileNotFoundError:
    d = None
    rows.append("verify_ext2_attn_olmoe.json missing  <-- FAIL")
    ok = False
if d is not None:
    # bf16 noise floor = the already-verified `layer` kind on the same cases and layers (per-case max, mean |diff|,
    # and the max deviation of the 20-case mean curve); the new kinds must not exceed 1.5x that floor
    floor = d["layer_vs_hf_maxdiff"]
    floor_mean = d["layer_vs_hf_meanabsdiff"]
    floor_curve = d["layer_mean_curve_maxdiff_vs_hf"]
    checks = [("attn_layer_vs_hf_maxdiff", "<=", max(0.7, 1.5 * floor)), ("block_vs_hf_maxdiff", "<=", max(0.7, 1.5 * floor)),
              ("resid_vs_hf_maxdiff", "<=", max(0.7, 1.5 * floor)),
              ("attn_layer_vs_hf_meanabsdiff", "<=", 1.5 * floor_mean), ("block_vs_hf_meanabsdiff", "<=", 1.5 * floor_mean),
              ("resid_vs_hf_meanabsdiff", "<=", 1.5 * floor_mean),
              ("attn_layer_mean_curve_maxdiff_vs_hf", "<=", max(0.1, 1.5 * floor_curve)),
              ("block_mean_curve_maxdiff_vs_hf", "<=", max(0.1, 1.5 * floor_curve)),
              ("resid_mean_curve_maxdiff_vs_hf", "<=", max(0.1, 1.5 * floor_curve)),
              ("attn_layer_rescue_corr_vs_hf", ">=", 0.95), ("block_rescue_corr_vs_hf", ">=", 0.95), ("resid_rescue_corr_vs_hf", ">=", 0.95),
              ("identity_attn_layer_maxdiff", "<=", 0.5), ("identity_block_maxdiff", "<=", 0.5), ("identity_resid_maxdiff", "<=", 0.5),
              ("block_vs_block_diff_maxdiff", "<=", 0.7), ("block_vs_block_diff_frac_within_0.1", ">=", 0.8)]
    rows.append(f"{'bf16 noise floor (layer_vs_hf_maxdiff)':45s} {floor:.4f}")
    for k, op, th in checks:
        v = d.get(k)
        good = v is not None and ((v >= th) if op == ">=" else (v <= th))
        ok &= good
        rows.append(f"{k:45s} {v!s:>10}  {op} {th:.3f}" + ("" if good else "  <-- FAIL"))
    for k in ("vnorm_vs_hf",):
        rows.append(f"{k}: {json.dumps(d[k])}")
print("\n".join(rows))
print("GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
