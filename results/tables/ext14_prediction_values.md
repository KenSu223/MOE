**Pre-registered predictions: observed values (p6-analysis; the verdict table is the coordinator's ext14_prereg_check.md)**

| ID | Cell | Prediction | Observed | Verdict | Verdict where testable |
|---|---|---|---|---|---|
| L1 | L-ant | <= 0.30 | qwen3: 0.51 [0.49, 0.54] (failed) | failed | failed (1 testable) |
| L2 | L-ant | <= 0.40 | mixtral_bos: 0.48 [0.45, 0.51] (failed) | failed | failed (1 testable) |
| L3 | L-prop | 0.30-0.55 | qwen3: 0.33 [0.30, 0.36] (held) | held | held (1 testable) |
| L4 | L-prop | 0.35-0.60 | mixtral_bos: 0.42 [0.40, 0.44] (held) | held | held (1 testable) |
| L5 | L-bind | >= 0.75 | qwen3: 0.79 [0.77, 0.81] (held) | held | held (1 testable) |
| L6 | L-bind | >= 0.65 | mixtral_bos: 0.71 [0.70, 0.73] (held) | held | held (1 testable) |
| L7 | L-ind | >= 0.85 | qwen3: 0.77 [0.73, 0.80] (insufficient (pooled only)) | insufficient (pooled only) | – |
| L8 | L-ind | >= 0.75 | mixtral_bos: 0.71 [0.69, 0.73] (failed) | failed | failed (1 testable) |
| L9 | ladder | >= 0.8 in every model | qwen3: 0.81 (held); mixtral_bos: 0.75 (failed); mistral7b_bos: 0.80 (held); qwen2_7b: 0.55 (failed); qwen2_57b: 0.74 (failed); qwen3_4b: 0.87 (held); qwen3_8b: 0.55 (failed); qwen3_14b: 0.72 (failed) | failed | failed (8 testable) |
| E1 | E1-a2e | >= 0.80 | qwen3: 0.81 [0.78, 0.83] (held) | held | held (1 testable) |
| E2 | E1-a2e | >= 0.70 | mixtral_bos: 0.62 [0.60, 0.64] (failed) | failed | failed (1 testable) |
| E3 | E1p-a2e vs E1-a2e | \|E1p - E1\| <= 0.10 | qwen3: 0.01 [-0.01, 0.02] (held); mixtral_bos: insufficient data | insufficient data | held (1 testable) |
| E4 | E2-a2e vs E1-a2e | E1 - E2 >= 0.10 | qwen3: insufficient data; mixtral_bos: insufficient data | insufficient data | – |
| E5 | E3-a2e | 0.65-0.80 (= Phase-5 F1 level) | qwen3: 0.73 [0.70, 0.76] (held) | held | held (1 testable) |
| E6 | E3-a2e | 0.45-0.65 | mixtral_bos: 0.52 [0.50, 0.54] (insufficient (pooled only)) | insufficient (pooled only) | – |
| E7 | E1-e2a | >= 0.70 | qwen3: 0.22 [0.20, 0.26] (failed) | failed | failed (1 testable) |
| E8 | E1-e2a | >= 0.65 | mixtral_bos: 0.50 [0.47, 0.53] (insufficient (pooled only)) | insufficient (pooled only) | – |
| E9 | E1p-e2a vs E1-e2a | E1 - E1p >= 0.10 | qwen3: 0.00 [-0.04, 0.04] (failed); mixtral_bos: -0.05 [-0.09, -0.00] (insufficient (pooled only)) | failed | failed (1 testable) |
| E10 | E2-e2a vs E1p-e2a | E1p - E2 > 0 | qwen3: insufficient data; mixtral_bos: insufficient data | insufficient data | – |
| E11 | E3-e2a | 0.10-0.30 (= option swap level) | qwen3: 0.18 [0.15, 0.21] (held) | held | held (1 testable) |
| E12 | E3-e2a | 0.25-0.45 | mixtral_bos: 0.34 [0.33, 0.35] (held) | held | held (1 testable) |
| E13 | direction gap a2e - e2a at E1 | H-inference (best guess): \|gap(E1)\| <= 0.10; H-type: gap(E1) >= 0.5 * gap(E3) | qwen3: H-type (failed); mixtral_bos: insufficient (pooled only) | failed | failed (1 testable) |
| E14 | dial E monotonicity | a2e: E1 >= E1p > E2 and E1 > E3; e2a: E1 > E1p > E2 >= E3 | qwen3:a2e: e1 0.81, e1p 0.83, e2 n/a, e3p3 0.77 (insufficient data); qwen3:e2a: e1 0.22, e1p 0.24, e2 n/a, e3 0.18 (insufficient data); mixtral_bos:a2e: e1 0.62, e1p n/a, e2 n/a, e3p3 0.55 (insufficient data); mixtral_bos:e2a: e1 0.50, e1p 0.47, e2 n/a, e3 0.34 (insufficient data) | insufficient data | – |
| A1 | Dial A (a2e and e2a) | conflict >= neutral(E1) > agree > E0 | qwen3:a2e: conflict 0.83, neutral 0.81, agree 0.79, E0 0.73 (held); qwen3:e2a: conflict n/a, neutral 0.22, agree 0.34, E0 0.44 (insufficient data); mixtral_bos:a2e: conflict n/a, neutral 0.62, agree n/a, E0 n/a (insufficient data); mixtral_bos:e2a: conflict n/a, neutral 0.50, agree n/a, E0 n/a (insufficient data) | insufficient data | held (1 testable) |
| A2 | A-conf | < 0 | qwen3:a2e: -0.30 [-0.35, -0.26] (held); qwen3:e2a: insufficient data; mixtral_bos:a2e: insufficient data; mixtral_bos:e2a: insufficient data | insufficient data | held (1 testable) |
| A3 | E0 | <= 0.35 | qwen3:e0_a2e: 0.73 [0.72, 0.74] (failed); qwen3:e0_e2a: 0.44 [0.42, 0.47] (insufficient (pooled only)) | failed | failed (1 testable) |
| A4 | E0 | <= 0.45 | mixtral_bos:e0_a2e: insufficient data; mixtral_bos:e0_e2a: insufficient data | insufficient data | – |
| A5 | Dial A dose-response | monotone decreasing with prior support (5 bins) | qwen3:a2e: -0.80 (held); qwen3:e2a: 0.70 (failed); mixtral_bos:a2e: insufficient data; mixtral_bos:e2a: insufficient data | failed | failed (2 testable) |
| C1 | Set C conflict (E3 + contradicting statement) | the explicit statement wins in >= 60 % of margin items | qwen3: 0.94 (held); mixtral_bos: 0.76 (held); mistral7b_bos: 0.57 (failed); qwen2_7b: 1.00 (held); qwen2_57b: 0.61 (held); qwen3_4b: 1.00 (held); qwen3_8b: 0.92 (held); qwen3_14b: 0.90 (held); qwen3_1p7b: 1.00 (held) | failed | failed (9 testable) |
| C2 | Set C statement-following subset | within 0.10 of E1-a2e | qwen3: -0.04 [-0.12, 0.03] (held); mixtral_bos: insufficient data | insufficient data | held (1 testable) |
| C3 | Set C event-following subset (if >= 64 val pairs) | within 0.10 of E3-a2e | qwen3: insufficient data; mixtral_bos: insufficient data | insufficient data | – |
| R1 | all Stage-1 cells | >= 0.8 | qwen3: nld 0.99, dp 0.97 (held); mixtral_bos: nld 0.99, dp 1.00 (held) | held | held (2 testable) |
| R2 | E1-a2e evidence-site vs question-site | \|diff\| <= 0.15 | qwen3: 0.03 [0.00, 0.05] (held); mixtral_bos: 0.05 [0.02, 0.07] (held) | held | held (2 testable) |
| R3 | E1-a2e, E3-e2a: knockout top-3 mover heads | attention AUC+ drops >= 20 % and share drops >= 0.05 more than with matched random heads | R3:e1_a2e/qwen3: rel_attn_vs_rand -0.23 [-0.27, -0.18], d_share_vs_rand -0.02 [-0.04, -0.00] (failed); R3:e1_a2e/mixtral_bos: rel_attn_vs_rand -0.31 [-0.38, -0.25], d_share_vs_rand -0.07 [-0.08, -0.06] (held); R3:e3_e2a/qwen3: rel_attn_vs_rand -0.12 [-0.28, 0.08], d_share_vs_rand -0.02 [-0.04, 0.00] (failed); R3:e3_e2a/mixtral_bos: rel_attn_vs_rand -0.16 [-0.20, -0.12], d_share_vs_rand -0.03 [-0.04, -0.03] (failed) | failed | failed (4 testable) |
| R4 | E1-a2e, E3-e2a: knockout top-3 experts | drops >= 20 % more than with matched random experts | R4:e1_a2e/qwen3: rel_moe_vs_rand -0.21 [-0.32, -0.07] (held); R4:e1_a2e/mixtral_bos: rel_moe_vs_rand -0.27 [-0.34, -0.18] (held); R4:e3_e2a/qwen3: rel_moe_vs_rand -0.15 [-0.23, -0.06] (failed); R4:e3_e2a/mixtral_bos: rel_moe_vs_rand -0.19 [-0.22, -0.16] (failed) | failed | failed (4 testable) |
| D0 | all Set-A cells | H-invariant for Mistral-7B <-> Mixtral (prior: Li et al. ACL 2025) | mistral7b_bos->mixtral_bos: 1.00 (held) | held | held (1 testable) |
| D1 | Set A | \|share_dense - share_MoE\| <= 0.10 in >= 80 % of cells | mistral7b_bos->mixtral_bos: 1.00 (held); qwen2_7b->qwen2_57b: 0.89 (held) | held | held (2 testable) |
| D2 | Set A | share_MoE - share_dense <= -0.10 in knowledge cells (L-ant, L-prop, L-cf, E3-e2a) and \|diff\| <= 0.10 in copy cells (L-ind, L-ioi, L-bind, E1-*) | mistral7b_bos->mixtral_bos: knowledge: L-ant antonym +0.03 [+0.01, +0.05], L-prop object property +0.06 [+0.04, +0.07], E3 e→a +0.03 [+0.01, +0.04]; copy: L-ind induction +0.03 [-0.00, +0.07], L-ioi IOI +0.02 [+0.01, +0.03], L-bind binding +0.04 [+0.02, +0.06], E1 a→e -0.03 [-0.06, +0.01], E1 e→a +0.02 [-0.02, +0.06] (failed); qwen2_7b->qwen2_57b: knowledge: L-ant antonym -0.11 [-0.13, -0.09], L-prop object property +0.00 [-0.01, +0.02], L-cf CounterFact +0.01 [-0.00, +0.03], E3 e→a -0.01 [-0.04, +0.02]; copy: L-ind induction +0.02 [-0.01, +0.05], L-ioi IOI +0.06 [+0.05, +0.08], E1 a→e -0.02 [-0.04, -0.00], E1 e→a -0.03 [-0.07, +0.02] (failed) | failed | failed (2 testable) |
| D3 | Set A | share_MoE - share_dense >= +0.10 in most cells | mistral7b_bos->mixtral_bos: 0.00 (failed); qwen2_7b->qwen2_57b: 0.00 (failed) | failed | failed (2 testable) |
| D4 | Set A | >= 0.8 | mistral7b_bos->mixtral_bos: 0.98 (held); qwen2_7b->qwen2_57b: 0.90 (held) | held | held (2 testable) |
| D5 | Set A | \|trend\| <= 0.10 per cell; ordering preserved (Spearman >= 0.8 with Qwen3-30B-A3B) | qwen3 series: 14B − 4B per cell L-ant antonym +0.03 [-0.00, +0.05], L-prop object property -0.03 [-0.05, -0.01], L-ioi IOI -0.01 [-0.02, -0.00], L-bind binding -0.02 [-0.03, -0.00], L-ind induction +0.04 [-0.01, +0.08], E1 a→e -0.04 [-0.06, -0.02], E1p a→e -0.02 [-0.04, -0.00], E3 a→e (F1 frame, Phase-5 anchor) -0.03 [-0.05, -0.01], E3 e→a -0.08 [-0.12, -0.04], E3 a→e (P3 frame) -0.01 [-0.03, +0.01]; Spearman with 30B-A3B Qwen3-4B 0.91, Qwen3-8B 0.86, Qwen3-14B 0.92 (held) | held | held (1 testable) |
| D6 | Set A knowledge cells | >= 0.5 | qwen2_57b: l_ant 0.75 [0.72, 0.78], l_prop 0.68 [0.66, 0.69], l_cf 0.75 [0.72, 0.77], e3_e2a 0.51 [0.48, 0.54] (held) | held | held (1 testable) |
| N1 | selected experts (L44E069, L41E117, L37E032, ...) | <= 50 of 768 | p6-circuit: pending | pending | – |
| N2 | selected experts (L20E000, L19E002, L18E001) | <= 200 of 14336 | p6-circuit: pending | pending | – |
| N3 | concept circuits | experts > heads | p6-circuit: pending | pending | – |
