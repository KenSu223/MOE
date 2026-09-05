# Progress log (append-only, UTC timestamps)

## 2026-09-03 02:25 — handoff state
- Plan: PLAN.md. Operational brief for the unattended agent: HANDOFF.md.
- Environment ready: /home/ubuntu/MOE/.venv (torch 2.14 CUDA verified, transformers 5.16.1, safetensors 0.8, pandas, pyarrow, scipy, matplotlib).
- Checkpoints fully downloaded to /opt/dlami/nvme/hf (HF_HOME): Qwen3-30B-A3B-Base (61 GB), Mixtral-8x7B-v0.1 (93 GB), OLMoE-1B-7B-0125 (fp32 shards).
- Data ready: data/counterfact.json, data/paper_case_ids.json.
- Code: nothing under moetrace/ yet. Next milestone: moetrace/weights.py + engine.py verified against transformers on OLMoE.
- Supervisor for the headless agent: scripts/run_agent.sh (launch from tmux; see HANDOFF.md section 0). The interactive VSCode
  session could not launch it itself (blocked by the permission classifier), so the user launches it manually.

## 2026-09-03 02:30 — headless agent running (run 1 started 02:24:30 UTC)
- Note for the agent: moetrace/__init__.py, arch.py, weights.py (timestamps 02:04) were written by a parallel interactive
  VSCode session before you started, not by you. They are unverified drafts: review them and keep, fix or replace as you see fit.
- The interactive sessions will stop editing code from now on and only monitor logs.
- 02:32 user update: the agent MAY use the internet freely (docs, transformers source, ROME reference code, web search).
  HANDOFF.md section 6 has been updated accordingly.

## 2026-09-03 02:40 — headless agent started (run 1, session 353b3fa2)
- Read HANDOFF.md, PLAN.md, TeX source (method, all appendix tables), HF modeling files for qwen3_moe/mixtral/olmoe (v5.16.1),
  model configs and safetensors key layouts. All match HANDOFF section 1.
- Found existing moetrace/arch.py and moetrace/weights.py (layer streamer with pinned double-buffer prefetch). Keeping them.
- Numerics decisions (from HF 5.16.1 source): RMSNorm fp32 -> cast -> *weight; RoPE cos/sin fp32 -> bf16; router logits bf16,
  softmax fp32, top-k, renorm if norm_topk; Qwen3/OLMoE cast routing weights to bf16, Mixtral keeps fp32 (5.16 behaviour);
  per-expert down-proj output (bf16) * weight -> fp32 contribution c_e; block output = fp32 sum of c_e cast to bf16; residual
  stream bf16. Delta for ALL rows computed via the same bf16 bmm on the (true, foil) lm_head rows so that null interventions
  give exactly zero rescue; full-vocab logits recorded for prefill rows as a cross-check and for the top-1 token.
- Next: moetrace/engine.py (two row types, wavefront spawns), verify vs HF on OLMoE.

## 2026-09-03 02:45 — engine written and verified on OLMoE (pilot)
- Code: moetrace/engine.py (streaming executor, prefill + wavefront rows, in-pass intervention vectors for kinds
  zero/layer/expert/expert_scaled/coalition_clean/coalition_union), data.py, noise.py, stats.py, models.py, verify.py;
  scripts/verify_olmoe.py, verify_olmoe_fp32.py, run_filter.py.
- Verification vs transformers 5.16.1 on OLMoE (results/verify_olmoe.json, 50 cases): embed std identical; routing sets
  agree 38/40 (2 flips at near-tied experts); top-1 agreement 98%; Delta_clean max |diff| 0.38 (72% within 0.1);
  Delta_noised max 0.61; layer patches vs HF hooks max 0.58 (58% within 0.1); union coalition == layer patch exactly;
  bmm-vs-full-matmul logits identical (100%).
- These differences looked large, so a noise-floor test was run (results/verify_olmoe_fp32.json, 12 cases, HF fp32 on
  CPU as ground truth): engine-bf16 error vs fp32 (clean mean 0.057/max 0.11; noised 0.146/0.55; L12 rescue 0.15/0.29)
  is the SAME size as HF-bf16's own error (eager: 0.041/0.11; 0.213/0.47; 0.175/0.62; sdpa similar), and HF eager vs
  HF sdpa differ by up to 0.47 on noised prompts. Conclusion: the engine is at the inherent bf16 noise floor; per-case
  rescue carries ~0.2-0.6 bf16 noise on noised prompts (means over 128 cases: ~0.02-0.05). Accepting.
- Zero-intervention wavefront rows reproduce prefill logits to within this same floor (not bit-exact: different matmul
  shapes -> different fp32 accumulation order -> occasional bf16 ulp flips). Documented as a tolerance, not a bug.
- Qwen3 per-layer load 0.65 s cold from NVMe (1.9 GB/s, disk-bound); no loader optimisation needed.
- OLMoE filter dry run (512 tokenizable records): 75% pass strict, 77% relaxed. Very high pass rates -> the paper's 256
  cases are probably the first ~350 tokenizable shuffled records if shuffles match. Checked when Qwen3 scan finishes.
- Launched: Qwen3 filter scan (logs/filter_qwen3.log). Next: sweep pass script.

## 2026-09-03 02:50 — Qwen3 filter scan + layer sweep done
- Filter scan (results/qwen3/filter_scan.parquet, 65 s, one chunk of 1,024 tokenizable records = first 1,048 shuffled
  records): 80% pass strict, 83% relaxed. Only 24 records rejected by the single-token rule.
- Funnel vs paper (case_sets.json -> paper_check): all 256 Table-8 IDs lie within the first 390 records of OUR seed-0
  shuffle (median rank 196) -> the paper's record order is random.Random(0).shuffle of the record list, same as ours.
  235/256 paper IDs pass our strict filter, 240 relaxed; overlap paper-vs-our-first-256-strict = 195. In the same
  390-record prefix, 66 non-paper records pass our strict filter and 21 paper records fail it (some with clean margin
  as low as -4.5, far beyond bf16 noise) -> the paper's Delta values differ from ours for a minority of cases
  (protocol difference, e.g. BOS handling or object tokenisation, not noise). Testing alternatives (BOS prepended;
  object token without leading space) in scripts/funnel_hypotheses.py; results reported either way. PRIMARY case set
  remains the paper's IDs as instructed.
- Layer sweep (results/qwen3/sweep_rows.parquet, 528 unique cases, 25,344 layer-patch rows, 62 s):
  paper set: L*=44 (discovery +0.973), validation +0.941 (paper +0.901 [0.752, 1.053]); next-best L42 +0.625 / L43 +0.620
  (paper L42 +0.592). Our strict 256: L*=44, val +0.921. Relaxed 512: L*=44, val +0.791 (paper +0.846).
- Launched: funnel hypotheses pass, then expert pass at L44 (all Qwen3 sets).

## 2026-09-03 02:58 — Qwen3 expert pass (L44) done; Mixtral chain launched
- Expert pass (results/qwen3/expert_rows.parquet, 528 cases, 36,847 spawn rows at L44, 56 s): expert patches for every
  clean-active expert (+ noised-only-active experts, tagged), all 56 ordered equal-norm pairs per case, coalitions, layer.
- Paper case set, recurrence >= 64/128: 5 candidates; L44E069 selected (all-case discovery mean +0.468; next E027 +0.029).
  Discovery active 114/128 (paper 112), validation active 116/128 (paper 116).
  Validation: expert rescue +0.503 [+0.362, +0.661] (paper +0.463 [+0.344, +0.590]); Spec +0.450 [+0.310, +0.608]
  (paper +0.400 [+0.276, +0.533]); active-random control mean +0.053. Sign-flip p < 1e-4.
  Gate-matched (n=116, paper 116): raw sel +0.555 / ctrl +0.054 / spec +0.501 (paper +0.513 / +0.054 / +0.459);
  equal-norm sel +0.247 / ctrl +0.046 / spec +0.202 (paper +0.239 / +0.051 / +0.188).
  All-active rank (n=116): top-1 61, top-2 82, mean rank 2.47, percentile 0.79, sel-minus-others +0.491 (paper 53/74/2.90/0.73/+0.441).
  Coalitions (extra for Qwen3): clean top-8 +0.916, union +0.932 (= layer patch +0.932 in the same pass; sweep-pass layer +0.941).
  Ratios: expert/layer 0.534 [0.428, 0.633] (paper 0.515), spec/layer 0.478 (paper 0.444).
  Relation-wise validation n's match the paper exactly (P103 12, P27 9, P136 7, ...), confirming the same split.
- Our strict 256 set gives the same selections (L44, E069, val rescue +0.475, spec +0.420).
- Funnel hypothesis round 1 (results/qwen3/funnel_hypotheses.json): BOS makes no difference. Object tokenised WITHOUT a
  leading space: on the 96 prefix cases where both no-space objects are single tokens, 74/75 paper cases pass strict vs
  2/21 non-paper -> the paper likely used tok(obj) (no leading space) for the object token. Round 2 (first-token variants
  over all 380 prefix cases) running in chain1. DEFAULT KEPT: leading-space continuation token (HANDOFF section 3);
  alternative reported.
- chain1 (logs/chain1.log): funnel round 2 -> Qwen3 noise pass (L44/E069, sigma 1,2,4) -> Mixtral filter -> sweep -> expert.

## 2026-09-03 02:56 — monitor note (interactive session, read-only observer)
- Qwen3 main results land inside the paper's CIs; good. Suggestion for the object-token finding: if funnel round 2
  confirms the paper used tok(obj) without a leading space, also run the Qwen3 sweep + L44 expert pass under that
  variant on the paper case set (2 passes, ~2 min) and report it as a "paper-tokenisation" column next to the default,
  since the target token id changes Delta itself, not just the funnel. Same for Mixtral if cheap. Default stays as is.

## 2026-09-03 03:05 — funnel round 2, Qwen3 noise pass, Mixtral filter done; report generator working
- Funnel round 2 (results/qwen3/funnel_hypotheses2.json, all 390 records of the paper prefix): agreement with paper
  membership (paper cases pass & non-paper fail): ours (leading-space continuation token) 0.774; first token of ' '+obj
  0.754; first token of obj 0.726; **no-space token if obj is a single token else leading-space token: 0.949**
  (252/256 paper pass, 16/134 non-paper pass, first-256 overlap 241). Conclusion: the paper most likely resolved object
  tokens with tok(obj) first and fell back to ' '+obj. Kept the HANDOFF default for all main results; added
  --token-rule paper_like to the pass scripts and queued a secondary run on the paper case set (results/*_alt) in chain2.
- Qwen3 noise pass (results/qwen3/noise_rows.parquet): Table 13 reproduces: sigma 1/2/3/4 -> rescue +0.095/+0.439/+0.503/+0.507
  (paper +0.098/+0.406/+0.459/+0.478), drop +1.36/+4.81/+5.62/+5.96 (paper +1.26/+4.91/+5.70/+6.00).
- Appendix D (post-processing): L44E069 selected in 25/25 (seed x threshold) settings, mean val rescue +0.491, spec +0.439
  (paper 25/25, +0.398, +0.344). Relation-held-out: E069 in 5/5 folds, active 230/256, rescue +0.485 [0.389, 0.593],
  spec +0.434 (paper 229/256, +0.443, +0.388).
- Mixtral filter (98 s pass): 2,034 records scanned, 1,010 rejected (32k vocab -> many multi-token objects), 86% strict
  pass; paper IDs within first 580 records; 233/256 pass our strict filter; overlap with our first-256 = 230.
- moetrace/analysis.py + report.py: Tables 1-16 (md + csv), Figure 1, results/summary.json; tested on Qwen3.
- Running: chain1 (Mixtral sweep -> expert). Queued: chain2 (paper_like token-rule runs; HF reference checks with offload).

## 2026-09-03 16:25 — interactive session took over (headless agent hit the account usage limit at 03:02 UTC)
- The agent's run 1 ended with "You've hit your session limit · resets 6:10am (UTC)"; runs 2-20 of the supervisor failed instantly for the same
  reason and the supervisor exited at 03:12. Everything up to the report generator was already done; chain1 finished (Mixtral sweep + expert);
  chain2 finished only the qwen3_alt sweep before the agent stopped it to add a no-BOS option (that edit never executed).
- Resumed at 16:07: implemented `--no-special-tokens` (data.py, protocol.py, run_sweep.py, run_expert.py); launched scripts/chain3.sh:
  Mixtral no-BOS sweep + expert (paper set), Mixtral paper_like token-rule sweep + expert, HF reference checks for both big models.
- KEY FINDING (results/mixtral_nobos, results/mixtral_compare.json): tokenising Mixtral WITHOUT BOS reproduces the paper exactly where the
  default run did not. L19 val rescue +0.446 (paper +0.457); 249/256 paper IDs pass the strict filter (233 with BOS); L19E006 is the sole
  recurrent candidate and is selected (paper E006), clean-active 91/128 disc and 83/128 val (paper 91/83, identical); E006 rescue +0.073
  [+0.007, +0.140] (paper +0.099), Spec -0.171 [-0.262, -0.082] (paper -0.175); active-pair equal-norm check and coalitions (+0.431/+0.454 vs
  +0.461/+0.490) inside the paper's CIs; validation top layer L21 then L19 as in the paper's Table 6. With BOS, routing shifts and E002 becomes
  recurrent (76/128) and specific (+0.205), which is why the default run selected E002.
- Report: scripts/build_final_report.py rebuilds everything and adds section 6b (no-BOS Mixtral) plus figures/fig1_mixtral_nobos.png.
  results/DONE is written once the HF reference checks finish.

## 2026-09-03 16:36 — COMPLETE
- HF reference checks (transformers 5.16.1, device_map=auto with CPU/disk offload, 5 validation prompts each): Qwen3 max |Delta diff| 0.25,
  mean 0.14, top-1 5/5; Mixtral max 0.0625, mean 0.0125, top-1 5/5. Within the bf16 noise floor measured on OLMoE.
- Final REPORT.md rebuilt (sections 1-8 + 6b Mixtral no-BOS); results/DONE written. All paper tables (1-16), Figure 1 (default and no-BOS
  Mixtral variants), row-level parquet for every pass, and the comparison JSONs are under results/.
