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

## 2026-09-14 — extensions phase started (RESEARCH_PLAN.md)
- Four directions planned: (1) joint layer×expert search vs the paper's two-stage selection; (2) generalisation to
  Qwen3-Instruct-2507, Qwen3-Coder-Instruct, Mixtral-Instruct, OLMoE base/Instruct + attention-vs-MoE attribution
  (new spawn kinds attn_layer/block); (3) mechanism of the BOS effect on Mixtral routing (H1 sink relocation,
  H2 BOS semantics, H3 position shift, H4 default expert); (4) CodeFact: CounterFact-style code counterfactuals
  (S1-S3 syntax, R1-R3 recall) on Python.
- User decisions: core model set only; instruct models under both raw and chat-template protocols; Python,
  HumanEval+MBPP+The Stack, paper's absolute thresholds primary; two waves (1+3 first, then 2+4).
- Infrastructure: scripts/gpu_queue.sh (flock-serialised GPU jobs; all agents must use it). Report target:
  results/EXTENSIONS_REPORT.md. Run dirs: results/<model>_<bos|nobos>_<experiment>/ with run_meta.json.
- Wave 1 agents launching: ext1-joint-search, ext3-bos-mechanism, ext3-literature.

## 2026-09-14 00:59 — ext3-literature
- Wrote docs/ext3_literature_review.md (Direction 3 desk research, no GPU): attention sinks / massive activations and BOS
  removal, first-token effects on MoE routing, ROME/MEMIT/EasyEdit tokenisation conventions, patching-robustness tooling
  (TransformerLens prepend_bos, PR #1773), Mistral/Mixtral BOS documentation; H1-H4 evidence table; 9 cheap engine
  experiments; unverified items listed. 30 references, all opened this session.
- Decision-relevant findings: (1) Peng et al. 2026 (arXiv 2603.06591): for Mistral-7B "[BOS] is the sole driver of the
  sink", unlike Llama where a position-0 sink re-forms without BOS; Oh et al. 2025 (arXiv 2410.01866): Mistral/Mixtral
  without BOS put massive activations on the first delimiter, and BOS at position 0 takes them over; Sun et al. 2024 App.
  A.3: Mixtral massive activations partly shift to <s> when BOS is prepended. (2) RoPE is relative (RoFormer), so H3
  cannot act alone; the shifted-position control should equal the no-BOS run up to bf16 noise. (3) ROME/MEMIT/EasyEdit
  rely on tokenizer defaults (no BOS for GPT-2/GPT-J, BOS for Llama/Mistral); the paper's Appendix A never mentions
  special tokens, consistent with a Qwen3-first pipeline that disables them. Literature favours H1 in a Mistral-specific
  form entangled with H2; H4 has only indirect support.

## 2026-09-14 01:10 UTC — ext3-bos-mechanism: engine diagnostics built and verified; GPU chain launched
- Engine (backward compatible, all behind new optional arguments): `PrefillSpec.pos_offset` (RoPE position of the first
  token), `PrefillSpec.sink_donor/sink_vscale` (an extra attendable key/value slot copied from another row's position 0,
  value optionally scaled to 0 = key-only absorber), `DiagSpec` passed to `Engine.run(diag=...)` recording per layer the
  final position's attention distribution per head, every position's residual norm, final-position router logits and
  residual, next-token log-probs, and (optionally) the routing of every token at chosen layers; `moe_forward(...,
  return_logits)`; row-chunked prefill attention (memory bound, identical numerics); chunked `_build_v`.
  `prepare_case(..., prefix_ids)` / `cases_by_id(..., prefix_ids)` prepend arbitrary tokens after object resolution.
- Verified on OLMoE vs transformers 5.16 (results/verify_ext3_diag_olmoe.json): attention probs mean max-diff 0.014,
  hidden norms (hidden_states[l+1] = output of layer l) <= 1.5 % rel., router logits mean max-diff 0.025, all-token
  routing 143/150, token log-probs max-diff 0.17, pos_offset vs HF position_ids max 0.19 (HF's own Delta moves 0.10 on
  average under a +1 shift = bf16 floor). Pilot check re-run and gate (scripts/ext3_gate.py) precede every experiment pass.
- Design changes after the literature review (docs/ext3_literature_review.md): H3 is degenerate (RoPE is relative), so
  `shift1` is only an identity check; the literal "sink transplant" (BOS K/V at every layer) equals the BOS run by
  construction and is used as an identity check of the mechanism (`sinkfull`), the informative variant is the key-only
  sink (`sinkkey`, value = 0); added ',' at position 0; diagnostics record every position for the sink-location analysis.
- Chain scripts/ext3_chain.sh (logs/ext3_chain.log): verify -> gate -> Mixtral passes A (bos, nobos, shift1, eos, bosbos),
  B (nl, dot, comma, the, rare), C (sinkfull, sinkkey) -> Qwen3 (default, eot) -> corpus wiki, code. 8 GPU jobs.

## 2026-09-14 01:12 UTC — ext3-bos-mechanism: gate passed, experiment passes running
- Pilot verification re-run after the engine changes (results/verify_olmoe.json vs verify_olmoe_before_ext3.json): every
  metric identical to the backup (all diffs 0.0000, routing 38/40, rescue curve max |diff| 0.0000): the default engine
  path is byte-identical. Sink-transplant identity check on OLMoE: transplanted-K/V rows reproduce the donor rows
  exactly (Delta max diff 0.0, routing 100 %, layer-patch rescues identical, sink-slot mass identical); shift-by-one
  rows differ from the unshifted ones by <= 0.36 in Delta (bf16 noise; HF itself moves 0.10 on average).
- GPU chain continues: Mixtral pass A started 01:08:54 (5 variants, 2,560 prefill rows, 55,040 spawn rows).

## 2026-09-14 01:15 UTC — ext3-bos-mechanism: Mixtral pass A done (bos, nobos, shift1, eos, bosbos; 88 s)
- bos reproduces the main run (L19 E002 77/84, E006 71/67, strict 233/256), nobos reproduces mixtral_nobos (E006 91/84,
  E002 59/70, strict 249/256). shift1 = nobos to bf16 noise (E006 90/83): RoPE relativity confirmed numerically.
- Final position's attention on position 0 at layer 1: 0.83 with <s>, 0.09 without any token (no position-0 sink
  re-forms in Mixtral without BOS), 0.19 with </s>, 0.45 with <s><s>. </s> at position 0 is destructive (strict pass
  154/256, L* moves to L21, E006 95/86) - not a BOS substitute. <s><s> behaves like <s> (E002 78/86, E006 67/66).

## 2026-09-14 01:20 UTC — ext3-bos-mechanism: Mixtral pass B done (nl, dot, comma, the, rare; 106 s)
- Routing follows the sink, not the token: '\n' (final-position attention on pos 0 at L1: 0.77) and ',' (0.81) reproduce
  the BOS run (E002 73-74/128 disc, E006 68-77, L19 selected, strict 237-239/256); '▁.' (0.08), '▁the' (0.15),
  '▁workspace' (0.06) behave like no token (E006 88-92/128, val argmax L21). H2 (BOS-specific semantics) rejected in its
  strict form; sink formation is token-selective (Mistral: <s> and some delimiters), consistent with Oh et al. 2025.
- Queued one extra batched pass (chain2): attached '.' (28723), ':', '▁of', lone '▁', <unk>.

## 2026-09-14 01:25 UTC — ext3-bos-mechanism: Mixtral pass C done (sink transplant)
- sinkfull (no token; <s> key+value from the BOS run as an extra attendable slot at every layer) = BOS run: L19 routing
  agreement 0.99, E002 77/84, E006 71/68, E002 selected with spec +0.189 - the whole BOS effect is carried by position
  0's K/V (as the causal-attention argument predicts). sinkkey (key only, value 0): attention parks on the slot
  (~0.75 mass) but the model breaks (strict 172/256, Delta_clean +2.9, agreement 0.37 with BOS / 0.29 with no-BOS):
  the sink is an absorber AND a constant value bias; pure mass absorption is not the mechanism.
- Strata (passes A+B): prompts whose final position carries the maximal residual norm without BOS (63/256) have L19
  routing agreement 0.06 (0.75 elsewhere), E006 active 31 -> 63, E002 44 -> 15; subject-at-position-0 prompts 0.51
  vs 0.88 when the subject comes later; E006's contribution norm at L19 is 61 without BOS vs 24 with BOS.

## 2026-09-14 01:18 UTC — ext1-joint-search
- GPU (all through scripts/gpu_queue.sh): expert pass at EVERY layer, `--no-pairs`, layer-chunked to bound wavefront rows (one Qwen3 pass
  at 86k rows used 16 GB, so 4 chunks per run): `results/qwen3_bos_alllayers` (48 layers, 350,056 rows, 4 x ~50 s), `results/mixtral_bos_alllayers`
  (32 layers, 91,142 rows, 4 x ~80 s), `results/mixtral_nobos_alllayers` (paper set, `--no-special-tokens`, 46,560 rows, 4 x ~78 s); ~14 min GPU.
  `scripts/run_expert.py` gained `--layer-chunks N` and `--dry-run`; per-chunk parquet merge; `run_meta.json` in each run dir.
- Analysis (`moetrace/ext1_analysis.py`, `scripts/ext1_analyze.py`): per-layer recurrence-first best expert (curves + Spec), joint argmax over
  all recurrent (layer, expert) pairs on discovery evaluated on validation, concentration Rescue(e*)/Rescue(block), Appendix-D grid for the joint
  search, exhaustive scan of L42/43/45 (Qwen3) and L20/21 (Mixtral), paired winner-vs-two-stage tests. Fast path verified identical to
  analysis.evaluate_expert. bf16 fingerprints vs the base passes: L44E069 +0.499 vs +0.503, L19E002 +0.363 vs +0.352, L19E006 +0.063 vs +0.073.
- Qwen3 (paper set): joint winner = two-stage winner L44E069 (val +0.499 [+0.357, +0.659], Spec +0.443). BUT L42E115 (active 126/128 disc,
  123/128 val) is a second near-equivalent expert: val +0.447 [+0.363, +0.537], Spec +0.423 [+0.339, +0.510]; joint discovery argmax on the
  strict and relaxed sets and in 30/75 grid cells (5/10/15); L42 concentrates 72% of its block rescue in E115 vs 53% at L44; per-case
  correlation with E069 r = 0.27, union rescues 89% of val cases (per-case max +0.75 vs L44 block +0.94). No expert of L43/L45 comes close.
- Mixtral BOS: joint = two-stage L19E002 in all sets, 17/25 grid cells; the 8 others are threshold 80/96 cells where L19 has no recurrent
  expert at all (two-stage returns nothing) and the joint search falls back to L21E001 (+0.27) / L18E001 (+0.24).
- Mixtral no-BOS (paper protocol): joint winner L18E001 (rank 1 disc.; val +0.139 [+0.081, +0.205], Spec +0.098 [+0.040, +0.162]) vs two-stage
  L19E006 (rank 4; +0.063 [-0.009, +0.134], Spec -0.159). Paired: rescue +0.077 (p = 0.08), Spec +0.257 (p < 1e-4). L19E006 never wins in
  25 grid cells (winners L21E001 x12 at thresholds 32-48, L22E001 x8, L18E001 x2); the per-split two-stage layer flips to L21 in 4/5 seeds
  where nothing is recurrent at >= 64. L21E001 has the best validation rescue of anything evaluated (+0.29) but is active in only 59/128.
  E001 is the strongest expert of L17/L18/L21/L22 under BOTH protocols (not a BOS artefact); L19E002 is the strongest L19 expert even
  without BOS (+0.22) but fails recurrence there.
- Outputs: results/sections/ext1_joint_search.md, results/ext1_summary.json, results/figures/ext1_curves_{qwen3_bos,mixtral_bos,mixtral_nobos}.{png,pdf},
  results/tables/ext1_{top10,stability,neighbours,concentration,best_expert_by_layer}_<run>.{md,csv}, ext1_layer_curve_*.csv,
  ext1_all_candidates_*.csv, ext1_neighbours_all_*.csv, ext1_mixtral_E001_by_layer.{md,csv}, ext1_summary.{md,csv}.

## 2026-09-14 05:15 UTC — ext3-bos-mechanism: all passes done, section written (agent resumed after the usage-limit reset)
- Passes D (dot2, colon, of, space, unk), Qwen3 (default, eot) and the two corpus passes (wiki, code; 400 x 127 tokens
  x 2 protocols each) finished 01:15-01:22 UTC (10 ext3 GPU jobs in total, ~16 GPU minutes). scripts/ext3_analyze.py
  rebuilt every table/figure (results/tables/ext3_*, results/figures/ext3_*) and results/sections/ext3_bos_mechanism.md
  (verdict in results/sections/ext3_conclusion.md, appended by the script).
- Verdict: H1 in a Mistral-specific form. <s>'s only channel is its prompt-independent per-layer K/V (sinkfull = BOS run,
  agreement 0.99); Mixtral forms no position-0 sink without it; the massive-norm state forms on the first delimiter /
  function word and, in 63/256 prompts (58 without any delimiter), on the FINAL preposition of the cloze. Those 63 are
  the whole L19 effect: their final residual is the sink state (router logits identical to <s>'s, sd 0.008), routed to
  E006 in 63/63 (31/63 with BOS; agreement 0.06 vs 0.75 elsewhere). E006 is the L19 sink expert (P(E006|<s>) = 1.00,
  P(E006|1st content token) = 0.67 vs base 0.25) and otherwise an ordinary expert (corpus usage 0.24-0.25, position
  entropy 0.997, protocol Delta < 0.01). H2 rejected in strict form: '\n', ',', ':', attached '.', lone space and 'of' at
  position 0 restore BOS-run routing (agreement 0.76-0.80, E002 selected, spec +0.09..+0.15); '▁.', 'the', 'workspace'
  do not; </s> and <unk> are destructive. Key-only sink (value 0) breaks the model: absorber AND value bias both needed.
  H3 degenerate (RoPE relative; shift1 = nobos to bf16 noise). Qwen3: L44E069 survives <|endoftext|> (rescue +0.46,
  spec +0.40; no position-0 sink in Qwen3 either way). OOD: prompts 0.16 nats/token less likely without BOS, weakly
  related to the routing change (AUC 0.585); norm ratio of the final position predicts it (r = 0.59).
- Engine changes verified: verify_olmoe.json identical to verify_olmoe_before_ext3.json on every metric (gate output in
  logs/ext3_chain.log). Files owned: moetrace/engine.py, data.py, protocol.py, moetrace/ext3_variants.py,
  scripts/ext3_{run_variants,corpus_routing,analyze,verify_diag,gate,chain,chain2}.{py,sh}. Not committed (coordinator).

## 2026-09-14 05:15 — coordinator: wave 1 complete (Directions 1 and 3)
- ext1-joint-search, ext3-bos-mechanism, ext3-literature all delivered; sections in results/sections/, assembled into
  results/EXTENSIONS_REPORT.md by scripts/build_extensions_report.py. Key findings recorded in CLAUDE.md section 2b.
- ext3-bos-mechanism was killed once by the account usage limit (01:2x UTC, reset 05:00) after all its GPU passes had
  finished; resumed at 05:02 from disk, no GPU work repeated. Engine diagnostics verified (verify_olmoe identical).
- Wave-1 GPU total: ext1 ~14 min, ext3 ~16 min.
- Next: wave 2 (ext2-model-zoo, ext2-attn-patch, ext4-codefact-data) with two brief changes from wave 1: model-zoo
  runs the all-layer expert pass (joint search) as standard, and every run reports the fraction of prompts whose
  final token carries the sink state (final-position norm ratio / attention-on-self) as a protocol diagnostic.

## 2026-09-14 05:26 UTC — ext2-model-zoo
- Setup: registry entries for `olmoe_instruct`, `qwen3_instruct`, `qwen3_coder`, `mixtral_instruct` in moetrace/models.py (+ family/base/instruct
  fields on all models); `moetrace/ext2_zoo.py` (protocols default/nobos/chat, chat-prefix rendering, run_meta, sink diagnostic, pattern label);
  scripts `ext2_zoo_usage.py` (data/model_usage/<key>.json from tokenizer_config/generation_config/config/chat template/model card + empirical
  tokenizer probe), `ext2_zoo_filter.py` (run_filter with --protocol/--out, prefix_ids, base model's paper set), `ext2_zoo_sweep.py` /
  `ext2_zoo_expert.py` (wrappers around run_sweep/run_expert: prefix_ids bound into cases_by_id, DiagSpec(attn_final, resid_norms) ->
  sink_diag.json, automatic layer chunking at <= 90k rows), `ext2_zoo_chain.sh` (filter -> sweep -> all-layer expert per protocol via gpu_queue),
  `ext2_zoo_download.sh` (sequential downloads with a 15 GB floor).
- Downloads (NVMe, HF cache): OLMoE-Instruct 13.8 GB (34 s), Qwen3-Instruct-2507 61 GB (127 s), Qwen3-Coder 61 GB (132 s) done; Mixtral-Instruct
  93 GB in progress. Model cards (README.md) fetched for all 7 repos. Free space after all four: ~42 GB (the 65 GB `/opt/dlami/nvme/offload`
  folder from hf_reference_check and 11 GB of `.incomplete` blobs in the OLMoE base cache are not mine and were left alone).
- Usage specs: OLMoE base/Instruct and Qwen3 family add no special tokens (`default` == `nobos`, run once). OLMoE-Instruct quirk: its tokenizer
  names id 50279 `|||IP_ADDRESS|||` (bos = eos) while the model card writes `<|endoftext|>`; the rendered template starts with id 50279 either
  way. Qwen3-Instruct-2507 / Coder: non-thinking templates, prefix `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` (19 tokens).
- OLMoE-1B-7B-0125 (base), `results/olmoe_default`: filter 1 pass (1249 records scanned, 1024 tokenizable, strict pass rate 0.74), sweep
  (512 cases, L*=13 strict / L*=12 relaxed, val +1.42 / +1.29), all-layer expert pass (16 layers, 114,660 rows, 2 chunks); sink diagnostic:
  position 0 is the max-norm position in 100% of prompts from layer 2, the final position never (0/512). GPU total ~1.5 min.
- Chains for qwen3_instruct, qwen3_coder (default + chat) and olmoe_instruct launched behind the GPU queue.

## 2026-09-14 05:30 UTC — ext2-attn-patch: engine extension verified, sweeps queued
- Engine (`moetrace/engine.py`, backward compatible): new SpawnSpec kinds `attn_layer` (spawn BEFORE the MoE of layer l as
  h_pre_noised + Attn_l^clean at the final position; that layer's MoE then recomputes on the patched residual), `block`
  ((h_pre_noised + Attn^clean) + MoE^clean, both sublayer outputs patched together), `resid` (clean residual after layer l,
  classic hidden-state restoration = block + upstream difference) and `block_diff` (numerics check: both sublayer
  differences added to the noised residual). `DiagSpec.attn_out_final` records the final-position attention-sublayer
  output per layer (bf16 [L, B, H]). `PassResult.sp_vnorm` carries |dAttn|, |dAttn + dMoE|, |dResid| for the new kinds.
- Identity gate: `results/verify_olmoe.json` after the change is bit-identical to `results/verify_olmoe_before_ext2.json`
  on all 23 non-timing metrics (only the cold-page-cache load-wait differs).
- New-kinds verification vs transformers hooks on OLMoE (20 cases x 16 layers, `scripts/ext2_attn_verify.py`,
  `results/verify_ext2_attn_olmoe.json`): mean |dDelta| vs HF 0.187 / 0.180 / 0.131 (attn_layer / block / resid) against
  0.150 for the already-verified `layer` kind on the same rows (max 1.61 / 1.19 / 0.95 vs 1.12); rescue correlation with
  HF 0.990 / 0.994 / 0.998 (layer 0.983); 20-case mean curves within 0.081 / 0.072 / 0.069 (layer 0.099). Invariants:
  identity on the clean run max 0.094 (mean 0.015); block vs block_diff mean |d| 0.042, 90% within 0.1, max 0.56
  (bf16 rounding of the residual; signed mean on the OLMoE smoke sweep ~0); vnorms match HF norms within 0.3% (means).
- OLMoE smoke sweep `results/olmoe_attnsweep` (strict set 16/16, 5 kinds, 8.5 s): attention output rescue +2.9 at L12
  vs MoE +1.4 at L13, block +4.0 at L12, block vs attn+moe gap -0.21 (r 0.97). Analysis code `moetrace/ext2_attn.py`,
  `scripts/ext2_attn_analyze.py` tested on it (figures/tables/section generated).
- GPU chain `scripts/ext2_attn_chain.sh` (gate -> Qwen3 tokenizer defaults -> Mixtral BOS -> Mixtral no-BOS, paper set,
  4 kinds x every layer, one pass each) launched through gpu_queue.sh; waits behind the model-zoo agent's jobs.

## 2026-09-14 05:36 UTC — ext4-codefact: Phase A (dataset) done
- Sources: HumanEval (164, MIT) + MBPP full (974, CC-BY-4.0) + CodeSearchNet Python test split (seed-0 sample of 6,000 functions <= 1,500 chars,
  ASCII; +16,000 validation-split functions used for R3 only). The Stack is gated on the Hub (no token here) -> CodeSearchNet fallback as
  documented; per-function repo + URL recorded (data/codefact/csn_sample_repos.csv). Docstrings removed, CRLF normalised, units must parse.
- Builder scripts/ext4_build_codefact.py (ast/tokenize extractors for S1 closer, S2 block keyword, S3 for-in / if-colon / from-import,
  R1 variable recall (name bound earlier, <= 2 prior occurrences, foil = most recently bound other name), R2 attribute of a literal-typed local
  or imported module (foil = same-type attribute), R3 str / single-digit literal reuse). Case construction moetrace/ext4_data.py: true and foil
  must be single-token continuations of the prefix; boundary back-off <= 4 punctuation/whitespace chars when the tokenizer merges (Qwen
  ` else`, `.append`); copy exclusion (true token in last 3 prefix tokens); prefix <= 160 tokens. CodeCase subclasses data.Case (category,
  source, item_id) so engine/noise/analysis code is reused unchanged.
- data/codefact/items.jsonl: 6,795 items; Qwen3-valid per category S1 964, S2 1200, S3 1195, R1 1132, R2 794, R3 1163; Mixtral-valid
  S1 1127, S2 1070, S3 1124, R1 955, R2 671, R3 824 (build_stats.md has raw -> capped -> written yields and reject reasons; samples.md 20
  random items per category). Qwen3-Coder tokenizer identical to Qwen3-Base on the items (2,000/2,000 identical ids); Coder snapshot complete.
- GPU scripts written (not yet run): scripts/ext4_scan.py (clean + noised + every-layer block patch + routing + resid-norm sink diagnostic per
  chunk of <= 1,024 items = filter and sweep in one job), ext4_select.py (case sets 256 -> 128/128 per category + mixed `all`, calibration
  tables/figure, sweep_* files for analysis.load_model), ext4_run_expert.py (all-layer expert pass, --no-pairs, layer-chunked; reuses
  run_expert.build_spawns/rows_from_result/merge_rows), ext4_analyze.py (per-category two-stage + joint search via ext1_analysis, Jaccard
  overlaps, factual-expert check, figures, section). Smoke test queued as ext4-smoke behind the ext2 chains.

## 2026-09-14 05:52 UTC — ext2-model-zoo: raw-protocol runs of the Qwen3 family and OLMoE-Instruct done
- `results/qwen3_instruct_default` (Qwen3-30B-A3B-Instruct-2507, tokenizer defaults = no special tokens): filter 1 pass (strict pass rate 0.82;
  234/256 of the base's paper IDs pass strict, exactly as for the base), sweep L*=44 on paper/strict/relaxed (paper val +1.065 vs base +0.941),
  all-layer expert pass (351,304 rows, 4 chunks, 18.5 GB peak). Two-stage and joint search both select **L44E069** (active 110/128 disc, 115/128 val;
  rescue +0.549 [+0.422, +0.690], Spec +0.464 [+0.333, +0.609]); L42E115 remains the second locus (124/123 active, +0.520 / Spec +0.490). Pattern A.
- `results/qwen3_coder_default` (Qwen3-Coder-30B-A3B-Instruct, raw): 218/256 paper IDs pass strict; L*=44 (paper val +0.962); **L44E069** selected
  (111/110 active, rescue +0.615 [+0.479, +0.763], Spec +0.551 [+0.413, +0.699]); L42E115 118/113 active (+0.514 / +0.490). Pattern A. Expert pass
  369,040 rows in 5 chunks (row cap lowered to 80k after the 18.5 GB reading).
- Final-position top-8 routing agreement with the base at L44 (paper prompts): Jaccard 0.80 (Instruct), 0.71 (Coder); identical sets in 27% / 13%.
- `results/olmoe_instruct_default`: filter (strict pass rate 0.72), sweep L*=13 (strict val +1.18), all-layer expert pass done. `results/olmoe_default_attnsweep`
  (kinds attn_layer, layer, block; strict set): L13 attention-output rescue +2.93, MoE-output +1.42, whole-layer +3.87 (val).
- Sink diagnostic: final position is never the max-norm position in any Qwen3/OLMoE run so far (0/512-1024 prompts); position 0 is the sink in 100%.
- Chat-protocol scans running (Qwen3-Instruct chat: T=37, strict pass rate 0.84, mean Delta_clean +7.6 vs +6.1 raw, 236/256 paper IDs pass strict).
  Mixtral-Instruct BOS filter done (236/256 paper IDs pass strict). Analysis driver `scripts/ext2_zoo_analyze.py` written; incremental tables in
  results/tables/ext2_zoo_*.md, section draft results/sections/ext2_model_zoo.md.

## 2026-09-14 05:55 UTC — ext2-attn-patch: sweeps done, analysis and section written
- GPU (gpu_queue.sh, chain scripts/ext2_attn_chain.sh after the gate passed): `results/qwen3_bos_attnsweep` (paper set, 256 cases,
  4 kinds x 48 layers = 49,152 rows, 61 s), `results/mixtral_bos_attnsweep` (32,768 rows, 98 s), `results/mixtral_nobos_attnsweep`
  (`--no-special-tokens`, 32,768 rows, 98 s); ~4.5 min GPU plus ~5 min of OLMoE verification/smoke runs. run_meta.json in each.
- Analysis `scripts/ext2_attn_analyze.py` -> results/tables/ext2_attn_{peaks,share,additivity,layer_curves}_<run>.*,
  ext2_attn_{summary,shares,protocol_mixtral}.*, results/figures/ext2_attn_curves_{qwen3_bos,mixtral_bos,mixtral_nobos,olmoe,
  mixtral_bos_vs_nobos}.{png,pdf}, results/ext2_attn_summary.json, results/sections/ext2_attn_patch.md (+ ext2_attn_interpretation.md).
- Qwen3 (validation, paper set): attention-output rescue peaks L40 +1.594 [+1.410, +1.791] (96% of cases positive) and L43 +1.10;
  MoE-output L44 +0.925 [+0.774, +1.096] (paper's layer), L42 +0.62; block L40 +1.946 [+1.734, +2.169]. Attention share at L44 = 2%
  [-2%, 6%] (pure MoE layer), overall (AUC+) 51% [48%, 55%]. Additivity: block = attn + moe to bf16 noise (per-case r 0.96-0.99),
  slight sub-additivity only at L40 (-0.08 [-0.12, -0.03]) and L43 (-0.10). Hidden-state restoration: half of the drop present by
  L33, largest increments +1.48 at L40, +0.36 at L43.
- Mixtral BOS: attention L18 +0.988 [+0.820, +1.164], L19 +0.92, L24 +0.84, L15 +0.44; MoE L19 +0.561 [+0.451, +0.683], L20/21
  +0.49/+0.48; block L19 +1.374 [+1.180, +1.577]. Attention share at L19 = 62% [58%, 66%]; overall 56%. Gap at L19 -0.11 [-0.18, -0.05],
  r 0.97. No BOS: same curves (layer correlation 0.96-0.98), block argmax L19 +1.18, attention argmax L24 +0.93 (L18 +0.80 within CI),
  MoE argmax L21 +0.53 (L19 +0.46 within CI); attention share at L19 64%, overall 60% [54%, 65%]. BOS run stronger at the L18
  attention step by +0.19 [+0.02, +0.36] (paired).
- Reading: information arrives at the final position in a few attention steps (Qwen3 L40/L43, Mixtral L15/L18/L19/L24) and is
  transformed by the MoE of the same and following layers; the paper's MoE-only patch sees the transformation channel only
  (right object for Qwen3 L44, minority share of the layer effect at Mixtral L19; misses the largest single-sublayer locus,
  Qwen3 L40 attention). Suggested follow-up: per-head attn_layer patch at those layers.
- Deviations: added `resid` (clean residual after layer l, the coordinator's "= h_clean_after_layer_l" reading) as a 4th kind next to
  `block` (both sublayer outputs of layer l, the reading consistent with the additivity invariant), and `block_diff` as a permanent
  numerics-check kind; gate threshold for block vs block_diff max |d| set to 0.7 (same as the HF maxdiff floors) after the first
  gate run failed at 0.557 (mean |d| 0.042, signed mean +0.002: bf16 rounding, no bias). Not committed (coordinator).
