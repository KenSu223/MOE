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

## 2026-09-14 05:58 UTC — ext4-codefact: GPU incident and fix
- ext4-scan-codefact_qwen3_raw (chunk 1024 items, sweep rows for 48 layers) OOMed at chunk 4/7 (T=71) in engine.attn_wavefront: the broadcast
  matmul expands the gathered K over the head-repeat factor, so the operand is [wf_chunk, n_heads, D, T] = 4.5 GiB at wf_chunk 8192, T 71
  (fine for CounterFact's T ~ 20). The Mixtral scan that started next would have failed the same way and was stopped (my own job, 1 chunk lost).
- Fix without touching engine.py: pass the engine's existing `wf_chunk` argument from my scripts (wf_chunk = 160,000 // T, i.e. 1,000 rows at
  T = 160), bound item chunks by items x T <= 80,000 (so 1,024 items at T <= 78, ~500 at T = 160) and set PYTORCH_CUDA_ALLOC_CONF=expandable_segments.
  Same in ext4_run_expert.py. Both chains re-queued at 05:55 UTC (ext4-scan-codefact_qwen3_raw, ext4-scan-codefact_mixtral_nobos). If a chunk still
  fails, fallback = the coordinator's plan (prefill-only calibration scan; sweep only the <= 256 filtered items per category with chunk <= 256).
- Smoke test (48 items): scan 61 s / 3.6 GB, expert pass 2 x ~55 s / 3.9 GB, ext4_select + ext4_run_expert + ext4_analyze verified end to end.

## 2026-09-14 10:15 UTC — ext4-codefact: resumed after the usage-limit reset; scans + selection done, expert passes re-planned
- Scans (filter + every-layer block patch in one job): codefact_qwen3_raw 6,448 items in 8 length-sorted chunks (466 s GPU, peak 13.3 GB);
  codefact_mixtral_nobos 4,671 items (<= 800 per category, --no-special-tokens) in 7 chunks (635 s, peak 16.3 GB). Selection (ext4_select.py):
  Qwen3 all six categories reach 256 (pass counts S1 871/964, S2 399/1200, S3 430/1195, R1 896/1132, R2 451/794, R3 690/1163), union 1,536;
  Mixtral S3 partial (241 pass -> 120/121), the rest 256 (S1 690/800, S2 316, R1 379, R2 321, R3 318), union 1,521. Mixed `all` set = 256
  stratified (43/43/43/43/42/42). Calibration tables results/tables/ext4_calibration_*.md, figure results/figures/ext4_calibration.png.
- Calibration finding (Qwen3): S1 passes 90 % (median Δ_clean +13.6, drop +4.4: noising the opener DOES move the closer), S2 33 % and S3 36 %
  (median drop +0.25 / +0.12: block keywords and `in`/`:` are redundantly determined), R1 79 %, R2 57 %, R3 59 %. Relative rule (25 %): 62/15/20/50/24/29 %.
  Top-1 = true: S1 18 % exact but 87 % "starts with true" (the model prefers merged tokens like `):`), S3 38 % / 96 %, R1 89 %, R2 90 %.
  No prompt has its final token as the max-norm (sink) position at L5 in either model (0.0 %).
- Layer sweeps: on code the LAST MoE block dominates the curve in both models (Qwen3 L47 selected for S1/S2/S3/R2/R3/all, R1 -> L43; Mixtral
  L31 for all but S2 -> L17), whereas CounterFact's last layer rescues nothing (-0.13 / -0.08). Interior maxima sit in a shared band: Qwen3
  L41-43 (S1 +1.26 at L42, R1 +0.39 at L43), Mixtral L17-22 (the ext1 E001 band). Analysis extended with an "interior" selection (layers <= L-5).
- Expert passes OOMed at 06:31 (prefill rmsnorm on 3,072 rows x T=160, not the wavefront). Rewritten ext4_run_expert.py: cases sorted by length
  and grouped so that 2 x cases x T <= 160k (Qwen3) / 110k (Mixtral) row-tokens, layers grouped so that wavefront rows <= 45k / 24k, engine
  wf_chunk = 160k // T, resumable per-pass part files. Plans: Qwen3 2 case chunks x 23 passes (925k spawn rows), Mixtral 3 x 13 passes (256k).
  Chain scripts/ext4_chain2.sh (detached) runs: Qwen3 expert -> Mixtral expert -> Coder raw scan/select/expert -> Coder chat scan/select/expert.

## 2026-09-14 10:40 UTC — ext2-model-zoo: all runs done, section written (agent resumed 10:03 after the usage-limit reset; no GPU work redone)
- GPU: 10 run configurations x (filter 1 pass, sweep 1 pass, all-layer expert pass 2-5 chunks) + 5 attention/MoE/block sweeps = 46 ext2zoo jobs,
  all rc=0 by 06:57 UTC (queue log), about 55 GPU-minutes; every run dir carries run_meta.json (rendered chat prefix, protocol, chunking, row
  counts) and sink_diag.json; raw diagnostics on /opt/dlami/nvme/moe_ext2/<run>/sweep_diag.npz.
- Runs: olmoe_default; olmoe_instruct_{default,chat}; qwen3_instruct_{default,chat}; qwen3_coder_{default,chat}; mixtral_instruct_{default,nobos,chat};
  attention sweeps {olmoe_default,olmoe_instruct_chat,qwen3_instruct_chat,qwen3_coder_chat,mixtral_instruct_chat}_attnsweep (kinds attn_layer,
  layer, block on the intended case set) plus the ext2-attn-patch agent's qwen3_bos/mixtral_bos/mixtral_nobos sweeps used for the base rows.
- Analysis (`scripts/ext2_zoo_analyze.py`, cached per run in results/<run>/zoo_summary.json): two-stage selection, evaluate_expert, coalitions,
  all-active rank, Appendix-D grid, joint search + per-layer best-expert curves (ext1 functions), base experts as fixed hypotheses, routing
  agreement with the base runs, sink fractions (+ sink position/token), drop-normalised rescue shares, attention/MoE/block peaks and additivity.
  Outputs: results/tables/ext2_zoo_{summary,usage,sink,reference_experts,base_vs_instruct,routing_agreement,attn}.{md,csv}, ext2_zoo_run_<run>.*,
  ext2_zoo_layer_curve_<run>_<set>.csv; figures ext2_zoo_{curves,expert_curves,attn_curves}.{png,pdf}; results/ext2_zoo_summary.json;
  results/sections/ext2_model_zoo.md (assembled into results/EXTENSIONS_REPORT.md).
- Headline: every model is pattern A under its intended protocol (OLMoE L13E056 Spec +0.87; OLMoE-Instruct chat L12E040 / L13E056; Qwen3-Instruct
  chat L44E069 +0.72 / Spec +0.62; Coder chat L44E069 +0.66 / +0.59; Mixtral-Instruct chat L31E002 +1.19 / +0.71 with the L19-L21 band still
  present). Pattern B only for Mixtral base/Instruct without BOS (E006 91/83 active in both, Spec -0.16 / -0.22, joint L18E001, final-token sink
  in 24-25% of prompts); pattern C never. Post-training keeps L44E069 and L42E115 in Qwen3-Instruct and Coder (rescue/Spec equal or higher than
  the base) although the L44 top-8 set matches the base in only 27% / 13% of prompts; Mixtral-Instruct with BOS = base (L19E002, 86% identical
  L19 routing). Chat wrapping raises margins, drops and absolute rescues but not the drop-normalised share (Qwen3 15-17%) and does not move the
  Qwen3/OLMoE loci; for Mixtral-Instruct it moves the argmax to the final layer L31. Attention-output patches peak earlier and higher than
  MoE-output patches in every model; block = attn + MoE within 0.01-0.11.

## 2026-09-14 16:05 UTC — ext4-codefact: Phases B and C complete, section written
- GPU (all via scripts/gpu_queue.sh, 14 jobs, 134 min total = 128 min of successful work + 6 min of failed attempts): scans 4 x 7-10 min
  (qwen3_raw 466 s, mixtral_nobos 635 s, qwen3_coder_raw 473 s, qwen3_coder_chat 521 s), memory-bounded all-layer expert passes
  (qwen3_raw 23 passes / 1,416 s, mixtral_nobos 13 / 1,057 s, coder_raw 23 / 1,377 s, coder_chat 25 / 1,492 s; peak 13.4-13.5 GB), smoke 221 s.
  OOM fix (no engine change): the 06:31 failures were the prefill rmsnorm on 3,072 rows x T=160; ext4_run_expert.py now sorts cases by length
  and groups them so that 2 x cases x T <= 160k row-tokens (Qwen3) / 110k (Mixtral), groups layers so that wavefront rows <= 45k / 24k, passes
  the engine's wf_chunk = 160k // T, writes resumable per-pass part files (results/<run>/expert_parts/) and concatenates them into
  expert_rows.parquet. The earlier 05:53 scan OOM (attn_wavefront at wf_chunk 8192, T 71) was fixed the same way (wf_chunk from T, item
  chunks bounded by items x T <= 80k). Coder tokenizer identical to Base on the items; chat protocol = user turn "Complete the following
  Python code." + open assistant turn + ```python fence (17 tokens), analogous to ext2's convention.
- Analysis (scripts/ext4_analyze.py, moetrace.analysis + ext1_analysis APIs; per-run pickle cache): per category two-stage selection at the
  paper's L*, the same restricted to interior layers (<= L-5), joint search over all recurrent pairs, coalitions, rank among active experts,
  factual-expert check, Jaccard overlaps (recurrent sets, top-10 pairs, interior variants), Coder-vs-Base on shared items. 172 tables
  results/tables/ext4_*, figures ext4_calibration*.png, ext4_curves_{qwen3,mixtral,coder_raw,coder_chat}.png, ext4_overlap_*.png,
  results/ext4_summary.json, section results/sections/ext4_codefact.md (+ ext4_codefact_narrative.md, embedded).
- Findings: (1) paper filter pass rates (Qwen3) S1 90 %, S2 33 %, S3 36 %, R1 79 %, R2 57 %, R3 59 % (Mixtral 86/40/30/47/48/40): block
  keywords and `in`/`:` are redundantly determined (median drop +0.25/+0.12), closing brackets are read from the opener like a fact from its
  subject (drop +4.4). (2) On code the LAST MoE block is a read-out: L47/L31 selected for nearly every category (CounterFact: -0.13/-0.08 there);
  interior peaks in a shared band L41-43 (Qwen3) / L17-22 (Mixtral); final-layer experts follow the final token's class (identifier vs
  indentation vs quote), so the mixed set has no recurrent expert at L47/L31 in any run. (3) S1 localises like a fact in both models
  (Qwen3 L47E025 rescue +1.15, Spec +0.97; Mixtral L31E000 +1.66 / +1.44, active 128/128; Qwen3 interior L42E048 +0.92 / +0.87); R1 in Qwen3 is
  the CounterFact-like case (L43E126 +0.33 / +0.32, 83 % of the block, rank-1 in 87/118); S2 Qwen3 interior L41E041 +0.55 / +0.55, Mixtral
  L17E003 +0.36 / +0.32; S3, R2, R3 weak (Spec <= 0.26) except Mixtral R2 L31E005 (Spec +0.50 on 70/128 active); Mixtral R1 -> L31E006 with
  Spec -0.24 (E006 negatively specific again, coalition +0.30). (4) Top-10 joint pairs almost disjoint across categories (mean Jaccard 0.05 /
  0.04); recurrent-set overlap 0.08-0.47 through always-on experts that rescue nothing; S1 is as close to R1/R2 as to S3 -> the S/R split is
  not the axis, single-token vs distributed determination is. (5) L44E069 / L42E115 clean-active in <= 4 / <= 17 of 128 and rescue nothing on
  every code category; L18E001 inactive; L19E002 / L19E006 recurrent only on Mixtral S2 (E002 +0.32, E006 Spec -0.27). (6) Coder-Instruct:
  same items pass (r(Δ_clean) 0.83-0.92), same experts for S1 (L47E025 in all three runs), R1 (L43E126) and R2/R3 (chat), S2's joint winner
  L41E041 in all runs; Coder adds a strong S3 expert L47E014 (+0.77/+0.99, Spec +0.70/+0.78); chat template raises block rescue, not experts.
- Storage note: results/codefact_* are 19-83 MB each, dominated by scan_routing.parquet (34 MB per Qwen3 run: routing of ALL 6,448 scanned
  items at every layer); it is only an intermediate for ext4_select.py (sweep_routing.parquet, 8 MB, is the subset the expert pass and the
  analysis use) and expert_rows.parquet (16-17 MB). scan_routing can be dropped from the commit; scan_cases/scan_rows (2.7 + 1.4 MB) hold
  every calibration statistic. expert_parts/ duplicates expert_rows.parquet (resume files) and can be deleted.

## 2026-09-14 16:20 — coordinator: wave 2 complete (Directions 2, 2b, 4); all four extension directions delivered
- ext2-attn-patch, ext2-model-zoo, ext4-codefact all delivered sections; assembled into results/EXTENSIONS_REPORT.md.
  CLAUDE.md section 2b summarises every direction and the open method questions.
- Usage-limit incidents: three wave-2 agents killed at ~06:30 (reset 10:00) and ext4 again at ~11:5x (reset 15:00);
  all GPU chains ran to completion detached; agents resumed from disk without repeating GPU work.
- Storage: results/codefact_*/scan_routing.parquet (34 MB each) and expert_parts/ are gitignored as regenerable
  intermediates; raw diagnostics live on the ephemeral NVMe (/opt/dlami/nvme/moe_ext2, moe_ext3).
- GPU minutes: wave 1 ≈ 30, wave 2 ≈ 10 (attn) + 55 (zoo) + 134 (codefact) → extensions total ≈ 3.8 h.

## 2026-09-21 — coordinator: Phase 2 wave 1 launched (F1, F2, F5 via ext5-engine + ext5-analysis; F4 via ext5-subject)
- User decisions: F4 before F3; F4 = subject-last-token column only; F5 metrics alongside Δ. Plan: RESEARCH_PLAN.md
  "Phase 2 execution plan". Engine ownership: ext5-engine only; ext5-subject implements suffix rows in a separate module.
- Pre-launch state: GPU idle, no queue waiters, git clean at 2a27fbc+, NVMe 49 GB free (new runs are small).

## 2026-09-21 01:55 UTC — ext5-subject: verification of the suffix-row executor done (F4, last-subject-token patching)
- New executor `moetrace/ext5_subject.py` (`SubjectEngine.run_subject`; engine.py untouched, helpers reused): suffix wavefront rows
  spawned at (layer l, position p) carry positions p..T-1, attend to the parent noised run's K/V for < p and their own for >= p;
  kinds zero/layer/expert/coalition_*/attn_layer/block/resid at p. Scripts `scripts/ext5_subject_{verify,gate,sweep,select,expert,analyze}.py`,
  chain `scripts/ext5_subject_chain.sh`.
- OLMoE verification (20 cases x 16 layers, `results/verify_ext5_subject_olmoe.json`, 93 s): vs transformers hooks replacing the
  self_attn / MoE / decoder-layer output at p: mean |dDelta| 0.139 / 0.150 / 0.168 (max 0.62 / 0.80 / 1.13) for attn_layer / layer /
  resid = ext2's final-token floor (0.19 / 0.15 / 0.13); 20-case mean-curve max dev 0.078 / 0.106 / 0.143; rescue corr 0.92 / 0.95 / 1.00
  (attention rescue at p is small in OLMoE, HF std 0.54, so its correlation is floor-limited). Null (zero vector at p = noised run) max
  0.49 mean 0.047; identity (clean donor on clean run) max 0.11. Consistency: the same prefill batch through run_subject at p = T-1 and
  through Engine.run is BIT-IDENTICAL (960/960 rows, prefill, vnorm); vs the stored olmoe_attnsweep rows (different batch) mean |dDelta|
  0.13-0.17, rescue corr 0.98-1.00, mean-curve max dev <= 0.094. Patched-vector norms vs HF within 2-10 %.
- Noise reference on these 20 cases: whole-span drop 7.12, last-subject-token-only 4.25, rest-of-span 5.68.
- Gate rule for the per-case correlation made noise-aware (corr >= min(0.95, r_exp - 0.03), r_exp = 1 - s_diff^2 / 2 s_HF^2); chain launched
  detached (re-runs verify with the extra field, gate, then 3 sweeps + 3 expert passes through gpu_queue.sh).

## 2026-09-21 01:55 UTC — ext5-analysis: F1.1 rankings and F1.2 population-level minimal sets done (zero GPU); F1.3 / F5 code ready, waiting for data
- Code (new files only): moetrace/ext5_rank.py (full ranking of every (layer, expert) pair with vectorised seed-0 bootstrap CIs identical to
  stats.summarize, all-case Spec matrix via protocol.active_controls, Kendall tau, additive greedy + per-case coverage greedy, cross-layer
  sum/max curves, F1.3 per-case minimal sets / interaction matrix / non-additivity decomposition), moetrace/ext5_metrics.py (F5: metric
  columns, rescue swap so analysis.py / ext1_analysis.py run unchanged under Δp, Δlog p, rank, KL, per-case ratio; funnel alternative),
  scripts/ext5_rank_analyze.py (+ ext5_rank_text.py), scripts/ext5_rank_subsets.py, scripts/ext5_metrics_analyze.py (+ ext5_metrics_text.py).
  Outputs: results/tables/ext5_rank_* (92 files), results/figures/ext5_rank_minimal_{qwen3_bos,mixtral_bos,mixtral_nobos}.png,
  results/ext5_rank_summary.json, results/sections/ext5_f1_rankings.md (F1.3 part pending), results/sections/ext5_f5_metrics.md (method + pending).
- Documented defaults: block = MoE-block patch (kind layer, same pass); orderings on discovery, evaluation on validation; block share defined
  only where the block CI excludes zero; recurrence 64/128. Observation: under additivity the greedy objective is linear in S, so greedy
  forward selection = descending all-case discovery rescue and the curve = its cumulative sum; a non-linear per-case coverage greedy was added.
- F1.1 headline: rankings agree wherever the effect is clear. Same 2-3 pairs lead under all-case rescue, active-only rescue, Spec and the
  discovery statistic in all three runs (tau rescue vs active-only 0.94-0.97; rescue vs Spec 0.53/0.35/0.43 among pairs with rescue CI > 0,
  0.64-0.73 within L*; over ALL recurrent pairs rescue vs Spec 0.38 Qwen3, 0.01-0.03 Mixtral = noise of near-zero pairs). Systematic
  disagreements: (a) junior partners (rescue > 0, Spec < 0: Qwen3 L43E046/E104; Mixtral L19E006, L20E005, L21E000, L20E000, L18E006),
  (b) Spec without rescue in layers whose block hurts (Qwen3 L47E032/E034, Mixtral L29-L31), (c) share/percentile reward weak layers.
  L44E069 rank 1/1/1/7/11/1, L42E115 2/2/2/2/1/2 (rescue, active-only, Spec, share, percentile, disc); Mixtral no-BOS L19E006 4/4/28/6/29/4.
- F1.2 headline (|S| for 50/80/90 % of the validation block rescue, discovery order): Qwen3 L44 1/6/9 (E069 alone 53 %), L42 1/2/6 (E115
  72 %), L43 3/7/8, L40 2/6/10; Mixtral BOS L19 1/3/4 (E002 63 %), L18 1/2/2 (E001 77 %), L21 1/3/5; no-BOS L19 1/4/5 (E002 51 %, fails
  recurrence), L18 1/3/3, L21 1/3/6. Additive end point vs exact coalition: equal on average at every layer (Qwen3 L44 sum +0.935 /
  coalition +0.916 / block +0.941, r 0.93, mean |diff| 0.36, 50 % of cases within 0.25; Mixtral 92-99 % within 0.25, r 0.90-0.98).
  Coverage (case reaches 80 % of own block): max 60-72 % Qwen3, 62-86 % Mixtral even with all active experts. Cross-layer: E069+E115 =
  80 % (per-case max) to 101 % (sum) of the L44 block; the sum over layers is an over-count (3.4x block at 59 pairs) -> F1.4.
- F1.3 driver tested on a synthetic subset_rows fixture (schema case_id, layer, experts, n_experts, rescue) for Qwen3 L44 (255 subsets x
  528 cases) and Mixtral L19; F5 driver tested on synthetic metric columns fabricated onto results/qwen3 and results/mixtral_nobos
  (identity Δ = logp_true - logp_foil exact by construction). Both write to a scratch dir in test mode. Waiting for results/*_subsets and
  results/*_metrics from ext5-engine.

## 2026-09-21 02:02 UTC — ext5-subject: Qwen3 subject-site sweep + expert pass done (Mixtral no-BOS/BOS queued)
- Gate passed 01:54 (noise-aware correlation rule). `results/qwen3_bos_subject`: 2 chunked passes (61 s + 49 s; 18,432 suffix rows /
  76k suffix tokens each, Smax 8, T 16), expert pass at L2,4,5 (+ fixed L42/L44) 76 s.
- Early site found: MoE-output patch at the last subject token peaks at **L4 +0.927 [0.647, 1.252]** (val; disc. argmax L4 +0.912), as large as
  the final-token L44 peak (+0.925); the curve is confined to L0-L9 (centre of mass 8.9 vs 38.6) and is ~0 at L42/L44. Attention output at p
  small (L1 +0.30); `resid` at p rises to +3.8 at L9-L11 (0.67 of the drop) then decays to 0 by L44 as the information leaves the token.
- No shared early-site expert: recurrent candidates exist (L4 E060 120/128, E046 96/128) but the best rescues +0.02-0.07 with Spec ~0; the clean
  top-8 coalition recovers the layer effect (+0.75 vs +0.88 at L4). Per case the best single expert gives ~90 % of the coalition, but it is a
  different expert per prompt (36-38 distinct winners / 128, most common in <= 19). L44E069 / L42E115 patched at p: rescue ~0.
- Noise on the last subject token alone causes 0.52 [0.46, 0.57] of the whole-span drop (5.95 -> 3.07); the rest of the span 0.87; strongly
  sub-additive. Clean/noised routing overlap at p only 0.33 (0.19-0.47) vs 0.65 at the final token.
- Analyzer `scripts/ext5_subject_analyze.py` produces tables ext5_subject_{peaks,experts,concentration,fixed}_<short>, noise_attribution,
  figure ext5_subject_curves_<short>.png, section results/sections/ext5_f4_subject.md (auto-filled).

## 2026-09-21 02:06 UTC — ext5-engine: engine extensions verified (F5 metrics, F2 attn_head, F1 coalition_set/multi); GPU chain launched
- Engine API (`moetrace/engine.py`, all backward compatible; `results/verify_olmoe.json` bit-identical to
  `results/verify_olmoe_before_ext5.json` on all 23 non-timing metrics, re-checked after the last edit):
  * `Engine.run(..., metrics=False, metrics_chunk=512)` -> `PassResult.metrics_prefill` / `.metrics_spawn` (dicts, keys
    `logp_true, logp_foil, p_true, p_foil` fp32, `rank_true` int32 (1 + #vocab logits strictly greater), `kl_to_clean` fp32 =
    KL(softmax(row) || softmax(clean prefill row of the same case)); arrays [B] and [S]). Clean reference of a prefill row:
    `PrefillSpec.clean_ref` (new, default -1 = inferred from spawns parent->clean, else itself -> KL = 0 exactly).
    Full log-softmax of the bf16 logits in fp32, row chunks; cost 0.03 s for 1.4k rows on OLMoE.
  * kind `attn_head` (`SpawnSpec.expert` = head index): v_h = W_o[:, h](H_h_clean - H_h_noised) in fp32, row starts before the
    MoE of layer l as h_pre_noised + bf16(Attn_noised + v_h); `DiagSpec.attn_heads_final=(layers)` records the pre-o_proj
    per-head outputs [B, nH, D] bf16; `DiagSpec.spawn_vectors=True` stores every spawn's fp32 vector (verification only).
  * kind `coalition_set` (`SpawnSpec.experts` = tuple): v = sum_{e in S} c_e_clean - c_e_noised (c_e = 0 if not routed).
  * kind `multi` (`SpawnSpec.steps` = ((layer, kind, experts), ...), increasing layers, kinds layer/expert/coalition_set/
    coalition_clean/zero): later steps replace the live row's OWN component by the clean one (v = clean - own).
  * `scripts/run_sweep.py` / `run_expert.py`: new flags `--metrics`, `--agent` (defaults unchanged; run_meta.json written when either is given).
- Verification `scripts/ext5_engine_verify.py` -> `results/verify_ext5_engine_olmoe.json` (OLMoE, 20 cases, 2 passes + HF):
  metrics: delta == logp_true - logp_foil to 9.5e-7 (prefill) / 1.5e-5 (spawn rows); KL(clean||clean) = 0 exactly, min KL 0;
  p_true + p_foil <= 1 on all 1400 rows; logp_true vs HF log-softmax max 0.33/0.54 (= the delta noise floor vs HF, 0.44/0.53);
  KL(noised||clean) 4.47 engine vs 4.49 HF; rank_true agrees with HF for 9/10 rows with HF rank <= 10 (max off by 2; tail ranks
  are bf16-sensitive). attn_head: sum_h v_h = W_o(H_clean - H_noised) to 6e-8 (fp32), = attn_layer vector to 0.2% rel. norm
  (bf16); vs HF hooks (o_proj input slice replaced) mean |dDelta| 0.147, 79% within 0.25, same floor as the whole-attention
  replace (0.147) and `layer` in verify_olmoe; identity on the clean run max 0.11. coalition_set/multi (on the 31/40 (case,
  layer) pairs whose in-pass routing equals pass 1's; the others are the known cross-pass bf16 routing flips): S = clean set ==
  coalition_clean EXACT (vector and delta), S = clean u noised == layer (vec 2e-7, delta exact), S = all == layer (exact delta),
  {e} == expert EXACT (248/248), single-step multi == single kind EXACT, multi (layer, coalition_set S=all) == multi (layer,
  layer) exact; multi (layer, layer) at 3 pairs + 1 triple vs HF double/triple replace hooks: mean |dDelta| 0.119, 91% within
  0.25, rescue r 0.99.
- Chain launched 02:05 UTC: `setsid nohup bash scripts/ext5_engine_chain.sh > logs/ext5_engine_chain.log 2>&1 &` (idempotent:
  steps with existing outputs are skipped). 12 gpu_queue jobs in order: ext5eng-sweep-qwen3_metrics, ext5eng-expert-qwen3_metrics,
  ext5eng-sweep-mixtral_nobos_metrics, ext5eng-expert-mixtral_nobos_metrics, ext5eng-sweep-mixtral_bos_metrics,
  ext5eng-expert-mixtral_bos_metrics, ext5eng-heads-qwen3 (L40,43,44), ext5eng-heads-mixtral_nobos (L15,18,19,24),
  ext5eng-heads-mixtral_bos (L15,18,19,24), ext5eng-subsets-qwen3-L44, ext5eng-subsets-qwen3-L42, ext5eng-subsets-mixtral_nobos
  (L18,19). ~16 min GPU; interleaves with ext5subj-* jobs.
- Output paths / schemas for ext5-analysis (all paper set, 128/128, Qwen3 defaults, Mixtral no-BOS and BOS):
  * F5 `results/{qwen3,mixtral_nobos,mixtral_bos}_metrics/`: `case_sets.json` (copied from base), `sweep_rows.parquet`
    (run_sweep schema + `logp_true, logp_foil, p_true, p_foil, rank_true, kl_to_clean` on clean/noised/layer rows; clean rows have
    kl_to_clean = 0), `sweep_routing.parquet`, `sweep_cases.parquet`, `sweep_summary.json`, `expert_rows.parquet` (layers 42,44 /
    18,19, `--no-pairs`: kinds layer, coalition_clean, coalition_union, expert, expert_noised_only; run_expert schema + the six
    metric columns of the patched row + `p_true_noised`, `logp_true_noised` of the same pass), `expert_prefill_L42_44.parquet` /
    `expert_prefill_L18_19.parquet` (case_id, delta_clean, delta_noised, `<metric>_clean`, `<metric>_noised` for all six),
    `run_meta.json` (keys `sweep`, `expert`).
  * F1.3 `results/{qwen3_subsets,mixtral_nobos_subsets}/subset_rows.parquet`: case_id, layer, kind (`coalition_set` | `layer` |
    `coalition_clean` | `block` reference rows of the SAME pass), experts (sorted comma-joined, '' for reference kinds), n_experts
    (0 for reference kinds), logit_true, logit_foil, delta, rescue, vnorm, delta_clean, delta_noised, n_clean_active, sigma_mult;
    `subset_prefill_L<l>.parquet`; inputs copied from the `_metrics` run (case_sets.json, sweep_cases, sweep_routing; the clean
    sets come from that routing, in-pass agreement logged in run_meta.json). Qwen3 L44 and L42 (255 subsets x 256 cases each),
    Mixtral no-BOS L18, L19 (3 subsets). Run names are in ext5-analysis' auto-detect list.
  * F2 `results/{qwen3,mixtral_nobos,mixtral_bos}_heads/head_rows.parquet`: case_id, layer, head (-1 for reference kinds), kind
    (`attn_head` | `attn_layer` | `layer` | `block`), logit_true, logit_foil, delta, rescue, vnorm, delta_clean, delta_noised,
    sigma_mult; `head_prefill.parquet`, `sweep_cases.parquet`, `sweep_routing.parquet`, `head_attn_final.npz` (attn [n_layers, 2n,
    nH, T] fp16, rows clean 0..n-1 then noised), `head_summary.json`, `run_meta.json`. Pilot `results/olmoe_heads` (strict+relaxed,
    L8,10,12,13) already written.

## 2026-09-21 02:13 UTC — ext5-engine: run_sweep.py meta crash fixed (pass outputs unaffected)
- `ext5eng-sweep-qwen3_metrics` (02:07-02:08) wrote all outputs (sweep_rows/routing/cases.parquet, sweep_summary.json; L*=44,
  val +0.952, 25,344 layer rows + 512 prefill rows with the six metric columns; clean-row KL = 0, Δ ≡ logp_true − logp_foil) and
  then crashed (rc=1) in `_write_meta`: the script's case-table code reuses the name `meta` for a DataFrame, which clobbered the
  new run-meta dict. Fix: dict renamed `run_meta` in run_sweep.py; `_write_meta` in run_sweep.py and run_expert.py now sanitises
  values to JSON-native types (`_json_safe`, numpy → python, other objects dropped), tolerates a corrupt existing run_meta.json
  (starts from {}) and dumps with `default=str`. `results/qwen3_metrics/run_meta.json` rebuilt on CPU from the summary/parquet.
  The chain skips the completed sweep (done-file present); the queued expert step picks up the fixed run_expert.py.

## 2026-09-21 02:25 UTC — ext5-subject: all runs done (chain "chain done" 02:15:07, six ext5subj-* jobs rc=0); section written
- Runs: `results/qwen3_bos_subject` (sweep 61+49 s, expert L2/4/5 + fixed 42/44, 76 s), `results/mixtral_nobos_subject` (sweep 97+93 s,
  expert L1/4/6 + fixed 18/19, 113 s), `results/mixtral_bos_subject` (sweep 2 passes, expert L3/4/6 + fixed 18/19). GPU total ≈ 15 min
  incl. two verification runs. All row-level Parquet + run_meta.json / expert_meta.json / expert_layers.json saved.
- Early site in all three runs (MoE-output patch at the last subject token, validation): Qwen3 L4 +0.927 [0.647, 1.252] (= final-token L44
  +0.925; both 0.16 of the drop); Mixtral no-BOS L6 +1.197 [0.940, 1.469] (val argmax L4 +1.349) vs final L21 +0.531 (0.24 vs 0.11 of the
  drop); Mixtral BOS L4 +2.312 [1.870, 2.779] vs final L19 +0.561 (0.47 vs 0.11). Bands disjoint (L0-L9 vs L40-L44 / L18-L24; curve
  correlation -0.16 to -0.28). Attention output at p only at L1 (+0.30 / +0.73 / +0.78). `resid` at p plateaus at 0.58-0.76 of the drop
  (L4-L11) then decays to 0 by the last layer.
- Noise attribution: last subject token alone = 0.52 / 0.47 / 0.51 of the whole-span drop; rest of span 0.87 / 0.94 / 0.93; sub-additive.
- No shared early-site expert: Qwen3 recurrent candidates at L2/4/5 rescue +0.02-0.07, Spec ~0 (L4 E046 -0.04); Mixtral has no expert above
  64/128 at L1/4/6 (max 38-60); per case the best single expert carries 88-99 % of the coalition but differs across prompts (Qwen3 36-38
  distinct winners /128; Mixtral all 8). Final-token experts at p: L44E069 -0.005 (active 70/128), L42E115 -0.005, L19E002 +0.01-0.02
  (Spec < 0), L19E006 / L18E001 active <= 6/128. Late layers at p: MoE patch -0.003 to +0.07.
- Mixtral BOS vs no BOS at p: curves r 0.94-0.96; MoE peak L4 under both, BOS larger (+0.96 [0.60, 1.35] paired at L4); attention identical.
- Outputs: tables `results/tables/ext5_subject_{peaks,experts,concentration,fixed}_{qwen3_bos,mixtral_nobos,mixtral_bos}`,
  `ext5_subject_noise_attribution`, `ext5_subject_protocol_mixtral`, `ext5_subject_comparison`; figures `results/figures/
  ext5_subject_curves_<short>.{png,pdf}`, `ext5_subject_mixtral_protocols`; `results/ext5_subject_summary.json`; section
  `results/sections/ext5_f4_subject.md` (auto-filled by `scripts/ext5_subject_analyze.py`, summary from `ext5_f4_subject_summary.md`,
  reading from `ext5_f4_subject_interpretation.md`). To append to EXTENSIONS_REPORT.md: add `ext5_f4_subject.md` to build_extensions_report.py.

## 2026-09-21 02:40 UTC — ext5-engine: F5 metrics runs, F2 head sweeps and F1.3 subset passes DONE (chain finished 02:33:23 UTC, 12/12 rc=0 after the meta fix)
- F5 `results/{qwen3,mixtral_nobos,mixtral_bos}_metrics/`: sweep_rows 12,800 / 8,704 / 8,704 rows (kinds clean, noised, layer; 16 columns incl.
  logp_true, logp_foil, p_true, p_foil, rank_true, kl_to_clean), expert_rows 6,883 / 2,848 / 2,732 rows (layers 42,44 / 18,19 / 18,19; kinds
  layer, coalition_clean, coalition_union, expert, expert_noised_only; 29 columns incl. the six metrics + p_true_noised, logp_true_noised),
  expert_prefill_L42_44 / L18_19.parquet (delta_clean, delta_noised, <metric>_clean, <metric>_noised), sweep_routing, sweep_cases,
  sweep_summary.json, run_meta.json (keys sweep, expert). Sanity on all three: clean-row KL = 0 exactly, p_true + p_foil <= 1 on every row,
  max |Δ − (logp_true − logp_foil)| = 0.125 (one bf16 ulp of a logit in [16, 32); Δ uses the bf16 pair dot product, logp the full-vocabulary
  GEMM). Selections unchanged: Qwen3 L44 E069 (114/128 active, all-case +0.483), L42 E115 (125/128, +0.477); Mixtral no-BOS L19 E006 (91/128,
  +0.074), L18 E001 (76/128, +0.162); Mixtral BOS L19 E002 (76/128, +0.378). Metrics cost 0.1 s per pass.
- F2 `results/{qwen3,mixtral_nobos,mixtral_bos}_heads/`: head_rows 26,880 / 35,840 / 35,840 rows (attn_head 32 heads x 3 or 4 layers x 256 cases +
  attn_layer, layer, block reference rows; layers 40,43,44 / 15,18,19,24), head_prefill, sweep_cases, sweep_routing, head_attn_final.npz,
  head_summary.json, run_meta.json. Pilot results/olmoe_heads (strict+relaxed, L8,10,12,13).
- F1.3 `results/qwen3_subsets/subset_rows.parquet` 132,096 rows (L44 and L42: 255 coalition_set subsets x 256 cases + layer, coalition_clean,
  block reference rows per case and layer, same pass) and `results/mixtral_nobos_subsets/subset_rows.parquet` 3,072 rows (L18, L19: 3 subsets);
  subset_prefill_L<l>.parquet; run_meta.json with per-pass routing agreement with the `_metrics` base routing (Qwen3 223/256 at L44, 228/256
  at L42; Mixtral 252/256) — for the non-matching cases the enumerated clean set is the base run's, so the in-pass clean set may differ by one
  expert (known cross-pass bf16 routing flips). Quick look (validation): Qwen3 L44 block +0.973, layer +0.950, coalition_clean +0.931, best
  single expert +0.821, best single >= 80% of block in 74/128 cases; L42 block +0.652 / best single +0.551 / 78/128; Mixtral no-BOS L18 block
  +1.003 vs layer +0.199 (34/128), L19 block +1.201 vs layer +0.450 (26/128).
- GPU: chain 12 jobs 1,259 s (21.0 min) incl. the 82 s crashed-then-complete sweep; verifications + OLMoE pilot 322 s (5.4 min); total 26.4 min.

## 2026-09-21 02:40 UTC — ext5-engine: F2 analysis done, section written
- `python scripts/ext5_heads_analyze.py` (moetrace/ext5_heads.py, ext5_heads_section.py) on qwen3_heads, mixtral_nobos_heads, mixtral_bos_heads
  (+ pilot olmoe_heads): tables results/tables/ext5_heads_{ranking_<run>_L<l>,ranking_<run>_all,additivity,minimal,attention}_*.{csv,md},
  figures results/figures/ext5_heads_{qwen3,mixtral_nobos,mixtral_bos,olmoe}.{png,pdf}, results/ext5_heads_summary.json, section
  results/sections/ext5_f2_heads.md (summary ≤ 250 words on top; method; OLMoE verification; per-run per-layer bullets; figures; tables;
  cross-run reading; caveats).
- Headline: mover heads carry the attention rescue — Qwen3 L40 h13 +0.95 [0.81, 1.09] (60% of +1.58, Spec +0.93; h15 +0.46, h14 +0.41; 2 heads
  for 80%), L43 distributed (h11/h15/h28, 4 heads), L44 null (attention +0.04, best head +0.014); Mixtral L18 h4 +0.61 / +0.79 (no BOS / BOS;
  76% / 80%, alone sufficient in 57% / 55% of cases), L24 h22 +0.78 / +0.70 (86% / 82%), L19 h29-31 (h29 +0.34 / +0.41; 4-5 heads for 80%),
  L15 h1/h3; identical heads under both Mixtral protocols. Mover heads read the last subject token (clean mass 0.4-0.5, halved by the noise,
  mass moves to relation tokens without BOS and to the position-0 sink with BOS). Σ heads = attention rescue on the mean at every layer (gaps
  within [-0.20, +0.06], CIs cover 0), per-case r 0.3-0.7 (bf16 noise x 32 rows) — minimal sets are additive estimates.

## 2026-09-21 02:38 UTC — ext5-analysis: F5 probability-scale metrics analysed (results/{qwen3,mixtral_nobos,mixtral_bos}_metrics), section written
- Code: moetrace/ext5_metrics.py (metric columns, same-pass references for expert rows from expert_prefill_L*.parquet, rescue-column swap so
  analysis.py / ext1_analysis.py run unchanged under Δp, Δlog p, rank, KL, per-case Δ/drop; funnel alternative; clean top-1 decoding; tail
  concentration), scripts/ext5_metrics_analyze.py (+ ext5_metrics_text.py). Outputs: results/tables/ext5_metrics_{layers,experts,fixed,joint,
  normalised,tail,funnel,funnel_sets,layer_curves,percase_*}_<run>.{md,csv}, results/figures/ext5_metrics_{curves,percase}_<run>.{png,pdf},
  results/ext5_metrics_summary.json, results/sections/ext5_f5_metrics.md.
- Identity: max |Δ − (log p_true − log p_foil)| = 0.125 in every run = one bf16 ulp of the stored Δ (holds to rounding). Expert rows must use
  their own pass's noised reference: cross-pass Δ_noised differs by up to 1.375 in 79 % of expert rows.
- Findings: (1) the clean p(true) is small (median 0.024-0.044); the true object is the clean top-1 in only 29-32 % of paper cases and otherwise
  the top-1 is ' the' / ' a' / ' of' / whitespace in 84-90 % of them; p_clean >= 0.5 would keep 26-45 of 256 paper cases (Jaccard 0.10-0.19
  with the Δ funnel). (2) Qwen3 and Mixtral-BOS: layer (L44 / L19) and expert (E069 / E002) identical under all six metrics; Spec positive under
  five (Δp underpowered: every Spec CI includes 0). L42E115 is joint top-1 under Δlog p and rank (Qwen3), L18E001 under Δp (BOS). (3) Mixtral
  no-BOS: Δ and Δ/drop -> L19E006 (Spec < 0 under all six metrics); Δp and rank -> L18E001 (Spec > 0); Δlog p and KL -> L0, a heavy-tail
  artefact (66 % of the L0 KL rescue from the 26 most disrupted noised cases, KL(noised||clean) >= 4.2, rank in the 100s-16,000s; r 0.87; with
  BOS the L0 KL rescue is 0.001) = the no-BOS sink-state prompts of ext3 that Δ cannot see. (4) Normalised rescue: 17 % (Qwen3 L44), 11 % / 9 %
  (Mixtral L19 BOS / no-BOS) of the lost log-odds vs 2.6-4.7 % of the lost probability mass; the 24-28 % top-1-flipped cases carry 52-83 % of
  the Δp rescue vs 24-41 % of the Δ rescue; per-case r(Δ, Δp) 0.08-0.21, r(Δ, rank) 0.42-0.87, r(Δ, Δlog p) 0.48-0.84.
- Recommendation (section): alongside Δ report normalised rescue, rank recovery (+ top-1 recovery), clean top-1 rate / median p_clean; Δp only
  descriptively; KL / Δlog p as protocol diagnostics; keep the Δ funnel.

## 2026-09-21 02:40 UTC — ext5-analysis: F1.3 per-case minimal sets analysed (results/{qwen3,mixtral_nobos}_subsets), F1 section complete
- scripts/ext5_rank_subsets.py on the delivered schema (kind column; same-pass layer / coalition_clean / block reference rows). Outputs:
  results/tables/ext5_rank_subsets_{overview,sizes,singletons,interactions,decomposition}_<short>.{md,csv}, per-layer percase / interaction
  matrix / counts / pairs / decomposition CSVs, results/figures/ext5_rank_subsets_{qwen3,mixtral_nobos}.{png,pdf}, results/ext5_subsets_summary.json,
  fragment results/tables/ext5_rank_subsets_section.md embedded in results/sections/ext5_f1_rankings.md (<= 250-word summary added at the top).
- Checks: exhaustive (255 x 256 per Qwen3 layer, 130,560 rows; 3 subsets Mixtral); full clean set = same-pass coalition_clean to 0.004; full
  set = 94-98 % of the block.
- Findings: per-case minimal set for 80 % of the case's own block: one expert in 59-60 % of eligible Qwen3 cases, <= 2 in 87-88 % (median 1;
  2 cases never); Mixtral one of two in 53-54 %. {E115} alone suffices (80 %) in 52 % of its active cases and is the size-1 set in 81 % of
  them; {E069} in 32 % / 48 %; {L18E001} 46 %, {L19E002} 43 %, {L19E006} 31 %. Additive prediction = exact minimal size in 97 % (L44) / 91 %
  (L42) of cases, same set in 90 % / 84 %, reaches the target when patched exactly in 95 % / 90 % (Mixtral 100 % / 97 %). Mean pairwise
  interaction +0.004 / +0.007 (CI > 0; 26-28 % negative; quantised at the bf16 ulp 0.125); E069 / E115 additive with partners (+0.005 / +0.003);
  redundancy only between strong pairs (L44 E060+E069 -0.10; L19 E002+E006 -0.034, E002+E004 -0.26); L18 E001+E006 synergistic +0.029
  [+0.007, +0.052]. Higher-order decomposition not resolvable at bf16 precision (pairwise sum overshoots, remainder cancels).

## 2026-09-21 02:50 UTC — coordinator: Phase 2 wave 1 complete (F4, F1, F2, F5)
- Agents ext5-subject, ext5-engine, ext5-analysis all delivered final reports. GPU: subject chain 6 jobs ≈ 15 min
  (01:52–02:15), engine chain 12 jobs 21 min + verifications/pilot 5.4 min (01:51–02:33); GPU idle since 02:33.
- Incidents: all three agents killed by the account usage limit at ~00:49 UTC (reset 01:50), resumed from disk at
  01:51 with no GPU work lost; `run_sweep.py` meta write crashed after a complete pass (variable name clash), fixed by
  ext5-engine within 2 min, before the queued expert job started; the subject agent's Monitor woke it on every queue
  event (stopped to save tokens).
- Deliverables: results/sections/ext5_{f4_subject,f1_rankings,f2_heads,f5_metrics}.md assembled into
  results/EXTENSIONS_REPORT.md (Directions 5-F4/5-F1/5-F2/5-F5 registered in scripts/build_extensions_report.py);
  CLAUDE.md sections 1/2b/3/5, README, RESEARCH_PLAN.md updated. Wave 2 (F3, F1.4, optional F2 Qwen3-Instruct, F4 grid)
  awaits the user's go-ahead.

## 2026-09-28 02:15 UTC — ext6-str: symmetric token replacement (Zhang & Nanda 2024) started
- Motivation: best-practices comparison with arXiv:2309.16042 (PDF gitignored at repo root): the paper's corruption (GN) is the method
  that paper recommends against; STR on CounterFact is feasible on the paper set.
- User decisions (2026-09-28): (1) case set = paper IDs + split restricted to cases with a symmetric donor, recurrence = half of the
  retained discovery cases; (2) up to 5 donors per case, uniformly random among qualifying (random.Random(2000 + case_id) order),
  per-case = donor mean, single-donor (slot 0) as sensitivity; (3) donor qualifies when logit(foil) - logit(true) >= 1.0 on the donor
  prompt (foil = target_new = the donor subject's true object); (4) runs: Qwen3, Mixtral no-BOS, Mixtral BOS.
- Donor = same relation, true object == case foil, same template, identical token positions and template tokens, same continuation ids.
  Candidates: Qwen3 3883 over 221/256 cases, Mixtral 3397 over 218/256 (quartiles 2/6/18 per case).
- Code: moetrace/ext6_str.py, scripts/ext6_str_{filter,sweep,expert,verify}.py (no engine change; corrupted run = plain prefill row).
- 02:11 UTC verification on OLMoE vs transformers hooks (results/verify_ext6_str_olmoe.json, 12 case/donor pairs): prefill Δ max diff 0.125
  (one ulp); layer patch max 0.31, mean |diff| 0.05, 98% within 0.25, rescue r 0.994; expert patch (111 rows) max 0.19, r 0.992; identity
  0.125. GN calibration (verify_ext2_attn_olmoe.json, layer kind): max 1.12, r 0.983. STR adds no new numerics.
- 02:18 UTC chain scripts/ext6_str_chain.sh launched (log logs/ext6_str_chain.log): phase A filter -> sweep -> key-layer expert for
  qwen3_str, mixtral_nobos_str, mixtral_bos_str; phase B remaining layers; analysis scripts/ext6_str_analyze.py.

## 2026-09-28 03:05 UTC — codewino design (interactive session): WinoGrande-style single-token twin benchmark on code
- User request: study WinoGrande in depth and design a code benchmark where one swapped token keeps the code valid and
  in-distribution but changes the expected next token, following Zhang & Nanda 2024 (STR, logit difference).
- Deliverable: docs/codewino_design.md (WinoGrande dossier summary, Z&N requirements, twin definition C1-C8, trigger-answer
  relations T1-T4, families BRK/CONT/NEG/COMPL/EXEC/GEN/NULL, build pipeline, tracing protocol incl. GN-vs-STR control,
  risks, 7 user decisions). No repo code written, no GPU used.
- CPU probe (scratchpad only) on the 36,996 CodeFact functions, both tokenizers aligned: BRK 6,742 (~1,657 tuple<->list in
  runtime-safe contexts; 2,770 in invalid contexts such as `except [A, B`), CONT 1,939, NEG 253 (45 True/False), COMPL 173;
  MBPP seed asserts for EXEC 474. NEG needs the CSN train split (412k functions).
- Verified from the official WinoGrande v1.1 zip: XL train = 40,398 lines (paper says 40,938); dev qID suffix equals the
  gold label in 1,267/1,267 items and test.jsonl keeps the suffixes.
- 02:14-03:10 UTC chain complete, all jobs rc=0 (GPU ≈ 58 min: filters 2+3.5+2.8 min, sweeps 2+3+3 min, key-layer expert passes
  2+3+3 min, remaining-layer passes 17+8.5+8 min). Retained cases 215 / 212 / 213 (disc 107/106/107, val 108/106/106), donor rows
  852 / 848 / 858 (4.0 per case), qualify rate 0.86-0.89; mean Δ_clean +5.8 / +5.4 / +6.6, Δ_corrupt -6.4 / -5.1 / -6.3.
- Results (results/sections/ext6_str.md, tables results/tables/ext6_str_*, results/ext6_str_summary.json): Qwen3 L44 (val +2.06) and
  L44E069 (Spec +0.95; per unit drop 0.082 vs GN 0.075), L42E115 second; all-layer joint top-1 L44E069 (GN same cases: L42E115/E069
  tie). Mixtral no BOS L19, E006 only recurrent candidate, Spec -0.29, top-2 coalition +0.80 = block +0.82, L18E001 +0.15. Mixtral BOS
  discovery L21 ≈ L20 ≈ L19 (+1.11/+1.07/+1.07) → L21E001 (Spec +0.40); L19E002 Spec +0.30; first donor picks L19E002. Normalised
  block rescue 0.177/0.077/0.085 vs GN 0.167/0.089/0.113; STR-GN curve r 0.99/0.94/0.94. Equal-norm active pair at L19 under STR ≈ 0
  for E006 and E002 (GN: E002 +0.07-0.11, E006 -0.07). Donor dispersion at the paper layer: median SD 0.34-0.42, sign agreement 0.89-0.95.
- Docs: section registered as Direction 6 in scripts/build_extensions_report.py, EXTENSIONS_REPORT.md rebuilt; CLAUDE.md, README updated.
  Not committed (awaiting the user).

## 2026-09-28 17:00 UTC — ext6-str grid: layer x position heatmap under STR (Zhang & Nanda Section 4.1 / Figure 4)
- User request: extend the STR run to the layer x position MoE-output patch grid; Qwen3 and Mixtral with BOS first, Mixtral without
  BOS only after the user has seen those two.
- Executor: moetrace/ext5_subject.py gained two backward-compatible options: SubjectSpawn.window (joint restoration of the MoE output
  at p over consecutive layers, kind 'layer') and run_subject(metrics=True) (full-softmax logp/p/rank of true and foil at the final
  position, no KL). Regression: scripts/ext5_subject_verify.py re-run, results/verify_ext5_subject_olmoe.json identical to the saved
  copy verify_ext5_subject_olmoe_before_ext6.json in all 70 non-timing fields.
- Verification (scripts/ext6_str_grid_verify.py -> results/verify_ext6_str_grid_olmoe.json; OLMoE, 8 STR pairs, 50 (case, position)
  units from the first subject token, 16 layers): window 1 max |ΔΔ| 0.59, mean 0.077, 93% within 0.25, rescue r 0.991; window 5 max
  0.50, mean 0.072, r 0.998; p(true) mean |diff| 0.0002 / 0.0005; zero-kind null invariant 0.125. GN calibration at p (F4): max 0.80,
  r 0.952.
- Grid: scripts/ext6_str_grid.py (units = case x position from the first subject token; prefix positions are exactly 0 under STR),
  all selected donors, packing by suffix length (<= 150k padded suffix tokens, <= 40k rows per pass): Qwen3 1518 units / 11 passes,
  Mixtral BOS 1584 units / 8 passes per window. Chain scripts/ext6_str_grid_chain.sh (w1 both, analysis, w5 both, analysis).
- 17:00-17:40 UTC grid chain complete, all rc=0 (GPU ≈ 40 min: Qwen3 w1 517 s, Mixtral BOS w1 651 s, Qwen3 w5 541 s, Mixtral BOS w5
  678 s; transient CUDA allocator retry warnings in the log, no failed allocation). Rows complete: Qwen3 284,832 per window, Mixtral
  BOS 198,304 per window, no NaN. Last-token column = final-position STR sweep (curve r 0.9999, per-case r 0.985 / 0.992).
- Results (results/sections/ext6_str_grid.md = Direction 6b, tables results/tables/ext6_str_grid_*, figures ext6_str_grid_*,
  numbers results/ext6_str_grid_summary.json): two sites as in Meng et al. Last subject token peaks at L0 (Qwen3 +0.187, Mixtral BOS
  +0.304 of the drop), layer sum 1.27 / 1.60 drops; final position peaks at the paper's band (Qwen3 L44 +0.176, Mixtral L19-L21
  +0.086); first subsequent / further tokens <= 0.007. Last / middle subject-token ratio, single layer: LD 2.87x / 4.79x, probability
  2.65x / 6.54x (agree); window 5: LD 3.18x / 5.04x, probability 3.99x / 14.9x. Sliding / summed single layers: LD 0.83-0.97 (Qwen3),
  0.60-1.09 (Mixtral); Δp 3.5-3.7 / 2.8-6.1 (softmax non-linearity). GN (F4 runs) vs STR at the last subject token, same cases,
  normalised: Qwen3 GN peak L4 +0.160 vs STR L0 +0.187, sums 1.34 vs 1.27 (1.06x); Mixtral BOS GN L4 +0.487 vs STR L0 +0.304, sums
  2.83 vs 1.60 (1.77x) - Zhang & Nanda's GN inflation at the early site.
- Mixtral without BOS not run (awaiting the user's decision). Not committed.

## 2026-10-04T06:07Z ext7 WinoGrande STR scan (interactive session)
- Built token-symmetric WinoGrande STR pairs (fill the blank with each twin answer, predict the sentence-final single-token trigger; rules W1-W6 in moetrace/ext7_wino.py; scripts/ext7_wino_build.py -> data/wino_str/): train_xl 20,199 twins -> 6,737 with a single sentence-final trigger word -> kept Qwen3 4,514, Mixtral 2,421 (no BOS = BOS), OLMoE 3,809. Raw WinoGrande 1.1 copied to data/winogrande_1.1 (gitignored).
- Launched scripts/ext7_wino_scan_chain.sh (competence scan, 6 prefill rows per pair: clean/GN/local x A/B), logs/ext7_wino_scan_chain.log.
- 2026-10-04T06:17Z User decision: Phase 3 uses STR only (no GN filter, no GN control). Primary WinoGrande pair set = margin both ways (Delta_A >= 1, Delta_B <= -1). The GN columns of the running scans are unused; GN rows to be removed from scripts/ext7_wino_scan.py after the chain ends. RESEARCH_PLAN.md Phase 3 updated.
- 2026-10-04T06:26Z ext7 scans complete (logs/ext7_wino_scan_chain.log, all rc=0; GPU ~17 min). STR-only funnel (results/tables/ext7_wino_funnel.md): margin pairs Qwen3 2,099 / Mixtral no BOS 1,109 / BOS 1,151 / OLMoE 1,325; shared by all three protocols 776; dev twins Qwen3 27/68, Mixtral 11/42. scripts/ext7_wino_scan.py now STR only (GN rows removed after the chain; smoke-tested on OLMoE). Phase 3 plan in RESEARCH_PLAN.md.
- 2026-10-04T07:04Z [ext8-addback] started; step E (engine extension: attn_layer/block steps in multi, noising direction, all-layer multi) in moetrace/engine_ext8_dev.py

## 2026-10-04T07:05Z [ext7-controls] start (sub-agent ext7-controls: CounterFact STR attention sweep, W7 role swap, W8 IOI, three-task table)
- [ext7-controls] Read CLAUDE.md, RESEARCH_PLAN.md Phase 3, ext6/ext2b code. GPU idle at start. Plan: task 1 (OLMoE check of attn_layer/block on STR donor pairs, then results/{qwen3,mixtral_bos}_str_attnsweep), data builders for data/wino_role/ and data/ioi/ (CPU), competence scans, then runs with ext7-wino's runner after EXT7 RUNNER READY.
- 2026-10-04T07:07:14Z [ext7-wino] started (agent ext7-wino): W1-W6 + W4 on WinoGrande STR pairs; status in logs/agent_status_ext7-wino.md
- 2026-10-04T07:20:20Z [ext7-wino] W1 verification (OLMoE, 20 WinoGrande margin pairs, 40 directed cases, vs transformers hooks): final-position layer/attn_layer/block/resid r 0.995/0.989/0.997/0.999, mean |diff| 0.068-0.071, attn_head r 0.905 (small effects, SD 0.15); STR-position grid layer/block/resid r 0.991/0.996/0.999, window 5 r 0.998. Found: SubjectEngine.run_subject (ext5_subject.py, not mine) turns attn_layer rows into block rows when the SAME pass contains a window>1 row (wf_winend = spawn layer for window-1 pre-MoE rows); ext6 grid and ext7 production passes never mix them; re-running the check with separate passes. Chain scripts/ext7_wino_chain1.sh started (logs/ext7_wino_chain1.log).
- 2026-10-04T07:25Z [ext7-controls] task 1 done (GPU ≈ 9 min): OLMoE check results/verify_ext7_controls_cf_olmoe.json (12 STR donor pairs x 16 layers vs HF hooks: rescue r attn_layer 0.996 / block 0.998 / layer 0.993, max |ΔΔ| 0.47; identity 0.25). Sweeps results/{qwen3,mixtral_bos}_str_attnsweep (scripts/ext7_cf_attnsweep.py, donor level, kinds attn_layer/layer/block at every layer; 852 / 858 donor rows). Validation (donor mean): Qwen3 attention L40 +3.96 (0.339 of the drop; GN same cases 0.284), MoE L44 +2.08 (0.178; GN 0.163), block L40 +4.74; Mixtral BOS attention L18 +2.33 (0.180; GN 0.192), MoE L21 +1.10 (0.085; GN L19 0.113), block L19 +3.47. Attention share of the positive rescue (AUC+) 0.544 [0.511, 0.572] / 0.578 [0.552, 0.604] (GN same cases 0.504 / 0.563). Tables results/tables/ext7_controls_cf_*, figure results/figures/ext7_controls_cf_curves.png.
- 2026-10-04T07:25Z [ext7-controls] W7 / W8 data: data/wino_role/pairs_{qwen3,mixtral_bos}.parquet (632 / 294 role pairs from 316 / 147 name twins of the margin pools; 213 Qwen3 twins fail token symmetry because a sentence-initial name tokenises differently); data/ioi/ (1,600 items from the Easy-Transformer ioi_dataset.py templates, 65 of 99 names + 8 places + 6 of 8 objects single-token under both tokenizers; corruptions s2io and s1io). Competence scans queued (scripts/ext7_role_ioi_scan_chain.sh).
- 2026-10-04T07:33:27Z [ext7-wino] EXT7 RUNNER READY (final-position sweep). Generic STR-pair runner, verified on OLMoE (final-position kinds vs transformers hooks, results/verify_ext7_wino_olmoe.json) and run on Qwen3 WinoGrande (results/wino_qwen3_str, 1384 directed cases, 4 passes, 235 s).
  Inputs: pair parquet with pair_id, ids_a, ids_b (JSON, equal length, tokenizer defaults), str_pos (JSON positions where A and B differ; fallback opt_pos), trig_a, trig_b (+ optional stratum_* columns, copied to sweep_cases); case-set JSON in the format of data/wino_str/case_sets.json ("pair_idx_of" {pair_id: idx}, "directed" {split: [2*idx+d]}, optional "own_pool_directed"). Splits discovery/validation -> family main, rep_<s> -> family rep, any other key k -> family k (split "all"). d=0: clean A / corrupted B / true trig_a; d=1 the reverse.
  Usage: python scripts/ext7_wino_sweep.py --model qwen3|mixtral --pairs <parquet> --case-sets <json> --out <run> [--splits k1,k2] [--own-pool <proto>] [--kinds layer,attn_layer,block(,resid)] [--max-rows 60000 (Mixtral 50000)] [--dry-run]  (submit via scripts/gpu_queue.sh) -> results/<run>/str_sweep_rows.parquet (kinds clean/corrupt/<kinds>, rescue = Delta_patched - Delta_corrupt), str_sweep_routing.parquet, sweep_cases.parquet (delta_clean, delta_corrupt, drop), case_sets.json (families), run_meta.json
  Analysis helpers (CPU): moetrace/ext7_pairs.py PairRun(run), sweep_curves, sweep_peaks, attention_share (Direction-2b AUC+ share, pair bootstrap), additivity, summ / ratio (pair-level bootstrap), model_data(run) -> analysis.ModelData. Expert / grid / heads / W4 runners announced separately.
- 2026-10-04T07:36Z [ext8-addback] engine.py replaced by the verified dev copy (additive: attn_layer/block steps in multi, SpawnSpec.kl_ref, DiagSpec.contrib_dla, row-chunked multi later steps); backups results/verify_olmoe_before_ext8.json, results/verify_ext5_engine_olmoe_before_ext8.json; regression runs queued
- 2026-10-04T07:43:27Z [ext7-wino] EXT7 RUNNER READY (expert, grid, heads; smoke-tested end to end on OLMoE WinoGrande pairs, grid executor verified vs transformers hooks; same common args as the sweep: --model --pairs --case-sets --out [--splits] [--own-pool]; all need the run's final-position sweep first, and the same --out):
  W6 experts: python scripts/ext7_wino_expert.py <common> [--families main] [--layers all|a-b|l1,l2] [--pairs-layers auto|l1,l2 (equal-norm expert_scaled rows; auto = discovery MoE argmax)] [--only-scaled] [--max-spawn 60000 (Mixtral 40000)] -> str_expert_rows.parquet (ext6 schema, case_id = directed id, slot 0)
  W3 grid: python scripts/ext7_wino_grid.py <common> [--families main] --kinds layer,attn_layer,resid(,block) --window 1  |  --kinds layer --window 5   (positions from min(str_pos) to T-1; NEVER mix window>1 and attn_layer rows in one pass: ext5_subject.run_subject then turns attn_layer into block) [--max-rows 60000 --token-budget 200000] -> str_grid_w<W>_rows.parquet (pos, cat = token group incl. 'between STR tokens' for non-contiguous sites, kind, layer, rescue, dp, p_true ...)
  W5 heads: python scripts/ext7_wino_heads.py <common> [--families main] --layers auto|l1,l2 (auto = top-4 discovery attn_layer layers + 1 null layer) -> head_rows.parquet, head_attn_final.npz, head_positions.parquet (classes str / ment_filled / ment_other from pair columns mention_a_pos / mention_b_pos JSON if present)
  Memory: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; --wf-chunk 2048 is the default in sweep/expert/heads/joint (a Mixtral sweep with 2632 prefill rows and 50k spawn rows OOMed at wf_chunk 8192).
- 2026-10-04T07:48Z [ext7-controls] competence scans done (GPU ≈ 5 min, results/wino_role_{qwen3,mixtral_bos}, results/ioi_{qwen3,mixtral_bos}). Role swap, margin both ways: Qwen3 558 / 632 role pairs (306 twins), Mixtral BOS 270 / 294 (146 twins), mean drop 7.46 / 6.54; case sets data/wino_role/case_sets_<proto>.json (seed 0 over twins): Qwen3 128/128 + replication 128/128, Mixtral 128/128 (no replication). IOI: margin both ways for S2 -> IO in 1,593 / 1,597 of 1,600 items (Qwen3 / Mixtral BOS), clean margin 1,597 / 1,599; Δ clean +6.27 / +5.55, Δ s2io −6.26 / −5.54, Δ s1io −1.65 / −1.68; 1,590 items pass both models -> data/ioi/case_sets_{s2io,s1io}.json (same 128/128 + 128/128 items for both corruptions and both models; s1io d = 0 only) and *_grid.json (first 64 validation items / pairs for W3).

## 2026-10-04T08:00Z [ext8-addback] ENGINE READY (ext8)
- moetrace/engine.py now = verified dev copy (md5 1603112687e707a0d68caf93b6736f1c). Regression: results/verify_olmoe.json and
  results/verify_ext5_engine_olmoe.json re-run, identical to *_before_ext8.json in all non-timing fields (23 / 225; scripts/ext8_regress_compare.py).
- New `multi` step kinds (SpawnSpec(kind="multi", steps=((layer, kind, experts), ...)), layers strictly increasing, layer == steps[0][0]):
    (l, "attn_layer", None)  later step: live row's final-position attention output := source's (h_mid = h_in_own + Attn_src), own MoE runs on h_mid
    (l, "block", None)       later step: attention AND MoE output := source's (h_out = h_mid + MoE_src); as first step = single-layer kind
    e.g. all attention:  SpawnSpec(0, parent, src, "multi", steps=tuple((l, "attn_layer", None) for l in range(L)))
         all MoE:        steps=tuple((l, "layer", None) for l in range(L));  both: steps=tuple((l, "block", None) for l in range(L))
         expert set S:   steps=tuple((l, "coalition_set", tuple(S_l)) for l in sorted layers of S)
- Noising direction = parent: clean prefill row, clean (=source) field: corrupted row (works for layer / attn_layer / block / coalition_set / multi);
  set PrefillSpec.clean_ref on prefill rows and the new SpawnSpec.kl_ref (default -1 = SpawnSpec.clean) for metrics in that direction.
- New DiagSpec.contrib_dla: per-expert DLA of prefill rows' final-position contributions [L, B, k] + final_rms [B].
- Verification results/verify_ext8_engine_olmoe.json (12 STR units: 6 CounterFact donors + 3 WinoGrande pairs both ways; 21 configs x 2 directions
  vs transformers hooks): |dDelta| max 1.05 (single-layer kinds) / 0.63 (multi-step kinds), mean 0.085, 92% within 0.25, effect r 0.9994;
  all-layer block == source Delta bit for bit (both directions, engine and HF); single-step multi attn_layer/block == single kinds exactly;
  stress 20k all-layer multi rows in one pass, 7.1 GB peak on OLMoE.
- STRUCTURAL NOTE for W4: at the final position the MoE is a per-token function, so attn_layer at EVERY layer restores the source run's final residual
  exactly when the final token is shared (CounterFact STR, WinoGrande, IOI): A = 1 by construction (all-attention == all-block, verified), hence the
  two-player Shapley split degenerates to phi_MoE = M/2, phi_attn = 1 - M/2, redundancy = M. Only M (all MoE outputs) is informative. A non-degenerate
  attention/MoE split of the final residual difference is the direct-path split: Delta(h_corrupt + sum_l dAttn_l) vs Delta(h_clean - sum_l dAttn_l)
  (exact final norm, from DiagSpec(attn_out_final=True, resid_final=True); code: direct_split() in scripts/ext8_addback_run.py).
- 2026-10-04T08:23Z [ext8-addback] OLMoE smoke of the add-back driver complete (results/olmoe_addback_src, results/olmoe_addback_smoke: 14 passes, 30k rows/pass, 6.9 GB peak); finding: subsets of experts overshoot the all-MoE ceiling (oracle max 0.48 vs ceiling 0.29 of the drop), so the greedy 95%-ceiling stop is off (15 steps, k@95% reported). Launched scripts/ext8_addback_chain.sh cf (logs/ext8_addback_chain_cf.log): Qwen3 then Mixtral BOS CounterFact STR, 5 passes per queue job
- 2026-10-04T08:25Z [ext7-controls] chains queued (all via gpu_queue.sh): scripts/ext7_role_ioi_chain.sh (W2 sweeps wino_role_<proto>_str, ioi_<proto>_{s2io,s1io}; W5 heads on IOI; W3 grids at named positions with the new scripts/ext7_{role,ioi}_grid.py = ext7-wino's grid executor restricted to mention / filled-option / S1 / IO / S2+ positions), scripts/ext7_role_ioi_w4_chain.sh (W4 joint, scripts/ext7_wino_joint.py), scripts/ext7_role_ioi_direct_chain.sh (direct-path split, python -m moetrace.ext7_controls direct; OLMoE smoke: fp32 vs engine Δ max 0.056). IOI attention (results/ioi_<proto>/attn_names.npz): name-mover-like heads with clean attention to IO that moves to S1 under S2 -> IO: Qwen3 L42H10 (0.86 -> S1 0.87), L44H15 (0.58), Mixtral L28H0 (0.73), L19H8 (0.67).
- 2026-10-04T08:25Z [ext7-controls] W7 Qwen3 role swap W2 (results/wino_role_qwen3_str, 128/128 pairs + replication): MoE L42 +1.22 [1.06, 1.37] (0.175 of the drop), attention L38 +0.91 [0.73, 1.09] (0.131), block L42 (0.217); attention share of the positive rescue 0.32 [0.29, 0.35] (replication 0.30) vs option swap 0.16 [0.13, 0.20] (ext7-wino run) and CounterFact 0.54.
- 2026-10-04T08:26:01Z [ext7-wino] EXT7 RUNNER READY (W4; smoke-tested on OLMoE with the ext8 engine): same common args as the sweep, needs the run's sweep (case_sets.json) only.
  W4 joint: python scripts/ext7_wino_joint.py <common> [--families main,rep] [--max-rows 30000] [--tail-layers K] -> joint_rows.parquet (direction denoise|noise, kind all_moe|all_attn|all_block, span, effect = restored (denoise) / destroyed (noise) Delta, drop). Checks on OLMoE: all_block and all_attn effect/drop = 1 exactly (structural, see ext8 note).
  W4 direct split: python scripts/ext7_wino_dla.py <common> [--families main,rep] [--chunk 512] (prefill-only pass, DiagSpec attn_out_final + resid_final) -> dla_rows.parquet (per layer DLA of attention / MoE outputs, clean and corrupted), dla_cases.parquet (den_attn = Delta(h_corrupt + sum dAttn) - Delta(h_corrupt), den_moe, noi_attn, noi_moe with the exact final norm; recon check 0).
  NOTE for pair data used both ways: the noising effect of direction d equals the denoising effect of direction 1-d EXACTLY (same intervention on the same prompt, metric sign-flipped), so pair means of noising and denoising coincide; report one.
- 2026-10-04T08:29:59Z [ext7-wino] EXPERT ROWS READY (ext7-wino): results/wino_qwen3_str/str_expert_rows.parquet, results/wino_mixtral_bos_str/str_expert_rows.parquet (schema of results/qwen3_str/str_expert_rows.parquet; case_id = directed id 2*pair_idx+d, slot 0; families in <run>/case_sets.json: main = 128/128 pairs, rep = 128/128 pairs; sweep_cases.parquet has delta_clean/delta_corrupt/drop; routing in str_sweep_routing.parquet)
- 2026-10-04T08:31:33Z [ext7-wino] W2 + W6 done (results/wino_qwen3_str, results/wino_mixtral_bos_str; main = 128/128 pairs, rep 128/128, own pool 128/128; directed cases both ways). Final-position single-layer patches, validation pairs, rescue / drop: Qwen3 MoE L41 0.218 [0.200, 0.238], attention max L38 0.009 (val argmax L42 0.03), block L41 0.238; attention share of the positive rescue (AUC+) 0.16 [0.13, 0.20] (rep 0.19, own pool 0.23; CounterFact STR from ext7-controls' run, same definition: 0.54). Mixtral BOS MoE L20 0.172 [0.162, 0.183], attention L13 0.141 and L19 0.114, block L19 0.251; share 0.34 (rep 0.35, own 0.37; CounterFact 0.58). Experts (two-stage, recurrence gate 128/256 directed): Qwen3 L41E117 val rescue +1.00 [0.84, 1.17], Spec +0.90 [0.74, 1.07], gate-matched equal-norm Spec +0.45 [0.37, 0.53], top-1 among active in 178/223, replicated on rep (L41E117, Spec +0.69); second locus L43E081 (Spec +0.61). Mixtral L20E000 rescue +1.11, Spec +0.94 [0.84, 1.04], equal-norm Spec +0.16 [0.13, 0.19], replicated; L19E006 Spec +0.84. Pattern A in both. CounterFact experts (L44E069, L42E115; L19E002, L21E001, L18E001) almost never routed at the WinoGrande final position (3-12 of 256 directed cases) and rescue ~0. Person-name pairs (23 of 256) are not carried by these experts (Qwen3 E117 Spec -0.06, Mixtral E000 +0.19). Remaining: W3 grid, W5 heads, W4 joint + direct split (chains 2, 3 running).
- 2026-10-04T08:38Z [ext8-addback] CounterFact Qwen3 add-back running (pass 0: 45k rows, 59 s, 7.3 GB peak; mean all-MoE rescue +6.2 vs drop ~12.2). EXPERT ROWS READY seen: launched scripts/ext8_addback_chain.sh wino (logs/ext8_addback_chain_wino.log; Qwen3 then Mixtral BOS; validation = main family 128 pairs / 256 directed cases; exact / noising singles / Shapley on subsets: Qwen3 128 / 128 / 64 rows, Mixtral 128 / 128 / 128; beam Qwen3 128, Mixtral 64)
- 2026-10-04T09:03Z [ext8-addback] preliminary A0 (pass 0 final; validation, ratio of means): Qwen3 CounterFact STR all-MoE M = 0.531 [0.489, 0.570] of the drop, direct-path split attention 0.50 / MoE 0.51; Qwen3 WinoGrande STR M = 0.843 [0.827, 0.859], direct split attention 0.045 / MoE 0.956 (all-attention = 1 by construction in both). Expert curves (partial): CounterFact oracle k80%-of-ceiling = 6 experts, WinoGrande 48 (more distributed, higher ceiling).
- 2026-10-04T09:15Z [ext7-controls] W2 / W4 so far (validation; attention share of the positive rescue, AUC+): IOI (i) S2 -> IO Qwen3 0.92 [0.90, 0.94] (attention L45 0.329 of the drop, MoE peak 0.017; the L45 MoE patch is -1.93), Mixtral BOS 0.80 [0.79, 0.81] (attention L19 0.204, MoE L17 0.047); IOI (ii) Qwen3 0.89 (attention L45 0.606); WinoGrande role swap Qwen3 0.32 / Mixtral 0.39 vs option swap (ext7-wino runs) 0.16 / 0.34; CounterFact STR 0.54 / 0.58. W4 joint at the final position is degenerate (all-attention A = 1.00 because the final token is shared and the MoE is per-token; M informative): all-MoE M role Qwen3 +0.77, Mixtral +0.70; IOI (i) Qwen3 -0.27, (ii) -0.24 (clean MoE writes at the final position push toward S). Direct-path split (exact norm, prefill): role A_dir / M_dir Qwen3 0.19 / 0.81, Mixtral 0.38 / 0.62. Remaining queued: Mixtral IOI (ii) W2, W5 heads x4, W3 grids x6, W4 x2, direct x4.
- 2026-10-04T09:14Z [coordinator] W4 redefined (joint all-attention patch at the final position is degenerate, A = 1): report A as sanity, M denoise/noise, direct-path split (ext7_controls.direct_split_pairs); relayed to ext7-wino. RESEARCH_PLAN.md W4 updated.
- 2026-10-04T09:14Z [ext8-addback] launched scripts/ext8_addback_chain_mixtral.sh (logs/ext8_addback_chain_mixtral.log): Mixtral BOS CounterFact then WinoGrande add-back in parallel with the cf / wino chains (jobs serialised by the GPU lock; state reloaded per job)
- 2026-10-04T09:18:48Z [ext7-wino] W4 switched to the revised form (coordinator): A only as sanity, M denoise + M noise / drop, direct split via moetrace/ext7_controls.direct_split_pairs (read-only, CLI 'python -m moetrace.ext7_controls direct', chain scripts/ext7_wino_chain4.sh -> results/wino_{qwen3,mixtral_bos}_str/direct_split.parquet, families main + rep), DLA shares; written under key 'w4' of results/ext7_wino_summary.json (main_validation, rep_validation, main_all, rep_all, strata). Joint rows done for both models (results/wino_*_str/joint_rows.parquet): Qwen3 M = 0.835 of the drop (main+rep), all-attention 0.994-0.998 (sanity), all-block 1.000. Note: for symmetric pairs used both ways noise(d) == denoise(1-d) exactly, so the pair-level M_noise equals M_denoise.
- 2026-10-04T10:10Z [ext7-controls] W5 IOI (i) Qwen3 heads (results/ioi_qwen3_s2io/head_rows.parquet; layers auto = 32, 39, 42, 43, 45): ≥ 2 SD on validation and discovery: L42H11 +2.25 (67 % of the L42 attention patch; attends S2 0.75 = reads the duplicated name, S-inhibition-like), L43H24 +1.27 (pos0 / other), L42H10 +0.87 (name mover: IO 0.87 clean -> S1 0.86 after S2 -> IO), L43H29 +0.80 (attends the final token 0.88), L43H28 +0.63; negative L42H14 -0.52 (IO 0.21 -> S1 0.21: negative name mover), L43H27 -0.47. L45 (the attention-peak layer) is distributed (13 heads for 80 %). W4 / direct: Qwen3 option swap direct A_dir / M_dir 0.05 / 0.95 (ext7-wino ran the direct split on wino_qwen3_str with this module), role 0.19 / 0.81, IOI (i) 1.40 / -0.40.
- 2026-10-04T10:26:56Z [ext7-wino] correction to my 08:23Z note: noise(d) == denoise(1-d) holds in exact arithmetic (identical on OLMoE), but on Qwen3 / Mixtral the per-directed-case values differ by bf16 recomputation noise amplified by routing flips (max 1.9 / 0.5 logits); the pair-level population ratios agree to <= 0.002. W4 shared direct split done: Qwen3 A_direct 0.047 [0.018, 0.075] / M_direct 0.953, Mixtral 0.288 [0.265, 0.313] / 0.712 (main validation); M (all MoE, denoise) 0.845 / 0.788; all-attention sanity 0.992 / 1.000; numbers under key 'w4' of results/ext7_wino_summary.json; direct_split.parquet in results/wino_{qwen3,mixtral_bos}_str/.
- 2026-10-04T10:39:52Z [ext7-wino] W3 (window 1) + W5 done. W3: the option's identity leaves the option token and reaches the final token directly (first subsequent token <= 0.12 of the drop): final-position residual restoration crosses 0.1/0.5/0.9 at L19/L30/L41 (Qwen3, gradual) and L10/L13/L23 (Mixtral BOS, a discrete step at L13); MoE at the option token = token identity at L0 (Z9). W5 (top-4 W2 attention layers + null layer, 32 heads each): Qwen3 heads L38H18 +0.34, L38H21 +0.29 vs L38H16 -0.55 (same GQA group 16-23) cancel within L38 (layer +0.10); Mixtral L25H9 +0.55, L19H13 +0.36, L13H18/H11/H4 ~+0.19; the Mixtral heads attend 0.2-0.5 to the option and shift attention to the first mention of the candidate the option names (L19H13 0.12 vs 0.03) - coreference-like; head patches are not additive (per-pair r 0.16-0.55). Only the Mixtral window-5 grid is still queued. Section draft results/sections/ext7_wino.md (scripts/ext7_wino_text.py).

## 2026-10-04T11:15Z [ext8-addback] A0-A3 COMPLETE (CounterFact STR + WinoGrande STR; Qwen3, Mixtral BOS)
- Runs (all complete): results/qwen3_str_addback (15 passes), mixtral_bos_str_addback (14), wino_qwen3_str_addback (14),
  wino_mixtral_bos_str_addback (14). GPU for all ext8 jobs ~82 min (add-back 71, verification/regression 7, OLMoE smoke 4).
- A0 (validation, fraction of drop): all-MoE M = CF Qwen3 0.531 [0.489,0.570], CF Mixtral 0.409 [0.367,0.451], Wino Qwen3 0.843
  [0.827,0.859], Wino Mixtral 0.787 [0.764,0.809]; all-attention = 1 by construction; direct-path split attn/MoE 0.50/0.51, 0.63/0.36,
  0.04/0.96, 0.29/0.71 (W4 keys w4_counterfact / w4_winogrande in results/ext8_addback_summary.json).
- A1: k for 80 % of the ceiling (oracle / greedy / pop / layerwise / random): CF Qwen3 6/5/24/48/320 of 384; CF Mixtral 5/4/8/9/64 of 64;
  Wino Qwen3 48/10/48/64/320; Wino Mixtral 8/7/12/16/48. Curves overshoot the ceiling (oracle max 0.61/0.52/0.85/0.81) -> greedy
  95%-ceiling stop not used. DLA ordering (patch-free) best or within 0.03 of the oracle (AUC log k 0.51/0.37/0.65/0.55).
- A2: beam - greedy <= 0.016 of drop, exact top-10 optimum - greedy10 <= 0.011 (noise level); greedy beats static oracle by +0.04..+0.12 at k=10.
- A3 deletion mirrors add-back. Section results/sections/ext8_addback.md; tables results/tables/ext8_*; figures results/figures/ext8_*.
- Note: run dirs hold ~100 MB of row parquet (addback_rows_p*.parquet); smoke dirs results/olmoe_addback_{src,smoke} (15 MB) are testing only.
- scripts/ext8_addback_chain_mixtral.sh still has one queued no-op job (state complete; it exits immediately when it gets the lock).
## 2026-10-04T11:08:42Z [ext7-wino] DONE (W1-W6, W4 revised form, strata)
- Section results/sections/ext7_wino.md (scripts/ext7_wino_text.py), tables results/tables/ext7_wino_* (17), figures results/figures/ext7_wino_{w2_curves,w3_grid_*,w4_dla,w5_heads}, numbers results/ext7_wino_summary.json (key "w4" in the revised form). GPU: ~55 min big-model passes + ~15 min OLMoE verification/smoke.
- Headlines (validation pairs): final-position MoE peak Qwen3 L41 0.218 of the drop, Mixtral BOS L20 0.172 (CounterFact STR 0.177 / 0.086); attention share (AUC+) 0.16 / 0.34 (CounterFact 0.54 / 0.58); all-MoE M 0.845 / 0.788; direct paths MoE 0.953 / 0.712, attention 0.047 / 0.288; experts L41E117 (Spec +0.90, equal-norm +0.45) and L20E000 (Spec +0.94, equal-norm +0.16), pattern A, replicated; CounterFact selections idle on WinoGrande. W3 hand-off: final residual 0.5 at L30 (Qwen3, gradual) / L13 (Mixtral, discrete step). W5: Qwen3 L38 heads cancel (H18/H21 vs H16, one GQA group); Mixtral L25H9, L19H13, L13 heads with coreference-like attention shift to the antecedent named by the option. Name pairs are the attention-leaning stratum.
- Open: Mixtral without BOS not run (decision (e)); ext5_subject.run_subject window/attn_layer mixing bug reported (not fixed, file not mine).
- 2026-10-04T11:20Z [ext7-controls] DONE. All GPU work finished (≈ 83 GPU-min in ext7c-* jobs: task-1 sweeps 9, scans + IOI attention 11, W2 15, W5 9, W3 grids 21, W4 9, direct split 9). Section results/sections/ext7_controls.md (python -m moetrace.ext7_controls section), tables results/tables/ext7_controls_* (incl. three_task), figures results/figures/ext7_controls_*, numbers results/ext7_controls_summary.json. Three-task result (validation, Qwen3 / Mixtral BOS): attention share of the positive final-position rescue IOI (i) 0.92 / 0.80, IOI (ii) 0.89 / 0.82, CounterFact STR 0.54 / 0.58, WinoGrande role swap 0.32 / 0.39, option swap 0.16 / 0.34; W4 all-MoE M IOI -0.27 / -0.02, CounterFact 0.53 / 0.41 (ext8), role 0.77 / 0.70, option 0.84 / 0.79 (A = 1 by construction at the final position); direct-path attention share 1.40 / 1.17, 0.50 / 0.63, 0.19 / 0.38, 0.05 / 0.29. WinoGrande is the most MoE-heavy task, IOI attention-driven (S2-reader heads Qwen3 L42H11 / Mixtral L21H6 under S2 -> IO, name movers Qwen3 L42H10 + L45 heads / Mixtral L19H8, L30H2 under S1, IO -> names). Remaining: scripts/ext7_role_ioi_chain.sh still steps through its six grid jobs, which are no-ops now (grids done by the single job ext7c-w3-all).
- 2026-10-04T11:15Z [coordinator] Phase 3 COMPLETE. Reviewed ext8-addback, ext7-wino, ext7-controls reports and sections; fixed ext5_subject window/attn_layer bug (regression identical: verify_ext5_subject_olmoe.json, verify_ext6_str_grid_olmoe.json vs *_before_ext7fix.json); wrote results/sections/ext7_synthesis.md; build_extensions_report.py + EXTENSIONS_REPORT.md rebuilt (Directions 7-8, 7, 7b, 8); CLAUDE.md and RESEARCH_PLAN.md updated. Nothing committed.
- 2026-10-05T05:58Z [coordinator] Phase 3 committed and pushed (81377e4; wording fix: layer-first ordering below every effect-based ranking; k ≤ 15 partial AUC table scripts/ext8_partial_auc.py). Phase 4 planned (RESEARCH_PLAN.md "Phase 4": items 1-4 = Directions 9 knockout, 10 head+expert circuits, 11 writer experts, 12 completeness); launching ext9-knockout, ext10-circuit, ext11-writer, ext12-complete.
- 2026-10-05T06:04Z [ext11-writer] started Direction 11 (writer vs computer experts): Part A (CPU, ext8 add-back DLA vs ext6/ext7 single-expert patches) first, then Part C routing diagnostics (GPU, existing DiagSpec), Part B after ENGINE READY (ext9)
- 2026-10-05T06:10Z [ext10-circuit] started (Phase 4 Direction 10, head + expert add-back to full repair). Step 1 = single-head patches at every (layer, head) on the ext8 validation rows + patch-free head DLA; Step 2 (greedy over heads + experts) after the ext9 engine. Status file logs/agent_status_ext10-circuit.md.
- 2026-10-05T06:06:47Z [ext12-complete] started Phase 4 item 4 (Direction 12: 4a add-back replication / CF fold swap, 4b role vs option swap on identical items, 4c Mixtral no BOS on WinoGrande); status logs/agent_status_ext12-complete.md
- 2026-10-05T06:13Z [ext11-writer] Part A done (CPU; results/ext11_writer_summary.json key A, tables ext11_A_*, figures ext11_A_*). Launched scripts/ext11_chain_routing.sh (logs/ext11_chain_routing.log): Part C prefill-only routing diagnostics, 3 passes per model via gpu_queue (results/{qwen3,mixtral_bos}_writer_routing, raw /opt/dlami/nvme/moe_ext11)
- 2026-10-05T06:20Z [ext10-circuit] Step 1 chain launched (scripts/ext10_chain1.sh, logs/ext10_chain1.log; gpu_queue jobs ext10-heads-<run>): single-head attn_head patches at every (layer, head) + attn_layer reference per layer on the ext8 validation rows (cf_qwen3 432 rows / 3 passes, wino_qwen3 256 / 2, cf_mixtral 436 / 4, wino_mixtral 256 / 2), prefill diag attn_heads_final (all layers) + contrib_dla -> head DLA / expert DLA; outputs results/{qwen3_str,wino_qwen3_str,mixtral_bos_str,wino_mixtral_bos_str}_circuit/. Smoke (6 rows Qwen3) ran 61 s; the analysis path was tested offline.
- 2026-10-05T06:20:35Z [ext12-complete] chains launched (all GPU via gpu_queue.sh): scripts/ext12_chain_inputs.sh (logs/ext12_chain_inputs.log: 4c Mixtral no-BOS WinoGrande sweep/expert/joint/direct/sink flags -> results/wino_mixtral_nobos_str; 4b option swap on the role-swap items -> results/wino_roleitems_{qwen3,mixtral_bos}_str, case sets results/wino_roleitems_inputs/ from scripts/ext12_roleitems_casesets.py), scripts/ext12_chain_addback.sh (logs/ext12_chain_addback.log: multi-task add-back driver scripts/ext12_addback_run.py, one engine pass serves several tasks of one model; Qwen3 = WinoGrande rep split + CounterFact fold swap, Mixtral = WinoGrande rep BOS + CounterFact fold swap BOS + WinoGrande no BOS). No edits to ext7/ext8 scripts or read-only modules: ext8_addback_run.dla_table/direct_split and ext8_addback_analyze.analyse are imported unchanged. 4c W2 (no BOS, main validation): MoE L20 0.179 of the drop, attention L13 0.145, block L19 0.253, attention share 0.349 vs BOS 0.344 (paired diff +0.005 [-0.002, +0.013]).
- 2026-10-05T06:26Z [ext11-writer] Part C Qwen3 pass done (3 chunks, 176 s GPU; results/qwen3_writer_routing). L41E117 routed at the final position of 84 % of WinoGrande prompts and 86 % of the context-free local prompts (DLA there +0.08 vs +0.54 logits), 0 % of IOI, 11 % of CounterFact; in wikitext 7 % of tokens, 66 % of copulas, 59 % of degree adverbs, 77 % of copula/degree tokens followed by a WinoGrande trigger word. L44E069 / L42E115: broad experts (45 % / 40 % of wikitext tokens), 94-95 % before CounterFact place names, 100 % at ' in' + place; DLA largest on place relations (E069 +1.08). Mixtral pass queued.
- 2026-10-05T06:43:41Z [ext12-complete] 4c W6 (Mixtral no BOS, WinoGrande 776-pair set, main): two-stage selection L20 -> L20E000 again (val rescue +1.12 [1.02, 1.23], Spec +0.95 [0.84, 1.06], equal-norm Spec +0.154 [0.130, 0.178]; BOS: +1.11 / +0.94 / +0.160), pattern A, re-selected on rep; joint top-3 L20E000, L19E006, L21E006 as with BOS. 4b (Qwen3, W2 only so far): on the 282 identical twins the option swap's attention share equals the role swap's (0.306 vs 0.304, paired diff -0.002 [-0.015, +0.009]): the Phase-3 gap (0.32 vs 0.16) was an item confound (name items). Joint / direct pending.
- 2026-10-05T06:52Z [ext11-writer] Parts A and C DONE (GPU 8.0 min: Qwen3 176 s, Mixtral BOS 303 s). Section draft results/sections/ext11_writer.md (scripts/ext11_writer_text.py), numbers results/ext11_writer_summary.json keys A, C, tables results/tables/ext11_{A,C}_*, figures results/figures/ext11_A_*. Direct share (validation, sum DLA / sum single-patch rescue where routed): Qwen3 L41E117 1.04 [0.99, 1.10], L42E115 0.96 [0.88, 1.06], L44E069 1.32 [1.20, 1.43]; Mixtral L20E000 0.56 [0.52, 0.62], L19E002 0.71, L21E001 0.87, L18E001 0.54; share rises with depth (crossover Qwen3 L41/L42, Mixtral WG L22). Routing is local (L41E117 at 86 % of context-free local prompts, 66 % of wikitext copulas), the write is contextual (DLA +0.58 full vs +0.09 local). Part B (vocabulary projections) scripted (scripts/ext11_writer_vocab.py, ..._vocab_analyze.py), waiting for the ext9 engine (E4c).
- 2026-10-05T06:58:29Z [ext9-knockout] ENGINE READY (ext9): moetrace/engine.py (md5 0f236b89) = E4a PrefillSpec.route_mask=((layer, expert), ...) + route_mask_pos "all"|"final" + route_mask_mode "reroute"|"zero" (expert knockout on prefill rows; recorded routing = masked routing; spawns on masked parents raise), E4b multi steps (layer, "attn_head", heads) / (layer, "heads_experts", (heads, experts)) as first and later steps in both directions (one-head step == attn_head kind bit for bit), E4c DiagSpec.contrib_final_vectors=(layers) -> extra["diag"]["contrib_final_vectors"][l] fp32 [B, k, H] + ["moe_out_final_fp32"][l]. Regression after the merge: verify_olmoe (23 fields) / verify_ext5_engine (225) / verify_ext8_engine (496) identical to *_before_ext9.json in every result field (scripts/ext9_regress_compare.py; only the GPU peak-memory fields of verify_ext8 differ by <1 MB, which is run-to-run nondeterministic with an unchanged engine: two 2026-10-04 runs with byte-identical code gave 2.010271 vs 2.010057 GB). OLMoE vs HF hooks (dev smoke): route masks |dDelta| mean 0.079 (baseline 0.063), all invariants 1.0; mixed head/expert steps vs HF mean 0.071, effect r 0.9993. API: logs/ext9_engine_api.md; full verification -> results/verify_ext9_engine_olmoe.json (queued).
- 2026-10-05T07:00Z [ext11-writer] after the ext9 engine (md5 0f236b89): launched scripts/ext11_chain_vocab.sh (logs/ext11_chain_vocab.log), Part B vocabulary projections, one prefill pass per model via gpu_queue (results/{qwen3,mixtral_bos}_writer_vocab)
- 2026-10-05T07:05Z [ext10-circuit] Step 1 Qwen3 done (cf_qwen3 3 passes, wino_qwen3 2 passes; peak 17-19 GB at 203-233k spawn rows); Mixtral running. First numbers (validation, rescue / drop): CounterFact STR top head L40H13 +0.192 [0.169, 0.216] (z +28.6; = the F2 GN mover, GN 0.171), then L40H15, L40H14, L43H11; WinoGrande STR L46H24 +0.101 vs L46H25 -0.105 (cancelling pair, layer not scanned in W5), L41H27 +0.065, W5 heads L38H18 / L38H21 / L38H16(-) recovered (row-level r 0.64 vs W5 rows, median |diff| 0.125 = 1 bf16 step). Sum of head DLA = ext8 attention DLA (ratio 0.999 / 1.03). Step-2 smoke (Qwen3, 4 rows per task, 2 passes) queued with the ext9 engine.
- 2026-10-05T07:11:33Z [ext9-knockout] E4 full verification on the merged engine: results/verify_ext9_engine_olmoe.json (route masks vs masked HF router mean |dDelta| 0.079 vs unmasked baseline 0.063, effect r 0.95-0.98, all invariants 1.0; head steps exact identities, mixed head/expert configurations vs HF mean 0.079, effect r 0.9992; contrib vectors = fp32 MoE output to 8e-8). Knockout chain scripts/ext9_chain.sh running (logs/ext9_chain.log).
- 2026-10-05T07:12:32Z [ext12-complete] 4a Qwen3 DONE (results/wino_qwen3_str_addback_rep, results/qwen3_str_addback_fold_rep; 14 shared passes): WinoGrande rep_validation ceiling 0.833 [0.815, 0.850] (main 0.843), greedy k80 12 [10, 14] (main 10), oracle 48 (48), random 320 (320), AUC log k oracle 0.581 / DLA 0.629 (main 0.591 / 0.647), greedy - static oracle at k=10 +0.104 [0.090, 0.117] (main +0.124), DLA - oracle AUC +0.048 (main +0.057); CounterFact fold swap ceiling 0.524 [0.494, 0.555] (main 0.531), k80 oracle / greedy 7 / 5 (6 / 5), DLA - oracle AUC +0.033 (+0.034), greedy - oracle r(10) +0.056 (+0.057); every difference CI contains 0. 4b Qwen3 complete: on 282 identical twins role vs option attention share 0.304 vs 0.306 (diff -0.002 [-0.015, +0.009]), A_dir 0.181 vs 0.183, all-MoE M 0.776 vs 0.778. Structural reason: for every role item the role-swapped corrupted prompt IS the option-swapped corrupted prompt with the two names exchanged (text check 512/512), so W7 on name items is the option swap up to a renaming; the Phase-3 role/option gap (0.32 vs 0.16) is the name-vs-object item difference (names stratum of the 776 set: option share 0.30).
- 2026-10-05T07:20Z [ext10-circuit] Step-2 smoke OK (results/qwen3_circuit_smoke, Qwen3, 4 rows per task, 2 passes, ext9 engine): all heads at all layers vs clean Delta |dev| max 0.31 (mean 0.00; o_proj bf16 rounding), all heads of one layer vs attn_layer median 0, max 0.375, r 0.995-0.999; greedy update logic and analysis tested. Launched scripts/ext10_chain2.sh (logs/ext10_chain2.log): IOI (i) S2->IO Qwen3 step 1 (attention-pole reference), then step 2 Qwen3 (CounterFact + WinoGrande + IOI rows in every pass, budget 100k rows), then step 2 Mixtral (CounterFact + WinoGrande), 5 passes per gpu_queue job; then analyses + section.
- 2026-10-05T07:31Z [ext10-circuit] Step 1 COMPLETE for the four main runs (GPU ~17 min: Qwen3 CF 3 passes / WG 2, Mixtral CF 4 / WG 3). Single-head rescue / drop, top head: Qwen3 CF L40H13 +0.19 [0.17, 0.22] (F2 GN mover), Mixtral CF L18H4 +0.15 [0.13, 0.18] and L24H22 +0.15 (F2 GN heads), Qwen3 WG L46H24 +0.10 (with L46H25 -0.10), Mixtral WG L25H9 +0.07 and L22H20 +0.07 (W5 heads all >= 2 SD); Z8 detects 12-18 positive / 5-12 negative heads per run; best single head vs best single expert per row (median) CF 0.19 / 0.19 (Qwen3), 0.21 / 0.12 (Mixtral), WG 0.15 / 0.20, 0.11 / 0.19; head DLA vs single-head patch: population r 0.96 / 0.92 / 0.87 / 0.79, top-1 agreement 0.71 / 0.44 / 0.41 / 0.47; sum of head DLA = ext8 attention DLA (ratio 0.999-1.03). Tables results/tables/ext10_heads_*, numbers results/ext10_circuit_summary.json (step1).
- 2026-10-05T07:43Z [ext10-circuit] chain2 restarted before any of its GPU work ran (it was waiting for the lock): step-2 jobs now carry 10 passes each (fewer lock waits under 4-agent contention; Qwen3 ~9 min, Mixtral ~17 min per job). logs/ext10_chain2.log.
- 2026-10-05T07:41:13Z [ext12-complete] Mixtral add-back group running (3 tasks share each pass: ~60k spawn rows; pass 0 78.6k incl. forced A0 rows, peak 19.4 GB, allocator 'expandable_segments ... OOM' warnings but passes complete; pass ~95-108 s). Engine: ext9 merge at 06:58Z; my jobs before that (4c sweep/expert, Qwen3 4b sweep/joint, Qwen3 add-back jobs 1-2) ran on the ext8 engine, later ones on the ext9 engine (regression identical per ext9).
- 2026-10-05T07:48:52Z [ext12-complete] 4b Mixtral BOS W2 on the 139 identical twins: attention share role 0.406 vs option 0.414 (paired diff -0.008 [-0.014, -0.001]); renaming identity 256/256; per-case agreement of the two corruptions r 0.96 (MoE) / 0.90 (attention), drops r 0.97. Same conclusion as Qwen3: on identical (name) items the role and option swaps give the same decomposition.
- 2026-10-05T07:54Z [ext11-writer] DONE (Parts A, B, C). Section results/sections/ext11_writer.md (scripts/ext11_writer_text.py), numbers results/ext11_writer_summary.json (keys A, C, B), tables results/tables/ext11_{A,B,C}_*, figures results/figures/ext11_A_*, runs results/{qwen3,mixtral_bos}_writer_{routing,vocab}. GPU 12.0 min total (C 8.0, B 4.0). Part B: L41E117 delta_e ranks the right trigger at median 98 of 151,936 (top-100 in 50 %), L44E069 5,758, L42E115 1,287; Mixtral L20E000 c_e at median 10 of 32,000 (top-10 in 51 %), L21E001 c_e 7; single-expert exact-norm direct effect = 0.996-0.999 x frozen DLA (mean |diff| <= 0.013 logits); delta DLA reproduces ext8's DLA (r >= 0.992). E4c verification (results/verify_ext9_engine_olmoe.json contrib_final_vectors): sum vs MoE 6e-08, vs HF median rel 0.022.
- 2026-10-05T08:07Z [ext10-circuit] IOI (i) S2->IO Qwen3 step 1 done (results/ioi_qwen3_s2io_circuit, 256 validation directed cases, 2 passes): top single heads L42H11 +0.178 [0.171, 0.184] of the drop (W5 S2 reader), L43H24 +0.100, L40H2 +0.071, L42H10 +0.069 (name mover), negative L44H15 -0.122, L47H29 -0.088; 29 positive / 14 negative heads >= 2 SD; best single expert by DLA only 0.015 of the drop per row (no single-expert patch table for IOI). Qwen3 step 2 (CF + WG + IOI rows per pass) waiting for the GPU lock.
- 2026-10-05T08:38Z [ext10-circuit] Qwen3 step 2: 10 of 20 greedy passes done (results/qwen3_circuit; ~50-65 s per pass, peak 12.5 GB at 100k rows). Preliminary (validation, population r = rescue / drop): mixed heads+experts greedy r(10) CounterFact 0.88 / WinoGrande 0.80 / IOI 0.78 vs ext8 expert-only greedy r(10) 0.52 / 0.68 (all-MoE ceiling 0.53 / 0.86); 80 % of the drop at k = 8 / 10 / 11. Static mixed DLA ordering reaches 80 % at k = 12 / 24 / 16 (oracle 16 / 48 / 16). In-pass sanity: all heads at all layers / clean Delta ratio 0.999-1.006 (row max |dev| 1.9-2.6 logits, median 0.125 = bf16 batch noise, same as all-attention 1.5), all heads of a layer vs attn_layer r 0.996-0.999.
- 2026-10-05T08:54:50Z [ext12-complete] 4a / 4c add-back COMPLETE (Mixtral group 15 shared passes: results/wino_mixtral_bos_str_addback_rep, results/mixtral_bos_str_addback_fold_rep, results/wino_mixtral_nobos_str_addback). Mixtral WinoGrande rep_validation ceiling 0.765 [0.744, 0.784] (main 0.787), k80 oracle / greedy / random 8 / 7 / 48 (8 / 7 / 48), DLA - oracle AUC -0.023 (-0.027); Mixtral CounterFact fold swap ceiling 0.435 [0.397, 0.470] (0.409), k80 5 / 4 / 64 (5 / 4 / 64), greedy - oracle r(10) +0.089 (+0.078); all independent-sample differences to Phase 3 contain 0. No BOS vs BOS (paired, identical pairs): ceiling 0.795 vs 0.787 (+0.008 [+0.002, +0.014]), greedy k80 7 vs 7, oracle AUC +0.005. Tables results/tables/ext12_4a_*.
- 2026-10-05T09:10Z [ext10-circuit] Qwen3 step 2 COMPLETE (results/qwen3_circuit, 19 passes, ~19 GPU-min; CF + WG + IOI rows in every pass). Validation, population r(k) = rescue / drop, mixed heads+experts greedy: CounterFact r(20) 0.97 [0.95, 1.00] (k for 50/80/90 % of the drop 3/8/12; experts alone cap at the all-MoE ceiling 0.53, ext8 greedy r(15) 0.55), WinoGrande 0.93 [0.91, 0.96] (4/10/17; all-MoE 0.86, ext8 0.72), IOI 0.94 [0.93, 0.96] (5/11/16; all-MoE -0.27); head-only greedy r(20) 0.86 / 0.58 / 0.94. Heads among the first 20 mixed picks 10.4 / 7.7 / 19.0; first picks L40H13/L40H15/L40H14 (CF), L41E117/L46H24/L43E081 (WG), L42H11/L43H24/L42H10 (IOI). Answer restored at k = 20 in 0.94 / 0.96 / 1.00 of cases. Mixtral step 2 queued.
- 2026-10-05T10:05:09Z [ext12-complete] 4b both models complete (direct split / all-MoE on the role items): Qwen3 A_dir role 0.181 vs option 0.183, M 0.776 vs 0.778; Mixtral A_dir 0.411 vs 0.410, M 0.668 vs 0.669 (all paired diffs <= 0.003). 4c W4 (no BOS vs BOS, identical main validation pairs): all-MoE M 0.795 vs 0.788 (+0.007 [+0.001, +0.013]), direct-path attention 0.288 vs 0.288 (-0.001 [-0.007, +0.006]). Only the sink-flag pass is left.
## 2026-10-05T10:25Z [ext10-circuit] Direction 10 COMPLETE (Step 1 + Step 2; head + expert add-back to full repair at the final position)
- Runs: step 1 results/{qwen3_str,mixtral_bos_str,wino_qwen3_str,wino_mixtral_bos_str,ioi_qwen3_s2io}_circuit (single heads at every (layer, head) + attn_layer refs + head / expert DLA); step 2 results/{qwen3,mixtral}_circuit (19 passes each; mixed greedy over top-32 heads U ext8 expert pool, head-only greedy, static mixed oracle / DLA orderings to k = 128, in-pass ceilings + sanity rows). GPU ~79 min (all ext10 queue jobs incl. smokes).
- Mixed heads+experts greedy, validation, r(20) of the drop: Qwen3 CF 0.97 [0.95, 1.00], Mixtral CF 1.00 [0.98, 1.03], Qwen3 WG 0.93 [0.91, 0.96], Mixtral WG 0.94 [0.93, 0.95], Qwen3 IOI 0.94 [0.93, 0.96]; k for 50/80/90 % of the drop 3/8/12, 4/9/13, 4/10/17, 4/10/16, IOI 5/11/16 (experts alone: ext8 greedy r(15) 0.55 / 0.56 / 0.72 / 0.79, all-MoE ceilings 0.53 / 0.41 / 0.86 / 0.79; heads alone r(20) 0.86 / 0.87 / 0.58 / 0.65 / 0.94). Answer restored at k = 20 in 0.94 / 0.99 / 0.96 / 1.00 / 1.00 of cases. Heads among the first 10 picks 6.2 / 6.4 / 3.6 / 2.8 / 9.9 (CF starts with mover heads L40H13 / L18H4 / L24H22, WG with the W6 experts L41E117 / L20E000 + L19E006, IOI heads only L42H11 / L43H24 / L42H10). AUC log k (1..20) greedy / static mixed oracle / static mixed DLA 0.64/0.57/0.57, 0.61/0.53/0.52, 0.59/0.50/0.50, 0.58/0.54/0.49.
- Section results/sections/ext10_circuit.md (scripts/ext10_circuit_text.py), tables results/tables/ext10_{heads,circuit}_*, figures results/figures/ext10_{heads,circuit}_*, numbers results/ext10_circuit_summary.json (step1, step2, step2_runs). Not done: Mixtral IOI (budget).
## 2026-10-05T10:37:07Z [ext12-complete] DONE (Phase 4 item 4 / Direction 12: completeness checks)
- Section results/sections/ext12_complete.md (scripts/ext12_complete_text.py from results/ext12_complete_summary.json, scripts/ext12_analyze.py); tables results/tables/ext12_*; figures results/figures/ext12_{4a_curves,4b_role_vs_option,4c_w2_nobos_vs_bos}. GPU ≈ 70 min in 19 ext12 jobs (add-back 41, 4c inputs 15, 4b 10, smoke 3).
- 4a: add-back replicates on the WinoGrande replication split and the CounterFact fold swap (ceilings Qwen3 0.833 / 0.524, Mixtral 0.765 / 0.435 vs Phase 3 0.843 / 0.531 / 0.787 / 0.409, all difference CIs contain 0; greedy k80 12 / 5 / 7 / 4 vs 10 / 5 / 7 / 4; DLA vs oracle, greedy > static, CF overshoot, greedy's first picks all as in Phase 3; layer-first < pop except Mixtral CF fold swap, tied).
- 4b: Phase-3 role-vs-option gap withdrawn: on identical twins attention share role vs option Qwen3 0.304 vs 0.306 (diff -0.002 [-0.015, +0.009]), Mixtral 0.406 vs 0.414 (-0.008 [-0.014, -0.001]); A_dir and M identical to <= 0.003. The role-swapped prompt equals the option-swapped prompt with the two names exchanged (all items), so W7 on name items is a renamed option swap.
- 4c: Mixtral without BOS on WinoGrande = BOS: L20E000 selected again (Spec +0.95 [0.84, 1.06], equal-norm +0.154), replicated; attention share 0.349 vs 0.344, direct-path attention 0.288 vs 0.288, all-MoE M 0.795 vs 0.788, add-back curves within 0.01; sink state at the final token in 2 / 1024 prompts (CounterFact: 63 / 256); L19E006 routed in 256/256 and positively specific (+0.86).
- 2026-10-05T10:37:12Z [ext12-complete] correction to the DONE line: GPU split = add-back 41, 4c inputs 17, 4b 10, smoke 3 min (total ≈ 70).
## 2026-10-05T10:38:43Z [ext9-knockout] DONE (Item 1 / Direction 9: expert knockout; engine E4)
- Engine E4 merged 06:58Z (ENGINE READY line above); results/verify_ext9_engine_olmoe.json; regression identical in every result field (scripts/ext9_regress_compare.py; ext8's peak-memory fields nondeterministic).
- Knockout (route mask, all positions, reroute; clean prompts; F = fraction of the clean margin lost, full scope): Qwen3 L41E117 WG +0.033 [0.028, 0.037] / CF -0.001; L44E069 CF +0.033 [0.026, 0.039] / WG +0.001; L42E115 CF +0.024 / WG -0.001; Mixtral BOS L20E000 WG +0.036 [0.033, 0.040] / CF -0.001; L18E001 CF +0.015; L19E002 / L21E001 CF +0.004 (barely necessary). Rank among same-layer experts on the own task: Qwen3 1/17 (all three targets), Mixtral L20E000 1/8, L18E001 1/8, L19E002 2/8, L21E001 3/8. Dissociation contrast L41E117 vs L44E069 +0.065 [0.057, 0.073]; L20E000 vs L21E001 +0.041 [0.036, 0.047]; top-10 sets +0.296 (Qwen3) / +0.133 (Mixtral).
- Population top-10 sets: Qwen3 WG 0.148 (pair acc 0.997 -> 0.944) / CF 0.144 on the own task, cross-task <= 0.004, frequency-matched random 10-sets ~0; Mixtral WG top-10 0.154 / CF top-10 0.038 (CF top-10 contains WG experts L19E006, L22E005 and costs WG 0.047). Wikitext NLL change single experts <= 0.008 nats/token; sets up to 0.065 (Mixtral WG top-10).
- Sufficiency vs necessity: knockout dDelta = 0.04-0.25 of the STR patch rescue in logits; zero mode ~ reroute; final-position-only knockout reproduces 0.91-0.96 of the all-position effect; damage confined to prompts whose final position routes to the expert. Noise floor (null, baseline in other passes): Qwen3 WG +0.008 / CF -0.003, Mixtral +0.001 / +0.000.
- Section results/sections/ext9_knockout.md (scripts/ext9_knockout_text.py), numbers results/ext9_knockout_summary.json, tables results/tables/ext9_knockout_*, figures results/figures/ext9_knockout_*; runs results/qwen3_knockout (70 MB), results/mixtral_bos_knockout (34 MB), smoke results/olmoe_knockout_smoke (testing only). GPU 105 min (engine 9, knockout 96: Qwen3 38, Mixtral 58).
- 2026-10-05T10:38Z [coordinator] Reviewed ext10-circuit, ext11-writer, ext12-complete reports and sections (spot checks vs summary JSONs OK). Direction 12 4b overturns the Phase-3 reading "role swap moves the balance towards attention": on identical name items role = option swap (Qwen3 0.304 vs 0.306); revised moetrace/ext7_controls.py reading/caveat text (section regenerated, only those lines changed), results/sections/ext7_synthesis.md, CLAUDE.md. Waiting for ext9-knockout.
- 2026-10-05T10:44Z [coordinator] Phase 4 COMPLETE. Reviewed ext9-knockout (engine E4 + knockout), ext10-circuit, ext11-writer, ext12-complete; added scripts/ext9_synthesis_selfrepair.py (knockout vs direct write on the same prompts: Qwen3 / Mixtral L20E000 lose 0.30-0.47 of the direct write, Mixtral L21E001 / L19E002 0.05-0.19; indirect L18E001 not compensated); wrote results/sections/ext9_synthesis.md; build_extensions_report.py SECTIONS 9-12; EXTENSIONS_REPORT.md rebuilt; CLAUDE.md, README.md, RESEARCH_PLAN.md updated. GPU ≈ 4.4 h. NOT committed.
- 2026-10-06T04:33Z [coordinator] Post-Phase-4 additions: simplified add-back / deletion figures (scripts/ext8_plot_simple.py → results/figures/ext8_a{1,3}_curves_simple.png; greedy/oracle/random and oracle/random), WinoGrande vs CounterFact final-position routing overlap (scripts/ext7_wg_cf_routing_overlap.py → results/tables/ext7_wg_cf_routing_overlap.md: cross-task shared-expert fraction Qwen3 0.10-0.15 vs within-task 0.36-0.59, chance 0.06; Mixtral 0.25-0.34 vs 0.40-0.67, chance 0.25), team brief docs/team_brief_winogrande_str.md. Committing Phase 4.
