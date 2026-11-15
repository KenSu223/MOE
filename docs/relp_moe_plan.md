# Phase 7: RelP -> MoE on SVA, and RelP vs Lu-style activation patching (plan and live status)

Owner: the coordinating Claude Code session (user instruction 2026-11-07: oversee every planned experiment end to end,
keep this file, CLAUDE.md and logs/PROGRESS.md current, check the runs periodically, deliver the final report).
Status legend: [x] done, [~] running / in progress, [ ] not started. Times UTC.

## Goal
1. RelP (docs/relp_moe_implementation_guide.md + docs/relp_sva_experiment_protocol.md, followed exactly) on SVA for
   Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1: neuron / expert-position / expert-all-positions rankings, held-out
   interventions (remove / keep / noise / denoise, zero and mean_q_routed baselines, matched random controls).
2. On the same SVA pairs, Lu et al.-style activation patching (STR version = Direction 6: pair A = clean, B = corrupt,
   metric F = logit(clean answer) - logit(patch answer)): layer x position grid, single-expert grid, Lu's two-stage
   selection (best layer by final-position rescue, recurrence gate, best expert, Spec). Compare the experts each method
   finds; classify every disagreement by cause.

## User decisions (2026-11-07)
- "Mistral" = Mixtral-8x7B-v0.1 (the MoE of the base reproduction). Mixtral keeps the tokenizer-default BOS (protocol rule 1),
  excluded from the scoring pool as a special token. Mixtral paper/simple discovery = 236 rows (--short-discovery).
- Test evaluation runs automatically after validation (plan locked at score time; nothing chosen on test).
- Ranking-stability check (re-score with another batch composition) is included.
- Comparison design (answer of 2026-11-07): same pairs, metric, position pool; directions matched
  (Lu denoising rescue <-> RelP endpoint B (q_A - q_B) . beta_B; noising drop <-> protocol RelP endpoint A);
  three comparisons: (a) same selection rule (RelP scores through Lu's two-stage rule), (b) each method's own rule,
  (c) all (layer, position, expert) units: rank correlation, top-1 / top-k agreement, sign agreement, prediction-vs-exact
  slope; selection on discovery, confirmation on validation, test never used for selection; group-bootstrap CIs;
  disagreement diagnosis order: tie within CI -> selection rule -> endpoint / direction -> linearisation (incl. patch with
  downstream routing frozen, attention pattern frozen) -> indirect effect / compensation (direct DLA vs total) ->
  neuron cancellation inside the expert.

## Stages
- [x] S0 preparation (2026-11-07): env .venv-relp, checkpoints at pinned revisions (41 shards sha256-verified), SVA data,
      qwen3_/mixtral_{smoke,paper,grouped}_v1 manifests, streamed engine moetrace/relp_moe_stream.py, runner
      scripts/relp_sva_moe_run.py, 33/33 tests, OLMoE fp32 cross-check vs FModel, real-checkpoint dev verify passed
      (Qwen3 2148/2148, Mixtral 1444/1444). Evidence: docs/relp_moe_stream_notes.md, results/relp_moe_dev/.
- [x] S1 RelP chain `scripts/relp_moe_chain.sh` (2026-11-07T23:57:52Z - 2026-11-08T03:15:43Z, 3 h 18 min; 20/20 runs, every verify passed, 0 failed jobs):
  - [x] smoke qwen3 (simple, nounpp; done 00:05Z, verify 2148/2148 both)    - [x] smoke mixtral (simple, nounpp; done 00:23Z)
  - [x] paper qwen3 (4 tasks + test; 00:23-00:59Z, all verify 2148/2148)     - [x] paper mixtral (4 tasks + test; done 01:49Z)
  - [x] grouped qwen3 (4 tasks + test; done 02:26Z)   - [x] grouped mixtral (4 tasks + test; done 03:15Z)
  Outputs results/relp_sva_moe/<model>/<profile>/<task>/{verify,score_zero,score_mean,validation,test}.
- [x] S2 ranking stability (done 04:41Z, 8/8 paper tasks): paper, both models, score with --score-chunk 100 (other batch composition) -> overlap /
      correlation of rankings vs S1 (est. 45 min).
- [x] S3 RelP endpoint B (done 04:03Z; 16/16 tasks; endpoint-A recompute bit-identical to the protocol score in all 16), (q_A - q_B) . beta_B, paper + grouped, both models (new module, est. 40 min).
- [x] S4 activation patching (new modules; the verified engine file is NOT modified while S1 runs):
  - [x] S4a implementation + tests: moetrace/relp_moe_patch.py (patch_pass, relp_backward_endpoint with per-row scores),
        scripts/relp_moe_compare_run.py (endpointB, patch), tests/test_relp_moe_patch.py (patch grid = brute force via the
        verified Override/out_patch paths, route-change counts exact, endpoint A = relp_backward, endpoint B = all-expert reference);
        analysis scripts/relp_moe_analyze.py (A/B/C, re-runnable); chain 2 scripts/relp_moe_patch_chain.sh (waits for S1)
  - [ ] S4b smoke timing run, then budget check against the estimate (report if clearly over)
  - [x] S4c layer x position grid (grouped done 05:38Z; paper grid is part of S4d), both directions, paper + grouped, both models (est. ~1 h)
  - [x] S4d full single-expert grid (05:38-06:53Z, 1 h 15 min; 8/8 paper tasks), both directions, paper, both models (est. 4-6 h)
  - [x] S4e Lu original scope (done 05:38Z) (final position, around the selected layer), grouped, both models (est. < 1 h)
  - [x] S4f diagnostics (06:53-07:40Z; 16/16 tasks, 143 units) + write-size probes (results/relp_moe_dev/probe_size_*.json) (relp_moe_patch.diagnose_pass: natural / frozen routing / frozen attention /
        both + direct DLA of final-position units; tests 9/9); to run on the disagreeing units once S5 identifies them
- [x] S5 analysis (scripts/relp_moe_analyze.py; results/relp_sva_moe/summary.json, results/tables/relp_moe_*.md): RelP results section; RelP vs patching comparison; disagreement classification.
- [x] S6 final report (2026-11-08) (results/sections/relp_moe_sva.md + figures/tables; team-readable summary), CLAUDE.md / README update.

## Post-hoc (2026-11-08/09, CPU only; report section 7, scripts/relp_moe_posthoc.py)
- [x] exact rescue of RelP's and Lu's first picks; Lu's rule with layers 0 / 0-2 excluded (all large disagreements vanish;
      Mixtral = RelP top-1, Qwen3 picks RelP rank 2 / 5); cross-task consistency (Jaccard of top-10: exact patching = RelP);
      grid coverage; routing status of the top-20 held-out patches (8-10 % injected without renormalisation).
- [x] RelP paper (arXiv 2601.22594) checked: agreement circuits are MLP neurons only; attention is never ablated (recomputed);
      edges between MLP neurons; open: mean vs counterfactual ablation in the paper's paired SVA evaluation.

## Open follow-ups (not run; need a user decision)
- add-back curve of top-k sets ranked by exact patching (~1 h GPU; expected = RelP for Qwen3, jump at k = 1 for Mixtral);
- neuron-to-neuron edges (extra backward pass per downstream node; new code) and path-patching verification;
- check the RelP authors' code for the ablation type; OLMoE formal run; Mixtral without BOS; dense Mistral-7B.

## Repository status
- Committed and pushed 2026-11-15 (see git log). Committed: code, docs, manifests, logs, tables, figures, run_meta / verification /
  rankings / summaries, intervention and patch parquet (~181 MB, 1,518 files). Kept on the root disk only (gitignored):
  aggregate*.pt (15.3 GB full neuron-level means; > 100 MB per file), per_row_expert_scores.pt (534 MB), routes.pt caches,
  interventions.jsonl (= the committed parquet), dev *.pt.

## Where things are
- Analysis / figures env: .venv-analysis (CPU torch 2.15.0, numpy/pandas/pyarrow at the experiment pins, + scipy, matplotlib);
  the experiment env .venv-relp is not modified. Run: .venv-analysis/bin/python scripts/relp_moe_{analyze,figures}.py
- Logs: logs/relp_moe_chain.log (chain steps), logs/relp_moe_jobs_<model>.log (job output), logs/gpu_queue.log,
  logs/PROGRESS.md ([relp-moe] lines).
- Watcher: scripts/chain_watch.py --chains "relp_moe_chain.sh|relp_moe_patch_chain.sh" --progress-tag "\[relp-moe\]".
- Resume: read this file, tail logs/PROGRESS.md and logs/relp_moe_chain.log, `pgrep -af relp_moe`; re-launching
  scripts/relp_moe_chain.sh skips complete steps and moves incomplete ones aside.
- Do not edit moetrace/relp_moe_stream.py, moetrace/relp_sva.py, moetrace/weights.py, moetrace/arch.py while a chain
  runs: the verification identity binds their sha256 and later steps would refuse to run.

## Notes for the final report (collected during the runs)
- Mixtral expert-level matched random controls: the protocol draws from the experts eligible at the same (layer, position)
  (2-4 for a top-2 MoE), so at small budgets a random set can coincide with RelP's choice (smoke simple, budget 1, seed 101:
  both L31E000 at the subject position). Report random baselines with this limitation; Qwen3 (8 of 128) is not affected.
- bf16 batch-composition noise of native HF (verify diagnostics): Qwen3 up to 0.26-0.38 margin, Mixtral 0.06-0.21.
- Native HF batch-composition noise per verify (8 sequences, batched vs one at a time; functional = native in both): Qwen3 up to
  0.13-0.94 margin (grouped/nounpp 0.94, mean |m| 2.9) with 47-484 top-8 set switches; Mixtral 0.04-0.37 with 0-27 top-2 switches.
  Qwen3's 8th-expert near-ties make single-row values batch-sensitive -> per-row numbers are reported with this caveat, and
  ranking stability (S2) is the check for the aggregate.
- Endpoint A vs B (S3, qwen3/paper/simple): per row they differ (corr 0.80, rel. diff 0.63) but their discovery MEANS nearly
  coincide (expert-level rel. diff 2.8 %). Reason: SVA contains each pair in both directions (A/B and target/foil swapped);
  for a mirrored pair, score_A(mirror) = (q_B - q_A) . beta_B(-F) = score_B(original). Balanced data -> equal means; the same
  algebra makes mean denoise rescue ~ mean noise drop. Direction matters per row, much less for aggregate rankings.
- S3 consistency check: endpoint A recomputed in the endpointB job (same batch composition) == protocol score bit for bit.
- S2 stability (score with 3 chunks of 100 vs 1 chunk of 300; first 5 tasks): top-1 and top-10 sets identical everywhere, top-50
  expert-position overlap 0.96-1.00, top-1000 neuron overlap 0.97-0.99, expert-level corr >= 0.9998 (rel. L2 0.5-2 %):
  per-row bf16 batch noise does not move the aggregate rankings, so it cannot explain RelP-vs-patching disagreements.
- First comparisons (qwen3/grouped, simple + rc): expert units agree closely (Pearson 0.94-0.985 RelP-B vs exact denoise; rank
  corr. 0.86-0.90 on units with effect >= 0.05; top-1 identical, bootstrap 100 % both); rc: Lu selects L41 -> E36 = RelP top-1.
  simple: Lu's layer stage picks L0 (the subject IS the final token; MoE-output patch at L0 restores 6.6 of ~10 logits) and inside L0
  no single expert matters (best rescue 0.06) -> Lu ends at L0E105, while the best expert anywhere = L46E30 = RelP top-1.
  Main RelP failure: early-layer whole-MoE patches at the subject position (exact 6.6 / 4.0 vs RelP 0.15 / 0.14) -> S4f diagnostics.
- Grouped comparisons complete (05:40Z). Subject NOT final (nounpp, rc): agreement in both models - Qwen3 Lu = joint = RelP =
  L41E36 (val rescue 2.4-2.6, Spec 2.4-2.6); Mixtral joint = RelP = L30E2 (Lu's 2-stage rule: rc L30E2, nounpp picks layer 31 ->
  L31E0); Pearson 0.97-1.00, bootstrap top-1 100 % both. Subject = final token (simple, within_rc): Lu's layer stage (final position
  only) picks L0 at the subject token. Qwen3: no single L0 expert matters (<= 0.07) -> Lu e* meaningless, best expert anywhere
  L46E30 = RelP. Mixtral: ONE L0 expert (E2) restores the whole paired difference (val rescue 10.2 / 7.5, Spec = same), RelP
  misses it (Pearson 0.27-0.31 driven by that unit; rank corr. on substantial units 0.78-0.79). Diagnose these L0 units (S4f).
- Qwen3 paper full grids (06:23Z; all positions x layers x union experts, 300 discovery pairs): expert units agree in all 4 tasks
  (RelP top-1 = exact top-1: L46E30 simple/within_rc, L41E36 nounpp/rc; Pearson 0.96-0.99, rank corr. substantial units
  0.93-0.98, top-10 overlap 0.9-1.0, bootstrap top-1 83-100 % both). RelP underestimates the strongest experts (slope 1.3-1.45:
  L41E36 exact 2.2-2.4 vs RelP 1.4-1.5). Lu 2-stage = L41E36 for nounpp/rc (val 2.3-2.7, Spec 2.3-2.6, test 1.9-2.4); for
  simple/within_rc it picks L0 (val ~0). Position units: L0 subject site exact 4.1-5.8 vs RelP 0.12-0.16 (only disagreement).
- Analysis run time with full grids ~10-15 min (CPU); run it in the background.
