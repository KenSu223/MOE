# Phase 8 plan: follow-ups to the RelP vs activation patching study (team meeting, week of 2026-11-09)

**Status: stages 1, 2 and 3 COMPLETE 2026-11-15 (report `results/sections/relp_moe_p8.md`, sections 1-2 = SVA, section 3 = CounterFact and
two-hop capitals; WinoGrande and IOI NOT run, by the user's decision). Committed and pushed 2026-11-15 as 6af6da3 (user request); 2026-11-16 follow-ups (rescue / disruption naming, five patch semantics,
report section 2.1b) committed and pushed 2026-11-18. Stage 4 (larger models) approved 2026-11-18 as part of Phase 9.** Scope (user, 2026-11-15):
run stage 1 and stage 2 (including the router-aware attribution S2.6 and the neuron-level check S2.7), then a detailed
analysis (section 8b); **stage 3 (other tasks) and stage 4 (larger models) are deferred and decided by the user after that
analysis.** Because stage 3 is deferred, the length-bucket / role generalisation is not built now (SVA has one template per
task); sections 4, 5 and 8 keep its specification for later. Builds on Phase 7 (`results/sections/relp_moe_sva.md`,
`docs/relp_moe_plan.md`). Status legend: [ ] not started, [~] running, [x] done. Times UTC.

Resume: read section 10 (order), tail `logs/PROGRESS.md` (`[relp-p8]` lines) and `logs/relp_moe_p8_chain.log`,
`pgrep -af relp_moe_p8`; every job writes `run_meta.json` with `"status": "complete"`, and chains skip complete steps.

## 1. Meeting notes (verbatim, from the user)

From the RelP vs activation patching experiment we ran on SVA:
- Number of experts added back vs. faithfulness + completeness on the y axis
- Top expert distribution: are they all clustered in the last few layers or distributed?
- Scatterplot: on RelP rank all experts, on Lu's method rank all experts - color based on which layer they are from
- RelP hypothesis: bc of approximation there may be more noise as we go down the list of ranked experts, while Lu's paper
  experiments one by one so noise shouldn't accumulate?
- Our contribution beyond applying the RelP method to MOE? - non intuitive / worth publishing answers / discoveries from the
  questions / experiments here
- RelP (= attribution patching?)
- RelP: over/underestimate the effect at a particular layer/position?
- In the Same-expert selection table: if same experts are selected in both A and B runs? If the two rows in the middle of the
  table (experts active in one run while inactive in the other) doesn't align with the selected experts from average in RelP
  (hypothesis: since expert selection gate is frozen as approximation while it varies a lot in those cases)?
- Can we find categories/samples where freezing nonlinearities and the expert gates don't yield faithful estimates, versus
  cases/categories where we can safely apply it?
  - On the samples that activation patching/RelP do not agree on: if the gate plays an important role
  - Separating tasks where the nonlinearity changes a lot - like the gates assigning experts
  - If different runs select different experts
- Can we probe into the model to see the values at the expert gates in MOEs in both runs (A and B), so we know how much they
  have changed in between the runs? Therefore, based on if they differ too much or not, we can separate the dataset into parts
  that RelP can faithfully apply on, and parts where RelP cannot (since the expert gate changed a lot but RelP froze it)
- Experiment on larger models like GPT OSS?
- Do experiment on a different task beyond SVA - like the texas question from Arora paper?
- Maybe another task? - a different type of task to do RelP on (e.g. CounterFact where we have activation patching to compare with)

## 2. What each note asks for

| note | work item | needs |
|---|---|---|
| experts added back vs faithfulness + completeness | S2.1 add-back curves under both rankings (named rescue / disruption and faithfulness / completeness, see section 4) | GPU |
| layer distribution of the top experts | S1.1 | existing data |
| rank-rank scatter coloured by layer | S1.2 | existing data |
| "RelP gets noisier down the list, patching does not" | S1.3 (agreement down the list, split-half reliability, noise floor), S2.1, S2.5 | existing data + GPU |
| contribution beyond applying RelP to MoE | section 15; tested by S2.3, S2.4, S2.6 and stage 3 | - |
| RelP = attribution patching? | answered in section 3; S2.4 plain gradient vs RelP rules | GPU |
| over / underestimate per layer and position | S1.4 bias map, S2.3 decomposition | existing data + GPU |
| same-expert table, units routed at one endpoint only | S1.5 routing classes | existing data |
| where freezing fails; role of the gate; tasks / samples | S1.6, S1.7, S2.2, S2.3 | existing data + GPU |
| probe the gate values in A and B, split the dataset | S2.2 gate probe, S1.7 | GPU |
| larger models (gpt-oss) | stage 4, deferred (section 14) | user decision |
| Texas task of the RelP paper | S3.2 two-hop state capitals | data + GPU |
| another task, e.g. CounterFact | S3.1 CounterFact, S3.3 WinoGrande and IOI | GPU |

## 3. Starting point

**Data on disk** (paper profile, both models, four SVA tasks; `results/relp_sva_moe/<model>/paper/<task>/`):
- `patch_full/part_*.parquet`: exact patches for every layer x every position from the changed word x every expert routed at A
  or B (and the whole MoE output), both directions, 380 pairs (discovery 300 / validation 40 / test 40). Per (pair, unit):
  `effect`, `active_A`, `active_B`, `route_changes` (later top-k sets the patch switches). Qwen3 0.40-1.96 M units per task,
  Mixtral 71 k-401 k.
- `score_endpointB/per_row_expert_scores.pt`: RelP per pair at endpoints A and B, expert level, `[300, L, T, E]`.
- `score_{zero,mean,endpointB}/aggregate.pt`: neuron-level discovery means `[L, T, E, I]`; rankings JSON.
- `patch_full/routes.pt`: top-k sets of both runs `[L, 380, T, k]`. Router weights / logits are NOT stored.
- `{validation,test}/*/interventions.parquet`: held-out interventions per pair for RelP's lists (expert budgets 1-50).
- `diagnose/diagnose.parquet`: natural / frozen later routing / frozen later attention / both, 143 hand-picked units.
- Grouped profile: layer x position grid and final-position experts only; shares 12-106 pairs per task with paper.

**RelP vs attribution patching.** Same family: unit score = (activation difference) x (coefficient from one backward pass at
one endpoint). Attribution patching uses the plain gradient. RelP changes only the backward pass (guide 3.1): RMSNorm
denominators, attention patterns, the SiLU sigmoid gate and the router's choice and weights are constants, and the product
SiLU(g)·u passes half of the gradient through each branch. Consequence used below: through every LATER expert RelP's
coefficient is half of the first-order term of the frozen model; the residual bypass is not halved.

**CPU preview** (2026-11-15; paper / nounpp, denoise, discovery; Pearson r over single (pair, expert unit) values, RelP at
endpoint B vs exact):

| slice | Qwen3 | Mixtral |
|---|---|---|
| last 8 layers, routed at both endpoints | 0.91 | 0.98 |
| last 8 layers, routed at one endpoint only | 0.49-0.61 | 0.77-0.87 |
| earlier layers | 0.05-0.25 | 0.03-0.75 |
| patch switches few later top-k sets (Qwen3 <= 5, Mixtral <= 1) | 0.87 | 0.92 |
| patch switches more (Qwen3 > 10, Mixtral > 1) | 0.09 | 0.42 |

Open after the preview: early-layer per-pair exact effects in Qwen3 average 0.05-0.11 logits with near-chance sign agreement
while 300-pair means agree (r 0.94-0.99). Either real pair-specific effects or sensitivity to routing near-ties (bf16);
S1.3 / S2.5 decide.

## 4. Definitions used by every item

- **Metric.** F = logit(clean answer) - logit(patch answer) at the final position, same pair of tokens at both endpoints.
- **Units.** neuron (layer, role, expert, neuron); expert-position (layer, role, expert); whole expert (layer, expert) =
  signed sum over the pool positions; position unit (layer, role) = the whole MoE output there.
- **Roles.** A role is an aligned position shared by all pairs of a task. SVA: every token position (one token per word, as in
  Phase 7). New tasks (stage 3): named word slots, position = last token of the slot; tokens outside the slots enter only
  whole-expert sums. With SVA's roles the new code must reproduce Phase 7 bit for bit (tests T3, T4, D1).
- **RelP scores.** Protocol score = endpoint A, (q_A - q_B)·beta_A; endpoint B, (q_A - q_B)·beta_B, labelled. Signed mean over
  discovery pairs (absent units count as 0); ranking by |mean| (protocol section 5).
- **Exact effects.** Complete replacement at one layer (Phase 7 semantics), later layers natural. denoise = F(B<-A) - F_B;
  noise = F_A - F(A<-B). Mean over discovery pairs (absent units 0); ranking by |mean|, signed ranking as a sensitivity.
- **Matched comparisons.** denoise <-> RelP at endpoint B (primary); noise <-> endpoint A.
- **Held-out curves.** Rescue(K) = E[F(B<-A on C_K) - F_B] / E[F_A - F_B] (only the K units carry the clean run; ideal 1);
  disruption(K) = E[F_A - F(A<-B on C_K)] / E[F_A - F_B] (share lost when only the K units carry the corrupted run; ideal 1).
  Faithfulness and completeness are the RelP paper's definitions with the protocol's baselines (keep / remove, zero and
  mean_q_routed): faithfulness = keep, completeness = remove (share left, ideal 0). Ratio of means; validation + test pairs.
  Naming decided by the user 2026-11-16, so the two families are not confused: before that, rescue was called "faithfulness"
  and 1 - disruption "completeness" in this plan, the report, tables and figures.
- **Uncertainty.** Group bootstrap over `group_id` (2,000 resamples). A unit mean counts as non-zero if |mean| > 2 SE; every
  rank statistic is reported for all units and for units non-zero under at least one method.
- **Selection.** Rankings, thresholds, temperatures and unit samples are fixed on discovery pairs; validation and test are
  only evaluated.

## 5. Code layout and verification gates

No edit to `moetrace/relp_moe_stream.py`, `relp_sva.py`, `weights.py`, `arch.py` (verification identity) and no change to
Phase-7 outputs. New files only; `scripts/relp_sva_moe_run.py` gets one change (task names read from the manifest instead of
a fixed list; Phase-7 commands unaffected).

| file | content |
|---|---|
| `moetrace/relp_moe_p8.py` | `RuleRelP(StreamRelP)` (backward-rule switches; forward values unchanged); role tables and length buckets; `multi_backward` (one reverse stream for several buckets and rule variants; scores (q_hi - q_lo)·beta_coef for arbitrary row triples); `patch_multi` (buckets, expert / position / neuron units, freeze variants); `gate_probe`; `forward_embeds` (interpolated embeddings) |
| `scripts/relp_moe_p8_run.py` | commands `probe`, `rules`, `path`, `addback`, `freeze`, `neuron`, `noise`, `score`, `patch`; requires the task's passed `verification.json` (same identity check as the Phase-7 runner) |
| `scripts/relp_moe_p8_prepare.py` | stage-3 manifests in the Phase-7 schema plus `roles` |
| `scripts/relp_moe_p8_analyze.py`, `relp_moe_p8_figures.py` | CPU analysis and figures (`.venv-analysis`) |
| `scripts/relp_moe_p8_chain_{a,b}.sh` | detached chains through `scripts/gpu_queue.sh` |
| `tests/test_relp_moe_p8.py` | tests T1-T12 on tiny random checkpoints of the three families |

**Tests (must pass before any formal job).**
- T1 `RuleRelP` with all rules on = `StreamRelP` (scores bit-identical).
- T2 plain gradient including routing weights = autograd through the native HF modules; with routing weights detached = the
  same with a detach hook.
- T3 `multi_backward` (several variants, several buckets, one reverse stream) = separate single runs, bit-identical;
  triples (A, B, B) and (A, B, A) = `relp_backward_endpoint`.
- T4 role aggregation with roles = all positions = Phase-7 totals.
- T5 `patch_multi` on one bucket = `patch_pass` (margins and route changes identical); several buckets = per-bucket calls.
- T6 neuron unit = the engine's verified `Override(value="source")` with a one-neuron mask.
- T7 freeze variants: nothing forced = `StreamRelP.layer` bitwise; everything forced on an unpatched row = the base run
  bitwise; frozen routing / attention = `diagnose_pass`; fp32: all-frozen effect of a patch scaled by eps, divided by eps,
  converges to the `relp_nohalf` score as eps -> 0.
- T8 `addback` on single-template data = the runner's `evaluate()` records.
- T9 router-weight and soft top-k variants: forward values bit-identical to the natural forward.
- T10 path: endpoints reproduce the A and B runs; one segment = the mean of the two endpoint scores.
- T11 probe: router logits reproduce the engine's top-k sets and weights; all distances 0 for identical inputs.
- T12 stage-3 manifests: checksums, equal pair lengths, single-token answers, valid roles.

**Dev checks on the real checkpoints (queue names `relp-dev-*`, low priority; not formal results).**
- D1 `rules`, variant `relp`, qwen3 and mixtral paper / nounpp: per-pair scores = Phase 7's `per_row_expert_scores.pt`
  (same 300-pair batch; bit-identical expected).
- D2 `addback`, RelP list, budgets 1-50, validation: normalised values within 0.02 of Phase 7 (batch composition differs).
- D3 `freeze`, natural variant: discovery means correlate >= 0.99 with `patch_full` for the same units.
- D4 timing of every command on one task per model; if the total exceeds 1.5 x section 11, report before the formal chains.

## 6. Stage 1: existing SVA data, CPU only

Script `scripts/relp_moe_p8_analyze.py` (re-runnable; also used for the stage-3 tasks). Paper profile, 2 models x 4 tasks.
Outputs: `results/relp_sva_moe/p8/units_<model>_<task>.parquet`, `results/relp_sva_moe/p8_summary.json`,
`results/tables/relp_moe_p8_*.md`, `results/figures/relp_moe_p8_*.png`.

- [x] **S1.0 Unit table.** One row per (kind, layer, position, expert): discovery mean and SE of the exact effect (both
  directions), RelP mean and SE (endpoints A and B), share of pairs per routing class (both / clean only / corrupt only /
  neither), mean `route_changes`, ranks under each method; the same exact means on validation + test.
- [x] **S1.1 Layer distribution.** Top-K expert-positions (K = 10, 20, 50, 100) under RelP and under exact patching: counts per
  layer, split by position class (subject / in between / final); cumulative share of the absolute score by layer.
  Figure `layerdist_<model>.png` (4 tasks x 2 methods), table `layerdist.md`.
- [x] **S1.2 Rank-rank scatter.** All expert units of a task: x = RelP rank, y = exact rank (log-log), colour = layer, marker =
  position class; grey zone = exact mean within 2 SE of zero. Value-value panel (signed, symmetric log) alongside.
  Figure `rankscatter_<model>.png`.
- [ ] **S1.3 Agreement down the list, reliability, noise floor.**
  - top-K overlap for K = 1...2000; rank correlation, sign agreement, median relative error per rank bin
    (1-10, 11-30, 31-100, 101-300, 301-1000, > 1000);
  - split-half reliability of each method: 100 random halves of the discovery groups, top-K overlap and rank correlation
    between halves as a function of K;
  - noise floor of exact patching from pairs shared by the paper and grouped profiles (same pair and final-position unit,
    two jobs with different batches): SD of the difference per layer band; extended by S2.5;
  - additivity: sum of single exact effects of the top-K vs sum of RelP scores vs measured joint effect (S2.1).
  Figure `depth_<model>.png`, tables `depth.md`, `splithalf.md`, `noisefloor.md`.
- [x] **S1.4 Bias map.** Per layer x position class: exact mean, RelP mean, difference and ratio, for whole-MoE-output units
  (RelP = sum over experts there) and for single experts (slope of exact on RelP per layer band).
  Figure `biasmap_<model>.png`, table `biasmap.md`.
- [x] **S1.5 Routing classes (the "same-expert" table).** Per (pair, unit): routed at both endpoints / clean only / corrupt only.
  Per class x layer band: share of units and of the absolute effect, Pearson, slope, sign agreement, mean absolute error;
  class composition of each method's top-20 / top-50; both rankings recomputed on both-routed pairs only.
  Write-up note: for a one-sided unit RelP's score is (q_A - 0)·beta, the correct first-order term; what RelP holds fixed is
  the routing of LATER layers. Table `classes.md`, figure `classes.png`.
- [x] **S1.6 Route switches.** |exact - RelP| and Pearson by quintile of `route_changes`, per layer band and position; map of
  the A-vs-B top-k overlap per layer x position and task. Table `switches.md`, figure `switches.png`.
- [x] **S1.7 Per-pair reliability and categories.** Per pair: agreement over its units (Pearson, relative L2, overlap of the
  per-pair top-20). Related to task, case (singular -> plural / plural -> singular), size of the paired difference, A-vs-B
  routing overlap, total route switches; later joined with the S2.2 probe features. Output: categories ordered by reliability
  and a threshold rule fixed on discovery. Table `pairs.md`, figure `pairs.png`.

## 7. Stage 2: GPU, SVA (paper profile), Qwen3 and Mixtral

All commands: `.venv-relp/bin/python scripts/relp_moe_p8_run.py <command> --model-dir <snapshot> --manifest-dir
data/relp_sva/<model>_paper_v1 --task <task> --verification results/relp_sva_moe/<model>/paper/<task>/verify/verification.json
--output-dir results/relp_sva_moe/<model>/paper/<task>/p8_<command>`, submitted through `scripts/gpu_queue.sh
relp-p8-<model>-<task>-<command>`.

- [x] **S2.2 `probe` (forward only).** (chain A, 2026-11-15) Every pair of the task (380), both runs, every layer x position:
  - router: full logits (stored fp16) -> top-k overlap A vs B, total variation between the weight vectors, margin between the
    k-th and (k+1)-th expert, RMS logit difference;
  - SiLU gates of experts routed in both runs: mean |sigma(g_A) - sigma(g_B)|, share of neurons with a change > 0.1;
  - RMSNorm denominators (both norms of the layer): log ratio A / B;
  - attention patterns: total variation A vs B per layer and query position (mean over heads);
  - residual distance ||h_A - h_B|| / ||h_B||.
  Analysis: maps of each quantity (layer x position, per task); which quantity predicts RelP's error per unit (features
  averaged over later layers and positions) and per pair; pairs split at the discovery median of the best predictor into
  "gates stable" / "gates change"; RelP-vs-exact agreement and held-out rescue / disruption (per-pair values of S2.1)
  on each half. Outputs `probe.npz`, `router_logits.npy`; tables `probe_maps.md`, `probe_predict.md`, `probe_split.md`.
- [x] **S2.4 + S2.6 `rules` (one forward, one reverse stream, all variants).** Discovery pairs, scores at endpoint B (and A for
  `relp`), expert-position means and per-pair values; neuron-level means kept for `relp` and the selected router-aware
  variant.

  | variant | backward rule |
  |---|---|
  | `relp` | the guide's rules (reference; must equal Phase 7) |
  | `grad` | plain gradient, router frozen (attribution patching) |
  | `grad_rw` | plain gradient including the routing weights (top-k set fixed): the true first-order term |
  | `relp_nohalf` | RelP's detached quantities, full product rule: first-order term of the frozen model |
  | `relp_nonorm`, `relp_noattn`, `relp_nogate` | RelP with one rule switched off |
  | `relp_rw` | RelP + gradient through the routing weights (router-aware, version 1) |
  | `relp_rw3` | as `relp_rw`, relevance split in three between gate, up branch and routing weight |
  | `relp_soft` x 3 | `relp_rw` + soft top-k boundary: mask sigma((logit - boundary) / tau) over the top-(k + r) experts as a straight-through term (forward unchanged); r = 2 (Qwen3) / 1 (Mixtral); tau = 0.5, 1, 2 x the layer's RMS router-logit difference between the two runs (from S2.2, discovery) |

  Analysis: each variant vs exact patching at expert and position-unit level (Pearson, slope, rank correlation on non-zero
  units, top-K overlap, bias map). Table `rules.md`, figure `rules_<model>.png`.
- [x] **S2.6 `path` (multi-step version; simple and nounpp, both models).** 9 points on the straight line between the B and A
  embeddings of the changed token, RelP backward at every point, score = sum over segments of (q_next - q_prev) x mean of
  the two coefficients. Tests whether a path average recovers large perturbations (Mixtral L0E002) and routing switches.
  Selection of the router-aware variant for S2.1: highest expert-level Pearson with exact patching on discovery, mean
  over tasks, one choice per model.
  Stated in advance: success = on the Qwen3 layer 0-2 subject position units the score reaches >= 0.5 of the exact effect
  (RelP: about 0.03), expert-level Pearson on the last 8 layers falls by <= 0.02, and held-out rescue at K = 20 is not
  more than 0.02 below RelP. Any other outcome is reported as it is.
- [x] **S2.5 `noise` (repeat patches).** (2026-11-15 02:34-03:00Z, 8/8, `p8_noise/`) 40 discovery pairs per task, changed and final positions, all layers, all routed
  experts, both directions, in another batch composition; difference to `patch_full` per (pair, unit) -> noise floor by
  layer band and by `route_changes`.
- [x] **S2.3 `freeze` (which frozen quantity causes the error).** First 100 discovery pairs, denoise. Units: every whole-MoE
  output (layer x position from the changed word) + top 200 expert units by exact effect + top 200 by RelP + 100 random
  expert units stratified by layer band. Each unit is patched 8 times, with these quantities of the LATER layers held at the
  unpatched run's values: nothing (natural); routing (set and weights); routing set only (weights free); attention patterns;
  norm denominators; routing + attention; + norm; + SiLU gates (everything RelP holds fixed).
  Decomposition per unit (means over pairs):
  exact - RelP = routing + attention + norm + SiLU gate + finite size (all-frozen patch - `relp_nohalf`) + half rule
  (`relp_nohalf` - `relp`); the routing term split into set switches and weights; solo effects reported next to the ladder
  (the ladder is order-dependent). Output `freeze.parquet`; tables `freeze_decomp.md`, `freeze_units.md`; figure
  `freeze_<model>.png`.
- [x] **S2.7 `neuron` (neuron-level check).** Units: RelP's top 1,000 neurons, 1,000 matched random neurons (protocol's matched
  draw, seed 101), and all neurons of the top expert-position (Qwen3: 768; Mixtral: its top 768 by RelP). 300 discovery
  pairs, denoise; neuron unit = this neuron's q replaced by the clean run's (protocol semantics). RelP per pair for the same
  neurons at endpoint B in the same job. Analysis: Pearson / slope / sign agreement by rank bin, noise floor from the random
  set, sum of single-neuron effects vs the expert's effect. Outputs `neuron_patch.parquet`, `relp_neuron_rows.pt`; table
  `neuron.md`, figure `neuron_<model>.png`.
- [x] **S2.1 `addback` (experts added back vs rescue / disruption and faithfulness / completeness).** Validation + test pairs (80 per task).
  Rankings fixed on discovery: `relp` (protocol, endpoint A), `relp_B`, `exact` (|mean rescue|), `exact_noL0-2` (layers 0-2
  excluded), `lu` (Lu's order: layers by final-position whole-MoE rescue, inside a layer the recurrent final-position
  experts by rescue), `grad`, the selected router-aware variant, `random` (3 seeds, matched to `relp`).
  Units and budgets: expert-position 1, 2, 5, 10, 20, 50, 100, 200, 500, all; whole expert 1, 2, 5, 10, 20, 50, 100, all
  ("all" = every eligible unit of the pool, the ceiling; budgets above the eligible count are dropped).
  Interventions: denoise and noise (rescue / disruption, the other run as baseline), keep and remove (faithfulness /
  completeness, the RelP paper's) with the mean_q_routed baseline (zero-baseline keep is ~0 by construction, Phase 7; dropped to halve the run).
  Added 2026-11-15 after the first path results: rankings `path_grad_rw` and `path_relp` (8-step path scores) for simple and nounpp. Outputs `interventions.parquet`, `summary.json`, `rankings.json`;
  figures `addback_<model>.png` (x = number of experts, y = rescue, disruption, faithfulness, completeness; one line per ranking), table
  `addback.md`.

## 8b. Detailed analysis after stage 2 (decision point)
Report `results/sections/relp_moe_p8.md`: every meeting note answered with its table / figure; the six-term decomposition of
RelP's error; where RelP can be trusted (rule from S1.5-S1.7 + S2.2, checked on held-out pairs); router-aware variants
against the criterion stated in S2.6; neuron-level agreement; each expectation of section 12 marked held / failed;
what the results imply for stage 3 (which tasks are worth running, which tools to carry over) and stage 4. Then a
plain-language summary to the user, who decides on stages 3 and 4.

## 8. Stage 3: other tasks (DEFERRED 2026-11-15; specification kept for the later decision)

Common: manifests in `data/relp_p8/<model>_<task>_v1/` (Phase-7 row schema + `roles`), built by
`scripts/relp_moe_p8_prepare.py` from pinned tokenizers (`data/relp_sva/models.lock.json`); Mixtral keeps `<s>` as in Phase 7.
Prompt lengths vary, so rows are processed in length buckets inside one streamed pass. Roles (single positions per pair):
the last changed token and the final token; rankings and add-back use these two roles (stated scope), the exact grid covers
every position from the first changed token to the final one and is reported by position class.
Per task and model, in order: `verify` (Phase-7 runner; 4 calibration pairs of the modal length) -> `score` (RelP endpoints
A and B, protocol aggregates and rankings) -> `rules` -> `patch` -> `probe` -> `freeze` (CounterFact, two-hop) -> `addback`
(as S2.1, plus the protocol's neuron budgets 1-1000 for the RelP list and matched random sets) -> analysis (S1.1-S1.7 on
the new task). Outputs `results/relp_p8/<model>/<task>/`.

- [ ] **S3.1 CounterFact.** Source: the STR pairs of Direction 6 (`results/{qwen3,mixtral_bos}_str/sweep_cases.parquet` +
  `str_candidates.parquet`), pair = case x donor slot 0 or 1, same length and subject positions by construction.
  Qwen3 403 pairs (215 cases), Mixtral 404 (213 cases); lengths 3-15 / 5-16 tokens. Splits by case (`group_id` = case):
  the paper's discovery cases -> discovery (200 / 201 pairs); the paper's validation cases, shuffled with seed 12 -> 15 cases
  calibration, the rest halved into validation and test. Roles: last subject token, final token. Both patch directions
  (pairs are not mirrored). Cross-check: final-position expert rescues of the old engine (`str_expert_rows.parquet`) for
  the same case and donor.
- [ ] **S3.2 Two-hop state capitals (the paper's "Texas" example).** Prompt `Fact: the capital of the state containing {city}
  is`; answer = first token of the capital. Candidate list: 2-8 non-capital cities per state (about 150; tokenizer check
  done: Qwen3 67 one-token and 49 two-token cities, Mixtral 56 two-token and 39 three-token; capital first tokens distinct
  except Alabama / Vermont in Mixtral, never paired).
  - Competence scan (GPU, forward only): a city passes if logit(own capital) - max logit(other capitals) >= 1.
  - Pair = two passing cities of different states with the same token count; both directions as rows; `group_id` = the
    unordered city pair. States shuffled with seed 12: 60 % discovery states, 20 % validation states, 20 % test states;
    pairs only inside a part; calibration = discovery-state pairs not used for discovery. Up to 300 / 40 / 40 / 40.
  - Rule fixed now: with fewer than 60 discovery or 20 held-out pairs for a model, relax to margin >= 0 and report; with
    fewer than 40 pairs in total, run discovery only and make no held-out claim for that model.
  - Roles: last city token, final token (`is`). Denoise direction only (rows are mirrored).
  - Differences from the paper stated in the report: base models and raw text (paper: Llama-3.1-8B-Instruct, chat format),
    paired score (paper: single prompt, sum of the top-5 logits), comparison with exact patching (paper: none).
- [ ] **S3.3 WinoGrande and IOI (light: no `freeze`).** WinoGrande: `data/wino_str/pairs_train_xl_<proto>.parquet`, the shared
  case set (128 discovery pairs; 128 validation pairs -> 16 calibration / 56 validation / 56 test), both directions as rows
  (256 discovery rows), lengths 14-32; roles: last token of the swapped option, final token. IOI: `data/ioi/pairs_s2io_<proto>.parquet`
  (second subject mention <-> indirect object), same split scheme, lengths 14-21; roles: the swapped name, final token.
  Denoise only (mirrored rows). Cross-check for WinoGrande: old-engine final-position expert rescues; IOI has no earlier
  expert-level patching.
- [ ] **S3.4 Cross-task summary.** Per task and model: RelP-vs-exact agreement (expert level), MoE-only add-back ceiling and
  rescue at K = 20, share of the exact effect at the early site, against the task's attention share (Phases 3-6).
  Figure `tasks.png`, table `tasks.md`.

## 8c. Live status
- 2026-11-15 02:34-03:00Z S2.5 noise repeats (Phase-7 patch command, 40 discovery pairs per task): 8/8 complete.
- Code: `moetrace/relp_moe_p8.py`, `scripts/relp_moe_p8_{run,data,select,analyze,figures,smoke}.py`, chains
  `scripts/relp_moe_p8_chain_{noise,a}.sh`; tests `tests/test_relp_moe_p8.py` 21/21 (T1-T3, T5-T7, T9-T11 incl. finite-difference
  checks: 'grad' = derivative of the model with later routing frozen, 'relp_nohalf' = derivative of the fully frozen model,
  'grad_rw' = derivative of the natural model). T4 / T8 / T12 belong to the deferred stage-3 generalisation.
- Smoke (qwen3 / nounpp, 8 rows, `results/relp_moe_dev/p8_smoke/`): all six commands rc=0 after two fixes (tracked-neuron
  accumulation wrote into a copy; dtype in the add-back exact ranking). Finding to carry into the analysis: the RANG variant
  (routing + attention + norm + SiLU gates frozen) diverges for layer-0 subject-position whole-output patches (effect ~1e7):
  with frozen denominators and linearised gates nothing bounds a large early perturbation; such units are flagged and the
  decomposition is summarised with medians and divergence counts.
- 2026-11-15 ~03:15Z chain A started (stage 2 GPU, 2 models x 4 tasks, then selection and add-back).
- 04:02-06:26Z Mixtral 4/4; router-aware selection on discovery (qwen3 relp_soft05, mixtral relp_soft2; both within 0.01 of RelP);
  add-back 8/8 (mixtral/nounpp OOM -> runner splits labels into memory-sized passes -> rerun 06:26-06:36Z). Chain A complete,
  ~4.0 GPU-h in total.
- 06:40Z final analysis, figures and report `results/sections/relp_moe_p8.md` (section 8b). Decision point for stages 3 / 4.
- 03:12-04:02Z Qwen3 probe / rules / path / freeze / neuron 4/4 (about 12 min per task). D1: 'relp' variant bit-identical to Phase 7
  at both endpoints. D3: freeze natural vs patch_full unit means r 0.998-1.000. Mixtral started 04:02Z.
- Stage 1 (S1.0-S1.7) and S2.5 analysis + figures done (`results/tables/relp_moe_p8_{layerdist,depth,biasmap,classes,switches,pairs,noisefloor}.md`).
- Analysis decisions taken while reading the first outputs (documented in the report): the freeze decomposition uses
  set / weights / attention / norm+nonlinearity / half (RAN and RANG, the norm- and gate-frozen patches, diverge for some early
  units, so their splits are given only where bounded); single-neuron effects are corrected for a per-(pair, layer) offset of
  the patched rows (bf16 GEMM-shape difference amplified by routing), estimated from random / top-expert neurons.

- 2026-11-16 follow-up (user request: "run all three"): the swap does not renormalise the routing weights at the patched site
  (guide section 8). Added `MIX_MODES` (full / content_both / content_all / renorm / route) and `evaluate_mix` to
  `moetrace/relp_moe_p8.py` (engine untouched; a shallow model copy whose layer() also applies the new overrides); tests 30/30
  (hand-built reference per mode on 3 families; a 1 % mutation is caught). Runner `scripts/relp_moe_p8_mix.py` (rankings and random
  sets rebuilt exactly as S2.1; full rows repeat S2.1 at r >= 0.999), chain `scripts/relp_moe_p8_chain_mix.sh` 17:41-18:04Z, 8/8,
  23 GPU-min; analysis `scripts/relp_moe_p8_mix_analyze.py` -> report section 2.1b. Result: no ranking comparison changes; renorm =
  full; content-only lowers Qwen3 by up to 0.15 (routing carried), Mixtral unchanged. Committed 2026-11-18.

## 8d. Stage 3 execution (approved 2026-11-15: CounterFact, then two-hop; no WinoGrande / IOI)

Design (replaces the sketch in section 8 where they differ):
- **Manifests** `data/relp_p8/<model>_<task>_v1/` (Phase-7 row schema + `roles: {subject: pos, final: pos}`), built by
  `scripts/relp_moe_p8_prepare.py`. CounterFact from the Direction-6 STR runs (`results/{qwen3,mixtral_bos}_str`: case x donor slot 0 / 1;
  discovery = the paper's discovery cases; the paper's validation cases shuffled with seed 12 -> 15 cases calibration, rest halved into
  validation / test; `group_id` = case). Two-hop from a fixed city list + a GPU competence scan (rule in section 8, S3.2).
- **Varying prompt lengths**: rows are bucketed by token length; every per-bucket function runs unchanged in its own thread and a
  scheduler (`moetrace/relp_moe_p8.py: lockstep`) serves all buckets from ONE streamed pass (clients run strictly one at a time).
  Test: lockstep results = separate per-bucket runs.
- **Roles**: `subject` = last token of the changed word (CounterFact: last subject token; two-hop: last city token), `final` = last
  token. RelP scores and exact effects are read at each pair's own role positions; rankings and add-back use (layer, role, expert);
  the exact grid still covers every position from the first changed token on and is also reported by position class.
- **Commands** `scripts/relp_moe_p8_run3.py`: `scan` (two-hop), `verify` (Phase-7 `verify()` on 4 calibration pairs of the modal
  length), `probe`, `rules` (relp, grad, grad_rw, relp_nohalf, relp_rw; both endpoints), `path` (relp, relp_rw, grad_rw; 8 steps),
  `patch` (full grid; CounterFact both directions, two-hop denoise only), `noise` (repeat of 40 discovery pairs), `freeze`, `addback`
  (rankings relp, relp_B, exact, exact_noL0-2, lu, grad, path_grad_rw, path_relp, random x 3; role units and whole experts).
- Outputs `results/relp_p8/<model>/<task>/<command>/`; chain `scripts/relp_moe_p8_chain_b.sh`; analysis
  `scripts/relp_moe_p8_analyze3.py`; report section appended to `results/sections/relp_moe_p8.md` (part II).
- Status: [x] code + tests (27/27; lockstep = separate runs bit for bit; evaluate_roles = Phase-7 evaluate)  [x] manifests CounterFact
  (qwen3 200 / 28 / 85 / 90 rows, mixtral 201 / 27 / 86 / 90; lengths 3-16)  [x] smoke qwen3 / counterfact (8/8 commands)
  [x] chain B (2026-11-15 07:22-11:06Z, ~4.0 GPU-h): CounterFact qwen3 (patch grid 2.94 M units, 47 min), mixtral (0.67 M); two-hop
  scans (qwen3 108 / 128 cities pass, mixtral 123 / 128; manifests 300 / 40 / 40 / 40 rows, held-out states unseen in discovery);
  two-hop qwen3, mixtral. One failure: mixtral / twohop / path OOM (whole length bucket at once) -> `--chunk` added, rerun with
  `--chunk 40` succeeded before the add-back step. All 32 runs `"status": "complete"`; verify 4 / 4 passed.
  [x] analysis + report (section 3 of results/sections/relp_moe_p8.md; tables `results/tables/relp_moe_p8s3_*.md`, figures
  `results/figures/relp_moe_p8s3_*.png`, cross-task table `relp_moe_p8s3_tasks.md`).
- Headline of stage 3: late final-token experts agree in all four cells (same top-1; Pearson 0.96-1.00); the early subject site is
  underestimated everywhere (RelP 8-22 % of exact at layers 0-2, plain gradient ~0) by a task-dependent mixture of later routing
  (20-61 %) and linearisation; the 8-step path gradient is the best or equal-best held-out ranking at K = 20-50; Lu's order the worst;
  Qwen3 L44E069 is the top unit of CounterFact AND two-hop (60 % of the two-hop difference).
- Committed 2026-11-15 (~451 MB, largest file 31 MB); gitignored and kept on the root disk only (~506 MB): resumable `*_part_NNN.npz` chunks (merged row for row into the parquet beside them), `p8_probe/router_logits.npy`, stage-3 `routes_T*.pt`.
  Check: every chunk directory's row count equals its merged parquet (28 / 28 directories).
- Stage-3 simplifications (stated in the report): expert-level scores only (no neuron-level aggregates); add-back uses noise / denoise
  only (no mean-baseline keep / remove: no per-position means exist across templates); rule variants reduced to relp, grad, grad_rw,
  relp_nohalf, relp_rw (the soft top-k and one-rule-off variants did nothing / little in stage 2).

## 9. Monitoring and logging
- Chains: `scripts/relp_moe_p8_chain_a.sh` (stage 2), `scripts/relp_moe_p8_chain_b.sh` (stage 3); detached (`nohup`), every
  GPU job through `scripts/gpu_queue.sh`, complete steps skipped on relaunch, log `logs/relp_moe_p8_chain.log`, job output
  `logs/relp_moe_p8_jobs_<model>.log`.
- Watcher: `scripts/chain_watch.py --chains "relp_moe_p8_chain" --progress-tag "\[relp-p8\]"` under the Monitor tool, re-armed
  every 30 min (failed jobs, tracebacks, chain exits, idle GPU with a live chain).
- `logs/PROGRESS.md` gets a `[relp-p8]` line per milestone; this file's checkboxes and CLAUDE.md are updated at every stage end.
- Nothing is committed unless the user asks (stages 1-3 were committed and pushed on request, 2026-11-15).

## 10. Order of execution
1. `noise` repeats first (existing Phase-7 patch command, no new code) so the GPU works while the new code is written.
2. `moetrace/relp_moe_p8.py`, runner and tests; tests (`relp-test-*` jobs), dev checks D1-D4 on the real checkpoints.
3. Chain A, per model and task: `probe` -> `rules` -> `path` (simple, nounpp) -> `freeze` -> `neuron`; then the analysis
   selects the router-aware variant on discovery; then `addback` for all tasks.
4. Stage 1 on the existing data (CPU), written and run while the GPU chain is busy.
5. Detailed analysis (section 8b), report, CLAUDE.md / README / this file updated, summary to the user.
Deferred: stage-3 manifests and chain B (CounterFact -> two-hop -> WinoGrande -> IOI).

## 11. Budget

| block | GPU time |
|---|---|
| tests and dev checks | ~0.5 h |
| S2.2 probe (8 task-models) | ~0.3 h |
| S2.4 / S2.6 rules (8) + path (4) | ~1.0 h + ~0.8 h |
| S2.5 noise repeats (8) | ~0.4 h |
| S2.3 freeze (8) | ~1.1 h |
| S2.7 neurons (8) | ~0.9 h |
| S2.1 add-back (8) | ~1.2 h |
| **total, stages 1-2** | **~6.5 h** (Phase 7: 7.7 h) |
| deferred stage 3: CounterFact, two-hop (full), WinoGrande, IOI (light), 2 models each | ~5.5 h |

Disk: about 3 GB of new outputs on the root disk (172 GB free); no new checkpoint.
Elapsed for stages 1-2 and the analysis: about 1.5-2 days including code, tests, the detached GPU chains and the report.

## 12. Expectations written down before the runs
1. Add-back: in Qwen3 the RelP and exact rankings give the same curve (within 0.03 up to K = 50). In Mixtral the exact
   ranking reaches >= 0.9 at K = 1 (L0E002 at the subject); without layers 0-2 it matches RelP.
2. Layer distribution: >= 80 % of RelP's top-20 sit in the last 8 layers in both models; exact patching adds the layer-0
   subject unit in Mixtral and nothing early in Qwen3 at expert level.
3. Down the list: top-K overlap >= 0.8 up to K = 20 and < 0.5 by K = 200-500; at K >= 100 RelP's list is at least as
   reproducible between halves as exact patching's in Qwen3.
4. Routing classes: one-sided units agree less than both-routed ones in the last 8 layers, in every task.
5. Decomposition: Qwen3 early position units, routing >= 70 % of exact - RelP; Mixtral L0E002, finite size dominates;
   late experts whose write is amplified (Qwen3 L41E036), the half rule dominates.
6. Plain gradient vs RelP: both correlate >= 0.9 with exact at expert level; `relp_nohalf` has a slope closer to 1.
7. Router-aware variants: open; criterion in S2.6.
8. Neurons: r >= 0.9 for the top 100; below rank ~300 the exact single-neuron effects are inside the noise floor.
9. CounterFact: RelP's final-position top-1 is Qwen3 L44E069 or L42E115 and Mixtral L21E001 or L19E002; the early subject
   site has no shared expert and is again underestimated.
10. Two-hop: open (do middle-layer experts at the city token appear, and do the methods agree on them?).
11. IOI: low MoE-only ceiling (attention share 0.8-0.9); expert-level agreement on late experts still high.

## 13. Risks and fallbacks
- bf16 batch effects: comparisons on means with SE; dev checks use tolerances, bit-identity only where the batch is the same.
- Length buckets give small batches and more bf16 variation between rows: reported, covered by S2.5-style repeats on
  CounterFact if agreement looks noise-limited.
- Two-hop competence may be low for base models: rule in S3.2.
- The soft top-k temperature is a modelling choice: fixed from the probe before any comparison, three values reported,
  labelled exploratory.
- Time overrun: D4 budget check; WinoGrande / IOI are already the light version; `path` is limited to two tasks.
- Session or usage-limit interruption: chains are detached and resumable; status in this file and `logs/PROGRESS.md`.

## 14. Deferred: larger models (stage 4; public configs checked 2026-11-15)

| model | size | layers x experts, top-k | engine | notes |
|---|---|---|---|---|
| Mixtral-8x22B-v0.1 | 141 B, ~281 GB bf16 (re-check at download) | 56 x 8, top-2 | supported family | NVMe has 311 GB free; ~4-5 GPU-h |
| gpt-oss-120b | 117 B, 65 GB as released | 36 x 128, top-4 | new family | see below |
| gpt-oss-20b | 21 B, 14 GB as released | 24 x 32, top-4 | new family | smaller than Qwen3-30B: an engine pilot |

gpt-oss needs: a decision on the no-quantisation rule (experts exist publicly only in 4-bit MXFP4); RelP rules re-derived
for its expert function ((up + 1)·gate·sigmoid(alpha·gate) with clamps and biases); a prompt-format decision (chat model,
no base checkpoint); engine additions (attention sinks, YaRN RoPE, top-k softmax router, MXFP4 reader) verified against
transformers' `GptOss` classes. One 120b layer in bf16 is 6.4 GB. Several days before a first result.

## 15. Candidate contributions (to be tested by the stages above)
1. Routing as an information channel: in Qwen3 the layer-0 subject effect reaches the output through which experts later
   layers choose (S2.3: set switches vs weights; stage 3: does it hold on other tasks).
2. A reliability criterion for gradient attribution in MoE from forward-only quantities (S1.5-S1.7, S2.2, S2.3).
3. Router-aware attribution (S2.6).
4. Why RelP underestimates amplified writes: the half rule halves every later-expert path (S2.3, S2.4).
5. Selection rules and noise: Lu's layer-first rule is fragile; exhaustive patching has a per-patch noise floor; RelP's own
   value is the neuron level (S1.3, S2.5, S2.7).
6. Engineering: exact RelP on 30-47 B MoE models on one 24 GB GPU by reverse layer streaming.

## 16. Deliverables
`results/sections/relp_moe_p8.md` (report), `results/tables/relp_moe_p8_*.md`, `results/figures/relp_moe_p8_*.png`,
`results/relp_sva_moe/p8_summary.json`, run directories with `run_meta.json`; status in this file, CLAUDE.md and
`logs/PROGRESS.md`; a plain-language summary to the user at the end.
