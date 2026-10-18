# agent_status p6-data (Phase 6)

Updated: 2026-10-17T16:55Z — ALL DONE (final report sent)

## Done (CPU)
- Builders (moetrace/ext14_data.py; scripts/ext14_build_{dial,ladder,dialA,pilot,qa,tokcheck}.py), QA (data/dial/qa/*), FRAME PILOT
  (P3 chosen for both models, confirmed by the coordinator), e3p3_a2e cell, Set C in P3 layout, PP-blank exclusion, string-wise
  reconstruction checks (e3p3 / Set C / R-site, 0 failures), E2 strata degree_adv / antonym_type, tokenizer identity (Mistral-7B and
  Qwen3 dense identical; Qwen2 = Qwen3 vocab minus 23 special tokens, all prompts identical -> reuse).
- Probes (both models) done; Dial A / E0 builds (dA_e1_a2e/e2a with stmt bind/swap, e0_a2e/e2a); ladder builds (l_bind 1600 items, l_ind 800).
- Ladder case sets registered (CASES READY l_ind/l_bind/l_ant/l_prop qwen3; l_bind/l_ant/l_prop mixtral_bos; REVIEW l_ind/mixtral_bos insufficient).
- Old-frame scans renamed results/dial_scan_<proto>/VOID_oldframe_*.parquet.

## Since 20:35
- IND PILOT built (scripts/ext14_build_indpilot.py; I0-I4, I3 behaviour-only because its STR is the final token); scans queued; l_ind withdrawn
  from the registry (registry key 'withdrawn'); builder supports --ind-frame I0|I1|I2 (scripts/ext14_build_ladder.py).
- P3 rescans done; CASES READY qwen3 e1_a2e / e3_e2a / rsite_e1_a2e, mixtral_bos e3_e2a; extension scans queued for the rest
  (qwen3 9 cells, mixtral 11 cells: scan2 tags dialE-ext1/2).
- Stage 2: PROTOS + PREFER (twin preference) in moetrace/ext14_data.py; full probe specs (probe_specs_full_<fam>); l_ioi build (Phase-3 IOI items);
  scripts/ext14_scan_cf.py (= ext6_str_filter with --paper-key) queued for mistral7b; scripts/ext14_build_stage2.py --what l_cf converts it;
  scripts/ext14_chain_base_stage2.sh mistral7b_bos queued (probes + 15 cells).
- Base analysis: depth-normalised fields (layer_frac, late_moe_last_ceil_L12); T1-T5 only (no expert stage) -> dense-safe.

## Since 21:05
- Coordinator rule for small cells implemented (pooled / behaviour only; registry flags pooled, behaviour_only, behaviour record);
  PROGRESS lines 'CASES READY ... (pooled)' / 'BEHAVIOUR ONLY ...'.
- Registered: qwen3 Dial A (e0_a2e, aagr_a2e, aagr_e2a, acon_a2e; e0_e2a pooled; acon_e2a behaviour only); mixtral_bos e1_a2e, e3p3_a2e,
  sc_agree (+ behaviour-only e2_*, sc_conflict_*).
- FIX: extension selection was split-unbalanced (rank cut = discovery twins); scripts/ext14_scan.py now interleaves splits and skips
  scanned items (files scan_<cell>_ext<k>); mixtral e1_e2a / e3_a2e / e1p_* / rsite withdrawn (registry 'withdrawn') and rescanning (tag dialE-ext3).
- Auto-registration chain scripts/ext14_chain_base_cases.sh (pid 812360): mixtral Dial A, mixtral ext3 cells, qwen3 Dial E after its ext scans.
- IND applier chain scripts/ext14_chain_base_ind.sh (pid 210543): waits for the IND PILOT line + coordinator / 30 min, rebuilds l_ind, scans, registers.
- Stage 2 mistral7b_bos: probes done, cf filter done (results/mistral7b_str), l_cf pairs built (212), cells + l_cf scans queued.

## Since the resume (01:48Z)
- Stage 1 status: BASE READY l_bind/qwen3, e1_a2e/qwen3+mixtral, e1_e2a/qwen3; tier-1 chain pid 3604990 continues (priority loop).
- l_ind in I2 registered both models (qwen3 pooled 66); Mixtral Dial A / E0 behaviour only (reasons in data/dial/build_report.md).
- Stage 2: mistral7b scans were complete; qwen3_4b / qwen3_8b scans started (stage2b); orchestrators scripts/ext14_chain_base_s2.sh
  for mistral7b_bos / qwen3_4b / qwen3_8b (ext scans -> case sets -> smoke -> full tier-1 chain), all nohup setsid.
- casesets --behaviour-only flag (Stage-2 Set C).

## Scope cut (coordinator 2026-10-17 ~02:00Z)
- Tier 1 = sweep + joint + DLA + direct; heads only for HEADS_NEEDED (e1_a2e, e3_e2a x qwen3 / mixtral_bos) and '[p6-circuit] NEED HEADS'
  lines; no grids. qwen2_57b adds joint_shared (scripts/ext14_scan_joint_shared.py). BASE READY = W2 + W4.
- Stage-1 chain restarted (pid 2400253 group) after stopping the old one while its next job was queued.

## Qwen2-57B hold (02:37-02:50Z)
- Watcher v2 matched a quoted 'PROTOCOL qwen2_57b' in a coordinator line and started qwen2_57b; cancelled before any GPU job started.
  Watcher v3 (scripts/ext14_chain_base_ready3.sh) only counts lines that start with '<UTC> [p6-engine] ...'. ext14_jobs.py PROGRESS
  patterns (REVIEW OK / VOID / NEED HEADS) are now line-anchored too.

## All-strata cells (coordinator 03:15Z)
- <cell>_all for e1_a2e / e1_e2a / e1p_*: Stage-1 e1_a2e_all / e1_e2a_all (mixtral_bos, qwen3) registered + chain; Stage-2 via
  scripts/ext14_chain_base_all.sh <proto> (armed for mistral7b_bos (done: 3 cells + 1 behaviour only), qwen3_4b, qwen3_8b, qwen2_7b,
  qwen2_57b, qwen3_14b). casesets: ALL_STRATA / TWINS, assoc_own / assoc_counterpart columns, twin-preferred items first.

## Second resume (06:45Z)
- All chains alive, no failures. SMOKE OK: mistral7b_bos (e3p3_a2e, 1.2 min), qwen3_4b (e1_a2e, 1.1), qwen3_8b (e1_a2e, 1.7).
- qwen2_7b fp32 verified (Engine applies models.py act_dtype by repo; scan meta now records engine_protocol). qwen3_14b / qwen2_7b
  orchestrators in progress; qwen3_1p7b scans running; qwen2_57b held (ready3 waits for a line starting '<UTC> [p6-engine] PROTOCOL qwen2_57b').
- Status counts posted in PROGRESS 06:5xZ.

## Throughput fix (07:0xZ)
- ext14_jobs.py emits one bundled job per call (ext14-<cell>-<proto>-t1 -> scripts/ext14_chain_base_cell.sh); --stages gives the per-stage
  list used inside the job; bundles 2 (Stage 1) / 4 (dense) / 1 (qwen2_57b); per-cell PID locks results/p6_<cell>_<proto>/.lock_t1.

## Third resume (15:00Z)
- Remaining tier 1: qwen3 acon_a2e (analysis pending), rsite_e1_a2e, sc_agree_a2e, sc_conflict_stmt_a2e; mixtral_bos rsite_e1_a2e,
  sc_agree_a2e (analysis pending); qwen2_57b l_ant (running), l_prop, l_cf, l_ioi. Missing heads e1_e2a/mixtral_bos queued
  (ext14-e1_e2a-mixtral_bos-t1heads, then re-analysis).
- build_report.md now has the status matrix (cell x model), notes and the problems log (scripts/ext14_build_qa.py --report).
- When all done: python scripts/ext14_build_qa.py && python scripts/ext14_build_qa.py --report; final report.

## DONE (16:51Z)
- All cells BASE READY / pooled / behaviour only / anchor; final build report data/dial/build_report.md; no chains running.

## Running (detached) — none
- Tier-1 chain scripts/ext14_chain_base.sh (pid 3560931, log logs/ext14_base_chain.log; out logs/ext14_base_chain_stage1.out): 23 cells x 2
  models, priority loop (first runnable cell next; unregistered / review-pending cells deferred).
- Scans: scripts/ext14_chain_base_scan2.sh a2eP3 (qwen3 7 cells, mixtral 3 cells), dialA (both); ext14_chain_base_scan.sh dialE_core
  (mixtral core2 pending).

## Next
1. When the a2eP3 + mixtral core2 scans are done: python scripts/ext14_casesets.py <proto> --plan (-> data/dial/build/need_ext_<proto>.txt),
   then bash scripts/ext14_chain_base_scan2.sh dialE_ext ext "$(paste -sd, data/dial/build/need_ext_<proto>.txt)" <proto>,
   then python scripts/ext14_casesets.py <proto> --cells e1_a2e,e1_e2a,e3p3_a2e,e3_a2e,e3_e2a,e1p_a2e,e1p_e2a,e2_a2e,e2_e2a,sc_agree_a2e,sc_conflict_a2e,rsite_e1_a2e
   (registers -> the tier-1 chain picks them up).
2. Dial A: after the dialA scans, python scripts/ext14_casesets.py <proto> --cells e0_a2e,e0_e2a,aagr_a2e,aagr_e2a,acon_a2e,acon_e2a.
3. Watch the first BASE READY (analysis log logs/ext14_base_analyze_<cell>_<proto>.log); then build_report (python scripts/ext14_build_qa.py --report).
4. Stage 2 on '[p6-engine] MODEL READY <proto>' lines: add PROTOS entries (family: mistral7b_bos -> mistral, qwen* -> qwen), probes, scans, case sets, tier 1.

## Open problems
- l_ind/mixtral_bos insufficient (Mixtral at chance on the 2-pair induction format); coordinator decision pending.
- Qwen3 E2 cells near 0 % margin (a2e 0.3 %, e2a 4 % in P0 / original frames): expected 'insufficient'.
- Accidental smoke-test write into results/p5_dir_f1_qwen3/base_summary.json was restored with git checkout (19:32Z); test mode writes to scratch.
