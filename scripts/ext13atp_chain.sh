#!/usr/bin/env bash
# p5-atp chain (detached): engine truth -> HF estimates -> add-back -> analysis. Every GPU step goes through gpu_queue.sh.
# Usage: setsid nohup bash scripts/ext13atp_chain.sh > logs/ext13atp_chain.log 2>&1 &
# Steps whose outputs exist are skipped (safe to re-launch after a failure or a usage-limit stop).
set -uo pipefail
cd /home/ubuntu/MOE
export HF_HOME=/opt/dlami/nvme/hf
. .venv/bin/activate
R=results
say() { echo "$(date -u +%FT%TZ) [chain] $*"; }
fail() { say "FAILED: $*"; echo "$(date -u +%FT%TZ) FAILED $*" > logs/ext13atp_chain.status; exit 1; }
echo "$(date -u +%FT%TZ) RUNNING" > logs/ext13atp_chain.status

# J1 engine: IOI scan + case sets, truth for both tasks
need=""
[ -f results/p5_atp_data/case_sets_ioi.json ] || need="$need scan"
[ -f $R/p5_atp_wino/truth_rows.parquet ] || need="$need wino"
[ -f $R/p5_atp_ioi/truth_rows.parquet ] || need="$need ioi"
if [ -n "$need" ]; then
  say "J1 engine:$need"
  bash scripts/gpu_queue.sh p5atp-engine-truth -- bash -c "
    set -e
    [ -f results/p5_atp_data/case_sets_ioi.json ] || python scripts/ext13atp_engine.py scan-ioi
    [ -f $R/p5_atp_wino/truth_rows.parquet ] || python scripts/ext13atp_engine.py truth --task wino
    [ -f $R/p5_atp_ioi/truth_rows.parquet ] || python scripts/ext13atp_engine.py truth --task ioi
    python scripts/ext13atp_grad.py --task wino --time-budget 150
  " > logs/ext13atp-engine-truth.log 2>&1 || fail "J1 engine truth (logs/ext13atp-engine-truth.log)"
fi
say "J1 done"

# J2 HF estimates, one task per queue job, repeated until complete (time budget per job)
for t in wino ioi; do
  for attempt in 1 2 3 4 5 6; do
    [ -f $R/p5_atp_$t/hf_nodes.parquet ] && break
    say "J2 HF $t attempt $attempt"
    bash scripts/gpu_queue.sh p5atp-hf-$t -- python scripts/ext13atp_grad.py --task $t --time-budget 780 \
      >> logs/ext13atp-hf-$t.log 2>&1 || fail "J2 HF $t (logs/ext13atp-hf-$t.log)"
  done
  [ -f $R/p5_atp_$t/hf_nodes.parquet ] || fail "J2 HF $t incomplete after 6 attempts"
done
say "J2 done"

# J3 add-back (CPU dry run first: validates the joined table and the plan)
for t in wino ioi; do
  python scripts/ext13atp_engine.py addback --task $t --dry-run >> logs/ext13atp-addback-plan.log 2>&1 || fail "add-back plan $t"
done
if [ ! -f $R/p5_atp_wino/addback_rows.parquet ] || [ ! -f $R/p5_atp_ioi/addback_rows.parquet ]; then
  say "J3 add-back"
  bash scripts/gpu_queue.sh p5atp-engine-addback -- bash -c "
    set -e
    [ -f $R/p5_atp_wino/addback_rows.parquet ] || python scripts/ext13atp_engine.py addback --task wino
    [ -f $R/p5_atp_ioi/addback_rows.parquet ] || python scripts/ext13atp_engine.py addback --task ioi
  " > logs/ext13atp-engine-addback.log 2>&1 || fail "J3 add-back (logs/ext13atp-engine-addback.log)"
fi
say "J3 done"
if [ -f scripts/ext13atp_analyze.py ]; then
  python scripts/ext13atp_analyze.py > logs/ext13atp-analyze.log 2>&1 || fail "analysis (logs/ext13atp-analyze.log)"
fi
say "chain complete"
echo "$(date -u +%FT%TZ) DONE" > logs/ext13atp_chain.status
