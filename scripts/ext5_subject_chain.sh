#!/usr/bin/env bash
# ext5-subject GPU chain (F4, subject-last-token column): verification gate, three subject-site sweeps (paper case sets),
# then per model the expert pass at the subject-site peak layers plus the final-token hypothesis layers. Every GPU step
# through scripts/gpu_queue.sh. Run dirs: results/{qwen3_bos,mixtral_nobos,mixtral_bos}_subject. Detach with
#   setsid nohup bash scripts/ext5_subject_chain.sh > logs/ext5_subject_chain.log 2>&1 &
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
log() { echo "$(date -u +%FT%TZ) $*"; }
. .venv/bin/activate; export HF_HOME=/opt/dlami/nvme/hf
log "chain start"
if [ "${SKIP_VERIFY:-0}" != "1" ]; then
  $Q ext5subj-verify -- python scripts/ext5_subject_verify.py 20 > logs/ext5_subject_verify.log 2>&1 || log "verify failed rc=$?"
fi
python scripts/ext5_subject_gate.py > logs/ext5_subject_gate.log 2>&1 || { log "GATE FAILED: sweeps not started"; exit 1; }
log "gate passed"

run_model() {  # model run base extra-flags fixed
  local model=$1 run=$2 base=$3 flags=$4 fixed=$5
  if [ ! -f results/$run/sweep_summary.json ]; then
    $Q ext5subj-sweep-$run -- python scripts/ext5_subject_sweep.py $model --out $run --base-run $base --sets paper $flags \
      > logs/ext5_sweep_$run.log 2>&1 || { log "$run sweep failed rc=$?"; return 1; }
  else log "$run sweep exists, skipping"; fi
  local layers
  layers=$(python scripts/ext5_subject_select.py $run --n 3 --fixed $fixed) || { log "$run select failed"; return 1; }
  log "$run expert layers: $layers"
  if [ ! -f results/$run/expert_meta.json ] || ! grep -q '"complete": true' results/$run/expert_meta.json; then
    $Q ext5subj-expert-$run -- python scripts/ext5_subject_expert.py $model --out $run --layers $layers --fixed $fixed $flags \
      > logs/ext5_expert_$run.log 2>&1 || { log "$run expert failed rc=$?"; return 1; }
  else log "$run expert pass exists, skipping"; fi
}
run_model qwen3 qwen3_bos_subject qwen3 "" "44:69,42:115"
run_model mixtral mixtral_nobos_subject mixtral_nobos "--no-special-tokens" "19:2,19:6,18:1"
run_model mixtral mixtral_bos_subject mixtral "" "19:2,19:6,18:1"
log "chain done"
