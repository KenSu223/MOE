#!/usr/bin/env bash
# ext9 knockout chain: every step is one GPU-queue job (lock shared with the other agents); resumable (each pass is saved,
# completed passes are skipped). Waits for the OLMoE driver smoke test (job ext9-ko-smoke-olmoe) to succeed first.
# Order = priority: full (baseline, targets, population sets) for both models, then the random sets (+ null), the
# same-layer controls, the final-only / zero-mode conditions.
#   nohup setsid bash scripts/ext9_chain.sh > logs/ext9_chain.log 2>&1 &
set -u
cd /home/ubuntu/MOE
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
if [ "${EXT9_SKIP_SMOKE_WAIT:-0}" != 1 ]; then
  until grep -q "END   ext9-ko-smoke-olmoe" logs/gpu_queue.log; do sleep 20; done
  if ! grep "END   ext9-ko-smoke-olmoe" logs/gpu_queue.log | tail -1 | grep -q "rc=0"; then echo "$(ts) chain: smoke failed; stop"; exit 1; fi
fi
for phase in full random samelayer final; do
  for m in qwen3 mixtral; do
    echo "$(ts) chain: $m $phase"
    for attempt in 1 2; do
      bash scripts/gpu_queue.sh "ext9-ko-$m-$phase" -- python scripts/ext9_knockout_run.py "$m" --phase "$phase" && break
      echo "$(ts) chain: $m $phase attempt $attempt failed"
    done
  done
done
echo "$(ts) chain: GPU work done; analysis"
python scripts/ext9_knockout_analyze.py && python scripts/ext9_knockout_text.py
echo "$(ts) chain: done"
