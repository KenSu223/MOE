#!/usr/bin/env bash
# p5-addback smoke test (2026-10-11): one queued job, Qwen3 then Mixtral on the pre-neutral F1 runs
# (results/wino_ref_F1_<proto>), 16 + 16 directed cases, greedy 3 / 2 steps -> results/p5_F1pre_<proto>_ab_smoke.
cd "$(dirname "$0")/.."
bash scripts/gpu_queue.sh ext13ab-smoke -- bash -c "python scripts/ext13ab_run.py qwen3 --smoke --greedy-steps 3 --max-passes 6 && python scripts/ext13ab_run.py mixtral --smoke --greedy-steps 2 --max-passes 5 --budget 50000" > logs/ext13ab-smoke.log 2>&1
echo "$(date -u +%FT%TZ) ext13ab-smoke rc=$?" >> logs/ext13ab_chain.log
