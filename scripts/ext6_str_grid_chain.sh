#!/usr/bin/env bash
# ext6 STR layer x position grid chain (2026-09-28). Priority order requested by the user: Qwen3 and Mixtral with BOS,
# single-layer patches first, then the 5-layer sliding window (Zhang & Nanda Figure 4 setting); Mixtral without BOS
# only on the user's go-ahead (not in this chain). Every GPU job goes through scripts/gpu_queue.sh.
# Log: logs/ext6_str_grid_chain.log
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
Q="bash scripts/gpu_queue.sh"
echo "$(date -u +%FT%TZ) grid chain start"
$Q ext6-grid-qwen3-w1 -- python scripts/ext6_str_grid.py qwen3 --out qwen3_str --window 1 || echo "FAILED qwen3 w1"
$Q ext6-grid-mixtral_bos-w1 -- python scripts/ext6_str_grid.py mixtral --out mixtral_bos_str --window 1 || echo "FAILED mixtral_bos w1"
python scripts/ext6_str_grid_analyze.py > logs/ext6_str_grid_analyze_w1.log 2>&1 || echo "analysis w1 failed"
echo "$(date -u +%FT%TZ) single-layer grids done"
$Q ext6-grid-qwen3-w5 -- python scripts/ext6_str_grid.py qwen3 --out qwen3_str --window 5 || echo "FAILED qwen3 w5"
$Q ext6-grid-mixtral_bos-w5 -- python scripts/ext6_str_grid.py mixtral --out mixtral_bos_str --window 5 || echo "FAILED mixtral_bos w5"
python scripts/ext6_str_grid_analyze.py > logs/ext6_str_grid_analyze.log 2>&1 || echo "analysis failed"
echo "$(date -u +%FT%TZ) grid chain done"
