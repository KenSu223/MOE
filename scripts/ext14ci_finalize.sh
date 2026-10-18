#!/usr/bin/env bash
# ext14 circuit (p6-circuit): waits for the C3 chains (qwen3 chain2, mixtral_bos chain3), re-runs every analysis and the section
# (scripts/ext14ci_all.py) and posts the C3 DONE handshake with the headline numbers. CPU only.
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
until grep -q "c3 chain2 done" logs/ext14ci_c3_chain2_qwen3.log 2>/dev/null && grep -q "c3 chain3 done" logs/ext14ci_c3_chain3_mixtral_bos.log 2>/dev/null; do sleep 300; done
python scripts/ext14ci_all.py > logs/ext14ci_all.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) [p6-circuit] C3 DONE (auto-finalised; ext14ci_all rc=$rc; section results/sections/ext14_circuit.md, tables results/tables/ext14ci_c3_{qwen3,mixtral_bos}.md, summary key c3): $(python scripts/ext14ci_c3_headline.py)" >> logs/PROGRESS.md
