# agent_status ext11-writer (Phase 4, Direction 11: writer vs computer experts)
Updated: 2026-10-05T07:58Z — ALL PARTS DONE (A, B, C); final report sent.
- Section: results/sections/ext11_writer.md (python scripts/ext11_writer_text.py regenerates it from results/ext11_writer_summary.json + tables).
- Re-run analysis only (CPU): python scripts/ext11_writer_partA.py; python scripts/ext11_writer_routing_analyze.py; python scripts/ext11_writer_vocab_analyze.py; python scripts/ext11_writer_text.py
- Runs: results/{qwen3,mixtral_bos}_writer_routing, results/{qwen3,mixtral_bos}_writer_vocab; raw /opt/dlami/nvme/moe_ext11/.
- GPU used: 12.0 min (Part C 8.0, Part B 4.0) of the 30-min budget.
