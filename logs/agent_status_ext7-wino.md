# ext7-wino agent status (WinoGrande STR study, Phase 3 W1-W6 + W4)

Updated: 2026-10-04T11:10Z — ALL DONE. Nothing running.

- GPU runs complete: verify (OLMoE), W2 sweeps, W6 expert passes, W3 grids (w1, w5), W5 heads, W4 joint rows, own DLA split,
  shared direct split (ext7_controls) for Qwen3 and Mixtral BOS.
- CPU: python scripts/ext7_wino_analyze.py && python scripts/ext7_wino_text.py regenerate summary JSON, tables, figures, section.
- Final report sent to the coordinator.
