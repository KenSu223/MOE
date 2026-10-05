# ext7-controls status (agent ext7-controls)

## Done (11:20Z) — all tasks complete, nothing to redo
- task 1 CounterFact STR attention sweep: results/{qwen3,mixtral_bos}_str_attnsweep, verify results/verify_ext7_controls_cf_olmoe.json
- W7 role swap: data/wino_role/, results/wino_role_<proto> (scan), wino_role_<proto>_str (W2, W4, direct), wino_role_<proto>_grid (W3)
- W8 IOI: data/ioi/, results/ioi_<proto> (scan, attn_names.npz), ioi_<proto>_{s2io,s1io} (W2, W5, W4, direct), *_grid (W3)
- three-task table, section results/sections/ext7_controls.md, summary results/ext7_controls_summary.json
- rebuild: python scripts/ext7_cf_analyze.py; python scripts/ext7_role_analyze.py; python scripts/ext7_ioi_analyze.py;
  python -m moetrace.ext7_controls section

## Running
- scripts/ext7_role_ioi_chain.sh finishes its last six grid steps as no-ops (grids already complete; skip check in
  moetrace/ext7_controls.run_grid_positions). Nothing else.

## Next
- none (coordinator: assemble into EXTENSIONS_REPORT.md; rerun the section if ext8 / ext7-wino numbers change)
