# p6-circuit status

Updated: 2026-10-17T17:30Z

## Done
- C1 DONE (chain finished 01:32Z; PROGRESS line "[p6-circuit] C1 DONE"): results/p6ci_c1_<task>/ for qwen3_cf, qwen3_wino,
  qwen3_f1, mixtral_cf, mixtral_wino, mistral7b_cf, mistral7b_wino; tables results/tables/ext14ci_c1_{neurons,agreement}.md,
  figure results/figures/ext14ci_c1_curves.png, summary key c1 (incl. dense sweeps and the Mixtral <-> Mistral neuron mapping).
- Upcycling weight check: results/p6ci_upcycle/upcycle.json (+ tables ext14ci_upcycle_{attn,ffn}.md after the next analysis).
- RelP: functional forward = HF (tiny fp32 exact; OLMoE bf16 within noise), gradients = finite differences; OLMoE smoke OK at
  22:15Z (the 21:10Z failure was an index bug, fixed). Conditional mean ablation for MoE added (experts routed in the live run
  replaced by their routed-token mean) after the smoke showed unconditional mean ablation wrecks OLMoE (F ~ 0 up to 30k nodes).

## Running chains (p6ci-* jobs are LOW priority in the queue)
- C3 Qwen3 COMPLETE (17:18Z; 8/8 tasks; results/tables/ext14ci_c3_qwen3.md). Finding: the COUNTRY circuit hurts the CounterFact
  language control as much as its own tasks (0.73 vs 0.70) -> CounterFact-recall circuit, not country-specific; SIZE/STATE and
  PERSON-ROLE are specific; N3 holds for SIZE (experts 0.98 vs heads 0.56) and PERSON (0.95 vs 0.82), not for COUNTRY (0.71 vs 0.82).
- C3 Mixtral: restarted 17:25Z (pid in logs/ext14ci_c3_chain3.pid, log logs/ext14ci_c3_chain3_mixtral_bos.log) with circuits rebuilt
  after the E1-e2a expert-DLA job (SIZE/STATE now 4 heads + 6 experts); 2 validation jobs queued; superseded files in
  results/p6ci_c3_data/superseded_mixtral_bos_noSSexperts/.
- Finalizer pid 2216044 (the duplicate 1733576 was stopped): waits for the Mixtral chain, runs scripts/ext14ci_all.py, posts C3 DONE.

## Next
- Nothing manual is required: the chains + finalizer complete C3 and rebuild results/sections/ext14_circuit.md.
- If resumed: check logs/ext14ci_c3_chain2_qwen3.log, logs/ext14ci_c3_chain3_mixtral_bos.log, logs/ext14ci_finalize.out; review
  the C3 tables (results/tables/ext14ci_c3_*.md) and the auto bullets in the section; adjust NOTES in scripts/ext14ci_text.py if needed.

## Open problems
- Mixtral PERSON_ROLE: IOI (64/64) + E1-a2e names (14/16 pairs) only; SIZE_STATE E1p-e2a 16/16 (pooled cells).
