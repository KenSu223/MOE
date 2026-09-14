# CLAUDE.md — project memory for agents working in /home/ubuntu/MOE

Read this first. It is the entry point for any new agent session in this repository: what the project is, what has
already been done and with what result, how the codebase is used, and the rules that keep work here consistent.
Details live in the documents listed in section 8; this file is the map.

## 1. What this project is

A full-scale, independent reproduction of **Lu, Modarressi, Liu, Schütze (2026), "Expert-Aware Causal Tracing of
Factual Recall in Sparse MoE Language Models"** (arXiv:2606.03780). The paper released no code. We re-implemented the
protocol from the text and re-ran every table and Figure 1 on the paper's exact models, Qwen3-30B-A3B-Base and
Mixtral-8x7B-v0.1, in bf16, on this machine's single 24 GB A10G.

**Status: COMPLETE** (2026-09-03 16:34 UTC, marker `results/DONE`). Deliverable: `results/REPORT.md` (all 16 tables
paper-vs-ours, both Figure 1 variants, verification, deviations). Repository: github.com/KenSu223/MOE (branch `main`).

## 2. Results in one screen

| Model | Run | Layer | Layer rescue (val) | Expert | Expert rescue | Specificity |
|---|---|---|---|---|---|---|
| Qwen3 | paper | L44 | +0.901 [+0.752, +1.053] | L44E069 | +0.463 | +0.400 |
| Qwen3 | ours (tokenizer defaults) | L44 | +0.941 | L44E069 | +0.503 | +0.450 |
| Mixtral | paper | L19 | +0.457 [+0.331, +0.579] | L19E006 | +0.099 | −0.175 |
| Mixtral | ours, BOS (tokenizer default) | L19 | +0.571 | **L19E002** | +0.352 | **+0.205** |
| Mixtral | ours, **no BOS** | L19 | +0.446 | L19E006 | +0.073 | −0.171 |

- **Qwen3 reproduces on every table** with tokenizer defaults; all validation means inside the paper's 95% CIs;
  Appendix D stability 25/25 and 5/5 relation folds as in the paper.
- **Mixtral reproduces only when prompts are tokenised WITHOUT the BOS token** (`add_special_tokens=False`). Then
  L19E006 is the sole recurrent candidate, its clean-active counts (91/128 discovery, 83/128 validation) equal the
  paper's exactly, specificity is negative, coalitions recover the layer effect (Tables 5, 6, 11, 16 all match). With
  BOS, routing at the final position shifts: E002 becomes recurrent (76/128) and positively specific and gets selected.
  Conclusion: the paper's Mixtral protocol omitted BOS (not stated in the paper). Qwen3's tokenizer adds no BOS, so
  Qwen3 is unaffected.
- Second under-specified detail: the paper most likely resolved object tokens as `tok(obj)` when that is a single
  token, else the leading-space token (`--token-rule paper_like`; 95% agreement with the paper's Qwen3 case funnel vs
  77% for our default). It does not change any conclusion; reported as a secondary run.
- Numerics: our engine's deviation from transformers equals transformers' own bf16 noise (fp32 CPU reference on
  OLMoE); big-model checks vs transformers-with-offload on 5 prompts each: max |ΔΔ| 0.25 (Qwen3), 0.06 (Mixtral).

## 3. What is NOT done / known gaps

- Mixtral **relaxed-filter set (Tables 14, 15)** was run only under the BOS default. A no-BOS filter scan → sweep →
  expert pass (about 4 passes, 7 min GPU) would complete it. Runs: `run_filter.py mixtral --out mixtral_nobos
  --no-special-tokens` then sweep/expert with the same flags.
- The paper's relaxed 512-case set used a different record order (only 15 overlaps with its strict set); its IDs are
  unpublished, so Table 15 subset sizes cannot match.
- Zero-row definition (Table 3), fold assignment (Table 12), split function (Appendix D), active-random draws and the
  exact noise samples are unrecoverable; only selections and CI-level agreement are comparable.

## 4. Machine and environment (verify before relying on it)

- AWS g5.4xlarge: 1× A10G 24 GB, 16 vCPU, 62 GB RAM. An unrelated user process (`repo-world-model`) holds ~9 GB RAM;
  do not kill processes you did not start.
- Python venv: `. /home/ubuntu/MOE/.venv/bin/activate` (Python 3.14, torch 2.14+cu130, transformers 5.16.1,
  safetensors, pandas, pyarrow, scipy, matplotlib).
- **Always `export HF_HOME=/opt/dlami/nvme/hf`** before any script. Checkpoints (Qwen3 61 GB, Mixtral 93 GB, OLMoE
  39 GB fp32) live on the NVMe instance store `/opt/dlami/nvme` (521 GB, 1.6 GB/s). It is **ephemeral**: an instance
  stop wipes it; re-download with `scripts/download_models.sh` (~20 min).
- Root disk holds code and results (`results/` is 9 MB). Never write weights or offload folders to the root disk.
- Before launching GPU work: `pgrep -af "run_sweep|run_expert|run_filter|hf_reference|chain"` and `nvidia-smi`.

## 5. Codebase map and how to use it

Engine idea: one decoder layer resident on the GPU at a time (streamed from safetensors with prefetch); all runs of a
pass advance layer by layer. *Prefill rows* = full clean/noised prompts (record final-position MoE in/out, routing,
per-expert contributions c_e = w_e·E_e(x)). *Wavefront rows* = single tokens implementing interventions from layer l
upward, attending to the parent noised run's K/V. One pass = one read of the checkpoint (Qwen3 ~60 s, Mixtral ~95 s)
regardless of how many interventions are batched. See README section "How it runs on one 24 GB GPU".

`moetrace/`
- `arch.py` ArchSpec from config.json (Qwen3-MoE, Mixtral, OLMoE key templates, q/k-norm variant, top-k renorm).
- `weights.py` CheckpointStore + LayerStreamer (mmap safetensors, per-layer expert stacking, pinned double buffer).
- `engine.py` `Engine(repo)`, `PrefillSpec(ids, true_id, foil_id, subject_pos=None, noise=None)`,
  `SpawnSpec(layer, parent_row, clean_row, kind, ...)`; kinds: `zero`, `layer`, `expert`, `expert_scaled`,
  `coalition_clean`, `coalition_union`. `eng.run(prefill, spawns, record_routing=...)` → deltas, logits, routing.
- `data.py` CounterFact loading, seed-0 shuffle, `prepare_case(rec, tok, token_rule="space"|"paper_like",
  special_tokens=True|False)`, filters (STRICT 1.0/0.5, relaxed 0.5/0.25), splits.
- `noise.py` σ = mult × embed std; per-case `torch.Generator(0 + case_id)`.
- `protocol.py` `cases_by_id`, `load_case_sets`, `active_controls` (random.Random(1000 + case_id)).
- `stats.py` `summarize` (mean, 5,000-resample percentile bootstrap CI, positive fraction, 10,000 sign-flip p), `fmt`.
- `analysis.py` post-processing: `load_model(run)`, `layer_analysis`, `expert_table`, `select_expert`
  (recurrence-first), `evaluate_expert`, `gate_matched_control`, `all_active_rank`, `active_pair_equal_norm`,
  `coalitions`, `stability_grid`, `relation_heldout`, `noise_table`, `funnel_check`.
- `report.py` `build()` → Tables 1–16 (md+csv) + `figure1()`; `alt_summary(suffix=...)` for secondary runs.
- `write_report.py` `write(res, tables, alt_table, extra_verdict, extra_sections)` → `results/REPORT.md`.
- `models.py` `MODELS` registry: `qwen3`, `mixtral`, `olmoe` → repo, label, n_controls, paper_layer, paper_expert.

`scripts/` (all take a model key; `--out <run_name>` selects `results/<run_name>/`; `--token-rule space|paper_like`;
`--no-special-tokens` = no BOS)
- `run_filter.py` filter scan → `case_sets.json` (keys `scan`, `strict`, `relaxed`, `paper`, `paper_check`).
- `run_sweep.py [--sets paper,strict,relaxed]` layer sweep → `sweep_rows`, `sweep_routing`, `sweep_cases`, `sweep_summary.json`.
- `run_expert.py --layers 44[,19]` expert pass → `expert_rows`, `expert_prefill_L*`. `auto_expert.sh` picks layers.
- `run_noise.py` σ ∈ {1,2,4} (Table 13). `hf_reference_check.py <model> 5` transformers-with-offload check (slow).
- `verify_olmoe.py`, `verify_olmoe_fp32.py` engine-vs-transformers verification on the pilot model.
- `mixtral_compare.py`, `mixtral_run_section.py` compare Mixtral runs / build the no-BOS report section.
- `build_final_report.py [--done]` rebuilds every table, both figures, secondary sections, REPORT.md (and DONE).
- `chain1/2/3.sh` the exact GPU sequences that were run; `run_agent.sh` headless-agent supervisor (section 7).

`results/` run directories: `qwen3`, `mixtral` (tokenizer defaults, all three case sets), `mixtral_nobos` (paper set,
no BOS), `qwen3_alt`, `mixtral_alt` (paper_like token rule, paper set), `olmoe` (pilot). Row-level Parquet schemas:
`sweep_rows` (case_id, kind clean|noised|layer, layer, logit_true, logit_foil, delta, rescue, ...);
`sweep_routing` (case_id, run, layer, slot, expert, weight, cnorm); `expert_rows` (case_id, layer, kind, expert,
partner, alpha, rescue, clean_active, noised_active, clean_weight, n_clean_active, ...). `summary.json` and
`mixtral_compare.json` hold the analysed numbers.

Typical new experiment: copy `results/<base>/case_sets.json` into a new run dir → `run_sweep.py --out <run>` →
`run_expert.py --layers <L> --out <run>` → analyse with `analysis.load_model("<run>")` → add to the report via
`report.alt_summary(suffix=...)` or a section script → `build_final_report.py`.

## 6. Conventions and rules for work here

- The paper's Table 8 case IDs (`data/paper_case_ids.json`) are the PRIMARY case set; our own strict/relaxed sets are
  secondary. Keep the discovery/validation assignment from the paper.
- Keep the main tables on the documented default protocol and report variants (no BOS, paper_like) as labelled
  secondary runs. For any Mixtral experiment meant to match the paper, use `--no-special-tokens`.
- Prefer counts and set membership (activity counts, funnel pass counts, selections) as "fingerprints" when comparing
  with the paper; means carry bf16 noise (per-case rescue ±0.1–0.6, 128-case means ±0.02–0.05).
- No quantization, ever: bf16 weights and activations are the object of study.
- Log every milestone to `logs/PROGRESS.md` (append-only, UTC timestamps). Save row-level Parquet after every pass.
- Commits: trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Pushing requires the user's GitHub
  credentials (VSCode source-control push or `gh auth login` in their terminal); agents in this shell have none.

## 7. Unattended / headless workflow (see PLAYBOOK.md)

Roles: an interactive session does research, planning and handoff; a headless `claude -p` agent in tmux executes
`HANDOFF.md` until `results/DONE`; `scripts/run_agent.sh` supervises it (resumes the same session, parses usage-limit
reset times and sleeps until then). Rules learned the hard way: the **user** must start the supervisor (the permission
classifier blocks an agent from launching another Claude process); queue GPU work as detached `setsid nohup` chains
that outlive the agent; never stop a running chain to edit it, append a new chain instead; on resume, read
`logs/agent_status.txt`, `logs/PROGRESS.md`, `logs/chain*.log`, then the last tool calls in the newest
`logs/agent_run*.jsonl` to find the first unexecuted step.

## 8. Document map

- `README.md` public overview, headline table, how to reproduce.
- `PLAN.md` feasibility analysis (why layer streaming), full implementation checklist, assumptions.
- `HANDOFF.md` the operational brief the headless agent executed (model facts, engine design, protocol defaults).
- `PLAYBOOK.md` the unattended-run process, timeline of 2026-09-03, problems and fixes.
- `results/REPORT.md` the deliverable: sections 1–8 plus 6b (Mixtral without BOS).
- `logs/PROGRESS.md` timestamped log of everything that happened, including the handover.
- Paper PDF / LaTeX are gitignored (`2606.03780.pdf`, `paper.txt`, `paper_src/`); present locally on this machine.
