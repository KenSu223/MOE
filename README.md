# Expert-aware causal tracing in sparse MoE LMs: a full-scale reproduction

Independent re-implementation and full rerun of **Lu, Modarressi, Liu, Schütze (2026), "Expert-Aware Causal Tracing of
Factual Recall in Sparse MoE Language Models"** ([arXiv:2606.03780](https://arxiv.org/abs/2606.03780)) on the paper's
exact models, Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1, in bf16, on a single 24 GB A10G. No code was released with the
paper; everything here was written from the paper text.

**Outcome.** Both models reproduce. Qwen3 matches the paper on every table with tokenizer defaults (layer 44,
expert L44E069, all validation means inside the paper's 95% intervals). Mixtral matches the paper once prompts are
tokenised **without a BOS token**: L19E006 is the sole recurrent candidate, its clean-active counts (91/128 discovery,
83/128 validation) equal the paper's exactly, and its specificity is negative while routed coalitions recover the
layer-level effect. With BOS, routing shifts and a different expert (E002) becomes recurrent and positively specific.

The full report with all 16 tables, both Figure 1 variants and the paper-versus-ours comparison is
[results/REPORT.md](results/REPORT.md). The timeline of the run is [logs/PROGRESS.md](logs/PROGRESS.md).

| Model | Run | Layer | Layer rescue | Expert | Expert rescue | Specificity |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | paper | L44 | +0.901 [+0.752, +1.053] | L44E069 | +0.463 [+0.344, +0.590] | +0.400 [+0.276, +0.533] |
| Qwen3-30B-A3B-Base | ours | L44 | +0.941 [+0.778, +1.124] | L44E069 | +0.503 [+0.362, +0.661] | +0.450 [+0.310, +0.608] |
| Mixtral-8x7B-v0.1 | paper | L19 | +0.457 [+0.331, +0.579] | L19E006 | +0.099 [+0.018, +0.175] | −0.175 [−0.284, −0.072] |
| Mixtral-8x7B-v0.1 | ours, BOS (tokenizer default) | L19 | +0.571 [+0.461, +0.692] | L19E002 | +0.352 [+0.257, +0.458] | +0.205 [+0.109, +0.308] |
| Mixtral-8x7B-v0.1 | ours, no BOS | L19 | +0.446 [+0.318, +0.569] | L19E006 | +0.073 [+0.007, +0.140] | −0.171 [−0.262, −0.082] |

## How it runs on one 24 GB GPU

The protocol is forward-only, uses 10 to 20 token prompts and intervenes only at the final position, so the model never
has to be resident. `moetrace/engine.py` is a **layer-streaming executor**: one decoder layer's weights are on the GPU
at a time (streamed from NVMe with a prefetch thread), and every run of a pass advances together, layer by layer:

- *prefill rows* (clean and noised prompts) record, at the final position of each layer, the MoE-block input and
  output, the routing decision and every routed expert's contribution c_e = w_e · E_e(x);
- *wavefront rows* (single tokens) implement each intervention: a run patched at layer l equals the noised run except
  at the final position from l upward, so it starts as `h_noised_after_l + v` and attends to the noised run's K/V for
  the earlier positions. The intervention vector v is built in-pass (layer patch: MoEOut_clean − MoEOut_noised;
  expert patch: c_e^clean − c_e^noised; scaled and coalition variants).

One pass over Qwen3 (61 GB) takes about 60 s, over Mixtral (93 GB) about 95 s; the whole study is about 20 passes.

## Reproducing

```bash
python3.14 -m venv .venv && . .venv/bin/activate
pip install torch transformers safetensors "huggingface_hub[hf_xet]" accelerate numpy scipy pandas pyarrow matplotlib tqdm
export HF_HOME=/path/with/160GB/free            # checkpoints: Qwen/Qwen3-30B-A3B-Base, mistralai/Mixtral-8x7B-v0.1, allenai/OLMoE-1B-7B-0125
bash scripts/download_models.sh                  # edit HF_HOME inside if needed
curl -L -o data/counterfact.json https://rome.baulab.info/data/dsets/counterfact.json   # if not present

python scripts/verify_olmoe.py                   # engine vs transformers on the OLMoE pilot
python scripts/run_filter.py qwen3               # filter scan -> results/qwen3/case_sets.json (paper IDs + our strict/relaxed sets)
python scripts/run_sweep.py qwen3                # layer sweep, all sets
bash   scripts/auto_expert.sh qwen3              # expert pass at the selected layer(s)
python scripts/run_noise.py qwen3                # Table 13
python scripts/run_filter.py mixtral && python scripts/run_sweep.py mixtral && bash scripts/auto_expert.sh mixtral
# paper protocol for Mixtral (no BOS), paper case set only:
mkdir -p results/mixtral_nobos && cp results/mixtral/case_sets.json results/mixtral_nobos/
python scripts/run_sweep.py mixtral --sets paper --out mixtral_nobos --no-special-tokens
python scripts/run_expert.py mixtral --layers 19 --out mixtral_nobos --no-special-tokens
python scripts/hf_reference_check.py qwen3 5     # optional, slow: transformers with CPU/disk offload
python scripts/build_final_report.py             # tables, figures, REPORT.md
```

`scripts/chain1.sh`, `chain2.sh`, `chain3.sh` are the exact sequences that were run.

## Layout

- `moetrace/` engine (`engine.py`, `weights.py`, `arch.py`), data and protocol (`data.py`, `noise.py`, `protocol.py`),
  statistics (`stats.py`), post-processing (`analysis.py`), tables/figures/report (`report.py`, `write_report.py`),
  verification (`verify.py`).
- `scripts/` pass CLIs, verification scripts, chains, the headless-agent supervisor used for the overnight run.
- `results/` `REPORT.md`, `tables/` (Tables 1 to 16 as Markdown and CSV, plus the Mixtral no-BOS tables),
  `figures/` (Figure 1 with tokenizer defaults and with the no-BOS Mixtral run), and one directory per run with
  row-level Parquet files (`sweep_rows`, `expert_rows`, `sweep_routing`, `filter_scan`, ...).
- `data/` CounterFact (21,919 records) and the paper's exact Table 8 case IDs (`paper_case_ids.json`).
- `PLAN.md` the feasibility analysis and plan; `HANDOFF.md` the brief the unattended agent executed;
  `logs/PROGRESS.md` the timestamped log of everything that happened.

## Protocol details that the paper leaves open

Recorded in `results/REPORT.md` sections 4, 6, 6b and 7. The two that matter: Mixtral prompts without BOS (decides
the Mixtral expert-level result), and object tokens resolved without a leading space when that is a single token
(reproduces the paper's Qwen3 case funnel at 95% agreement; does not change any conclusion). Numerics are bf16 with
fp32 accumulation; the engine's deviation from transformers equals transformers' own bf16 noise (report section 3).

## Provenance

Paper: Lu, Modarressi, Liu, Schütze, arXiv:2606.03780 (LMU Munich). CounterFact: Meng et al., 2022
(rome.baulab.info). Models: Qwen (Apache-2.0), Mistral AI (Apache-2.0), AllenAI OLMoE (Apache-2.0).
The paper PDF and LaTeX source are not redistributed here.
