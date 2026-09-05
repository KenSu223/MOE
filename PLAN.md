# Reproduction plan: "Expert-Aware Causal Tracing of Factual Recall in Sparse MoE Language Models"

arXiv 2606.03780 (Lu, Modarressi, Liu, Schütze; LMU Munich / TU Darmstadt, June 2026).
Local copies: `2606.03780.pdf`, `paper.txt`, `paper_src/acl_latex.tex` (TeX source), `data/counterfact.json`
(21,919 records), `data/paper_case_ids.json` (the paper's exact 4 x 128 case IDs from Table 8).

## 1. Verdict on feasibility

**A full-scale reproduction with the paper's exact models (Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1) in bf16 is
feasible on this machine.** No scale-down is required for the main results.

The protocol is forward-only, uses short prompts (about 10 to 20 tokens), and intervenes only at the final token
position. The GPU never has to hold the whole model; it only has to hold one decoder layer at a time. A
layer-streaming executor that processes every clean run, noised run, and intervention variant for all 256 cases
together, one layer at a time, turns each experiment into a handful of sequential passes over the checkpoint. Each pass
reads the weights once from the 521 GB NVMe scratch disk (measured 1.6 GB/s), so a pass costs roughly one minute for
Qwen3 and one to two minutes for Mixtral. The whole experiment suite for one model is roughly 10 passes.

What is **not** feasible is the conventional approach the authors almost certainly used: load the model with
Hugging Face `from_pretrained`, register hooks, and run one forward pass per intervention. Neither model fits
(61 GB and 93 GB versus 24 GB GPU plus about 45 GB of free RAM), and CPU-offloaded forwards would take 10 to 20 s each,
times about 20,000 forwards per model.

Expected fidelity: same weights, same precision, same protocol. Bit-level results will still differ from the paper
(different GPU kernels, batched bf16 accumulation, our re-implementation versus their HF-version hooks), so a few
borderline cases in the filtering funnel may flip. Two safeguards: we run the paper's exact case IDs (Table 8) as the
primary case set, and our own filter funnel as a secondary check of how well the funnel itself reproduces.

Wall-clock estimate: downloads about 75 min (35 MB/s single stream; likely faster with parallel download), engine
development and verification 1 to 1.5 days, each model's full run 1 to 2 hours, reporting half a day.

## 2. Released code

None found. Checked: the arXiv source tarball (LaTeX, bib, one figure PDF; no scripts, no URLs), the arXiv abstract
page (no ancillary files), the Hugging Face paper page (404), GitHub repository search for "expert-aware causal
tracing", "L44E069", "moe causal tracing counterfact" (no hits), GitHub users yuetianlu (an unrelated iOS developer)
and LuYuetian (zero repos), and the MCML publication page (links only to arXiv). The paper refers to a "supplementary
artifact" with scripts and row-level results that was submitted to the venue but is not public. The paper notes
GPT-5.5 was used as a coding assistant, which suggests a self-contained research codebase rather than a public
framework. Option: email the corresponding author (yuetianlu@cis.lmu.de) for the artifact; it would let us diff
row-level results directly.

Everything below assumes we implement from the paper.

## 3. Hardware versus requirements

| Resource | This machine (g5.4xlarge) | Needed for the exact models |
|---|---|---|
| GPU | 1 x NVIDIA A10G, 24 GB, bf16 | One resident decoder layer: Qwen3 1.3 GB, Mixtral 2.9 GB, plus embeddings/head under 1.3 GB. Fits with room for large batches. |
| RAM | 62 GB total, about 45 GB free (another project holds 9.3 GB) | Page cache for the streamed checkpoint; our own state is under 5 GB. |
| Disk (fast) | 521 GB free NVMe instance store at `/opt/dlami/nvme`, 1.6 GB/s read | Qwen3 61.1 GB + Mixtral 93.4 GB = 155 GB. Fits together. Ephemeral: wiped if the instance is stopped. |
| Disk (root) | 153 GB free EBS, 131 MB/s | Too slow and too small for both checkpoints at once. Code, results, and figures live here. |
| Network | 35 MB/s single stream from Hugging Face | 30 min (Qwen3) + 45 min (Mixtral) worst case. |
| Software | Python 3.14, no torch installed | torch 2.14 (cp314 wheels exist), transformers 5.16, safetensors, huggingface_hub, numpy, scipy, pandas, matplotlib. No quantization libraries needed. |

Both checkpoints are public and ungated.

## 4. What the paper does (implementation checklist)

Models: Qwen3-30B-A3B-Base (48 MoE layers, 128 experts, top-8, norm_topk_prob=true, hidden 2048, expert inter 768,
no shared expert). Mixtral-8x7B-v0.1 (32 MoE layers, 8 experts, top-2, softmax then renormalize, hidden 4096,
inter 14336). Layer indices are 0-based (L44 of 0..47, L19 of 0..31).

Data and metric
- CounterFact records shuffled with seed 0; scan until 256 usable cases; random split (seed 0) into 128 discovery,
  128 validation. Appendix D varies the split seed over {0..4}.
- Prompt = relation template with subject filled in; object not in the prompt. True object = target_true, foil =
  target_new. Both must be single tokens (with leading space) in the model's tokenizer.
- Delta(x) = logit(true) - logit(foil) at the next-token position (raw logits).
- Filters: clean margin Delta_clean >= 1.0; subject-noise drop Delta_clean - Delta_noised >= 0.5. Relaxed
  (Appendix F): 0.5 and 0.25, giving 512 cases.
- Corruption: Gaussian noise on the subject-token input embeddings, sigma = 3.0 x std of the embedding matrix,
  one draw per case with seed 0 + case_id.

Stage 1: layer-level tracing
- Cache the final-position MoE-block output at every layer in the clean run; in the noised run replace the layer-l
  final-position MoE-block output with the clean one. Rescue_l = Delta_patched,l - Delta_noised.
- Sweep all layers on discovery cases, pick the argmax, evaluate as a fixed hypothesis on validation cases.
  Figure 1 left shows the validation-rescue curve over all layers for both models.

Stage 2: expert-level tracing (at the selected layer)
- c_e(x) = MoEOut(x) - MoEOut_without_e(x): ablation difference under the original top-k routing, no rerouting, no
  renormalization. For these architectures this equals w_e * E_e(h) but we implement it literally.
- delta_e = c_e(x_clean) - c_e(x_noised); add delta_e to the noised MoE-block output at the final position.
  If e is not routed in a run, its contribution in that run is zero.
- Recurrence-first selection on discovery cases: candidates must be clean-active in >= 64 of 128 discovery cases;
  choose the highest mean rescue. Report the all-case mean (zero rows retained, Table 3) and the active-only mean.
- Specificity Spec = Rescue(e*) - mean Rescue(active-random controls), controls drawn from the other clean-active
  experts at the same layer for the same prompt: 3 per case for Qwen3, 1 per case (the unique other) for Mixtral.
- Statistics: means, positive fraction, 5,000-resample bootstrap CIs, 10,000-sample two-sided sign-flip tests.

Controls and appendices
- Table 3: selected-expert activity counts (discovery, validation, zero rows).
- Table 5: relation-wise breakdown. Table 4: retained relation counts.
- Table 6: layer-sweep sharpness (top vs next layer). Table 7: expert/layer ratios.
- Appendix C.1 (Qwen3): gate-weight-matched control (other clean-active expert with closest clean router weight),
  equal-norm control (scale both patch vectors to the smaller norm), on the 116 cases where L44E069 is clean-active.
  Table 10: rank of L44E069 among all clean-active layer-44 experts per case.
- Appendix C.2 (Mixtral): equal-norm active-pair check on the 83 anchor-active validation cases.
- Appendix D (Qwen3): selection stability over split seeds {0..4} x recurrence thresholds {32,48,64,80,96}; plus
  relation-held-out selection over 5 relation folds (Table 12). Both are post-processing over saved rows if we store
  the rescue of every clean-active expert for every case.
- Appendix E (Qwen3): noise-scale sensitivity sigma in {1,2,3,4} with L44/E069 fixed (Table 13).
- Appendix F: relaxed-filter 512-case rerun (Tables 14, 15, including overlap with the strict set).
- Appendix G (Mixtral): coalition patches D_top2 = sum over clean-routed experts of delta_e, and D_union over the
  union of clean- and noised-routed experts (for experts not clean-routed, delta_e = -c_e(x_noised)). Table 16.
- Figure 1: three panels (layer sweep, single-expert specificity, Mixtral coalition rescue).

Target numbers (validation, 128 cases each)

| Model | Layer rescue | Expert | Expert rescue | Spec |
|---|---|---|---|---|
| Qwen3 | L44 +0.901 [+0.752, +1.053] | L44E069 | +0.463 [+0.344, +0.590] | +0.400 [+0.276, +0.533] |
| Mixtral | L19 +0.457 [+0.331, +0.579] | L19E006 | +0.099 [+0.018, +0.175] | -0.175 [-0.284, -0.072] |

Mixtral coalitions: clean top-2 +0.461 [+0.343, +0.572], routing union +0.490 [+0.367, +0.613], both 0.75 positive.
Sweep sharpness: Qwen3 next-best L42 +0.592 (gap 0.309); Mixtral top L21 +0.496, L19 +0.457 (gap 0.038).
Selected-expert activity: L44E069 clean-active 112/128 discovery, 116/128 validation, 26 zero rows; L19E006 91, 83, 54.

## 5. Design: layer-streaming executor

One decoder layer is resident on the GPU at a time. For layer l we load its weights from the safetensors shards on
NVMe (stacking per-expert gate/up/down matrices), run every pending unit of work through it, then move on. Two kinds
of work units:

1. Full-sequence rows (prefill): clean and noised prompts for the filter scan and the caching pass. At each layer we
   record, for the final position, the MoE-block input and output, the routing indices and weights, and every routed
   expert's contribution. We also keep each row's K/V for every layer so that later single-token work can attend to it.
2. Single-token rows (wavefront): a patched run differs from its noised run only at the final position, and causal
   attention means the other positions are unaffected. So a layer-l patch is: take the noised run's final-position
   residual entering layer l, apply the intervention at layer l's MoE block, and run only that one token through
   layers l..L-1 attending to the noised run's cached K/V. At layer l the batch holds every case's variants whose patch
   layer is <= l; for the Qwen3 sweep that is at most 256 x 48 = 12,288 rows of size 2048 at the last layer. The whole
   layer sweep for all cases is one pass over the weights.

Interventions supported at the MoE block of the final position: replace output (layer patch), add a vector (expert
patch, coalition patch, scaled patch), suppress one expert under fixed routing (ablation difference), and record only.

Cost per pass: I/O about 40 s (Qwen3) or 60 s (Mixtral) cold from NVMe, less once the page cache is warm; compute
well under a minute for Qwen3 and a few minutes for Mixtral at the largest batches. Weight loading for layer l+1 is
prefetched on a background thread while layer l computes.

Attention, RMSNorm, RoPE, Qwen3's q/k norms, and Mixtral's router are re-implemented from the HF modeling files
(pinned copies fetched from transformers main) so that we control residency and interventions. Correctness is checked
against the reference HF implementation, see section 7.

## 6. Code layout (`/home/ubuntu/MOE/moetrace/`)

- `data.py`: CounterFact loading, seed-0 shuffle, prompt construction, subject token span (ROME-style character
  offsets to token indices), single-token object check, filter funnel, discovery/validation split.
- `weights.py`: safetensors index to per-layer lazy loader; per-architecture key maps (Qwen3-MoE, Mixtral, OLMoE);
  expert stacking; background prefetch.
- `engine.py`: the layer-streaming executor with the two row types, K/V store, MoE hook points, and intervention ops.
- `noise.py`: embedding-std sigma, per-case seeded Gaussian draw, subject-span injection.
- `protocol.py`: layer sweep, selection rules, expert patches, active-random / gate-matched / equal-norm controls,
  coalitions, relaxed filter, noise sweep.
- `stats.py`: bootstrap CI, sign-flip test, positive fraction.
- `report.py`: Tables 1 to 16 as CSV and Markdown, Figure 1 in matplotlib.
- `verify.py`: equivalence and invariance tests.
- `scripts/`: `download.py`, `run_filter.py`, `run_cache.py`, `run_layer_sweep.py`, `run_expert.py`,
  `run_appendix.py`, `make_report.py`. All row-level outputs saved as Parquet under `results/<model>/`.

## 7. Verification before trusting any number

1. OLMoE-1B-7B-0125 (14 GB bf16, fits the GPU fully) is the pilot model: HF `from_pretrained` and our engine must
   produce identical logits (bf16 tolerance) on 50 prompts, and the pipeline runs end to end in minutes.
2. Qwen3 and Mixtral: HF reference forward with CPU/disk offload on 5 prompts each (slow, about a minute per forward)
   compared with our engine's logits.
3. Invariances: patching a layer's own clean output into the clean run changes nothing; sum of expert contributions
   equals the MoE-block output; a single-token wavefront row with no intervention reproduces the noised run's logits;
   suppressing a non-routed expert has zero effect.
4. Funnel agreement: fraction of the paper's Table 8 case IDs that pass our filters, and our Delta_clean on those cases.

## 8. Run schedule

| Phase | Work | Passes per model | Estimated time |
|---|---|---|---|
| 0 | venv with torch 2.14 / transformers 5.16; start both downloads to `/opt/dlami/nvme/hf` in the background | - | 10 min setup, 75 min download in background |
| 1 | Engine + data + verification on OLMoE | - | 0.5 to 1 day |
| 2 | Qwen3: filter scan in chunks of 1,024 records (2,048 sequences per pass) until 256 strict and 512 relaxed cases | 2 to 4 | 15 min |
| 3 | Qwen3: caching pass over the paper's 256 IDs plus our 256 plus relaxed 512 (K/V, MoE in/out, routing, per-expert contributions at all layers) | 1 | 5 min |
| 4 | Qwen3: layer sweep for all cases and sets | 1 | 5 min |
| 5 | Qwen3: expert patches for every clean-active expert, controls, equal-norm, gate-matched, all-active ranks | partial pass from L44 | 5 min |
| 6 | Qwen3: noise-scale sweep sigma in {1,2,4} | 1 | 5 min |
| 7 | Mixtral: phases 2 to 5 plus coalition patches and active-pair equal-norm | 5 to 7 | 30 min |
| 8 | HF reference checks on the big models | - | 30 min |
| 9 | Post-processing: stability grid, relation folds, all tables, Figure 1, comparison report vs paper | - | 0.5 day |

## 9. Under-specified details and the assumptions I will make

- Tokenizer defaults for special tokens: Mixtral prepends BOS, Qwen3 does not.
- Single-token check uses the object string with a leading space, tokenized standalone.
- Embedding std is the standard deviation over the whole embedding matrix in float32; noise is drawn in float32
  with a torch generator seeded 0 + case_id, added to the bf16 embeddings of every subject token.
- Layer rescue patches the MoE-block output only (the MoE sublayer's residual update), not the whole layer output.
- Expert selection uses the all-case mean over discovery cases; the active-only alternative is reported alongside.
- Active-random controls are drawn with a per-case seeded RNG; the paper's exact draws cannot be recovered.
- Delta values are computed from float32 copies of the bf16 logits.
- Appendix F reruns the full layer sweep on the relaxed set rather than fixing L44/L19, and reports both.
- Any of these can be changed to match the artifact if the authors share it.

## 10. Decisions needed

1. Approve the full-scale path (recommended) rather than a scaled-down model swap. OLMoE is used only as a pilot.
2. Approve using `/opt/dlami/nvme` for the 155 GB of checkpoints. It is instance-store scratch and is wiped if the
   instance stops; re-downloading takes about 75 min.
3. Approve creating a Python venv with torch (about 4 GB on the root disk).
4. Whether to contact the authors for the supplementary artifact in parallel.
