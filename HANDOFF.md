# HANDOFF: autonomous end-to-end reproduction of arXiv 2606.03780

You are a headless Claude Code agent running unattended overnight on the user's AWS g5.4xlarge (1x A10G 24 GB,
16 vCPU, 62 GB RAM). The user (Ken) is asleep and cannot answer questions. Your job: implement the full plan in
`PLAN.md` and run every experiment of the paper end to end, producing `results/REPORT.md`, all tables, Figure 1 and
row-level parquet files. Work only inside `/home/ubuntu/MOE` and `/opt/dlami/nvme`. Never ask for input; make the
documented default choice and record it in `logs/PROGRESS.md`. Never stop early: if something fails, diagnose, fix,
and continue. When everything is done, write `results/DONE` (a marker file) as the very last step.

Read `PLAN.md` first (sections 4 to 9 are the specification), then this file (operational detail and decisions already
made), then `paper.txt` / `paper_src/acl_latex.tex` whenever a protocol detail matters.

## 0. State at handoff (2026-09-03 ~02:20 UTC)

Done:
- Paper, TeX source, `paper.txt`, `data/counterfact.json` (21,919 records), `data/paper_case_ids.json` (Table 8 exact
  case IDs: `{"Qwen3": {"discovery": [...128], "validation": [...128]}, "Mixtral": {...}}`).
- Python venv at `/home/ubuntu/MOE/.venv` (Python 3.14): torch 2.14 CUDA (verified bf16 matmul on GPU), transformers
  5.16.1, safetensors 0.8, huggingface_hub, accelerate, numpy, scipy, pandas, pyarrow, matplotlib, tqdm.
  Activate with `. /home/ubuntu/MOE/.venv/bin/activate`. Do not install quantization libraries.
- All three checkpoints fully downloaded to `HF_HOME=/opt/dlami/nvme/hf` (always `export HF_HOME=/opt/dlami/nvme/hf`):
  - `Qwen/Qwen3-30B-A3B-Base` (61 GB, 16 shards), `mistralai/Mixtral-8x7B-v0.1` (93 GB, 19 shards),
    `allenai/OLMoE-1B-7B-0125` (fp32, 6 shards; use as bf16 pilot).
  - Snapshot dirs: `/opt/dlami/nvme/hf/hub/models--<org>--<name>/snapshots/<hash>/` (one hash each).
- Not started: any code under `moetrace/`.

Machine constraints: another user process (unrelated project) holds ~9 GB RAM; about 42 GB RAM is free. GPU is idle.
Root disk has ~150 GB free; write results there. NVMe reads at 1.6 GB/s. Do not kill processes you did not start.

## 1. Model facts (from the downloaded configs and safetensors indices)

| | OLMoE-1B-7B-0125 | Qwen3-30B-A3B-Base | Mixtral-8x7B-v0.1 |
|---|---|---|---|
| layers | 16 | 48 | 32 |
| hidden | 2048 | 2048 | 4096 |
| heads / kv heads / head_dim | 16 / 16 / 128 | 32 / 4 / 128 | 32 / 8 / 128 |
| experts / top-k | 64 / 8 | 128 / 8 | 8 / 2 |
| expert intermediate | 1024 | 768 (`moe_intermediate_size`) | 14336 |
| router norm | softmax, top-k, `norm_topk_prob=False` | softmax, top-k, renormalize (`norm_topk_prob=True`) | softmax, top-2, always renormalize |
| q/k norm | RMSNorm over full q (2048) and k (2048) BEFORE head reshape | RMSNorm per head (128) after reshape, before RoPE | none |
| rms eps / rope theta | 1e-5 / 1e4 | 1e-6 / 1e6 | 1e-5 / 1e6 |
| vocab / BOS added by tokenizer | 50304 / no | 151936 / no | 32000 / yes (`<s>`) |
| checkpoint dtype | float32 (cast to bf16 on load) | bfloat16 | bfloat16 |

Safetensors keys (per layer `model.layers.{l}.`):
- OLMoE and Qwen3: `input_layernorm.weight`, `post_attention_layernorm.weight`, `self_attn.{q,k,v,o}_proj.weight`,
  `self_attn.{q,k}_norm.weight`, `mlp.gate.weight` (router, shape [E, hidden]),
  `mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`.
- Mixtral: same attention/norm names but no q/k norm; router `block_sparse_moe.gate.weight`;
  experts `block_sparse_moe.experts.{e}.{w1,w3,w2}.weight` where w1 = gate_proj, w3 = up_proj, w2 = down_proj.
- Global: `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight` (untied for all three).
- All linear layers have no bias. Activation is SiLU: `down(silu(gate(x)) * up(x))`.

Reference implementations (transformers main, pinned copies you can re-fetch):
`src/transformers/models/{qwen3_moe,mixtral,olmoe}/modeling_*.py`. Match their numerics: RMSNorm computes in fp32 and
casts back before multiplying by weight; RoPE cos/sin computed in fp32 then cast to bf16, `rotate_half` convention,
positions 0..T-1; router logits from a bf16 matmul, softmax in fp32, top-k, optional renormalize, weights cast back to
bf16; expert outputs multiplied by routing weight then summed.

## 2. Engine design (already decided; see PLAN.md section 5)

Implement a layer-streaming executor in plain PyTorch (`moetrace/engine.py`), not HF `from_pretrained`:
- Resident on GPU: embeddings, final norm, lm_head, and ONE decoder layer's weights at a time. Load layer l from the
  safetensors shards (`safetensors.safe_open`, mmap) and stack per-expert weights into fused tensors
  `gate_up[E, 2*I, H]`, `down[E, H, I]`. Prefetch layer l+1 on a background thread while computing layer l.
- Two row types advanced together, layer by layer:
  1. Prefill rows: full short sequences (clean prompt, and noised prompts for each requested sigma). Right-pad within
     the batch; RoPE positions 0..len-1; causal attention (right padding needs no extra key mask for real tokens).
     At each layer record, for the final position (index len-1) of every prefill row: MoE-block input, routing
     (top-k indices and weights), MoE output, and each routed expert's contribution c_e = w_e * E_e(x) (accumulate the
     block output as the fp32 sum of the c_e so that sum(c_e) == MoEOut exactly). Keep this layer's K and V for all
     positions so wavefront rows can attend to them (they are only needed for the current layer).
  2. Wavefront (single-token) rows: a patched run equals its parent noised run except at the final position from the
     patch layer upward. Under causal attention the other positions are untouched, so a wavefront row spawned at
     layer l starts as `h_noised_after_layer_l[final] + v` (v = intervention vector, below) and then runs layers
     l+1..L-1 as a single token attending to the parent noised prefill row's K/V of positions 0..len-2 plus its own
     K/V at position len-1. No recomputation at the patch layer itself.
- Intervention vectors, all built in-pass at layer l from the clean and noised prefill rows of the same case:
  - layer patch: v = MoEOut_clean - MoEOut_noised (this equals replacing the noised block output with the clean one).
  - expert patch e: v = delta_e = c_e^clean - c_e^noised, where c_e = 0 in a run where e is not routed.
  - scaled expert patch: v = alpha * delta_e (equal-norm control: alpha = min(||delta_a||, ||delta_b||) / ||delta_a||).
  - coalition: v = sum of delta_e over a set (clean top-k set, or union of clean- and noised-routed sets).
    Note: the union coalition equals the layer patch exactly in exact arithmetic; report both anyway.
- After the last layer: final RMSNorm, then logits. For prefill rows compute full-vocab logits at the final position
  (bf16 matmul) to record Delta and the top-1 token; for wavefront rows compute only the two needed logits
  (true and foil rows of lm_head). Delta = logit(true) - logit(foil) computed from float32 copies of bf16 logits.
- Batch sizes: filter passes ~1,024 records = 2,048 prefill rows; sweep passes up to 768 cases x (2 prefill + 48
  wavefront) rows. Wavefront rows at the top layer for the Qwen3 sweep: ~37k rows x 2048 bf16 = 150 MB. Fine.
- A pass over Qwen3 reads 61 GB (about 40 s from NVMe cold); Mixtral 93 GB (about 60 s). Aim for 5 to 8 passes per
  model total. Log per-layer timing on the first pass.

Passes per model (each pass = one sweep over all layers with a job list):
1. Filter scan: chunks of 1,024 records in the seed-0 shuffled order; rows = clean + noised (sigma 3.0) per record;
   outputs Delta_clean, Delta_noised; keep strict (margin >= 1.0, drop >= 0.5) and relaxed (>= 0.5, >= 0.25) sets;
   continue chunks until 256 strict AND 512 relaxed cases exist. Also verify which of the paper's Table 8 IDs pass.
2. Sweep pass: for the case set(s) (paper IDs 256; our strict 256; relaxed 512; run them in one pass if memory
   allows, otherwise per set): clean, noised, and a layer patch at every layer. Save per-case Rescue_l for all l,
   and per-layer final-position routing (clean and noised top-k indices and weights) and ||c_e||.
3. Expert pass at the selected layer(s): clean, noised, plus expert patches for EVERY clean-active expert of every
   case (this covers selection, active-random controls, gate-matched control, all-active ranks, stability and
   relation folds in post-processing), equal-norm rows for every ordered pair (a, b) of clean-active experts
   (56 per case for Qwen3, 2 for Mixtral), and coalition rows (clean top-k, union). Run at the discovery-selected
   layer; also run at the paper's layer (44 for Qwen3, 19 for Mixtral) if selection differs.
4. Noise-scale pass (Qwen3): sigma in {1, 2, 4} (3.0 already done): clean, noised_sigma, layer patch at L*, expert
   patch for the selected expert at L*. Report Table 13 rows.
5. HF reference check (verification, see section 4).

## 3. Protocol details and assumptions (defaults; record any deviation in PROGRESS.md)

- Records: `data/counterfact.json`. Prompt = `requested_rewrite.prompt.format(subject)`. True object =
  `target_true.str`, foil = `target_new.str`. Relation = `relation_id`. Case id = `case_id`.
- Shuffle: `random.Random(0).shuffle(indices)` over all 21,919 records; scan in that order.
- Object tokens: continuation tokenization. `ids(prompt + " " + obj)` must equal `ids(prompt) + [one token]`; that
  one token is the target id. Reject otherwise. Tokenize with the model tokenizer's defaults for special tokens
  (Mixtral adds BOS; Qwen3 and OLMoE add nothing).
- Subject span: character offsets of the subject inside the formatted prompt (from the template's `{}` position);
  subject token positions = tokens whose offset span overlaps it (`return_offsets_mapping=True`; skip special tokens).
- Noise: sigma = 3.0 * embed_tokens.weight.float().std() (over the whole matrix). Per case:
  `g = torch.Generator().manual_seed(0 + case_id)`; `eps = torch.randn(n_subject_tokens, hidden, generator=g) * sigma`
  (fp32), added to the fp32 copy of the subject-token embeddings, then cast to bf16. Same draw reused for every
  intervention on that case. For sigma variants use `sigma_mult * embed_std` with the same generator.
- Filters: strict margin >= 1.0 and drop >= 0.5; relaxed 0.5 / 0.25. Split: `random.Random(split_seed)` shuffle of the
  collected cases, first 128 discovery (first 256 of 512 for relaxed), split_seed = 0 for the main run. The paper's
  Table 8 IDs are the PRIMARY case set: use their given discovery/validation assignment.
- Layer selection: argmax over layers of the mean Rescue_l on discovery cases; validate on validation cases; report
  the full per-layer validation curve (Figure 1a), Table 6 sharpness (top vs next-best layer on validation).
- Expert selection (recurrence-first): at L*, candidates = experts clean-active (routed in the clean run at the final
  position) in >= 64 of 128 discovery cases (threshold = half the discovery split); pick the highest mean expert
  rescue over ALL discovery cases (rescue = 0 where not clean-active). Also report the active-only mean.
- Specificity: Spec = Rescue(e*) - mean over controls; controls = other clean-active experts of the same case at L*:
  Qwen3 3 controls sampled with `random.Random(1000 + case_id)`; Mixtral the single other expert. Cases where e* is
  not clean-active contribute Rescue(e*) = 0 and, for Spec, use controls drawn from the clean-active set anyway
  (record both "all-case" and "anchor-active" summaries; Table 3 reports zero-row counts).
- Gate-weight-matched control (Qwen3, Table 9): among other clean-active experts pick the one whose clean router
  weight is closest to e*'s. Equal-norm: scale both patch vectors to the smaller of the two norms.
- All-active rank (Table 10): rank of Rescue(e*) among all clean-active experts of the case; mean rank, percentile,
  top-1 and top-2 counts, and Rescue(e*) minus mean of all other active experts.
- Mixtral active-pair equal-norm (Table 11) on validation cases where e* is clean-active.
- Appendix D: re-run selection for split seeds {0..4} x thresholds {32,48,64,80,96} using the saved per-case
  per-expert rescues over the 256 cases; report how often L44E069 is selected, mean validation rescue and Spec.
  Relation-held-out (Table 12): 5 folds over relation ids; select on non-held-out relations, evaluate on held-out.
- Appendix F: relaxed 512 cases; same protocol; Tables 14 and 15 (overlap of relaxed validation with strict set).
- Appendix G: Mixtral coalition rows (Table 16). Also compute them for Qwen3 as an extra.
- Statistics: mean, positive fraction, 5,000-resample percentile bootstrap CI of the mean, two-sided sign-flip test
  with 10,000 resamples (p = fraction of |flipped mean| >= |observed mean|). Seed 0 for both.

## 4. Verification you must do before trusting numbers

1. OLMoE pilot: load `allenai/OLMoE-1B-7B-0125` with transformers (`dtype=torch.bfloat16`, on GPU) and compare final-
   position logits with the engine on 50 prompts. Expect max |diff| of order 0.1 and near-perfect agreement of Delta
   (|diff| < 0.1 on almost all prompts). Also verify: wavefront row with v = 0 spawned at layer l reproduces the noised
   prefill logits; sum of c_e equals MoEOut; layer patch spawned on the clean run as parent with v = 0 is identity.
2. Big models: after the main results, load Qwen3 and Mixtral with transformers using
   `device_map="auto"`, `max_memory={0: "16GiB", "cpu": "28GiB"}`, `offload_folder="/opt/dlami/nvme/offload"`,
   `dtype=torch.bfloat16`, and compare logits on 5 prompts each (slow: minutes per model). If transformers 5.16 fails
   to load with offload, try `accelerate`-free CPU-only loading of a truncated model (first 4 layers) built from the
   same safetensors and compare the residual stream after 4 layers instead. Record the outcome; do not block on it.
3. Funnel agreement: fraction of the paper's Table 8 IDs that pass our strict filter per model, with our Delta_clean
   and drop for each.

## 5. Deliverables and layout

- Code: `moetrace/{weights,engine,data,noise,protocol,stats,report,verify}.py`, `scripts/*.py` CLIs.
- Row-level: `results/<model>/*.parquet` (case_id, split, set, sigma, kind, layer, expert, partner, alpha,
  logit_true, logit_foil, delta, plus routing tables).
- Tables: `results/tables/table_01.md` ... `table_16.md` (and CSV), matching the paper's table numbering; add
  paper-vs-ours columns where a paper number exists (all paper numbers are in `paper.txt` / the TeX source).
- Figure 1: `results/figures/fig1.pdf` and `.png` (three panels like the paper: layer sweep validation curves for both
  models, single-expert specificity with CIs, Mixtral coalition rescue vs layer rescue).
- `results/REPORT.md`: what was run, verification outcomes, paper-vs-ours comparison for every table, deviations and
  assumptions, timing per pass, and an honest statement of what did or did not reproduce.
- `logs/PROGRESS.md`: append a timestamped entry after every milestone (what finished, key numbers, next step).
  This is how the user will follow along in the morning. Also `logs/*.log` for long-running commands.
- Final step: `touch results/DONE`.

## 6. Working rules

- Long runs: run via `nohup ... > logs/x.log 2>&1 &` or with generous timeouts; poll logs; never leave the GPU idle for
  long while waiting on nothing.
- Save intermediate row-level results after every pass so a crash never loses more than one pass.
- Prefer correctness over speed, but do not do optimization work beyond the design above unless a pass takes over
  10 minutes.
- If a result disagrees with the paper, first suspect the implementation (verification suite), then the protocol
  assumptions in section 3; try the alternative reading if cheap, and report both.
- Do not email anyone, do not modify files outside `/home/ubuntu/MOE` and `/opt/dlami/nvme`.
- Internet access IS allowed (user permission, 2026-09-03 02:32 UTC): fetch documentation, transformers source, the
  ROME/CounterFact reference code, arXiv pages, PyPI, Hugging Face, and web search whenever it helps correctness.
- Suggested order: weights + engine + verify on OLMoE (HF comparison) -> data/noise/filter on OLMoE (quick end-to-end
  dry run of the whole protocol on OLMoE, ~minutes) -> Qwen3 filter, sweep, expert, noise passes -> Mixtral filter,
  sweep, expert passes -> HF reference checks on the big models -> post-processing, tables, figure, REPORT -> DONE.
