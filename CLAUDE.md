# CLAUDE.md — project memory for agents working in /home/ubuntu/MOE

Read this first. It is the entry point for any new agent session in this repository: what the project is, what has
already been done and with what result, how the codebase is used, and the rules that keep work here consistent.
Details live in the documents listed in section 8; this file is the map. Last full refresh: 2026-09-20; Phase 2 wave 1 added 2026-09-21.

## 1. What this project is

A full-scale, independent reproduction of **Lu, Modarressi, Liu, Schütze (2026), "Expert-Aware Causal Tracing of
Factual Recall in Sparse MoE Language Models"** (arXiv:2606.03780), followed by four extension studies. The paper
released no code. We re-implemented the protocol from the text and re-ran every table and Figure 1 on the paper's exact
models, Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1, in bf16, on this machine's single 24 GB A10G.

**Status.** Base reproduction COMPLETE (2026-09-03, marker `results/DONE`, deliverable `results/REPORT.md`).
Extensions COMPLETE (2026-09-14, deliverable `results/EXTENSIONS_REPORT.md`, plan `RESEARCH_PLAN.md`).
**Phase 2 wave 1 COMPLETE (2026-09-21: F4 subject-token patching, F1 rankings/minimal sets, F2 attention heads,
F5 probability metrics; sections `results/sections/ext5_*.md`, assembled into `EXTENSIONS_REPORT.md` as Directions
5-F4/5-F1/5-F2/5-F5).** Wave 2 (F3 gradient attribution, F1.4 multi-layer sets) NOT started. Everything is committed
and pushed to github.com/KenSu223/MOE (branch `main`). Nothing is running. Open items are in section 3.

## 2. Results in one screen (base reproduction)

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

Vocabulary used everywhere: Δ = logit(true) − logit(foil) at the final position; drop = Δ_clean − Δ_noised;
rescue = Δ_patched − Δ_noised (all in logits, validation-set means of 128 cases unless stated); Spec = rescue of the
selected expert minus the mean rescue of other clean-active experts of the same prompt and layer; "clean-active" =
routed to in the clean run at the final position; recurrence gate = clean-active in ≥ 64 of 128 discovery cases.

## 2b. Extensions (RESEARCH_PLAN.md; results in results/EXTENSIONS_REPORT.md, sections in results/sections/)

Both waves done 2026-09-14 (wave 1: Directions 1 and 3; wave 2: Directions 2, 2b and 4). GPU total for the extensions ≈ 4 h.
- **Direction 1 (joint layer×expert search, all-layer expert passes in `results/*_alllayers`)**: the two-stage
  procedure does not miss a *better* expert in Qwen3 (L44E069 is the joint argmax) but misses a *second locus*:
  L42E115 (126/128 recurrent, val +0.447 [0.363, 0.537], Spec +0.423) wins the discovery argmax on the strict and
  relaxed sets and in 30/75 grid cells. Mixtral under the paper protocol (no BOS): the joint winner is **L18E001**
  (Spec +0.098 [0.040, 0.162]) while the paper's L19E006 has Spec −0.159; E001 is the strongest expert of L17/18/21/22
  under both protocols; L19E002 is the strongest L19 expert without BOS too but fails the 64/128 recurrence gate.
- **Direction 3 (BOS mechanism)**: `<s>` is a prompt-independent attention sink whose per-layer K/V alone reproduce
  the BOS run when transplanted into no-BOS prompts (routing agreement 0.99); a key-only sink breaks the model.
  Without BOS, Mixtral forms no position-0 sink; in 63/256 prompts the final token itself becomes the sink state and
  is routed to E006 (63/63), which is the L19 "sink expert" (P(E006|`<s>`)=1.00) but an ordinary content expert on
  corpus text. Any position-0 token that forms a sink (`\n`, `,`, `:`, attached `.`, `▁`) restores the BOS-run routing;
  `the`, `▁.`, rare words do not. RoPE shift is a no-op (relative positions). Qwen3's L44E069 survives a prepended
  `<|endoftext|>`. Consequence: the paper's negative Mixtral specificity is expected once a quarter of its prompts
  route the final token like a sink. Engine gained optional diagnostics (`DiagSpec`), `PrefillSpec.pos_offset`,
  sink transplant (`sink_donor`), `prefix_ids` in `prepare_case`; verify_olmoe unchanged.
- Literature: docs/ext3_literature_review.md (Mistral is the documented exception: BOS is its sole sink driver).
- **Direction 2b (attention vs MoE patching; engine kinds `attn_layer`, `block`, `resid`, `block_diff`, runs
  `results/*_attnsweep`)**: the paper's MoE-only patch misses the largest single-sublayer locus. Qwen3: attention
  output at **L40 +1.59 [1.41, 1.79]** (block L40 +1.95) vs MoE L44 +0.93; L44 is a pure-MoE layer (attention share
  2%) but overall attention carries 51% of the positive rescue. Mixtral: attention L18 +0.99, MoE L19 +0.56, block L19
  +1.37; at L19 attention carries 62% [58, 66]. block = attn + MoE to bf16 noise (r 0.96-0.99), slight sub-additivity
  only at shared peaks. Reading: attention moves the information in a few discrete steps (Qwen3 L40/L43, Mixtral
  L15/L18/L19), MoE of the same and following layers transforms it. Verified vs transformers hooks on OLMoE.
- **Direction 2 (model zoo; runs `results/{qwen3_instruct,qwen3_coder,mixtral_instruct,olmoe,olmoe_instruct}_
  {default,nobos,chat}`, usage specs in `data/model_usage/`)**: all five new checkpoints show **pattern A** (one
  layer, one positive specific expert) under their intended protocol: Qwen3-Instruct-2507 and Qwen3-Coder keep
  **L44E069** (and L42E115 as the second locus) under raw and chat protocols with rescue/Spec at or above the base;
  Mixtral-Instruct with BOS = base (L19E002, 86% identical L19 routing); OLMoE base L13E056, OLMoE-Instruct
  L12E040/L13E056. **Pattern B** (selected expert not specific) occurs only for Mixtral base and Instruct under the
  paper's no-BOS protocol (identical 91/83 E006 activity in both; joint winner L18E001); pattern C never. Post-training
  moves neither layer nor expert. Chat wrapping raises margins and absolute rescues but not the drop-normalised layer
  share; one exception: Mixtral-Instruct chat's argmax jumps to the last layer L31 (+1.79) while the L19-L21 band
  stays. Sink-carrying final tokens: 24% of prompts in both Mixtrals without BOS, 0% in every other run.
- **Direction 4 (CodeFact; `data/codefact/items.jsonl`, 6,795 Python next-token counterfactuals in six categories from
  HumanEval, MBPP and CodeSearchNet; runs `results/codefact_{qwen3_raw,mixtral_nobos,qwen3_coder_raw,qwen3_coder_chat}`)**:
  pass rates under the paper's thresholds separate categories by how the answer is determined: S1 closing bracket 90%
  (read from the opener like a fact, drop +4.4), R1 variable recall 79%, R2/R3 57-59%, S2/S3 keywords 33-36% (redundantly
  determined, median drop +0.25/+0.12). On code the last MoE block acts as a read-out (L47/L31 selected almost
  everywhere), so an interior-layer variant (≤ L−5) is reported alongside. Localised single experts: S1 (Qwen3
  L47E025 Spec +0.97, interior L42E048 +0.87; Mixtral L31E000 +1.44), R1 in Qwen3 (L43E126 +0.32, CounterFact-like),
  S2 (Qwen3 L41E041 +0.55, Mixtral L17E003 +0.32); S3/R2/R3 weak. Categories are near-disjoint in their top experts
  (mean Jaccard 0.05); the factual experts L44E069/L42E115 rescue nothing on code; Mixtral E006 is negatively specific
  again (R1). Qwen3-Coder keeps the same code experts as the base (L47E025, L43E126, L41E041) and adds an S3 expert
  L47E014. Code prompts are 36-84 tokens median (CounterFact: 8) with a single-token "subject"; the axis that matters
  is single-token vs distributed determination, not syntax vs recall.
- **Phase 2 wave 1 (2026-09-21, ≈ 45 GPU min; plan in RESEARCH_PLAN.md "Phase 2 execution plan")**:
  - *F4 subject-token patching* (`moetrace/ext5_subject.py`, suffix wavefront rows from the last subject token p;
    runs `results/{qwen3_bos,mixtral_nobos,mixtral_bos}_subject`; verified vs transformers hooks on OLMoE, bit-identical
    to `Engine.run` at p = T−1): the MoE-output patch at p peaks in the **first ten layers** — Qwen3 **L4 +0.93**
    [0.65, 1.25] (= the final-token L44 peak +0.93), Mixtral no-BOS L6 +1.20, Mixtral BOS **L4 +2.31** (4× the L19
    peak); ≈ 0 from L9 on and at L42/L44/L18/L19. **No shared early-site expert**: recurrent candidates rescue
    ≤ +0.07 with Spec ≈ 0, the clean top-k coalition recovers the layer effect, and the per-case best expert (88–99% of
    the coalition) differs across prompts (36–38 distinct winners/128 in Qwen3). L44E069/L42E115/L19E002 patched at p
    rescue nothing. Noise on the last subject token alone = 0.47–0.52 of the whole-span drop. Reading: expert-level
    localisation is a property of the shared late read-out, not of the early subject-enrichment site.
  - *F1 rankings / minimal sets* (`moetrace/ext5_rank.py`, existing `*_alllayers` passes + exhaustive subset passes
    `results/{qwen3,mixtral_nobos}_subsets`, engine kind `coalition_set`): rankings by all-case rescue, active-only
    rescue, Spec and the discovery statistic agree wherever the effect is clear (τ 0.94–0.97 rescue vs active-only;
    L44E069 rank 1/1/1, L42E115 2/2/2, L19E002 1/1/1); disagreements are systematic: "junior partners" (rescue > 0,
    Spec < 0: Mixtral L19E006, Qwen3 L43E046) and Spec-without-rescue in layers whose block hurts. Minimal sets for
    50/80/90 % of the block rescue: Qwen3 L44 1/6/9 experts (E069 alone 53 %), L42 1/2/6 (E115 72 %); Mixtral BOS
    L19 1/3/4 (E002 63 %), L18 1/2/2 (E001 77 %). Exact subsets: one expert reaches 80 % of a case's own block in
    59–60 % of Qwen3 cases (≤ 2 in 87–88 %); pairwise interactions ≈ 0 (bf16 ulp), redundancy only between strong
    pairs; the additive approximation predicts the exact minimal size in 91–97 % of cases. Cross-layer sums over-count
    (F1.4 open).
  - *F2 attention heads* (engine kind `attn_head`, runs `results/{qwen3,mixtral_nobos,mixtral_bos}_heads`): the
    attention rescue is carried by a few **mover heads reading the last subject token**: Qwen3 L40 **head 13 +0.95**
    [0.81, 1.09] (60 % of +1.58, Spec +0.93; two heads for 80 %), L43 distributed (4 heads), L44 null; Mixtral L18
    **head 4** +0.61/+0.79 (no BOS/BOS, 76–80 %), L24 head 22 (82–86 %), L19 heads 29–31 (one GQA group), L15 heads
    1/3; identical heads under both protocols. Mover heads put 0.4–0.5 of clean attention on the last subject token,
    noise halves it (→ relation tokens without BOS, → the position-0 sink with BOS). Σ heads = attention rescue on the
    mean; per-case r 0.3–0.7, so minimal head sets are additive estimates.
  - *F5 probability metrics* (engine `--metrics`: `logp_true, logp_foil, p_true, p_foil, rank_true, kl_to_clean`;
    runs `results/{qwen3,mixtral_nobos,mixtral_bos}_metrics`): Δ = log p(true) − log p(foil) exactly (log-odds; checked
    to one bf16 ulp). Qwen3 and Mixtral-BOS selections are metric-independent (L44E069, L19E002 under all six).
    Mixtral no-BOS: Δp and rank select **L18E001** (Spec > 0) instead of L19E006 (Spec < 0 under every metric); Δlog p
    and KL pick L0, a heavy-tail artefact of the 26 most-disrupted (sink-state) cases. Dataset descriptor: the true
    object is the clean top-1 in only 29–32 % of paper cases (median clean p(true) 0.02–0.04). Recommendation adopted
    in the section: keep Δ primary, report normalised rescue, rank recovery and clean top-1 rate alongside; Δp
    descriptive only; KL/Δlog p as protocol diagnostics.
  - Incident: `run_sweep.py` crashed after a complete pass when writing `run_meta.json` (variable `meta` reused for a
    DataFrame); fixed with JSON sanitising in `run_sweep.py`/`run_expert.py`; no GPU work lost.
- Open method questions raised by the agents, **not yet decided by the user**: (a) how strongly to state that the
  paper's Mixtral expert claim is a search-scope artefact (L18E001); (b) last-layer read-out vs localisation
  (interior-layer rule for code / chat?); (c) select layers by block (attention + MoE) rescue rather than MoE rescue?;
  (d) flag sink-carrying prompts and the object-token rule as protocol checks; (e) recurrence threshold relative to top-k.

## 3. What is NOT done / known gaps

Base reproduction:
- Mixtral **relaxed-filter set (Tables 14, 15)** was run only under the BOS default. A no-BOS filter scan → sweep →
  expert pass (about 4 passes, 7 min GPU) would complete it: `run_filter.py mixtral --out mixtral_nobos_relaxed
  --no-special-tokens`, then `run_sweep.py` / `run_expert.py` with the same flags.
- The paper's relaxed 512-case set used a different record order (only 15 overlaps with its strict set); its IDs are
  unpublished, so Table 15 subset sizes cannot match.
- Zero-row definition (Table 3), fold assignment (Table 12), split function (Appendix D), active-random draws and the
  exact noise samples are unrecoverable; only selections and CI-level agreement are comparable.

Extensions:
- **Phase 2 wave 2 NOT started**: F3 gradient / attribution patching (reverse layer streaming), F1.4 multi-layer
  minimal sets (engine kind `multi` exists and is verified), optional F2 on Qwen3-Instruct, F4 full position × layer
  grid (the executor supports any p per row). Wave-1 loose ends: exact `attn_head_set` kind (minimal head sets are
  additive estimates); MoE-side cross-check of mover heads (patch L40 heads, read E069 routing at L44) needs wavefront
  routing; consolidate `ext5_subject._build_v_at` into `engine.py`; join ext3 per-case sink flags to the F5 L0 KL tail;
  early-site expert statistic stratified by subject token / relation instead of recurrence.
- The five open method questions in 2b (user decisions pending); the final wording of EXTENSIONS_REPORT.md follows them.
- CodeFact: Mixtral S3 is partial (241 passing items → 120/121 split); The Stack was not used (gated; CodeSearchNet
  instead); R3 has a single-digit sub-category that may deserve exclusion; no HF-hook verification of the code runs
  beyond the shared engine.
- Model zoo: Qwen3-Coder has no public Base checkpoint (Instruct only); OLMoE-Instruct's strict/relaxed sets pick
  L13E056 vs L12E040 (no shared case set); Mixtral-Instruct chat's last-layer argmax is unresolved (question b).
- Direction 2b: no per-head decomposition of the attention patches (natural follow-up: mover heads at Qwen3 L40/L43,
  Mixtral L15/L18/L19/L24).
- Raw diagnostics (`/opt/dlami/nvme/moe_ext2`, `moe_ext3`, 3 GB) and all checkpoints sit on the ephemeral NVMe.

## 4. Machine and environment (verify before relying on it)

- AWS g5.4xlarge: 1× A10G 24 GB, 16 vCPU, 62 GB RAM. An unrelated user process (`repo-world-model`) holds ~9 GB RAM;
  do not kill processes you did not start.
- Python venv: `. /home/ubuntu/MOE/.venv/bin/activate` (Python 3.14, torch 2.14+cu130, transformers 5.16.1,
  safetensors, pandas, pyarrow, scipy, matplotlib, datasets). `~/.local/bin/claude` and `~/.local/bin/gh` exist.
- **Always `export HF_HOME=/opt/dlami/nvme/hf`** before any script (`scripts/gpu_queue.sh` does it for you).
  Seven checkpoints live there (≈ 407 GiB): Qwen3-30B-A3B-Base 61 G, Qwen3-30B-A3B-Instruct-2507 57 G,
  Qwen3-Coder-30B-A3B-Instruct 57 G, Mixtral-8x7B-v0.1 93 G, Mixtral-8x7B-Instruct-v0.1 87 G, OLMoE-1B-7B-0125 39 G
  (fp32 shards), OLMoE-1B-7B-0125-Instruct 13 G. NVMe free ≈ 49 GB; `/opt/dlami/nvme/offload` (65 GB, transformers
  offload scratch from the reference checks) is reclaimable. The NVMe is **ephemeral**: an instance stop wipes it;
  re-download with `scripts/download_models.sh` (base three) and `scripts/ext2_zoo_download.sh` (zoo four).
- Root disk holds code and results: `results/` is 380 MB on disk, ~110 MB tracked (gitignored: `codefact_*/
  scan_routing.parquet`, `codefact_*/expert_parts/`, `codefact_smoke/`); `.git` is 240 MB. Never write weights or
  offload folders to the root disk.
- Before launching GPU work: `pgrep -af "run_sweep|run_expert|run_filter|hf_reference|chain|ext[1-4]_|gpu_queue"`
  and `nvidia-smi`. Submit every GPU job through `bash scripts/gpu_queue.sh <job-name> -- python ...` (flock lock,
  `logs/gpu_queue.log`), even when you believe you are alone.

## 5. Codebase map and how to use it

Engine idea: one decoder layer resident on the GPU at a time (streamed from safetensors with prefetch); all runs of a
pass advance layer by layer. *Prefill rows* = full clean/noised prompts (record final-position MoE in/out, routing,
per-expert contributions c_e = w_e·E_e(x), optionally attention outputs and diagnostics). *Wavefront rows* = single
tokens implementing interventions from layer l upward, attending to the parent noised run's K/V. One pass = one read
of the checkpoint (Qwen3 ~60 s, Mixtral ~95 s, OLMoE ~4 s) regardless of how many interventions are batched; the only
limit is wavefront rows × prompt length (about 45k rows at T ≈ 15, far fewer at code lengths T ≈ 40-140: chunk by
layers and cases, see `run_expert.py --layer-chunks` and `scripts/ext4_scan.py`).

`moetrace/` (base)
- `arch.py` ArchSpec from config.json (Qwen3-MoE, Mixtral, OLMoE key templates, q/k-norm variant, top-k renorm).
- `weights.py` CheckpointStore + LayerStreamer (mmap safetensors, per-layer expert stacking, pinned double buffer).
- `engine.py` `Engine(repo)`; `PrefillSpec(ids, true_id, foil_id, noise_pos=None, noise_eps=None, pos_offset=0,
  sink_donor=-1, sink_vscale=1.0)`; `SpawnSpec(layer, parent, clean, kind, expert=-1, partner=-1)`; `DiagSpec(...)`
  (per-layer attention mass, residual norms, router logits, residuals, token log-probs, all-token routing, attention
  outputs); kinds = `zero`, `layer` (MoE output), `expert`, `expert_scaled`, `coalition_clean`, `coalition_union`,
  `attn_layer`, `block` (attention + MoE), `resid` (whole residual after the layer), `block_diff` (numerics check).
  `eng.run(prefill, spawns, record_routing=True, diag=None)` → `PassResult` (delta, logits, sp_delta, sp_vnorm,
  route_idx/route_w/route_cnorm [L, B, k], extra["diag"]).
- `data.py` CounterFact loading, seed-0 shuffle, `prepare_case(rec, tok, token_rule="space"|"paper_like",
  special_tokens=True|False, prefix_ids=None)`, filters (STRICT 1.0/0.5, RELAXED 0.5/0.25), splits.
- `noise.py` σ = mult × embed std; per-case `torch.Generator(0 + case_id)`.
- `protocol.py` `cases_by_id(model, ids, token_rule, special_tokens, prefix_ids)`, `load_case_sets`,
  `active_controls` (random.Random(1000 + case_id)).
- `stats.py` `summarize` (mean, 5,000-resample percentile bootstrap CI, positive fraction, 10,000 sign-flip p), `fmt`.
- `analysis.py` post-processing on a run dir: `load_model(run)`, `layer_analysis`, `expert_table`, `select_expert`
  (recurrence-first), `evaluate_expert`, `gate_matched_control`, `all_active_rank`, `active_pair_equal_norm`,
  `coalitions`, `stability_grid`, `relation_heldout`, `noise_table`, `funnel_check`.
- `report.py` `build()` → Tables 1–16 (md+csv) + `figure1()`; `alt_summary(suffix=...)` for secondary runs;
  `write_report.py` `write(...)` → `results/REPORT.md`.
- `models.py` `MODELS` registry: `qwen3`, `mixtral`, `olmoe`, `olmoe_instruct`, `qwen3_instruct`, `qwen3_coder`,
  `mixtral_instruct` → repo, label, n_controls (3 for top-8, 1 for top-2), paper_layer, paper_expert.

`moetrace/` (extensions, one module per direction, none edits the base modules)
- `ext1_analysis.py` per-layer best expert, joint (layer, expert) search, concentration, grid robustness.
- `ext2_attn.py` attention / MoE / block curves, peaks, shares, additivity.
- `ext2_zoo.py` chat-template prefixes, usage specs, cross-model summaries, sink fractions.
- `ext3_variants.py` position-0 substitutions, sink transplant, corpus routing helpers.
- `ext4_data.py` CodeFact item → `data.Case` (categories S1-S3, R1-R3; single-token continuation with boundary back-off).
- Phase 2 (ext5): `ext5_subject.py` (`SubjectEngine.run_subject`, `SubjectPrefill(rec_pos)`, `SubjectSpawn(pos)`: suffix
  wavefront rows patched at position p; copies engine's `_build_v` as `_build_v_at`), `ext5_rank.py` (full (layer,
  expert) rankings, Kendall τ, greedy/coverage minimal sets, exact-subset analysis), `ext5_metrics.py` (F5 metric
  columns, rescue swap so `analysis.py`/`ext1_analysis.py` run under Δp/Δlog p/rank/KL), `ext5_heads.py` +
  `ext5_heads_section.py` (per-head rankings, Spec, additivity, attention-mass classes, section writer).
  Engine additions by ext5-engine (in `engine.py`, backward compatible): `Engine.run(..., metrics=True)` →
  `PassResult.metrics_prefill/.metrics_spawn`; `PrefillSpec.clean_ref`; kinds `attn_head` (`SpawnSpec.expert` = head),
  `coalition_set` (`SpawnSpec.experts`), `multi` (`SpawnSpec.steps`); `DiagSpec.attn_heads_final`, `spawn_vectors`;
  `run_sweep.py`/`run_expert.py` flags `--metrics`, `--agent`. Verification: `results/verify_ext5_engine_olmoe.json`,
  `results/verify_ext5_subject_olmoe.json`; `verify_olmoe.json` unchanged vs `verify_olmoe_before_ext5.json`.

`scripts/` — base CLIs take a model key; `--out <run>` selects `results/<run>/`; `--token-rule space|paper_like`;
`--no-special-tokens` = no BOS; `--layer-chunks N` bounds wavefront rows.
- Base: `run_filter.py`, `run_sweep.py [--sets paper,strict,relaxed]`, `run_expert.py --layers ... [--no-pairs]`,
  `auto_expert.sh`, `run_noise.py`, `hf_reference_check.py`, `verify_olmoe.py`, `verify_olmoe_fp32.py`,
  `mixtral_compare.py`, `mixtral_run_section.py`, `build_final_report.py [--done]`, `chain1/2/3.sh`.
- Infrastructure: `gpu_queue.sh` (mandatory lock for GPU jobs), `build_extensions_report.py` (assembles
  `results/sections/ext*.md` into `results/EXTENSIONS_REPORT.md`), `run_agent.sh` (headless supervisor, section 7),
  `download_models.sh`, `ext2_zoo_download.sh`, `setup_env.sh`.
- Direction 1: `ext1_analyze.py` (reads `results/<run>_alllayers`), `ext1_chain*.sh`.
- Direction 2b: `ext2_attn_sweep.py <model> --out <run> --sets paper [--no-special-tokens]`, `ext2_attn_verify.py`,
  `ext2_attn_gate.py`, `ext2_attn_analyze.py`, `ext2_attn_chain.sh`.
- Direction 2: `ext2_zoo_usage.py` (writes `data/model_usage/<key>.json`), `ext2_zoo_filter.py <key> --protocol
  default|nobos|chat`, `ext2_zoo_sweep.py`, `ext2_zoo_expert.py [--resume]`, `ext2_zoo_attn.py`, `ext2_zoo_analyze.py`,
  `ext2_zoo_chain.sh <key>`.
- Direction 3: `ext3_run_variants.py <model> --variants bos,nobos,nl,dot,...,sinkfull,sinkkey`, `ext3_corpus_routing.py
  <model> --corpus wiki|code`, `ext3_verify_diag.py`, `ext3_gate.py`, `ext3_analyze.py`, `ext3_chain*.sh`.
- Direction 4: `ext4_build_codefact.py` → `data/codefact/items.jsonl`; `ext4_scan.py <model> --out <run> --protocol
  raw|nobos|chat` (calibration scan + layer patches, token-budget chunking); `ext4_select.py`; `ext4_run_expert.py
  --no-pairs`; `ext4_analyze.py`; `ext4_chain*.sh`.
- Phase 2: `ext5_subject_{verify,gate,sweep,select,expert,analyze}.py`, `ext5_subject_chain.sh` (F4);
  `ext5_engine_verify.py`, `ext5_engine_chain.sh`, `ext5_heads_sweep.py <model> --out <run> --layers ... --base-run
  <run>`, `ext5_heads_analyze.py`, `ext5_subsets_run.py <model> --out <run> --layers ... --base-run <metrics-run>`
  (F2, F1.3, F5 runs); `ext5_rank_analyze.py`, `ext5_rank_subsets.py`, `ext5_rank_text.py`, `ext5_metrics_analyze.py`,
  `ext5_metrics_text.py` (F1, F5 analysis; CPU only).

`results/` run directories (each has `run_meta.json`; row-level Parquet: `sweep_rows`, `sweep_routing`,
`sweep_cases`, `expert_rows`, `expert_prefill_*`; extension runs add `zoo_summary.json`, `sink_diag.json`, ...):
- base: `qwen3`, `mixtral` (tokenizer defaults, three case sets), `mixtral_nobos` (paper set, no BOS), `qwen3_alt`,
  `mixtral_alt` (paper_like token rule), `olmoe` (pilot);
- Direction 1: `{qwen3_bos,mixtral_bos,mixtral_nobos}_alllayers`;
- Direction 3: `mixtral_{bos,nobos}_diag`, `mixtral_nobos_prefix_<tok>`, `mixtral_nobos_shift1`,
  `mixtral_nobos_sink_{full,keyonly}`, `mixtral_bos_prefix_bos`, `mixtral_{bos,nobos}_corpus_{wiki,code}`,
  `qwen3_nobos_diag`, `qwen3_bos_prefix_eot`;
- Direction 2b: `{qwen3_bos,mixtral_bos,mixtral_nobos,olmoe}_attnsweep`;
- Direction 2: `<key>_{default,nobos,chat}` for the five zoo models (+ `_attnsweep`), `olmoe_default`;
- Direction 4: `codefact_{qwen3_raw,mixtral_nobos,qwen3_coder_raw,qwen3_coder_chat}`;
- Phase 2: `{qwen3_bos,mixtral_nobos,mixtral_bos}_subject` (F4; `sweep_rows` has `pos`, prefill kinds clean/noised/
  noised_lastonly/noised_exceptlast, expert kinds incl. `expert_fixed`), `{qwen3,mixtral_nobos,mixtral_bos}_metrics`
  (F5; six metric columns), `{qwen3,mixtral_nobos,mixtral_bos,olmoe}_heads` (`head_rows.parquet`, `head_attn_final.npz`),
  `{qwen3,mixtral_nobos}_subsets` (`subset_rows.parquet`, all 2^k−1 clean-active subsets + same-pass reference rows).
Tables: `results/tables/table_01..16` (base), `ext{1,2,2_attn,2_zoo,3,4}_*` and `ext5_{subject,rank,heads,metrics}_*`;
figures likewise; sections in `results/sections/`; analysed numbers in `results/{summary,ext1_summary,
ext2_attn_summary,ext2_zoo_summary,ext3_numbers,ext4_summary,mixtral_compare,ext5_subject_summary,ext5_rank_summary,
ext5_subsets_summary,ext5_heads_summary,ext5_metrics_summary}.json`.

Recipes:
- *Paper protocol on a new MoE model*: add a `MODELS` entry (family must be Qwen3-MoE, Mixtral or OLMoE; other
  families need `arch.py` key templates and possibly engine work), write `data/model_usage/<key>.json` with
  `ext2_zoo_usage.py`, then `ext2_zoo_filter.py` → `ext2_zoo_sweep.py` → `ext2_zoo_expert.py` → `ext2_zoo_analyze.py`.
- *New intervention*: add a kind to `KINDS` and its vector in the engine's spawn-vector builder; verify on OLMoE
  against transformers hooks as `ext2_attn_verify.py` does; re-run `verify_olmoe.py` and confirm identity with the
  previous `results/verify_olmoe*.json`.
- *New counterfactual dataset*: produce `data.Case`-compatible objects (ids, subject_pos, true_id, foil_id) as
  `ext4_data.py` does, then reuse `ext4_scan.py` / `ext4_run_expert.py` / `ext4_analyze.py`.
- *Rebuild deliverables*: `python scripts/build_final_report.py` (base) and `python scripts/build_extensions_report.py`.

## 6. Conventions and rules for work here

- The paper's Table 8 case IDs (`data/paper_case_ids.json`) are the PRIMARY case set; our own strict/relaxed sets are
  secondary. Keep the discovery/validation assignment from the paper. Instruct variants are evaluated on their base
  model's paper case set for comparability, plus their own strict set.
- Protocol labels: `default` = tokenizer defaults, raw cloze; `nobos` = no special tokens (the paper's protocol; for
  Qwen3 and OLMoE `default` ≡ `nobos`); `chat` = chat template with the cloze prompt inside the assistant turn. Keep the
  main tables on the documented default and report variants as labelled runs. For any Mixtral experiment meant to
  match the paper, use `--no-special-tokens`.
- Prefer counts and set membership (activity counts, funnel pass counts, selections) as "fingerprints" when comparing
  with the paper; means carry bf16 noise (per-case rescue ±0.1–0.6, 128-case means ±0.02–0.05).
- No quantization, ever: bf16 weights and activations are the object of study.
- Every run dir gets a `run_meta.json` (model, flags, case sets, chunking, command). Save row-level Parquet after
  every pass. Log every milestone to `logs/PROGRESS.md` (append-only, UTC timestamps).
- Parallel agents own disjoint files (base modules vs `ext*` modules, one script prefix per direction) and share the
  GPU only through `gpu_queue.sh`; never stop a running chain to edit it, append a new chain.
- Commits: trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Pushing works from this shell (the
  user's GitHub credentials are configured); commit sub-agent outputs selectively with `git add <paths>`.

## 7. Unattended / headless workflow (see PLAYBOOK.md)

Two patterns were used and both are documented in PLAYBOOK.md:
- *Headless agent* (2026-09-03): an interactive session wrote `HANDOFF.md`, the **user** started
  `scripts/run_agent.sh` in tmux (the permission classifier blocks an agent from launching another Claude process),
  the agent executed until `results/DONE`; the supervisor now parses usage-limit reset times and sleeps until then.
- *Coordinator + sub-agents* (2026-09-14): one session planned (`RESEARCH_PLAN.md`), launched one sub-agent per
  direction with a brief (goal, owned files, deliverable section, PROGRESS entries, final-report format), serialised
  the GPU with `gpu_queue.sh`, reviewed sections and committed. Sub-agents die at the account usage limit; GPU chains
  run detached and finish anyway; on reset, resume the agent with "resume from disk, do not redo GPU work".
- On any resume: read `logs/PROGRESS.md` (tail), `logs/gpu_queue.log`, `logs/chain*.log`, `logs/agent_status.txt`
  if present, then the run dirs' `run_meta.json`, to find the first unexecuted step.

## 8. Document map

- `README.md` public overview, headline table, how to reproduce, extension summary.
- `PLAN.md` feasibility analysis (why layer streaming), full implementation checklist, assumptions.
- `RESEARCH_PLAN.md` the four extension directions, experiments, agent assignment, user decisions.
- `HANDOFF.md` the operational brief the headless agent executed for the base reproduction.
- `PLAYBOOK.md` the unattended-run process, timelines, problems and fixes (both patterns).
- `results/REPORT.md` base deliverable: sections 1–8 plus 6b (Mixtral without BOS).
- `results/EXTENSIONS_REPORT.md` extension deliverable, assembled from `results/sections/ext*.md`.
- `docs/ext3_literature_review.md` attention sinks, BOS, MoE routing, tokenisation conventions (30 references).
- `data/codefact/build_stats.md`, `data/codefact/samples.md` CodeFact construction and eyeballed examples;
  `data/model_usage/*.json` per-model tokenisation / template specs.
- `logs/PROGRESS.md` timestamped log of everything that happened, including handovers and incidents.
- Paper PDF / LaTeX are gitignored (`2606.03780.pdf`, `paper.txt`, `paper_src/`); present locally on this machine.
