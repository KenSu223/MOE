# Brief: p6-engine (dense / Qwen2 / Qwen2-MoE support, head masks, neuron-level kinds, verification)

Read logs/agent_brief_p6_common.md first. Its rules apply.

## Goal

Extend the layer-streaming engine to the Phase-6 models and interventions without changing any existing result, verify everything, and announce readiness.

## Files you own

- `moetrace/arch.py`, `moetrace/weights.py`, `moetrace/engine.py` — additive changes only; existing behaviour must stay bit-identical.
- `moetrace/models.py` — new MODELS entries only.
- `scripts/ext14_engine_*.py`
- `results/verify_ext14_*.json`
- `logs/ext14_engine_api.md`, `logs/agent_status_p6-engine.md`

Generic runners (`scripts/ext7_wino_*`) and analysis code (`moetrace/analysis.py`, ext* modules) may contain model-specific assumptions: number of experts, top-k, ``qwen3``/``mixtral`` keys, layer counts.
- Do NOT edit them.
- List every assumption you find, with file:line, in `logs/ext14_engine_api.md` under "Runner compatibility". Post a `[p6-engine] REQUEST p6-data/p6-analysis: …` line when one blocks a model.

## Safe development

Other chains import the engine from the main tree while you work.
- Develop and test in a git worktree: `git worktree add /home/ubuntu/MOE_wt_p6engine -b p6-engine`. Same checkpoints via HF_HOME; symlink `data/` and `results/` read-only as needed.
- When everything passes, copy the changed files into /home/ubuntu/MOE atomically (write a temp file, then `os.replace`).
- Re-run the regression checks in the main tree afterwards.
- Never leave a half-edited engine file in the main tree.

## Tasks

1. **Dense families.**
   - `MistralForCausalLM` (Mistral-7B-v0.1), `Qwen2ForCausalLM` (Qwen2-7B: q/k/v biases), `Qwen3ForCausalLM` (Qwen3-1.7B / 4B / 8B / 14B-Base: per-head q/k norm; tied embeddings for ≤ 4B).
   - The MLP is one "expert" with weight 1 (n_experts = 1, top_k = 1, no router). Kinds `layer` / `expert` (expert 0) patch the MLP output; routing records are trivial. Route masks raise a clear error.
   - Prompts of the Mistral family keep BOS (protocol `mistral7b_bos`, as `mixtral_bos`).
2. **`Qwen2MoeForCausalLM`** (Qwen2-57B-A14B).
   - q/k/v biases; 64 routed experts, top-8, `norm_topk_prob` from config (False).
   - Shared expert `mlp.shared_expert.{gate,up,down}_proj` with scalar gate sigmoid(`mlp.shared_expert_gate.weight` · x).
   - Record the shared contribution separately (per-expert contribution arrays get an extra slot or a separate field; document it).
   - Allow a spawn to patch it: e.g. expert index = n_experts means "shared"; `layer` patches routed + shared.
3. **`PrefillSpec.head_mask`.** A tuple of (layer, head) with mode `zero` (primary) | `mean` (caller supplies a per-head mean vector). Applied at every position of that prefill row. Used by p6-analysis for R-causal: knockouts in both the clean and the corrupted runs, then the usual spawns.
4. **Neuron level** (for p6-circuit C1):
   - `DiagSpec.expert_neurons = [(layer, expert), …]` records the expert's intermediate activations a = SiLU(W_gate x) ⊙ (W_up x) at the final position, for every prefill row.
   - A new spawn kind `expert_neurons` (SpawnSpec layer, expert, neuron index set) replaces those neurons' activations by the source row's values at the final position, everything else clean.
   - Dense: expert 0 = the MLP.
5. **`moetrace/models.py` entries:** `mistral7b`, `qwen2_7b`, `qwen2_57b`, `qwen3_1p7b`, `qwen3_4b`, `qwen3_8b`, `qwen3_14b`, and `mixtral22` (Mixtral-8x22B-v0.1; not downloaded, entry only). Include n_controls (top-8 → 3, top-2 → 1, dense → 0) and paper_layer None.
6. **Verification** (GPU jobs through the queue; short):
   - **Dense:** compare with transformers hooks on the dense model itself (fits on the GPU) for Mistral-7B, Qwen2-7B and Qwen3-4B (and Qwen3-8B if cheap). ≥ 5 STR prompts. Kinds layer, attn_layer, attn_head, block, resid, multi; metrics; head_mask. Thresholds as in earlier verifications (r ≥ 0.99 on rescues, max |ΔΔ| reported). Qwen3-14B (28 GB) cannot be held on the GPU: verify per layer.
   - **Qwen2-MoE:**
     - (a) A tiny randomly initialised `Qwen2MoeForCausalLM` (2–4 layers, small dims, saved under `/opt/dlami/nvme/p6_tiny/`), fp32, compared with transformers: exact to fp32 tolerance.
     - (b) On the real Qwen2-57B, a per-layer check: load one decoder layer's weights into transformers' `Qwen2MoeDecoderLayer` and compare with the engine's layer output for 3 layers × 5 prompts (bf16 tolerance).
     - No full-model offload (disk is tight).
   - **Neuron kind:** patching all neurons of expert e = the `expert` kind (identity); subsets are consistent with linearity in the down-projection.
   - **Regression:** `scripts/verify_olmoe.py` output identical to `results/verify_olmoe.json`; the ext8 / ext9 engine checks (`results/verify_ext8_engine_olmoe.json`, `verify_ext9_engine_olmoe.json`) re-run identical. One small Phase-5 cell re-run (a single sweep chunk) identical to its stored rows.
7. **Handshakes** (PROGRESS, exact):
   - `[p6-engine] HEADMASK READY` (needed first, for the Stage-1 R-causal check);
   - `[p6-engine] NEURONS READY`;
   - `[p6-engine] MODEL READY <proto>` for each of mistral7b_bos, qwen2_7b, qwen3_4b, qwen3_8b, qwen3_14b, qwen3_1p7b, qwen2_57b — each only after its verification passed.
   - Order: HEADMASK → dense models (Mistral-7B first) → NEURONS → Qwen2-57B.

Checkpoints: `HF_HOME=/opt/dlami/nvme/hf`. Downloads are running (`logs/ext14_download.log`); Qwen2-57B-A14B arrives last. Pass rows such as STR prompts from `data/wino_ref` pair files or simple CounterFact prompts.

## Done when

All handshakes are written, `logs/ext14_engine_api.md` documents the new fields, kinds, model keys and runner compatibility, and the regression is identical. Then send your final report.
