"""ext3 (RESEARCH_PLAN Direction 3): position-0 substitution variants and shared helpers for the BOS-mechanism runs.

A variant fixes how the paper's prompts are tokenised: whether the tokenizer's special tokens (Mixtral: BOS `<s>`) are
added, which token ids are prepended in front of the prompt, and the RoPE position of the first token. Each variant is
written to its own run directory `results/<run_dir>/` in the same row-level schema as the main runs (sweep_rows,
sweep_routing, sweep_cases, expert_rows), so `moetrace.analysis` works on it unchanged. Large raw diagnostics go to the
NVMe scratch (DIAG_ROOT), derived summaries stay under results/.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

DIAG_ROOT = "/opt/dlami/nvme/moe_ext3"
RESULTS = "/home/ubuntu/MOE/results"


@dataclass(frozen=True)
class Variant:
    key: str
    model: str
    run_dir: str
    label: str
    special_tokens: bool
    prefix_ids: tuple = ()
    pos_offset: int = 0
    note: str = ""
    sink_from: str = ""  # ext3 transplant: key of the variant (same pass) whose clean row of the same case donates its position-0 K/V
    sink_vscale: float = 1.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["prefix_ids"] = list(self.prefix_ids)
        return d


# Mixtral-8x7B-v0.1 token ids (SentencePiece, 32k): <s>=1, </s>=2, '\n'=<0x0A>=13, '▁.'=842, ','=28725, '▁the'=272,
# '▁workspace'=28640 (one of the highest-id, i.e. rarest, whole-word ASCII tokens of the vocabulary).
# H3 note (coordinator / literature review): RoPE encodes relative position only, so 'shift1' is mathematically the
# no-BOS run; it is kept in a batched pass as a numerics identity check, not as an experiment.
MIXTRAL_VARIANTS: dict[str, Variant] = {v.key: v for v in [
    Variant("bos", "mixtral", "mixtral_bos_diag", "<s> at position 0 (tokenizer default)", True, (), 0,
            "the main run's protocol, re-run with diagnostics"),
    Variant("nobos", "mixtral", "mixtral_nobos_diag", "no token (paper protocol)", False, (), 0,
            "the paper's evident protocol, re-run with diagnostics"),
    Variant("shift1", "mixtral", "mixtral_nobos_shift1", "no token, RoPE positions start at 1", False, (), 1,
            "identity check: RoPE is relative, so this must equal 'nobos' up to bf16 noise"),
    Variant("eos", "mixtral", "mixtral_nobos_prefix_eos", "</s> at position 0", False, (2,), 0, "special token, not BOS"),
    Variant("bosbos", "mixtral", "mixtral_bos_prefix_bos", "<s><s> at positions 0-1", True, (1,), 0, "two BOS tokens"),
    Variant("nl", "mixtral", "mixtral_nobos_prefix_nl", "'\\n' at position 0", False, (13,), 0, "delimiter token"),
    Variant("dot", "mixtral", "mixtral_nobos_prefix_dot", "'.' at position 0", False, (842,), 0, "delimiter token ('▁.')"),
    Variant("comma", "mixtral", "mixtral_nobos_prefix_comma", "',' at position 0", False, (28725,), 0, "delimiter token (',')"),
    Variant("the", "mixtral", "mixtral_nobos_prefix_the", "'the' at position 0", False, (272,), 0, "frequent content token ('▁the')"),
    Variant("rare", "mixtral", "mixtral_nobos_prefix_rare", "'workspace' at position 0", False, (28640,), 0, "rare content token ('▁workspace')"),
    Variant("dot2", "mixtral", "mixtral_nobos_prefix_dot2", "'.' (attached form, id 28723) at position 0", False, (28723,), 0,
            "delimiter token without the SentencePiece space prefix: the form that follows words in running text"),
    Variant("colon", "mixtral", "mixtral_nobos_prefix_colon", "':' at position 0", False, (28747,), 0, "delimiter token (':')"),
    Variant("of", "mixtral", "mixtral_nobos_prefix_of", "'of' at position 0", False, (302,), 0, "function word ('▁of'), a massive-activation carrier in Mixtral per Sun et al. 2024"),
    Variant("space", "mixtral", "mixtral_nobos_prefix_space", "'▁' (lone space) at position 0", False, (28705,), 0, "whitespace token"),
    Variant("unk", "mixtral", "mixtral_nobos_prefix_unk", "<unk> at position 0", False, (0,), 0, "the third special token"),
    Variant("sinkfull", "mixtral", "mixtral_nobos_sink_full", "no token; <s> key+value transplanted as an extra slot", False, (), 1,
            "identity check of the transplant mechanism: must equal the BOS run (position 0's K/V is the only channel of <s>)", "bos", 1.0),
    Variant("sinkkey", "mixtral", "mixtral_nobos_sink_keyonly", "no token; <s> KEY transplanted, value = 0 (pure absorber)", False, (), 1,
            "H1 test: attention can be parked on a sink that injects nothing", "bos", 0.0),
]}

# Qwen3-30B-A3B-Base: the tokenizer adds no BOS; <|endoftext|> = 151643 is its document separator.
QWEN3_VARIANTS: dict[str, Variant] = {v.key: v for v in [
    Variant("default", "qwen3", "qwen3_nobos_diag", "no token (tokenizer default = paper protocol)", True, (), 0,
            "the main run's protocol, re-run with diagnostics"),
    Variant("eot", "qwen3", "qwen3_bos_prefix_eot", "<|endoftext|> at position 0", False, (151643,), 0,
            "symmetric test: Qwen's document separator prepended"),
]}

VARIANTS = {"mixtral": MIXTRAL_VARIANTS, "qwen3": QWEN3_VARIANTS}


def diag_dir(run_dir: str) -> str:
    d = os.path.join(DIAG_ROOT, run_dir)
    os.makedirs(d, exist_ok=True)
    return d


def write_run_meta(run_dir: str, meta: dict) -> None:
    os.makedirs(os.path.join(RESULTS, run_dir), exist_ok=True)
    with open(os.path.join(RESULTS, run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)


def load_diag(run_dir: str) -> dict:
    """Raw diagnostics of a variant run: npz arrays (attn_final, resid_norms, router_logits_final, token_logprobs,
    logprob_true, logprob_foil, lens, row_kind, case_ids) plus resid_final (torch bf16 [L, B, H]) if present."""
    import numpy as np
    d = diag_dir(run_dir)
    out = dict(np.load(os.path.join(d, "diag.npz"), allow_pickle=True))
    rf = os.path.join(d, "resid_final.pt")
    if os.path.exists(rf):
        import torch
        out["resid_final"] = torch.load(rf)
    return out
