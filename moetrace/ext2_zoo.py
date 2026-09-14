"""ext2-model-zoo (RESEARCH_PLAN Direction 2): protocols, chat-prefix rendering, run-directory naming, run_meta and the
sink diagnostic for the five additional models (Qwen3-30B-A3B-Instruct-2507, Qwen3-Coder-30B-A3B-Instruct,
Mixtral-8x7B-Instruct-v0.1, OLMoE-1B-7B-0125 base and Instruct).

Protocols (RESEARCH_PLAN "Decisions" 2: instruct models are run both raw and chat-wrapped):
  default  tokenizer defaults (BOS if the tokenizer adds one), raw cloze prompt   = "the way the model is meant to be used" for base models
  nobos    no special tokens, raw cloze prompt                                    = the paper's protocol (only differs from `default`
           when the tokenizer adds a BOS token, i.e. the Mistral family)
  chat     the chat template rendered with a short user instruction and an OPENED assistant turn, tokenised as `prefix_ids`;
           the raw cloze prompt follows inside the assistant turn (special_tokens=False for the cloze part because the
           rendered template already carries the BOS token where the model has one). The final position is still the
           prompt's last token and subject noise still hits the subject tokens (positions shifted by the prefix length),
           so the next-token semantics of the cloze task are unchanged.   Instruct models only.

Every run directory is results/<modelkey>_<protocol>[_<experiment>]/ and carries a run_meta.json.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .models import MODELS, RESULTS

USAGE_DIR = "/home/ubuntu/MOE/data/model_usage"
DIAG_ROOT = "/opt/dlami/nvme/moe_ext2"  # raw diagnostics (large) stay on the NVMe scratch, summaries under results/
INSTRUCTION = "Complete the sentence with the single most likely next word."
ZOO_KEYS = ("olmoe", "olmoe_instruct", "qwen3_instruct", "qwen3_coder", "mixtral_instruct")
PROTOCOLS = ("default", "nobos", "chat")
MAX_ROWS_PER_CHUNK = 80_000  # one Qwen3 pass at 85k wavefront rows used 18.5 GB of the 23 GB (ext1: 86k rows, 16 GB); keep chunks at or below this


def usage_path(key: str) -> str:
    return os.path.join(USAGE_DIR, f"{key}.json")


def load_usage(key: str) -> dict:
    with open(usage_path(key)) as f:
        return json.load(f)


def protocol_list(key: str, usage: Optional[dict] = None) -> list[dict]:
    """Protocols to run for a model: `default` always; `nobos` only when it differs from default (tokenizer adds BOS);
    `chat` for instruct models. Each entry: {protocol, run, equivalent_to}."""
    u = usage or load_usage(key)
    m = MODELS[key]
    out = [dict(protocol="default", run=True, equivalent_to=None if u["tokenizer"]["adds_bos"] else "nobos")]
    if u["tokenizer"]["adds_bos"]:
        out.append(dict(protocol="nobos", run=True, equivalent_to=None))
    else:
        out.append(dict(protocol="nobos", run=False, equivalent_to="default"))
    if m.get("instruct") and u.get("chat_template"):
        out.append(dict(protocol="chat", run=True, equivalent_to=None))
    return out


def protocols_to_run(key: str) -> list[str]:
    return [p["protocol"] for p in protocol_list(key) if p["run"]]


def run_dir_name(key: str, protocol: str, experiment: Optional[str] = None) -> str:
    return f"{key}_{protocol}" + (f"_{experiment}" if experiment else "")


def chat_prefix(tok, usage: dict) -> tuple[list[int], str]:
    """Render the chat template with the fixed user instruction and an opened assistant turn; tokenise without adding
    special tokens (the template carries them). Returns (ids, rendered text)."""
    msgs = []
    if usage.get("default_system_prompt"):
        msgs.append({"role": "system", "content": usage["default_system_prompt"]})
    msgs.append({"role": "user", "content": INSTRUCTION})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [int(t) for t in ids], text


@dataclass
class ProtocolSpec:
    model: str
    protocol: str
    special_tokens: bool
    prefix_ids: list
    prefix_text: str
    run_dir: str
    label: str
    equivalent_to: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


def protocol_spec(key: str, protocol: str, tok=None, experiment: Optional[str] = None) -> ProtocolSpec:
    u = load_usage(key)
    m = MODELS[key]
    plist = {p["protocol"]: p for p in protocol_list(key, u)}
    if protocol not in plist:
        raise ValueError(f"{key}: protocol {protocol} not applicable ({list(plist)})")
    eq = plist[protocol]["equivalent_to"]
    if protocol == "default":
        st, pre, txt = True, [], ""
        lab = f"{m['label']}, tokenizer defaults" + (" (adds BOS)" if u["tokenizer"]["adds_bos"] else " (no special tokens; = paper protocol)")
    elif protocol == "nobos":
        st, pre, txt = False, [], ""
        lab = f"{m['label']}, no special tokens (paper protocol)"
    elif protocol == "chat":
        if tok is None:
            from transformers import AutoTokenizer
            from .arch import snapshot_dir
            tok = AutoTokenizer.from_pretrained(snapshot_dir(m["repo"]))
        pre, txt = chat_prefix(tok, u)
        st = False
        lab = f"{m['label']}, chat template (cloze inside the assistant turn)"
    else:
        raise ValueError(protocol)
    return ProtocolSpec(key, protocol, st, pre, txt, run_dir_name(key, protocol, experiment), lab, eq)


def write_run_meta(run_dir: str, meta: dict, update: bool = False) -> dict:
    od = os.path.join(RESULTS, run_dir)
    os.makedirs(od, exist_ok=True)
    p = os.path.join(od, "run_meta.json")
    out = {}
    if update and os.path.exists(p):
        with open(p) as f:
            out = json.load(f)
    out.update(meta)
    out.setdefault("created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=str)
    return out


def base_meta(spec: ProtocolSpec, stage: str, command: str, extra: Optional[dict] = None) -> dict:
    m = MODELS[spec.model]
    u = load_usage(spec.model)
    d = dict(model=spec.model, repo=m["repo"], label=m["label"], family=m.get("family"), base_model=m.get("base"), instruct=bool(m.get("instruct")),
             protocol=spec.protocol, protocol_label=spec.label, protocol_equivalent_to=spec.equivalent_to,
             special_tokens=spec.special_tokens, prefix_ids=list(spec.prefix_ids), prefix_text=spec.prefix_text, prefix_len=len(spec.prefix_ids),
             chat_instruction=INSTRUCTION if spec.protocol == "chat" else None, token_rule="space", sigma_mult=3.0,
             tokenizer_adds_bos=u["tokenizer"]["adds_bos"], usage_spec=usage_path(spec.model),
             experiment="ext2 model zoo (RESEARCH_PLAN Direction 2)", agent="ext2-model-zoo", stage=stage, command=command)
    if extra:
        d.update(extra)
    return d


# ---------------------------------------------------------------------------------------------------------------
# sink diagnostic (ext3 follow-up requested by the coordinator): per run, the fraction of prompts whose FINAL position
# carries the sink state, from the engine's DiagSpec(resid_norms, attn_final) recorded on the clean prefill rows.
# ---------------------------------------------------------------------------------------------------------------
def sink_summary(resid_norms: np.ndarray, attn_final: np.ndarray, lens: np.ndarray, prefix_len: int = 0) -> dict:
    """resid_norms fp32 [L, n, T]; attn_final fp16 [L, n, nH, T] (final-position attention per head); lens [n].

    Per layer: fraction of prompts whose final position has the maximal residual norm over all positions; fraction whose
    final position puts more head-mean attention mass on itself than on any other single position; mean self mass; mean
    mass on position 0; mean norm ratio final / median-of-others. The "sink layer" is the layer (in the first half of the
    network) where the massive-activation state is most pronounced (largest mean ratio max-norm / median-norm), and the
    headline numbers are read there. `prefix_len` positions (chat prefix) are part of the prompt and count as positions.
    """
    L, n, T = resid_norms.shape
    ar = np.arange(n)
    fin = lens - 1
    per_layer = []
    for l in range(L):
        rn = resid_norms[l].astype(np.float64)  # [n, T]
        valid = np.arange(T)[None, :] < lens[:, None]
        rn_m = np.where(valid, rn, -np.inf)
        argmax = rn_m.argmax(1)
        final_is_max = argmax == fin
        med_others = np.array([np.median(np.delete(rn[i, : lens[i]], fin[i])) if lens[i] > 1 else np.nan for i in range(n)])
        ratio = rn[ar, fin] / np.maximum(med_others, 1e-6)
        mx_ratio = rn_m.max(1) / np.maximum(np.array([np.median(rn[i, : lens[i]]) for i in range(n)]), 1e-6)
        att = attn_final[l].astype(np.float32).mean(1)  # head-mean [n, T]
        att = np.where(valid, att, -1.0)
        self_mass = att[ar, fin]
        att_no_self = att.copy()
        att_no_self[ar, fin] = -1.0
        self_is_max = self_mass > att_no_self.max(1)
        per_layer.append(dict(layer=l, frac_final_max_norm=float(final_is_max.mean()), frac_final_self_attn_max=float(self_is_max.mean()),
                              mean_self_mass=float(self_mass.mean()), mean_mass_pos0=float(att[:, 0].mean()),
                              frac_pos0_max_norm=float((argmax == 0).mean()), mean_final_norm_ratio=float(np.nanmean(ratio)),
                              median_final_norm_ratio=float(np.nanmedian(ratio)), mean_max_over_median=float(np.nanmean(mx_ratio)),
                              frac_self_mass_gt_half=float((self_mass > 0.5).mean())))
    half = max(1, L // 2)
    sink_layer = int(np.argmax([p["mean_max_over_median"] for p in per_layer[1:half]]) + 1) if half > 1 else 0
    sl = per_layer[sink_layer]
    # a prompt "carries the sink at its final position" if that holds at the sink layer; also report the any-layer version
    fm_any = np.zeros(n, dtype=bool)
    for l in range(1, half):
        rn = resid_norms[l]
        valid = np.arange(T)[None, :] < lens[:, None]
        fm_any |= np.where(valid, rn, -np.inf).argmax(1) == fin
    return dict(n=int(n), n_layers=int(L), T=int(T), prefix_len=int(prefix_len), sink_layer=sink_layer,
                frac_final_max_norm_at_sink_layer=sl["frac_final_max_norm"], frac_final_self_attn_max_at_sink_layer=sl["frac_final_self_attn_max"],
                frac_pos0_max_norm_at_sink_layer=sl["frac_pos0_max_norm"], mean_mass_pos0_at_sink_layer=sl["mean_mass_pos0"],
                mean_self_mass_at_sink_layer=sl["mean_self_mass"], frac_final_max_norm_any_early_layer=float(fm_any.mean()),
                mean_mass_pos0_L1=per_layer[1]["mean_mass_pos0"] if L > 1 else None, per_layer=per_layer)


def diag_dir(run_dir: str) -> str:
    d = os.path.join(DIAG_ROOT, run_dir)
    os.makedirs(d, exist_ok=True)
    return d


def pattern_label(layer_val: dict, spec_all: Optional[dict], rescue_all: Optional[dict], coalition_clean: Optional[dict]) -> str:
    """A: one positive, specific expert (Spec CI above 0 and rescue CI above 0); B: layer localised but the selected
    expert is not specific (or none is recurrent) while a coalition carries the effect; C: no layer-level localisation
    (validation rescue CI at L* includes 0)."""
    if layer_val is None or not (layer_val["ci_lo"] > 0):
        return "C"
    if spec_all is not None and rescue_all is not None and spec_all["n"] > 0 and spec_all["ci_lo"] > 0 and rescue_all["ci_lo"] > 0:
        return "A"
    return "B"


def sink_positions(resid_norms: np.ndarray, lens: np.ndarray, layer: int) -> dict:
    """Where the maximal-norm (sink) position sits at `layer`: histogram of the argmax position over prompts (mode, fraction
    at the mode, fraction at position 0, fraction at the final position)."""
    rn = resid_norms[layer].astype(np.float64)
    n, T = rn.shape
    valid = np.arange(T)[None, :] < lens[:, None]
    am = np.where(valid, rn, -np.inf).argmax(1)
    vals, counts = np.unique(am, return_counts=True)
    order = np.argsort(-counts)
    hist = [(int(vals[i]), int(counts[i])) for i in order[:5]]
    mode = hist[0][0]
    return dict(layer=int(layer), mode_position=mode, frac_at_mode=float(hist[0][1] / n), top5=hist,
                frac_pos0=float((am == 0).mean()), frac_final=float((am == lens - 1).mean()))
