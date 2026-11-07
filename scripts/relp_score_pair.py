#!/usr/bin/env python3
"""Portable, single-example RelP scorer using the existing OLMoE/Mistral FModel.

See docs/relp_moe_implementation_guide.md. --validate-only needs only Python's
standard library; inference needs the existing torch/transformers environment.
This is a scoring entry point, not a native-forward equivalence certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_inputs(model_dir: Path, input_path: Path):
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    data = json.loads(input_path.read_text())
    if not isinstance(config, dict) or not isinstance(data, dict):
        raise ValueError("config.json and input JSON must contain objects")
    if config.get("model_type") not in {"olmoe", "mistral"}:
        raise ValueError("FModel supports only the implemented olmoe/mistral layouts")
    vocab = config.get("vocab_size")
    if type(vocab) is not int or vocab <= 0:
        raise ValueError("config.json needs a positive integer vocab_size")
    for name in ("input_ids", "reference_ids"):
        if name == "reference_ids" and name not in data:
            continue
        ids = data.get(name)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{name} must be a nonempty list of token IDs")
        if any(type(i) is not int or not 0 <= i < vocab for i in ids):
            raise ValueError(f"{name} contains invalid token IDs for vocab_size={vocab}")
    if "reference_ids" in data and len(data["reference_ids"]) != len(data["input_ids"]):
        raise ValueError("Paired scoring requires equal, semantically aligned token positions")
    for name in ("target_id", "foil_id"):
        value = data.get(name)
        if type(value) is not int or not 0 <= value < vocab:
            raise ValueError(f"{name} must be one valid vocabulary ID")
    if data["target_id"] == data["foil_id"]:
        raise ValueError("target_id and foil_id must be different")
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError("This FModel implements SiLU experts only")
    for name in ("attention_bias", "mlp_bias", "num_shared_experts", "n_shared_experts",
                 "shared_expert_intermediate_size"):
        if config.get(name):
            raise ValueError(f"Unsupported FModel configuration: {name}")
    if config.get("clip_qkv") is not None:
        raise ValueError("clip_qkv requires a separate RelP-rule audit before using this entry point")
    for name in ("sliding_window", "max_position_embeddings"):
        limit = config.get(name)
        if isinstance(limit, int) and limit > 0 and len(data["input_ids"]) > limit:
            raise ValueError(f"Prompt exceeds {name}={limit}; this entry point uses full causal attention")
    return config, data


def check_loaded_layout(fm):
    """Fail explicitly if a transformers version changes the expected tensor layout."""
    if fm.nH * fm.D != fm.H:
        raise ValueError("FModel requires num_attention_heads * head_dim == hidden_size")
    if fm.nH % fm.nKV:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if getattr(fm.model.lm_head, "bias", None) is not None:
        raise ValueError("FModel does not include an lm_head bias")
    for layer, weights in zip(fm.model.model.layers, fm.layers):
        for module in (layer.self_attn, layer.mlp):
            for name, child in module.named_modules():
                if getattr(child, "bias", None) is not None:
                    raise ValueError(f"FModel does not include module bias: {name}")
        if fm.moe:
            if tuple(weights["gate_up"].shape) != (fm.E, 2 * fm.I, fm.H):
                raise ValueError("Expected fused OLMoE gate_up weights [E, 2I, H]")
            if tuple(weights["down"].shape) != (fm.E, fm.H, fm.I):
                raise ValueError("Expected fused OLMoE down weights [E, H, I]")
        if "qn" in weights and (weights["qn"].numel() != fm.nH * fm.D
                                or weights["kn"].numel() != fm.nKV * fm.D):
            raise ValueError("Only full-projection OLMoE Q/K normalization is implemented")


def serialize_routes(route):
    return [{"layer": layer, "expert_ids": ids.cpu().tolist(),
             "weights": weights.float().cpu().tolist()}
            for layer, (ids, weights) in enumerate(route)]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run(args, config, data):
    # Lazy imports keep --help and --validate-only usable on a documentation laptop.
    import torch
    import transformers

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from moetrace.ext14_circuit_relp import FModel

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    fm = FModel(str(args.model_dir), device=args.device, dtype=dtype)
    check_loaded_layout(fm)
    target, foil = [data["target_id"]], [data["foil_id"]]

    def tensor(ids):
        return torch.tensor([ids], dtype=torch.long, device=args.device)

    reference_q = None
    reference_metric = None
    reference_routes = None
    if "reference_ids" in data:
        with torch.no_grad():
            reference = fm.forward(tensor(data["reference_ids"]), target, foil, record=True)
        reference_q = [q[0].float().cpu() for q in reference["q"]]
        reference_metric = float(reference["delta"][0])
        reference_routes = serialize_routes(reference["route"])
        del reference

    # The main forward IS the differentiable forward. Calling node_grads here
    # would recompute it and add an unnecessary forward to the minimal protocol.
    with torch.enable_grad():
        main = fm.forward(tensor(data["input_ids"]), target, foil, relp=True, grad=True)
        gradients = torch.autograd.grad(main["delta"].sum(), main["q"])
    current_metric = float(main["delta"][0].detach())
    current_routes = serialize_routes(main["route"])
    modes = {"unpaired": []}
    if reference_q is not None:
        modes["paired"] = []
    length = len(data["input_ids"])
    for layer, (activation, gradient) in enumerate(zip(main["q"], gradients)):
        q = activation[0].detach().float().cpu().reshape(length, fm.E, fm.I)
        beta = gradient[0].detach().float().cpu().reshape_as(q)
        ref = reference_q[layer].reshape_as(q) if reference_q is not None else None
        values = (q, beta) if ref is None else (q, beta, ref)
        if not all(bool(torch.isfinite(v).all()) for v in values):
            raise ValueError(f"Non-finite activation or pseudo-gradient at layer {layer}")
        for mode in modes:
            score = (q if mode == "unpaired" else q - ref) * beta
            if not bool(torch.isfinite(score).all()):
                raise ValueError(f"Non-finite {mode} score at layer {layer}")
            # Per-layer top K contains every possible global top K candidate.
            _, indices = torch.topk(score.abs().flatten(), min(args.top_k, score.numel()))
            for flat in indices.tolist():
                position, remainder = divmod(flat, fm.E * fm.I)
                expert, neuron = divmod(remainder, fm.I)
                value = float(score[position, expert, neuron])
                if value == 0:
                    continue
                modes[mode].append({
                    "layer": layer, "position": position, "expert": expert, "neuron": neuron,
                    "score": value, "q_input": float(q[position, expert, neuron]),
                    "q_reference": None if ref is None else float(ref[position, expert, neuron]),
                    "beta_input": float(beta[position, expert, neuron]),
                })
    for mode in modes:
        modes[mode].sort(key=lambda row: (-abs(row["score"]), row["layer"], row["position"],
                                         row["expert"], row["neuron"]))
        modes[mode] = modes[mode][:args.top_k]
    changed_sets = None
    if reference_routes is not None:
        changed_sets = sum(set(a) != set(b)
                           for current, reference in zip(current_routes, reference_routes)
                           for a, b in zip(current["expert_ids"], reference["expert_ids"]))
    metadata = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(args.model_dir), "model_type": config["model_type"],
        "config_sha256": hashlib.sha256((args.model_dir / "config.json").read_bytes()).hexdigest(),
        "fmodel_sha256": hashlib.sha256((project / "moetrace/ext14_circuit_relp.py").read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "device": args.device, "dtype": args.dtype,
        "layers": fm.L, "experts": fm.E, "neurons_per_expert": fm.I, "tokens": length,
        "native_topk_renormalization": fm.norm_topk if fm.moe else None,
        "metric": "logit(target_id) - logit(foil_id), final input position, fp32 readout",
        "metric_input": current_metric, "metric_reference": reference_metric,
        "pseudo_gradient_endpoint": "input_ids", "routing": "natural per forward; frozen backward",
        "forward_calls": 2 if reference_q is not None else 1, "backward_calls": 1,
        "intervention_calls": 0, "native_forward_equivalence_checked": False,
        "changed_topk_sets_across_layer_positions": changed_sets,
        "top_k_per_mode": args.top_k, "ranking": "absolute score; preserve signed score",
        "scope": "single-example neuron ranking; not a validated circuit or router attribution",
    }
    # Do not overwrite previous runs, including if another process created the path while loading.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "input.json", data)
    write_json(args.output_dir / "run_meta.json", metadata)
    write_json(args.output_dir / "scores_top.json", modes)
    write_json(args.output_dir / "routes.json", {"input": current_routes, "reference": reference_routes})
    print(f"Wrote {args.output_dir} ({metadata['forward_calls']} forwards, 1 backward; no interventions)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Local checkpoint/snapshot directory")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="New directory; required for inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true", help="Check input/config only; do not load weights")
    args = parser.parse_args()
    try:
        if args.top_k <= 0:
            raise ValueError("--top-k must be positive")
        args.model_dir = args.model_dir.expanduser().resolve()
        args.input_json = args.input_json.expanduser().resolve()
        config, data = load_inputs(args.model_dir, args.input_json)
        if args.validate_only:
            print("Input/config contract OK. Weights, tokenizer alignment, forward equivalence, and GPU were NOT tested.")
            return
        if args.output_dir is None:
            raise ValueError("--output-dir is required for inference")
        args.output_dir = args.output_dir.expanduser().resolve()
        if args.output_dir.exists():
            raise ValueError("--output-dir already exists; choose a fresh run directory")
        run(args, config, data)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"relp_score_pair: {error}\n")


if __name__ == "__main__":
    main()
