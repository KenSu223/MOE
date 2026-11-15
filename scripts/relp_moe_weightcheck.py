#!/usr/bin/env python3
"""Real-checkpoint weight-loading check for the streamed RelP engine: HF from_pretrained (its own WeightConverter:
per-expert keys -> fused gate_up_proj / down_proj) vs moetrace.weights.CheckpointStore (used by both the functional
engine and the native reference in `verify`), tensor by tensor, bit for bit, for decoder layer 0 and the
embedding / final norm / unembedding. CPU only. Layer 0 only: from_pretrained of a later layer would load every
layer before it, and the key templates differ only by the layer index.

Usage: python scripts/relp_moe_weightcheck.py <model_dir> --out results/relp_moe_dev/weightcheck_<model>.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from moetrace.weights import HostLayerBuffers
    cfg = AutoConfig.from_pretrained(a.model_dir)
    L = cfg.num_hidden_layers
    res = {"model_dir": a.model_dir, "layers": {}}
    for layer in (0,):
        # a model whose only decoder layer is checkpoint layer `layer`: rename keys by loading with a key remap
        c = AutoConfig.from_pretrained(a.model_dir)
        c.num_hidden_layers = layer + 1
        hf = AutoModelForCausalLM.from_pretrained(a.model_dir, config=c, dtype=torch.bfloat16, device_map="cpu")
        lay = hf.model.layers[layer]
        from moetrace.arch import spec_from_config
        from moetrace.weights import CheckpointStore
        cj = json.loads((Path(a.model_dir) / "config.json").read_text())
        spec = spec_from_config(cj)
        store = CheckpointStore(a.model_dir, spec, device="cpu", dtype=torch.bfloat16)
        buf = HostLayerBuffers(spec, torch.bfloat16, pinned=False)
        store._fill_layer_host(layer, buf)
        sm = buf.small
        pairs = {"input_layernorm.weight": sm["ln1"], "post_attention_layernorm.weight": sm["ln2"],
                 "self_attn.q_proj.weight": sm["wq"], "self_attn.k_proj.weight": sm["wk"],
                 "self_attn.v_proj.weight": sm["wv"], "self_attn.o_proj.weight": sm["wo"],
                 "mlp.gate.weight": sm["router"], "mlp.experts.gate_up_proj": buf.gate_up,
                 "mlp.experts.down_proj": buf.down}
        if sm["q_norm"] is not None:
            pairs.update({"self_attn.q_norm.weight": sm["q_norm"], "self_attn.k_norm.weight": sm["k_norm"]})
        sd = lay.state_dict()
        res["layers"][layer] = {k: bool(torch.equal(sd[k], v.to(torch.bfloat16))) for k, v in pairs.items()}
        res["layers"][layer]["hf_param_names_covered"] = sorted(sd) == sorted(pairs)
        if layer == 0:
            g = {"embed": store.cpu_tensor(spec.k_embed).to(torch.bfloat16), "norm": store.cpu_tensor(spec.k_norm).to(torch.bfloat16),
                 "head": store.cpu_tensor(spec.k_head).to(torch.bfloat16)}
            res["globals"] = {"embed": bool(torch.equal(hf.model.embed_tokens.weight, g["embed"])),
                              "norm": bool(torch.equal(hf.model.norm.weight, g["norm"])),
                              "head": bool(torch.equal(hf.lm_head.weight, g["head"]))}
        del hf, lay, sd, buf
    res["passed"] = all(all(v.values()) for v in res["layers"].values()) and all(res["globals"].values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
