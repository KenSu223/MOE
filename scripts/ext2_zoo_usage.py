"""ext2-model-zoo: write data/model_usage/<key>.json for a downloaded model, verified from the snapshot's
tokenizer_config.json, generation_config.json, config.json, chat template (tokenizer_config or chat_template.jinja) and
model card (README.md), plus an empirical tokenizer check (does tok(text) add BOS/EOS?) and the rendered chat prefix.

Usage: python scripts/ext2_zoo_usage.py <model_key> [<model_key> ...]
"""
import json, os, re, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
from transformers import AutoTokenizer
from moetrace.models import MODELS
from moetrace.arch import snapshot_dir
from moetrace.ext2_zoo import INSTRUCTION, USAGE_DIR, usage_path


def read_json(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def model_card_facts(d: str) -> dict:
    p = os.path.join(d, "README.md")
    if not os.path.exists(p):
        return {"present": False}
    txt = open(p, errors="replace").read()
    facts = {"present": True, "chars": len(txt)}
    dt = sorted(set(m.lower() for m in re.findall(r"torch_dtype\s*=\s*[\"']?([A-Za-z0-9.]+)|(bfloat16|bf16|float16|fp16)", txt) for m in m if m))
    facts["dtype_mentions"] = dt
    facts["mentions_system_prompt"] = bool(re.search(r"system prompt", txt, re.I))
    facts["mentions_bos"] = bool(re.search(r"\bBOS\b|<s>|beginning.of.sequence", txt))
    facts["mentions_thinking"] = bool(re.search(r"thinking mode|enable_thinking|<think>", txt))
    facts["mentions_apply_chat_template"] = "apply_chat_template" in txt
    m = re.search(r"(?:recommend|suggest)[^.\n]{0,120}(?:temperature|Temperature)[^.\n]{0,200}", txt)
    facts["recommended_sampling"] = m.group(0).strip() if m else None
    facts["license_line"] = (re.search(r"license:\s*(\S+)", txt).group(1) if re.search(r"license:\s*(\S+)", txt) else None)
    return facts


def main(keys):
    os.makedirs(USAGE_DIR, exist_ok=True)
    for key in keys:
        m = MODELS[key]
        d = snapshot_dir(m["repo"])
        tc = read_json(d, "tokenizer_config.json") or {}
        gc = read_json(d, "generation_config.json") or {}
        cfg = read_json(d, "config.json") or {}
        stm = read_json(d, "special_tokens_map.json") or {}
        tok = AutoTokenizer.from_pretrained(d)
        probe = "The Eiffel Tower is located in"
        ids_default = tok(probe)["input_ids"]
        ids_plain = tok(probe, add_special_tokens=False)["input_ids"]
        added_front = ids_default[: len(ids_default) - len(ids_plain)] if ids_default[-len(ids_plain):] == ids_plain else None
        added_back = ids_default[len(ids_plain):] if ids_default[: len(ids_plain)] == ids_plain else None
        adds_bos = bool(len(ids_default) > len(ids_plain) and ids_default[0] != ids_plain[0])
        adds_eos = bool(len(ids_default) > len(ids_plain) and ids_default[-1] != ids_plain[-1])
        chat_template = tc.get("chat_template")
        src = "tokenizer_config.json" if chat_template else None
        jin = os.path.join(d, "chat_template.jinja")
        if not chat_template and os.path.exists(jin):
            chat_template = open(jin).read()
            src = "chat_template.jinja"
        if not chat_template and getattr(tok, "chat_template", None):
            chat_template = tok.chat_template
            src = "tokenizer.chat_template"
        rendered = None
        rendered_ids = None
        sys_prompt = None
        if chat_template:
            try:
                rendered = tok.apply_chat_template([{"role": "user", "content": INSTRUCTION}], tokenize=False, add_generation_prompt=True)
                rendered_ids = tok(rendered, add_special_tokens=False)["input_ids"]
                # a default system prompt is one that the template inserts on its own when no system message is given
                rendered_user_only = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
                ms = re.search(r"system\n(.*?)<\|im_end\|>", rendered_user_only, re.S)
                if ms:
                    sys_prompt = ms.group(1)
            except Exception as ex:
                rendered = f"RENDER ERROR: {ex!r}"
        usage = {
            "key": key, "repo": m["repo"], "label": m["label"], "family": m.get("family"), "base_counterpart": m.get("base"), "instruct": bool(m.get("instruct")),
            "snapshot": d, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "architecture": cfg.get("architectures"), "n_layers": cfg.get("num_hidden_layers"),
            "n_experts": cfg.get("num_experts") or cfg.get("num_local_experts"), "top_k": cfg.get("num_experts_per_tok"),
            "recommended_dtype": cfg.get("torch_dtype") or cfg.get("dtype"), "dtype_source": "config.json torch_dtype" if (cfg.get("torch_dtype") or cfg.get("dtype")) else None,
            "tokenizer": {"class": tc.get("tokenizer_class"), "add_bos_token_field": tc.get("add_bos_token"), "add_eos_token_field": tc.get("add_eos_token"),
                          "bos_token": tc.get("bos_token") if not isinstance(tc.get("bos_token"), dict) else tc["bos_token"].get("content"),
                          "eos_token": tc.get("eos_token") if not isinstance(tc.get("eos_token"), dict) else tc["eos_token"].get("content"),
                          "pad_token": tc.get("pad_token") if not isinstance(tc.get("pad_token"), dict) else tc["pad_token"].get("content"),
                          "special_tokens_map": {k: (v if not isinstance(v, dict) else v.get("content")) for k, v in stm.items()},
                          "adds_bos": adds_bos, "adds_eos": adds_eos, "empirical_probe": probe, "ids_default": ids_default, "ids_no_special": ids_plain,
                          "tokens_added_front": added_front, "tokens_added_back": added_back,
                          "verification": "tok(probe) vs tok(probe, add_special_tokens=False) on the downloaded snapshot"},
            "generation_config": {k: gc.get(k) for k in ("bos_token_id", "eos_token_id", "pad_token_id", "do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens") if k in gc},
            "config_token_ids": {k: cfg.get(k) for k in ("bos_token_id", "eos_token_id", "pad_token_id")},
            "chat_template": chat_template, "chat_template_source": src, "default_system_prompt": sys_prompt,
            "chat_prefix_instruction": INSTRUCTION if chat_template else None, "rendered_chat_prefix": rendered,
            "rendered_chat_prefix_ids": rendered_ids, "rendered_chat_prefix_len": len(rendered_ids) if rendered_ids else None,
            "model_card": model_card_facts(d),
            "protocols": {"default": "tokenizer defaults, raw cloze" + (" (adds BOS)" if adds_bos else " (identical to nobos: tokenizer adds nothing)"),
                          "nobos": "no special tokens, raw cloze (paper protocol)" + ("" if adds_bos else " (= default)"),
                          "chat": ("chat template + opened assistant turn as prefix_ids, cloze inside the assistant turn" if (m.get("instruct") and chat_template) else "not applicable")},
        }
        with open(usage_path(key), "w") as f:
            json.dump(usage, f, indent=1, ensure_ascii=False)
        print(f"{key}: adds_bos={adds_bos} adds_eos={adds_eos} dtype={usage['recommended_dtype']} template={src} prefix_len={usage['rendered_chat_prefix_len']} "
              f"sys_prompt={sys_prompt!r}\n  rendered: {rendered!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
