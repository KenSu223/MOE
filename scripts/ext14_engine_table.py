"""ext14 (p6-engine): markdown table of the checkpoint verifications (results/verify_ext14_<key>[_perlayer].json).
Usage: python scripts/ext14_engine_table.py [keys...]"""
import json, os, sys

R = "/home/ubuntu/MOE/results"
KEYS = sys.argv[1:] or ["mistral7b", "qwen2_7b", "qwen3_1p7b", "qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen2_57b"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


print("| model | full mode vs transformers (spawns; rescue r; max / mean-of-groups |dDelta|; prefill max) | per-layer update relerr (max; routing) | engine identities | sanity (answer ranks) |")
print("|---|---|---|---|---|")
for k in KEYS:
    f, p = load(f"{R}/verify_ext14_{k}.json"), load(f"{R}/verify_ext14_{k}_perlayer.json")
    if f is None and p is None:
        print(f"| {k} | pending | | | |")
        continue
    if f is not None:
        g = f["spawn_delta_vs_hf"]
        mg = sum(v["meanabs"] for v in g.values()) / len(g)
        pm = max(v["maxabs"] for v in f["prefill_delta_vs_hf"].values())
        full = (f"{f['n_spawns']}; r {f['spawn_rescue_corr_all']:.4f}; {f['spawn_delta_maxabs_all']:.3g} / {mg:.3f}; {pm:.3g}"
                + (" (HF offloaded to CPU RAM, light plan)" if f.get("hf_offload") else ""))
    else:
        full = "n/a (per-layer only)"
    if p is not None:
        pl = p["per_layer"]
        lay = ", ".join(f"L{l} {v['update_relerr']['maxabs']:.2g}" + (f" ({v['routing_set_agreement']:.2f})" if v["routing_set_agreement"] is not None else "")
                        for l, v in pl["layers"].items())
        ide = pl["identities"]
        idt = (f"union/set(all) vs layer {max(ide['union_vs_layer']['maxabs'], ide['setall_vs_layer']['maxabs']):.2g}; neurons(all) vs expert "
               f"{ide['neurons_all_vs_expert']['maxabs']:.2g}; linearity {ide['neurons_linearity_relerr_max']:.1e}"
               + (f"; shared expert vs set {ide['shared_expert_vs_set']['maxabs']:.2g}, shared neurons(all) {ide['shared_neurons_all_vs_expert']['maxabs']:.2g}"
                  if "shared_expert_vs_set" in ide else ""))
        san = " ".join(f"{x['answer'].strip()}#{x['rank']}" for x in pl["behaviour"]["sanity"])
    else:
        lay = idt = san = "pending"
    print(f"| {k} | {full} | {lay} | {idt} | {san} |")
