"""Noise-floor diagnostic: engine(bf16) vs HF(bf16, eager) vs HF(bf16, sdpa) vs HF(fp32 on CPU) on OLMoE."""
import sys, time, json, os
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, torch
from moetrace.verify import pick_cases
from moetrace.arch import snapshot_dir
from moetrace.engine import Engine, PrefillSpec, SpawnSpec
from moetrace.noise import noise_draw
from transformers import AutoModelForCausalLM
REPO = "allenai/OLMoE-1B-7B-0125"
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
cases = pick_cases(REPO, n)
snap = snapshot_dir(REPO)

def run_hf(model, dev, dtype, sigma, hooks_layer=None):
    out_c, out_n, out_p = [], [], []
    emb = model.get_input_embeddings()
    for c in cases:
        ids = torch.tensor([c.ids], device=dev)
        e = emb(ids)
        with torch.no_grad():
            lc = model(inputs_embeds=e).logits[0, -1].float().cpu().numpy()
        eps = noise_draw(c.case_id, len(c.subject_pos), e.shape[-1], sigma).to(dev)
        en = e.clone(); pos = torch.tensor(c.subject_pos, device=dev)
        en[0, pos] = (en[0, pos].float() + eps).to(e.dtype)
        with torch.no_grad():
            ln = model(inputs_embeds=en).logits[0, -1].float().cpu().numpy()
        out_c.append(lc[c.true_id]-lc[c.foil_id]); out_n.append(ln[c.true_id]-ln[c.foil_id])
        if hooks_layer is not None:
            store = {}
            def rec(m, a, o): store['o'] = (o[0] if isinstance(o, tuple) else o)[0, -1].detach().clone()
            def rep(m, a, o):
                oo = (o[0] if isinstance(o, tuple) else o).clone(); oo[0, -1] = store['o']; return oo
            mod = model.model.layers[hooks_layer].mlp
            h = mod.register_forward_hook(rec)
            with torch.no_grad(): model(inputs_embeds=e)
            h.remove(); h = mod.register_forward_hook(rep)
            with torch.no_grad(): lp = model(inputs_embeds=en).logits[0, -1].float().cpu().numpy()
            h.remove(); out_p.append(lp[c.true_id]-lp[c.foil_id])
    return np.array(out_c), np.array(out_n), np.array(out_p)

PL = 12  # patch layer for the layer-patch comparison
res = {}
# --- HF bf16 eager and sdpa on GPU
for attn in ["eager", "sdpa"]:
    m = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn).eval()
    sigma = 3.0 * m.get_input_embeddings().weight.float().std().item()
    res[f"hf_bf16_{attn}"] = run_hf(m, "cuda", torch.bfloat16, sigma, hooks_layer=PL)
    log(f"HF bf16 {attn} done"); del m; torch.cuda.empty_cache()
# --- engine bf16
eng = Engine(REPO); sigma = 3.0 * eng.embed_std
pre = [PrefillSpec(c.ids, c.true_id, c.foil_id) for c in cases] + [PrefillSpec(c.ids, c.true_id, c.foil_id, c.subject_pos, noise_draw(c.case_id, len(c.subject_pos), 2048, sigma)) for c in cases]
sp = [SpawnSpec(PL, n+i, i, "layer") for i in range(n)]
r = eng.run(pre, sp, record_routing=False)
res["engine_bf16"] = (r.delta[:n], r.delta[n:], r.sp_delta)
del eng; torch.cuda.empty_cache(); log("engine done")
# --- HF fp32 on CPU (ground truth)
torch.set_num_threads(16)
m = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32, device_map="cpu", attn_implementation="eager").eval()
sigma32 = 3.0 * m.get_input_embeddings().weight.float().std().item()
log(f"fp32 model loaded; sigma fp32 {sigma32:.6f} vs bf16-derived {sigma:.6f}")
t0 = time.time(); res["hf_fp32_cpu"] = run_hf(m, "cpu", torch.float32, sigma, hooks_layer=PL); log(f"fp32 forwards done in {time.time()-t0:.0f}s")
ref = res["hf_fp32_cpu"]
summary = {}
for k, (dc, dn, dp) in res.items():
    summary[k] = {
        "clean_vs_fp32_mean_abs": float(np.abs(dc-ref[0]).mean()), "clean_vs_fp32_max": float(np.abs(dc-ref[0]).max()),
        "noised_vs_fp32_mean_abs": float(np.abs(dn-ref[1]).mean()), "noised_vs_fp32_max": float(np.abs(dn-ref[1]).max()),
        f"patchL{PL}_vs_fp32_mean_abs": float(np.abs(dp-ref[2]).mean()), f"patchL{PL}_vs_fp32_max": float(np.abs(dp-ref[2]).max()),
        f"rescueL{PL}_vs_fp32_mean_abs": float(np.abs((dp-dn)-(ref[2]-ref[1])).mean()), f"rescueL{PL}_vs_fp32_max": float(np.abs((dp-dn)-(ref[2]-ref[1])).max()),
    }
e, h = res["engine_bf16"], res["hf_bf16_eager"]
summary["engine_vs_hf_eager"] = {"clean_max": float(np.abs(e[0]-h[0]).max()), "noised_max": float(np.abs(e[1]-h[1]).max()), "patch_max": float(np.abs(e[2]-h[2]).max())}
h2 = res["hf_bf16_sdpa"]
summary["hf_eager_vs_hf_sdpa"] = {"clean_max": float(np.abs(h2[0]-h[0]).max()), "noised_max": float(np.abs(h2[1]-h[1]).max()), "patch_max": float(np.abs(h2[2]-h[2]).max())}
summary["deltas_fp32"] = {"clean": ref[0].round(3).tolist(), "noised": ref[1].round(3).tolist(), "patch": ref[2].round(3).tolist()}
summary["deltas_engine"] = {"clean": e[0].round(3).tolist(), "noised": e[1].round(3).tolist(), "patch": e[2].round(3).tolist()}
summary["deltas_hf_eager"] = {"clean": h[0].round(3).tolist(), "noised": h[1].round(3).tolist(), "patch": h[2].round(3).tolist()}
json.dump(summary, open("results/verify_olmoe_fp32.json", "w"), indent=1)
for k, v in summary.items():
    if not k.startswith("deltas"): log(k, json.dumps(v))
