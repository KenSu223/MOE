"""HF reference forward (device_map=auto with CPU/disk offload) vs engine on 5 prompts for a big model."""
import sys, json, time, os, gc
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, torch
from moetrace.models import MODELS
from moetrace.arch import snapshot_dir
from moetrace.protocol import cases_by_id, load_case_sets
from moetrace.engine import Engine, PrefillSpec
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
model = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
m = MODELS[model]; sets = load_case_sets(model)
ids = sets["paper"]["validation"][:n]
cases, _ = cases_by_id(model, ids)
cs = [cases[c] for c in ids]
eng = Engine(m["repo"])
res = eng.run([PrefillSpec(c.ids, c.true_id, c.foil_id) for c in cs], [], record_routing=False)
# full logits for the engine: recompute head on the prefill rows is internal; use recorded true/foil + top1
eng_delta = res.delta.copy(); eng_top1 = res.top1.copy(); eng_lt = res.logit_true_full.copy(); eng_lf = res.logit_foil_full.copy()
del eng, res; gc.collect(); torch.cuda.empty_cache()
log("engine done; loading HF with offload")
from transformers import AutoModelForCausalLM
t0 = time.time()
os.makedirs("/opt/dlami/nvme/offload", exist_ok=True)
hf = AutoModelForCausalLM.from_pretrained(snapshot_dir(m["repo"]), dtype=torch.bfloat16, device_map="auto",
                                          max_memory={0: "16GiB", "cpu": "28GiB"}, offload_folder="/opt/dlami/nvme/offload")
hf.eval(); log(f"HF loaded in {time.time()-t0:.0f}s; device map sample: {list(hf.hf_device_map.items())[:3]} ... {list(hf.hf_device_map.items())[-2:]}")
out = []
for c, dl, t1, lt, lf in zip(cs, eng_delta, eng_top1, eng_lt, eng_lf):
    t0 = time.time()
    with torch.no_grad():
        lg = hf(input_ids=torch.tensor([c.ids], device="cuda")).logits[0, -1].float().cpu().numpy()
    d_hf = float(lg[c.true_id] - lg[c.foil_id])
    out.append(dict(case_id=c.case_id, prompt=c.prompt, delta_engine=float(dl), delta_hf=d_hf, diff=float(dl - d_hf),
                    logit_true_engine=float(lt), logit_true_hf=float(lg[c.true_id]), logit_foil_engine=float(lf), logit_foil_hf=float(lg[c.foil_id]),
                    top1_engine=int(t1), top1_hf=int(lg.argmax()), forward_s=time.time() - t0))
    log(json.dumps(out[-1]))
summary = {"n": len(out), "max_abs_delta_diff": max(abs(o["diff"]) for o in out), "mean_abs_delta_diff": float(np.mean([abs(o["diff"]) for o in out])),
           "top1_agree": sum(o["top1_engine"] == o["top1_hf"] for o in out), "rows": out}
json.dump(summary, open(f"results/{model}/hf_reference_check.json", "w"), indent=1)
log("summary", json.dumps({k: v for k, v in summary.items() if k != "rows"}))
