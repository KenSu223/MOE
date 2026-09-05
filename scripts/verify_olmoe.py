"""Verification pilot on OLMoE-1B-7B-0125: HF reference vs streaming engine."""
import json, sys, time, os
sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace.verify import pick_cases, hf_reference, engine_check

REPO = "allenai/OLMoE-1B-7B-0125"
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    cases = pick_cases(REPO, n)
    log(f"picked {len(cases)} cases; first prompt: {cases[0].prompt!r} true={cases[0].true_str} foil={cases[0].foil_str} subj_pos={cases[0].subject_pos}")
    layers = [0, 4, 8, 12, 15]
    ref = hf_reference(REPO, cases, layers, n_patch_cases=10, n_expert_cases=4, log=log)
    rep = engine_check(REPO, cases, ref, layers, log=log)
    os.makedirs("results", exist_ok=True)
    with open("results/verify_olmoe.json", "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in rep.items():
        if k not in ("layer_times_first_pass", "mean_layer_rescue_curve"):
            log(f"{k}: {v}")
    log("curve:", rep["mean_layer_rescue_curve"])
