"""Phase-6 Stage-2 chain (agent p6-analysis): Qwen2-57B-A14B shared vs routed experts (prediction D6).

Waits for '[p6-engine] MODEL READY qwen2_57b' and the '[p6-data] BASE READY <cell>/qwen2_57b' lines; for every Set-A cell
(knowledge cells first) runs scripts/ext14_t2_shared.py <cell> through the GPU queue (one job at a time, one pass of
~3 x 28 x n spawn rows), then refreshes scripts/ext14_all.py and writes '<UTC> [p6-analysis] SHARED done <cell>/qwen2_57b ...'.
Exits when every Set-A cell registered for qwen2_57b has its shared_summary.json (or after --hours). Stop: touch
logs/ext14_t2_s2_chain.stop.

Usage: nohup setsid python scripts/ext14_t2_s2_chain.py > logs/ext14_t2_s2_chain.log 2>&1 < /dev/null &
"""
import argparse, json, os, re, subprocess, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
from moetrace import ext14_analysis as XA
from moetrace.models import RESULTS

ROOT = "/home/ubuntu/MOE"
PY = os.path.join(ROOT, ".venv/bin/python")
PROTO = "qwen2_57b"
ORDER = ("l_ant", "l_prop", "l_cf", "e3_e2a", "e3_a2e", "e1_a2e", "e1_e2a", "e1p_a2e", "e1p_e2a", "e2_a2e", "e2_e2a", "l_bind", "l_ind", "l_ioi")
STOP = os.path.join(ROOT, "logs", "ext14_t2_s2_chain.stop")


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, flush=True)


def engine_ready():
    p = os.path.join(ROOT, "logs", "PROGRESS.md")
    return any(re.match(r"^\s*(-\s*)?\S+\s+\[p6-engine\]\s+MODEL READY qwen2_57b\b", ln) for ln in open(p, errors="replace"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=120)
    ap.add_argument("--poll", type=int, default=180)
    a = ap.parse_args()
    deadline = time.time() + a.hours * 3600
    fails = {}
    log(f"stage-2 shared chain start (pid {os.getpid()})")
    last = 0
    while time.time() < deadline and not os.path.exists(STOP):
        try:
            if engine_ready():
                lines = XA.ready_lines()
                reg = XA.registry()
                todo = [c for c in ORDER if (c, PROTO) in lines and PROTO in reg.get(c, {})
                        and not os.path.exists(os.path.join(RESULTS, f"p6_{c}_{PROTO}_t2", "shared_summary.json")) and fails.get(c, 0) < 2]
                if todo:
                    c = todo[0]
                    job = f"p6an-shared-{c}-{PROTO}"
                    log(f"submit {job}")
                    with open(os.path.join(ROOT, "logs", f"ext14_t2_{job}.log"), "a") as lf:
                        rc = subprocess.call(["bash", "scripts/gpu_queue.sh", job, "--", "python", "scripts/ext14_t2_shared.py", c], cwd=ROOT,
                                             stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "GPU_QUEUE_WAIT_S": "86400"})
                    log(f"{job} rc={rc}")
                    if rc != 0:
                        fails[c] = fails.get(c, 0) + 1
                        with open(os.path.join(ROOT, "logs", "PROGRESS.md"), "a") as f:
                            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [p6-analysis] SHARED {c}/{PROTO} job FAILED rc={rc} (attempt {fails[c]})\n")
                        continue
                    s = json.load(open(os.path.join(RESULTS, f"p6_{c}_{PROTO}_t2", "shared_summary.json")))
                    with open(os.path.join(ROOT, "logs", "ext14_analysis.log"), "a") as lf:
                        subprocess.call([PY, "-W", "ignore", "scripts/ext14_all.py"], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
                    with open(os.path.join(ROOT, "logs", "PROGRESS.md"), "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [p6-analysis] SHARED done {c}/{PROTO}: shared share of the MoE rescue "
                                f"{XA.fr(s['shared_share_of_moe_auc'])}, shared / whole-MoE AUC+ {XA.fr(s['shared_over_layer_auc'])}\n")
                    continue
                regc = [c for c in ORDER if PROTO in reg.get(c, {}) and not reg[c][PROTO].get("behaviour_only")]
                if regc and all(os.path.exists(os.path.join(RESULTS, f"p6_{c}_{PROTO}_t2", "shared_summary.json")) or fails.get(c, 0) >= 2 for c in regc) \
                        and len(regc) >= len(ORDER) - 2:
                    log("all registered Set-A cells done")
                    break
            if time.time() - last > 1800:
                log(f"waiting (engine ready: {engine_ready()})")
                last = time.time()
        except Exception as ex:  # noqa: BLE001
            import traceback
            log(f"iteration error {ex!r}\n{traceback.format_exc()}")
        time.sleep(a.poll)
    log("exit")


if __name__ == "__main__":
    main()
