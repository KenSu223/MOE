"""Phase 6 (p6-analysis): detached analysis refresher. The tier-2 / R-causal chain only refreshes the analyses between its GPU
jobs, and its low-priority R-causal jobs can wait for hours in the queue, so this CPU-only loop re-runs scripts/ext14_all.py
(flock logs/ext14_analysis.lock) whenever the set of '[p6-data] BASE READY' lines, the registry or the R-causal / tier-2
summaries change. Stop: touch logs/ext14_analyze_chain.stop.

Usage: nohup setsid python scripts/ext14_analyze_chain.py > logs/ext14_analyze_chain.log 2>&1 < /dev/null &
"""
import os, subprocess, sys, time

ROOT = "/home/ubuntu/MOE"
PY = os.path.join(ROOT, ".venv/bin/python")
STOP = os.path.join(ROOT, "logs", "ext14_analyze_chain.stop")
WATCH = ("data/dial/p6_conditions.json", "results/p6_causal_summary.json", "results/ext14_t2_summary.json")


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, flush=True)


def state():
    n = sum(1 for ln in open(os.path.join(ROOT, "logs", "PROGRESS.md"), errors="replace") if "[p6-data] BASE READY" in ln)
    m = tuple(int(os.stat(os.path.join(ROOT, f)).st_mtime) if os.path.exists(os.path.join(ROOT, f)) else 0 for f in WATCH)
    return (n,) + m


def main():
    last = None
    log(f"analysis refresher start (pid {os.getpid()})")
    while not os.path.exists(STOP):
        s = state()
        if s != last:
            t0 = time.time()
            with open(os.path.join(ROOT, "logs", "ext14_analysis.log"), "a") as lf:
                lf.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} scripts/ext14_all.py (refresher)\n")
                lf.flush()
                rc = subprocess.call([PY, "-W", "ignore", "scripts/ext14_all.py"], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
            log(f"ext14_all.py rc={rc} ({time.time() - t0:.0f}s) state {s}")
            last = s
        time.sleep(180)
    log("stop file; exit")


if __name__ == "__main__":
    main()
