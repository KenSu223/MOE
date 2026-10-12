"""Chain health watcher for unattended GPU runs (written for Phase 5; reusable). Prints one line per event, silent otherwise,
so it can run under Claude Code's Monitor tool (each stdout line = one notification):
    python3 -u scripts/chain_watch.py [--chains REGEX] [--progress-tag REGEX] [--state PATH]
Defaults are the Phase-5 chain scripts and PROGRESS tags; pass --chains / --progress-tag for a new phase.
Events: failed GPU jobs, errors in analysis / chain logs, chain processes exiting, GPU idle while chains are alive,
long-running jobs, stage-completion lines from the agents in logs/PROGRESS.md.
State (file offsets, known chain PIDs) persists in p5_watch_state.json so a re-armed watcher does not repeat events."""
import glob
import json
import os
import re
import subprocess
import time

R = "/home/ubuntu/MOE"
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--chains", default=None)
_ap.add_argument("--progress-tag", default=None)
_ap.add_argument("--state", default="/tmp/chain_watch_state.json")
_A, _ = _ap.parse_known_args()
ST = _A.state
CHAIN_RE = None  # set below
CHAIN_RE_DEFAULT = r"ext13_p5_chain_base.sh|ext13_p5_fix_cpnat|ext13ab_chain.sh|ext13ci_chain.py|ext13ko_chain.sh|ext13ko_nomore_watch"
ERR_RE = re.compile(r"Traceback|Error|rc=[1-9]|Killed|FAILED|abort")
ERR_FILES = ["logs/ext13_p5_analyze_*.log", "logs/ext13ab_analyze.log", "logs/ext13ko_analyze.log", "logs/ext13wr_analyze.log",
             "logs/ext13ab_chain.log", "logs/ext13ko_chain_*.log", "logs/ext13_p5_chain.log",
             "logs/ext13_p5_fix_cpnat_mixtral.out", "logs/ext13ci_analyze*.log", "logs/ext13ci_chain_*.log"]
PROG_RE = re.compile(_A.progress_tag or r"\[p5-(data|addback|circuit|expert|atp)\]")
IDLE_MIN, LONG_MIN = 20, 50
CHAIN_RE = _A.chains or CHAIN_RE_DEFAULT


def emit(s):
    print(time.strftime("%H:%MZ", time.gmtime()), s[:300], flush=True)


def load():
    try:
        return json.load(open(ST))
    except Exception:
        return {"off": {}, "pids": {}, "idle_flag": False, "long_flag": ""}


def new_lines(state, path):
    try:
        size = os.path.getsize(path)
    except OSError:
        return []
    off = state["off"].get(path)
    if off is None or off > size:  # first sight: start at the end (only new lines count)
        state["off"][path] = size
        return []
    if size == off:
        return []
    with open(path, "rb") as f:
        f.seek(off)
        data = f.read()
    state["off"][path] = size
    return data.decode("utf-8", "replace").splitlines()


def chains():
    """Top-level chain processes only (a chain's own subshells / command substitutions are skipped)."""
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        parts = ln.split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, cmd = parts
        if re.search(CHAIN_RE, cmd) and "pgrep" not in cmd and "p5_watch" not in cmd and not cmd.startswith("/bin/bash -c source"):
            rows.append((pid, ppid, cmd))
    pids = {r[0] for r in rows}
    return {pid: " ".join(cmd.split()[-3:]) for pid, ppid, cmd in rows if ppid not in pids}


def queue_state():
    """(running job name or None, minutes since its START or since the last END)."""
    try:
        lines = open(f"{R}/logs/gpu_queue.log").read().splitlines()[-40:]
    except OSError:
        return None, 0
    started = {}
    last_end = None
    for ln in lines:
        m = re.match(r"(\S+) (START|END)\s+(\S+)", ln)
        if not m:
            continue
        t = time.mktime(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
        if m.group(2) == "START":
            started[m.group(3)] = t
        else:
            started.pop(m.group(3), None)
            last_end = t
    now = time.time()
    if started:
        name, t = max(started.items(), key=lambda kv: kv[1])
        return name, (now - t) / 60
    return None, (now - last_end) / 60 if last_end else 0


def main():
    state = load()
    first = not state["pids"]
    while True:
        # failed GPU jobs
        for ln in new_lines(state, f"{R}/logs/gpu_queue.log"):
            m = re.search(r"END\s+(\S+) rc=(\d+)", ln)
            if m and m.group(2) != "0":
                emit(f"GPU JOB FAILED {m.group(1)} rc={m.group(2)} (log logs/{m.group(1)}.log)")
        # errors in analysis / chain logs
        for pat in ERR_FILES:
            for p in glob.glob(f"{R}/{pat}"):
                for ln in new_lines(state, p):
                    if ERR_RE.search(ln):
                        emit(f"ERROR in {os.path.relpath(p, R)}: {ln.strip()}")
        # stage completions from the agents' chains
        for ln in new_lines(state, f"{R}/logs/PROGRESS.md"):
            if PROG_RE.search(ln) and re.search(r"BASE READY|DONE|COMPLETE|done|complete|FAIL|fail|error", ln):
                emit("PROGRESS " + ln.strip("- ").strip())
        # chain processes
        cur = chains()
        if not first:
            for pid, cmd in state["pids"].items():
                if pid not in cur:
                    emit(f"CHAIN EXITED pid {pid}: {cmd} (still alive: {len(cur)})")
        state["pids"] = cur
        first = False
        # idle / long-running
        job, mins = queue_state()
        if job is None and cur and mins >= IDLE_MIN:
            if not state["idle_flag"]:
                emit(f"GPU IDLE {mins:.0f} min while {len(cur)} chain processes are alive")
                state["idle_flag"] = True
        else:
            state["idle_flag"] = False
        if job and mins >= LONG_MIN and state["long_flag"] != job:
            emit(f"LONG JOB {job} running {mins:.0f} min")
            state["long_flag"] = job
        if not cur:
            emit("ALL WATCHED CHAINS EXITED")
            json.dump(state, open(ST, "w"))
            return
        json.dump(state, open(ST, "w"))
        time.sleep(120)


if __name__ == "__main__":
    main()
