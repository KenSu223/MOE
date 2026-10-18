"""Phase-6 tier-2 detached chain (agent p6-analysis). One process for all tier-2 cells; ONE GPU job in the queue at a time.

Waits for the exact PROGRESS lines '<UTC> [p6-data] BASE READY <cell>/<proto>' of the tier-2 cells
(scripts/ext14_t2_common.T2_CELLS: Stage 1 e1_a2e / e1_e2a on qwen3 and mixtral_bos; Stage 2 e1_a2e / e3_a2e on qwen2_57b,
which also needs '[p6-engine] MODEL READY qwen2_57b'), then runs per cell, in this order, skipping stages with a
results/p6_<cell>_<proto>_t2/.done_<stage> marker:
  prepare (CPU)  tier-2 mirror dir;
  expert  (GPU)  scripts/ext7_wino_expert.py --out <t2 run> (all layers, equal-norm pairs at the MoE argmax layer);
  shared  (GPU)  Qwen2-57B only: scripts/ext14_t2_shared.py --singles (shared expert 64 appended to the expert table);
  select  (CPU)  scripts/ext14_t2_select.py;
  ab      (GPU)  scripts/ext14_t2_run.py ab ... until run_meta 'complete' (budget x 3/4 after a failed job);
  ci1     (GPU)  scripts/ext14_t2_run.py ci1 ... (single heads, every layer) until complete;
  ci2     (GPU)  scripts/ext14_t2_run.py ci2 ... (mixed head + expert greedy) until complete;
  ko      (GPU)  scripts/ext14_t2_run.py ko ... --with-wr (knockout + writer pass) until complete;
and after every stage the CPU analyses (scripts/ext14_t2_analyze.py, scripts/ext14_all.py; flock logs/ext14_analysis.lock),
plus a PROGRESS line '<UTC> [p6-analysis] T2 <stage> done <cell>/<proto> ...'. The cross-cell analysis
(scripts/ext14_all.py) is also refreshed whenever new BASE READY lines appear, so the ext14 tables / figures / sections fill in
while p6-data works. R-causal (scripts/ext14_causal_run.py, if present) runs after '[p6-engine] HEADMASK READY' and the
BASE READY lines of its cells. Resumable: all decisions are re-derived from disk; failures and add-back budgets in
logs/ext14_t2_chain_state.json. Stop: touch logs/ext14_t2_chain.stop.

Usage: nohup setsid python scripts/ext14_t2_chain.py > logs/ext14_t2_chain.log 2>&1 < /dev/null &
"""
import argparse, json, os, re, subprocess, sys, time, traceback
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
import ext14_t2_common as T2
from moetrace import ext14_analysis as XA

ROOT = "/home/ubuntu/MOE"
PY = os.path.join(ROOT, ".venv/bin/python")
STATE = os.path.join(ROOT, "logs", "ext14_t2_chain_state.json")
STOP = os.path.join(ROOT, "logs", "ext14_t2_chain.stop")
ORDER = [("e1_a2e", "qwen3"), ("e1_a2e", "mixtral_bos"), ("e1_e2a", "qwen3"), ("e1_e2a", "mixtral_bos"),
         ("e1_a2e", "qwen2_57b"), ("e3_a2e", "qwen2_57b")]
AB = {"qwen3": dict(model="qwen3", budget=45000, passes=18), "mixtral_bos": dict(model="mixtral", budget=50000, passes=12),
      "qwen2_57b": dict(model="qwen2_57b", budget=40000, passes=12)}
EXPERT_SPAWN = {"qwen3": 60000, "mixtral_bos": 40000, "qwen2_57b": 40000}
S1_PASSES = {"qwen3": 6, "mixtral_bos": 4, "qwen2_57b": 4}
S2_PASSES = {"qwen3": 12, "mixtral_bos": 10, "qwen2_57b": 10}
MAX_FAIL = 3


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, flush=True)


def progress(msg):
    with open(os.path.join(ROOT, "logs", "PROGRESS.md"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [p6-analysis] {msg}\n")


def gpu(job, cmd):
    log(f"submit {job}: {' '.join(cmd)}")
    t0 = time.time()
    with open(os.path.join(ROOT, "logs", f"ext14_t2_{job}.log"), "a") as lf:
        rc = subprocess.call(["bash", "scripts/gpu_queue.sh", job, "--"] + cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                             env={**os.environ, "GPU_QUEUE_WAIT_S": "86400", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    log(f"{job} rc={rc} ({time.time() - t0:.0f}s incl. queue wait)")
    return rc


def cpu(args, logname="ext14_analysis.log"):
    with open(os.path.join(ROOT, "logs", logname), "a") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(args)}\n")
        lf.flush()
        rc = subprocess.call([PY, "-W", "ignore"] + args, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    log(f"cpu {' '.join(args)} rc={rc}")
    return rc


def analyse(cell=None, proto=None):
    if cell:
        cpu(["scripts/ext14_t2_analyze.py", "--cells", f"{cell}/{proto}"], "ext14_analysis_t2.log")
    cpu(["scripts/ext14_all.py"], "ext14_analysis.log")


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"fail": {}, "budget": {}, "analysed_lines": 0}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=1)


def engine_line(what):
    p = os.path.join(ROOT, "logs", "PROGRESS.md")
    pat = re.compile(r"^\s*(-\s*)?\S+\s+\[p6-engine\]\s+" + re.escape(what) + r"\b")
    return any(pat.match(ln) for ln in open(p, errors="replace"))


def ab_complete(cell, proto):
    p = os.path.join(T2.t2_dir(cell, proto), "ab", "run_meta.json")
    return os.path.exists(p) and bool(json.load(open(p)).get("complete"))


def ci1_complete(cell, proto):
    p = os.path.join(T2.t2_dir(cell, proto), "ci", "heads_state.json")
    if not os.path.exists(p):
        return False
    s = json.load(open(p))
    return len(s["done"]) == len(s["chunks"]) and len(s["chunks"]) > 0


def ci2_complete(cell, proto):
    p = os.path.join(T2.t2_dir(cell, proto), "ci2", "run_meta.json")
    return os.path.exists(p) and bool(json.load(open(p)).get("complete"))


def ko_complete(cell, proto):
    d = T2.t2_dir(cell, proto)
    p = os.path.join(d, "ko", "run_meta.json")
    ok = os.path.exists(p) and bool(((json.load(open(p)).get("phases") or {}).get("main") or {}).get("complete"))
    return ok and os.path.exists(os.path.join(d, "wr", "vec_rows.parquet"))


def step_cell(cell, proto, st):
    """Run the next pending stage of one cell. Returns True if a job (GPU or CPU) ran, False if nothing to do."""
    tag = f"{cell}/{proto}"
    m = T2.T2_MODEL[proto]
    fails = st["fail"]

    def failed(stage):
        return fails.get(f"{tag}:{stage}", 0) >= MAX_FAIL

    def fail(stage, rc):
        k = f"{tag}:{stage}"
        fails[k] = fails.get(k, 0) + 1
        save_state(st)
        progress(f"T2 {stage} {tag} job FAILED rc={rc} (attempt {fails[k]}/{MAX_FAIL}); see logs/ext14_t2_chain.log")

    e = XA.entry_of(cell, proto)
    if not done(cell, proto, "prepare"):
        T2.prepare(cell, proto)
        T2.mark(cell, proto, "prepare")
        log(f"{tag}: mirror prepared")
        return True
    if not done(cell, proto, "expert"):
        if failed("expert"):
            return False
        cmd = ["python", "scripts/ext7_wino_expert.py", "--model", m, "--pairs", XA._abs(e["pair_file"]), "--case-sets", XA._abs(e["case_sets"]),
               "--out", T2.t2_run(cell, proto), "--agent", "p6-analysis", "--families", "main", "--layers", "all", "--pairs-layers", "auto",
               "--max-spawn", str(EXPERT_SPAWN[proto])]
        rc = gpu(f"p6an-expert-{cell}-{proto}", cmd)
        if rc == 0:
            T2.mark(cell, proto, "expert")
        else:
            fail("expert", rc)
        return True
    if T2.has_shared(m) and not done(cell, proto, "shared"):
        if failed("shared"):
            return False
        rc = gpu(f"p6an-shared-{cell}-{proto}", ["python", "scripts/ext14_t2_shared.py", cell, "--proto", proto, "--singles"])
        if rc == 0:
            T2.mark(cell, proto, "shared")
        else:
            fail("shared", rc)
        return True
    if not done(cell, proto, "select"):
        rc = cpu(["scripts/ext14_t2_select.py", cell, proto], "ext14_analysis_t2.log")
        if rc == 0:
            T2.mark(cell, proto, "select")
            sel = (T2.selection(cell, proto) or {}).get("selected_expert") or {}
            progress(f"T2 expert done {tag}: selected {sel.get('name')} (pattern {sel.get('pattern')}, Spec {fmt3(sel.get('spec'))}); "
                     f"pop top-3 {[f'L{l}E{x:03d}' for l, x, _ in ((T2.selection(cell, proto) or {}).get('pop_top10') or [])[:3]]}")
            analyse(cell, proto)
        else:
            fail("select", rc)
        return True
    stages = [("ab", ab_complete), ("ci1", ci1_complete), ("ci2", ci2_complete), ("ko", ko_complete)]
    for stage, comp in stages:
        if done(cell, proto, stage):
            continue
        if comp(cell, proto):
            T2.mark(cell, proto, stage)
            analyse(cell, proto)
            progress(f"T2 {stage} done {tag}" + summary_line(cell, proto, stage))
            return True
        if failed(stage):
            continue  # later stages do not depend on the add-back; circuit / knockout continue
        if stage == "ab":
            b = st["budget"].get(f"{tag}:ab", AB[proto]["budget"])
            cmd = ["python", "scripts/ext14_t2_run.py", "ab", AB[proto]["model"], "--tasks", f"{cell}_{proto}", "--max-passes", str(AB[proto]["passes"]),
                   "--budget", str(b), "--max-tasks", "1"]
        elif stage == "ci1":
            cmd = ["python", "scripts/ext14_t2_run.py", "ci1", cell, proto, "--max-passes", str(S1_PASSES[proto])]
        elif stage == "ci2":
            if not ci1_complete(cell, proto):
                continue
            cmd = ["python", "scripts/ext14_t2_run.py", "ci2", proto, "--group", cell, "--conds", cell, "--max-passes", str(S2_PASSES[proto])]
        else:
            cmd = ["python", "scripts/ext14_t2_run.py", "ko", proto, "--conds", cell, "--with-wr", "--time-budget", "1320"]
        rc = gpu(f"p6an-{stage}-{cell}-{proto}", cmd)
        if rc != 0:
            if stage == "ab":
                st["budget"][f"{tag}:ab"] = max(15000, int(st["budget"].get(f"{tag}:ab", AB[proto]["budget"]) * 3 / 4))
            fail(stage, rc)
        elif comp(cell, proto):
            T2.mark(cell, proto, stage)
            analyse(cell, proto)
            progress(f"T2 {stage} done {tag}" + summary_line(cell, proto, stage))
        return True
    return False


def done(cell, proto, stage):
    return T2.done(cell, proto, stage)


def fmt3(x):
    try:
        return f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]"
    except Exception:  # noqa: BLE001
        return "n/a"


def summary_line(cell, proto, stage):
    p = os.path.join(T2.t2_dir(cell, proto), "t2_summary.json")
    if not os.path.exists(p):
        return ""
    try:
        s = json.load(open(p))
        if stage == "ab":
            a = s.get("addback") or {}
            return (f": all-MoE ceiling {fmt3(a.get('ceiling'))}; k80 greedy / oracle / random "
                    f"{'/'.join(str((a.get(o) or {}).get('k80', ['?'])[0]) for o in ('greedy', 'oracle', 'rand'))}")
        if stage == "ci2":
            c = (s.get("circuit") or {}).get("step2") or {}
            r = c.get("r20") or {}
            return f": mixed greedy r(20) {r.get('ratio', float('nan')):.2f}, heads in first 10 {c.get('heads_first10', float('nan')):.1f}"
        if stage == "ci1":
            c = (s.get("circuit") or {}).get("step1") or {}
            return ": top heads " + ", ".join(f"{h['head']} {h['rescue_over_drop']['ratio']:+.3f}" for h in (c.get("top") or [])[:3])
        if stage == "ko":
            k = s.get("knockout") or {}
            sel = ((s.get("selection") or {}).get("selected_expert") or {}).get("name")
            f = (k.get("F") or {}).get(sel)
            t10 = ((k.get("sets") or {}).get("k10") or {})
            return f": knockout F selected {sel} {fmt3(f)}; top-10 {t10.get('pop')} vs random {t10.get('rand_mean')}"
    except Exception:  # noqa: BLE001
        return ""
    return ""


def causal_step(st):
    """R-causal (scripts/ext14_causal_run.py --next): runs one pending causal job if the script exists and its inputs are ready."""
    sc = os.path.join(ROOT, "scripts", "ext14_causal_run.py")
    if not os.path.exists(sc) or not engine_line("HEADMASK READY"):
        return False
    if st["fail"].get("causal", 0) >= MAX_FAIL:
        return False
    r = subprocess.run([PY, "-W", "ignore", sc, "--plan"], cwd=ROOT, capture_output=True, text=True)
    nxt = [ln for ln in r.stdout.splitlines() if ln.startswith("NEXT ")]
    if not nxt:
        return False
    args = nxt[0][5:].split()
    rc = gpu(f"p6an-causal-{args[0]}-{args[1]}", ["python", "scripts/ext14_causal_run.py"] + args)
    if rc != 0:
        st["fail"]["causal"] = st["fail"].get("causal", 0) + 1
        save_state(st)
        progress(f"R-causal job {' '.join(args)} FAILED rc={rc} (attempt {st['fail']['causal']})")
    else:
        cpu([sc.replace("ext14_causal_run.py", "ext14_causal_analyze.py")], "ext14_analysis_causal.log") if os.path.exists(
            sc.replace("ext14_causal_run.py", "ext14_causal_analyze.py")) else None
        analyse()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--hours", type=float, default=96)
    ap.add_argument("--no-t2", action="store_true", help="scope cut 2026-10-17: no tier-2 stages; only R-causal and the analysis refresh")
    args = ap.parse_args()
    deadline = time.time() + args.hours * 3600
    st = load_state()
    log(f"chain start (pid {os.getpid()}); order {ORDER}")
    last_wait = 0
    while time.time() < deadline:
        if os.path.exists(STOP):
            log("stop file")
            break
        try:
            lines = XA.ready_lines()
            # cross-cell refresh on new BASE READY lines
            if len(lines) != st.get("analysed_lines", 0):
                st["analysed_lines"] = len(lines)
                save_state(st)
                analyse()
            did = False
            for cell, proto in ([] if args.no_t2 else ORDER):
                if (cell, proto) not in lines:
                    continue
                if proto == "qwen2_57b" and not engine_line("MODEL READY qwen2_57b"):
                    continue
                if XA.entry_of(cell, proto) is None:
                    continue
                if step_cell(cell, proto, st):
                    did = True
                    break
            if not did:
                did = causal_step(st)
            if did:
                continue
            pending = [] if args.no_t2 else [f"{c}/{p}" for c, p in ORDER if not all(T2.done(c, p, s) for s in ("prepare", "expert", "select", "ab", "ci1", "ci2", "ko"))]
            if time.time() - last_wait > 1800:
                log(f"waiting; BASE READY seen for {sorted(f'{c}/{p}' for c, p in lines)}; tier-2 cells not finished: {pending}")
                last_wait = time.time()
        except Exception as ex:  # noqa: BLE001
            log(f"chain iteration error: {ex!r}\n{traceback.format_exc()}")
            time.sleep(args.poll * 3)
        time.sleep(args.poll)
    log("chain exit")


if __name__ == "__main__":
    main()
