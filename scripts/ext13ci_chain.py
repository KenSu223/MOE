"""ext13ci detached chain (Phase 5, p5-circuit): one process per protocol (model). Waits for the exact PROGRESS lines
'[p5-data] BASE READY <cond>/<proto>', runs step 1 (scripts/ext13ci_heads_run.py) for every ready condition in plan order,
and step 2 (scripts/ext13ci_circuit_run.py) for a group of conditions (moetrace.ext13ci.GROUPS) once all its members have
step 1; groups larger than the per-pass budget are split into balanced subgroups. Every GPU job goes through
scripts/gpu_queue.sh and is capped in passes (Qwen3 step 1 6 passes, Mixtral 4; step 2 Qwen3 12, Mixtral 10) so the queue
interleaves with the other agents (jobs <= ~25 min). After each job: CPU analysis of the finished parts
(scripts/ext13ci_analyze.py) and a [p5-circuit] PROGRESS line. Resumable: all decisions are re-derived from disk
(heads_state.json, circuit_state.pkl, run_meta.json "complete"); chain state (failures, subgroup plan) in
results/p5_<proto>_ci/chain_state.json. Conditions listed in results/p5_<proto>_ci/skip.txt (one cond per line) are
skipped. Exits when every group of the protocol is complete (or failed twice).

Usage: nohup python scripts/ext13ci_chain.py <proto> [--poll 120] > logs/ext13ci_chain_<proto>.log 2>&1 &
"""
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
from moetrace import ext13ci as E13
from moetrace import ext10_circuit as C

ROOT = "/home/ubuntu/MOE"
S1_PASSES = {"qwen3": 6, "mixtral": 4}
S2_PASSES = {"qwen3": 12, "mixtral": 10}
CAP_ROWS = {"qwen3": 940, "mixtral": 670}  # rows per step-2 pass (adaptive rows ~ 96 / 128 per row + static)
MAX_FAIL = 2


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, flush=True)


def progress(msg):
    with open(os.path.join(ROOT, "logs", "PROGRESS.md"), "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [p5-circuit] {msg}\n")


def run(job, cmd):
    log(f"submit {job}: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.call(["bash", "scripts/gpu_queue.sh", job, "--"] + cmd, cwd=ROOT)
    log(f"{job} rc={rc} ({time.time() - t0:.0f}s incl. queue wait)")
    return rc


def cpu(cmd):
    log(f"cpu: {' '.join(cmd)}")
    rc = subprocess.call([os.path.join(ROOT, ".venv/bin/python")] + cmd, cwd=ROOT)
    log(f"cpu rc={rc}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto", choices=list(E13.PROTOS))
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()
    proto = args.proto
    model = E13.MODEL_OF[proto]
    base = os.path.join(E13.RESULTS, f"p5_{proto}_ci")
    os.makedirs(base, exist_ok=True)
    sp = os.path.join(base, "chain_state.json")
    cs = json.load(open(sp)) if os.path.exists(sp) else {"fail": {}, "subgroups": {}, "done_groups": []}

    def save():
        json.dump(cs, open(sp, "w"), indent=1)

    def skipped():
        p = os.path.join(base, "skip.txt")
        return {ln.strip() for ln in open(p) if ln.strip()} if os.path.exists(p) else set()

    log(f"chain {proto} started (pid {os.getpid()})")
    state = {"waited_note": 0, "errors": 0}
    while True:
        try:
            r = step(proto, model, cs, save, skipped, state, args.poll)
        except Exception as ex:  # malformed registry entry, missing file, ...: log, back off, retry
            import traceback
            state["errors"] += 1
            log(f"chain iteration error #{state['errors']}: {ex!r}\n{traceback.format_exc()}")
            if state["errors"] in (1, 10, 50):
                progress(f"chain {proto}: iteration error #{state['errors']} ({ex!r}); retrying every {args.poll * 5} s")
            time.sleep(args.poll * 5)
            continue
        if r == "exit":
            break


def step(proto, model, cs, save, skipped, state, poll):
    """One chain iteration: returns 'exit' when everything is done, else None (after running a job or sleeping)."""
    if True:
        regs = E13.load_conditions()
        ready = E13.base_ready_lines()
        skip = skipped()
        conds = [c for c in E13.COND_ORDER if c not in skip and not (regs.get((c, proto)) or {}).get("behaviour_only")]
        did = False
        # ---- step 1 for ready conditions, plan order
        for c in conds:
            if (c, proto) not in ready or (c, proto) not in regs or E13.step1_complete(c, proto):
                continue
            k = f"s1:{c}"
            if cs["fail"].get(k, 0) >= MAX_FAIL:
                continue
            rc = run(f"ext13ci-s1-{c}-{proto}", ["python", "scripts/ext13ci_heads_run.py", c, proto, "--max-passes", str(S1_PASSES[model])])
            if rc != 0:
                cs["fail"][k] = cs["fail"].get(k, 0) + 1
                save()
                progress(f"step 1 {c}/{proto} job FAILED rc={rc} (attempt {cs['fail'][k]}); see logs/ext13ci_chain_{proto}.log")
            elif E13.step1_complete(c, proto):
                meta = json.load(open(os.path.join(E13.step1_dir(c, proto), "run_meta.json")))
                gs = sum(n.get("pass_s") or 0 for n in meta.get("pass_notes", []))
                cpu(["scripts/ext13ci_analyze.py", "--step", "1", "--keys", E13.key_of(c, proto)])
                try:
                    s1 = json.load(open(E13.SUMMARY))["step1"][E13.key_of(c, proto)]
                    th = ", ".join(f"{h['head']} {h['rescue_over_drop']['ratio']:+.3f}" for h in s1["top"][:3])
                    extra = f"; top heads {th}; Z8 +{s1['z8']['n_pos']}/-{s1['z8']['n_neg']}"
                except Exception:
                    extra = ""
                progress(f"step 1 done {c}/{proto} ({meta.get('n_val_rows')} rows, {len(meta.get('pass_notes', []))} passes, "
                         f"{gs / 60:.1f} GPU-min){extra}")
            did = True
            break
        if did:
            return None
        # ---- step 2 per group
        all_done = True
        for g, members in E13.GROUPS:
            mem = [c for c in members if c in conds]
            if not mem or g in cs["done_groups"]:
                continue
            all_done = False
            if not all(E13.step1_complete(c, proto) for c in mem):
                if all(cs["fail"].get(f"s1:{c}", 0) >= MAX_FAIL or E13.step1_complete(c, proto) for c in mem) and \
                        any(E13.step1_complete(c, proto) for c in mem):
                    mem = [c for c in mem if E13.step1_complete(c, proto)]
                else:
                    continue
            if g not in cs["subgroups"]:
                nrows = {}
                for c in mem:
                    t = E13.load_task(c, proto)
                    nrows[c] = len(E13.val_rows(t))
                tot = sum(nrows.values())
                nsub = max(1, -(-tot // CAP_ROWS[model]))
                target = tot / nsub
                subs, cur, acc = [], [], 0
                for c in mem:
                    if cur and acc + nrows[c] > target * 1.15 and len(subs) < nsub - 1:
                        subs.append(cur); cur, acc = [], 0
                    cur.append(c); acc += nrows[c]
                subs.append(cur)
                cs["subgroups"][g] = {f"{g}" if len(subs) == 1 else f"{g}{chr(97 + i)}": s for i, s in enumerate(subs)}
                cs["rows"] = {**cs.get("rows", {}), **nrows}
                save()
                log(f"group {g}: subgroups {cs['subgroups'][g]} (rows {nrows})")
            pending = False
            for sg, sm in cs["subgroups"][g].items():
                od = E13.step2_dir(proto, sg)
                mp = os.path.join(od, "run_meta.json")
                if os.path.exists(mp) and json.load(open(mp)).get("complete"):
                    continue
                k = f"s2:{sg}"
                if cs["fail"].get(k, 0) >= MAX_FAIL:
                    continue
                pending = True
                rc = run(f"ext13ci-s2-{sg}-{proto}", ["python", "scripts/ext13ci_circuit_run.py", proto, "--group", sg, "--conds", ",".join(sm),
                                                       "--max-passes", str(S2_PASSES[model])])
                if rc != 0:
                    cs["fail"][k] = cs["fail"].get(k, 0) + 1
                    save()
                    progress(f"step 2 {proto}/{sg} job FAILED rc={rc} (attempt {cs['fail'][k]})")
                elif os.path.exists(mp) and json.load(open(mp)).get("complete"):
                    cpu(["scripts/ext13ci_analyze.py", "--step", "2", "--groups", f"{proto}/{sg}"])
                    if os.path.exists(os.path.join(ROOT, "scripts/ext13ci_text.py")):
                        cpu(["scripts/ext13ci_text.py"])
                    try:
                        s2 = json.load(open(E13.SUMMARY))["step2"]
                        parts = []
                        for c in sm:
                            r = s2[E13.key_of(c, proto)]["curves"]["greedy_mix"]
                            parts.append(f"{c} r(20) {r['r20']['ratio']:.2f}, heads in first 10 {r['composition']['10']['mean_heads']:.1f}, "
                                         f"k80 {r['k_q']['0.8']:.0f}")
                        extra = "; " + "; ".join(parts)
                    except Exception:
                        extra = ""
                    meta = json.load(open(mp))
                    gs = sum(n.get("pass_s") or 0 for n in meta.get("pass_notes", []) if "pass" in n)
                    progress(f"step 2 done {proto}/{sg} ({','.join(sm)}; {meta.get('passes')} passes, {gs / 60:.1f} GPU-min){extra}")
                did = True
                break
            if did:
                break
            if not pending:
                cs["done_groups"].append(g)
                save()
        if did:
            return None
        if all_done:
            log("all groups done")
            if os.path.exists(os.path.join(ROOT, "scripts/ext13ci_text.py")):
                cpu(["scripts/ext13ci_text.py"])
            progress(f"chain {proto} finished (all step-1 / step-2 runs of the runnable conditions done)")
            return "exit"
        if time.time() - state["waited_note"] > 1800:
            missing = [c for c in conds if (c, proto) not in ready or not E13.step1_complete(c, proto)]
            log(f"waiting (BASE READY seen: {sorted(c for c, p in ready if p == proto)}; not finished: {missing})")
            state["waited_note"] = time.time()
        time.sleep(poll)
        return None


if __name__ == "__main__":
    main()
