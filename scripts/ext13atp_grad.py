"""p5-atp HF stage: gradient estimates (AtP, EAP-IG, DLA), HF activation patches with routing records (free and frozen
routing) and AtP* cut estimates, per directed case (moetrace/ext13atp.py run_case_hf).

GPU (through scripts/gpu_queue.sh):
  python scripts/ext13atp_grad.py --task wino [--bs 256] [--time-budget 840]
    -> results/p5_atp_<task>/hf_parts/part_<first case>.parquet (resumable: finished cases are skipped), merged into
       hf_nodes.parquet + hf_cases.parquet when every case is done; gradient samples of the fp32-check cases
       (hf_grad_sample_bf16.pt). Exit code 0 in both cases; run_meta.json "hf".complete says whether it finished.
CPU fp32 check (no GPU; the checkpoint's native fp32 weights):
  python scripts/ext13atp_grad.py --task wino --fp32-check [--n-check 4]
    -> hf_fp32_nodes.parquet, hf_grad_sample_fp32.pt for the first n validation directed cases (gradients only).
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace import ext13atp as A

log = A.log


def _meta(task, key, upd):
    p = os.path.join(A.run_dir(task), "run_meta.json")
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.setdefault(key, {}).update(upd)
    json.dump(cur, open(p, "w"), indent=1, default=str)


def check_cases(cs, n):
    return cs["directed"]["validation"][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(A.TASKS))
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--time-budget", type=float, default=840.0)
    ap.add_argument("--part-size", type=int, default=16)
    ap.add_argument("--fp32-check", action="store_true")
    ap.add_argument("--n-check", type=int, default=4)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--fp32-patch", action="store_true", help="fp32 check with the activation patches too (fp32 truth)")
    args = ap.parse_args()
    t_start = time.time()
    pairs, cs, fam, ids, dc, split = A.load_task(args.task)
    od = A.run_dir(args.task)
    chk = check_cases(cs, args.n_check)
    if args.fp32_check:
        torch.set_num_threads(args.threads)
        hf = A.HFAtP(device="cpu", dtype=torch.float32)
        log(f"fp32 model loaded on CPU ({time.time() - t_start:.0f}s)")
        outs, samp = [], {}
        with torch.no_grad():
            for c in chk:
                t0 = time.time()
                est, info, gp = A.run_case_hf(hf, dc[c], patch=args.fp32_patch)
                est.insert(0, "case_id", c)
                outs.append(est)
                samp[c] = {"g_mid": gp["g_mid"].clone(), "g_post": gp["g_post"].clone(), "delta": gp["delta"]}
                log(f"fp32 case {c}: {time.time() - t0:.0f}s, delta corrupt {info['delta_corrupt_hf']:.4f} clean {info['delta_clean_hf']:.4f}")
        suf = "_patch" if args.fp32_patch else ""
        pd.concat(outs).to_parquet(os.path.join(od, f"hf_fp32{suf}_nodes.parquet"), index=False)
        if not args.fp32_patch:
            torch.save(samp, os.path.join(od, "hf_grad_sample_fp32.pt"))
        _meta(args.task, "fp32_check" + suf, {"cases": chk, "cpu_s": time.time() - t_start, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return
    pd_dir = os.path.join(od, "hf_parts")
    os.makedirs(pd_dir, exist_ok=True)
    done = set()
    for f in glob.glob(os.path.join(pd_dir, "cases_*.parquet")):
        done |= set(pd.read_parquet(f).case_id.astype(int))
    todo = [c for c in ids if c not in done]
    log(f"{args.task}: {len(ids)} cases, {len(done)} done, {len(todo)} to do")
    _meta(args.task, "hf", {"command": "python " + " ".join(sys.argv), "bs": args.bs, "ig_m": A.IG_M, "complete": False,
                            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    hf = A.HFAtP(device="cuda", dtype=torch.bfloat16)
    log(f"model loaded ({time.time() - t_start:.0f}s)")
    samp_path = os.path.join(od, "hf_grad_sample_bf16.pt")
    samp = torch.load(samp_path, weights_only=False) if os.path.exists(samp_path) else {}
    buf_n, buf_c, t_case = [], [], []
    stopped = False

    def flush():
        if not buf_n:
            return
        first = int(buf_c[0]["case_id"])
        pd.concat(buf_n).to_parquet(os.path.join(pd_dir, f"nodes_{first:05d}.parquet"), index=False)
        pd.DataFrame(buf_c).to_parquet(os.path.join(pd_dir, f"cases_{first:05d}.parquet"), index=False)
        buf_n.clear(); buf_c.clear()

    with torch.no_grad():
        for c in todo:
            if time.time() - t_start > args.time_budget:
                stopped = True
                break
            t0 = time.time()
            nodes, info, gp = A.run_case_hf(hf, dc[c], bs=args.bs, patch=True)
            nodes.insert(0, "case_id", c)
            info["split"] = split[c]
            info["s"] = time.time() - t0
            buf_n.append(nodes); buf_c.append(info)
            t_case.append(info["s"])
            if c in chk:
                samp[c] = {"g_mid": gp["g_mid"].cpu().clone(), "g_post": gp["g_post"].cpu().clone(), "delta": gp["delta"]}
                torch.save(samp, samp_path)
            if len(buf_c) >= args.part_size:
                flush()
                log(f"{len(t_case)} cases this run, {np.mean(t_case):.2f}s/case, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    flush()
    done = set()
    for f in glob.glob(os.path.join(pd_dir, "cases_*.parquet")):
        done |= set(pd.read_parquet(f).case_id.astype(int))
    complete = all(c in done for c in ids)
    if complete:
        N = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(pd_dir, "nodes_*.parquet")))])
        Cs = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(pd_dir, "cases_*.parquet")))])
        N.to_parquet(os.path.join(od, "hf_nodes.parquet"), index=False)
        Cs.to_parquet(os.path.join(od, "hf_cases.parquet"), index=False)
    prev = json.load(open(os.path.join(od, "run_meta.json"))).get("hf", {}).get("gpu_s_total", 0.0)
    _meta(args.task, "hf", {"complete": complete, "gpu_s_total": prev + time.time() - t_start,
                            "mean_case_s": float(np.mean(t_case)) if t_case else None,
                            "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "last_run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    log(f"{args.task}: {'COMPLETE' if complete else 'partial (time budget)'}; {len(done)}/{len(ids)} cases; run {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
