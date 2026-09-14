"""Direction 4: expert-level pass over the union of the CodeFact case sets (all layers, no pairs), memory-bounded.

Same job as scripts/run_expert.py (whose build_spawns / rows_from_result are imported unchanged) but (i) the cases come
from data/codefact/items.jsonl via moetrace.ext4_data, (ii) the routing comes from the scan pass (results/<out>/
sweep_routing.parquet written by ext4_select.py) and (iii) the work is split into passes that fit the 24 GB GPU with
code-length prompts (T up to 160): cases are sorted by length and grouped so that (2 x cases) x T <= --prefill-tokens
(prefill memory is O(rows x T x hidden) in fp32), and inside each case chunk the layers are grouped so that the number
of wavefront rows (layer + 2 coalitions + every clean-active and noised-only expert per case and layer) stays under
--wf-rows; the engine's wavefront attention chunk is --wf-tokens // T. Every pass writes
results/<out>/expert_parts/part_c<ci>_L<a>-<b>.parquet (and prefill_c<ci>.parquet); existing parts are skipped (resume),
and the parts are concatenated into expert_rows.parquet after every pass.

Usage: python scripts/ext4_run_expert.py <model_key> --out codefact_<model>_<protocol> --protocol raw|nobos|chat
          [--prefill-tokens 160000] [--wf-rows 45000] [--wf-tokens 160000] [--no-pairs] [--dry-run]
"""
import argparse, glob, json, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS, RESULTS
from moetrace.ext4_data import load_cases
from moetrace.noise import noise_draw
from run_expert import build_spawns, rows_from_result


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def case_chunks(ids: list[int], cases: dict, prefill_tokens: int) -> list[list[int]]:
    """Length-sorted case groups with 2 x n x T_max <= prefill_tokens."""
    ids = sorted(ids, key=lambda c: len(cases[c].ids))
    out, cur = [], []
    for c in ids:
        T = len(cases[c].ids)
        if cur and 2 * (len(cur) + 1) * T > prefill_tokens:
            out.append(cur)
            cur = []
        cur.append(c)
    if cur:
        out.append(cur)
    return out


def rows_per_layer(ch: list[int], layers: list[int], rt: dict) -> dict[int, int]:
    out = {}
    for l in layers:
        n = 0
        for c in ch:
            ce, ne = rt[(c, "clean", l)], rt[(c, "noised", l)]
            n += 3 + len(ce) + len(set(ne) - set(ce))
        out[l] = n
    return out


def layer_groups(ch: list[int], layers: list[int], rt: dict, wf_rows: int) -> list[list[int]]:
    rpl = rows_per_layer(ch, layers, rt)
    groups, cur, tot = [], [], 0
    for l in layers:
        if cur and tot + rpl[l] > wf_rows:
            groups.append(cur)
            cur, tot = [], 0
        cur.append(l)
        tot += rpl[l]
    if cur:
        groups.append(cur)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default="raw", choices=["raw", "nobos", "chat"])
    ap.add_argument("--layers", default="all")
    ap.add_argument("--sigma-mult", type=float, default=3.0)
    ap.add_argument("--no-pairs", action="store_true")
    ap.add_argument("--prefill-tokens", type=int, default=160000, help="max (2 x cases) x T per pass")
    ap.add_argument("--wf-rows", type=int, default=45000, help="max wavefront (spawn) rows per pass")
    ap.add_argument("--wf-tokens", type=int, default=160000, help="wavefront attention chunk = wf_tokens // T")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    m = MODELS[args.model]
    od = os.path.join(RESULTS, args.out)
    parts_dir = os.path.join(od, "expert_parts")
    os.makedirs(parts_dir, exist_ok=True)
    sets = json.load(open(os.path.join(od, "case_sets.json")))
    ct = pd.read_parquet(os.path.join(od, "sweep_cases.parquet"))
    ids = ct.case_id.tolist()
    routing = pd.read_parquet(os.path.join(od, "sweep_routing.parquet"))
    n_layers = int(routing.layer.max()) + 1
    layers = list(range(n_layers)) if args.layers == "all" else sorted(set(int(x) for x in args.layers.split(",")))
    routing = routing[routing.layer.isin(layers)]
    cases, rej, meta = load_cases(args.model, ids, protocol=args.protocol)
    assert not rej, rej[:5]
    cti = ct.set_index("case_id")
    for c in ids[:50]:
        assert json.loads(cti.loc[c, "ids"]) == cases[c].ids, c  # the tokenisation must be the one the scan used
    rt = {}
    for (cid, run, layer), g in routing.groupby(["case_id", "run", "layer"]):
        rt[(int(cid), run, int(layer))] = dict(zip(g.expert.astype(int).tolist(), g.weight.tolist()))
    cch = case_chunks(ids, cases, args.prefill_tokens)
    plan = []
    for ci, ch in enumerate(cch):
        T = max(len(cases[c].ids) for c in ch)
        for lg in layer_groups(ch, layers, rt, args.wf_rows):
            plan.append((ci, ch, lg, T))
    tot_rows = 0
    for ci, ch, lg, T in plan:
        spawns, _ = build_spawns(ch, lg, rt, len(ch), args.no_pairs)
        tot_rows += len(spawns)
        if args.dry_run:
            log(f"[dry-run] case chunk {ci}: {len(ch)} cases T={T} ({2 * len(ch) * T} prefill row-tokens), layers {lg[0]}..{lg[-1]} ({len(lg)}): {len(spawns)} spawn rows, wf_chunk {max(512, min(8192, args.wf_tokens // T))}")
    log(f"{args.model} {args.protocol}: {len(ids)} cases, {len(cch)} case chunks, {len(plan)} passes, {tot_rows} spawn rows total")
    if args.dry_run:
        return
    from moetrace.engine import Engine, PrefillSpec
    eng = Engine(m["repo"])
    sigma = args.sigma_mult * eng.embed_std
    Hd = eng.hidden
    times, peaks, skipped = [], [], 0
    path = os.path.join(od, "expert_rows.parquet")

    def concat_parts():
        files = sorted(glob.glob(os.path.join(parts_dir, "part_c*_L*.parquet")))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df.to_parquet(path, index=False)
        pf = sorted(glob.glob(os.path.join(parts_dir, "prefill_c*.parquet")))
        pd.concat([pd.read_parquet(f) for f in pf], ignore_index=True).drop_duplicates("case_id").to_parquet(os.path.join(od, "expert_prefill_all.parquet"), index=False)
        return df

    for pi, (ci, ch, lg, T) in enumerate(plan):
        tag = f"part_c{ci:02d}_L{lg[0]:02d}-{lg[-1]:02d}"
        fpath = os.path.join(parts_dir, tag + ".parquet")
        if os.path.exists(fpath):
            skipped += 1
            continue
        n = len(ch)
        pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ch]
        pre += [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id, cases[c].subject_pos,
                            noise_draw(c, len(cases[c].subject_pos), Hd, sigma)) for c in ch]
        spawns, tags = build_spawns(ch, lg, rt, n, args.no_pairs)
        wf_chunk = int(max(512, min(8192, args.wf_tokens // T)))
        log(f"pass {pi + 1}/{len(plan)} {tag}: {n} cases T={T}, {len(pre)} prefill rows, {len(spawns)} spawn rows, wf_chunk={wf_chunk}")
        t0 = time.time()
        res = eng.run(pre, spawns, record_routing=False, log=log if pi == 0 else None, wf_chunk=wf_chunk)
        times.append(round(time.time() - t0, 1))
        peaks.append(round(torch.cuda.max_memory_allocated() / 2**30, 2))
        torch.cuda.reset_peak_memory_stats()
        log(f"pass time {times[-1]}s; peak GPU memory {peaks[-1]} GB")
        df = rows_from_result(res, tags, ch, rt, args.sigma_mult)
        df["case_chunk"] = ci
        df.to_parquet(fpath, index=False)
        d = res.delta
        pd.DataFrame(dict(case_id=ch, delta_clean=d[:n], delta_noised=d[n:])).to_parquet(os.path.join(parts_dir, f"prefill_c{ci:02d}.parquet"), index=False)
        all_df = concat_parts()
        log(f"{tag}: {len(df)} rows; expert_rows.parquet now {len(all_df)} rows, {all_df.case_id.nunique()} cases, {all_df.layer.nunique()} layers")
        del res, df
        torch.cuda.empty_cache()
    all_df = concat_parts()
    rm = {"model": args.model, "repo": m["repo"], "protocol": args.protocol, "special_tokens": meta["special_tokens"], "chat_prefix_len": meta["chat_prefix_len"],
          "sigma_mult": args.sigma_mult, "case_sets": [s for s in sets if s != "scan"], "n_cases": len(ids), "n_layers": len(layers),
          "pairs": not args.no_pairs, "prefill_tokens": args.prefill_tokens, "wf_rows": args.wf_rows, "wf_tokens": args.wf_tokens,
          "n_case_chunks": len(cch), "n_passes": len(plan), "passes_skipped_resume": skipped, "pass_times_s": times, "peak_mem_gb": peaks,
          "n_rows": int(len(all_df)), "command": " ".join(sys.argv), "agent": "ext4-codefact",
          "experiment": "ext4 CodeFact expert pass, all layers, no pairs (RESEARCH_PLAN Direction 4)", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(od, "run_meta.json"), "w") as f:
        json.dump(rm, f, indent=1)
    log(f"done: {len(all_df)} rows over {all_df.layer.nunique()} layers, {sum(times):.0f}s GPU in {len(times)} passes (+{skipped} resumed)")


if __name__ == "__main__":
    main()
