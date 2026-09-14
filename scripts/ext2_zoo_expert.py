"""ext2-model-zoo: all-layer expert pass (Direction-1 machinery) under a named protocol.

Thin wrapper around scripts/run_expert.py (not modified): binds the protocol's chat prefix into `cases_by_id`, runs
`--layers 0..L-1 --no-pairs`, and picks the number of layer chunks automatically from a dry-run row count so that no
pass holds more than MAX_ROWS_PER_CHUNK (90k) wavefront rows (one Qwen3 pass at 86k rows used 16 GB in ext1).
Rows are merged into results/<run>/expert_rows.parquet after every chunk (run_expert.merge_rows), so a killed job
resumes by re-running (finished layers are simply recomputed unless --resume skips them).

Usage: python scripts/ext2_zoo_expert.py <model_key> --protocol default|nobos|chat [--layers 0,1,...] [--chunks N] [--resume]
"""
import argparse, functools, json, math, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import pandas as pd
from moetrace.models import MODELS, RESULTS
from moetrace.arch import load_spec
from moetrace.protocol import cases_by_id, load_case_sets
from moetrace.ext2_zoo import protocol_spec, write_run_meta, base_meta, MAX_ROWS_PER_CHUNK
import run_expert


def count_rows(od: str, layers: list[int], model: str, spec) -> int:
    """Exact spawn-row count for the given layers from the recorded routing (same rule as run_expert.build_spawns)."""
    routing = pd.read_parquet(os.path.join(od, "sweep_routing.parquet"))
    routing = routing[routing.layer.isin(layers)]
    n_cases = routing.case_id.nunique()
    per = routing.groupby(["case_id", "layer", "run"]).expert.apply(set).unstack("run")
    n_expert = int(per.apply(lambda r: len(r["clean"]) + len(r["noised"] - r["clean"]), axis=1).sum())
    return n_expert + 3 * n_cases * len(layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", default=None, help="comma list; default all layers")
    ap.add_argument("--chunks", type=int, default=None, help="override the automatic chunk count")
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS_PER_CHUNK)
    ap.add_argument("--resume", action="store_true", help="skip layers already present in expert_rows.parquet")
    args = ap.parse_args()
    spec = protocol_spec(args.model, args.protocol)
    run = args.out or spec.run_dir
    od = os.path.join(RESULTS, run)
    for f in ("case_sets.json", "sweep_cases.parquet", "sweep_routing.parquet"):
        assert os.path.exists(os.path.join(od, f)), f"{od}/{f} missing: run ext2_zoo_sweep.py first"
    arch, _ = load_spec(MODELS[args.model]["repo"])
    layers = sorted(set(int(x) for x in args.layers.split(","))) if args.layers else list(range(arch.n_layers))
    if args.resume and os.path.exists(os.path.join(od, "expert_rows.parquet")):
        done = set(pd.read_parquet(os.path.join(od, "expert_rows.parquet"), columns=["layer"]).layer.unique().tolist())
        layers = [l for l in layers if l not in done]
        if not layers:
            print("all layers present, nothing to do")
            return
    n_rows = count_rows(od, layers, args.model, spec)
    n_chunks = args.chunks or max(1, math.ceil(n_rows / args.max_rows))
    n_chunks = min(n_chunks, len(layers))
    cmd = "python scripts/ext2_zoo_expert.py " + " ".join(sys.argv[1:])
    write_run_meta(run, base_meta(spec, "expert", cmd, {"expert_layers": f"{layers[0]}-{layers[-1]} ({len(layers)})", "expert_no_pairs": True,
                                                         "expert_rows_planned": n_rows, "expert_layer_chunks": n_chunks}), update=True)
    print(time.strftime("%H:%M:%S"), f"{args.model}/{spec.protocol} -> {run}: {len(layers)} layers, {n_rows} spawn rows, {n_chunks} chunks", flush=True)
    prefix = list(spec.prefix_ids) or None
    run_expert.cases_by_id = functools.partial(cases_by_id, prefix_ids=prefix)
    argv = ["run_expert.py", args.model, "--layers", ",".join(map(str, layers)), "--no-pairs", "--layer-chunks", str(n_chunks), "--out", run]
    if not spec.special_tokens:
        argv.append("--no-special-tokens")
    sys.argv = argv
    t0 = time.time()
    run_expert.main()
    df = pd.read_parquet(os.path.join(od, "expert_rows.parquet"))
    write_run_meta(run, {"expert_done_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "expert_wall_s": round(time.time() - t0, 1),
                         "expert_n_rows": int(len(df)), "expert_n_layers_done": int(df.layer.nunique()), "expert_complete": bool(df.layer.nunique() == arch.n_layers),
                         "expert_rows_by_kind": df.kind.value_counts().to_dict(),
                         "outputs": ["filter_scan.parquet, filter_rejects.parquet, case_sets.json (filter)", "sweep_rows/sweep_routing/sweep_cases.parquet, sweep_summary.json, sink_diag.json (sweep)",
                                     "expert_rows.parquet (kinds: layer, coalition_clean, coalition_union, expert, expert_noised_only; all layers; no expert_scaled), expert_prefill_L*.parquet"]},
                   update=True)


if __name__ == "__main__":
    main()
