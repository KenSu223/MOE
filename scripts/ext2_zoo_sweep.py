"""ext2-model-zoo: layer sweep under a named protocol, plus the sink diagnostic.

Thin wrapper around scripts/run_sweep.py (not modified): the protocol's chat prefix is injected by binding
`prefix_ids` into the `cases_by_id` the sweep uses, and `Engine.run` is wrapped so that the pass also records
DiagSpec(attn_final, resid_norms) on the prefill rows. From the CLEAN rows the wrapper writes
results/<run>/sink_diag.json (moetrace.ext2_zoo.sink_summary: fraction of prompts whose final position carries the
maximal residual norm / attends mostly to itself, per layer and at the sink layer) and the raw arrays to
/opt/dlami/nvme/moe_ext2/<run>/sweep_diag.npz.

Usage: python scripts/ext2_zoo_sweep.py <model_key> --protocol default|nobos|chat [--sets paper,strict,relaxed]
"""
import argparse, functools, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np
from moetrace.models import MODELS, RESULTS
from moetrace.protocol import cases_by_id
from moetrace.engine import Engine, DiagSpec
from moetrace.ext2_zoo import protocol_spec, write_run_meta, base_meta, sink_summary, diag_dir
import run_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sets", default=None)
    ap.add_argument("--no-diag", action="store_true")
    args = ap.parse_args()
    spec = protocol_spec(args.model, args.protocol)
    run = args.out or spec.run_dir
    od = os.path.join(RESULTS, run)
    assert os.path.exists(os.path.join(od, "case_sets.json")), f"{od}/case_sets.json missing: run ext2_zoo_filter.py first"
    cmd = "python scripts/ext2_zoo_sweep.py " + " ".join(sys.argv[1:])
    write_run_meta(run, base_meta(spec, "sweep", cmd), update=True)
    prefix = list(spec.prefix_ids) or None
    run_sweep.cases_by_id = functools.partial(cases_by_id, prefix_ids=prefix)

    state = {}
    orig_run = Engine.run

    def run_with_diag(self, prefill, spawns, **kw):
        if not args.no_diag:
            kw["diag"] = DiagSpec(attn_final=True, resid_norms=True)
        res = orig_run(self, prefill, spawns, **kw)
        state["res"] = res
        state["n_prefill"] = len(prefill)
        return res

    Engine.run = run_with_diag
    argv = ["run_sweep.py", args.model, "--out", run]
    if not spec.special_tokens:
        argv.append("--no-special-tokens")
    if args.sets:
        argv += ["--sets", args.sets]
    sys.argv = argv
    t0 = time.time()
    run_sweep.main()
    res = state["res"]
    meta = {"sweep_done_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sweep_pass_time_s": res.extra.get("total_s"), "sweep_T": res.extra.get("T"),
            "sweep_wall_s": round(time.time() - t0, 1)}
    if not args.no_diag and "diag" in res.extra:
        dg = res.extra["diag"]
        n = state["n_prefill"] // 2  # clean rows first, then noised rows (run_sweep order)
        lens = res.lens[:n]
        summ = sink_summary(dg["resid_norms"][:, :n], dg["attn_final"][:, :n], lens, prefix_len=len(spec.prefix_ids))
        summ["model"], summ["protocol"], summ["run"] = args.model, spec.protocol, run
        with open(os.path.join(od, "sink_diag.json"), "w") as f:
            json.dump(summ, f, indent=1)
        dd = diag_dir(run)
        np.savez_compressed(os.path.join(dd, "sweep_diag.npz"), resid_norms=dg["resid_norms"], attn_final=dg["attn_final"], lens=res.lens,
                            n_clean=n, prefix_len=len(spec.prefix_ids))
        meta["sink_diag"] = {k: v for k, v in summ.items() if k != "per_layer"}
        meta["sweep_diag_raw"] = os.path.join(dd, "sweep_diag.npz")
        print(time.strftime("%H:%M:%S"), f"sink diag: layer {summ['sink_layer']}: final-is-max-norm {summ['frac_final_max_norm_at_sink_layer']:.3f}, "
              f"final-attends-mostly-to-itself {summ['frac_final_self_attn_max_at_sink_layer']:.3f}, pos0 mass {summ['mean_mass_pos0_at_sink_layer']:.3f}, "
              f"any-early-layer final-is-max {summ['frac_final_max_norm_any_early_layer']:.3f}", flush=True)
    write_run_meta(run, meta, update=True)


if __name__ == "__main__":
    main()
