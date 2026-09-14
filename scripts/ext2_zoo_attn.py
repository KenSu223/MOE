"""ext2-model-zoo: attention-output / MoE-output / whole-layer sweep (ext2-attn-patch's scripts/ext2_attn_sweep.py, kinds
attn_layer, layer, block) for a zoo run under its protocol. Thin wrapper: binds the protocol's chat prefix into the
`cases_by_id` the sweep script uses and reuses the run's case_sets.json. Output: results/<run>_attnsweep/.

Usage: python scripts/ext2_zoo_attn.py <model_key> --protocol default|nobos|chat [--sets paper] [--kinds attn_layer,layer,block]
"""
import argparse, functools, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
from moetrace.models import RESULTS
from moetrace.protocol import cases_by_id
from moetrace.ext2_zoo import protocol_spec, write_run_meta, base_meta
import ext2_attn_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--sets", default=None, help="default: paper if the run has it, else strict")
    ap.add_argument("--kinds", default="attn_layer,layer,block")
    args = ap.parse_args()
    spec = protocol_spec(args.model, args.protocol)
    run = spec.run_dir
    out = run + "_attnsweep"
    sets = json.load(open(os.path.join(RESULTS, run, "case_sets.json")))
    s = args.sets or ("paper" if "paper" in sets else "strict")
    prefix = list(spec.prefix_ids) or None
    ext2_attn_sweep.cases_by_id = functools.partial(cases_by_id, prefix_ids=prefix)
    argv = ["ext2_attn_sweep.py", args.model, "--out", out, "--base-run", run, "--sets", s, "--kinds", args.kinds, "--agent", "ext2-model-zoo"]
    if not spec.special_tokens:
        argv.append("--no-special-tokens")
    sys.argv = argv
    t0 = time.time()
    ext2_attn_sweep.main()
    meta = json.load(open(os.path.join(RESULTS, out, "run_meta.json")))
    meta.update(base_meta(spec, "attn_sweep", "python scripts/ext2_zoo_attn.py " + " ".join(sys.argv[1:]),
                          {"wall_s": round(time.time() - t0, 1), "zoo_run": run, "kinds": args.kinds.split(","), "case_sets": [s]}))
    meta["experiment"] = "ext2 attention-vs-MoE sweep for the model zoo (RESEARCH_PLAN Direction 2, Question B)"
    write_run_meta(out, meta, update=True)


if __name__ == "__main__":
    main()
