"""Phase-6 tier 2 (p6-analysis): run a Phase-5 tier-2 driver on Phase-6 cells (GPU jobs; called through gpu_queue.sh).

The Phase-5 scripts are imported and their main() is called after scripts/ext14_t2_common.py has patched the module
registries / output paths to the Phase-6 tier-2 dirs results/p6_<cell>_<proto>_t2/ (no ext13 file is edited).

Usage:
  python scripts/ext14_t2_run.py ab  <qwen3|mixtral> --tasks e1_a2e_qwen3[,..] [--max-passes N --budget B --max-tasks T]
  python scripts/ext14_t2_run.py ci1 <cell> <proto> [--max-passes N --rows-per-pass R --limit N]
  python scripts/ext14_t2_run.py ci2 <proto> --group <cell> --conds <cell> [--max-passes N --budget B --limit N]
  python scripts/ext14_t2_run.py ko  <proto> --conds <cell> --with-wr [--time-budget S --limit N]
(the arguments after the component are passed to scripts/ext13ab_run.py, ext13ci_heads_run.py, ext13ci_circuit_run.py,
ext13ko_run.py unchanged)
"""
import os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import ext14_t2_common as T2

SCRIPT = {"ab": "ext13ab_run", "ci1": "ext13ci_heads_run", "ci2": "ext13ci_circuit_run", "ko": "ext13ko_run"}


def main():
    comp = sys.argv[1]
    rest = sys.argv[2:]
    if comp == "ab":
        T2.patch_ab()
    elif comp in ("ci1", "ci2"):
        T2.patch_ci()
    elif comp == "ko":
        T2.patch_wr()
    else:
        raise SystemExit(f"unknown component {comp}")
    T2.patch_argparse()
    T2.patch_engine_shared()  # no-op for models without a shared expert
    T2.patch_loaders_shared()
    import importlib
    mod = importlib.import_module(SCRIPT[comp])
    sys.argv = [f"scripts/{SCRIPT[comp]}.py (via ext14_t2_run.py {comp})"] + rest
    mod.main()


if __name__ == "__main__":
    main()
