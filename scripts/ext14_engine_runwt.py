"""ext14 (p6-engine) dev helper: run a script with `moetrace` imported from a given tree, even when the script itself
inserts /home/ubuntu/MOE at the front of sys.path (the package is cached in sys.modules first).
Usage: python scripts/ext14_engine_runwt.py <tree> <script.py> [args ...]"""
import os, runpy, sys

tree = os.path.abspath(sys.argv[1])
sys.path.insert(0, tree)
import moetrace.engine  # noqa: E402

assert moetrace.engine.__file__.startswith(tree + "/"), moetrace.engine.__file__
print(f"[runwt] moetrace from {os.path.dirname(moetrace.engine.__file__)}", flush=True)
script = sys.argv[2]
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
