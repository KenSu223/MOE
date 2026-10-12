"""ext13ab chain helper: which add-back tasks of a model can run now.

A condition / protocol is ready when logs/PROGRESS.md has the exact line fragment "[p5-data] BASE READY <cond>/<proto>"
AND data/wino_ref/p5_conditions.json lists it (status.base true, not behaviour-only). Prints one line:
  RUN <key,key,...>   ready, incomplete add-back tasks (main and replication families)
  WAIT                nothing to run now, conditions still pending
  DONE                every non-behaviour-only condition of the protocol in the JSON is ready and complete, and p5-data
                      has declared all bases ready (all PLAN_ORDER conditions present in the JSON with status.base, or a
                      '[p5-data] ALL BASE READY' / '[p5-data] DONE' line)
Usage: python scripts/ext13ab_ready.py <qwen3|mixtral>
"""
import re, sys
sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace import ext13ab as AB

model = sys.argv[1]
proto = AB.MODEL_PROTO[model]
txt = open("/home/ubuntu/MOE/logs/PROGRESS.md").read()
# only lines written by p5-data (its tag is the line's first bracket tag), so other agents' lines quoting the string do not count
lines = [ln for ln in txt.splitlines() if re.match(r"^\s*(-\s*)?(\S+\s+){0,2}\[p5-data\]", ln)]
ready = set()
for ln in lines:
    ready |= set(re.findall(r"BASE READY (\w+)/(\w+)", ln))
ready = {c for c, p in ready if p == proto}
reg = AB.registry(ready_only=True)
mine = {k: s for k, s in reg.items() if s.model == model and s.cond in ready}
todo = [k for k, s in mine.items() if not AB.run_complete(s.out)]
if todo:
    print("RUN " + ",".join(todo))
    sys.exit(0)
allent = [e for e in AB.cond_entries() if e.get("proto") == proto]
entries = [e for e in allent if not AB._behaviour_only(e)]
all_listed = all(any(e.get("cond") == c for e in allent) for c in AB.PLAN_ORDER)  # behaviour-only entries count as listed
declared = any(re.search(r"\[p5-data\] (ALL BASE READY|DONE)", ln) for ln in lines) or (all_listed and all(AB._status(e, "base") for e in entries))
pending = [e["cond"] for e in entries if e["cond"] not in ready or not AB._status(e, "base")]
print("DONE" if declared and not pending else "WAIT " + ",".join(sorted(set(pending))))
