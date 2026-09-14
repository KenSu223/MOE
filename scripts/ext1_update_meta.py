"""Fill run_meta.json of the ext1 all-layer runs with completion facts read from the pass logs and the parquet output."""
import json, os, re, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import pandas as pd
from moetrace.models import RESULTS

LOGS = {"qwen3_bos_alllayers": "logs/ext1_expert_qwen3_bos_alllayers.log", "mixtral_bos_alllayers": "logs/ext1_expert_mixtral_bos_alllayers.log",
        "mixtral_nobos_alllayers": "logs/ext1_expert_mixtral_nobos_alllayers.log"}
for run, logf in LOGS.items():
    od = os.path.join(RESULTS, run)
    p = os.path.join(od, "expert_rows.parquet")
    if not (os.path.exists(p) and os.path.exists(logf)):
        print(run, "not finished"); continue
    txt = open(logf).read()
    times = [float(x) for x in re.findall(r"pass time ([0-9.]+)s", txt)]
    chunks = re.findall(r"chunk \d+/\d+: (\d+) cases, layers (\d+)\.\.(\d+) \((\d+)\), (\d+) prefill rows, (\d+) spawn rows", txt)
    df = pd.read_parquet(p)
    meta = json.load(open(os.path.join(od, "run_meta.json")))
    n_layers = int(df.layer.nunique())
    meta.update(completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(p))), n_passes=len(times), pass_times_s=times,
                gpu_time_s=round(sum(times), 1), n_cases=int(df.case_id.nunique()), n_layers_done=n_layers, n_rows=int(len(df)),
                rows_by_kind=df.kind.value_counts().to_dict(), chunks=[dict(cases=int(c[0]), layers=f"{c[1]}-{c[2]}", n_layers=int(c[3]), prefill_rows=int(c[4]), spawn_rows=int(c[5])) for c in chunks],
                complete=(n_layers == (48 if meta["model"] == "qwen3" else 32)), log=logf)
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    print(run, "rows", len(df), "layers", n_layers, "passes", len(times), "gpu s", round(sum(times), 1), "complete", meta["complete"])
