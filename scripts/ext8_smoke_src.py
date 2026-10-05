"""ext8 smoke-test source run on OLMoE: the ext6 STR filter on the Qwen3 paper case IDs with the OLMoE tokenizer
(OLMoE has no paper case set), K = 2 donors, written to results/olmoe_addback_src. Then run ext6_str_sweep.py and
ext6_str_expert.py on it (see scripts/ext8_smoke_chain.sh). Verification / smoke testing only.
"""
import runpy, sys
sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace.models import MODELS
MODELS["olmoe"]["paper_key"] = "Qwen3"
sys.argv = ["ext6_str_filter.py", "olmoe", "--out", "olmoe_addback_src", "--k", "2"]
runpy.run_path("/home/ubuntu/MOE/scripts/ext6_str_filter.py", run_name="__main__")
