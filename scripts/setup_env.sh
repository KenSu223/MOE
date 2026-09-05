#!/usr/bin/env bash
# Phase 0a: main Python environment for the reproduction (CUDA PyTorch + HF stack).
set -euo pipefail
cd /home/ubuntu/MOE
[ -d .venv ] || python3.14 -m venv .venv
. .venv/bin/activate
pip install -U pip wheel
pip install torch
pip install transformers safetensors "huggingface_hub[hf_xet]" accelerate numpy scipy pandas pyarrow matplotlib tqdm
python - <<'PY'
import torch, transformers, safetensors, pandas, scipy, matplotlib
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
print("transformers", transformers.__version__, "| safetensors", safetensors.__version__)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); y = x @ x; torch.cuda.synchronize()
print("bf16 matmul on GPU ok:", tuple(y.shape))
PY
echo SETUP_DONE
