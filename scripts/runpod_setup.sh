#!/usr/bin/env bash
# One-time setup on a RunPod GPU pod (RTX 4090 recommended; a CUDA 12.x PyTorch
# template). Run from the repo root after cloning.
set -euo pipefail

pip install -U pip
# The RunPod PyTorch template usually ships a matching torch already. Only install
# torch explicitly if it is missing, to avoid clobbering the pod's CUDA build.
python -c "import torch" 2>/dev/null || pip install torch
pip install chatterbox-tts matplotlib

command -v nsys >/dev/null && nsys --version || \
  echo "nsys not found — torch profiler still works; nsys is optional for the timeline figure."

python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY
