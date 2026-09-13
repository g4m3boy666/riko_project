#!/usr/bin/env bash
set -euo pipefail

# Check your NVIDIA/CUDA version before using this CUDA 12.6 PyTorch index.
uv venv --python 3.10

uv pip install --python .venv/bin/python torch==2.6.0 torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -r extra-req.txt -r requirements.txt

.venv/bin/python - <<'PYCODE'
import nltk
for pkg in ["averaged_perceptron_tagger", "cmudict"]:
    nltk.download(pkg)
PYCODE
