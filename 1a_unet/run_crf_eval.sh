#!/usr/bin/env bash
# Runs eval_crf.py using the venv_crf environment.
# The LD_LIBRARY_PATH is needed so PyTorch can find CUDA libs in the venv.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../../venv_crf"
NVIDIA_LIB="$VENV/lib/python3.12/site-packages/nvidia/cublas/lib"

export LD_LIBRARY_PATH="$NVIDIA_LIB:${LD_LIBRARY_PATH:-}"

exec "$VENV/bin/python" "$SCRIPT_DIR/eval_crf.py" "$@"
