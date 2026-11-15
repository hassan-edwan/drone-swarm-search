#!/bin/bash
# One-time setup on a login node: a virtual environment with the CPU or CUDA
# build of PyTorch, then the test suite (about 2 minutes on 4 cores).
set -euo pipefail
source "$(dirname "$0")/phase2.env"
cd "$SWARM_DIR"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
