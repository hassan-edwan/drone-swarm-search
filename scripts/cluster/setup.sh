#!/bin/bash
# One-time setup on a login node: the cluster's Python module, then a virtual
# environment with PyTorch (CPU build by default) and the requirements.
# The tests run in the smoke job (smoke.sbatch), not on the login node.
set -euo pipefail
source "$(dirname "$0")/phase2.env"
cd "$SWARM_DIR"
if command -v module >/dev/null 2>&1; then set +u; module load "$PY_MODULE"; set -u; fi
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
if [ -n "$TORCH_INDEX" ]; then pip install torch --index-url "$TORCH_INDEX"; fi
pip install -r requirements.txt
mkdir -p logs
echo "set up in $SWARM_DIR; next: sbatch -A <account> scripts/cluster/smoke.sbatch"
