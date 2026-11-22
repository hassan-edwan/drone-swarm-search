#!/bin/bash
# One-time setup on a login node: the cluster's Python module, then a virtual
# environment with PyTorch (CPU build by default) and the requirements.
# The tests run in the smoke job (smoke.sbatch), not on the login node.
set -euo pipefail
source "$(dirname "$0")/phase2.env"
cd "$SWARM_DIR"
if command -v module >/dev/null 2>&1; then set +u; module load "$PY_MODULE"; set -u; fi
unset PYTHONPATH   # the module points it at its own packages, which would shadow the venv's
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
rm -rf "$VENV"   # start clean (a failed setup leaves a broken one)
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --no-cache-dir --upgrade pip
if [ -n "$TORCH_INDEX" ]; then python -m pip install --no-cache-dir torch --index-url "$TORCH_INDEX"; fi
python -m pip install --no-cache-dir -r requirements.txt
python -c "import torch, numpy, gymnasium, pettingzoo; print('ok: torch', torch.__version__, 'numpy', numpy.__version__)"
mkdir -p logs
echo "set up in $SWARM_DIR; next: sbatch -A <account> scripts/cluster/smoke.sbatch"
