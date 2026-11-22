# Sourced by the job scripts.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/phase2.env"
cd "$SWARM_DIR"
if command -v module >/dev/null 2>&1; then set +u; module load "$PY_MODULE"; set -u; fi
unset PYTHONPATH   # the module points it at its own packages, which would shadow the venv's
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
mkdir -p "$RUN"
export OMP_NUM_THREADS=1
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
