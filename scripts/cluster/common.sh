# Sourced by the job scripts.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/phase2.env"
cd "$SWARM_DIR"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
mkdir -p "$RUN"
export OMP_NUM_THREADS=1
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
