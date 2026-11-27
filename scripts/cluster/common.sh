# Sourced by the job scripts.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/phase2.env"
cd "$SWARM_DIR"
if command -v module >/dev/null 2>&1; then set +u; module load "$PY_MODULE"; set -u; fi
unset PYTHONPATH   # the module points it at its own packages, which would shadow the venv's
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"
else echo "ERROR: no Python environment at $VENV (set VENV=..., or run scripts/cluster/setup.sh)" >&2; exit 1; fi
python -c "import numpy, scipy, torch" 2>/dev/null || {
  echo "ERROR: $VENV is missing packages; run: source $VENV/bin/activate && pip install -r $SWARM_DIR/requirements.txt" >&2; exit 1; }
echo "python: $(command -v python)"
mkdir -p "$RUN"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1   # progress lines reach the log file as they happen
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
echo "settings: POLICY_ARCH=$POLICY_ARCH PLAN=$PLAN FAMILIES=$FAMILIES SEEDS=$SEEDS RL_STEPS=$RL_STEPS N_ENVS=$N_ENVS CPUS=$CPUS RUN=$RUN"
