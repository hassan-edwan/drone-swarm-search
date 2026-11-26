#!/bin/bash
#SBATCH --job-name=swarm-realism
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Old vs calibrated sensing, with and without under-canopy checks, on generated
# wilderness and the real Catoctin site: REAL_EPISODES missions per row (the rule).
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/scripts/cluster/common.sh"
source scripts/cluster/realism.env
python scripts/eval_realism.py --episodes "$REAL_EPISODES" --workers "$CPUS" --out "$REAL_RUN/eval"
