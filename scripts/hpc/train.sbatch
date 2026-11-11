#!/bin/bash
# One training run + evaluation on a CPU node (written for UMD Zaratan).
#
# One-time setup on a login node:
#   python -m venv ~/venvs/swarm && source ~/venvs/swarm/bin/activate
#   pip install -r requirements.txt
# Then, from the repo root (set --account first):
#   sbatch scripts/hpc/train.sbatch
#   ARGS="--guide heuristic" TAG=guided sbatch scripts/hpc/train.sbatch
#
#SBATCH --job-name=swarm
#SBATCH --account=CHANGE_ME
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=runs/hpc/slurm_%j.out

set -euo pipefail
source ~/venvs/swarm/bin/activate
STEPS=${STEPS:-2000000}
TAG=${TAG:-run}
ARGS=${ARGS:-}
out="runs/hpc/${TAG}"
mkdir -p runs/hpc

python scripts/train.py --algo mappo --steps "$STEPS" --threads "$SLURM_CPUS_PER_TASK" $ARGS \
  --out "$out" > "$out.log" 2>&1
python scripts/evaluate.py --models "$out/model.pt" --episodes 100 --out "results/${TAG}" >> "$out.log" 2>&1
