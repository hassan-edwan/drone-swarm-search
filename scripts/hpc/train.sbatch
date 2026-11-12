#!/bin/bash
# One training run + evaluation on a CPU node (written for UMD Zaratan).
#
# One-time setup on a login node:
#   python -m venv ~/venvs/swarm && source ~/venvs/swarm/bin/activate
#   pip install -r requirements.txt
# Then, from the repo root (set --account first):
#   sbatch scripts/hpc/train.sbatch
#   ARGS="--obstacles 0.3 --init models/m1_infer_2m.pt" EVAL=--m2 TAG=m2 sbatch scripts/hpc/train.sbatch
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
EVAL=${EVAL:-}
out="runs/hpc/${TAG}"
mkdir -p runs/hpc

python scripts/train.py --algo mappo --steps "$STEPS" --threads "$SLURM_CPUS_PER_TASK" $ARGS \
  --out "$out" > "$out.log" 2>&1
python scripts/evaluate.py --models "$out/model.pt" --episodes 100 $EVAL --out "results/${TAG}" >> "$out.log" 2>&1
