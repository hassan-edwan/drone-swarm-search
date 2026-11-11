#!/bin/bash
# Evaluate every trained model from train_array.sbatch on the same held-out
# scenarios. Submit after the array finishes, or chain it:
#   jid=$(sbatch --parsable scripts/hpc/train_array.sbatch)
#   sbatch --dependency=afterok:$jid scripts/hpc/evaluate.sbatch
#
#SBATCH --job-name=swarm-eval
#SBATCH --account=CHANGE_ME
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=runs/hpc/slurm_eval_%j.out

set -euo pipefail
source ~/venvs/swarm/bin/activate
TAG=${TAG:-m1}
SEED=${SEED:-0}

python scripts/evaluate.py --episodes 100 --out "results/${TAG}_hpc_s${SEED}" \
  --models runs/hpc/${TAG}_*_s${SEED}/model.pt
