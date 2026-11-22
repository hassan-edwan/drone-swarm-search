#!/bin/bash
# Submit the whole Phase 2 pipeline with dependencies:
#   imitate -> train (one job per seed, requeues itself) -> evaluate
# Pass extra sbatch options (account, partition) as arguments:
#   bash scripts/cluster/submit.sh -A <account>
# Memory is the cluster default per core (4 GB on Zaratan): 128 GB for imitation.
set -euo pipefail
source "$(dirname "$0")/phase2.env"
cd "$SWARM_DIR"
mkdir -p logs
N=$(($(wc -w <<< "$SEEDS") - 1))
J1=$(sbatch --parsable "$@" scripts/cluster/imitate.sbatch)
J2=$(sbatch --parsable "$@" --dependency=afterok:$J1 --array=0-$N scripts/cluster/train.sbatch)
J3=$(sbatch --parsable "$@" --dependency=afterok:$J2 scripts/cluster/eval.sbatch)
echo "imitate $J1 -> train $J2 (seeds: $SEEDS) -> eval $J3"
echo "watch: squeue --me; logs in $SWARM_DIR/logs and $SWARM_DIR/$RUN"
