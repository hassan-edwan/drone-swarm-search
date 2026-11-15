#!/bin/bash
# Submit the whole Phase 2 pipeline with dependencies:
#   imitate -> train (one job per seed, requeues itself) -> evaluate
# Pass extra sbatch options (account, partition) as arguments:
#   bash scripts/cluster/submit.sh --account=myproj --partition=cpu
set -euo pipefail
cd "$(dirname "$0")"
source phase2.env
N=$(($(wc -w <<< "$SEEDS") - 1))
J1=$(sbatch --parsable "$@" imitate.sbatch)
J2=$(sbatch --parsable "$@" --dependency=afterok:$J1 --array=0-$N train.sbatch)
J3=$(sbatch --parsable "$@" --dependency=afterok:$J2 eval.sbatch)
echo "imitate $J1 -> train $J2 (seeds: $SEEDS) -> eval $J3"
