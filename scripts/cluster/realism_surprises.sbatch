#!/bin/bash
#SBATCH --job-name=swarm-surprises
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Under-canopy checks planned on the MAP flown in the REAL world (what the map lacks, drawn at random)
# with and without onboard sensing, GPS or camera navigation, wind, drone size, winter snow:
# collisions, turn-backs, extra time. Results in $REAL_RUN/surprises/results.md; also updates
# sites/catoctin_west/exec_stats.json, which the field simulator uses.
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/scripts/cluster/common.sh"
source scripts/cluster/realism.env
python scripts/real_site/surprise_eval.py --worlds "$SURPRISE_WORLDS" --checks "$SURPRISE_CHECKS" \
  --workers "$CPUS" --out "$REAL_RUN/surprises"
