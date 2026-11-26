#!/bin/bash
# Submit the three realism jobs (independent of each other):
#   realism_eval    the comparison table at scale
#   realism_sitl    ArduPilot flights: tracking and clearances
#   realism_targets retrain the learned target policy under the new defaults, then evaluate
# Pass the account (and partition) as arguments:  bash scripts/cluster/submit_realism.sh -A msml642-class
set -euo pipefail
source "$(dirname "$0")/phase2.env"
source "$(dirname "$0")/realism.env"
cd "$SWARM_DIR"
mkdir -p logs
for J in realism_eval realism_sitl realism_targets; do
  echo "$J: $(sbatch --parsable "$@" scripts/cluster/$J.sbatch)"
done
echo "watch: squeue --me;  logs: $SWARM_DIR/logs;  results: $SWARM_DIR/$REAL_RUN"
