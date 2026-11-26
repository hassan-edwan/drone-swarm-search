#!/bin/bash
#SBATCH --job-name=swarm-sitl
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Missions on the hamlet site flown in ArduPilot SITL (one copter process per drone)
# in each wind, with GPS-only and with camera-based navigation under the canopy;
# tracking and clearance statistics (from the true track) in $REAL_RUN/sitl/summary.md.
# Re-submitting skips flights already done.
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/scripts/cluster/common.sh"
source scripts/cluster/realism.env
SLOTS=$(( CPUS / 6 ))   # each flight: up to 5 copter processes and the controller
python scripts/real_site/sitl_batch.py --site sites/catoctin_west --seeds "$SITL_SEEDS" --winds $SITL_WINDS \
  --steps "$SITL_STEPS" --slots "$SLOTS" --ardupilot "$ARDUPILOT" --out "$REAL_RUN/sitl"
