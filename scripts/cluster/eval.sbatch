#!/bin/bash
#SBATCH --job-name=swarm-eval
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Baselines, the copy and every seed's RL model on the core conditions and on
# the mission suite (docs/MISSIONS.md), EVAL_EPISODES missions each.
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/scripts/cluster/common.sh"
MODELS="$RUN/bc.pt"
for s in $SEEDS; do   # best on validation; the final model when validation was off
  B="$RUN/rl_seed$s/best.pt"; [ -f "$B" ] || B="$RUN/rl_seed$s/model.pt"; MODELS="$MODELS $B"
done
for SUITE in field missions; do
  python scripts/evaluate.py --$SUITE --episodes $EVAL_EPISODES --models $MODELS --temps $EVAL_TEMPS \
    --workers "$CPUS" --out "$RUN/results" --tag $SUITE | tail -3
done
