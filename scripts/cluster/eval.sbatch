#!/bin/bash
#SBATCH --job-name=swarm-eval
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Baselines, the copy and every seed's RL model on the core conditions and on
# the mission suite (docs/MISSIONS.md), EVAL_EPISODES missions each.
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/common.sh"
MODELS="$RUN/bc.pt"
for s in $SEEDS; do MODELS="$MODELS $RUN/rl_seed$s/model.pt"; done
for SUITE in field missions; do
  python scripts/evaluate.py --$SUITE --episodes $EVAL_EPISODES --models $MODELS --temps 1 0 \
    --workers "$CPUS" --out "$RUN/results" --tag $SUITE | tail -3
done
