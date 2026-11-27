#!/bin/bash
#SBATCH --job-name=swarm-targets
#SBATCH --cpus-per-task=128
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
# Retrain the target-choosing policy under the new defaults, then evaluate it
# (greedy and sampled) against the baselines, real sites included.
# Re-submitting resumes training from the last checkpoint.
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/scripts/cluster/common.sh"
source scripts/cluster/realism.env
OUT="$REAL_RUN/targets"
python scripts/train_targets.py --field wilderness urban --iters "$T_ITERS" --episodes "$T_EPISODES" \
  --workers $(( CPUS - 8 )) --threads 8 --val-every 20 --val-episodes "$T_VAL_EPISODES" --resume --out "$OUT"
M="$OUT/best.pt"; [ -f "$M" ] || M="$OUT/model.pt"
python scripts/evaluate.py --conditions $T_CONDITIONS --episodes "$T_EVAL_EPISODES" --models "$M" --temps 0 1 \
  --workers "$CPUS" --out "$OUT/results" --tag targets | tail -3
