#!/bin/bash
#SBATCH --job-name=swarm-train
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
# RL fine-tune, one seed per array task. Checkpoints every CKPT_EVERY updates;
# when the time limit or a preemption nears, the job requeues itself and
# continues from the checkpoint (--resume). Add --gres=gpu:1 and DEVICE=cuda
# to put the network on a GPU (the missions stay on the CPUs).
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/common.sh"
SEEDS_ARR=($SEEDS)
SEED=${SEEDS_ARR[${SLURM_ARRAY_TASK_ID:-0}]}
OUT="$RUN/rl_seed$SEED"
LEARNER_THREADS=4
trap 'echo "requeue at $(date)"; kill -TERM $PID; wait $PID; scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' USR1
python scripts/train.py --field $FAMILIES --expand-share $EXPAND_SHARE --long-share $LONG_SHARE \
  --init "$RUN/bc_sharp.pt" --bc-anchor "$RUN/bc_sharp.pt" --bc-coef $BC_COEF --bc-anneal $BC_ANNEAL \
  --critic-warmup 30 --ent-coef $ENT_COEF --steps $RL_STEPS --envs $N_ENVS \
  --workers $((CPUS - LEARNER_THREADS)) --threads $LEARNER_THREADS --device "${DEVICE:-cpu}" \
  --seed "$SEED" --ckpt-every $CKPT_EVERY --resume --out "$OUT" >> "$OUT.log" 2>&1 &
PID=$!
wait $PID
