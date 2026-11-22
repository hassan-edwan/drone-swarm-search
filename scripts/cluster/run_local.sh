#!/bin/bash
# The same pipeline on one large machine without a scheduler (e.g. a cloud VM).
# Seeds run one after another; rerun the script to continue after a stop
# (training resumes from its checkpoints; finished stages are skipped).
source "$(dirname "$0")/common.sh"
[ -f "$RUN/bc_sharp.pt" ] || bash -c "SLURM_CPUS_PER_TASK=$CPUS bash scripts/cluster/imitate.sbatch"
for s in $SEEDS; do
  OUT="$RUN/rl_seed$s"
  python scripts/train.py --field $FAMILIES --expand-share $EXPAND_SHARE --long-share $LONG_SHARE \
    --init "$RUN/bc_sharp.pt" --bc-anchor "$RUN/bc_sharp.pt" --bc-coef $BC_COEF --bc-anneal $BC_ANNEAL \
    --critic-warmup 30 --ent-coef $ENT_COEF --steps $RL_STEPS --envs $N_ENVS \
    --arch "$POLICY_ARCH" $PLAN --gamma $GAMMA --bc-floor $BC_FLOOR $REWARD_NORM \
    --val-every $VAL_EVERY --val-episodes $VAL_EPISODES \
    --workers $((CPUS > 8 ? CPUS - 4 : CPUS)) --threads 4 --device "${DEVICE:-cpu}" \
    --seed "$s" --ckpt-every $CKPT_EVERY --resume --out "$OUT" >> "$OUT.log" 2>&1
done
SLURM_CPUS_PER_TASK=$CPUS bash scripts/cluster/eval.sbatch
