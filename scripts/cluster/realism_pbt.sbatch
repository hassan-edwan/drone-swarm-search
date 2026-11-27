#!/bin/bash
#SBATCH --job-name=swarm-pbt
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
# Population-based tuning of the target policy (scripts/pbt.py): one array task per member. Each trains
# in segments of PBT_SEG iterations; between segments a member in the bottom quarter (paired validation
# against the rule) copies a top-quarter member's weights and perturbs its settings. Members read each
# other's files, so they must share the output folder. Re-submitting resumes every member.
#   sbatch -A <account> scripts/cluster/realism_pbt.sbatch          # 8 members (the array size)
#   python scripts/pbt.py report --out runs/realism/pbt              # the population, any time
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/scripts/cluster/common.sh"
source scripts/cluster/realism.env
M=${SLURM_ARRAY_TASK_ID:-0}
OUT="$REAL_RUN/pbt"
python scripts/pbt.py init --pop "$PBT_POP" --out "$OUT"
D=$(printf "%s/m%02d" "$OUT" "$M")
for (( s = 1; s <= PBT_SEGMENTS; s++ )); do
  python scripts/pbt.py step --member "$M" --out "$OUT"
  python scripts/train_targets.py --field wilderness urban --iters $(( s * PBT_SEG )) --episodes $(( CPUS - 8 )) \
    --workers $(( CPUS - 8 )) --threads 8 --val-every "$PBT_SEG" --val-episodes "$T_VAL_EPISODES" --seed "$M" \
    $(python scripts/pbt.py args --member "$M" --out "$OUT") --resume --out "$D"
done
python scripts/pbt.py report --out "$OUT"
