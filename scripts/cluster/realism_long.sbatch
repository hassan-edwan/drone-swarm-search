#!/bin/bash
#SBATCH --job-name=swarm-long
#SBATCH --cpus-per-task=128
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
# A longer training run of the target policy, starting from a population-based-tuning member (default:
# member 4, the one ahead of the rule in every condition) with that member's settings; then a paired
# comparison with the rule and the starting model on the same missions, real sites included.
# Re-submitting resumes training from the last checkpoint (and runs the comparison once it is done).
#   sbatch -A <account> scripts/cluster/realism_long.sbatch
#   FROM=runs/realism/pbt/m06 L_ITERS=600 sbatch -A <account> scripts/cluster/realism_long.sbatch
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/scripts/cluster/common.sh"
source scripts/cluster/realism.env
FROM=${FROM:-$REAL_RUN/pbt/m04}        # the member to start from (its model.pt and hp.json)
L_ITERS=${L_ITERS:-400}                # iterations of this run
L_LR_SCALE=${L_LR_SCALE:-0.75}         # a slightly lower learning rate for the longer run
OUT=${L_OUT:-$REAL_RUN/long_$(basename "$FROM")}
[ -f "$FROM/model.pt" ] && [ -f "$FROM/hp.json" ] || { echo "ERROR: no model.pt / hp.json in $FROM" >&2; exit 1; }
mkdir -p "$OUT"
[ -f "$OUT/start.pt" ] || cp "$FROM/model.pt" "$OUT/start.pt"   # keep the starting weights (PBT may overwrite FROM)
[ -f "$OUT/hp.json" ] || cp "$FROM/hp.json" "$OUT/hp.json"
HP=$(python - "$OUT/hp.json" "$L_LR_SCALE" <<'EOF'
import json, sys
hp = json.load(open(sys.argv[1]))
hp["lr"] = hp["lr"] * float(sys.argv[2])
print(" ".join(f"--{k.replace('_', '-')} {v}" for k, v in hp.items()))
EOF
)
echo "start: $FROM  settings: $HP  out: $OUT"

# Training: missions per update = one per worker; validation (paired with the rule) every 20 iterations;
# best.pt is the best validation so far, starting with the starting model itself (validated at iteration 0).
python scripts/train_targets.py --field wilderness urban --init "$OUT/start.pt" --iters "$L_ITERS" \
  --episodes "$T_EPISODES" --workers $(( CPUS - 8 )) --threads 8 --val-every 20 --val-episodes "$T_VAL_EPISODES" \
  $HP --resume --out "$OUT"

# Comparison on the same 300 missions per condition: the rule, the starting model, the best and last of this run.
python scripts/paired_eval.py --methods rule "targets:$OUT/start.pt" "targets:$OUT/best.pt" "targets:$OUT/model.pt" \
  --conditions $T_CONDITIONS --episodes "$T_EVAL_EPISODES" --workers "$CPUS" --out "$OUT/compare.json"
