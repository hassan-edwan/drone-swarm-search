#!/bin/bash
#SBATCH --job-name=swarm-imitate
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Copy the rule (DAgger) on the field missions, then sharpen the copy.
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/common.sh"
python scripts/imitate.py --field $FAMILIES --expand-share $EXPAND_SHARE --long-share $LONG_SHARE \
  --episodes $IMITATE_EPISODES --betas $BETAS --workers "$CPUS" --arch $ARCH $PLAN --every $IMITATE_EVERY \
  --out "$RUN/bc.pt" | tee "$RUN/bc_imitate.log"
python scripts/sharpen.py "$RUN/bc.pt" "$RUN/bc_sharp.pt" --factor 4
