#!/bin/bash
#SBATCH --job-name=swarm-smoke
#SBATCH --cpus-per-task=16
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x-%j.out
# Before booking the full run: the tests, the whole pipeline at tiny settings,
# then the training speed on 16 cores (steps per second, for sizing).
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/scripts/cluster/common.sh"
python -m pytest -q
RUN=runs/smoke IMITATE_EPISODES=8 BETAS="1.0 0.0" RL_STEPS=8192 N_ENVS=4 SEEDS=0 EVAL_EPISODES=2 VAL_EVERY=0 \
  bash scripts/cluster/run_local.sh
ls runs/smoke/results
python scripts/train.py --field $FAMILIES --arch "$POLICY_ARCH" $PLAN --steps ${BENCH_STEPS:-300000} --envs 48 --workers 12 --threads 4 \
  --out runs/smoke/bench > runs/smoke/bench.log 2>&1
python - <<'PY'
import csv
r = list(csv.DictReader(open("runs/smoke/bench/train_log.csv")))[-1]
steps, secs = float(r["env_steps"]), float(r["secs"])
print(f"speed on 16 cores: {steps / secs:.0f} steps/s")
PY
