#!/usr/bin/env bash
# Train MAPPO and IPPO (several seeds each) in parallel, then evaluate.
# Each run is one process; the environment stepping is single-threaded, so
# running several runs side by side is what uses a many-core machine.
#
#   bash scripts/run_experiments.sh            # M1, 2M steps, seeds 0 1
#   STEPS=1000000 SEEDS="0" bash scripts/run_experiments.sh
#   OBSTACLES=0.3 TAG=m2 bash scripts/run_experiments.sh   # M2
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-2000000}
SEEDS=${SEEDS:-"0 1"}
OBSTACLES=${OBSTACLES:-0.0}
TAG=${TAG:-m1}
THREADS=${THREADS:-2}

mkdir -p runs
pids=()
for algo in mappo ippo; do
  for seed in $SEEDS; do
    out="runs/${TAG}_${algo}_s${seed}"
    echo "starting $out"
    python scripts/train.py --algo "$algo" --steps "$STEPS" --seed "$seed" \
      --obstacles "$OBSTACLES" --threads "$THREADS" --out "$out" > "$out.log" 2>&1 &
    pids+=($!)
  done
done
echo "logs: runs/${TAG}_*.log   (tail -f to watch)"
for p in "${pids[@]}"; do wait "$p"; done

first_seed=$(echo $SEEDS | awk '{print $1}')
python scripts/evaluate.py --episodes 100 --out "results/${TAG}" \
  --models "runs/${TAG}_mappo_s${first_seed}/model.pt" "runs/${TAG}_ippo_s${first_seed}/model.pt"
