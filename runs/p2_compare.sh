#!/bin/bash
# Same data budget for four imitation set-ups, then each copy flown alone (greedy) against the rule.
cd /home/user/drone-swarm-search
export OMP_NUM_THREADS=1
OUT=runs/p2_compare; mkdir -p $OUT
C="--field wilderness urban --episodes 60 --betas 1.0 0.5 0.0 --workers 4 --epochs 4"
python scripts/imitate.py $C --hard-labels --out $OUT/A_mlp_hard.pt > $OUT/A.log 2>&1
python scripts/imitate.py $C --out $OUT/B_mlp_sets.pt > $OUT/B.log 2>&1
python scripts/imitate.py $C --plan-feats --out $OUT/C_mlp_plan.pt > $OUT/C.log 2>&1
python scripts/imitate.py $C --plan-feats --arch cnn --every 2 --out $OUT/D_cnn_plan.pt > $OUT/D.log 2>&1
python scripts/evaluate.py --conditions wild_mix wild_drones_only urban_mix --episodes 40 --workers 4 \
  --models $OUT/A_mlp_hard.pt $OUT/B_mlp_sets.pt $OUT/C_mlp_plan.pt $OUT/D_cnn_plan.pt --temps 0 \
  --out $OUT --tag compare > $OUT/eval.log 2>&1
touch $OUT/done
