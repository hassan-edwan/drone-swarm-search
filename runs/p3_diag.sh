#!/bin/bash
# Diagnostic: planning features from travel distance (the rule's own score) vs straight-line.
cd /home/user/drone-swarm-search
export OMP_NUM_THREADS=1
OUT=runs/p3_diag; mkdir -p $OUT
python scripts/imitate.py --field wilderness urban --episodes 60 --betas 1.0 0.5 0.0 --workers 4 --epochs 4 \
  --plan-path --out $OUT/E_mlp_path.pt > $OUT/E.log 2>&1
python scripts/evaluate.py --conditions wild_mix urban_mix flood_mix_heldout --episodes 100 --workers 4 \
  --models runs/p2_compare/C_mlp_plan.pt $OUT/E_mlp_path.pt --temps 1 0 --out $OUT --tag diag > $OUT/eval.log 2>&1
touch $OUT/done
