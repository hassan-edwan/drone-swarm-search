#!/bin/bash
# Phase 3 comparison on 4 cores: the rollout planner, the target policy trained by RL
# from the rule, and the target policy trained to imitate the planner.
cd /home/user/drone-swarm-search
export OMP_NUM_THREADS=1
OUT=runs/p3; mkdir -p $OUT
C="--conditions wild_mix urban_mix flood_mix_heldout wild_hard_radio --episodes 100 --workers 4"
[ -f $OUT/planner_eval.done ] || { python scripts/evaluate.py $C --planner --out $OUT --tag planner > $OUT/planner_eval.log 2>&1 && touch $OUT/planner_eval.done; }
[ -f $OUT/planner_bc.done ] || { python scripts/imitate_planner.py --field wilderness urban --episodes 40 --betas 1.0 0.5 \
    --workers 4 --out $OUT/planner_bc.pt > $OUT/planner_bc.log 2>&1 && touch $OUT/planner_bc.done; }
[ -f $OUT/rl.done ] || { python scripts/train_targets.py --field wilderness urban --iters 60 --episodes 16 --workers 4 \
    --val-every 10 --val-episodes 8 --resume --out $OUT/rl > $OUT/rl.log 2>&1 && touch $OUT/rl.done; }
[ -f $OUT/models_eval.done ] || { python scripts/evaluate.py $C --no-baselines --temps 1 0 \
    --models $OUT/rl/best.pt $OUT/planner_bc.pt --out $OUT --tag models > $OUT/models_eval.log 2>&1 && touch $OUT/models_eval.done; }
touch $OUT/done
