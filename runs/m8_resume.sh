#!/bin/bash
# Overnight chain (after the camera run and the field baseline evaluation):
# resume: copy the rule on the field environment, sharpen it, RL fine-tune with a KL
# pull that stays on, evaluate on all field conditions (flood held out), render.
cd /home/user/drone-swarm-search
LOG=runs/m8_chain.log
exec >> $LOG 2>&1
commit() {  # $1 = message, rest = paths
  msg="$1"; shift
  git add -f "$@" && git -c user.name="Hassan Edwan" -c user.email="hameedubaid65@gmail.com" commit -q -m "$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LJpntNMZaWkqDVmVzFxJsU"
  for i in 1 2 3 4; do git push -q origin main && break; sleep $((i * 20)); done
}
echo "$(date) RL fine-tune resumed from update ~1550 (container restart)"
python scripts/train.py --field wilderness urban --init runs/field_rl_part1_model.pt --bc-anchor models/bc_field_sharp.pt \
  --bc-coef 0.3 --bc-anneal 1000000 --ent-coef 0.001 --steps 2400000 --seed 1 --out runs/field_rl2 > runs/m8_field_train2.log 2>&1
cp runs/field_rl2/model.pt models/m8_field_rl_4m.pt
cat runs/m8_field_train.log runs/m8_field_train2.log > models/m8_field_rl_4m_train.log
cp runs/field_rl_part1_log.csv models/m8_field_rl_4m_part1.csv; cp runs/field_rl2/train_log.csv models/m8_field_rl_4m_part2.csv
commit "M8: RL fine-tune of the field copy (4M steps, KL pull kept on)" models/m8_field_rl_4m.pt models/m8_field_rl_4m_train.log models/m8_field_rl_4m_part1.csv models/m8_field_rl_4m_part2.csv

echo "$(date) evaluation"
export OMP_NUM_THREADS=1
E="python scripts/evaluate.py --field --episodes 50 --models models/bc_field.pt models/m8_field_rl_4m.pt --temps 1 0 --out results/m8"
$E --conditions wild_mix urban_mix flood_mix_heldout --tag learned_a > runs/m8_lev_a.log 2>&1 &
$E --conditions wild_mix_adaptive_map wild_hard_unresponsive wild_hard_evasive --tag learned_b > runs/m8_lev_b.log 2>&1 &
$E --conditions wild_hard_wrong_ipp wild_hard_outside wild_hard_weak_sensor --tag learned_c > runs/m8_lev_c.log 2>&1 &
$E --conditions wild_hard_late_start wild_hard_radio wild_hard_off_model --tag learned_d > runs/m8_lev_d.log 2>&1 &
wait
python scripts/render_field.py --condition wild_mix --seed 4 --every 1 --fps 10 --model models/m8_field_rl_4m.pt --out results/figures/field_learned.mp4 > /dev/null 2>&1
ffmpeg -loglevel error -y -sseof -3 -i results/figures/field_learned.mp4 -frames:v 1 -update 1 results/figures/field_learned_frame.png
commit "M8: field evaluation of the learned models (copy, copy + RL) and a mission video" results/m8 results/figures/field_learned.mp4 results/figures/field_learned_frame.png results/figures/field_learned.txt
echo "$(date) done"
touch runs/m8_chain.done
