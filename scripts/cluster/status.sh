#!/bin/bash
# Check on the cluster jobs: queue, recent job states, progress of each realism job,
# the tail of each job's latest log and any errors in it.
#   bash scripts/cluster/status.sh          # everything
#   bash scripts/cluster/status.sh 40       # show 40 log lines per job instead of 8
#   watch -n 60 bash scripts/cluster/status.sh   # refresh every minute
N=${1:-8}
source "$(dirname "$0")/phase2.env"
source "$(dirname "$0")/realism.env"
cd "$SWARM_DIR" || exit 1
hr() { printf '\n=== %s ===\n' "$1"; }

hr "queue (squeue --me)"
squeue --me -o "%.10i %.18j %.8T %.10M %.10l %.5C %R" 2>/dev/null || echo "squeue not available"

hr "jobs since yesterday (sacct)"
sacct -X -S "$(date -d yesterday +%F)" --name=swarm-realism,swarm-sitl,swarm-targets,swarm-surprises,swarm-train,swarm-imitate,swarm-eval,swarm-smoke \
  -o JobID%14,JobName%16,State%12,Elapsed,ExitCode 2>/dev/null || echo "sacct not available"

hr "progress"
# 1. realism evaluation: writes its table only at the end
if [ -f "$REAL_RUN/eval/results.md" ]; then echo "realism_eval:      DONE -> $REAL_RUN/eval/results.md"
else echo "realism_eval:      running or not started (table appears at the end)"; fi

# 2. ArduPilot flights: one report per seed x wind x GPS mode
count_seeds() { local n=0 a b; for p in ${1//,/ }; do
  if [[ $p == *-* ]]; then a=${p%-*}; b=${p#*-}; n=$((n + b - a + 1)); else n=$((n + 1)); fi; done; echo $n; }
S=$(count_seeds "$SITL_SEEDS"); W=$(wc -w <<< "$SITL_WINDS")
TOTAL=$((S * W * 2))
DONE=$(ls "$REAL_RUN"/sitl/sitl_seed*_wind*_gps*.json 2>/dev/null | wc -l)
PLANS=$(ls "$REAL_RUN"/sitl/flight_seed*.json 2>/dev/null | wc -l)
printf "realism_sitl:      plans %d/%d, flights %d/%d (%d%%)%s\n" "$PLANS" "$S" "$DONE" "$TOTAL" $((100 * DONE / TOTAL)) \
  "$([ -f "$REAL_RUN/sitl/summary.md" ] && echo '  DONE')"

# 3. target policy: training iterations, best validation, then the evaluation
TL="$REAL_RUN/targets/train_log.csv"; VL="$REAL_RUN/targets/val_log.csv"
if [ -f "$TL" ]; then
  IT=$(tail -1 "$TL" | cut -d, -f1)
  printf "realism_targets:   iteration %s/%s (%d%%)\n" "$IT" "$T_ITERS" $((100 * ${IT:-0} / T_ITERS)) 2>/dev/null
  echo "                   last train row: $(tail -1 "$TL")"
  [ -f "$VL" ] && echo "                   validation ($(head -1 "$VL")): $(tail -1 "$VL")"
  ls "$REAL_RUN"/targets/results/*.md >/dev/null 2>&1 && echo "                   evaluation DONE -> $REAL_RUN/targets/results"
else echo "realism_targets:   not started"; fi

# 4. map vs real world: one 'done' line per variant x world (8 variants)
SL=$(ls -t logs/swarm-surprises-*.out 2>/dev/null | head -1)
if [ -f "$REAL_RUN/surprises/results.md" ]; then echo "realism_surprises: DONE -> $REAL_RUN/surprises/results.md"
elif [ -n "$SL" ]; then
  D=$(grep -c " done (" "$SL"); T=$((SURPRISE_WORLDS * 8))
  printf "realism_surprises: %d/%d variant-worlds (%d%%)\n" "$D" "$T" $((100 * D / T))
else echo "realism_surprises: not started"; fi

for J in swarm-realism swarm-sitl swarm-targets swarm-surprises swarm-imitate swarm-train swarm-eval swarm-smoke; do
  L=$(ls -t logs/$J-*.out 2>/dev/null | head -1)
  [ -z "$L" ] && continue
  hr "$L  (updated $(date -r "$L" '+%F %H:%M'))"
  tail -n "$N" "$L"
  E=$(grep -inE "error|traceback|killed|oom|cancelled|time limit" "$L" | tail -3)
  [ -n "$E" ] && printf -- "--- possible problems:\n%s\n" "$E"
done
echo
