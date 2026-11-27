#!/bin/bash
# Push the cluster's results (logs, tables, settings; no model weights) to GitHub, on their own branch,
# without touching the checkouts that jobs are running from.
#
#   bash scripts/cluster/push_results.sh                    # both checkouts (see DIRS below)
#   bash scripts/cluster/push_results.sh ~/some-checkout    # just these
#   BRANCH=hpc-results MAX_MB=5 bash scripts/cluster/push_results.sh
#
# Files land on branch $BRANCH under hpc/<checkout name>/..., e.g.
#   hpc/drone-swarm-search/runs/realism/targets/val_log.csv
#   hpc/drone-swarm-search-p4/runs/realism/pbt/m03/val_log.csv
# Each run adds one commit with whatever changed. Safe to run while jobs are running and as often as you like
# (e.g. from cron: 0 */2 * * * bash ~/drone-swarm-search/scripts/cluster/push_results.sh >/dev/null 2>&1).
set -euo pipefail
BRANCH=${BRANCH:-hpc-results}
MAX_MB=${MAX_MB:-5}                 # skip any file bigger than this (GitHub refuses > 100 MB)
DIRS=("$@")
[ ${#DIRS[@]} -eq 0 ] && DIRS=("$HOME/drone-swarm-search" "$HOME/drone-swarm-search-p4")
REPO=$(cd "${DIRS[0]}" && git rev-parse --show-toplevel)
WT=$(mktemp -d "${TMPDIR:-/tmp}/push-results.XXXXXX")
cleanup() { git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"; git -C "$REPO" branch -qD tmp-results 2>/dev/null || true; }
trap cleanup EXIT

# a separate worktree on the results branch (made from nothing the first time)
if git -C "$REPO" ls-remote -q --exit-code origin "refs/heads/$BRANCH" >/dev/null 2>&1; then
  git -C "$REPO" fetch -q origin "$BRANCH"
  git -C "$REPO" worktree add -q --detach "$WT" FETCH_HEAD
else
  git -C "$REPO" worktree add -q --detach "$WT"
  git -C "$WT" checkout -q --orphan tmp-results && git -C "$WT" rm -rq --cached . && find "$WT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
fi

n=0
for D in "${DIRS[@]}"; do
  [ -d "$D" ] || { echo "skip $D (not found)"; continue; }
  name=$(basename "$D")
  # results worth keeping: tables, logs, settings, figures; not weights or caches
  while IFS= read -r -d '' f; do
    rel=${f#"$D"/}
    mkdir -p "$WT/hpc/$name/$(dirname "$rel")"
    cp -p "$f" "$WT/hpc/$name/$rel"; n=$((n + 1))
  done < <(cd "$D" && find runs logs results -type f -size -"${MAX_MB}"M \
             \( -name '*.csv' -o -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.txt' \
                -o -name '*.out' -o -name '*.err' -o -name '*.log' -o -name '*.png' -o -name '*.svg' \) \
             -print0 2>/dev/null | sed -z "s|^|$D/|")
  # where each checkout's code was, so results can be matched to it
  git -C "$D" log -1 --format="%H %cd %s" > "$WT/hpc/$name/CODE_VERSION.txt" 2>/dev/null || true
done

cd "$WT"
git add -A hpc
if git diff --cached --quiet; then echo "nothing new ($n files checked)"; exit 0; fi
git commit -qm "cluster results $(date '+%F %H:%M') ($(hostname -s))"
git push -q origin "HEAD:refs/heads/$BRANCH"
echo "pushed $(git diff --stat HEAD~1 2>/dev/null | tail -1 || echo "$n files") to branch $BRANCH"
