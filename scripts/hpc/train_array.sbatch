#!/bin/bash
# Slurm job array: every (algo, arch, seed) combination as its own job.
# Written for UMD Zaratan; works on any Slurm cluster.
#
#   1. Set --account below (your Zaratan allocation, e.g. from `sbalance`).
#   2. One-time setup on a login node:
#        python -m venv ~/venvs/swarm && source ~/venvs/swarm/bin/activate
#        pip install -r requirements.txt
#   3. Submit from the repo root:
#        sbatch scripts/hpc/train_array.sbatch
#        STEPS=5000000 sbatch scripts/hpc/train_array.sbatch     # longer runs
#   4. Watch: squeue -u $USER ; tail -f runs/hpc/*.log
#
# 2 algos x 3 archs x 3 seeds = 18 jobs. Each job is single-process: env
# stepping is CPU-bound and the networks are small, so many CPU jobs in
# parallel beat one GPU job. For a GPU job instead, use --partition=gpu,
# --gres=gpu:1 and add "--device cuda" to the train command.
#
#SBATCH --job-name=swarm
#SBATCH --account=CHANGE_ME
#SBATCH --array=0-17
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=runs/hpc/slurm_%A_%a.out

set -euo pipefail
source ~/venvs/swarm/bin/activate

ALGOS=(mappo ippo)
ARCHS=(mlp cnn transformer)
SEEDS=(0 1 2)
STEPS=${STEPS:-2000000}
OBSTACLES=${OBSTACLES:-0.0}
TAG=${TAG:-m1}

i=$SLURM_ARRAY_TASK_ID
algo=${ALGOS[$(( i / 9 ))]}
arch=${ARCHS[$(( (i / 3) % 3 ))]}
seed=${SEEDS[$(( i % 3 ))]}
out="runs/hpc/${TAG}_${algo}_${arch}_s${seed}"
mkdir -p runs/hpc

echo "$out on $(hostname)"
python scripts/train.py --algo "$algo" --arch "$arch" --seed "$seed" --steps "$STEPS" \
  --obstacles "$OBSTACLES" --threads "$SLURM_CPUS_PER_TASK" --out "$out" > "$out.log" 2>&1
