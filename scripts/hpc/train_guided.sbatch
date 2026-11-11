#!/bin/bash
# One LLM-guided MAPPO training run on a GPU node (written for UMD Zaratan).
#
# One-time setup on a login node (compute nodes may have no internet):
#   python -m venv ~/venvs/swarm && source ~/venvs/swarm/bin/activate
#   pip install -r requirements.txt -r requirements-llm.txt
#   export HF_HOME=$SCRATCH/hf            # model cache on scratch, not home
#   hf download Qwen/Qwen2.5-7B-Instruct  # ~15 GB, once
#
# Then, from the repo root:  sbatch scripts/hpc/train_guided.sbatch
#   MODEL=Qwen/Qwen2.5-3B-Instruct sbatch ...   # smaller / faster guide
#   STEPS=5000000 sbatch ...
#
# The LLM runs on the GPU, batched over all envs; env stepping and the small
# policy run on the CPUs. More envs = bigger LLM batches = faster.
#
#SBATCH --job-name=swarm-guided
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=runs/hpc/slurm_%j.out

set -euo pipefail
source ~/venvs/swarm/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/hf}
export HF_HUB_OFFLINE=1

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
STEPS=${STEPS:-2000000}
ARCH=${ARCH:-mlp}
ENVS=${ENVS:-32}
TAG=${TAG:-m1_guided}
out="runs/hpc/${TAG}"
mkdir -p runs/hpc

python scripts/train.py --algo mappo --arch "$ARCH" --steps "$STEPS" --envs "$ENVS" \
  --guide "llm:$MODEL" --guide-device cuda --device cpu --threads "$SLURM_CPUS_PER_TASK" \
  --out "$out" > "$out.log" 2>&1

# Evaluate against the lawnmower baseline on the held-out scenarios.
python scripts/evaluate.py --models "$out/model.pt" --guide-device cuda --episodes 100 \
  --out "results/${TAG}" >> "$out.log" 2>&1
