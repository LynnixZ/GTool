#!/usr/bin/env bash
# Submit ONE offline Slurm job that runs the full GTool sweep ON ONE NODE.
# #SBATCH lines are STATIC (Slurm parses them before any shell runs), so we pass the
# resource request + log paths on the sbatch COMMAND LINE here, and --export=ALL carries
# WORK_DIR / REPO / MODELS / MODEL / SMOKE into the job. The job (job_unites.sbatch) runs
# run_grid.sh, which distributes the (model x domain) runs across the GPUs this job got.
#
# Usage:
#   GPUS_PER_JOB=2 bash scripts/submit_unites.sh                  # full 3x3 sweep on 2 GPUs
#   MODELS="mistral" GPUS_PER_JOB=1 bash scripts/submit_unites.sh # one model x 3 domains
#   SMOKE=1 MODEL=qwen2.5-0.5b GPUS_PER_JOB=1 bash scripts/submit_unites.sh
set -e

# ===== EDIT: how many GPUs this node/job gets (China nodes often 1-2; <= your cap) =====
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
# =======================================================================================
# cu121 torch runs on Ampere/Ada (a100/a6000/ada) but NOT Blackwell ("no kernel image").
# Submit to all compatible partitions -> Slurm picks whichever frees up first.
PARTITION="${PARTITION:-a100,ada,a6000}"
CPUS="${CPUS:-$(( GPUS_PER_JOB * 12 ))}"
MEM="${MEM:-$(( GPUS_PER_JOB * 50 ))G}"           # ~50G/GPU; stays under a node's ~472GiB limit

WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"
mkdir -p "$WORK_DIR/logs"

echo "[submit] partition=$PARTITION  gpus=$GPUS_PER_JOB  cpus=$CPUS  mem=$MEM"
echo "[submit] WORK_DIR=$WORK_DIR  REPO=${REPO:-(\$USER default)}  MODELS='${MODELS:-vicuna mistral qwen3-8b}'  SMOKE=${SMOKE:-0}"

sbatch -J gtool -p "$PARTITION" --gres=gpu:"$GPUS_PER_JOB" --cpus-per-task="$CPUS" --mem="$MEM" \
  --output="$WORK_DIR/logs/gtool-%j.out" --error="$WORK_DIR/logs/gtool-%j.err" \
  --export=ALL scripts/job_unites.sbatch
