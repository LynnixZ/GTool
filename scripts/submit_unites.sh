#!/usr/bin/env bash
# Submit GTool's offline Slurm job, choosing GPU count (and carrying MODEL/SMOKE).
#
# Why a wrapper: #SBATCH lines are STATIC -- Slurm parses them before any shell runs,
# so they can't read a shell variable. We pass the resource request + log paths on the
# sbatch COMMAND LINE here (overrides the #SBATCH defaults), and --export=ALL carries
# WORK_DIR / REPO / MODEL / SMOKE into the job. Log paths follow WORK_DIR, so they land
# in YOUR shared dir even when it != $USER (the static --output=%u default would be wrong).
#
# Usage:
#   GPUS_PER_JOB=1 MODEL=mistral bash scripts/submit_unites.sh
#   SMOKE=1 MODEL=qwen2.5-0.5b GPUS_PER_JOB=1 bash scripts/submit_unites.sh
set -e

# ===== EDIT: how many GPUs (GTool: 1 fits a frozen 7B/8B + GNN; <= your per-user cap) =====
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
# ==========================================================================================
# cu121 torch runs on Ampere/Ada (a100/a6000/ada) but NOT Blackwell ("no kernel image").
# Submit to all compatible partitions -> Slurm picks whichever frees up first, never blackwell.
PARTITION="${PARTITION:-a100,ada,a6000}"
CPUS="${CPUS:-$(( GPUS_PER_JOB * 12 ))}"
MEM="${MEM:-$(( GPUS_PER_JOB * 50 ))G}"           # ~50G/GPU; stays under a node's ~472GiB limit

WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"
mkdir -p "$WORK_DIR/logs"

echo "[submit] partition=$PARTITION  gpus=$GPUS_PER_JOB  cpus=$CPUS  mem=$MEM"
echo "[submit] WORK_DIR=$WORK_DIR  REPO=${REPO:-(\$USER default)}  MODEL=${MODEL:-mistral}  SMOKE=${SMOKE:-0}"

sbatch -p "$PARTITION" --gres=gpu:"$GPUS_PER_JOB" --cpus-per-task="$CPUS" --mem="$MEM" \
  --output="$WORK_DIR/logs/gtool-%j.out" --error="$WORK_DIR/logs/gtool-%j.err" \
  --export=ALL scripts/job_unites.sbatch
