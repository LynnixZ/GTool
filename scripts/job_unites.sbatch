#!/bin/bash
# PART 2 -- the OFFLINE Slurm job. Submit from the login node:
#   sbatch scripts/job_unites.sbatch                       # default: mistral
#   MODEL=qwen3-8b sbatch scripts/job_unites.sbatch
#   SMOKE=1 MODEL=qwen2.5-0.5b sbatch -p a100 --gres=gpu:1 scripts/job_unites.sbatch
# Runs on a compute node (no internet) against the cache staged by PART 1.
#
# ⚠️ DEPLOY CHECKS (UNITES):
#   1. repo + venv + cache MUST live on shared NFS (/playpen-shared), NOT $HOME
#      (/home is login-node local and INVISIBLE to compute nodes).
#   2. shared dir name may DIFFER from $USER (e.g. dir 'xinyu' vs 'xinyuzh') ->
#      edit WORK_DIR (prep_env.sh/job_env.sh), REPO below, and --output/--error.
#   3. GTool graph build needs a GPU (SBERT on cuda:0) -> it runs inside this job.
#SBATCH --job-name=gtool
#SBATCH --partition=a100            # Ampere (A100) or Ada -- cu121 torch OK. AVOID Blackwell.
#SBATCH --gres=gpu:1                # 1 GPU fits frozen 7B/8B + small GNN; raise only if sharding.
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G                   # partial-GPU job -> concrete value. For gpu:8 use --mem=0
                                    # (whole node); NOTE 480G EXCEEDS a100/ada limit (~472GiB) -> rejected.
#SBATCH --time=1-00:00:00
#SBATCH --output=/playpen-shared/%u/tb_work/logs/gtool-%j.out   # edit %u if your dir != login name
#SBATCH --error=/playpen-shared/%u/tb_work/logs/gtool-%j.err
set -Eeuo pipefail

export REPO="${REPO:-/playpen-shared/$USER/GTool}"   # shared NFS copy (compute-node visible); NOT $HOME
cd "$REPO"

source scripts/job_env.sh                            # offline env + venv + GPUS

if [ "${SMOKE:-0}" = 1 ]; then
  bash run_experiment.sh --smoke "${MODEL:-qwen2.5-0.5b}"
else
  bash run_experiment.sh "${MODEL:-mistral}"
fi
