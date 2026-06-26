#!/bin/bash
# PART 2 -- the OFFLINE Slurm job. Submit from the login node:
#   sbatch scripts/job_unites.sbatch                       # default: mistral
#   MODEL=qwen3-8b sbatch scripts/job_unites.sbatch
#   SMOKE=1 MODEL=qwen2.5-0.5b sbatch -p a100 --gres=gpu:1 scripts/job_unites.sbatch
# Runs on a compute node (no internet) against the cache staged by PART 1.
#
# ⚠️ DEPLOY CHECKS (UNITES) -- READ BEFORE FIRST SUBMIT:
#   1. repo + venv + cache MUST live on shared NFS (/playpen-shared), NOT $HOME
#      (/home is login-node local and INVISIBLE to compute nodes).
#   2. shared dir name may DIFFER from $USER (e.g. dir 'xinyu' vs login 'xinyuzh').
#      If so, set ONE base in 3 places that can't share a variable:
#        - the two #SBATCH --output/--error lines below (replace %u with the real dir),
#        - REPO + WORK_DIR exported below (or pass them on the CLI / in prep_env.sh).
#      Submit-with-real-paths example (no file edits needed):
#        WORK_DIR=/playpen-shared/xinyu/tb_work REPO=/playpen-shared/xinyu/GTool \
#          sbatch --output=/playpen-shared/xinyu/tb_work/logs/gtool-%j.out \
#                 --error=/playpen-shared/xinyu/tb_work/logs/gtool-%j.err \
#                 scripts/job_unites.sbatch
#   3. GTool graph build needs a GPU (SBERT on cuda:0) -> it runs inside this job.
#   4. To vary GPU count, pass it on the CLI (overrides the static directives below):
#        sbatch --gres=gpu:4 --cpus-per-task=48 --mem=200G scripts/job_unites.sbatch
#SBATCH --job-name=gtool
#SBATCH --partition=a100            # Ampere (A100) or Ada -- cu121 torch OK. AVOID Blackwell (cu121 -> "no kernel image").
#SBATCH --gres=gpu:1                # 1 GPU fits frozen 7B/8B + small GNN; raise only if sharding.
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G                   # partial-GPU job -> concrete value (~50G/GPU). For a WHOLE node use --mem=0;
                                    # NOTE a fixed 480G EXCEEDS a typical a100/ada node (~472GiB) -> rejected.
#SBATCH --time=1-00:00:00
#SBATCH --output=/playpen-shared/%u/tb_work/logs/gtool-%j.out   # edit %u if your shared dir != login name
#SBATCH --error=/playpen-shared/%u/tb_work/logs/gtool-%j.err
set -Eeuo pipefail

# These must match PART 1 (prep_env.sh) exactly. WORK_DIR feeds job_env.sh (venv + cache).
export WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"   # shared NFS, NOT $HOME
export REPO="${REPO:-/playpen-shared/$USER/GTool}"             # shared NFS copy (compute-node visible)
cd "$REPO"

source scripts/job_env.sh                            # offline env + venv + GPUS

if [ "${SMOKE:-0}" = 1 ]; then
  bash run_experiment.sh --smoke "${MODEL:-qwen2.5-0.5b}"
else
  bash run_experiment.sh "${MODEL:-mistral}"
fi
