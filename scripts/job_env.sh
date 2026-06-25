#!/usr/bin/env bash
# SOURCE me from inside the sbatch job (or a China offline shell): PART 2 OFFLINE env.
#   source scripts/job_env.sh
# Reads ONLY the cache that PART 1 (prep_env + prestage_all) staged. WORK_DIR must
# match PART 1 exactly (and your shared dir may differ from $USER -> set it here).
# GTool does NOT use wandb, so no WANDB_* needed.

export WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"   # China: /root/autodl-tmp/tb_work
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
source "$WORK_DIR/venv/bin/activate"

# Cache-only: never touch the network. (Do NOT source setup_US.sh here -- it UNSETs these.)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Use exactly the GPUs Slurm pinned to this job (they appear as 0..N-1 inside it).
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
export GPUS="${GPUS:-$(seq -s' ' 0 $((NGPU-1)))}"

echo "[job_env] $(hostname)  GPUS='$GPUS'  WORK_DIR=$WORK_DIR  (offline cache)"
