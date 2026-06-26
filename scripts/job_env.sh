#!/usr/bin/env bash
# SOURCE me from inside the sbatch job (or a China offline shell): PART 2 OFFLINE env.
#   source scripts/job_env.sh
# Reads ONLY the cache that PART 1 (prep_env + prestage_all) staged. WORK_DIR must
# match PART 1 exactly (and your shared dir may differ from $USER -> set it here).
# GTool does NOT use wandb, so no WANDB_* needed.

export WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"   # China: /root/autodl-tmp/tb_work
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"

# Fail loudly if WORK_DIR is wrong (e.g. shared dir != $USER), rather than a cryptic
# "source: no such file" from activate. The venv must have been built by PART 1.
if [ ! -f "$WORK_DIR/venv/bin/activate" ]; then
  echo "[job_env] ERROR: no venv at $WORK_DIR/venv -- WORK_DIR wrong (shared dir != \$USER?) or PART 1 not run." >&2
  return 1 2>/dev/null || exit 1
fi
source "$WORK_DIR/venv/bin/activate"

# Cache-only: never touch the network. Offline != auto-skip -- HF/transformers will TRY
# to reach the hub (even to freshness-check the cache) and hang/fail offline unless these
# are set. (Do NOT source setup_US.sh here -- it UNSETs these.) GTool uses no wandb.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Use exactly the GPUs Slurm pinned to this job (they appear as 0..N-1 inside it).
# Guard NGPU=0: on a no-GPU box `seq 0 -1` is empty on GNU but prints "0 -1" on BSD/macOS,
# which would silently export a garbage GPUS. PART 2 needs a GPU -> warn loudly if none.
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [ "${NGPU:-0}" -ge 1 ]; then
  export GPUS="${GPUS:-$(seq -s' ' 0 $((NGPU-1)))}"
else
  export GPUS="${GPUS:-}"
  echo "[job_env] WARN: no GPU detected (NGPU=0). PART 2 (build graphs + train) NEEDS a GPU." >&2
  echo "[job_env]       China: reboot into a GPU instance.  US: this should run inside the sbatch job." >&2
fi

echo "[job_env] $(hostname)  GPUS='$GPUS'  WORK_DIR=$WORK_DIR  (offline cache)"
