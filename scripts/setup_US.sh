#!/usr/bin/env bash
# Convenience env for a US/EU server -- official sources, NO mirrors, NO secrets.
#   source scripts/setup_US.sh
# Values respect anything you already exported (override per box).

# ⚠️ On a Slurm cluster (e.g. UNITES) OVERRIDE this to shared NFS (/playpen-shared/<dir>)
# BEFORE sourcing -- $HOME is login-node local and INVISIBLE to compute nodes. prep_env.sh
# already does this; only the bare-server fallback uses $HOME.
export WORK_DIR="${WORK_DIR:-$HOME/tb_work}"        # override to a big/scratch/shared disk
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
# Official cu121 wheels (Ampere/Ada OK; avoid Blackwell which needs cu128+).
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"        # parallel Xet (newer models)
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"   # parallel classic-LFS

# Clear any China-mirror / offline leftovers so we use official international sources.
unset HF_ENDPOINT HF_HUB_DISABLE_XET HF_HUB_OFFLINE PIP_INDEX_URL

echo "[setup_us] WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME  (official HF/PyPI)"
[ -n "${HF_TOKEN:-}" ] || echo "[setup_us] NOTE: HF_TOKEN unset -> gated Mistral skipped (set it + accept license to fetch)"
