#!/usr/bin/env bash
# SOURCE me (don't execute): PART 1 ONLINE env on a CHINA box.
#   source scripts/prep_env_china.sh
# Mirror of prep_env.sh -- only the data disk + download sources differ (mirrors).

export WORK_DIR="${WORK_DIR:-/root/autodl-tmp/tb_work}"   # China data disk (AutoDL)
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
mkdir -p "$WORK_DIR" "$WORK_DIR/logs"

# China mirrors (hf-mirror.com, Tsinghua PyPI, SJTU cu121 torch) + parallel downloads.
source "$(dirname "${BASH_SOURCE[0]}")/setup_china.sh"

echo "[prep_env_china] WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME"
[ -n "${HF_TOKEN:-}" ] || echo "[prep_env_china] NOTE: HF_TOKEN unset -> gated Mistral skipped (fine for smoke/Qwen)"
