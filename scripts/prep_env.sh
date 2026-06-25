#!/usr/bin/env bash
# SOURCE me (don't execute): PART 1 ONLINE env on the US login node.
#   source scripts/prep_env.sh
# Edit WORK_DIR if your shared dir name differs from $USER (e.g. dir 'xinyu' vs 'xinyuzh').

# Shared NFS visible from EVERY compute node (NOT $HOME / local disk).
export WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
mkdir -p "$WORK_DIR" "$WORK_DIR/logs"

# Official HF/PyPI sources + parallel downloads; clears any mirror/offline leftovers.
source "$(dirname "${BASH_SOURCE[0]}")/setup_US.sh"

echo "[prep_env] WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME"
[ -n "${HF_TOKEN:-}" ] || echo "[prep_env] WARN: HF_TOKEN unset -> gated Mistral will be SKIPPED"
