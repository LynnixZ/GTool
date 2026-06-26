#!/usr/bin/env bash
# Convenience env for a CHINA GPU box -- mirrors only, NO secrets, safe in git.
#   source scripts/setup_china.sh
# All values respect anything you already exported (override per box).
# On a US/EU server use scripts/setup_US.sh instead.

export WORK_DIR="${WORK_DIR:-/root/autodl-tmp/tb_work}"   # big data disk (AutoDL)
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"

# --- China mirrors / network workarounds ---
# AutoDL TIP: if /etc/network_turbo exists, `source /etc/network_turbo` BEFORE this file
# gives proxied access to the OFFICIAL HF + github (often faster/more complete than mirrors).
# pip + torch still keep the China mirrors below even under turbo, so this stays correct.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# torch: download.pytorch.org is ~3 MB/s from China. Prefer SJTU's MIRROR of the cu121
# wheels (same driver-safe cu121 build, fast from China). DO NOT use plain Tsinghua PyPI
# 'torch' -- it resolves to the latest cu13 build and fails on a 12.x driver.
# CAVEAT: SJTU's cu121 wheels START at 2.2.0 -- they DO NOT carry torch==2.1.0, which GTool
# needs (its PyG companion wheels are pinned pt21cu121). prestage_all.sh handles this: it
# tries this index first, then AUTO-FALLS-BACK to the official download.pytorch.org/whl/cu121
# (which does have 2.1.0). To make that fallback fast on AutoDL, run
# `source /etc/network_turbo` BEFORE prestage (proxies download.pytorch.org + huggingface.co).
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu121}"
# Parallel downloads (Xet for new models like Qwen3; hf_transfer for classic LFS).
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "[setup_china] WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME  HF_ENDPOINT=$HF_ENDPOINT"
[ -n "${HF_TOKEN:-}" ] || echo "[setup_china] NOTE: HF_TOKEN unset -> gated Mistral skipped (fine for smoke / Qwen)"
