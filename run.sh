#!/usr/bin/env bash
# ============================================================================
# GTool 美国节点启动脚本 — per-domain 实验。COMMITTED（已入库，无密钥，可直接 clone/pull）。
#
# 用法：
#   1) clone 到【共享 NFS】（不是 $HOME，计算节点看不到 /home）：
#        git clone https://github.com/LynnixZ/GTool.git /playpen-shared/<你的共享目录>/GTool
#        cd /playpen-shared/<你的共享目录>/GTool
#   2) 在你自己的 shell export 密钥（别写进这个文件）：
#        export HF_TOKEN=hf_xxx          # 下 gated 的 Mistral 用；token 须在 HF 接受过 Mistral 许可
#   3) 改下面【要改①】SHARED、【要改②】GPU 数，然后：  bash run.sh
#   ★ US Slurm 必带 VENV_PYTHON（登录节点 conda 建的 venv 在计算节点会失效）：
#        VENV_PYTHON=/usr/bin/python3 bash run.sh      # China 不用设；它会透传给 prestage 建 venv
#   以后更新只需 `git pull`。
# ============================================================================
set -e

# ======================【 要改①：共享目录名 SHARED 】======================
#   填你 /playpen-shared/<这个> 的真实目录名（可能 ≠ 登录名）；不确定就 `ls /playpen-shared/`。
export SHARED="${SHARED:-xinyu}"                                   # <-- 改这里（默认 xinyu；不确定 `ls /playpen-shared/`）
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # 本 repo 位置（自动，须在 /playpen-shared）
export WORK_DIR="${WORK_DIR:-/playpen-shared/$SHARED/tb_work}"     # venv + 缓存（共享 NFS）
export HF_HOME="$WORK_DIR/hf_home"
case "$REPO" in /playpen-shared/*) : ;; *) echo "⚠️ REPO=$REPO 不在 /playpen-shared，计算节点看不到 -> 请 clone 到共享盘";; esac
[ -n "${HF_TOKEN:-}" ] || echo "WARN: HF_TOKEN 没设 -> Mistral(gated) 会被跳过；export HF_TOKEN=... 再跑可补上。"

# ===== PART1：登录节点（有网）下环境 + 模型 =====
source scripts/prep_env.sh               # 官方源 cu121 + WORK_DIR/HF_HOME（共享 NFS）
bash scripts/prestage_all.sh             # 隔离 venv + cu121 torch + PyG + 下模型

# ===== PART2：提交【一个】离线作业占【一个】节点的 N 张卡 =====
#   job 内 run_grid 把 3 模型 × 3 域 = 9 个 run 铺到这 N 卡（一卡一 run，排队）。
#   分区自动投 a100,ada,a6000（避 blackwell）。Mistral 没许可会自动跳过，vicuna/qwen3 照跑。
# ======================【 要改②：申请几张 GPU 】======================
GPUS_PER_JOB="${GPUS_PER_JOB:-2}" bash scripts/submit_unites.sh    # <-- 改卡数；也可 GPUS_PER_JOB=4 bash run.sh 覆盖

# 其他用法（按需）：
#   只跑某个模型（×3 域）：MODELS=mistral GPUS_PER_JOB=2 bash scripts/submit_unites.sh
#   烟测：SMOKE=1 MODEL=qwen2.5-0.5b GPUS_PER_JOB=1 bash scripts/submit_unites.sh

squeue -u "$USER"
# tail -f "$WORK_DIR"/logs/gtool-*.out
