#!/bin/bash
# US Slurm — offline GTool job (submit from the LOGIN node with: sbatch job.sbatch).
# Edit REPO / WORK_DIR / partition / paths to match your cluster, then submit.
# Override per-submit, e.g.:  MODEL=qwen3-8b sbatch --gres=gpu:1 job.sbatch
#
#SBATCH --job-name=gtool
#SBATCH --partition=a100            # any cu121-supported arch: Ampere(A100/3090) or Ada(4090). AVOID Blackwell.
#SBATCH --gres=gpu:1                # 1 GPU fits a frozen 7B/8B + small GNN. On 24G (4090): lower --batch_size/--max_txt_len.
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out          # consider an absolute path on shared FS

set -euo pipefail

# ── re-export everything: the compute node is a clean, OFFLINE environment ──
export REPO="${REPO:-/playpen-shared/$USER/GTool}"          # <-- repo on shared FS
export WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"
export HF_HOME="$WORK_DIR/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1              # never touch the network
export OMP_NUM_THREADS=1                                    # silence the OMP warning
# GTool has NO wandb — no WANDB_* needed.

cd "$REPO"
source "$WORK_DIR/venv/bin/activate"

# sanity: CUDA must be usable, else the offline run wastes the allocation
python -c "import torch;assert torch.cuda.is_available(), 'CUDA not available'; print('cuda OK, devices:', torch.cuda.device_count())"

bash run_experiment.sh "${MODEL:-mistral}"
