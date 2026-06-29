#!/usr/bin/env bash
# PART 2 (offline / compute node) — full GTool pipeline for one model, PER DOMAIN.
#
# GTool's paper trains per-domain (train.py --dataset <one domain>). So for the
# given MODEL we loop over the 3 domains and run an INDEPENDENT split+train+test
# per domain; nothing mixes domains. Each (model,domain) gets its own TAG /
# output_dir / checkpoint and its own split dir, so runs never collide and
# inference always reloads the exact checkpoint training wrote.
#
# Usage:
#   bash run_experiment.sh [MODEL_ALIAS] [TAG_PREFIX] [extra train_zou args...]
#   bash run_experiment.sh mistral                       # 3 domains
#   bash run_experiment.sh qwen3-8b zou_qwen3_8b --num_epochs 10
#
#   # Run a single domain only (e.g. the grid dispatches one (model,domain) pair):
#   ONLY_DOMAIN=huggingface bash run_experiment.sh mistral
#
#   # Smoke test (tiny split, 1 epoch, small model — verifies the node end-to-end):
#   bash run_experiment.sh --smoke                 # defaults to qwen3-0.6b
#   bash run_experiment.sh --smoke qwen2.5-0.5b
#
# Offline env must already be set (see RUNBOOK §2/§3):
#   export HF_HOME=$WORK_DIR/hf_home
#   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
#
# Data: vendored GNN4TaskPlan (RAW_ROOT=dataset_gnn4plan, scripts/download_gnn4plan.sh).
# Split: GNN4plan (fixed split_ids.json test + capped 85/15 pool) -- same as GNN4Plan/GRAFT/GTool.
# NOTE: graph building (src.dataset.preprocess_gnn4plan) needs a GPU (SBERT on cuda:0) and the
# SBERT model in cache, so this whole script runs on the GPU compute node. First run builds the
# domain's graphs (~one-time); later runs reuse them.
set -euo pipefail

SMOKE=0
if [ "${1:-}" = "--smoke" ]; then SMOKE=1; shift; fi

RAW_ROOT="${RAW_ROOT:-dataset_gnn4plan}"        # vendored GNN4TaskPlan data (scripts/download_gnn4plan.sh)
SPLIT_ROOT="${SPLIT_ROOT:-artifacts/splits_gnn4plan}" # per-domain GNN4plan split at $SPLIT_ROOT/<domain>/

if [ "$SMOKE" = "1" ]; then
  MODEL="${1:-qwen3-0.6b}"
  TAG_PREFIX="${2:-smoke_${MODEL//[^a-zA-Z0-9]/_}}"
  SPLIT_ROOT="${SPLIT_ROOT_SMOKE:-artifacts/splits_smoke}"
  SMOKE_TRAIN_CAP="${SMOKE_TRAIN_CAP:-30}"     # tiny gnn4plan train pool for a fast smoke
  SMOKE_TEST_CAP="${SMOKE_TEST_CAP:-8}"        # shrink the FIXED test set for smoke only
  # Smoke defaults to ALL THREE domains (verify the whole protocol). SMOKE_DOMAIN
  # accepts one or more space-separated domains; for the fastest single-domain check:
  #   SMOKE_DOMAIN=huggingface bash run_experiment.sh --smoke
  SMOKE_DOMAIN="${SMOKE_DOMAIN:-huggingface multimedia dailylife}"
  DOMAINS=($SMOKE_DOMAIN)
  # tiny + fast + LOW-MEMORY train args so a 7B fits a 24G card (4090) out of the box.
  # (the graph token needs grad -> backprop runs through the whole frozen LLM -> full
  # activations; batch 1 + short max_txt_len keeps it under 24G.) Overridable by appending.
  TRAIN_ARGS=(--num_epochs 1 --batch_size 1 --eval_batch_size 1 --patience 1 --max_txt_len 1024)
else
  MODEL="${1:-mistral}"
  TAG_PREFIX="${2:-g4p_${MODEL//[^a-zA-Z0-9]/_}}"
  DOMAINS=(huggingface multimedia dailylife)
  TRAIN_ARGS=()
fi
# Restrict to a single domain when ONLY_DOMAIN is set (used by run_grid.sh fan-out).
if [ -n "${ONLY_DOMAIN:-}" ]; then DOMAINS=("$ONLY_DOMAIN"); fi
EXTRA=("${@:3}")

echo "SMOKE=$SMOKE  MODEL=$MODEL  TAG_PREFIX=$TAG_PREFIX  RAW_ROOT=$RAW_ROOT  SPLIT_ROOT=$SPLIT_ROOT  DOMAINS=${DOMAINS[*]}"
python -c "import torch;print('cuda available:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

for d in "${DOMAINS[@]}"; do
  TAG="${TAG_PREFIX}_${d}"
  SPLIT_DIR="$SPLIT_ROOT/$d"
  echo "===== [$MODEL / $d] TAG=$TAG SPLIT_DIR=$SPLIT_DIR ====="

  # 1) Build this domain's GTool graphs from the vendored GNN4TaskPlan data (idempotent).
  #    Compact format -> graph_base.pt + requests.pt (one each per domain).
  if [ -f "$RAW_ROOT/$d/graph_base.pt" ] && [ -f "$RAW_ROOT/$d/requests.pt" ]; then
    echo "[skip] graphs exist for $d"
  else
    echo "[build] graphs for $d (from GNN4TaskPlan data)"
    python -m src.dataset.preprocess_gnn4plan --root "$RAW_ROOT" --domains "$d"
  fi

  # 2) GNN4plan split for THIS domain only (idempotent): test = fixed split_ids.json
  #    chains; train/val = pool minus test, shuffled seed42, cap 3000, 85/15.
  if [ -f "$SPLIT_DIR/train.jsonl" ]; then
    echo "[skip] split exists at $SPLIT_DIR"
  elif [ "$SMOKE" = "1" ]; then
    echo "[build] smoke split -> $SPLIT_DIR (domain=$d, gnn4plan tiny: train_cap=$SMOKE_TRAIN_CAP test_cap=$SMOKE_TEST_CAP)"
    python -m src.dataset.preprocess_zou.split_subset \
        --raw_root "$RAW_ROOT" --out_dir "$SPLIT_DIR" --domains "$d" \
        --mode gnn4plan --train_cap "$SMOKE_TRAIN_CAP" --test_cap "$SMOKE_TEST_CAP"
  else
    echo "[build] gnn4plan split -> $SPLIT_DIR (domain=$d)"
    python -m src.dataset.preprocess_zou.split_subset \
        --raw_root "$RAW_ROOT" --out_dir "$SPLIT_DIR" --domains "$d" --mode gnn4plan
  fi

  # 3) Train (best checkpoint by val loss + early stop; auto-tests on test_all at the end).
  echo "[train] $MODEL / $d"
  python train_zou.py --dataset "$TAG" --llm_model_name "$MODEL" \
      --split_dir "$SPLIT_DIR" --raw_root "$RAW_ROOT" "${TRAIN_ARGS[@]}" "${EXTRA[@]}"

  # 4) Test (per-bucket node/chain breakdown + overall).
  # MUST pass the same TRAIN_ARGS/EXTRA + identical TAG/SPLIT_DIR: the checkpoint
  # filename encodes num_epochs & patience and lives under output/<TAG>/, so
  # inference has to use identical values to locate ..._checkpoint_best.pth.
  echo "[test] $MODEL / $d"
  python inference_zou.py --dataset "$TAG" --llm_model_name "$MODEL" \
      --split_dir "$SPLIT_DIR" --raw_root "$RAW_ROOT" "${TRAIN_ARGS[@]}" "${EXTRA[@]}"

  echo "DONE [$MODEL / $d]. Results: output/$TAG/*.csv"
done
