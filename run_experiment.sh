#!/usr/bin/env bash
# PART 2 (offline / compute node) — full GTool pipeline for one model.
# Builds graphs + split ONCE (model-independent, reused), then trains + tests.
#
# Usage:
#   bash run_experiment.sh [MODEL_ALIAS] [OUTPUT_TAG] [extra train_zou args...]
#   bash run_experiment.sh mistral
#   bash run_experiment.sh qwen3-8b zou_qwen3_8b --num_epochs 10
#
#   # Smoke test (tiny split, 1 epoch, small model — verifies the node end-to-end):
#   bash run_experiment.sh --smoke                 # defaults to qwen3-0.6b
#   bash run_experiment.sh --smoke qwen2.5-0.5b
#
# Offline env must already be set (see RUNBOOK §2/§3):
#   export HF_HOME=$WORK_DIR/hf_home
#   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
#
# NOTE: graph building (src.dataset.preprocess.*) needs a GPU (SBERT on cuda:0)
# and the SBERT model in cache, so this whole script runs on the GPU compute node.
# The first run builds ALL graphs (~one-time, even in smoke); later runs reuse them.
set -euo pipefail

SMOKE=0
if [ "${1:-}" = "--smoke" ]; then SMOKE=1; shift; fi

RAW_ROOT="${RAW_ROOT:-dataset}"                 # GTool preprocess writes to dataset/<domain>/

if [ "$SMOKE" = "1" ]; then
  MODEL="${1:-qwen3-0.6b}"
  TAG="${2:-smoke_${MODEL//[^a-zA-Z0-9]/_}}"
  SPLIT_DIR="${SPLIT_DIR:-artifacts/splits_smoke}"
  SMOKE_LIMIT="${SMOKE_LIMIT:-40}"              # usable samples per domain
  # tiny + fast train args (overridable by appending more args)
  TRAIN_ARGS=(--num_epochs 1 --batch_size 2 --eval_batch_size 2 --patience 1)
else
  MODEL="${1:-mistral}"
  TAG="${2:-zou_${MODEL//[^a-zA-Z0-9]/_}}"
  SPLIT_DIR="${SPLIT_DIR:-artifacts/splits_subset}"
  TRAIN_ARGS=()
fi
EXTRA=("${@:3}")

echo "SMOKE=$SMOKE  MODEL=$MODEL  TAG=$TAG  RAW_ROOT=$RAW_ROOT  SPLIT_DIR=$SPLIT_DIR"
python -c "import torch;print('cuda available:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

# 1) Build GTool graphs per domain (idempotent: skip if already built).
for d in huggingface multimedia dailylife; do
  if [ -d "$RAW_ROOT/$d/graphs" ] && [ -n "$(ls -A "$RAW_ROOT/$d/graphs" 2>/dev/null)" ]; then
    echo "[skip] graphs exist for $d"
  else
    echo "[build] graphs for $d"
    python -m "src.dataset.preprocess.$d"
  fi
done

# 2) Stratified split (idempotent). Smoke uses a tiny, coverage-relaxed split.
if [ -f "$SPLIT_DIR/train.jsonl" ]; then
  echo "[skip] split exists at $SPLIT_DIR"
elif [ "$SMOKE" = "1" ]; then
  echo "[build] smoke split -> $SPLIT_DIR (limit_per_domain=$SMOKE_LIMIT)"
  python -m src.dataset.preprocess_zou.split_subset \
      --raw_root "$RAW_ROOT" --out_dir "$SPLIT_DIR" \
      --limit_per_domain "$SMOKE_LIMIT" --skip_coverage
else
  echo "[build] split -> $SPLIT_DIR"
  python -m src.dataset.preprocess_zou.split_subset --raw_root "$RAW_ROOT" --out_dir "$SPLIT_DIR"
fi

# 3) Train (best checkpoint by val loss + early stop; auto-tests on test_all at the end).
echo "[train] $MODEL"
python train_zou.py --dataset "$TAG" --llm_model_name "$MODEL" \
    --split_dir "$SPLIT_DIR" --raw_root "$RAW_ROOT" "${TRAIN_ARGS[@]}" "${EXTRA[@]}"

# 4) Test (per-bucket node/chain breakdown + overall).
echo "[test] $MODEL"
python inference_zou.py --dataset "$TAG" --llm_model_name "$MODEL" \
    --split_dir "$SPLIT_DIR" --raw_root "$RAW_ROOT"

echo "DONE. Results: output/$TAG/*.csv"
