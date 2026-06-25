#!/usr/bin/env bash
# Task-parallel runner: run several models AT ONCE, one per GPU.
#
# Why this (and not just multi-GPU for one run): GTool training is single-process
# with NO data-parallel (DDP). device_map="auto" only *shards* one model across
# GPUs (model-parallel) to fit bigger weights -- for a 7B/8B that fits on one card
# it gives NO speedup and is slower. The right use of 2-4 GPUs here is to run
# independent models concurrently, one pinned per GPU via CUDA_VISIBLE_DEVICES.
#
# Builds graphs + split ONCE first (serial, to avoid races), then fans out. If
# there are more models than GPUs, each GPU runs its share sequentially (queue).
#
#   bash run_grid.sh vicuna mistral qwen3-8b           # GPUs auto-detected
#   GPUS="0 1" MODELS="vicuna mistral qwen3-8b" bash run_grid.sh
#   GPUS="0 1 2 3" bash run_grid.sh vicuna mistral qwen3-8b
#
# Per-model extra train args via EXTRA, e.g. for 24G cards:
#   EXTRA="--batch_size 1 --max_txt_len 1536" bash run_grid.sh mistral qwen3-8b
set -Eeuo pipefail
cd "$(dirname "$0")"

RAW_ROOT="${RAW_ROOT:-dataset}"
SPLIT_DIR="${SPLIT_DIR:-artifacts/splits_subset}"
LOGDIR="${LOGDIR:-output/grid_logs}"; mkdir -p "$LOGDIR"
EXTRA="${EXTRA:-}"

# models from args or $MODELS
[ "$#" -gt 0 ] && MODELS="$*"
MODELS="${MODELS:-vicuna mistral qwen3-8b}"

# GPUs from $GPUS or auto-detect
if [ -z "${GPUS:-}" ]; then
  NGD=$(nvidia-smi -L 2>/dev/null | wc -l); GPUS=$(seq -s' ' 0 $((NGD-1)))
fi
GPU_ARR=($GPUS); NG=${#GPU_ARR[@]}
[ "$NG" -ge 1 ] || { echo "[grid] no GPUs found"; exit 1; }
echo "[grid] MODELS='$MODELS'  GPUS='${GPU_ARR[*]}'  ($NG GPU)  EXTRA='$EXTRA'"

# 1) one-time build (serial; avoids parallel workers racing on the same files)
for d in huggingface multimedia dailylife; do
  if [ -d "$RAW_ROOT/$d/graphs" ] && [ -n "$(ls -A "$RAW_ROOT/$d/graphs" 2>/dev/null)" ]; then
    echo "[grid] graphs ok: $d"
  else
    echo "[grid] building graphs: $d (on GPU ${GPU_ARR[0]})"
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -m "src.dataset.preprocess.$d"
  fi
done
[ -f "$SPLIT_DIR/train.jsonl" ] || \
  python -m src.dataset.preprocess_zou.split_subset --raw_root "$RAW_ROOT" --out_dir "$SPLIT_DIR"

# 2) round-robin assign models -> per-GPU queues (indexed arrays = bash 3.2 safe)
queues=(); i=0
for m in $MODELS; do
  idx=$(( i % NG )); queues[$idx]="${queues[$idx]:-} $m"; i=$(( i + 1 ))
done

# 3) one background worker per GPU; each runs its queue sequentially
pids=()
for ((k=0; k<NG; k++)); do
  g="${GPU_ARR[$k]}"; ml="${queues[$k]:-}"; [ -n "${ml// /}" ] || continue
  (
    for m in $ml; do
      tag="grid_${m//[^a-zA-Z0-9]/_}"
      echo "[grid][gpu $g] START $m -> $LOGDIR/${tag}.log"
      if CUDA_VISIBLE_DEVICES="$g" RAW_ROOT="$RAW_ROOT" SPLIT_DIR="$SPLIT_DIR" \
           bash run_experiment.sh "$m" "$tag" $EXTRA > "$LOGDIR/${tag}.log" 2>&1; then
        echo "[grid][gpu $g] DONE  $m"
      else
        echo "[grid][gpu $g] FAIL  $m  (see $LOGDIR/${tag}.log)"
      fi
    done
  ) &
  pids+=($!)
done
wait "${pids[@]}"
echo "[grid] all workers finished. logs: $LOGDIR/  results: output/grid_*/"
