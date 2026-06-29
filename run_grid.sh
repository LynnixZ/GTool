#!/usr/bin/env bash
# Task-parallel runner: run several (model,domain) runs AT ONCE, one per GPU.
#
# GTool trains PER DOMAIN (paper protocol), so the unit of work is a
# (model, domain) pair, not a model. With 3 models x 3 domains that is 9 runs.
#
# Why one-run-per-GPU (and not multi-GPU for one run): GTool training is
# single-process with NO data-parallel (DDP). device_map="auto" only *shards*
# one model across GPUs (model-parallel) to fit bigger weights -- for a 7B/8B
# that fits on one card it gives NO speedup and is slower. The right use of
# 2-4 GPUs here is to run independent runs concurrently, one pinned per GPU via
# CUDA_VISIBLE_DEVICES.
#
# Builds graphs ONCE first (serial, to avoid races), then fans out the 9 pairs.
# If there are more pairs than GPUs, each GPU runs its share sequentially (queue).
# Each pair builds its own per-domain split lazily inside run_experiment.sh.
#
#   bash run_grid.sh vicuna mistral qwen3-8b           # GPUs auto-detected
#   GPUS="0 1" MODELS="vicuna mistral qwen3-8b" bash run_grid.sh
#   GPUS="0 1 2 3" bash run_grid.sh vicuna mistral qwen3-8b
#   DOMAINS="huggingface multimedia" bash run_grid.sh vicuna   # subset of domains
#
# Per-run extra train args via EXTRA, e.g. for 24G cards:
#   EXTRA="--batch_size 1 --max_txt_len 1536" bash run_grid.sh mistral qwen3-8b
set -Eeuo pipefail
cd "$(dirname "$0")"

RAW_ROOT="${RAW_ROOT:-dataset_gnn4plan}"        # vendored GNN4TaskPlan data
SPLIT_ROOT="${SPLIT_ROOT:-artifacts/splits_gnn4plan}"
LOGDIR="${LOGDIR:-output/grid_logs}"; mkdir -p "$LOGDIR"
EXTRA="${EXTRA:-}"

# models from args or $MODELS
[ "$#" -gt 0 ] && MODELS="$*"
MODELS="${MODELS:-vicuna mistral qwen3-8b}"
DOMAINS="${DOMAINS:-huggingface multimedia dailylife}"

# GPUs from $GPUS or auto-detect
if [ -z "${GPUS:-}" ]; then
  NGD=$(nvidia-smi -L 2>/dev/null | wc -l); GPUS=$(seq -s' ' 0 $((NGD-1)))
fi
GPU_ARR=($GPUS); NG=${#GPU_ARR[@]}
[ "$NG" -ge 1 ] || { echo "[grid] no GPUs found"; exit 1; }
echo "[grid] MODELS='$MODELS'  DOMAINS='$DOMAINS'  GPUS='${GPU_ARR[*]}'  ($NG GPU)  EXTRA='$EXTRA'"

# 1) one-time graph build (serial; avoids parallel workers racing on the same
#    files). Splits are per-domain and built lazily inside run_experiment.sh.
for d in $DOMAINS; do
  if [ -d "$RAW_ROOT/$d/graphs" ] && [ -n "$(ls -A "$RAW_ROOT/$d/graphs" 2>/dev/null)" ]; then
    echo "[grid] graphs ok: $d"
  else
    echo "[grid] building graphs: $d (GNN4TaskPlan data, on GPU ${GPU_ARR[0]})"
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -m src.dataset.preprocess_gnn4plan --root "$RAW_ROOT" --domains "$d"
  fi
done

# 1b) PRE-BUILD per-domain splits SERIALLY (idempotent). Multiple (model,domain)
#     pairs sharing the same domain can launch concurrently on different GPUs; if
#     they each built $SPLIT_ROOT/<domain> lazily they'd race/corrupt the JSONL.
#     Build them here once, up front, with the IDENTICAL invocation + out_dir that
#     run_experiment.sh uses, so the workers' lazy build becomes a no-op skip.
for d in $DOMAINS; do
  if [ -f "$SPLIT_ROOT/$d/train.jsonl" ]; then
    echo "[grid] split ok: $d"
  else
    echo "[grid] building gnn4plan split: $d -> $SPLIT_ROOT/$d"
    python -m src.dataset.preprocess_zou.split_subset \
        --raw_root "$RAW_ROOT" --out_dir "$SPLIT_ROOT/$d" --domains "$d" --mode gnn4plan
  fi
done

# 2) build the (model,domain) work list, then round-robin -> per-GPU queues
pairs=()
for m in $MODELS; do
  for d in $DOMAINS; do
    pairs+=("$m:$d")
  done
done
echo "[grid] ${#pairs[@]} (model,domain) runs queued"

queues=(); i=0
for p in "${pairs[@]}"; do
  idx=$(( i % NG )); queues[$idx]="${queues[$idx]:-} $p"; i=$(( i + 1 ))
done

# 3) one background worker per GPU; each runs its queue sequentially
pids=()
for ((k=0; k<NG; k++)); do
  g="${GPU_ARR[$k]}"; pl="${queues[$k]:-}"; [ -n "${pl// /}" ] || continue
  (
    for p in $pl; do
      m="${p%%:*}"; d="${p##*:}"
      tag="grid_${m//[^a-zA-Z0-9]/_}_${d}"
      echo "[grid][gpu $g] START $m / $d -> $LOGDIR/${tag}.log"
      if CUDA_VISIBLE_DEVICES="$g" RAW_ROOT="$RAW_ROOT" SPLIT_ROOT="$SPLIT_ROOT" \
           ONLY_DOMAIN="$d" \
           bash run_experiment.sh "$m" "grid_${m//[^a-zA-Z0-9]/_}" $EXTRA > "$LOGDIR/${tag}.log" 2>&1; then
        echo "[grid][gpu $g] DONE  $m / $d"
      else
        echo "[grid][gpu $g] FAIL  $m / $d  (see $LOGDIR/${tag}.log)"
      fi
    done
  ) &
  pids+=($!)
done
wait "${pids[@]}"
echo "[grid] all workers finished. logs: $LOGDIR/"

# ---- Package lightweight results (per-(model,domain) csv reports + logs; exclude the
# big *.pth checkpoints) into ONE tarball under output/, and print its ABSOLUTE path at
# the very end so it's easy to scp/download. ----
TAR="output/gtool_results_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}.tar.gz"
if tar czf "$TAR" --exclude='*.pth' output/grid_* 2>/dev/null; then
  echo "[grid] results tarball -> $(cd "$(dirname "$TAR")" && pwd)/$(basename "$TAR")"
else
  echo "[grid] WARN: packaging failed (results still under output/grid_*/)"
fi
echo "[grid] ALL DONE."
