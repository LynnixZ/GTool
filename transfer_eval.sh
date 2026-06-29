#!/usr/bin/env bash
# Cross-domain (transfer) eval: take a model TRAINED on ONE source domain and test it on
# EVERY domain's test set (incl. its own). GTool's GNN operates on SBERT features and the
# LLM sees the target domain's tool list in the prompt, so a source-domain checkpoint can
# plan over another domain's graph -- this measures cross-domain generalization.
#
# Reuses the checkpoint trained by run_experiment.sh (TAG=<TAG_PREFIX>_<source>). Output
# csvs are tagged with the domain under test (--eval_tag) so nothing is overwritten:
#   output/<TAG>/..._evalon_<target>_test_all.csv
#
# Usage:
#   bash transfer_eval.sh <model> <source_domain> [TAG_PREFIX] [extra args matching training...]
#   bash transfer_eval.sh mistral huggingface                  # hf-trained -> hf/mm/dl test
#   bash transfer_eval.sh qwen3-8b multimedia g4p_qwen3_8b
#
# IMPORTANT: pass the SAME hyper-params training used (num_epochs/patience/seed/...) so the
# checkpoint path resolves -- defaults match run_experiment.sh's full (non-smoke) run.
set -Eeuo pipefail
cd "$(dirname "$0")"

MODEL="${1:?usage: bash transfer_eval.sh <model> <source_domain> [TAG_PREFIX] [extra args]}"
SRC="${2:?source domain required (huggingface|multimedia|dailylife)}"
TAG_PREFIX="${3:-g4p_${MODEL//[^a-zA-Z0-9]/_}}"
EXTRA=("${@:4}")

RAW_ROOT="${RAW_ROOT:-dataset_gnn4plan}"
SPLIT_ROOT="${SPLIT_ROOT:-artifacts/splits_gnn4plan}"
TARGETS="${TARGETS:-huggingface multimedia dailylife}"

TAG="${TAG_PREFIX}_${SRC}"            # the trained source-domain checkpoint (output/<TAG>/)
echo "[transfer] source checkpoint TAG=$TAG  targets='$TARGETS'  RAW_ROOT=$RAW_ROOT"

for TGT in $TARGETS; do
  echo "===== [$MODEL: train=$SRC -> test=$TGT] ====="
  python inference_zou.py --dataset "$TAG" --llm_model_name "$MODEL" \
      --split_dir "$SPLIT_ROOT/$TGT" --raw_root "$RAW_ROOT" \
      --eval_tag "$TGT" --test_split test_all "${EXTRA[@]}"
done
echo "[transfer] done. results: output/$TAG/..._evalon_<target>_test_all.csv"
