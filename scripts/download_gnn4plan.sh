#!/usr/bin/env bash
# Vendor GNN4TaskPlan's data (the dataset GNN4Plan / GRAFT / GTool all benchmark on)
# into GTool's per-domain layout, so the experiment uses the EXACT same samples + the
# FIXED split_ids.json test set as the papers (true 1:1 comparison). Run on a node WITH
# internet (PART 1). China: `source /etc/network_turbo` first makes raw.githubusercontent fast.
#
#   bash scripts/download_gnn4plan.sh [DEST]      # default DEST=dataset_gnn4plan
#
# GNN4TaskPlan's data.json is TaskBench-format (user_request/task_nodes/task_links/type),
# field-compatible with GTool. split_ids.json = {"test_ids": {"chain": [...500 ids...]}}.
# node_desc.json is NOT shipped -> GTool's gnn4plan graph build derives it from tool_desc.json.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-dataset_gnn4plan}"
BASE="https://raw.githubusercontent.com/WxxShirley/GNN4TaskPlan/main/data"
# GNN4TaskPlan dir name : GTool domain dir name (GTool uses 'dailylife' for dailylife).
PAIRS="huggingface:huggingface multimedia:multimedia dailylife:dailylife"
FILES="data.json tool_desc.json graph_desc.json user_requests.json split_ids.json"

for pair in $PAIRS; do
  g="${pair%%:*}"; o="${pair##*:}"
  mkdir -p "$DEST/$o"
  for f in $FILES; do
    echo "[gnn4plan] $g/$f -> $DEST/$o/$f"
    curl -fsSL "$BASE/$g/$f" -o "$DEST/$o/$f"
  done
  n=$(python3 -c "import json;print(len(json.load(open('$DEST/$o/split_ids.json'))['test_ids']['chain']))" 2>/dev/null || echo '?')
  m=$(wc -l < "$DEST/$o/data.json" 2>/dev/null || echo '?')
  echo "[gnn4plan] $o: samples=$m  test_chains=$n"
done
echo "[gnn4plan] done -> $DEST/  (use RAW_ROOT=$DEST in run_experiment.sh / run_grid.sh)"
