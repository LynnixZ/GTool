#!/usr/bin/env bash
# PART 1 one-shot pre-stage onto the DATA DISK: Python venv + deps + models.
# (GTool ships its TaskBench subset inside the repo under dataset/, so NO data
# download step is needed -- unlike the SFT-comparison repo.)
#
# Gated auto-detect: each model is tried WITHOUT a token; if gated and HF_TOKEN is
# set + license accepted it retries with the token, else it prints NEEDS_TOKEN and
# skips (so the free models + SBERT still cache). Run on a node WITH internet.
#
#   source scripts/prep_env_china.sh      # China (mirrors)   -- or scripts/prep_env.sh on US
#   export HF_TOKEN=hf_xxx                 # OPTIONAL: only to also fetch gated Mistral
#   bash scripts/prestage_all.sh
set -Eeuo pipefail
cd "$(dirname "$0")/.."

WORK_DIR="${WORK_DIR:-/root/autodl-tmp/tb_work}"
HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
VENV_DIR="${VENV_DIR:-$WORK_DIR/venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
PYG_FIND_LINKS="https://data.pyg.org/whl/torch-2.1.0+cu121.html"
export HF_HOME
mkdir -p "$WORK_DIR" "$HF_HOME"

# Models: smoke + SBERT (mandatory for graph building) + the two comparison LLMs.
DEFAULT_MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"                  # smoke (non-gated)
  "sentence-transformers/all-roberta-large-v1"  # SBERT -- MANDATORY (builds the tool graphs)
  "Qwen/Qwen3-8B"                               # non-gated
  "lmsys/vicuna-7b-v1.5"                         # non-gated
  "mistralai/Mistral-7B-Instruct-v0.3"          # gated -> needs HF_TOKEN
)
if [ -n "${MODELS:-}" ]; then read -ra MODEL_LIST <<< "$MODELS"; else MODEL_LIST=("${DEFAULT_MODELS[@]}"); fi

log() { echo "[$(date -u +%H:%M:%S)] [prestage] $*"; }
log "WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME"
log "PIP_INDEX_URL=${PIP_INDEX_URL:-(default PyPI)}  HF_ENDPOINT=${HF_ENDPOINT:-(default hf.co)}  TORCH_INDEX_URL=$TORCH_INDEX_URL"
[ -n "${HF_TOKEN:-}" ] && log "HF_TOKEN set -> will also try gated Mistral" || log "HF_TOKEN unset -> gated Mistral skipped"

# ---- 1. venv + deps ----
# --system-site-packages REUSES a usable CUDA torch from the image/base env when
# possible. GTool needs torch 2.1.x specifically (the pinned PyG wheels are pt21cu121),
# so we only reuse the base torch if it is 2.1.x AND CUDA works; otherwise install 2.1.0.
VENV_FLAGS=""; [ "${VENV_ISOLATED:-0}" = 1 ] || VENV_FLAGS="--system-site-packages"
[ -d "$VENV_DIR" ] || { log "creating venv ($VENV_FLAGS)"; python3 -m venv $VENV_FLAGS "$VENV_DIR"; }
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null

if python -c "import torch,sys; sys.exit(0 if (torch.cuda.is_available() and torch.__version__.startswith('2.1')) else 1)" 2>/dev/null; then
  log "reusing base torch $(python -c 'import torch;print(torch.__version__)')  (cuda OK)"
else
  log "installing torch==2.1.0 from $TORCH_INDEX_URL"
  pip install --force-reinstall "torch==2.1.0" --index-url "$TORCH_INDEX_URL"
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available (likely cu13 mismatch)'; print('torch', torch.__version__, 'cuda OK')"

log "installing PyG companion wheels (pt21cu121)"
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 -f "$PYG_FIND_LINKS"

log "installing requirements-node.txt"
pip install hf_transfer || log "WARN: hf_transfer failed (downloads just slower)"
pip install -r requirements-node.txt

# ---- 2. Models (auto-detect gated). SBERT + Qwen are free; Mistral is gated. ----
if [ "${SKIP_MODELS:-0}" = 1 ]; then log "SKIP_MODELS=1 -> env ready, no models pre-downloaded."; exit 0; fi
log "downloading models to $HF_HOME ..."
SUMMARY="$WORK_DIR/prestage_models_summary.txt"; : > "$SUMMARY"
for model in "${MODEL_LIST[@]}"; do
  log "=== $model ==="
  status=$(MODEL_ID="$model" python - <<'PY'
import os
from huggingface_hub import snapshot_download
model = os.environ["MODEL_ID"]; token = os.environ.get("HF_TOKEN") or None
try:
    # KEEP .bin (SBERT ships pytorch_model.bin); skip raw consolidated / GGUF.
    snapshot_download(model, token=token, ignore_patterns=["original/*", "*.pth", "*.gguf", "consolidated*"])
    print("OK")
except Exception as e:
    msg = str(e).lower()
    print("NEEDS_TOKEN" if any(s in msg for s in ("gated","restricted","401","403","awaiting","access to model")) else "ERROR:"+type(e).__name__)
PY
)
  echo "$status  $model" | tee -a "$SUMMARY"
done
log "================= SUMMARY ================="; cat "$SUMMARY"
log "OK = cached; NEEDS_TOKEN = gated (set HF_TOKEN + accept license, re-run). done."
