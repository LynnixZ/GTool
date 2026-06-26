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
# ISOLATED venv is the DEFAULT (do NOT use --system-site-packages). We learned that
# coupling to the base image (root + --system-site-packages) caused two failures:
#   (a) `pip install --force-reinstall torch` DELETED the base image's torch, and
#   (b) the base image's newer CUDA libs SHADOWED the cu121 ones -> cuda.is_available()
#       stayed False even after installing torch 2.1.0+cu121.
# An isolated venv avoids both. Set VENV_SYSTEM_SITE=1 only if you deliberately want to
# reuse a base torch (then it must already be 2.1.x+cu121; the PyG wheels are pt21cu121).
VENV_FLAGS=""; [ "${VENV_SYSTEM_SITE:-0}" = 1 ] && VENV_FLAGS="--system-site-packages"
[ -d "$VENV_DIR" ] || { log "creating venv (${VENV_FLAGS:-isolated})"; python3 -m venv $VENV_FLAGS "$VENV_DIR"; }
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null

# Install torch unless this venv ALREADY has EXACTLY 2.1.x (the pinned PyG wheels are
# pt21cu121). Isolated venv -> import fails first time -> we install. Version-only check:
# CUDA can't be verified on a no-GPU prep box (AutoDL 无卡模式), and a plain
# startswith('2.1') would wrongly match 2.12.x.
if python -c "import torch,sys; v=torch.__version__.split('+')[0].split('.'); sys.exit(0 if (v[0]=='2' and v[1]=='1') else 1)" 2>/dev/null; then
  log "torch already present: $(python -c 'import torch;print(torch.__version__)')"
else
  log "installing torch==2.1.0 from $TORCH_INDEX_URL"
  # No --force-reinstall: in an isolated venv torch isn't present, and force-reinstall
  # is what nuked the base image's torch under --system-site-packages.
  pip install "torch==2.1.0" --index-url "$TORCH_INDEX_URL"
fi
# PIN torch so the unpinned deps below (transformers/accelerate/PyG) can never swap it
# for a cu13 wheel during their resolve. cu13 on a 12.x driver is the #1 failure mode.
TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
echo "torch==$TORCH_VER" > "$WORK_DIR/torch.constraint"
log "pinned torch==$TORCH_VER -> $WORK_DIR/torch.constraint"

# GPU status is INFORMATIONAL: pre-staging (deps + model download) does NOT need a
# GPU. Training (PART2) does. So we warn, not fail, when CUDA is unavailable.
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  log "CUDA OK: cuda=$(python -c 'import torch;print(torch.version.cuda)')  devices=$(python -c 'import torch;print(torch.cuda.device_count())')"
else
  log "WARN: torch.cuda.is_available()=False."
  log "      - If this is AutoDL 无卡模式 (no GPU attached for cheap prep): FINE here --"
  log "        finish prestage, then reboot into a GPU instance and run training (PART2)."
  log "      - If you DO have a GPU now: check 'nvidia-smi' (cu121 needs driver >= 525)."
fi

log "installing PyG companion wheels (pt21cu121)"
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 \
  -f "$PYG_FIND_LINKS" -c "$WORK_DIR/torch.constraint"

log "installing requirements-node.txt"
pip install hf_transfer || log "WARN: hf_transfer failed (downloads just slower)"
pip install -r requirements-node.txt -c "$WORK_DIR/torch.constraint"

# Final env verify (informational; cuda False is OK in 无卡模式 prep -- see warn above).
python -c "import torch; print('[verify] venv='+'$VENV_DIR'); print('[verify] torch', torch.__version__, '| cuda_build', torch.version.cuda, '| cuda.is_available', torch.cuda.is_available())"

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
