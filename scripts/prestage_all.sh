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
PYG_FIND_LINKS="https://data.pyg.org/whl/torch-2.2.0+cu121.html"   # pt22cu121 wheels (match torch 2.2.x)
export HF_HOME
mkdir -p "$WORK_DIR" "$HF_HOME"

# SBERT is MANDATORY (builds the tool graphs for EVERY run) -> always downloaded,
# regardless of MODELS. `MODELS="repo1 repo2"` only chooses which LLM(s) to stage;
# SBERT is added automatically. Unset MODELS -> the full comparison set below.
SBERT_MODEL="sentence-transformers/all-roberta-large-v1"
DEFAULT_LLMS=(
  "Qwen/Qwen2.5-0.5B-Instruct"          # smoke (non-gated)
  "Qwen/Qwen3-8B"                       # non-gated
  "lmsys/vicuna-7b-v1.5"                # non-gated
  "mistralai/Mistral-7B-Instruct-v0.3"  # gated -> needs HF_TOKEN
)
if [ -n "${MODELS:-}" ]; then read -ra LLM_LIST <<< "$MODELS"; else LLM_LIST=("${DEFAULT_LLMS[@]}"); fi
# SBERT first, then the chosen LLMs (dedup if SBERT was also passed in MODELS).
MODEL_LIST=("$SBERT_MODEL")
for m in "${LLM_LIST[@]}"; do [ "$m" = "$SBERT_MODEL" ] || MODEL_LIST+=("$m"); done

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
# reuse a base torch (then it must already be 2.2.x+cu121; the PyG wheels are pt22cu121).
VENV_FLAGS=""; [ "${VENV_SYSTEM_SITE:-0}" = 1 ] && VENV_FLAGS="--system-site-packages"
[ -d "$VENV_DIR" ] || { log "creating venv (${VENV_FLAGS:-isolated})"; python3 -m venv $VENV_FLAGS "$VENV_DIR"; }
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null

# torch 2.2.2+cu121 (pinned PyG companion wheels are pt22cu121 -- keep torch 2.2.x).
# NOTE: torch 2.1.0+cu121 is no longer distributed -- both SJTU and the official cu121
# index now START at 2.2.0 ("No matching distribution found for torch==2.1.0"). So the
# whole stack targets 2.2.2+cu121, which SJTU has (fast in China) and matches cu12.x drivers.
# Install with a fallback: try the configured $TORCH_INDEX_URL first, then the OFFICIAL index.
# (No --force-reinstall: in an isolated venv torch isn't present, and force-reinstall is
# what nuked the base image's torch under --system-site-packages.)
TORCH_VERSION="2.2.2"
TORCH_OFFICIAL_URL="https://download.pytorch.org/whl/cu121"
install_torch() {
  # $1 = index url; returns nonzero if torch==$TORCH_VERSION isn't found there.
  pip install "torch==$TORCH_VERSION" --index-url "$1"
}
# Install unless this venv ALREADY has EXACTLY 2.2.x. Isolated venv -> import fails first
# time -> we install. Version-only check: CUDA can't be verified on a no-GPU prep box
# (AutoDL 无卡模式).
if python -c "import torch,sys; v=torch.__version__.split('+')[0].split('.'); sys.exit(0 if (v[0]=='2' and v[1]=='2') else 1)" 2>/dev/null; then
  log "torch already present: $(python -c 'import torch;print(torch.__version__)')"
else
  log "installing torch==$TORCH_VERSION (try $TORCH_INDEX_URL first)"
  if install_torch "$TORCH_INDEX_URL"; then
    log "torch installed from $TORCH_INDEX_URL"
  elif [ "$TORCH_INDEX_URL" = "$TORCH_OFFICIAL_URL" ]; then
    log "ERROR: torch==$TORCH_VERSION not found at official $TORCH_OFFICIAL_URL"; exit 1
  else
    log "WARN: torch==$TORCH_VERSION not at $TORCH_INDEX_URL; falling back to OFFICIAL $TORCH_OFFICIAL_URL"
    log "      (tip: 'source /etc/network_turbo' makes the official-source download fast on AutoDL)"
    install_torch "$TORCH_OFFICIAL_URL" || { log "ERROR: torch==$TORCH_VERSION install failed on both indexes"; exit 1; }
    log "torch installed from $TORCH_OFFICIAL_URL (fallback)"
  fi
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

log "installing PyG companion wheels (pt22cu121)"
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
