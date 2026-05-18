#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

IMAGE_PATH="${1:-}"
VIDEO_PATH="${2:-}"
PROMPT="${3:-surgical tool . forceps . grasper . catheter . guidewire .}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ ! -x .venv/bin/python ]]; then
  rm -rf .venv
  "${PYTHON_BIN}" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip

ensure_python_package() {
  local module_name="$1"
  shift
  if ! python -c "import ${module_name}" >/dev/null 2>&1; then
    pip install "$@"
  fi
}

ensure_torch_stack() {
  if ! python - <<'PY' >/dev/null 2>&1
import torch
import torchvision
PY
  then
    pip install torch torchvision
  fi
}

ensure_python_package wheel wheel setuptools
ensure_torch_stack
ensure_python_package cv2 opencv-python
ensure_python_package yaml pyyaml
ensure_python_package tqdm tqdm
ensure_python_package pycocotools pycocotools
if ! python - <<'PY' >/dev/null 2>&1
from torchmetrics.detection.mean_ap import MeanAveragePrecision
PY
then
  pip install "torchmetrics[detection]"
fi
ensure_python_package transformers "transformers<5"
ensure_python_package sam2 "git+https://github.com/facebookresearch/sam2.git"
ensure_python_package groundingdino --no-build-isolation "git+https://github.com/IDEA-Research/GroundingDINO.git"

pip install -e .

if [[ ! -s checkpoints/groundingdino_swint_ogc.pth || ! -s checkpoints/sam2.1_hiera_large.pt ]]; then
  bash scripts/download_checkpoints.sh
fi

if [[ -n "${IMAGE_PATH}" ]]; then
  python scripts/run_image.py \
    --image "${IMAGE_PATH}" \
    --prompt "${PROMPT}" \
    --output outputs/smoke_frame_overlay.jpg \
    --device "${DEVICE}"
fi

if [[ -n "${VIDEO_PATH}" ]]; then
  python scripts/run_video.py \
    --video "${VIDEO_PATH}" \
    --prompt "${PROMPT}" \
    --output-dir outputs/smoke_video \
    --device "${DEVICE}"
fi
