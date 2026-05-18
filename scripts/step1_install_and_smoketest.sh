#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

IMAGE_PATH="${1:-}"
VIDEO_PATH="${2:-}"
PROMPT="${3:-surgical tool . forceps . grasper . catheter . guidewire .}"
DEVICE="${DEVICE:-cuda}"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
bash scripts/download_checkpoints.sh

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
