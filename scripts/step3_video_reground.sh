#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <video_path> <output_dir> [reground_every_frames] [prompt]"
  exit 1
fi

VIDEO_PATH="$1"
OUTPUT_DIR="$2"
REGROUND_EVERY="${3:-30}"
PROMPT="${4:-surgical tool . forceps . grasper . catheter . guidewire .}"
DEVICE="${DEVICE:-cuda}"

source .venv/bin/activate

python scripts/run_video.py \
  --video "${VIDEO_PATH}" \
  --prompt "${PROMPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --reground-every "${REGROUND_EVERY}"
