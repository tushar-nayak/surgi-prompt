#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 4 ]]; then
  echo "Usage:"
  echo "  $0 coco <images_dir> <annotations_json> <output_dir> [prompt] [max_images]"
  echo "  $0 kvasir <images_dir> <masks_dir> <output_dir> [prompt] [max_images]"
  exit 1
fi

DATASET_TYPE="$1"
IMAGES_DIR="$2"
THIRD_ARG="$3"
OUTPUT_DIR="$4"
PROMPT="${5:-surgical tool . forceps . grasper . catheter . guidewire . snare . balloon .}"
MAX_IMAGES="${6:-0}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f .venv/bin/activate ]]; then
  echo ".venv is missing. Run scripts/step1_install_and_smoketest.sh first."
  exit 1
fi

source .venv/bin/activate

if [[ "${DATASET_TYPE}" == "coco" ]]; then
  python scripts/evaluate_dataset.py \
    --dataset-type coco \
    --images-dir "${IMAGES_DIR}" \
    --annotations "${THIRD_ARG}" \
    --prompt "${PROMPT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --label-map configs/tool_label_map.yaml \
    --max-images "${MAX_IMAGES}"
else
  python scripts/evaluate_dataset.py \
    --dataset-type kvasir \
    --images-dir "${IMAGES_DIR}" \
    --masks-dir "${THIRD_ARG}" \
    --prompt "${PROMPT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --label-map configs/tool_label_map.yaml \
    --max-images "${MAX_IMAGES}"
fi
