#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage:"
  echo "  $0 coco <train_images_dir> <train_annotations.json> <output_dir> [val_images_dir] [val_annotations.json]"
  echo "  $0 kvasir <train_images_dir> <train_masks_dir> <output_dir> [val_images_dir] [val_masks_dir]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install -r requirements.txt

DATASET_TYPE="$1"
TRAIN_IMAGES_DIR="$2"
DATASET_ARG="$3"
OUTPUT_DIR="$4"
VAL_IMAGES_DIR="${5:-}"
VAL_DATASET_ARG="${6:-}"

COMMON_ARGS=(
  --dataset-type "$DATASET_TYPE"
  --images-dir "$TRAIN_IMAGES_DIR"
  --output-dir "$OUTPUT_DIR"
  --device cuda
  --model-id IDEA-Research/grounding-dino-base
  --epochs 5
  --batch-size 2
  --grad-accum-steps 4
  --learning-rate 1e-5
  --weight-decay 1e-4
)

VAL_ARGS=()
if [[ -n "$VAL_IMAGES_DIR" ]]; then
  VAL_ARGS+=(--val-images-dir "$VAL_IMAGES_DIR")
fi

if [[ "$DATASET_TYPE" == "coco" ]]; then
  if [[ -n "$VAL_DATASET_ARG" ]]; then
    VAL_ARGS+=(--val-annotations "$VAL_DATASET_ARG")
  fi
  python scripts/train_grounding_dino.py \
    "${COMMON_ARGS[@]}" \
    --annotations "$DATASET_ARG" \
    "${VAL_ARGS[@]}"
elif [[ "$DATASET_TYPE" == "kvasir" ]]; then
  if [[ -n "$VAL_DATASET_ARG" ]]; then
    VAL_ARGS+=(--val-masks-dir "$VAL_DATASET_ARG")
  fi
  python scripts/train_grounding_dino.py \
    "${COMMON_ARGS[@]}" \
    --masks-dir "$DATASET_ARG" \
    "${VAL_ARGS[@]}" \
    --prompt-labels "surgical tool"
else
  echo "Unsupported dataset type: $DATASET_TYPE"
  exit 1
fi
