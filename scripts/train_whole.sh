#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train_whole.sh coco   <train_images_dir> <train_annotations.json> <output_dir> [val_images_dir] [val_annotations.json]
  ./scripts/train_whole.sh kvasir <train_images_dir> <train_masks_dir>       <output_dir> [val_images_dir] [val_masks_dir]

Examples:
  ./scripts/train_whole.sh coco /data/endoscapes/train_seg /data/endoscapes/train_seg/annotation_coco.json outputs/train_endoscapes /data/endoscapes/val_seg /data/endoscapes/val_seg/annotation_coco.json
  ./scripts/train_whole.sh kvasir /data/kvasir-instrument/images /data/kvasir-instrument/masks outputs/train_kvasir
EOF
}

if [[ $# -lt 4 ]]; then
  usage
  exit 1
fi

die_missing() {
  echo "Missing or invalid path: $1"
  exit 1
}

DATASET_TYPE="$1"
TRAIN_IMAGES_DIR="$2"
DATASET_ARG="$3"
BASE_OUTPUT_DIR="$4"
VAL_IMAGES_DIR="${5:-}"
VAL_DATASET_ARG="${6:-}"

case "$DATASET_TYPE" in
  coco|kvasir)
    ;;
  *)
    echo "Unsupported dataset type: $DATASET_TYPE"
    usage
    exit 1
    ;;
esac

if [[ ! -x scripts/train_real_detector.sh ]]; then
  echo "scripts/train_real_detector.sh is missing or not executable."
  exit 1
fi

[[ -d "$TRAIN_IMAGES_DIR" ]] || die_missing "$TRAIN_IMAGES_DIR"
if [[ "$DATASET_TYPE" == "coco" ]]; then
  [[ -f "$DATASET_ARG" ]] || die_missing "$DATASET_ARG"
  if [[ -n "$VAL_IMAGES_DIR" ]]; then
    [[ -d "$VAL_IMAGES_DIR" ]] || die_missing "$VAL_IMAGES_DIR"
  fi
  if [[ -n "$VAL_DATASET_ARG" ]]; then
    [[ -f "$VAL_DATASET_ARG" ]] || die_missing "$VAL_DATASET_ARG"
  fi
else
  [[ -d "$DATASET_ARG" ]] || die_missing "$DATASET_ARG"
  if [[ -n "$VAL_IMAGES_DIR" ]]; then
    [[ -d "$VAL_IMAGES_DIR" ]] || die_missing "$VAL_IMAGES_DIR"
  fi
  if [[ -n "$VAL_DATASET_ARG" ]]; then
    [[ -d "$VAL_DATASET_ARG" ]] || die_missing "$VAL_DATASET_ARG"
  fi
fi

bash scripts/train_real_detector.sh \
  "$DATASET_TYPE" \
  "$TRAIN_IMAGES_DIR" \
  "$DATASET_ARG" \
  "$BASE_OUTPUT_DIR" \
  "$VAL_IMAGES_DIR" \
  "$VAL_DATASET_ARG"
