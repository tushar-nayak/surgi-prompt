#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage:"
  echo "  $0 coco <train_images_dir> <train_annotations.json> <output_dir> [val_images_dir] [val_annotations.json]"
  echo "  $0 kvasir <train_images_dir> <train_masks_dir> <output_dir> [val_images_dir] [val_masks_dir]"
  exit 1
fi

die_missing() {
  echo "Missing or invalid path: $1"
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
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
pip install --no-build-isolation "git+https://github.com/IDEA-Research/GroundingDINO.git"

pip install -e .

DATASET_TYPE="$1"
TRAIN_IMAGES_DIR="$2"
DATASET_ARG="$3"
BASE_OUTPUT_DIR="$4"
VAL_IMAGES_DIR="${5:-}"
VAL_DATASET_ARG="${6:-}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${DATASET_TYPE}_train_${TIMESTAMP}"
RUN_DIR="${BASE_OUTPUT_DIR%/}/${RUN_NAME}"
mkdir -p "$RUN_DIR"

[[ -d "$TRAIN_IMAGES_DIR" ]] || die_missing "$TRAIN_IMAGES_DIR"
[[ -d "$(dirname "$BASE_OUTPUT_DIR")" ]] || die_missing "$(dirname "$BASE_OUTPUT_DIR")"
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

COMMON_ARGS=(
  --dataset-type "$DATASET_TYPE"
  --images-dir "$TRAIN_IMAGES_DIR"
  --output-dir "$RUN_DIR/train"
  --device cuda
  --model-id IDEA-Research/grounding-dino-base
  --epochs 5
  --batch-size 1
  --grad-accum-steps 8
  --learning-rate 1e-5
  --weight-decay 1e-4
  --freeze-text-encoder
  --freeze-vision-backbone
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

readarray -t RUN_INFO < <(python - "$RUN_DIR/train/prompt_labels.json" <<'PY'
import json, sys
labels = json.loads(open(sys.argv[1]).read())
prompt = " . ".join(labels) + " ."
print(prompt)
print(",".join(labels))
PY
)
PROMPT="${RUN_INFO[0]}"
PROMPT_LABELS_CSV="${RUN_INFO[1]}"
EVAL_IMAGES_DIR="${VAL_IMAGES_DIR:-$TRAIN_IMAGES_DIR}"
EVAL_DATASET_ARG="${VAL_DATASET_ARG:-$DATASET_ARG}"
DETECTION_EVAL_DIR="$RUN_DIR/eval_detection"
TRACKING_EVAL_DIR="$RUN_DIR/eval_tracking"

if [[ "$DATASET_TYPE" == "coco" ]]; then
  python scripts/evaluate_dataset.py \
    --dataset-type coco \
    --images-dir "$EVAL_IMAGES_DIR" \
    --annotations "$EVAL_DATASET_ARG" \
    --prompt "$PROMPT" \
    --output-dir "$DETECTION_EVAL_DIR" \
    --device cuda \
    --grounding-hf-model-id "$RUN_DIR/train/best" \
    --grounding-force-hf-backend

  TRACKING_VIDEO_ID="$(python - "$EVAL_DATASET_ARG" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
counter = Counter()
for image in payload["images"]:
    raw = image.get("video_id", "") or Path(image["file_name"]).stem.split("_")[0]
    counter[str(raw)] += 1
if not counter:
    raise SystemExit(1)
print(counter.most_common(1)[0][0])
PY
)"
  python scripts/evaluate_tracking.py \
    --images-dir "$EVAL_IMAGES_DIR" \
    --annotations "$EVAL_DATASET_ARG" \
    --video-id "$TRACKING_VIDEO_ID" \
    --prompt "$PROMPT" \
    --output-dir "$TRACKING_EVAL_DIR" \
    --device cuda \
    --reground-mode fixed \
    --reground-every 0 \
    --min-active-tracks 1 \
    --motion-iou-threshold 0.1 \
    --area-ratio-threshold 0.3 \
    --grounding-hf-model-id "$RUN_DIR/train/best" \
    --grounding-force-hf-backend
fi

python - "$RUN_DIR" "$BASE_OUTPUT_DIR" "$DATASET_TYPE" "$RUN_NAME" "$TRAIN_IMAGES_DIR" "$EVAL_IMAGES_DIR" "$PROMPT" "$PROMPT_LABELS_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
base_dir = Path(sys.argv[2])
dataset_type = sys.argv[3]
run_name = sys.argv[4]
train_images_dir = sys.argv[5]
eval_images_dir = sys.argv[6]
prompt = sys.argv[7]
prompt_labels_csv = sys.argv[8]

train_summary = json.loads((run_dir / "train" / "train_summary.json").read_text())
detection_metrics = None
tracking_metrics = None
if (run_dir / "eval_detection" / "metrics.json").exists():
    detection_metrics = json.loads((run_dir / "eval_detection" / "metrics.json").read_text())
if (run_dir / "eval_tracking" / "tracking_metrics.json").exists():
    tracking_metrics = json.loads((run_dir / "eval_tracking" / "tracking_metrics.json").read_text())

summary = {
    "run_name": run_name,
    "run_dir": str(run_dir.resolve()),
    "dataset_type": dataset_type,
    "train_images_dir": train_images_dir,
    "eval_images_dir": eval_images_dir,
    "prompt": prompt,
    "prompt_labels_csv": prompt_labels_csv,
    "train": train_summary,
    "detection_eval": detection_metrics,
    "tracking_eval": tracking_metrics,
}
(run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

jsonl_path = base_dir / "runs_summary.jsonl"
with jsonl_path.open("a") as fh:
    fh.write(json.dumps(summary) + "\n")

row = {
    "run_name": run_name,
    "run_dir": str(run_dir.resolve()),
    "dataset_type": dataset_type,
    "prompt_labels_csv": prompt_labels_csv,
    "train_samples": train_summary.get("num_train_samples"),
    "val_samples": train_summary.get("num_val_samples"),
    "best_val_loss": train_summary.get("best_val_loss"),
    "final_train_loss": train_summary.get("final_train_loss"),
    "final_val_loss": train_summary.get("final_val_loss"),
    "bbox_map": None if detection_metrics is None else detection_metrics.get("bbox_map"),
    "segm_map": None if detection_metrics is None else detection_metrics.get("segm_map"),
    "mean_mask_iou": None if detection_metrics is None else detection_metrics.get("mean_mask_iou"),
    "eval_fps": None if detection_metrics is None else detection_metrics.get("fps"),
    "tracking_mean_iou": None if tracking_metrics is None else tracking_metrics.get("mean_frame_best_iou"),
    "tracking_recall50": None if tracking_metrics is None else tracking_metrics.get("mean_frame_recall_at_50"),
    "tracking_fps": None if tracking_metrics is None else tracking_metrics.get("fps"),
    "tracking_video_id": None if tracking_metrics is None else tracking_metrics.get("video_id"),
}

csv_path = base_dir / "runs_summary.csv"
fieldnames = list(row.keys())
write_header = not csv_path.exists()
with csv_path.open("a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY

echo "Run directory: $RUN_DIR"
echo "Training artifacts: $RUN_DIR/train"
if [[ -d "$DETECTION_EVAL_DIR" ]]; then
  echo "Detection eval: $DETECTION_EVAL_DIR"
fi
if [[ -d "$TRACKING_EVAL_DIR" ]]; then
  echo "Tracking eval: $TRACKING_EVAL_DIR"
fi
echo "Run summary: $RUN_DIR/run_summary.json"
