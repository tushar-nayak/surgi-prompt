#!/usr/bin/env bash
# ==============================================================================
# openvocab-surgical-tooling: Exhaustive Ablation Study Runner
# Designed for safe execution on a partitioned GPU (e.g., 33% of an NVIDIA 3090 / ~8GB VRAM)
# Runs sequentially to avoid OOM, logs metrics, and generates a comparative report.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_exhaustive_ablation.sh coco   <train_images_dir> <train_annotations.json> <output_dir> [val_images_dir] [val_annotations.json]
  ./scripts/run_exhaustive_ablation.sh kvasir <train_images_dir> <train_masks_dir>       <output_dir> [val_images_dir] [val_masks_dir]
EOF
}

if [[ $# -lt 4 ]]; then
  usage
  exit 1
fi

die_missing() {
  echo "Error: Missing or invalid path: $1"
  exit 1
}

DATASET_TYPE="$1"
TRAIN_IMAGES_DIR="$2"
DATASET_ARG="$3"
BASE_OUTPUT_DIR="$4"
VAL_IMAGES_DIR="${5:-}"
VAL_DATASET_ARG="${6:-}"

# Check environment
if [[ ! -f .venv/bin/activate ]]; then
  echo "Error: Virtual environment (.venv) is missing. Run step1_install_and_smoketest.sh first."
  exit 1
fi
source .venv/bin/activate

# Path validations
[[ -d "$TRAIN_IMAGES_DIR" ]] || die_missing "$TRAIN_IMAGES_DIR"
if [[ "$DATASET_TYPE" == "coco" ]]; then
  [[ -f "$DATASET_ARG" ]] || die_missing "$DATASET_ARG"
  if [[ -n "$VAL_IMAGES_DIR" ]]; then [[ -d "$VAL_IMAGES_DIR" ]] || die_missing "$VAL_IMAGES_DIR"; fi
  if [[ -n "$VAL_DATASET_ARG" ]]; then [[ -f "$VAL_DATASET_ARG" ]] || die_missing "$VAL_DATASET_ARG"; fi
else
  [[ -d "$DATASET_ARG" ]] || die_missing "$DATASET_ARG"
  if [[ -n "$VAL_IMAGES_DIR" ]]; then [[ -d "$VAL_IMAGES_DIR" ]] || die_missing "$VAL_IMAGES_DIR"; fi
  if [[ -n "$VAL_DATASET_ARG" ]]; then [[ -d "$VAL_DATASET_ARG" ]] || die_missing "$VAL_DATASET_ARG"; fi
fi

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
ABLATION_DIR="${BASE_OUTPUT_DIR%/}/ablation_study_${TIMESTAMP}"
mkdir -p "$ABLATION_DIR"

echo "======================================================================"
echo " Starting Exhaustive Ablation Study"
echo " Run Directory: $ABLATION_DIR"
echo " Dataset Type:  $DATASET_TYPE"
echo " GPU Device:    cuda (8GB VRAM Constraint Active)"
echo "======================================================================"

# Base parameters tailored for 8GB VRAM limits
# Uses batch size 1 and grad accumulation steps 8 for extreme safety
COMMON_TRAIN_ARGS=(
  --dataset-type "$DATASET_TYPE"
  --images-dir "$TRAIN_IMAGES_DIR"
  --device cuda
  --model-id IDEA-Research/grounding-dino-base
  --epochs 20
  --batch-size 1
  --grad-accum-steps 8
  --weight-decay 1e-4
  --early-stopping-patience 5
  --freeze-text-encoder
)

VAL_TRAIN_ARGS=()
if [[ -n "$VAL_IMAGES_DIR" ]]; then
  VAL_TRAIN_ARGS+=(--val-images-dir "$VAL_IMAGES_DIR")
fi
if [[ "$DATASET_TYPE" == "coco" ]]; then
  if [[ -n "$VAL_DATASET_ARG" ]]; then
    VAL_TRAIN_ARGS+=(--val-annotations "$VAL_DATASET_ARG")
  fi
  VAL_TRAIN_ARGS+=(--annotations "$DATASET_ARG")
else
  if [[ -n "$VAL_DATASET_ARG" ]]; then
    VAL_TRAIN_ARGS+=(--val-masks-dir "$VAL_DATASET_ARG")
  fi
  VAL_TRAIN_ARGS+=(--masks-dir "$DATASET_ARG")
  VAL_TRAIN_ARGS+=(--prompt-labels "surgical tool")
fi

declare -A EXPERIMENTS
EXPERIMENTS=(
  ["0_proposed_best"]="--learning-rate 1e-5 --backbone-lr-factor 0.1"
  ["1_no_augment"]="--learning-rate 1e-5 --backbone-lr-factor 0.1 --no-augment"
  ["2_frozen_backbone"]="--learning-rate 1e-5 --freeze-vision-backbone"
  ["3_backbone_lr_0_01"]="--learning-rate 1e-5 --backbone-lr-factor 0.01"
  ["4_higher_lr"]="--learning-rate 2e-5 --backbone-lr-factor 0.1"
)

# ------------------------------------------------------------------------------
# Phase 1: Sequential Training Ablations
# ------------------------------------------------------------------------------
for exp_name in "0_proposed_best" "1_no_augment" "2_frozen_backbone" "3_backbone_lr_0_01" "4_higher_lr"; do
  exp_args="${EXPERIMENTS[$exp_name]}"
  exp_output_dir="$ABLATION_DIR/$exp_name"
  echo "----------------------------------------------------------------------"
  echo " Running Training Run: $exp_name"
  echo " Args: $exp_args"
  echo " Output: $exp_output_dir"
  echo "----------------------------------------------------------------------"

  python scripts/train_grounding_dino.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    "${VAL_TRAIN_ARGS[@]}" \
    --output-dir "$exp_output_dir" \
    $exp_args \
    > "$ABLATION_DIR/train_${exp_name}.log" 2>&1 || {
      echo "Warning: Experiment $exp_name failed. Check $ABLATION_DIR/train_${exp_name}.log. Continuing..."
    }
done

# ------------------------------------------------------------------------------
# Find best checkpoint for inference & tracking sweeps
# ------------------------------------------------------------------------------
echo "----------------------------------------------------------------------"
echo " Selecting best checkpoint from all runs"
echo "----------------------------------------------------------------------"

BEST_RUN_NAME="0_proposed_best"
BEST_MAP="0.0"

for exp_name in "0_proposed_best" "1_no_augment" "2_frozen_backbone" "3_backbone_lr_0_01" "4_higher_lr"; do
  summary_file="$ABLATION_DIR/$exp_name/train_summary.json"
  if [[ -f "$summary_file" ]]; then
    val_map=$(python -c "import json; d=json.load(open('$summary_file')); print(d.get('best_val_bbox_map') or d.get('best_val_bbox_map_50') or 0.0)")
    echo "Run $exp_name achieved Val mAP: $val_map"
    if (( $(echo "$val_map > $BEST_MAP" | bc -l) )); then
      BEST_MAP="$val_map"
      BEST_RUN_NAME="$exp_name"
    fi
  fi
done

BEST_CHECKPOINT_DIR="$ABLATION_DIR/$BEST_RUN_NAME/best"
echo "Best run selected: $BEST_RUN_NAME (Val mAP: $BEST_MAP)"
echo "Best checkpoint path: $BEST_CHECKPOINT_DIR"

if [[ ! -d "$BEST_CHECKPOINT_DIR" ]]; then
  echo "Error: Best checkpoint not found at $BEST_CHECKPOINT_DIR. Falling back to proposed best checkpoint."
  BEST_CHECKPOINT_DIR="$ABLATION_DIR/0_proposed_best/best"
fi

# Obtain Prompt configurations from training summary
readarray -t RUN_INFO < <(python - "$ABLATION_DIR/$BEST_RUN_NAME/prompt_labels.json" <<'PY'
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

# ------------------------------------------------------------------------------
# Phase 2: Exhaustive Threshold Sweeps (Inference)
# ------------------------------------------------------------------------------
echo "----------------------------------------------------------------------"
echo " Starting Exhaustive Bounding Box & Text Threshold Sweep"
echo " Evaluating on Detection Validation Dataset"
echo "----------------------------------------------------------------------"

SWEEP_OUTPUT_DIR="$ABLATION_DIR/sweeps"
mkdir -p "$SWEEP_OUTPUT_DIR"

for bt in 0.20 0.25 0.30 0.35; do
  for tt in 0.15 0.20 0.25 0.30; do
    echo "-> Sweeping Box Thr: $bt, Text Thr: $tt"
    python scripts/evaluate_dataset.py \
      --dataset-type "$DATASET_TYPE" \
      --images-dir "$EVAL_IMAGES_DIR" \
      ${DATASET_TYPE=="coco"?"--annotations":"--masks-dir"} "$EVAL_DATASET_ARG" \
      --prompt "$PROMPT" \
      --output-dir "$SWEEP_OUTPUT_DIR/det_bt_${bt}_tt_${tt}" \
      --device cuda \
      --box-threshold "$bt" \
      --text-threshold "$tt" \
      --grounding-hf-model-id "$BEST_CHECKPOINT_DIR" \
      --grounding-force-hf-backend \
      > /dev/null 2>&1 || true
  done
done

# Find best threshold combination from Sweep
BEST_BT="0.30"
BEST_TT="0.25"
BEST_DET_MAP="0.0"

for bt in 0.20 0.25 0.30 0.35; do
  for tt in 0.15 0.20 0.25 0.30; do
    met_file="$SWEEP_OUTPUT_DIR/det_bt_${bt}_tt_${tt}/metrics.json"
    if [[ -f "$met_file" ]]; then
      det_map=$(python -c "import json; d=json.load(open('$met_file')); print(d.get('bbox_map', 0.0))")
      if (( $(echo "$det_map > $BEST_DET_MAP" | bc -l) )); then
        BEST_DET_MAP="$det_map"
        BEST_BT="$bt"
        BEST_TT="$tt"
      fi
    fi
  done
done

echo "Best Detection Thresholds found: Box Threshold = $BEST_BT, Text Threshold = $BEST_TT (mAP: $BEST_DET_MAP)"

# ------------------------------------------------------------------------------
# Phase 3: Exhaustive Tracking Parameter Sweep
# ------------------------------------------------------------------------------
if [[ "$DATASET_TYPE" == "coco" ]]; then
  echo "----------------------------------------------------------------------"
  echo " Starting Exhaustive Tracking Ablation & Parameter Sweep"
  echo " Evaluating on Tracking Dataset"
  echo "----------------------------------------------------------------------"

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

  for reground_every in 0 15 30; do
    for motion_thr in 0.1 0.2 0.3; do
      for area_thr in 0.3 0.45 0.6; do
        echo "-> Sweeping Reground Every: $reground_every, Motion Thr: $motion_thr, Area Thr: $area_thr"
        python scripts/evaluate_tracking.py \
          --images-dir "$EVAL_IMAGES_DIR" \
          --annotations "$EVAL_DATASET_ARG" \
          --video-id "$TRACKING_VIDEO_ID" \
          --prompt "$PROMPT" \
          --output-dir "$SWEEP_OUTPUT_DIR/track_reground_${reground_every}_motion_${motion_thr}_area_${area_thr}" \
          --device cuda \
          --box-threshold "$BEST_BT" \
          --text-threshold "$BEST_TT" \
          --reground-mode "fixed" \
          --reground-every "$reground_every" \
          --min-active-tracks 1 \
          --motion-iou-threshold "$motion_thr" \
          --area-ratio-threshold "$area_thr" \
          --grounding-hf-model-id "$BEST_CHECKPOINT_DIR" \
          --grounding-force-hf-backend \
          > /dev/null 2>&1 || true
      done
    done
  done
fi

# ------------------------------------------------------------------------------
# Phase 4: Compile Final Markdown and JSON Reports
# ------------------------------------------------------------------------------
echo "----------------------------------------------------------------------"
echo " Compiling Ablation Study Final Summary Report"
echo "----------------------------------------------------------------------"

python - "$ABLATION_DIR" "$DATASET_TYPE" "$BEST_RUN_NAME" "$BEST_BT" "$BEST_TT" <<'PY'
import json, sys
from pathlib import Path

ablation_dir = Path(sys.argv[1])
dataset_type = sys.argv[2]
best_run = sys.argv[3]
best_bt = sys.argv[4]
best_tt = sys.argv[5]

# 1. Gather Training Results
train_rows = []
for exp_name in ["0_proposed_best", "1_no_augment", "2_frozen_backbone", "3_backbone_lr_0_01", "4_higher_lr"]:
    summary_file = ablation_dir / exp_name / "train_summary.json"
    history_file = ablation_dir / exp_name / "history.json"
    
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        history = json.loads(history_file.read_text()) if history_file.exists() else []
        epochs_run = len(history)
        
        train_rows.append({
            "Experiment": exp_name,
            "Best Val Loss": summary.get("best_val_loss"),
            "Best Val mAP": summary.get("best_val_bbox_map") or summary.get("best_val_bbox_map_50") or "N/A",
            "Final Train Loss": summary.get("final_train_loss"),
            "Epochs Run (Early Stopped)": epochs_run
        })

# 2. Gather Detection Sweep Results
det_rows = []
sweeps_dir = ablation_dir / "sweeps"
if sweeps_dir.exists():
    for det_dir in sweeps_dir.glob("det_bt_*_tt_*"):
        metrics_file = det_dir / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            parts = det_dir.name.split("_")
            det_rows.append({
                "Box Threshold": parts[2],
                "Text Threshold": parts[4],
                "bbox_map": metrics.get("bbox_map", 0.0),
                "bbox_map_50": metrics.get("bbox_map_50", 0.0),
                "mean_mask_iou": metrics.get("mean_mask_iou", 0.0),
                "fps": metrics.get("fps", 0.0)
            })
    det_rows = sorted(det_rows, key=lambda x: x["bbox_map"], reverse=True)

# 3. Gather Tracking Sweep Results
track_rows = []
if sweeps_dir.exists():
    for tr_dir in sweeps_dir.glob("track_reground_*_motion_*_area_*"):
        metrics_file = tr_dir / "tracking_metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            parts = tr_dir.name.split("_")
            track_rows.append({
                "Reground Every": parts[2],
                "Motion Threshold": parts[4],
                "Area Ratio Threshold": parts[6],
                "tracking_mean_iou": metrics.get("mean_frame_best_iou", 0.0),
                "tracking_recall_50": metrics.get("mean_frame_recall_at_50", 0.0),
                "fps": metrics.get("fps", 0.0)
            })
    track_rows = sorted(track_rows, key=lambda x: x["tracking_mean_iou"], reverse=True)

# Write Consolidated JSON
consolidated = {
    "dataset_type": dataset_type,
    "best_training_run": best_run,
    "best_box_threshold": best_bt,
    "best_text_threshold": best_tt,
    "training_ablations": train_rows,
    "detection_threshold_sweeps": det_rows[:15],  # Top 15
    "tracking_parameter_sweeps": track_rows[:15]   # Top 15
}
(ablation_dir / "ablation_summary.json").write_text(json.dumps(consolidated, indent=2))

# Write Markdown Report
md = []
md.append("# Exhaustive Ablation Study & Threshold Sweep Report\n")
md.append("## 🏆 Champion Configurations")
md.append(f"- **Best Trained Model**: `{best_run}`")
md.append(f"- **Best Detection Thresholds**: Box Threshold = `{best_bt}`, Text Threshold = `{best_tt}`")
if track_rows:
    top_t = track_rows[0]
    md.append(f"- **Best Tracking Config**: Reground Every = `{top_t['Reground Every']}`, Motion Thr = `{top_t['Motion Threshold']}`, Area Thr = `{top_t['Area Ratio Threshold']}` (Tracking Mean IoU: `{top_t['tracking_mean_iou']:.4f}`)\n")

md.append("## 📊 1. Training Pipeline Ablation Study")
md.append("| Experiment | Best Val Loss | Best Val mAP | Final Train Loss | Epochs Run |")
md.append("| :--- | :---: | :---: | :---: | :---: |")
for row in train_rows:
    map_str = f"{row['Best Val mAP']:.4f}" if isinstance(row['Best Val mAP'], float) else str(row['Best Val mAP'])
    md.append(f"| {row['Experiment']} | {row['Best Val Loss']:.4f} | {map_str} | {row['Final Train Loss']:.4f} | {row['Epochs Run (Early Stopped)']} |")

md.append("\n## 🔍 2. Bounding Box & Text Threshold Sweep (Top 10)")
md.append("| Box Threshold | Text Threshold | bbox mAP | bbox mAP@50 | Mean Mask IoU | FPS |")
md.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
for row in det_rows[:10]:
    md.append(f"| {row['Box Threshold']} | {row['Text Threshold']} | {row['bbox_map']:.4f} | {row['bbox_map_50']:.4f} | {row['mean_mask_iou']:.4f} | {row['fps']:.2f} |")

if track_rows:
    md.append("\n## 🔄 3. Tracking Parameter Grid Sweep (Top 10)")
    md.append("| Reground Every | Motion Threshold | Area Ratio Threshold | Tracking Mean IoU | Recall@50 | FPS |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in track_rows[:10]:
        md.append(f"| {row['Reground Every']} | {row['Motion Threshold']} | {row['Area Ratio Threshold']} | {row['tracking_mean_iou']:.4f} | {row['tracking_recall_50']:.4f} | {row['fps']:.2f} |")

(ablation_dir / "ablation_summary.md").write_text("\n".join(md))
print("Final ablation report written to:", str(ablation_dir / "ablation_summary.md"))

PY

echo "======================================================================"
echo " Ablation study complete! Report located at:"
echo " $ABLATION_DIR/ablation_summary.md"
echo "======================================================================"
