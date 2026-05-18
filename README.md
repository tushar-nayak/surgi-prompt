# Open-Vocabulary Surgical Tool Detection using Grounding DINO + SAM2

Pipeline:

`endoscopic frame/video -> text prompt -> Grounding DINO boxes -> SAM2 mask refinement -> video tracking + overlay -> mAP / IoU / FPS / failure cases`

This project supports both inference and real-data fine-tuning. It is set up for real endoscopic and laparoscopic datasets only. No synthetic dataset loader is included.

## Current Results

These are repository-tracked smoke results on real Endoscapes data.

### Step 1: real image and real video smoke run

Single real Endoscapes frame:

| Metric | Value |
| --- | ---: |
| image fps | `0.5137` |
| detections | `4` |

Real 120-frame Endoscapes sequence video:

| Run | FPS | Frames | Re-grounding |
| --- | ---: | ---: | ---: |
| baseline tracking | `5.3907` | `120` | `0` |
| periodic re-grounding | `5.5816` | `120` | `30` |

### Step 2: class-aware evaluation on a real Endoscapes subset

Tool-only subset from Endoscapes `test`:

| Metric | Value |
| --- | ---: |
| bbox mAP | `0.2139` |
| bbox mAP@50 | `0.2492` |
| segm mAP | `0.0990` |
| segm mAP@50 | `0.0990` |
| mean mask IoU | `0.3926` |
| eval fps | `2.1140` |
| images | `9` |

## Supported real datasets

- `Endoscapes2023` for COCO-style tool/anatomy boxes and instance masks
- `Kvasir-Instrument` for real GI tool masks and derived boxes
- Other real datasets can be used through the generic COCO loader if they are exported to COCO

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Checkpoints

```bash
bash scripts/download_checkpoints.sh
```

Default files:

- `checkpoints/groundingdino_swint_ogc.pth`
- `checkpoints/sam2.1_hiera_large.pt`

## Run on one image

```bash
python scripts/run_image.py \
  --image /path/to/frame.jpg \
  --prompt "surgical tool . forceps . grasper . catheter . guidewire ." \
  --output outputs/frame_overlay.jpg
```

## Run on video

```bash
python scripts/run_video.py \
  --video /path/to/video.mp4 \
  --prompt "surgical tool . forceps . grasper . catheter . guidewire ." \
  --output-dir outputs/video_run
```

Outputs:

- overlay video
- per-frame masks
- JSON summary with `fps`, detections, and track metadata

## Evaluate on real datasets

Endoscapes / generic COCO:

```bash
python scripts/evaluate_dataset.py \
  --dataset-type coco \
  --images-dir /data/endoscapes/test_seg \
  --annotations /data/endoscapes/test_seg/annotation_coco.json \
  --prompt "tool . forceps . grasper . hook . clip applier . scissors ." \
  --output-dir outputs/endoscapes_eval \
  --max-images 100
```

Kvasir-Instrument:

```bash
python scripts/evaluate_dataset.py \
  --dataset-type kvasir \
  --images-dir /data/kvasir-instrument/images \
  --masks-dir /data/kvasir-instrument/masks \
  --prompt "surgical tool . forceps . snare . balloon . catheter ." \
  --output-dir outputs/kvasir_eval

## Train on real datasets

Fine-tune the Hugging Face Grounding DINO backend on real COCO-style endoscopy data:

```bash
bash scripts/train_real_detector.sh \
  coco \
  /data/endoscapes/train_seg \
  /data/endoscapes/train_seg/annotation_coco.json \
  outputs/train_endoscapes \
  /data/endoscapes/val_seg \
  /data/endoscapes/val_seg/annotation_coco.json
```

Kvasir-Instrument:

```bash
bash scripts/train_real_detector.sh \
  kvasir \
  /data/kvasir-instrument/images \
  /data/kvasir-instrument/masks \
  outputs/train_kvasir
```

The shell wrapper targets the stronger `IDEA-Research/grounding-dino-base` checkpoint by default and keeps training on real data only.
```

## Notes

- Grounding DINO prompting follows the official recommendation to separate class names with periods.
- SAM 2 video propagation uses box prompts from frame 0 by default.
- `scripts/step1_install_and_smoketest.sh` is incremental and skips reinstalling dependencies or redownloading checkpoints when they are already present.
- `scripts/step2_class_aware_eval.sh` accepts an optional `max_images` argument for bounded real-data smoke evaluation.
- This repository includes a GitHub Pages site under `docs/` with tracked run artifacts and current results.
- This scaffold avoids any synthetic train/val/test split. Use official real-data splits from the dataset publishers.

## Sources

- Grounding DINO official repo: https://github.com/IDEA-Research/GroundingDINO
- SAM 2 official repo: https://github.com/facebookresearch/sam2
- Endoscapes official repo: https://github.com/CAMMA-public/Endoscapes
- Kvasir-Instrument dataset: https://datasets.simula.no/kvasir-instrument/
