# Open-Vocabulary Surgical Tool Detection using Grounding DINO + SAM2

Pipeline:

`endoscopic frame/video -> text prompt -> Grounding DINO boxes -> SAM2 mask refinement -> video tracking + overlay -> mAP / IoU / FPS / failure cases`

This project is inference-first. It is set up for real endoscopic and laparoscopic datasets only. No synthetic dataset loader is included.

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
  --output-dir outputs/endoscapes_eval
```

Kvasir-Instrument:

```bash
python scripts/evaluate_dataset.py \
  --dataset-type kvasir \
  --images-dir /data/kvasir-instrument/images \
  --masks-dir /data/kvasir-instrument/masks \
  --prompt "surgical tool . forceps . snare . balloon . catheter ." \
  --output-dir outputs/kvasir_eval
```

## Notes

- Grounding DINO prompting follows the official recommendation to separate class names with periods.
- SAM 2 video propagation uses box prompts from frame 0 by default.
- This scaffold avoids any synthetic train/val/test split. Use official real-data splits from the dataset publishers.

## Sources

- Grounding DINO official repo: https://github.com/IDEA-Research/GroundingDINO
- SAM 2 official repo: https://github.com/facebookresearch/sam2
- Endoscapes official repo: https://github.com/CAMMA-public/Endoscapes
- Kvasir-Instrument dataset: https://datasets.simula.no/kvasir-instrument/
