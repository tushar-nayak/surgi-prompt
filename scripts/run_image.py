from __future__ import annotations

import argparse
from pathlib import Path

from endotool.pipeline import OpenVocabSurgicalPipeline, PipelineConfig
from endotool.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    args = parser.parse_args()

    pipeline = OpenVocabSurgicalPipeline(PipelineConfig(
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    ))
    detections, fps = pipeline.run_image(args.image, args.prompt)
    pipeline.annotate_image(args.image, detections, args.output)
    write_json(
        Path(args.output).with_suffix(".json"),
        {
            "image": args.image,
            "prompt": args.prompt,
            "fps": fps,
            "detections": [
                {
                    "label": det.label,
                    "score": det.score,
                    "box_xyxy": det.box_xyxy.tolist(),
                    "mask_pixels": int(det.mask.sum()) if det.mask is not None else 0,
                }
                for det in detections
            ],
        },
    )


if __name__ == "__main__":
    main()
