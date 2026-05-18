from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from endotool.pipeline import OpenVocabSurgicalPipeline, PipelineConfig
from endotool.utils.io import ensure_dir, write_json
from endotool.utils.visualization import overlay_detections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reground-every", type=int, default=0)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    pipeline = OpenVocabSurgicalPipeline(PipelineConfig(device=args.device))

    cap = cv2.VideoCapture(args.video)
    ok, first_frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {args.video}")

    detections = pipeline.detector.detect(first_frame, args.prompt)
    detections = pipeline.segmenter.refine_image_masks(first_frame, detections)
    tracked, fps = pipeline.segmenter.track_video(
        args.video,
        detections,
        output_frames_dir=output_dir / "frames",
        detector=pipeline.detector,
        prompt=args.prompt,
        reground_every=args.reground_every,
    )

    _render_video(args.video, tracked, output_dir / "overlay.mp4")
    write_json(
        output_dir / "summary.json",
        {
            "video": args.video,
            "prompt": args.prompt,
            "fps": fps,
            "num_frames": len(tracked),
            "reground_every": args.reground_every,
            "seed_detections": [
                {"label": det.label, "score": det.score, "box_xyxy": det.box_xyxy.tolist()}
                for det in detections
            ],
        },
    )


def _render_video(video_path: str, tracked: dict[int, list], output_path: Path) -> None:
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(overlay_detections(frame, tracked.get(frame_idx, [])))
        frame_idx += 1
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
