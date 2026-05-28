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
    parser.add_argument("--reground-mode", choices=["fixed", "adaptive", "hybrid"], default="fixed")
    parser.add_argument("--min-active-tracks", type=int, default=1)
    parser.add_argument("--motion-iou-threshold", type=float, default=0.2)
    parser.add_argument("--area-ratio-threshold", type=float, default=0.45)
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-hf-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--grounding-force-hf-backend", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    pipeline = OpenVocabSurgicalPipeline(
        PipelineConfig(
            device=args.device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            grounding_hf_model_id=args.grounding_hf_model_id,
            grounding_force_hf_backend=args.grounding_force_hf_backend,
        )
    )

    cap = cv2.VideoCapture(args.video)
    ok, first_frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {args.video}")

    detections = pipeline.detector.detect(first_frame, args.prompt)
    detections = pipeline.segmenter.refine_image_masks(first_frame, detections)
    tracked, fps, reground_stats = pipeline.segmenter.track_video(
        args.video,
        detections,
        output_frames_dir=output_dir / "frames",
        detector=pipeline.detector,
        prompt=args.prompt,
        reground_every=args.reground_every,
        reground_mode=args.reground_mode,
        min_active_tracks=args.min_active_tracks,
        motion_iou_threshold=args.motion_iou_threshold,
        area_ratio_threshold=args.area_ratio_threshold,
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
            "reground_mode": args.reground_mode,
            "mean_active_tracks": (sum(len(items) for items in tracked.values()) / max(len(tracked), 1)),
            "reground_stats": reground_stats,
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
