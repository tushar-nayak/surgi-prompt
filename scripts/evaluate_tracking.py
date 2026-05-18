from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from endotool.pipeline import OpenVocabSurgicalPipeline, PipelineConfig
from endotool.utils.io import ensure_dir, write_json
from endotool.utils.tracking import box_iou


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reground-every", type=int, default=0)
    parser.add_argument("--reground-mode", choices=["fixed", "adaptive", "hybrid"], default="fixed")
    parser.add_argument("--min-active-tracks", type=int, default=1)
    parser.add_argument("--motion-iou-threshold", type=float, default=0.2)
    parser.add_argument("--area-ratio-threshold", type=float, default=0.45)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    frames, gt_by_frame = load_video_keyframes(args.images_dir, args.annotations, args.video_id)
    video_path = build_temp_video(frames)

    try:
        pipeline = OpenVocabSurgicalPipeline(PipelineConfig(device=args.device))
        first_frame = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        detections = pipeline.detector.detect(first_frame, args.prompt)
        detections = pipeline.segmenter.refine_image_masks(first_frame, detections)
        tracked, fps, reground_stats = pipeline.segmenter.track_video(
            video_path,
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
        metrics = compute_tracking_metrics(tracked, gt_by_frame)
        metrics["fps"] = fps
        metrics["reground_mode"] = args.reground_mode
        metrics["reground_every"] = args.reground_every
        metrics["min_active_tracks"] = args.min_active_tracks
        metrics["motion_iou_threshold"] = args.motion_iou_threshold
        metrics["area_ratio_threshold"] = args.area_ratio_threshold
        metrics["reground_stats"] = reground_stats
        metrics["video_id"] = args.video_id
        metrics["num_annotated_frames"] = len(gt_by_frame)
        write_json(output_dir / "tracking_metrics.json", metrics)
        render_video(video_path, tracked, output_dir / "overlay.mp4")
    finally:
        Path(video_path).unlink(missing_ok=True)


def load_video_keyframes(images_dir: str, annotations_path: str, video_id: str) -> tuple[list[Path], dict[int, list[np.ndarray]]]:
    payload = json.loads(Path(annotations_path).read_text())
    images = [img for img in payload["images"] if str(img.get("video_id", "") or Path(img["file_name"]).stem.split("_")[0]) == str(video_id)]
    images = sorted(images, key=lambda item: item["file_name"])
    selected_ids = {img["id"] for img in images}
    gt_by_image: dict[int, list[np.ndarray]] = {img["id"]: [] for img in images}
    for ann in payload["annotations"]:
        if ann["image_id"] not in selected_ids:
            continue
        if ann["category_id"] != 6:
            continue
        x, y, w, h = ann["bbox"]
        gt_by_image[ann["image_id"]].append(np.array([x, y, x + w, y + h], dtype=np.float32))
    frame_paths = [Path(images_dir) / img["file_name"] for img in images]
    gt_by_frame = {idx: gt_by_image[img["id"]] for idx, img in enumerate(images)}
    return frame_paths, gt_by_frame


def build_temp_video(frame_paths: list[Path]) -> str:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(frame_paths[0])
    height, width = first.shape[:2]
    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    handle.close()
    writer = cv2.VideoWriter(handle.name, cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height))
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(frame_path)
        writer.write(frame)
    writer.release()
    return handle.name


def compute_tracking_metrics(tracked: dict[int, list], gt_by_frame: dict[int, list[np.ndarray]]) -> dict[str, float | list[float]]:
    frame_best_ious: list[float] = []
    frame_recalls: list[float] = []
    for frame_idx, gt_boxes in gt_by_frame.items():
        preds = tracked.get(frame_idx, [])
        pred_boxes = [pred.box_xyxy for pred in preds]
        if not gt_boxes:
            continue
        best_ious = []
        recalled = 0
        for gt in gt_boxes:
            best = max((box_iou(gt, pred) for pred in pred_boxes), default=0.0)
            best_ious.append(best)
            if best >= 0.5:
                recalled += 1
        frame_best_ious.append(float(np.mean(best_ious)))
        frame_recalls.append(recalled / len(gt_boxes))
    return {
        "mean_frame_best_iou": float(np.mean(frame_best_ious)) if frame_best_ious else 0.0,
        "mean_frame_recall_at_50": float(np.mean(frame_recalls)) if frame_recalls else 0.0,
        "frame_best_ious": frame_best_ious,
        "frame_recalls_at_50": frame_recalls,
    }


def render_video(video_path: str, tracked: dict[int, list], output_path: Path) -> None:
    from endotool.utils.visualization import overlay_detections

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 2.0
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
