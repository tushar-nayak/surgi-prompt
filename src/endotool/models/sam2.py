from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from endotool.types import Detection
from endotool.utils.io import ensure_dir
from endotool.utils.tracking import match_detections_to_tracks


class Sam2Segmenter:
    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
        vos_optimized: bool = False,
    ) -> None:
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device
        self.autocast_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        image_model = build_sam2(str(config_path), str(checkpoint_path), device=device)
        self.image_predictor = SAM2ImagePredictor(image_model)
        self.video_predictor = build_sam2_video_predictor(
            str(config_path),
            str(checkpoint_path),
            device=device,
            vos_optimized=vos_optimized,
        )

    def refine_image_masks(self, image_bgr: np.ndarray, detections: list[Detection]) -> list[Detection]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), torch.autocast(device_type=self.device.split(":")[0], dtype=self.autocast_dtype, enabled=self.device.startswith("cuda")):
            self.image_predictor.set_image(image_rgb)
            for det in detections:
                masks, scores, _ = self.image_predictor.predict(
                    box=det.box_xyxy[None, :],
                    multimask_output=False,
                )
                det.mask = masks[0].astype(bool)
                det.score = max(det.score, float(scores[0]))
        return detections

    def track_video(
        self,
        video_path: str | Path,
        detections: list[Detection],
        output_frames_dir: str | Path | None = None,
        detector=None,
        prompt: str | None = None,
        reground_every: int = 0,
    ) -> tuple[dict[int, list[Detection]], float]:
        frame_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
        try:
            frame_paths = _extract_video_frames(video_path, frame_dir)
            state = self.video_predictor.init_state(video_path=str(frame_dir))
            object_seeds: dict[int, Detection] = {}
            with torch.inference_mode(), torch.autocast(device_type=self.device.split(":")[0], dtype=self.autocast_dtype, enabled=self.device.startswith("cuda")):
                for obj_id, det in enumerate(detections, start=1):
                    object_seeds[obj_id] = det
                    self.video_predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=obj_id,
                        box=det.box_xyxy.astype(np.float32),
                    )
                started = time.perf_counter()
                tracked: dict[int, list[Detection]] = {}
                reground_stats = {"events": 0, "matched_refreshes": 0, "new_tracks": 0}
                for frame_idx, object_ids, mask_logits in self.video_predictor.propagate_in_video(state):
                    current: list[Detection] = []
                    active_tracks: dict[int, Detection] = {}
                    image = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
                    for obj_id, mask_logit in zip(object_ids, mask_logits):
                        mask = mask_logit[0].cpu().numpy() > 0.0
                        box = _mask_to_box(mask)
                        if box is None:
                            continue
                        seed = object_seeds.get(int(obj_id))
                        if seed is None:
                            continue
                        det = Detection(
                            label=seed.label,
                            score=seed.score,
                            box_xyxy=box.astype(np.float32),
                            mask=mask,
                            label_id=seed.label_id,
                        )
                        current.append(det)
                        active_tracks[int(obj_id)] = det
                    if detector is not None and prompt and reground_every > 0 and frame_idx > 0 and frame_idx % reground_every == 0:
                        reground_stats["events"] += 1
                        refreshed = detector.detect(image, prompt)
                        refreshed = self.refine_image_masks(image, refreshed)
                        matches, unmatched = match_detections_to_tracks(refreshed, active_tracks)
                        for track_id, det in matches:
                            reground_stats["matched_refreshes"] += 1
                            object_seeds[track_id] = det
                            self.video_predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=frame_idx,
                                obj_id=track_id,
                                box=det.box_xyxy.astype(np.float32),
                            )
                        next_obj_id = max(object_seeds.keys(), default=0) + 1
                        for offset, det in enumerate(unmatched):
                            add_idx = next_obj_id + offset
                            reground_stats["new_tracks"] += 1
                            object_seeds[add_idx] = det
                            self.video_predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=frame_idx,
                                obj_id=add_idx,
                                box=det.box_xyxy.astype(np.float32),
                            )
                        if refreshed:
                            current = [det for _, det in matches] + unmatched
                    tracked[frame_idx] = current
                    if output_frames_dir is not None:
                        ensure_dir(output_frames_dir)
                        overlay = image.copy()
                        from endotool.utils.visualization import overlay_detections

                        overlay = overlay_detections(overlay, current)
                        cv2.imwrite(str(Path(output_frames_dir) / f"{frame_idx:05d}.png"), overlay)
                elapsed = time.perf_counter() - started
            fps = len(frame_paths) / max(elapsed, 1e-6)
            return tracked, fps
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)


def _extract_video_frames(video_path: str | Path, output_dir: Path) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    paths: list[Path] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        path = output_dir / f"{idx:05d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
        idx += 1
    cap.release()
    if not paths:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return paths


def _mask_to_box(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
