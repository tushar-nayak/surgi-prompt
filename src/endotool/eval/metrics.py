from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from endotool.types import DatasetSample, Detection
from endotool.utils.visualization import overlay_detections


class Evaluator:
    def __init__(self) -> None:
        self.det_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        self.seg_map = MeanAveragePrecision(iou_type="segm")
        self.mask_ious: list[float] = []

    def update(self, sample: DatasetSample, predictions: list[Detection]) -> None:
        pred_boxes = np.array([pred.box_xyxy for pred in predictions], dtype=np.float32) if predictions else np.zeros((0, 4), dtype=np.float32)
        pred_scores = np.array([pred.score for pred in predictions], dtype=np.float32) if predictions else np.zeros((0,), dtype=np.float32)
        pred_labels = np.array([pred.label_id or 1 for pred in predictions], dtype=np.int64)
        target_labels = np.array(sample.label_ids or [1] * len(sample.labels), dtype=np.int64)

        self.det_map.update(
            [dict(boxes=torch.tensor(pred_boxes), scores=torch.tensor(pred_scores), labels=torch.tensor(pred_labels))],
            [dict(boxes=torch.tensor(sample.boxes_xyxy), labels=torch.tensor(target_labels))],
        )

        if sample.masks:
            height, width = sample.masks[0].shape
            pred_masks = (
                np.stack([pred.mask.astype(np.uint8) for pred in predictions if pred.mask is not None], axis=0)
                if any(pred.mask is not None for pred in predictions)
                else np.zeros((0, height, width), dtype=np.uint8)
            )
            target_masks = np.stack([mask.astype(np.uint8) for mask in sample.masks], axis=0)
            self.seg_map.update(
                [dict(masks=torch.tensor(pred_masks), scores=torch.tensor(pred_scores[: len(pred_masks)]), labels=torch.tensor(pred_labels[: len(pred_masks)]))],
                [dict(masks=torch.tensor(target_masks), labels=torch.tensor(target_labels[: len(target_masks)]))],
            )
            self.mask_ious.append(best_mask_iou(sample.masks, [pred.mask for pred in predictions if pred.mask is not None]))

    def compute(self) -> dict[str, float]:
        det = self.det_map.compute()
        seg = self.seg_map.compute()
        return {
            "bbox_map": float(det["map"]),
            "bbox_map_50": float(det["map_50"]),
            "segm_map": float(seg["map"]),
            "segm_map_50": float(seg["map_50"]),
            "mean_mask_iou": float(np.mean(self.mask_ious)) if self.mask_ious else 0.0,
        }


def best_mask_iou(target_masks: list[np.ndarray], pred_masks: list[np.ndarray | None]) -> float:
    valid_preds = [mask for mask in pred_masks if mask is not None]
    if not target_masks or not valid_preds:
        return 0.0
    best = 0.0
    for target in target_masks:
        for pred in valid_preds:
            inter = np.logical_and(target, pred).sum()
            union = np.logical_or(target, pred).sum()
            if union > 0:
                best = max(best, inter / union)
    return float(best)


def save_failure_case(image: np.ndarray, predictions: list[Detection], output_path: str | Path) -> None:
    cv2.imwrite(str(output_path), overlay_detections(image, predictions))
