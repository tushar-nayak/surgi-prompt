from __future__ import annotations

import numpy as np

from endotool.types import Detection


GENERIC_LABELS = {"surgical tool", "tool", "instrument"}


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.tolist()
    bx1, by1, bx2, by2 = box_b.tolist()
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box.tolist()
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def box_containment(inner: np.ndarray, outer: np.ndarray) -> float:
    ix1, iy1, ix2, iy2 = inner.tolist()
    ox1, oy1, ox2, oy2 = outer.tolist()
    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_inner = box_area(inner)
    if area_inner <= 0.0:
        return 0.0
    return float(inter / area_inner)


def suppress_duplicate_detections(
    detections: list[Detection],
    iou_threshold: float = 0.6,
    containment_threshold: float = 0.9,
) -> list[Detection]:
    ranked = sorted(detections, key=lambda det: det.score, reverse=True)
    kept: list[Detection] = []

    for det in ranked:
        should_keep = True
        for prev in kept:
            iou = box_iou(det.box_xyxy, prev.box_xyxy)
            contained = box_containment(det.box_xyxy, prev.box_xyxy)
            same_label = det.label == prev.label
            generic_vs_specific = (det.label in GENERIC_LABELS) != (prev.label in GENERIC_LABELS)

            if same_label and iou >= iou_threshold:
                should_keep = False
                break

            if generic_vs_specific and contained >= containment_threshold:
                if det.label in GENERIC_LABELS:
                    should_keep = False
                    break

            if iou >= 0.85:
                should_keep = False
                break

        if should_keep:
            kept.append(det)
    return kept


def match_detections_to_tracks(
    refreshed: list[Detection],
    tracks: dict[int, Detection],
    iou_threshold: float = 0.35,
) -> tuple[list[tuple[int, Detection]], list[Detection]]:
    assigned_tracks: set[int] = set()
    matches: list[tuple[int, Detection]] = []
    unmatched: list[Detection] = []

    for det in sorted(refreshed, key=lambda item: item.score, reverse=True):
        best_track_id = None
        best_iou = 0.0
        for track_id, track_det in tracks.items():
            if track_id in assigned_tracks:
                continue
            if not labels_are_compatible(det.label, track_det.label):
                continue
            iou = box_iou(det.box_xyxy, track_det.box_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id
        if best_track_id is not None and best_iou >= iou_threshold:
            assigned_tracks.add(best_track_id)
            matches.append((best_track_id, det))
        else:
            unmatched.append(det)
    return matches, unmatched


def labels_are_compatible(label_a: str, label_b: str) -> bool:
    if label_a == label_b:
        return True
    return label_a in GENERIC_LABELS or label_b in GENERIC_LABELS
