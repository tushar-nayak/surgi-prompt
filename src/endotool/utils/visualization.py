from __future__ import annotations

import cv2
import numpy as np

from endotool.types import Detection


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (51, 153, 255),
        (0, 200, 120),
        (255, 170, 0),
        (255, 99, 71),
        (180, 80, 255),
        (0, 220, 220),
    ]
    return palette[index % len(palette)]


def overlay_detections(image_bgr: np.ndarray, detections: list[Detection], alpha: float = 0.35) -> np.ndarray:
    overlay = image_bgr.copy()
    canvas = image_bgr.copy()

    for idx, det in enumerate(detections):
        color = color_for_index(idx)
        x1, y1, x2, y2 = det.box_xyxy.astype(int).tolist()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{det.label}: {det.score:.2f}"
        cv2.putText(canvas, text, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if det.mask is not None:
            mask = det.mask.astype(bool)
            overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array(color)).astype(np.uint8)

    blended = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0)
    return blended

