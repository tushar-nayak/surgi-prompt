from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class Detection:
    label: str
    score: float
    box_xyxy: np.ndarray
    mask: np.ndarray | None = None
    label_id: int | None = None


@dataclass(slots=True)
class DatasetSample:
    image_id: int | str
    image_path: Path
    boxes_xyxy: np.ndarray
    labels: list[str]
    label_ids: list[int] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    video_id: str | None = None
    frame_idx: int | None = None
