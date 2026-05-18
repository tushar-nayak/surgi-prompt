from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from endotool.types import DatasetSample


class KvasirInstrumentDataset:
    def __init__(self, images_dir: str | Path, masks_dir: str | Path) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        image_paths = sorted(self.images_dir.glob("*"))
        self.samples: list[DatasetSample] = []
        for image_path in image_paths:
            if not image_path.is_file():
                continue
            mask_path = self.masks_dir / image_path.name
            if not mask_path.exists():
                alt = self.masks_dir / f"{image_path.stem}.png"
                if alt.exists():
                    mask_path = alt
                else:
                    continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = mask > 0
            box = _mask_to_box(mask)
            boxes = np.array([box], dtype=np.float32) if box is not None else np.zeros((0, 4), dtype=np.float32)
            masks = [mask] if box is not None else []
            labels = ["surgical tool"] if box is not None else []
            self.samples.append(
                DatasetSample(
                    image_id=image_path.stem,
                    image_path=image_path,
                    boxes_xyxy=boxes,
                    labels=labels,
                    label_ids=[1] if box is not None else [],
                    masks=masks,
                )
            )

    def __iter__(self):
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


def _mask_to_box(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
