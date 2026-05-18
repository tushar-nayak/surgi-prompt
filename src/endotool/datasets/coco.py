from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from endotool.types import DatasetSample
from endotool.utils.labels import normalize_label


class CocoEndoscopyDataset:
    def __init__(
        self,
        images_dir: str | Path,
        annotations_path: str | Path,
        label_map: dict[str, str] | None = None,
        category_id_to_label_id: dict[str, int] | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        payload = json.loads(Path(annotations_path).read_text())
        categories = {cat["id"]: cat["name"] for cat in payload["categories"]}
        images = {img["id"]: img for img in payload["images"]}
        grouped: dict[int, list[dict]] = defaultdict(list)
        for ann in payload["annotations"]:
            grouped[ann["image_id"]].append(ann)

        self.samples: list[DatasetSample] = []
        for image_id, image_meta in images.items():
            anns = grouped.get(image_id, [])
            boxes = []
            labels = []
            label_ids = []
            masks = []
            for ann in anns:
                normalized = normalize_label(categories[ann["category_id"]], label_map)
                if category_id_to_label_id:
                    label_id = category_id_to_label_id.get(normalized)
                    if label_id is None:
                        continue
                    label_ids.append(label_id)
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(normalized)
                segmentation = ann.get("segmentation")
                if segmentation:
                    masks.append(_decode_segmentation(segmentation, image_meta["height"], image_meta["width"]))
            if not boxes:
                continue
            self.samples.append(
                DatasetSample(
                    image_id=image_id,
                    image_path=self.images_dir / image_meta["file_name"],
                    boxes_xyxy=np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
                    labels=labels,
                    label_ids=label_ids,
                    masks=masks,
                    video_id=str(image_meta["file_name"]).split("_")[0] if "_" in image_meta["file_name"] else None,
                    frame_idx=_parse_frame_idx(image_meta["file_name"]),
                )
            )

    def __iter__(self):
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


def _parse_frame_idx(file_name: str) -> int | None:
    stem = Path(file_name).stem
    if "_" not in stem:
        return None
    _, _, *rest = stem.partition("_")
    try:
        return int("".join(rest))
    except ValueError:
        return None


def _decode_segmentation(segmentation: list | dict, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        merged = mask_utils.merge(rles)
        return mask_utils.decode(merged).astype(bool)
    if isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            segmentation = mask_utils.frPyObjects(segmentation, height, width)
        return mask_utils.decode(segmentation).astype(bool)
    raise TypeError("Unsupported segmentation format")
