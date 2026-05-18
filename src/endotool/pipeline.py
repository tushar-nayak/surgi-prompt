from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from endotool.models.grounding_dino import GroundingDinoDetector
from endotool.models.sam2 import Sam2Segmenter
from endotool.types import Detection
from endotool.utils.labels import build_label_index, load_label_map
from endotool.utils.visualization import overlay_detections
from endotool.utils.paths import grounding_dino_default_config, sam2_default_config


@dataclass(slots=True)
class PipelineConfig:
    grounding_config: str = grounding_dino_default_config()
    grounding_checkpoint: str = "checkpoints/groundingdino_swint_ogc.pth"
    sam2_config: str = sam2_default_config()
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    device: str = "cuda"
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    label_map_path: str | None = "configs/tool_label_map.yaml"
    grounding_hf_model_id: str = "IDEA-Research/grounding-dino-tiny"


class OpenVocabSurgicalPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.alias_to_canonical = load_label_map(config.label_map_path) if config.label_map_path else None
        self.label_to_id = build_label_index(list(set(self.alias_to_canonical.values()))) if self.alias_to_canonical else {}
        self.detector = GroundingDinoDetector(
            config_path=config.grounding_config,
            checkpoint_path=config.grounding_checkpoint,
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
            device=config.device,
            label_map=self.alias_to_canonical,
            label_to_id=self.label_to_id,
            hf_model_id=config.grounding_hf_model_id,
        )
        self.segmenter = Sam2Segmenter(
            config_path=config.sam2_config,
            checkpoint_path=config.sam2_checkpoint,
            device=config.device,
        )

    def run_image(self, image_path: str | Path, prompt: str) -> tuple[list[Detection], float]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        started = time.perf_counter()
        detections = self.detector.detect(image, prompt)
        detections = self.segmenter.refine_image_masks(image, detections)
        elapsed = time.perf_counter() - started
        return detections, 1.0 / max(elapsed, 1e-6)

    def annotate_image(self, image_path: str | Path, detections: list[Detection], output_path: str | Path) -> None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        annotated = overlay_detections(image, detections)
        cv2.imwrite(str(output_path), annotated)
