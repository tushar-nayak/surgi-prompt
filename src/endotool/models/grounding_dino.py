from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from endotool.types import Detection
from endotool.utils.labels import normalize_label
from endotool.utils.tracking import suppress_duplicate_detections


class GroundingDinoDetector:
    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        device: str = "cuda",
        label_map: dict[str, str] | None = None,
        label_to_id: dict[str, int] | None = None,
        hf_model_id: str = "IDEA-Research/grounding-dino-tiny",
    ) -> None:
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.label_map = label_map
        self.label_to_id = label_to_id or {}
        self.hf_model_id = hf_model_id
        self.backend = "official"
        try:
            from groundingdino.datasets import transforms as T
            from groundingdino.util.inference import load_model, predict

            self._transforms = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self._predict = predict
            self.model = load_model(str(config_path), str(checkpoint_path), device=device)
        except Exception:
            self._init_hf_backend()

    def detect(self, image_bgr: np.ndarray, prompt: str) -> list[Detection]:
        if self.backend == "hf":
            return self._detect_hf(image_bgr, prompt)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        transformed, _ = self._transforms(pil_image, None)
        try:
            boxes_cxcywh, scores, phrases = self._predict(
                model=self.model,
                image=transformed,
                caption=prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
        except Exception:
            self._init_hf_backend()
            return self._detect_hf(image_bgr, prompt)

        height, width = image_bgr.shape[:2]
        detections: list[Detection] = []
        for box_cxcywh, score, phrase in zip(boxes_cxcywh, scores, phrases):
            normalized = normalize_label(phrase or "surgical tool", self.label_map)
            cx, cy, bw, bh = box_cxcywh.tolist()
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            detections.append(
                Detection(
                    label=normalized,
                    score=float(score.item() if hasattr(score, "item") else score),
                    box_xyxy=np.clip(box, [0, 0, 0, 0], [width - 1, height - 1, width - 1, height - 1]),
                    label_id=self.label_to_id.get(normalized),
                )
            )
        return suppress_duplicate_detections(detections)

    def _init_hf_backend(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.backend = "hf"
        self.processor = AutoProcessor.from_pretrained(self.hf_model_id)
        self.hf_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.hf_model_id).to(self.device)
        self.hf_model.eval()

    def _detect_hf(self, image_bgr: np.ndarray, prompt: str) -> list[Detection]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.hf_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        detections: list[Detection] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            normalized = normalize_label(str(label), self.label_map)
            detections.append(
                Detection(
                    label=normalized,
                    score=float(score.item()),
                    box_xyxy=box.detach().cpu().numpy().astype(np.float32),
                    label_id=self.label_to_id.get(normalized),
                )
            )
        return suppress_duplicate_detections(detections)
