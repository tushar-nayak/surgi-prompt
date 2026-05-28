from __future__ import annotations

import argparse
import time
from pathlib import Path

from endotool.datasets.coco import CocoEndoscopyDataset
from endotool.datasets.kvasir import KvasirInstrumentDataset
from endotool.eval.metrics import Evaluator, best_mask_iou, save_failure_case
from endotool.pipeline import OpenVocabSurgicalPipeline, PipelineConfig
from endotool.utils.labels import build_label_index, load_label_map
from endotool.utils.io import ensure_dir, read_image, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", choices=["coco", "kvasir"], required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations")
    parser.add_argument("--masks-dir")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--failure-iou-threshold", type=float, default=0.30)
    parser.add_argument("--label-map", default="configs/tool_label_map.yaml")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-hf-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--grounding-force-hf-backend", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    failures_dir = ensure_dir(output_dir / "failure_cases")
    alias_to_canonical = load_label_map(args.label_map) if args.label_map else None
    label_to_id = build_label_index(list(set(alias_to_canonical.values()))) if alias_to_canonical else {}

    if args.dataset_type == "coco":
        if not args.annotations:
            raise ValueError("--annotations is required for --dataset-type coco")
        dataset = CocoEndoscopyDataset(args.images_dir, args.annotations, alias_to_canonical, label_to_id)
    else:
        if not args.masks_dir:
            raise ValueError("--masks-dir is required for --dataset-type kvasir")
        dataset = KvasirInstrumentDataset(args.images_dir, args.masks_dir)

    pipeline = OpenVocabSurgicalPipeline(
        PipelineConfig(
            device=args.device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            label_map_path=args.label_map,
            grounding_hf_model_id=args.grounding_hf_model_id,
            grounding_force_hf_backend=args.grounding_force_hf_backend,
        )
    )
    evaluator = Evaluator()
    started = time.perf_counter()
    num_seen = 0

    for sample in dataset:
        if args.max_images > 0 and num_seen >= args.max_images:
            break
        image = read_image(sample.image_path)
        preds = pipeline.detector.detect(image, args.prompt)
        preds = pipeline.segmenter.refine_image_masks(image, preds)
        evaluator.update(sample, preds)
        iou = best_mask_iou(sample.masks, [pred.mask for pred in preds]) if sample.masks else 0.0
        if sample.masks and iou < args.failure_iou_threshold:
            save_failure_case(image, preds, failures_dir / f"{sample.image_id}.png")
        num_seen += 1

    elapsed = time.perf_counter() - started
    metrics = evaluator.compute()
    metrics["fps"] = num_seen / max(elapsed, 1e-6)
    metrics["num_images"] = num_seen
    metrics["label_to_id"] = label_to_id
    write_json(output_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
