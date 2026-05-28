from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from endotool.datasets.coco import CocoEndoscopyDataset
from endotool.datasets.kvasir import KvasirInstrumentDataset
from endotool.types import DatasetSample
from endotool.utils.labels import load_label_map


@dataclass(slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float | None
    learning_rate: float


class RealGroundingDataset(Dataset):
    def __init__(self, samples: list[DatasetSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self.samples[index]


class GroundingCollator:
    def __init__(self, processor, prompt_labels: list[str], augment: bool = False) -> None:
        self.processor = processor
        self.prompt_labels = prompt_labels
        self.label_to_prompt_idx = {label: idx for idx, label in enumerate(prompt_labels)}
        self.augment = augment

    def __call__(self, batch: list[DatasetSample]) -> dict[str, object]:
        images: list[Image.Image] = []
        annotations: list[dict[str, object]] = []
        sample_ids: list[str] = []
        for sample in batch:
            image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(sample.image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            boxes_xyxy = sample.boxes_xyxy.copy() if sample.boxes_xyxy is not None else np.empty((0, 4), dtype=np.float32)
            labels = list(sample.labels)
            if self.augment:
                pil_image, boxes_xyxy, valid_mask = _apply_augmentations(pil_image, boxes_xyxy)
                labels = [l for l, v in zip(labels, valid_mask) if v]
            images.append(pil_image)

            ann_objects = []
            for box_xyxy, label in zip(boxes_xyxy, labels):
                prompt_idx = self.label_to_prompt_idx.get(label)
                if prompt_idx is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width <= 1e-3 or height <= 1e-3:
                    continue
                ann_objects.append(
                    {
                        "bbox": [x1, y1, width, height],
                        "category_id": prompt_idx,
                        "area": width * height,
                        "iscrowd": 0,
                    }
                )

            annotations.append({"image_id": int(index_or_hash(sample.image_id)), "annotations": ann_objects})
            sample_ids.append(str(sample.image_id))

        encoded = self.processor(
            images=images,
            text=[self.prompt_labels] * len(images),
            annotations=annotations,
            return_tensors="pt",
            padding=True,
        )
        encoded["sample_ids"] = sample_ids
        return encoded


def _apply_augmentations(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Apply training-time augmentations to image and boxes.

    Augmentations: horizontal flip, small affine (rotation/translate/scale),
    color jitter, and occasional Gaussian blur.  Spatial transforms update
    the bounding-box coordinates accordingly.

    Returns (image, boxes, valid_mask) where valid_mask is a bool array
    indicating which input boxes survived (not clipped to degenerate).
    """
    width, height = image.size
    boxes = boxes_xyxy.copy().astype(np.float64)

    # --- horizontal flip (p=0.5) ---
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if len(boxes):
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = width - x2
            boxes[:, 2] = width - x1

    # --- random affine (rotation, translate, scale) ---
    angle = random.uniform(-15, 15)
    tx = random.uniform(-0.05, 0.05) * width
    ty = random.uniform(-0.05, 0.05) * height
    scale = random.uniform(0.85, 1.15)

    cx, cy = width / 2.0, height / 2.0
    cos_a = math.cos(math.radians(angle)) * scale
    sin_a = math.sin(math.radians(angle)) * scale
    # PIL affine transform coefficients (inverse mapping)
    a = cos_a
    b = sin_a
    c = cx - cos_a * cx - sin_a * cy + tx
    d = -sin_a
    e = cos_a
    f = cy + sin_a * cx - cos_a * cy + ty
    image = image.transform(
        image.size,
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BILINEAR,
    )
    if len(boxes):
        # forward-transform box corners
        fwd_a = cos_a
        fwd_b = -sin_a
        fwd_d = sin_a
        fwd_e = cos_a
        fwd_c = cx - fwd_a * cx - fwd_b * cy - tx
        fwd_f = cy - fwd_d * cx - fwd_e * cy - ty
        new_boxes = []
        for box in boxes:
            corners = np.array([
                [box[0], box[1]],
                [box[2], box[1]],
                [box[2], box[3]],
                [box[0], box[3]],
            ])
            transformed = np.column_stack([
                fwd_a * corners[:, 0] + fwd_b * corners[:, 1] + fwd_c,
                fwd_d * corners[:, 0] + fwd_e * corners[:, 1] + fwd_f,
            ])
            new_boxes.append([
                transformed[:, 0].min(),
                transformed[:, 1].min(),
                transformed[:, 0].max(),
                transformed[:, 1].max(),
            ])
        boxes = np.array(new_boxes, dtype=np.float64)

    # --- color jitter ---
    if random.random() < 0.8:
        brightness = random.uniform(0.7, 1.3)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        contrast = random.uniform(0.7, 1.3)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        saturation = random.uniform(0.8, 1.2)
        image = ImageEnhance.Color(image).enhance(saturation)

    # --- Gaussian blur (p=0.15) ---
    if random.random() < 0.15:
        radius = random.uniform(0.5, 1.5)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    # --- clip boxes to image bounds and filter degenerate ---
    valid = np.ones(len(boxes), dtype=bool)
    if len(boxes):
        boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
        valid = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
        boxes = boxes[valid]

    return image, boxes.astype(np.float32), valid


def index_or_hash(value: int | str) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except ValueError:
        return abs(hash(value)) % (2**31)


def load_samples(
    dataset_type: str,
    images_dir: str,
    annotations_path: str | None,
    masks_dir: str | None,
    label_map_path: str | None,
    max_images: int | None,
) -> list[DatasetSample]:
    label_map = load_label_map(label_map_path) if label_map_path else None
    if dataset_type == "coco":
        if not annotations_path:
            raise ValueError("--annotations is required for dataset-type=coco")
        dataset = CocoEndoscopyDataset(images_dir, annotations_path, label_map=label_map)
    elif dataset_type == "kvasir":
        if not masks_dir:
            raise ValueError("--masks-dir is required for dataset-type=kvasir")
        dataset = KvasirInstrumentDataset(images_dir, masks_dir)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    samples = list(dataset)
    if max_images is not None:
        samples = samples[:max_images]
    return samples


def derive_prompt_labels(samples: list[DatasetSample]) -> list[str]:
    labels = sorted({label for sample in samples for label in sample.labels})
    if not labels:
        raise ValueError("No training labels found in dataset.")
    return labels


def move_batch_to_device(batch: dict[str, object], device: str) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if key == "sample_ids":
            moved[key] = value
        elif key == "labels":
            moved[key] = [{k: v.to(device) for k, v in item.items()} for item in value]
        elif hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_scheduler(optimizer: AdamW, num_training_steps: int, num_warmup_steps: int) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader: DataLoader, device: str, use_amp: bool) -> float:
    model.eval()
    running_loss = 0.0
    num_batches = 0
    for batch in tqdm(loader, desc="val", leave=False):
        batch = move_batch_to_device(batch, device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    pixel_mask=batch.get("pixel_mask"),
                    input_ids=batch["input_ids"],
                    token_type_ids=batch.get("token_type_ids"),
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
        running_loss += float(outputs.loss.item())
        num_batches += 1
    return running_loss / max(1, num_batches)


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_samples(
        dataset_type=args.dataset_type,
        images_dir=args.images_dir,
        annotations_path=args.annotations,
        masks_dir=args.masks_dir,
        label_map_path=args.label_map_path,
        max_images=args.max_train_images,
    )
    if not train_samples:
        raise ValueError("No training samples found.")

    val_samples: list[DatasetSample] = []
    if args.val_images_dir:
        val_samples = load_samples(
            dataset_type=args.dataset_type,
            images_dir=args.val_images_dir,
            annotations_path=args.val_annotations,
            masks_dir=args.val_masks_dir,
            label_map_path=args.label_map_path,
            max_images=args.max_val_images,
        )

    prompt_labels = (
        [label.strip().lower() for label in args.prompt_labels.split(",") if label.strip()]
        if args.prompt_labels
        else derive_prompt_labels(train_samples)
    )
    if not prompt_labels:
        raise ValueError("Prompt label vocabulary is empty.")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(args.device)

    if args.freeze_text_encoder:
        for param in model.model.text_backbone.parameters():
            param.requires_grad = False

    if args.freeze_vision_backbone:
        for param in model.model.backbone.parameters():
            param.requires_grad = False

    train_loader = DataLoader(
        RealGroundingDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=GroundingCollator(processor, prompt_labels, augment=not args.no_augment),
    )
    val_loader = (
        DataLoader(
            RealGroundingDataset(val_samples),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            collate_fn=GroundingCollator(processor, prompt_labels),
        )
        if val_samples
        else None
    )
    # Build parameter groups with layer-wise learning rate
    backbone_param_ids = set()
    if not args.freeze_vision_backbone:
        backbone_param_ids = {id(p) for p in model.model.backbone.parameters() if p.requires_grad}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) in backbone_param_ids]
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_param_ids]
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.learning_rate})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.learning_rate * args.backbone_lr_factor})
    trainable_params = head_params + backbone_params
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / max(1, args.grad_accum_steps)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda") and not args.disable_amp)

    (output_dir / "prompt_labels.json").write_text(json.dumps(prompt_labels, indent=2))
    (output_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2))

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    global_step = 0
    use_amp = args.device.startswith("cuda") and not args.disable_amp

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        num_batches = 0
        progress = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, args.device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    pixel_mask=batch.get("pixel_mask"),
                    input_ids=batch["input_ids"],
                    token_type_ids=batch.get("token_type_ids"),
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                loss = outputs.loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running_loss += float(outputs.loss.item())
            num_batches += 1
            progress.set_postfix(loss=f"{running_loss / max(1, num_batches):.4f}")

        train_loss = running_loss / max(1, num_batches)
        val_loss = evaluate(model, val_loader, args.device, use_amp) if val_loader is not None else None
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
        )
        history.append(asdict(metrics))
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))

        epoch_dir = output_dir / f"checkpoint-epoch{epoch:02d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_dir = output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
        elif val_loss is not None:
            epochs_without_improvement += 1
        elif val_loss is None and epoch == args.epochs:
            best_dir = output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

        # early stopping check
        if args.early_stopping_patience > 0 and val_loss is not None:
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping: no improvement for {args.early_stopping_patience} epochs.")
                break

    final_dir = output_dir / "last"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    summary = {
        "model_id": args.model_id,
        "prompt_labels": prompt_labels,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "epochs": args.epochs,
        "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "best_checkpoint": str((output_dir / "best").resolve()),
        "last_checkpoint": str(final_dir.resolve()),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Hugging Face Grounding DINO on real surgical data.")
    parser.add_argument("--dataset-type", choices=["coco", "kvasir"], required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations")
    parser.add_argument("--masks-dir")
    parser.add_argument("--val-images-dir")
    parser.add_argument("--val-annotations")
    parser.add_argument("--val-masks-dir")
    parser.add_argument("--label-map-path", default="configs/tool_label_map.yaml")
    parser.add_argument("--prompt-labels", help="Comma-separated label vocabulary. Default: derive from training data.")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-val-images", type=int)
    parser.add_argument("--freeze-text-encoder", action="store_true")
    parser.add_argument("--freeze-vision-backbone", action="store_true")
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1,
                        help="LR multiplier for vision backbone params (default 0.1x head LR).")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-augment", action="store_true", help="Disable training data augmentation.")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Stop if val loss does not improve for N epochs. 0 = disabled.")
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
