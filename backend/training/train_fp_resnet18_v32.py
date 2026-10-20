"""Train the v3.2 2.5D ResNet18 false-positive reduction model.

Key changes from v3.1:
  - optional ImageNet initialization with matching input normalization
  - synchronized 2.5D augmentation
  - class-balanced and hard-negative-aware sampling
  - dropout, label smoothing, differential learning rates and early stopping
  - evaluation never updates model weights
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fp_resnet18 import FPResNet18, count_parameters


class CandidatePatchDataset(Dataset):
    """Load validated [3, 96, 96] PNG-derived patch tensors."""

    def __init__(self, csv_path, split, augment=False):
        self.csv_dir = os.path.dirname(os.path.abspath(csv_path))
        self.augment = augment
        self.samples = []

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] != split:
                    continue
                label = int(row["label"])
                if label not in (0, 1):
                    raise ValueError(f"Invalid label={label} for sample {row.get('sample_id')}")

                patch_path = self._resolve_patch_path(row["patch_path"], split)
                sample_weight = float(row.get("sample_weight") or 1.0)
                self.samples.append(
                    {
                        "patch_path": patch_path,
                        "label": label,
                        "sample_id": row["sample_id"],
                        "sample_weight": max(sample_weight, 0.01),
                    }
                )

        if not self.samples:
            raise RuntimeError(f"No {split} samples found in {csv_path}")

        self.labels = [sample["label"] for sample in self.samples]
        positives = sum(self.labels)
        print(
            f"  {split}: {len(self.samples)} samples "
            f"(pos={positives}, neg={len(self.samples) - positives}, augment={augment})"
        )

    def _resolve_patch_path(self, stored_path, split):
        if os.path.isfile(stored_path):
            return stored_path
        fallback = os.path.join(self.csv_dir, "patches", split, os.path.basename(stored_path))
        if os.path.isfile(fallback):
            return fallback
        raise FileNotFoundError(f"Patch missing: {stored_path}; fallback: {fallback}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tensor = torch.load(sample["patch_path"], map_location="cpu", weights_only=False)
        tensor = tensor.float()
        if list(tensor.shape) != [3, 96, 96]:
            raise RuntimeError(
                f"Bad patch shape {list(tensor.shape)} for {sample['patch_path']}; "
                "expected [3, 96, 96]"
            )
        value_min = float(tensor.min())
        value_max = float(tensor.max())
        if value_min < -1e-4 or value_max > 1.0001:
            raise RuntimeError(
                f"Patch outside [0,1]: [{value_min:.4f}, {value_max:.4f}] "
                f"for {sample['patch_path']}"
            )
        if self.augment:
            tensor = augment_25d_patch(tensor)
        return tensor, sample["label"]


def augment_25d_patch(tensor):
    """Apply identical spatial transforms to z-1/z/z+1 channels."""
    if random.random() < 0.5:
        tensor = torch.flip(tensor, dims=(-1,))

    angle = random.uniform(-8.0, 8.0)
    translate = [random.randint(-4, 4), random.randint(-4, 4)]
    scale = random.uniform(0.95, 1.05)
    tensor = TF.affine(
        tensor,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )

    contrast = random.uniform(0.90, 1.10)
    brightness = random.uniform(-0.04, 0.04)
    channel_mean = tensor.mean(dim=(-2, -1), keepdim=True)
    tensor = (tensor - channel_mean) * contrast + channel_mean + brightness

    if random.random() < 0.3:
        tensor = tensor + torch.randn_like(tensor) * random.uniform(0.002, 0.012)
    return tensor.clamp_(0.0, 1.0)


def build_sampler(dataset, balance_classes, hard_negative_sampling):
    if not balance_classes and not hard_negative_sampling:
        return None

    class_counts = np.bincount(dataset.labels, minlength=2)
    class_weights = [
        len(dataset) / (2.0 * max(int(class_counts[label]), 1))
        if balance_classes
        else 1.0
        for label in range(2)
    ]
    negative_hardness = [
        sample["sample_weight"]
        for sample in dataset.samples
        if sample["label"] == 0
    ]
    mean_negative_hardness = (
        sum(negative_hardness) / len(negative_hardness)
        if negative_hardness
        else 1.0
    )
    weights = []
    for sample in dataset.samples:
        weight = class_weights[sample["label"]]
        if hard_negative_sampling and sample["label"] == 0:
            # Redistribute probability inside the negative class without
            # changing the intended 1:1 positive/negative class balance.
            weight *= sample["sample_weight"] / mean_negative_hardness
        weights.append(weight)

    print(
        f"  sampler: class_weights={class_weights}, "
        f"hard_negative_sampling={hard_negative_sampling}, "
        f"mean_negative_hardness={mean_negative_hardness:.4f}"
    )
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def confusion_metrics(probabilities, labels, threshold=0.5):
    predictions = (probabilities >= threshold).long()
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced_accuracy = (recall + specificity) / 2
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "positive_count": tp + fn,
        "negative_count": tn + fp,
    }


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    probabilities = []
    labels_all = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            amp_enabled = scaler is not None
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.detach()) * images.size(0)
            total += images.size(0)
            probabilities.append(torch.softmax(logits.detach(), dim=1)[:, 1].cpu())
            labels_all.append(labels.detach().cpu())

    metrics = confusion_metrics(torch.cat(probabilities), torch.cat(labels_all))
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train RadScout v3.2 FPResNet18")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_dir", default="outputs/fp_resnet18_v32")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Classifier learning rate")
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced_sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard_negative_sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--negative_class_weight", type=float, default=1.0)
    parser.add_argument("--freeze_epochs", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        help=(
            "Optional existing FPResNet18 checkpoint used to initialize the "
            "model before hard-negative fine-tuning."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested CUDA device, but torch.cuda.is_available() is False. "
            "Check D:/Python/python.exe and its CUDA-enabled PyTorch installation."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    csv_path = os.path.abspath(args.csv)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"torch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = CandidatePatchDataset(csv_path, "train", augment=args.augment)
    train_eval_dataset = CandidatePatchDataset(csv_path, "train", augment=False)
    val_dataset = CandidatePatchDataset(csv_path, "val", augment=False)
    sampler = build_sampler(
        train_dataset,
        balance_classes=args.balanced_sampler,
        hard_negative_sampling=args.hard_negative_sampling,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=sampler is None,
        **loader_options,
    )
    train_eval_loader = DataLoader(train_eval_dataset, shuffle=False, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = FPResNet18(
        pretrained=args.pretrained,
        dropout=args.dropout,
        normalize_input=args.pretrained,
    ).to(device)
    if args.init_checkpoint:
        init_checkpoint_path = os.path.abspath(args.init_checkpoint)
        if not os.path.isfile(init_checkpoint_path):
            raise FileNotFoundError(
                f"Initial checkpoint not found: {init_checkpoint_path}"
            )
        init_checkpoint = torch.load(
            init_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(init_checkpoint["model_state_dict"])
        print(
            "Initialized from checkpoint: "
            f"{init_checkpoint_path} (epoch={init_checkpoint.get('epoch', 'unknown')})"
        )
    if args.freeze_epochs > 0:
        model.set_backbone_trainable(False)
    print(
        f"Model: FPResNet18-v3.2 | pretrained={args.pretrained} | "
        f"dropout={args.dropout} | parameters={count_parameters(model):,}"
    )

    backbone_parameters = []
    classifier_parameters = []
    for name, parameter in model.backbone.named_parameters():
        if name.startswith("fc."):
            classifier_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    optimizer = optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.lr * args.backbone_lr_ratio},
            {"params": classifier_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
    )
    class_weights = torch.tensor(
        [args.negative_class_weight, 1.0], dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    scaler_or_none = scaler if scaler.is_enabled() else None

    log_path = os.path.join(output_dir, "train_log.csv")
    fieldnames = [
        "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
        "val_precision", "val_recall", "val_specificity", "val_f1",
        "val_balanced_accuracy", "TP", "FP", "TN", "FN", "backbone_lr",
        "head_lr", "epoch_seconds",
    ]
    with open(log_path, "w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    model_config = {
        "pretrained": args.pretrained,
        "dropout": args.dropout,
        "normalize_input": args.pretrained,
        "patch_size": 96,
        "input_channels": ["z-1", "z", "z+1"],
    }
    best_path = os.path.join(output_dir, "best_fp_resnet18_v32.pt")
    last_path = os.path.join(output_dir, "last_fp_resnet18_v32.pt")
    best_f1 = -1.0
    best_specificity = -1.0
    epochs_without_improvement = 0
    final_epoch = 0

    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            model.set_backbone_trainable(True)
            print("Backbone unfrozen.")

        started = time.time()
        train_metrics = run_epoch(
            model, train_loader, device, criterion, optimizer, scaler_or_none
        )
        val_metrics = run_epoch(model, val_loader, device, criterion)
        scheduler.step(val_metrics["loss"])
        elapsed = time.time() - started
        cm = val_metrics["confusion_matrix"]

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"P={val_metrics['precision']:.4f} "
            f"R={val_metrics['recall']:.4f} "
            f"Spec={val_metrics['specificity']:.4f} "
            f"F1={val_metrics['f1']:.4f} | "
            f"TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']} | "
            f"{elapsed:.1f}s"
        )

        row = {
            "epoch": epoch,
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_accuracy": f"{train_metrics['accuracy']:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_accuracy": f"{val_metrics['accuracy']:.6f}",
            "val_precision": f"{val_metrics['precision']:.6f}",
            "val_recall": f"{val_metrics['recall']:.6f}",
            "val_specificity": f"{val_metrics['specificity']:.6f}",
            "val_f1": f"{val_metrics['f1']:.6f}",
            "val_balanced_accuracy": f"{val_metrics['balanced_accuracy']:.6f}",
            "TP": cm["TP"], "FP": cm["FP"], "TN": cm["TN"], "FN": cm["FN"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "epoch_seconds": f"{elapsed:.2f}",
        }
        with open(log_path, "a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "model_config": model_config,
            "training_args": vars(args),
        }
        torch.save(checkpoint, last_path)

        improved = (
            val_metrics["f1"] > best_f1 + 1e-6
            or (
                abs(val_metrics["f1"] - best_f1) <= 1e-6
                and val_metrics["specificity"] > best_specificity
            )
        )
        if improved:
            best_f1 = val_metrics["f1"]
            best_specificity = val_metrics["specificity"]
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
            print(f"  -> Best model saved: F1={best_f1:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Early stopping after {epoch} epochs "
                    f"(patience={args.early_stopping_patience})."
                )
                break

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    train_final = run_epoch(model, train_eval_loader, device, criterion)
    val_final = run_epoch(model, val_loader, device, criterion)
    metrics = {
        "model": "FPResNet18-v3.2",
        "dataset_csv": csv_path,
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "best_epoch": best_checkpoint["epoch"],
        "stopped_epoch": final_epoch,
        "model_config": model_config,
        "train": train_final,
        "val": val_final,
        "note": (
            "Patch-level classification metrics. Final product quality must be "
            "validated with detection-level two-stage evaluation."
        ),
    }
    with open(
        os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(f"Best checkpoint: {best_path}")
    print(
        f"Best epoch={best_checkpoint['epoch']} | "
        f"val_F1={val_final['f1']:.4f} | "
        f"val_specificity={val_final['specificity']:.4f}"
    )


if __name__ == "__main__":
    main()
