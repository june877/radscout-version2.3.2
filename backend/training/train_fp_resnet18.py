"""
train_fp_resnet18.py — 训练 2.5D ResNet18 假阳性削减模型

用法:
    cd D:/python_gjw/radscout2/version2.3/backend
    D:/Python/python.exe -u training/train_fp_resnet18.py \
        --csv ../datasets/luna_candidate_patches/candidate_dataset.csv \
        --output_dir outputs/fp_resnet18 \
        --epochs 20 --batch_size 32 --lr 1e-4 --device cuda --amp

输出:
    outputs/fp_resnet18/
        best_fp_resnet18.pt      验证 F1 最高的模型
        last_fp_resnet18.pt      最后一轮模型
        train_log.csv            每轮详细指标
        metrics.json             最终评估
"""

import os, sys, json, time, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fp_resnet18 import FPResNet18, count_parameters


# ============================================================
# Dataset
# ============================================================

class CandidatePatchDataset(Dataset):
    """从 candidate_dataset.csv 加载 patch 和 label"""

    def __init__(self, csv_path, split):
        self.samples = []
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                patch_path = row["patch_path"]
                if not os.path.isfile(patch_path):
                    raise FileNotFoundError(f"Patch missing: {patch_path}")
                self.samples.append({
                    "patch_path": patch_path,
                    "label": int(row["label"]),
                    "sample_id": row["sample_id"],
                })

        labels = [s["label"] for s in self.samples]
        print(f"  {split}: {len(self.samples)} samples "
              f"(pos={sum(labels)}, neg={len(labels)-sum(labels)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tensor = torch.load(s["patch_path"], weights_only=False)
        if list(tensor.shape) != [3, 96, 96]:
            raise RuntimeError(
                f"Bad patch shape {list(tensor.shape)} for {s['patch_path']}, expected [3,96,96]")
        t_min, t_max = tensor.min().item(), tensor.max().item()
        if t_min < -0.1 or t_max > 1.1:
            raise RuntimeError(
                f"Patch value out of [0,1]: [{t_min:.3f}, {t_max:.3f}] for {s['patch_path']}")
        return tensor, s["label"]


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def compute_metrics(model, loader, device):
    """在验证集上计算详细指标"""
    model.eval()
    all_probs, all_labels = [], []
    total_loss, total = 0.0, 0
    loss_fn = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]  # class=1 prob
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds = (probs > 0.5).long()

    # Confusion matrix
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    acc = (tp + tn) / (tp + fp + tn + fn) if (tp+fp+tn+fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "loss": total_loss / total,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "pos_count": tp + fn,
        "neg_count": tn + fp,
    }


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    loss_fn = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train FPResNet18")
    p.add_argument("--csv", type=str, required=True, help="candidate_dataset.csv path")
    p.add_argument("--output_dir", type=str, default="outputs/fp_resnet18")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="Enable AMP mixed precision")
    p.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = os.path.abspath(args.csv)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── Device ──
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.amp and device.type != "cuda":
        print("AMP disabled (CPU)")
        args.amp = False

    # ── Data ──
    print("\nLoading data...")
    train_ds = CandidatePatchDataset(csv_path, "train")
    val_ds = CandidatePatchDataset(csv_path, "val")
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)

    # ── Model ──
    print(f"\nModel: FPResNet18 (pretrained={args.pretrained})")
    model = FPResNet18(pretrained=args.pretrained).to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    # ── Optimizer ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    # ── Training ──
    best_val_f1 = 0.0
    csv_log = os.path.join(out_dir, "train_log.csv")
    with open(csv_log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc",
                     "val_loss", "val_acc", "val_precision", "val_recall",
                     "val_specificity", "val_f1", "TP", "FP", "TN", "FN", "lr"])

    print(f"\n{'='*60}")
    print(f"  Training {args.epochs} epochs...")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_metrics = compute_metrics(model, val_loader, device)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        cm = val_metrics["confusion_matrix"]
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss={train_metrics['loss']:.4f} | "
              f"train_acc={train_metrics['accuracy']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | "
              f"P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} "
              f"F1={val_metrics['f1']:.3f} | "
              f"TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']} | "
              f"{elapsed:.1f}s")

        with open(csv_log, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch,
                         f"{train_metrics['loss']:.6f}", f"{train_metrics['accuracy']:.6f}",
                         f"{val_metrics['loss']:.6f}", f"{val_metrics['accuracy']:.6f}",
                         f"{val_metrics['precision']:.4f}", f"{val_metrics['recall']:.4f}",
                         f"{val_metrics['specificity']:.4f}", f"{val_metrics['f1']:.4f}",
                         cm['TP'], cm['FP'], cm['TN'], cm['FN'],
                         f"{lr:.6f}"])

        # Save best by val_f1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, os.path.join(out_dir, "best_fp_resnet18.pt"))
            print(f"  -> Best model (F1={best_val_f1:.4f})")

    # Save last
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
    }, os.path.join(out_dir, "last_fp_resnet18.pt"))

    # ── Final metrics ──
    best_ckpt = torch.load(os.path.join(out_dir, "best_fp_resnet18.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_val = compute_metrics(model, val_loader, device)
    final_train = train_one_epoch(model, train_loader, optimizer, device, scaler)

    metrics = {
        "model": "FPResNet18",
        "training_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_epoch": best_ckpt["epoch"],
        "train": {"loss": round(final_train["loss"], 4), "accuracy": round(final_train["accuracy"], 4)},
        "val": final_val,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"  Best F1: {best_val_f1:.4f} (epoch {best_ckpt['epoch']})")
    cm = final_val["confusion_matrix"]
    print(f"  TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']}")
    print(f"  P={final_val['precision']:.4f} R={final_val['recall']:.4f} F1={final_val['f1']:.4f}")
    print(f"  Output: {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
