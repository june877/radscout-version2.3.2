"""
训练脚本 — 2D CT 肺结节检测模型

用法:
    cd D:/python_gjw/radscout2/version2.1/backend
    python training/train.py --data_dir D:/python_gjw/radscout2/data/luna16 --epochs 20 --batch_size 8

GPU 训练:
    python training/train.py --data_dir ... --epochs 20 --batch_size 8 --device cuda
    python training/train.py --data_dir ... --epochs 20 --batch_size 8 --device cuda --amp

仅评估已有模型:
    python training/train.py --data_dir ... --evaluate --device cuda

输出:
    weights/best_model.pt     验证集准确率最高的模型权重
    outputs/train_log.csv     每轮训练/验证指标
    outputs/metrics.json      最终评估结果（含混淆矩阵、precision、recall、F1 等）
    outputs/eval_samples/     FP / FN / TP / TN 样本图像
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.simple_detector import SimpleCTDetector, count_parameters
from training.luna_dataset import create_dataloaders


# ============================================================
# 命令行参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="RadScout v2.1 — Train SimpleCTDetector"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--box_lambda", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--evaluate", action="store_true",
                        help="仅评估已有模型，不训练")
    parser.add_argument("--eval_samples", type=int, default=5,
                        help="每类保存的错误样本数（默认5）")
    return parser.parse_args()


# ============================================================
# Loss & 训练辅助
# ============================================================
def compute_loss(cls_logits, bbox_pred, targets, box_lambda=0.5):
    cls_labels = targets["has_nodule"].unsqueeze(1)
    bbox_labels = targets["bbox"]
    cls_loss_fn = nn.BCEWithLogitsLoss()
    cls_loss = cls_loss_fn(cls_logits, cls_labels)
    pos_mask = cls_labels.squeeze() > 0.5
    n_pos = pos_mask.sum().item()
    if n_pos > 0:
        box_loss_fn = nn.SmoothL1Loss()
        box_loss = box_loss_fn(bbox_pred[pos_mask], bbox_labels[pos_mask])
    else:
        box_loss = torch.tensor(0.0, device=cls_logits.device)
    total_loss = cls_loss + box_lambda * box_loss
    return total_loss, cls_loss, box_loss, n_pos


def compute_accuracy(cls_logits, targets):
    probs = torch.sigmoid(cls_logits).squeeze()
    preds = (probs > 0.5).float()
    labels = targets["has_nodule"]
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def train_one_epoch(model, loader, optimizer, device, box_lambda, scaler=None):
    model.train()
    total_loss, total_cls_loss, total_box_loss = 0.0, 0.0, 0.0
    correct, total = 0, 0
    for images, targets in loader:
        images = images.to(device)
        targets = {k: v.to(device) if hasattr(v, 'to') else v for k, v in targets.items()}
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                cls_logits, bbox_pred = model(images)
                loss, cls_loss, box_loss, n_pos = compute_loss(
                    cls_logits, bbox_pred, targets, box_lambda)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            cls_logits, bbox_pred = model(images)
            loss, cls_loss, box_loss, n_pos = compute_loss(
                cls_logits, bbox_pred, targets, box_lambda)
            loss.backward()
            optimizer.step()
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_cls_loss += cls_loss.item() * bs
        total_box_loss += box_loss.item() * bs
        correct += compute_accuracy(cls_logits, targets) * bs
        total += bs
    return {
        "loss": total_loss / total, "cls_loss": total_cls_loss / total,
        "box_loss": total_box_loss / total, "accuracy": correct / total,
    }


# ============================================================
# 详细验证 & 指标计算
# ============================================================
@torch.no_grad()
def detailed_validate(model, loader, device, box_lambda):
    """
    对验证集做完整评估，收集每个样本的预测结果。

    返回:
        metrics: dict 包含 loss, accuracy, confusion_matrix,
                 precision, recall, specificity, f1, pos_count, neg_count
        records: list of dict，每个元素:
                 {image, prob, pred, label, has_nodule, bbox, pred_bbox, seriesuid}
    """
    model.eval()
    total_loss, total_cls_loss, total_box_loss = 0.0, 0.0, 0.0
    total = 0

    # 混淆矩阵元素
    tp = fp = tn = fn = 0

    # 收集每个样本的信息
    all_records = []

    for images, targets in loader:
        images = images.to(device)
        targets_dev = {k: v.to(device) if hasattr(v, 'to') else v
                       for k, v in targets.items()}

        cls_logits, bbox_pred = model(images)
        loss, cls_loss, box_loss, n_pos = compute_loss(
            cls_logits, bbox_pred, targets_dev, box_lambda)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_cls_loss += cls_loss.item() * bs
        total_box_loss += box_loss.item() * bs
        total += bs

        # 分类预测
        probs = torch.sigmoid(cls_logits).squeeze()  # [B]
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        preds = (probs > 0.5).float()
        labels = targets_dev["has_nodule"]

        # 累计混淆矩阵
        for i in range(bs):
            p = int(preds[i].item())
            l = int(labels[i].item())
            if p == 1 and l == 1:   tp += 1
            elif p == 1 and l == 0: fp += 1
            elif p == 0 and l == 0: tn += 1
            elif p == 0 and l == 1: fn += 1

            # 记录样本详情
            all_records.append({
                "image": images[i].cpu(),
                "prob": float(probs[i].item()),
                "pred": p,
                "label": l,
                "has_nodule": int(labels[i].item()),
                "bbox": targets_dev["bbox"][i].cpu().numpy(),
                "pred_bbox": bbox_pred[i].cpu().numpy(),
                "seriesuid": targets["seriesuid"][i],
            })

    # ── 计算派生指标 ──
    total_samples = tp + fp + tn + fn
    acc = (tp + tn) / total_samples if total_samples > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "loss": total_loss / total,
        "cls_loss": total_cls_loss / total,
        "box_loss": total_box_loss / total,
        "accuracy": acc,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "sensitivity": round(recall, 4),  # same as recall
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "pos_count": tp + fn,
        "neg_count": tn + fp,
        "total": total_samples,
    }
    return metrics, all_records


# ============================================================
# 错误样本保存
# ============================================================
def save_eval_samples(records, output_dir, num_per_class=5, threshold=0.5):
    """
    保存 FP / FN / TP / TN 样本图像到 output_dir/eval_samples/

    每张图像标注预测概率、真实标签和 bbox 信息。
    """
    import random
    from PIL import Image, ImageDraw, ImageFont

    sample_dir = os.path.join(output_dir, "eval_samples")
    for sub in ["FP", "FN", "TP", "TN"]:
        os.makedirs(os.path.join(sample_dir, sub), exist_ok=True)

    # 分类记录
    fp_records = [r for r in records if r["pred"] == 1 and r["label"] == 0]
    fn_records = [r for r in records if r["pred"] == 0 and r["label"] == 1]
    tp_records = [r for r in records if r["pred"] == 1 and r["label"] == 1]
    tn_records = [r for r in records if r["pred"] == 0 and r["label"] == 0]

    categories = {
        "FP": fp_records, "FN": fn_records,
        "TP": tp_records, "TN": tn_records,
    }

    for cat_name, cat_records in categories.items():
        if len(cat_records) > num_per_class:
            selected = random.sample(cat_records, num_per_class)
        else:
            selected = cat_records

        for i, rec in enumerate(selected):
            # 转图像
            img_tensor = rec["image"]  # [1, 512, 512]
            img_arr = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_arr).convert("RGB")
            draw = ImageDraw.Draw(img)

            # 标注文字
            lines = [
                f"{cat_name}  prob={rec['prob']:.3f}",
                f"label={rec['label']}  pred={rec['pred']}",
                f"seriesuid: {rec['seriesuid'][:40]}",
            ]
            if rec["has_nodule"]:
                bbox = rec["bbox"]
                lines.append(f"bbox (norm): [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")

            y = 5
            for line in lines:
                draw.text((5, y), line, fill=(255, 255, 0))
                y += 14

            # 如果有结节，画框
            if rec["has_nodule"]:
                bx, by, bw, bh = rec["bbox"]  # normalized
                bx_px = int(bx * 512)
                by_px = int(by * 512)
                bw_px = int(bw * 512)
                bh_px = int(bh * 512)
                x1 = int(bx_px - bw_px // 2)
                y1 = int(by_px - bh_px // 2)
                x2 = x1 + bw_px
                y2 = y1 + bh_px
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

            img.save(os.path.join(sample_dir, cat_name, f"{i:03d}.png"))

    # 摘要
    summary = {
        "num_fp_saved": min(len(fp_records), num_per_class),
        "num_fn_saved": min(len(fn_records), num_per_class),
        "num_tp_saved": min(len(tp_records), num_per_class),
        "num_tn_saved": min(len(tn_records), num_per_class),
        "total_fp": len(fp_records),
        "total_fn": len(fn_records),
        "total_tp": len(tp_records),
        "total_tn": len(tn_records),
    }
    with open(os.path.join(sample_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEval samples saved to {sample_dir}/")
    for cat_name, cat_records in categories.items():
        print(f"  {cat_name}: saved {min(len(cat_records), num_per_class)} / {len(cat_records)} total")


# ============================================================
# 指标保存
# ============================================================
def save_metrics(output_dir, model_name, train_metrics, val_metrics, best_epoch=None):
    """保存完整评估结果到 metrics.json"""
    metrics = {
        "model_name": model_name,
        "dataset": "LUNA16 subset0",
        "best_epoch": best_epoch,
        "metrics": {
            "train_loss": round(train_metrics["loss"], 4),
            "train_accuracy": round(train_metrics["accuracy"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_cls_loss": round(val_metrics["cls_loss"], 4),
            "val_box_loss": round(val_metrics["box_loss"], 4),
            "confusion_matrix": val_metrics.get("confusion_matrix", {}),
            "precision": val_metrics.get("precision", 0),
            "recall": val_metrics.get("recall", 0),
            "sensitivity": val_metrics.get("sensitivity", 0),
            "specificity": val_metrics.get("specificity", 0),
            "f1_score": val_metrics.get("f1_score", 0),
            "pos_count": val_metrics.get("pos_count", 0),
            "neg_count": val_metrics.get("neg_count", 0),
        },
        "note": (
            "Accuracy is slice-level nodule/background classification accuracy, "
            "not clinical diagnostic accuracy. Trained on LUNA16 subset0 only. "
            "This is a teaching/portfolio-grade prototype."
        ),
    }
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {path}")


# ============================================================
# 设备选择
# ============================================================
def setup_device(args):
    if args.device == "cpu":
        device = torch.device("cpu")
        use_amp = False
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("\n" + "=" * 60)
            print("  ERROR: --device cuda but CUDA unavailable.")
            print(f"  torch: {torch.__version__}, cuda build: {torch.version.cuda}")
            print(f"  Run: {sys.executable} tools/check_gpu.py")
            print("=" * 60)
            sys.exit(1)
        device = torch.device("cuda")
        use_amp = args.amp
    else:  # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = args.amp if device.type == "cuda" else False

    print(f"\n{'='*60}")
    print(f"  Device Configuration")
    print(f"{'='*60}")
    print(f"  Device:          {device}")
    print(f"  torch version:   {torch.__version__}")
    print(f"  cuda build:      {torch.version.cuda or 'None (CPU)'}")

    scaler = None
    if device.type == "cuda":
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU memory:      {gpu_mem:.1f} GB")
        print(f"  AMP:             {use_amp}")
        torch.backends.cudnn.benchmark = True
        if use_amp:
            scaler = torch.amp.GradScaler("cuda")
    else:
        print(f"  CUDA available:  False")

    use_pin_memory = (device.type == "cuda")
    print(f"  pin_memory:      {use_pin_memory}")
    print(f"{'='*60}")
    return device, scaler, use_pin_memory


# ============================================================
# 主函数
# ============================================================
def main():
    args = parse_args()

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(backend_dir, "weights")
    outputs_dir = os.path.join(backend_dir, "outputs")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    annotations_path = os.path.join(args.data_dir, "annotations.csv")
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(f"annotations.csv not found: {annotations_path}")

    # ── 设备 ──
    device, scaler, use_pin_memory = setup_device(args)

    # ── 缓存 ──
    if args.cache_dir is None:
        cache_dir = os.path.join(backend_dir, "outputs", "cache_slices")
    elif args.cache_dir == "":
        cache_dir = None
    else:
        cache_dir = args.cache_dir

    if args.rebuild_cache and cache_dir is not None:
        print("\n" + "=" * 60)
        print("Rebuilding slice cache...")
        print("=" * 60)
        from training.luna_dataset import build_slice_cache
        build_slice_cache(
            data_dir=args.data_dir, annotations_path=annotations_path,
            cache_dir=cache_dir, limit_samples=args.limit_samples,
        )

    # ── 数据加载 ──
    print(f"\n{'='*60}")
    print("Loading data...")
    if cache_dir is not None:
        meta_path = os.path.join(cache_dir, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  Cache: {meta['num_train']} train + {meta['num_val']} val slices")
    print(f"  num_workers: {args.num_workers}")
    print(f"{'='*60}")

    train_loader, val_loader, train_ds, val_ds = create_dataloaders(
        data_dir=args.data_dir, annotations_path=annotations_path,
        limit_samples=args.limit_samples, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=use_pin_memory,
        cache_dir=cache_dir,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── 模型 ──
    model = SimpleCTDetector().to(device)
    print(f"\nModel: SimpleCTDetector, {count_parameters(model):,} params")

    best_model_path = os.path.join(weights_dir, "best_model.pt")

    # ── 仅评估模式 ──
    if args.evaluate:
        if not os.path.isfile(best_model_path):
            print(f"\nERROR: {best_model_path} not found. Train first.")
            sys.exit(1)
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}, "
              f"val_acc={checkpoint.get('val_accuracy', '?')}")

        val_metrics, records = detailed_validate(model, val_loader, device, args.box_lambda)

        print(f"\n{'='*60}")
        print("  Evaluation Results")
        print(f"{'='*60}")
        _cm = val_metrics["confusion_matrix"]
        print(f"  Confusion Matrix:  TP={_cm['TP']}  FP={_cm['FP']}  "
              f"TN={_cm['TN']}  FN={_cm['FN']}")
        print(f"  Accuracy:     {val_metrics['accuracy']:.4f}")
        print(f"  Precision:    {val_metrics['precision']:.4f}")
        print(f"  Recall:       {val_metrics['recall']:.4f}")
        print(f"  Specificity:  {val_metrics['specificity']:.4f}")
        print(f"  F1 Score:     {val_metrics['f1_score']:.4f}")
        print(f"  Pos/Neg:      {val_metrics['pos_count']}/{val_metrics['neg_count']}")
        print(f"{'='*60}")

        # 保存错误样本
        save_eval_samples(records, outputs_dir, num_per_class=args.eval_samples)
        # 保存指标
        save_metrics(outputs_dir, "RadScout-SimpleCTDetector-v2.1",
                      {"loss": 0, "accuracy": 0}, val_metrics,
                      best_epoch=checkpoint.get("epoch"))
        return

    # ── 优化器 ──
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── 训练循环 ──
    best_val_acc = 0.0
    best_epoch = 0
    csv_path = os.path.join(outputs_dir, "train_log.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,"
                "precision,recall,f1,cls_loss,box_loss,lr\n")

    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs...")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, args.box_lambda, scaler)
        val_metrics, _ = detailed_validate(
            model, val_loader, device, args.box_lambda)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # 每 epoch 打印（含详细指标）
        _cm = val_metrics["confusion_matrix"]
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss={train_metrics['loss']:.4f} | "
              f"train_acc={train_metrics['accuracy']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | "
              f"P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} "
              f"F1={val_metrics['f1_score']:.3f} | "
              f"TP={_cm['TP']} FP={_cm['FP']} TN={_cm['TN']} FN={_cm['FN']} | "
              f"{elapsed:.1f}s")

        # CSV
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_metrics['loss']:.6f},{train_metrics['accuracy']:.6f},"
                    f"{val_metrics['loss']:.6f},{val_metrics['accuracy']:.6f},"
                    f"{val_metrics['precision']:.4f},{val_metrics['recall']:.4f},"
                    f"{val_metrics['f1_score']:.4f},"
                    f"{val_metrics['cls_loss']:.6f},{val_metrics['box_loss']:.6f},"
                    f"{current_lr:.6f}\n")

        # 最佳模型
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
            }, best_model_path)
            print(f"  -> Best model saved (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete! Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")

    # ── 最终评估（使用最佳模型） ──
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_val_metrics, records = detailed_validate(
        model, val_loader, device, args.box_lambda)

    # 最终训练指标
    final_train_metrics = train_one_epoch(
        model, train_loader, optimizer, device, args.box_lambda)

    # 保存
    _cm = final_val_metrics["confusion_matrix"]
    print(f"\n{'='*60}")
    print("  Final Evaluation (best model)")
    print(f"{'='*60}")
    print(f"  TP={_cm['TP']}  FP={_cm['FP']}  TN={_cm['TN']}  FN={_cm['FN']}")
    print(f"  Accuracy={final_val_metrics['accuracy']:.4f}  "
          f"Precision={final_val_metrics['precision']:.4f}  "
          f"Recall={final_val_metrics['recall']:.4f}")
    print(f"  Specificity={final_val_metrics['specificity']:.4f}  "
          f"F1={final_val_metrics['f1_score']:.4f}")
    print(f"{'='*60}")

    save_metrics(outputs_dir, "RadScout-SimpleCTDetector-v2.1",
                  final_train_metrics, final_val_metrics, best_epoch=best_epoch)
    save_eval_samples(records, outputs_dir, num_per_class=args.eval_samples)

    print(f"\nTrain log: {csv_path}")
    print(f"Model: {best_model_path}")


if __name__ == "__main__":
    main()
