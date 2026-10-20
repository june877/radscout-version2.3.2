"""
analyze_fp_thresholds.py — best_fp_resnet18.pt 阈值敏感性分析

用法:
    cd D:/python_gjw/radscout2/version3.0/backend
    D:/Python/python.exe training/analyze_fp_thresholds.py \
        --csv D:/python_gjw/radscout2/version2.3/datasets/luna_candidate_patches/candidate_dataset.csv \
        --model outputs/fp_resnet18/best_fp_resnet18.pt \
        --out_dir outputs/fp_resnet18
"""

import os, sys, csv, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fp_resnet18 import load_fp_resnet18_checkpoint
from training.train_fp_resnet18 import CandidatePatchDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/fp_resnet18")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def compute_at_threshold(probs, labels, thresh):
    preds = (probs >= thresh).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    balanced_acc = (recall + specificity) / 2

    return {
        "threshold": round(thresh, 2),
        "accuracy": round(acc, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "specificity": round(specificity, 4),
        "f1": round(f1, 4), "balanced_acc": round(balanced_acc, 4),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──
    print(f"\nLoading model: {args.model}")
    model, ckpt = load_fp_resnet18_checkpoint(args.model, device)
    print(f"  Loaded epoch {ckpt['epoch']}, F1={ckpt['val_metrics']['f1']}")

    # ── Load val data ──
    print(f"\nLoading val data...")
    ds = CandidatePatchDataset(args.csv, "val")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    # ── Run inference ──
    print("Running inference on val set...")
    all_probs, all_labels = [], []
    all_sample_ids = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # class 1 = real nodule
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    # ── Save val_predictions.csv （补全所有字段） ──
    preds_path = os.path.join(args.out_dir, "val_predictions.csv")
    with open(args.csv) as f_in, open(preds_path, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = ["sample_id", "label", "resnet_score", "yolo_score", "max_iou", "patch_path"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        idx = 0
        for row in reader:
            if row["split"] != "val":
                continue
            writer.writerow({
                "sample_id": row["sample_id"],
                "label": row["label"],
                "resnet_score": round(float(probs[idx]), 6),
                "yolo_score": row["yolo_score"],
                "max_iou": row["max_iou"],
                "patch_path": row["patch_path"],
            })
            idx += 1
    print(f"  Saved {idx} rows → {preds_path}")

    # ── Sweep thresholds ──
    print(f"\nSweeping thresholds 0.05 → 0.95...")
    thresholds = np.arange(0.05, 1.0, 0.05)
    results = [compute_at_threshold(probs, labels, t) for t in thresholds]

    # Summaries
    best_f1 = max(results, key=lambda r: r["f1"])
    recall_ge_095 = [r for r in results if r["recall"] >= 0.95]
    recall_ge_090 = [r for r in results if r["recall"] >= 0.90]
    best_spec_095 = max(recall_ge_095, key=lambda r: r["specificity"]) if recall_ge_095 else None
    best_spec_090 = max(recall_ge_090, key=lambda r: r["specificity"]) if recall_ge_090 else None
    best_bal = max(results, key=lambda r: r["balanced_acc"])

    # Print
    print(f"\n{'='*60}")
    print(f"  Threshold Analysis Results")
    print(f"{'='*60}")
    header = f"{'Thr':>5}  {'Acc':>7}  {'P':>7}  {'R':>7}  {'Spec':>7}  {'F1':>7}  {'bAcc':>7}  {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}"
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r['threshold']:5.2f}  {r['accuracy']:7.4f}  {r['precision']:7.4f}  "
              f"{r['recall']:7.4f}  {r['specificity']:7.4f}  {r['f1']:7.4f}  "
              f"{r['balanced_acc']:7.4f}  {r['TP']:5d} {r['FP']:5d} {r['TN']:5d} {r['FN']:5d}")

    print(f"\n{'='*60}")
    print(f"  Key Thresholds")
    print(f"{'='*60}")
    print(f"  Best F1:          thr={best_f1['threshold']:.2f}  "
          f"F1={best_f1['f1']:.4f}  P={best_f1['precision']:.4f}  "
          f"R={best_f1['recall']:.4f}  Spec={best_f1['specificity']:.4f}")
    if best_spec_095:
        print(f"  Recall>=0.95:     thr={best_spec_095['threshold']:.2f}  "
              f"Spec={best_spec_095['specificity']:.4f}  R={best_spec_095['recall']:.4f}")
    if best_spec_090:
        print(f"  Recall>=0.90:     thr={best_spec_090['threshold']:.2f}  "
              f"Spec={best_spec_090['specificity']:.4f}  R={best_spec_090['recall']:.4f}")
    print(f"  Best balanced acc: thr={best_bal['threshold']:.2f}  "
          f"bAcc={best_bal['balanced_acc']:.4f}  "
          f"R={best_bal['recall']:.4f}  Spec={best_bal['specificity']:.4f}")
    print(f"{'='*60}")

    # Default 0.5
    r05 = compute_at_threshold(probs, labels, 0.5)
    print(f"\n  Default threshold=0.5:")
    print(f"    P={r05['precision']:.4f} R={r05['recall']:.4f} "
          f"F1={r05['f1']:.4f} Spec={r05['specificity']:.4f}")
    print(f"    TP={r05['TP']} FP={r05['FP']} TN={r05['TN']} FN={r05['FN']}")

    # ── Save CSVs ──
    tsv_path = os.path.join(args.out_dir, "threshold_metrics.csv")
    with open(tsv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n  Saved threshold sweep → {tsv_path}")
    print(f"  Saved predictions → {preds_path}")


if __name__ == "__main__":
    main()
