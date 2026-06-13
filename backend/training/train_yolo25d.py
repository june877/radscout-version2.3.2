"""
train_yolo25d.py — YOLO11s 3-slice 2.5D 训练

与 v2.2 train_yolo.py 完全相同的 API，只是默认路径指向 2.5D 数据集。

用法:
    cd D:/python_gjw/radscout2/version2.3/backend
    D:/Python/python.exe training/train_yolo25d.py \
        --model yolo11s.pt \
        --data ../datasets/luna_yolo_25d_3slice/data.yaml \
        --epochs 100 --imgsz 640 --batch 8 --device 0 --workers 2 --amp
"""

import os, sys, json, shutil, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11s 2.5D on LUNA16")
    p.add_argument("--model", type=str, default="yolo11s.pt")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


def extract_metrics(results_dir):
    results_csv = os.path.join(results_dir, "results.csv")
    if not os.path.isfile(results_csv): return {}
    import pandas as pd
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    if "metrics/mAP50-95(B)" in df.columns:
        best_idx = df["metrics/mAP50-95(B)"].idxmax()
    else:
        best_idx = df.index[-1]
    best = df.iloc[best_idx]
    return {
        "epoch": int(best.get("epoch", 0)),
        "precision": round(float(best.get("metrics/precision(B)", 0)), 4),
        "recall": round(float(best.get("metrics/recall(B)", 0)), 4),
        "mAP50": round(float(best.get("metrics/mAP50(B)", 0)), 4),
        "mAP50_95": round(float(best.get("metrics/mAP50-95(B)", 0)), 4),
        "train_box_loss": round(float(best.get("train/box_loss", 0)), 4),
        "train_cls_loss": round(float(best.get("train/cls_loss", 0)), 4),
        "val_box_loss": round(float(best.get("val/box_loss", 0)), 4),
        "val_cls_loss": round(float(best.get("val/cls_loss", 0)), 4),
    }


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed."); sys.exit(1)

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(backend_dir, "weights")
    outputs_dir = os.path.join(backend_dir, "outputs")
    runs_dir = os.path.join(backend_dir, "runs", "yolo25d_3slice")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    data_path = os.path.abspath(args.data)
    if not os.path.isfile(data_path):
        print(f"ERROR: data.yaml not found: {data_path}")
        print("  Run build_yolo25d_dataset.py first!"); sys.exit(1)

    print("\n" + "="*60)
    print("  YOLO11s 3-Slice 2.5D Training")
    print("  R=z-1  G=z(center)  B=z+1")
    print("="*60)
    print(f"  Model:  {args.model}")
    print(f"  Data:   {data_path}")
    print(f"  Epochs: {args.epochs}, Imgsz: {args.imgsz}, Batch: {args.batch}")
    print(f"  Device: {args.device}, Workers: {args.workers}")
    print("="*60)

    model = YOLO(args.model)
    print(f"\nStarting 2.5D training...\n")

    try:
        results = model.train(
            data=data_path, epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, device=args.device, workers=args.workers,
            amp=args.amp, lr0=args.lr0, patience=args.patience,
            project=runs_dir, name="train", exist_ok=True, verbose=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "="*60 + "\n  CUDA OOM!\n  Try: --batch " +
                  str(max(1, args.batch//2)) + "\n" + "="*60)
        raise

    # 复制最佳权重
    best_src = os.path.join(runs_dir, "train", "weights", "best.pt")
    best_dst = os.path.join(weights_dir, "yolo11s_25d_3slice_best.pt")
    if os.path.isfile(best_src):
        shutil.copy2(best_src, best_dst)
        print(f"\nBest weights: {best_dst}")
    else:
        last_src = os.path.join(runs_dir, "train", "weights", "last.pt")
        if os.path.isfile(last_src):
            shutil.copy2(last_src, best_dst)

    # 指标
    metrics = extract_metrics(os.path.join(runs_dir, "train"))
    yolo_metrics = {
        "model_name": "RadScout-YOLO11s-2.5D-v2.3",
        "dataset": "LUNA16 subset0-6 (3-slice 2.5D)",
        "model": args.model,
        "input": "R=z-1, G=z, B=z+1",
        "metrics": metrics if metrics else {},
        "note": "YOLO11s 3-slice 2.5D CT nodule detector. Teaching/portfolio prototype.",
    }
    metrics_path = os.path.join(outputs_dir, "yolo25d_3slice_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(yolo_metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_path}")

    if metrics:
        print(f"\n  Best (by mAP50-95):")
        print(f"    Precision: {metrics.get('precision')}")
        print(f"    Recall:    {metrics.get('recall')}")
        print(f"    mAP50:     {metrics.get('mAP50')}")
        print(f"    mAP50-95:  {metrics.get('mAP50_95')}")


if __name__ == "__main__":
    main()
