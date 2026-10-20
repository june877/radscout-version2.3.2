"""
build_candidate_dataset_png.py — YOLO candidates → PNG 2.5D patch 数据集

与 v3.0 rawCT 版本的区别:
  旧版: 从 .mhd/.raw 原始 CT 裁剪 HU patch
  新版: 从 YOLO 数据集 RGB PNG 的三通道 (R=z-1, G=z, B=z+1) 裁剪 patch

用途: 保证 ResNet18 训练输入与产品推理输入 (连续PNG→RGB合成) 一致。

用法:
    cd D:/python_gjw/radscout2/version3.1/backend
    D:/Python/python.exe training/build_candidate_dataset_png.py \
        --yolo_weights weights/yolo11s_25d_3slice_best.pt \
        --yolo_dataset D:/python_gjw/radscout2/version2.3/datasets/luna_yolo_25d_3slice \
        --out_dir ../datasets/luna_candidate_patches_png \
        --device cuda
"""

import os, sys, json, time, csv, argparse, random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, r"D:/python_gjw/radscout2/version2.3/backend")
from training.bbox_utils import yolo_to_xyxy, expand_bbox, clamp_bbox, compute_iou


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo_weights", type=str, required=True)
    p.add_argument("--yolo_dataset", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--conf_threshold", type=float, default=0.10)
    p.add_argument("--expand_ratio", type=float, default=1.5)
    p.add_argument("--patch_size", type=int, default=96)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit_images", type=int, default=None)
    p.add_argument("--pos_iou", type=float, default=0.3)
    p.add_argument("--neg_iou", type=float, default=0.1)
    return p.parse_args()


def read_yolo_label(path):
    """读取 YOLO label → [(cx,cy,w,h), ...] 归一化坐标"""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    boxes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 5:
                boxes.append(tuple(float(x) for x in parts[1:5]))
    return boxes


def pil_to_patch_tensor(pil_img, x1, y1, x2, y2, expand_ratio, patch_size, img_w, img_h):
    """
    从 PIL RGB 图像中按 bbox 位置裁剪并 resize → [3, 96, 96] patch tensor (float32, [0,1])。
    返回 (patch_tensor, expanded_bbox) 或 (None, None)。
    """
    ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, expand_ratio)
    ex1, ey1, ex2, ey2 = clamp_bbox(ex1, ey1, ex2, ey2, img_w, img_h, min_size=16)
    if ex2 <= ex1 + 8 or ey2 <= ey1 + 8:
        return None, None
    crop = pil_img.crop((ex1, ey1, ex2, ey2)).resize((patch_size, patch_size), Image.BILINEAR)
    arr = np.array(crop).astype(np.float32) / 255.0  # [96,96,3]
    return torch.from_numpy(arr.transpose(2, 0, 1)), (ex1, ey1, ex2, ey2)


def build_dataset(args):
    t0 = time.time()
    out_dir = os.path.abspath(args.out_dir)
    yolo_dir = os.path.abspath(args.yolo_dataset)
    os.makedirs(os.path.join(out_dir, "patches", "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "patches", "val"), exist_ok=True)

    # ── Load YOLO ──
    print("=" * 60)
    print("  Loading YOLO...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    print(f"  Loaded: {args.yolo_weights}")

    # ── Scan splits ──
    csv_rows = []
    sample_id = 0
    stats = {"train": {"imgs": 0, "pos": 0, "neg": 0, "skip": 0, "err": 0},
             "val":   {"imgs": 0, "pos": 0, "neg": 0, "skip": 0, "err": 0}}

    for split in ["train", "val"]:
        img_dir = os.path.join(yolo_dir, "images", split)
        lbl_dir = os.path.join(yolo_dir, "labels", split)
        png_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        if args.limit_images:
            png_files = png_files[:args.limit_images]
        print(f"\n  --- {split}: {len(png_files)} images ---")

        for fname in tqdm(png_files, desc=f"YOLO {split}"):
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, fname.replace(".png", ".txt"))
            stats[split]["imgs"] += 1

            # YOLO inference
            try:
                results = yolo.predict(img_path, conf=args.conf_threshold,
                                       device=args.device, imgsz=args.imgsz, verbose=False)
            except Exception:
                continue

            yolo_preds = []  # list of (cx,cy,w,h,conf)
            if results and len(results) > 0 and results[0].boxes is not None:
                xyxy = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                iw, ih = results[0].orig_shape[1], results[0].orig_shape[0]
                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j]
                    cx = (x1 + x2) / 2 / iw; cy = (y1 + y2) / 2 / ih
                    w = (x2 - x1) / iw; h = (y2 - y1) / ih
                    yolo_preds.append((cx, cy, w, h, float(confs[j])))

            if len(yolo_preds) == 0:
                continue

            # GT labels
            gt_boxes = read_yolo_label(lbl_path)

            # IoU matching
            ious = []
            for yp in yolo_preds:
                px1, py1, px2, py2 = yolo_to_xyxy(yp[0], yp[1], yp[2], yp[3], args.imgsz, args.imgsz)
                max_iou = 0.0
                for gb in gt_boxes:
                    gx1, gy1, gx2, gy2 = yolo_to_xyxy(gb[0], gb[1], gb[2], gb[3], args.imgsz, args.imgsz)
                    max_iou = max(max_iou, compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2)))
                ious.append(max_iou)

            # Open image once
            pil_img = Image.open(img_path).convert("RGB")
            iw, ih = pil_img.width, pil_img.height

            for j, (yp, iou) in enumerate(zip(yolo_preds, ious)):
                if iou >= args.pos_iou:
                    label = 1; stats[split]["pos"] += 1
                elif iou <= args.neg_iou:
                    label = 0; stats[split]["neg"] += 1
                else:
                    stats[split]["skip"] += 1; continue

                cx, cy, w, h, conf = yp
                px1, py1, px2, py2 = yolo_to_xyxy(cx, cy, w, h, args.imgsz, args.imgsz)

                patch, expanded = pil_to_patch_tensor(
                    pil_img, px1, py1, px2, py2, args.expand_ratio, args.patch_size, iw, ih)
                if patch is None:
                    stats[split]["err"] += 1; continue

                patch_fname = f"{sample_id:08d}.pt"
                patch_path = os.path.join(out_dir, "patches", split, patch_fname)
                torch.save(patch, patch_path)

                ex1, ey1, ex2, ey2 = expanded
                csv_rows.append({
                    "sample_id": sample_id, "split": split,
                    "image_path": img_path, "label_path": lbl_path,
                    "x1": round(px1, 1), "y1": round(py1, 1),
                    "x2": round(px2, 1), "y2": round(py2, 1),
                    "expanded_x1": ex1, "expanded_y1": ey1,
                    "expanded_x2": ex2, "expanded_y2": ey2,
                    "yolo_score": round(conf, 4), "max_iou": round(iou, 4),
                    "label": label, "patch_path": patch_path,
                })
                sample_id += 1

    # ── Save CSV ──
    csv_path = os.path.join(out_dir, "candidate_dataset_png.csv")
    fields = ["sample_id", "split", "image_path", "label_path",
              "x1", "y1", "x2", "y2", "expanded_x1", "expanded_y1", "expanded_x2", "expanded_y2",
              "yolo_score", "max_iou", "label", "patch_path"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(csv_rows)

    # ── Summary ──
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "PNG 2.5D patches (from YOLO dataset RGB images, NOT raw CT)",
        "config": {"conf_threshold": args.conf_threshold, "expand_ratio": args.expand_ratio,
                   "patch_size": args.patch_size, "imgsz": args.imgsz},
        "total_samples": sample_id,
        "train": stats["train"], "val": stats["val"],
    }
    with open(os.path.join(out_dir, "build_summary_png.json"), "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  BUILD COMPLETE ({elapsed:.1f}s)")
    print(f"  Total: {sample_id} samples")
    for s in ["train", "val"]:
        st = stats[s]
        print(f"  {s}: {st['imgs']} imgs → pos={st['pos']} neg={st['neg']} skip={st['skip']} err={st['err']}")
    print(f"  CSV: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    build_dataset(parse_args())
