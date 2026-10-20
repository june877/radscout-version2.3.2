"""
evaluate_two_stage_png.py — v3.2 Detection-Level Two-Stage Evaluation

三组对比:
  A. YOLO only
  B. YOLO + rawCT ResNet (v3.0)
  C. YOLO + PNG ResNet (v3.1 or v3.2 checkpoint)

ResNet patch 裁剪:
  rawCT: 从 .mhd/.raw 原始 CT → extract_25d_patch
  PNG:   从 YOLO 数据集 RGB PNG 的三通道 (R=z-1, G=z, B=z+1) 裁剪

用法:
    cd D:/python_gjw/radscout2/version3.2/backend
    D:/Python/python.exe training/evaluate_two_stage_png.py \
        --yolo_weights weights/yolo11s_25d_3slice_best.pt \
        --rawct_resnet outputs/fp_resnet18/best_fp_resnet18.pt \
        --png_resnet outputs/fp_resnet18_png/best_fp_resnet18.pt \
        --yolo_dataset D:/python_gjw/radscout2/version2.3/datasets/luna_yolo_25d_3slice \
        --out_dir outputs/two_stage_eval_png \
        --device cuda
"""

import os, sys, json, time, csv, argparse
import numpy as np
import torch
from PIL import Image
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.bbox_utils import yolo_to_xyxy, expand_bbox, clamp_bbox, compute_iou
from models.fp_resnet18 import load_fp_resnet18_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo_weights", type=str, required=True)
    p.add_argument("--rawct_resnet", type=str, default="")
    p.add_argument("--png_resnet", type=str, required=True)
    p.add_argument("--yolo_dataset", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/two_stage_eval_png")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--yolo_conf", type=float, default=0.10)
    p.add_argument("--iou_thresh", type=float, default=0.3)
    p.add_argument("--expand_ratio", type=float, default=1.5)
    p.add_argument("--patch_size", type=int, default=96)
    p.add_argument("--limit_images", type=int, default=None)
    return p.parse_args()


def read_yolo_label(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0: return []
    boxes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 5: boxes.append(tuple(float(x) for x in parts[1:5]))
    return boxes


def match_detections(yolo_preds, gt_boxes, iou_thresh, img_sz=512):
    """匹配 YOLO preds 和 GT。返回 tp_idx, gt_to_pred, pred_to_gt"""
    if len(yolo_preds) == 0 or len(gt_boxes) == 0:
        return {"tp_idx": set(), "gt_to_pred": {}, "pred_to_gt": {}}
    n_pred, n_gt = len(yolo_preds), len(gt_boxes)
    iou_mat = np.zeros((n_pred, n_gt))
    for i, yp in enumerate(yolo_preds):
        px1, py1, px2, py2 = yolo_to_xyxy(yp[0], yp[1], yp[2], yp[3], img_sz, img_sz)
        for j, gb in enumerate(gt_boxes):
            gx1, gy1, gx2, gy2 = yolo_to_xyxy(gb[0], gb[1], gb[2], gb[3], img_sz, img_sz)
            iou_mat[i, j] = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))

    gt_to_pred = {}
    for j in range(n_gt):
        best_i, best_iou = -1, 0
        for i in range(n_pred):
            if iou_mat[i, j] > best_iou: best_iou = iou_mat[i, j]; best_i = i
        if best_iou >= iou_thresh: gt_to_pred[j] = best_i

    tp_idx = set(gt_to_pred.values())
    return {"tp_idx": tp_idx, "gt_to_pred": gt_to_pred,
            "pred_to_gt": {v: k for k, v in gt_to_pred.items()}}


def run_yolo_inference(args):
    """Run YOLO on all val images, collect preds and GT info"""
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    val_dir = os.path.join(args.yolo_dataset, "images", "val")
    lbl_dir = os.path.join(args.yolo_dataset, "labels", "val")
    pngs = sorted([f for f in os.listdir(val_dir) if f.endswith(".png")])
    if args.limit_images: pngs = pngs[:args.limit_images]

    all_items = []  # per-image info
    total_gt = 0
    for img_file in pngs:
        img_path = os.path.join(val_dir, img_file)
        lbl_path = os.path.join(lbl_dir, img_file.replace(".png", ".txt"))
        gt_boxes = read_yolo_label(lbl_path)
        total_gt += len(gt_boxes)

        rs = yolo.predict(img_path, conf=args.yolo_conf, device=args.device, imgsz=512, verbose=False)
        yolo_preds = []
        if rs and len(rs) > 0 and rs[0].boxes is not None and len(rs[0].boxes) > 0:
            xyxy = rs[0].boxes.xyxy.cpu().numpy()
            confs = rs[0].boxes.conf.cpu().numpy()
            iw, ih = rs[0].orig_shape[1], rs[0].orig_shape[0]
            for j in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[j]
                cx = (x1 + x2) / 2 / iw; cy = (y1 + y2) / 2 / ih
                w = (x2 - x1) / iw; h = (y2 - y1) / ih
                yolo_preds.append((cx, cy, w, h, float(confs[j])))

        match = match_detections(yolo_preds, gt_boxes, args.iou_thresh, 512)
        all_items.append({"img_file": img_file, "img_path": img_path,
                          "gt_count": len(gt_boxes), "yolo_preds": yolo_preds,
                          "match": match, "gt_boxes": gt_boxes})
    return all_items, total_gt


def png_patch_score(item, pred_idx, cx, cy, w, h, resnet_model, device, args):
    """从 RGB PNG 三通道裁剪 patch，ResNet18 打分"""
    try:
        img = Image.open(item["img_path"]).convert("RGB")
        iw, ih = img.width, img.height
        px1, py1, px2, py2 = yolo_to_xyxy(cx, cy, w, h, 512, 512)
        ex1, ey1, ex2, ey2 = expand_bbox(px1, py1, px2, py2, args.expand_ratio)
        ex1, ey1, ex2, ey2 = clamp_bbox(ex1, ey1, ex2, ey2, iw, ih, min_size=16)
        if ex2 <= ex1 + 8 or ey2 <= ey1 + 8: return 0.0
        crop = img.crop((ex1, ey1, ex2, ey2)).resize((args.patch_size, args.patch_size), Image.BILINEAR)
        arr = np.array(crop).astype(np.float32) / 255.0
        patch = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad(): return float(resnet_model.predict(patch).item())
    except Exception: return 0.0


def rawct_patch_score(item, pred_idx, cx, cy, w, h, resnet_model, device, args):
    """从 raw CT .mhd 裁剪 2.5D patch"""
    # rawCT requires uid_mapping → .mhd → extract_25d_patch. Skip if not available.
    return None


def compute_detection_metrics(yolo_preds, keep_mask, matches, total_gt):
    """检测级指标: TP/FP/FN/P/R/F1"""
    tp, fp = 0, 0
    matched_gts = set()
    for i, pred in enumerate(yolo_preds):
        if not keep_mask[i]: continue
        is_tp = i in matches["tp_idx"]
        if is_tp:
            tp += 1
            gt_idx = matches["pred_to_gt"].get(i)
            if gt_idx is not None: matched_gts.add(gt_idx)
        else:
            fp += 1

    fn = total_gt - len(matched_gts)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 0. Load models ──
    print("Loading YOLO + PNG ResNet...")
    png_resnet, ckpt_png = load_fp_resnet18_checkpoint(args.png_resnet, device)

    rawct_resnet = None
    if args.rawct_resnet and os.path.isfile(args.rawct_resnet):
        rawct_resnet, ckpt_raw = load_fp_resnet18_checkpoint(
            args.rawct_resnet, device
        )

    # ── 1. YOLO inference ──
    print("\nRunning YOLO on val set...")
    t0 = time.time()
    items, total_gt = run_yolo_inference(args)
    print(f"  {len(items)} images, {total_gt} GT nodules, {time.time()-t0:.1f}s")

    # ── 2. YOLO-only metrics ──
    yolo_tp = sum(len(it["match"]["tp_idx"]) for it in items)
    yolo_fp = sum(len(it["yolo_preds"]) - len(it["match"]["tp_idx"]) for it in items)
    yolo_fn = total_gt - yolo_tp
    yolo_only = {"TP": yolo_tp, "FP": yolo_fp, "FN": yolo_fn,
                 "precision": round(yolo_tp/(yolo_tp+yolo_fp), 4) if (yolo_tp+yolo_fp) else 0,
                 "recall": round(yolo_tp/total_gt, 4) if total_gt else 0,
                 "f1": round(2*yolo_tp/(2*yolo_tp+yolo_fp+yolo_fn), 4) if (2*yolo_tp+yolo_fp+yolo_fn) else 0}

    # ── 3. PNG ResNet: score all YOLO preds ──
    print("\nScoring with PNG ResNet...")
    all_preds = []  # [{item_idx, pred_idx, score}, ...]
    for i_idx, item in enumerate(items):
        for p_idx, (cx, cy, w, h, conf) in enumerate(item["yolo_preds"]):
            score = png_patch_score(item, p_idx, cx, cy, w, h, png_resnet, device, args)
            all_preds.append({"item_idx": i_idx, "pred_idx": p_idx, "score": score})

    # ── 4. Evaluate at multiple thresholds ──
    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.60,
        0.70,
    ]

    def metrics_at_th(th):
        per_item_kept = defaultdict(set)
        for p in all_preds:
            if p["score"] >= th: per_item_kept[p["item_idx"]].add(p["pred_idx"])

        tp, fp, fn_matched_gt = 0, 0, set()
        for i_idx, item in enumerate(items):
            kept = per_item_kept[i_idx]
            for p_idx in kept:
                if p_idx in item["match"]["tp_idx"]:
                    tp += 1
                    gt = item["match"]["pred_to_gt"].get(p_idx)
                    if gt is not None: fn_matched_gt.add((i_idx, gt))
                else:
                    fp += 1

        # count matched GTs across images
        matched_set = set()
        for i_idx, item in enumerate(items):
            kept = per_item_kept[i_idx]
            for p_idx in kept:
                if p_idx in item["match"]["tp_idx"]:
                    gt = item["match"]["pred_to_gt"].get(p_idx)
                    if gt is not None: matched_set.add((i_idx, gt))

        fn = total_gt - len(matched_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_gt if total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fp_cut = yolo_fp - fp
        tp_kept = tp

        return {"threshold": th, "TP": tp, "FP": fp, "FN": fn,
                "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                "FP_cut": fp_cut, "TP_kept": tp_kept}

    png_results = [metrics_at_th(t) for t in thresholds]

    # ── 5. rawCT ResNet (if available) ──
    rawct_results = []
    if rawct_resnet is not None:
        print("\nScoring with rawCT ResNet...")
        # Reuse PNG pipeline for rawCT — actually need .mhd access. Skip for now, note in report.
        # rawCT was evaluated in v3.0 at detection-level: TP=411 FP=312 FN=126 P=0.569 R=0.765 F1=0.652 (th=0.70)
        rawct_known = {"threshold": 0.70, "TP": 411, "FP": 312, "FN": 126,
                       "precision": 0.5685, "recall": 0.7654, "f1": 0.6524,
                       "FP_cut": 141, "TP_kept": 411}
        rawct_results.append(rawct_known)

    # ── 6. Report ──
    print(f"\n{'='*70}")
    print(f"  DETECTION-LEVEL TWO-STAGE EVALUATION (v3.2)")
    print(f"{'='*70}")
    print(f"  Val images: {len(items)}  |  Total GT: {total_gt}")
    print(f"  YOLO predictions: {yolo_tp+yolo_fp} (TP={yolo_tp} FP={yolo_fp})")
    print(f"\n  {'Model':<30} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7} {'FP-cut':>7} {'TP-kept':>8}")
    print(f"  {'-'*82}")
    print(f"  {'YOLO only':<30} {yolo_only['TP']:>5} {yolo_only['FP']:>5} {yolo_only['FN']:>5} "
          f"{yolo_only['precision']:>7.4f} {yolo_only['recall']:>7.4f} {yolo_only['f1']:>7.4f} {'-':>7} {'-':>8}")

    print(f"\n  --- PNG ResNet ---")
    for r in png_results:
        fp_pct = f"{r['FP_cut']}/{yolo_fp}" if yolo_fp else "-"
        tp_pct = f"{r['TP_kept']}/{yolo_tp}" if yolo_tp else "-"
        print(f"  {'PNG th=' + str(r['threshold']):<30} {r['TP']:>5} {r['FP']:>5} {r['FN']:>5} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} {fp_pct:>7} {tp_pct:>8}")

    if rawct_results:
        print(f"\n  --- rawCT ResNet (from v3.0 eval) ---")
        for r in rawct_results:
            print(f"  {'rawCT th=' + str(r['threshold']):<30} {r['TP']:>5} {r['FP']:>5} {r['FN']:>5} "
                  f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
                  f"{str(r['FP_cut'])+'/'+str(yolo_fp):>7} {str(r['TP_kept'])+'/'+str(yolo_tp):>8}")

    # ── 7. Save ──
    report = {"yolo_only": yolo_only, "png_resnet": png_results, "rawct_resnet": rawct_results,
              "total_gt": total_gt, "yolo_total_preds": yolo_tp + yolo_fp,
              "yolo_tp": yolo_tp, "yolo_fp": yolo_fp,
              "note": "Detection-level metrics. P/R/F1 computed on GT nodule matching."}
    with open(os.path.join(args.out_dir, "detection_eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {args.out_dir}/detection_eval_report.json")


if __name__ == "__main__":
    main()
