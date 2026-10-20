"""
evaluate_two_stage.py — YOLO vs YOLO+ResNet18 二阶段检测级评估

用法:
    cd D:/python_gjw/radscout2/version3.0/backend
    D:/Python/python.exe training/evaluate_two_stage.py \
        --yolo_weights weights/yolo11s_25d_3slice_best.pt \
        --resnet_model outputs/fp_resnet18/best_fp_resnet18.pt \
        --yolo_dataset D:/python_gjw/radscout2/version2.3/datasets/luna_yolo_25d_3slice \
        --candidate_csv D:/python_gjw/radscout2/version2.3/datasets/luna_candidate_patches/candidate_dataset.csv \
        --data_dir D:/python_gjw/radscout2/data/luna16 \
        --out_dir outputs/two_stage_eval \
        --device cuda
"""

import os, sys, json, time, csv, argparse
import numpy as np
import torch
from PIL import Image
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 复用 v2.3 的 bbox_utils
sys.path.insert(0, r"D:/python_gjw/radscout2/version2.3/backend")
from training.bbox_utils import yolo_to_xyxy, compute_iou


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo_weights", type=str, required=True)
    p.add_argument("--resnet_model", type=str, required=True)
    p.add_argument("--yolo_dataset", type=str, required=True)
    p.add_argument("--candidate_csv", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="D:/python_gjw/radscout2/data/luna16")
    p.add_argument("--out_dir", type=str, default="outputs/two_stage_eval")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--conf_threshold", type=float, default=0.1,
                   help="YOLO minimum confidence")
    p.add_argument("--iou_threshold", type=float, default=0.3,
                   help="IoU threshold for TP matching")
    p.add_argument("--limit_images", type=int, default=None)
    p.add_argument("--explicit", type=str, default=None,
                   help="Comma-separated image filenames to debug (e.g. 'img1.png,img2.png')")
    return p.parse_args()


def read_gt_labels(label_path):
    """读取 YOLO GT label → [(cx,cy,w,h), ...]"""
    if not os.path.isfile(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                boxes.append(tuple(float(x) for x in parts[1:5]))
    return boxes


def evaluate_detection(yolo_preds, gt_boxes, iou_thresh=0.3):
    """
    检测级评估: 匹配 YOLO preds 和 GT。

    返回:
        TP, FP, FN 计数
        matched_gt: set of matched GT indices
        gt_to_pred: {gt_idx: pred_idx} 映射
        tp_indices: set of pred indices that matched a GT
        fp_indices: set of unmatched pred indices
    """
    if len(yolo_preds) == 0:
        return {"TP": 0, "FP": 0, "FN": len(gt_boxes),
                "matched_gt": set(), "gt_to_pred": {},
                "tp_indices": set(), "fp_indices": set()}

    n_pred = len(yolo_preds)
    n_gt = len(gt_boxes)
    iou_mat = np.zeros((n_pred, n_gt))
    for i, yp in enumerate(yolo_preds):
        px1, py1, px2, py2 = yolo_to_xyxy(yp[0], yp[1], yp[2], yp[3], 512, 512)
        for j, gb in enumerate(gt_boxes):
            gx1, gy1, gx2, gy2 = yolo_to_xyxy(gb[0], gb[1], gb[2], gb[3], 512, 512)
            iou_mat[i, j] = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))

    # Greedy matching: each GT → best IoU pred
    gt_to_pred = {}  # gt_idx → pred_idx
    for j in range(n_gt):
        best_iou = 0; best_i = -1
        for i in range(n_pred):
            if iou_mat[i, j] > best_iou:
                best_iou = iou_mat[i, j]; best_i = i
        if best_iou >= iou_thresh:
            gt_to_pred[j] = best_i

    tp_indices = set(gt_to_pred.values())
    fp_indices = set(range(n_pred)) - tp_indices
    matched_gt = set(gt_to_pred.keys())
    fn_count = n_gt - len(matched_gt)

    return {
        "TP": len(tp_indices), "FP": len(fp_indices), "FN": fn_count,
        "matched_gt": matched_gt, "gt_to_pred": gt_to_pred,
        "tp_indices": tp_indices, "fp_indices": fp_indices,
    }


def main():
    args = parse_args()
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Load YOLO ──
    print("=" * 60)
    print("  Loading YOLO...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    print(f"  {args.yolo_weights}")

    # ── 2. Load ResNet18 ──
    print("\n  Loading ResNet18...")
    from models.fp_resnet18 import FPResNet18
    ckpt = torch.load(args.resnet_model, map_location=device)
    resnet = FPResNet18(pretrained=False).to(device)
    resnet.load_state_dict(ckpt["model_state_dict"])
    resnet.eval()
    print(f"  {args.resnet_model} (epoch {ckpt['epoch']})")

    # ── 3. Scan val images ──
    print("\n  Scanning val images...")
    val_img_dir = os.path.join(args.yolo_dataset, "images", "val")
    val_lbl_dir = os.path.join(args.yolo_dataset, "labels", "val")
    img_files = sorted([f for f in os.listdir(val_img_dir) if f.endswith(".png")])

    if args.limit_images:
        img_files = img_files[:args.limit_images]
    print(f"  {len(img_files)} val images")

    # ── 4. Load uid_mapping for CT volume lookup ──
    import json as _json
    uid_map_path = os.path.join(args.yolo_dataset, "uid_mapping.json")
    with open(uid_map_path) as f:
        uid_mapping = _json.load(f)

    # Build mhd index
    import SimpleITK as sitk
    mhd_map = {}
    for entry in os.listdir(args.data_dir):
        sp = os.path.join(args.data_dir, entry)
        if not os.path.isdir(sp) or not entry.startswith("subset"):
            continue
        for root, dirs, files in os.walk(sp):
            for fn in files:
                if fn.endswith(".mhd"):
                    mhd_map[fn.replace(".mhd", "")] = os.path.join(root, fn)

    # ── 5. Run YOLO on all val images, collect preds ──
    print("\n  Running YOLO inference on val set...")
    all_results = []  # [{img, gt_count, yolo_count, eval_result, yolo_preds, ...}]
    t0 = time.time()

    for img_file in img_files:
        img_path = os.path.join(val_img_dir, img_file)
        lbl_path = os.path.join(val_lbl_dir, img_file.replace(".png", ".txt"))
        stem = img_file.replace(".png", "")

        # GT
        gt_boxes = read_gt_labels(lbl_path)

        # YOLO
        results = yolo.predict(img_path, conf=args.conf_threshold, device=device,
                               imgsz=512, verbose=False)
        yolo_preds = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                iw, ih = results[0].orig_shape[1], results[0].orig_shape[0]
                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j]
                    cx = (x1 + x2) / 2 / iw
                    cy = (y1 + y2) / 2 / ih
                    w = (x2 - x1) / iw
                    h = (y2 - y1) / ih
                    yolo_preds.append((cx, cy, w, h, float(confs[j])))

        eval_res = evaluate_detection(yolo_preds, gt_boxes, args.iou_threshold)

        # Full UID from mapping
        full_uid = uid_mapping.get(stem, "unknown")
        if full_uid in mhd_map:
            mhd_path = mhd_map[full_uid]
            try:
                vol_info = sitk.ReadImage(mhd_path)
                vol_arr = sitk.GetArrayFromImage(vol_info)
                orig_shape = vol_arr.shape
            except Exception:
                orig_shape = (0, 512, 512)
        else:
            mhd_path = None
            orig_shape = (0, 512, 512)

        all_results.append({
            "img_file": img_file,
            "gt_count": len(gt_boxes),
            "yolo_count": len(yolo_preds),
            "yolo_preds": yolo_preds,
            "gt_boxes": gt_boxes,
            "eval": eval_res,
            "full_uid": full_uid,
            "mhd_path": mhd_path,
            "orig_shape": orig_shape,
        })

    yolo_time = time.time() - t0

    # ── 6. YOLO-only metrics ──
    total_gt = sum(r["gt_count"] for r in all_results)
    yolo_tp = sum(r["eval"]["TP"] for r in all_results)
    yolo_fp = sum(r["eval"]["FP"] for r in all_results)
    yolo_fn = sum(r["eval"]["FN"] for r in all_results)
    yolo_precision = yolo_tp / (yolo_tp + yolo_fp) if (yolo_tp + yolo_fp) else 0
    yolo_recall = yolo_tp / (yolo_tp + yolo_fn) if (yolo_tp + yolo_fn) else 0
    yolo_f1 = 2 * yolo_precision * yolo_recall / (yolo_precision + yolo_recall) if (yolo_precision + yolo_recall) else 0

    yolo_only = {
        "stage": "YOLO only",
        "total_gt": total_gt,
        "yolo_predictions": yolo_tp + yolo_fp,
        "TP": yolo_tp, "FP": yolo_fp, "FN": yolo_fn,
        "precision": round(yolo_precision, 4),
        "recall": round(yolo_recall, 4),
        "f1": round(yolo_f1, 4),
    }

    # ── 7. Run ResNet18 on all YOLO preds (once) ──
    print("\n  Running ResNet18 on all YOLO candidates...")
    from training.ct_patch_utils import extract_25d_patch, clear_volume_cache

    # Collect all YOLO preds with their patches and GT matching info
    all_yolo_items = []  # [{img_idx, pred_idx, cx,cy,w,h, conf, is_tp, matched_gt_idx, resnet_score}]
    for img_idx, r in enumerate(all_results):
        tp_set = r["eval"]["tp_indices"]
        gt_to_pred = r["eval"]["gt_to_pred"]  # {gt_idx: pred_idx}
        # Reverse: pred_idx → gt_idx
        pred_to_gt = {v: k for k, v in gt_to_pred.items()}
        for pi, (cx, cy, w, h, conf) in enumerate(r["yolo_preds"]):
            is_tp = pi in tp_set
            matched_gt_idx = pred_to_gt.get(pi) if is_tp else None
            all_yolo_items.append({
                "img_idx": img_idx,
                "pred_idx": pi,
                "cx": cx, "cy": cy, "w": w, "h": h,
                "conf": conf,
                "is_tp": is_tp,
                "matched_gt_idx": matched_gt_idx,
                "resnet_score": None,
                "patch_ok": False,
            })

    # Batch inference
    batch_size = 64
    for start in range(0, len(all_yolo_items), batch_size):
        end = min(start + batch_size, len(all_yolo_items))
        batch_items = all_yolo_items[start:end]
        batch_tensors = []

        for item in batch_items:
            r = all_results[item["img_idx"]]
            if r["mhd_path"] is None:
                item["resnet_score"] = 0.0
                continue

            stem = r["img_file"].replace(".png", "")
            parts = stem.rsplit("_", 2)
            slice_z = int(parts[1]) if len(parts) >= 3 else 0

            # Convert YOLO normalized bbox → original CT voxel coords
            N, orig_H, orig_W = r["orig_shape"]
            px1, py1, px2, py2 = yolo_to_xyxy(item["cx"], item["cy"], item["w"], item["h"], 512, 512)
            sw = orig_W / 512.0
            sh = orig_H / 512.0
            ox1 = px1 * sw
            oy1 = py1 * sh
            ox2 = px2 * sw
            oy2 = py2 * sh
            # expand 1.5x
            cx_o = (ox1 + ox2) / 2
            cy_o = (oy1 + oy2) / 2
            bw_o = (ox2 - ox1) * 1.5
            bh_o = (oy2 - oy1) * 1.5
            ex1 = max(0, int(cx_o - bw_o / 2))
            ey1 = max(0, int(cy_o - bh_o / 2))
            ex2 = min(orig_W - 1, int(cx_o + bw_o / 2))
            ey2 = min(orig_H - 1, int(cy_o + bh_o / 2))

            patch = extract_25d_patch(r["mhd_path"], slice_z, ex1, ey1, ex2, ey2, patch_size=96)
            if patch is None:
                item["resnet_score"] = 0.0
                continue

            batch_tensors.append(patch)
            item["_batch_idx"] = len(batch_tensors) - 1
            item["patch_ok"] = True

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                scores = resnet.predict(batch).cpu().numpy()
            for item in batch_items:
                if "_batch_idx" in item:
                    item["resnet_score"] = float(scores[item.pop("_batch_idx")])

    clear_volume_cache()

    # ── 8. Two-stage metrics at different thresholds ──
    print("\n  Computing two-stage metrics...")

    def two_stage_metrics(items, resnet_thresh):
        """Filter items by resnet_score >= resnet_thresh, compute correct per-image FN"""
        # Per image: keep preds that pass ResNet18
        per_img_kept_preds = defaultdict(set)       # img_idx → set of kept pred_idx
        per_img_kept_tp_preds = defaultdict(set)     # img_idx → set of kept pred_idx that are TP

        for item in items:
            idx = item["img_idx"]
            keep = item["resnet_score"] is not None and item["resnet_score"] >= resnet_thresh
            if not keep:
                continue
            per_img_kept_preds[idx].add(item["pred_idx"])
            if item["is_tp"]:
                per_img_kept_tp_preds[idx].add(item["pred_idx"])

        # Per image: count TP, FP, FN
        tp = 0; fp = 0; fn = 0
        for img_idx, r in enumerate(all_results):
            gt_to_pred = r["eval"]["gt_to_pred"]     # {gt_idx: pred_idx} from YOLO
            kept_tp = per_img_kept_tp_preds[img_idx]  # pred_idx of kept TPs

            # TP: YOLO TPs that passed ResNet18 filter
            img_tp = 0
            matched_gts = set()
            for gt_idx, pred_idx in gt_to_pred.items():
                if pred_idx in kept_tp:
                    img_tp += 1
                    matched_gts.add(gt_idx)
            tp += img_tp

            # FN: GT nodules NOT matched after filtering
            img_fn = r["gt_count"] - len(matched_gts)
            fn += img_fn

            # FP: kept preds that are NOT TP
            total_kept = per_img_kept_preds[img_idx]
            fp_preds = total_kept - per_img_kept_tp_preds[img_idx]
            fp += len(fp_preds)

        fp_removed = yolo_fp - fp
        fp_reduction_rate = fp_removed / yolo_fp if yolo_fp else 0
        tp_retained = tp
        tp_retention_rate = tp_retained / yolo_tp if yolo_tp else 0

        final_recall = tp / total_gt if total_gt else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * precision * final_recall / (precision + final_recall) if (precision + final_recall) else 0

        return {
            "resnet_threshold": resnet_thresh,
            "TP": tp, "FP": fp, "FN": fn,
            "precision": round(precision, 4),
            "final_recall": round(final_recall, 4),
            "f1": round(f1, 4),
            "fp_reduction_rate": round(fp_reduction_rate, 4),
            "tp_retention_rate": round(tp_retention_rate, 4),
            "fp_removed": fp_removed,
            "tp_lost": yolo_tp - tp_retained,
        }

    thresholds = [0.50, 0.70, 0.85]
    two_stage_results = [two_stage_metrics(all_yolo_items, t) for t in thresholds]

    # ── 9. Report ──
    print(f"\n{'='*60}")
    print(f"  TWO-STAGE EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Val images: {len(img_files)}")
    print(f"  Total GT nodules: {total_gt}")
    print(f"  YOLO candidates: {yolo_tp + yolo_fp}")
    print(f"  YOLO inference time: {yolo_time:.1f}s")

    print(f"\n  {'Stage':<30} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7} {'FP-cut':>7} {'TP-kept':>8}")
    print(f"  {'-'*88}")
    # YOLO only: R = yolo_recall (TP / total_gt)
    print(f"  {'YOLO only':<30} {yolo_only['TP']:>5} {yolo_only['FP']:>5} {yolo_only['FN']:>5} "
          f"{yolo_only['precision']:>7.4f} {yolo_only['recall']:>7.4f} {yolo_only['f1']:>7.4f} "
          f"{'-':>7} {'100.00%':>8}")
    for stage in two_stage_results:
        name = f"YOLO+ResNet th={stage['resnet_threshold']:.2f}"
        fp_cut = stage['fp_removed']
        tp_keep = stage['tp_retention_rate']
        print(f"  {name:<30} {stage['TP']:>5} {stage['FP']:>5} {stage['FN']:>5} "
              f"{stage['precision']:>7.4f} {stage['final_recall']:>7.4f} {stage['f1']:>7.4f} "
              f"{fp_cut:>7d} {tp_keep:>7.2%}")

    print(f"\n  --- FP Reduction vs TP Retention ---")
    for stage in two_stage_results:
        print(f"  th={stage['resnet_threshold']:.2f}: "
              f"FP cut {stage['fp_reduction_rate']:.1%} ({stage['fp_removed']}/{yolo_fp}), "
              f"TP kept {stage['tp_retention_rate']:.1%} ({stage['TP']}/{yolo_tp}), "
              f"FN {yolo_only['FN']}→{stage['FN']}")

    # ── 10. Save ──
    report = {
        "yolo_only": yolo_only,
        "two_stage": {f"th_{s['resnet_threshold']:.2f}": s for s in two_stage_results},
        "config": {
            "yolo_conf_threshold": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "val_images": len(img_files),
            "total_gt": total_gt,
        },
    }
    with open(os.path.join(args.out_dir, "two_stage_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {args.out_dir}/two_stage_report.json")


if __name__ == "__main__":
    main()
