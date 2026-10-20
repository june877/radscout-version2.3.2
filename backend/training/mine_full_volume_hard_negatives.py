"""Mine new 2.5D hard negatives from complete LUNA16 training volumes.

The YOLO detector is fixed. For each training CT volume, this script:
  1. builds R=z-1, G=z, B=z+1 lung-window PNG inputs;
  2. runs YOLO with imgsz=512;
  3. rejects predictions near annotated nodules;
  4. crops [3, 96, 96] patches from the same RGB input;
  5. scores patches with the current ResNet18;
  6. saves high-value false positives for a later training round.

Only train seriesuids reconstructed from the existing YOLO split are used.
Validation volumes are never mined.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fp_resnet18 import load_fp_resnet18_checkpoint
from training.bbox_utils import clamp_bbox, compute_iou, expand_bbox


LUNG_MIN = -1000.0
LUNG_MAX = 400.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine complete-volume hard negatives for RadScout v3.2"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--yolo_weights", required=True)
    parser.add_argument("--resnet_weights", required=True)
    parser.add_argument("--yolo_dataset", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--subsets", type=int, nargs="+", default=list(range(7)))
    parser.add_argument("--yolo_conf", type=float, default=0.10)
    parser.add_argument("--neg_iou", type=float, default=0.10)
    parser.add_argument("--pos_iou", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--expand_ratio", type=float, default=1.5)
    parser.add_argument("--min_box_size", type=int, default=6)
    parser.add_argument("--max_per_slice", type=int, default=5)
    parser.add_argument("--max_per_volume", type=int, default=100)
    parser.add_argument("--slice_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug_count", type=int, default=100)
    parser.add_argument("--limit_volumes", type=int, default=None)
    parser.add_argument(
        "--body_fraction_min",
        type=float,
        default=0.02,
        help="Skip near-empty superior/inferior slices",
    )
    parser.add_argument(
        "--nodule_safety_slices",
        type=int,
        default=1,
        help="Additional z margin around the physical nodule radius",
    )
    return parser.parse_args()


def hu_to_uint8(slice_hu):
    clipped = np.clip(slice_hu, LUNG_MIN, LUNG_MAX)
    return ((clipped - LUNG_MIN) / (LUNG_MAX - LUNG_MIN) * 255).astype(
        np.uint8
    )


def safe_slice(volume, z):
    z = max(0, min(int(z), volume.shape[0] - 1))
    return volume[z]


def make_rgb_25d(volume, z, image_size):
    rgb = np.stack(
        [
            hu_to_uint8(safe_slice(volume, z - 1)),
            hu_to_uint8(safe_slice(volume, z)),
            hu_to_uint8(safe_slice(volume, z + 1)),
        ],
        axis=-1,
    )
    source_height, source_width = rgb.shape[:2]
    if (source_width, source_height) != (image_size, image_size):
        rgb = np.asarray(
            Image.fromarray(rgb, mode="RGB").resize(
                (image_size, image_size), Image.BILINEAR
            )
        )
    return rgb, image_size / source_width, image_size / source_height


def scan_mhd_files(data_dir, subset_ids):
    mapping = {}
    used_subsets = []
    for subset_id in subset_ids:
        subset_dir = Path(data_dir) / f"subset{subset_id}"
        nested_dir = subset_dir / f"subset{subset_id}"
        actual_dir = nested_dir if nested_dir.is_dir() else subset_dir
        if not actual_dir.is_dir():
            print(f"[WARN] subset{subset_id} not found, skipping")
            continue
        files = list(actual_dir.glob("*.mhd"))
        if not files:
            print(f"[WARN] subset{subset_id} has no .mhd files, skipping")
            continue
        used_subsets.append(subset_id)
        for path in files:
            mapping[path.stem] = str(path.resolve())
    return mapping, used_subsets


def reconstruct_split_uids(yolo_dataset, annotations, mhd_map):
    """Reproduce the exact split used by build_yolo25d_dataset.py.

    The historical uid_mapping.json cannot be authoritative because image
    names use a truncated UID and train/val counters restart independently,
    allowing mapping-key collisions. The original split itself is
    deterministic, so we reconstruct it from sorted matched UIDs.
    """
    dataset_dir = Path(yolo_dataset)
    matched_uids = sorted(set(annotations["seriesuid"]) & set(mhd_map))
    train_list, val_list = train_test_split(
        matched_uids,
        test_size=0.2,
        random_state=42,
    )
    train_uids = set(train_list)
    val_uids = set(val_list)
    overlap = train_uids & val_uids
    if overlap:
        raise RuntimeError(
            f"Train/val leakage detected: {len(overlap)} overlapping seriesuids"
        )

    summary_path = dataset_dir / "split_summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        expected_train = int(summary.get("train_volumes", -1))
        expected_val = int(summary.get("val_volumes", -1))
        if expected_train != len(train_uids) or expected_val != len(val_uids):
            raise RuntimeError(
                "Reconstructed split count does not match split_summary.json: "
                f"reconstructed={len(train_uids)}/{len(val_uids)}, "
                f"expected={expected_train}/{expected_val}"
            )

    mapping_path = dataset_dir / "uid_mapping.json"
    if mapping_path.is_file():
        with mapping_path.open("r", encoding="utf-8") as handle:
            uid_mapping = json.load(handle)
        mapped_by_split = {}
        for split in ("train", "val"):
            image_dir = dataset_dir / "images" / split
            mapped_by_split[split] = {
                uid_mapping[image_path.stem]
                for image_path in image_dir.glob("*.png")
                if image_path.stem in uid_mapping
            }
        mapping_overlap = mapped_by_split["train"] & mapped_by_split["val"]
        if mapping_overlap:
            print(
                f"[WARN] Historical uid_mapping.json reports "
                f"{len(mapping_overlap)} overlapping UIDs due to truncated "
                "filename-key collisions. It is ignored for full-volume mining."
            )
    return train_uids, val_uids


def build_nodule_metadata(image, annotations, image_size, safety_margin):
    spacing = image.GetSpacing()
    source_size = image.GetSize()
    scale_x = image_size / source_size[0]
    scale_y = image_size / source_size[1]
    nodules = []

    for _, row in annotations.iterrows():
        physical_point = (
            float(row["coordX"]),
            float(row["coordY"]),
            float(row["coordZ"]),
        )
        continuous_index = image.TransformPhysicalPointToContinuousIndex(
            physical_point
        )
        vx, vy, vz = continuous_index
        diameter_mm = float(row["diameter_mm"])
        width_px = diameter_mm / spacing[0] * scale_x
        height_px = diameter_mm / spacing[1] * scale_y
        cx = vx * scale_x
        cy = vy * scale_y
        z_radius = max(
            1,
            int(math.ceil((diameter_mm / 2.0) / spacing[2])) + safety_margin,
        )
        nodules.append(
            {
                "cx": cx,
                "cy": cy,
                "z": int(round(vz)),
                "z_radius": z_radius,
                "bbox": (
                    cx - width_px / 2,
                    cy - height_px / 2,
                    cx + width_px / 2,
                    cy + height_px / 2,
                ),
                "diameter_mm": diameter_mm,
            }
        )
    return nodules


def active_gt_boxes(nodules, z):
    return [
        nodule["bbox"]
        for nodule in nodules
        if abs(z - nodule["z"]) <= nodule["z_radius"]
    ]


def nms_xyxy(candidates, iou_threshold=0.5):
    ordered = sorted(candidates, key=lambda item: item["yolo_score"], reverse=True)
    kept = []
    for candidate in ordered:
        if all(
            compute_iou(candidate["bbox"], existing["bbox"]) < iou_threshold
            for existing in kept
        ):
            kept.append(candidate)
    return kept


def crop_patch(rgb, bbox, expand_ratio, patch_size):
    height, width = rgb.shape[:2]
    expanded = expand_bbox(*bbox, expand_ratio)
    expanded = clamp_bbox(*expanded, width, height, min_size=16)
    ex1, ey1, ex2, ey2 = expanded
    if ex2 <= ex1 + 8 or ey2 <= ey1 + 8:
        return None, None
    crop = Image.fromarray(rgb, mode="RGB").crop(expanded).resize(
        (patch_size, patch_size), Image.BILINEAR
    )
    array = np.asarray(crop, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    return tensor, expanded


def informative_slice(slice_hu, minimum_fraction):
    # Air outside the body is close to -1000 HU. This cheaply excludes
    # near-empty slices while retaining lung apices and bases.
    return float(np.mean(slice_hu > -950)) >= minimum_fraction


def score_resnet(model, patch_tensors, device):
    if not patch_tensors:
        return []
    batch = torch.stack(patch_tensors).to(device, non_blocking=True)
    with torch.no_grad():
        return model.predict(batch).cpu().tolist()


def save_debug_panel(path, volume, z, candidate, image_size):
    rgb, _, _ = make_rgb_25d(volume, z, image_size)
    channel_titles = ("z-1", f"z={z}", "z+1")
    panels = []
    for channel_index, title in enumerate(channel_titles):
        image = Image.fromarray(rgb[:, :, channel_index], mode="L").convert("RGB")
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = candidate["bbox"]
        ex1, ey1, ex2, ey2 = candidate["expanded_bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        draw.rectangle((ex1, ey1, ex2, ey2), outline=(255, 215, 0), width=2)
        draw.rectangle((0, 0, image_size, 24), fill=(0, 0, 0))
        draw.text((6, 5), title, fill=(255, 255, 255))
        panels.append(image)

    canvas = Image.new("RGB", (image_size * 3, image_size + 34), "black")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * image_size, 34))
    title = (
        f"YOLO={candidate['yolo_score']:.3f} "
        f"ResNet={candidate['resnet_score']:.3f} "
        f"Hardness={candidate['hardness_score']:.3f} "
        f"IoU={candidate['max_iou']:.3f}"
    )
    ImageDraw.Draw(canvas).text((8, 9), title, fill=(255, 255, 255))
    canvas.save(path)


def process_volume(
    uid,
    mhd_path,
    annotations,
    yolo,
    resnet,
    device,
    args,
):
    image = sitk.ReadImage(mhd_path)
    volume = sitk.GetArrayFromImage(image)
    nodules = build_nodule_metadata(
        image, annotations, args.imgsz, args.nodule_safety_slices
    )

    stats = defaultdict(int)
    volume_candidates = []
    pending_images = []
    pending_metadata = []

    def flush_batch():
        if not pending_images:
            return
        results = yolo.predict(
            pending_images,
            conf=args.yolo_conf,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch_size,
            verbose=False,
        )
        for rgb_pil, metadata, result in zip(
            pending_images, pending_metadata, results
        ):
            z = metadata["z"]
            rgb = np.asarray(rgb_pil)
            gt_boxes = active_gt_boxes(nodules, z)
            predictions = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = (float(value) for value in box)
                    if x2 - x1 < args.min_box_size or y2 - y1 < args.min_box_size:
                        stats["small_box_skipped"] += 1
                        continue
                    max_iou = max(
                        (compute_iou((x1, y1, x2, y2), gt) for gt in gt_boxes),
                        default=0.0,
                    )
                    predictions.append(
                        {
                            "bbox": (x1, y1, x2, y2),
                            "yolo_score": float(score),
                            "max_iou": max_iou,
                        }
                    )

            predictions = nms_xyxy(predictions)
            safe_candidates = []
            for prediction in predictions:
                if prediction["max_iou"] >= args.pos_iou:
                    stats["positive_predictions_skipped"] += 1
                    continue
                if prediction["max_iou"] > args.neg_iou:
                    stats["ambiguous_iou_skipped"] += 1
                    continue
                if gt_boxes:
                    # Conservative safety rule: do not create negatives from
                    # any slice intersecting an annotated nodule z extent.
                    stats["near_nodule_skipped"] += 1
                    continue
                patch, expanded_bbox = crop_patch(
                    rgb,
                    prediction["bbox"],
                    args.expand_ratio,
                    args.patch_size,
                )
                if patch is None:
                    stats["invalid_patch_skipped"] += 1
                    continue
                prediction.update(
                    {
                        "z": z,
                        "patch_tensor": patch,
                        "expanded_bbox": expanded_bbox,
                    }
                )
                safe_candidates.append(prediction)

            safe_candidates.sort(
                key=lambda item: item["yolo_score"], reverse=True
            )
            safe_candidates = safe_candidates[: args.max_per_slice]
            scores = score_resnet(
                resnet,
                [item["patch_tensor"] for item in safe_candidates],
                device,
            )
            for candidate, score in zip(safe_candidates, scores):
                candidate["resnet_score"] = float(score)
                candidate["hardness_score"] = (
                    candidate["yolo_score"] * candidate["resnet_score"]
                )
                volume_candidates.append(candidate)
            stats["safe_candidates"] += len(safe_candidates)

        pending_images.clear()
        pending_metadata.clear()

    for z in range(0, volume.shape[0], args.slice_stride):
        stats["slices_seen"] += 1
        if not informative_slice(volume[z], args.body_fraction_min):
            stats["empty_slices_skipped"] += 1
            continue
        rgb, _, _ = make_rgb_25d(volume, z, args.imgsz)
        pending_images.append(Image.fromarray(rgb, mode="RGB"))
        pending_metadata.append({"z": z})
        stats["slices_inferred"] += 1
        if len(pending_images) >= args.batch_size:
            flush_batch()
    flush_batch()

    volume_candidates.sort(
        key=lambda item: item["hardness_score"], reverse=True
    )
    return (
        image,
        volume,
        volume_candidates[: args.max_per_volume],
        dict(stats),
    )


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested CUDA, but torch.cuda.is_available() is False. "
            "Check D:/Python/python.exe before mining."
        )
    device = torch.device(args.device)
    data_dir = os.path.abspath(args.data_dir)
    yolo_dataset = os.path.abspath(args.yolo_dataset)
    out_dir = os.path.abspath(args.out_dir)
    patch_dir = os.path.join(out_dir, "patches", "train")
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    annotations_path = os.path.join(data_dir, "annotations.csv")
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(f"annotations.csv not found: {annotations_path}")
    annotations = pd.read_csv(annotations_path)

    mhd_map, used_subsets = scan_mhd_files(data_dir, args.subsets)
    train_uids, val_uids = reconstruct_split_uids(
        yolo_dataset, annotations, mhd_map
    )
    missing_train_uids = sorted(train_uids - set(mhd_map))
    available_train_uids = sorted(train_uids & set(mhd_map))
    if args.limit_volumes is not None:
        available_train_uids = available_train_uids[: args.limit_volumes]

    if set(available_train_uids) & val_uids:
        raise RuntimeError("Validation UID entered the mining list")
    if not available_train_uids:
        raise RuntimeError("No train volumes are available for mining")

    print(f"Using device: {device}")
    print(f"Train UIDs: {len(train_uids)}, Val UIDs: {len(val_uids)}, overlap=0")
    print(
        f"Mining volumes: {len(available_train_uids)} "
        f"(missing train volumes={len(missing_train_uids)})"
    )

    from ultralytics import YOLO

    yolo = YOLO(os.path.abspath(args.yolo_weights))
    resnet, resnet_checkpoint = load_fp_resnet18_checkpoint(
        os.path.abspath(args.resnet_weights), device
    )
    print(
        f"ResNet epoch={resnet_checkpoint.get('epoch')} "
        f"normalize_input={resnet.normalize_input}"
    )

    started = time.time()
    rows = []
    global_stats = defaultdict(int)
    debug_saved = 0
    debug_per_volume = max(
        1, math.ceil(args.debug_count / len(available_train_uids))
    )

    fields = [
        "sample_id",
        "split",
        "seriesuid",
        "subset",
        "slice_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "expanded_x1",
        "expanded_y1",
        "expanded_x2",
        "expanded_y2",
        "yolo_score",
        "resnet_score",
        "hardness_score",
        "max_iou",
        "label",
        "patch_path",
        "source_mhd",
    ]

    for uid in tqdm(available_train_uids, desc="Mining train volumes"):
        uid_annotations = annotations[annotations["seriesuid"] == uid]
        _, volume, candidates, stats = process_volume(
            uid,
            mhd_map[uid],
            uid_annotations,
            yolo,
            resnet,
            device,
            args,
        )
        for key, value in stats.items():
            global_stats[key] += value
        global_stats["processed_volumes"] += 1

        subset_name = Path(mhd_map[uid]).parent.name
        for candidate in candidates:
            sample_id = len(rows)
            patch_path = os.path.join(patch_dir, f"hn_{sample_id:08d}.pt")
            patch = candidate.pop("patch_tensor")
            if list(patch.shape) != [3, args.patch_size, args.patch_size]:
                raise RuntimeError(f"Unexpected patch shape: {list(patch.shape)}")
            if float(patch.min()) < -1e-4 or float(patch.max()) > 1.0001:
                raise RuntimeError("Mined patch value is outside [0,1]")
            torch.save(patch, patch_path)

            x1, y1, x2, y2 = candidate["bbox"]
            ex1, ey1, ex2, ey2 = candidate["expanded_bbox"]
            rows.append(
                {
                    "sample_id": f"hn_{sample_id:08d}",
                    "split": "train",
                    "seriesuid": uid,
                    "subset": subset_name,
                    "slice_index": candidate["z"],
                    "x1": round(x1, 3),
                    "y1": round(y1, 3),
                    "x2": round(x2, 3),
                    "y2": round(y2, 3),
                    "expanded_x1": ex1,
                    "expanded_y1": ey1,
                    "expanded_x2": ex2,
                    "expanded_y2": ey2,
                    "yolo_score": round(candidate["yolo_score"], 6),
                    "resnet_score": round(candidate["resnet_score"], 6),
                    "hardness_score": round(candidate["hardness_score"], 6),
                    "max_iou": round(candidate["max_iou"], 6),
                    "label": 0,
                    "patch_path": patch_path,
                    "source_mhd": mhd_map[uid],
                }
            )

        for candidate in candidates[:debug_per_volume]:
            if debug_saved >= args.debug_count:
                break
            debug_path = os.path.join(
                debug_dir,
                f"debug_{debug_saved:04d}_{uid[:12]}_z{candidate['z']:04d}.png",
            )
            save_debug_panel(debug_path, volume, candidate["z"], candidate, args.imgsz)
            debug_saved += 1

    csv_path = os.path.join(out_dir, "hard_negatives.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    unique_uids = {row["seriesuid"] for row in rows}
    validation_overlap = unique_uids & val_uids
    if validation_overlap:
        raise RuntimeError(
            f"Mined output contains {len(validation_overlap)} validation UIDs"
        )

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 2),
        "config": vars(args),
        "used_subsets": used_subsets,
        "train_uid_count": len(train_uids),
        "val_uid_count": len(val_uids),
        "processed_volume_count": len(available_train_uids),
        "missing_train_uids": missing_train_uids,
        "mined_patch_count": len(rows),
        "mined_unique_uid_count": len(unique_uids),
        "train_val_overlap": len(validation_overlap),
        "debug_image_count": debug_saved,
        "stats": dict(global_stats),
        "hardness": {
            "minimum": min((row["hardness_score"] for row in rows), default=None),
            "mean": (
                sum(row["hardness_score"] for row in rows) / len(rows)
                if rows
                else None
            ),
            "maximum": max((row["hardness_score"] for row in rows), default=None),
        },
        "safety_note": (
            "Predictions on slices within each annotated nodule's physical z "
            "radius plus a safety margin were excluded from negative mining."
        ),
    }
    summary_path = os.path.join(out_dir, "mining_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\nMining complete")
    print(f"  Patches: {len(rows)}")
    print(f"  Debug images: {debug_saved}")
    print(f"  Train/val overlap: {len(validation_overlap)}")
    print(f"  CSV: {csv_path}")
    print(f"  Summary: {summary_path}")
    print("  STOP: visually inspect debug images before merging or retraining.")


if __name__ == "__main__":
    main()
