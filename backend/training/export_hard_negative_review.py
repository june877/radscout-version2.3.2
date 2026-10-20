"""Export a focused human-review set from mined full-volume negatives."""

import argparse
import csv
import os
import sys
from collections import Counter

import SimpleITK as sitk
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mine_full_volume_hard_negatives import save_debug_panel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export globally hardest mined negatives for human review"
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max_per_uid", type=int, default=2)
    parser.add_argument(
        "--sort_by",
        choices=("hardness_score", "resnet_score", "yolo_score"),
        default="hardness_score",
    )
    parser.add_argument("--imgsz", type=int, default=512)
    return parser.parse_args()


def save_contact_sheets(out_dir, manifest_rows, columns=2, rows_per_sheet=5):
    items_per_sheet = columns * rows_per_sheet
    thumb_width = 768
    thumb_height = 273
    caption_height = 26
    sheet_paths = []

    for sheet_index, start in enumerate(
        range(0, len(manifest_rows), items_per_sheet), start=1
    ):
        page_rows = manifest_rows[start : start + items_per_sheet]
        canvas = Image.new(
            "RGB",
            (
                columns * thumb_width,
                rows_per_sheet * (thumb_height + caption_height),
            ),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for item_index, row in enumerate(page_rows):
            source = Image.open(os.path.join(out_dir, row["review_image"])).convert("RGB")
            source.thumbnail((thumb_width, thumb_height))
            x = (item_index % columns) * thumb_width
            y = (item_index // columns) * (thumb_height + caption_height)
            canvas.paste(source, (x, y))
            draw.text(
                (x + 5, y + thumb_height + 5),
                row["review_image"],
                fill=(0, 0, 0),
            )
        sheet_path = os.path.join(out_dir, f"contact_sheet_{sheet_index:02d}.jpg")
        canvas.save(sheet_path, quality=92)
        sheet_paths.append(sheet_path)
    return sheet_paths


def main():
    args = parse_args()
    csv_path = os.path.abspath(args.csv)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found: {csv_path}")

    rows.sort(key=lambda row: float(row[args.sort_by]), reverse=True)
    selected = []
    uid_counts = Counter()
    for row in rows:
        uid = row["seriesuid"]
        if uid_counts[uid] >= args.max_per_uid:
            continue
        selected.append(row)
        uid_counts[uid] += 1
        if len(selected) >= args.count:
            break

    volume_cache = {}
    manifest_rows = []
    for index, row in enumerate(selected):
        mhd_path = row["source_mhd"]
        if mhd_path not in volume_cache:
            volume_cache[mhd_path] = sitk.GetArrayFromImage(sitk.ReadImage(mhd_path))
        volume = volume_cache[mhd_path]
        z = int(row["slice_index"])
        candidate = {
            "bbox": (
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            ),
            "expanded_bbox": (
                float(row["expanded_x1"]),
                float(row["expanded_y1"]),
                float(row["expanded_x2"]),
                float(row["expanded_y2"]),
            ),
            "yolo_score": float(row["yolo_score"]),
            "resnet_score": float(row["resnet_score"]),
            "hardness_score": float(row["hardness_score"]),
            "max_iou": float(row["max_iou"]),
        }
        filename = (
            f"review_{index:03d}_{row['seriesuid'][:12]}_"
            f"z{z:04d}_h{candidate['hardness_score']:.3f}.png"
        )
        save_debug_panel(
            os.path.join(out_dir, filename),
            volume,
            z,
            candidate,
            args.imgsz,
        )
        output_row = dict(row)
        output_row["review_image"] = filename
        output_row["review_status"] = ""
        output_row["review_note"] = ""
        manifest_rows.append(output_row)

    manifest_path = os.path.join(out_dir, "review_manifest.csv")
    fieldnames = list(manifest_rows[0]) if manifest_rows else []
    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    contact_sheets = save_contact_sheets(out_dir, manifest_rows)

    print(f"Review images: {len(manifest_rows)}")
    print(f"Unique seriesuids: {len(uid_counts)}")
    print(f"Contact sheets: {len(contact_sheets)}")
    print(f"Sorted by: {args.sort_by}")
    print(f"Output directory: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
