"""Merge newly mined full-volume hard negatives into the v3.2 manifest.

The original validation rows are kept unchanged. New samples must all be
training negatives, because validation data must remain fixed for a fair
comparison with the previous v3.2 model.
"""

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def count_rows(rows):
    counts = Counter((row["split"], int(row["label"])) for row in rows)
    return {
        "train_positive": counts[("train", 1)],
        "train_negative": counts[("train", 0)],
        "val_positive": counts[("val", 1)],
        "val_negative": counts[("val", 0)],
        "total": len(rows),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge full-volume hard negatives into the v3.2 manifest"
    )
    parser.add_argument("--base_csv", required=True)
    parser.add_argument("--hard_negative_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_summary", required=True)
    parser.add_argument("--hard_negative_boost", type=float, default=3.0)
    parser.add_argument("--yolo_score_boost", type=float, default=0.5)
    parser.add_argument(
        "--min_resnet_score",
        type=float,
        default=0.0,
        help="Only append newly mined negatives at or above this score.",
    )
    parser.add_argument(
        "--max_new_samples",
        type=int,
        default=None,
        help="Optional cap after sorting new negatives by hardness_score.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_csv = os.path.abspath(args.base_csv)
    hard_csv = os.path.abspath(args.hard_negative_csv)
    out_csv = os.path.abspath(args.out_csv)
    out_summary = os.path.abspath(args.out_summary)

    base_fields, base_rows = read_csv(base_csv)
    hard_fields, hard_rows = read_csv(hard_csv)
    if not base_rows:
        raise RuntimeError(f"Base manifest is empty: {base_csv}")
    if not hard_rows:
        raise RuntimeError(f"Hard-negative manifest is empty: {hard_csv}")

    base_sample_ids = {row["sample_id"] for row in base_rows}
    base_patch_paths = {
        os.path.normcase(os.path.abspath(row["patch_path"])) for row in base_rows
    }
    eligible_hard_rows = [
        row
        for row in hard_rows
        if float(row.get("resnet_score") or 0.0) >= args.min_resnet_score
    ]
    eligible_hard_rows.sort(
        key=lambda row: float(row.get("hardness_score") or 0.0),
        reverse=True,
    )
    if args.max_new_samples is not None:
        eligible_hard_rows = eligible_hard_rows[: args.max_new_samples]

    new_rows = []
    new_sample_ids = set()
    new_patch_paths = set()

    for row in eligible_hard_rows:
        if row.get("split") != "train" or int(row.get("label", -1)) != 0:
            raise RuntimeError(
                "New hard negatives must all have split=train and label=0: "
                f"{row.get('sample_id')}"
            )

        sample_id = row["sample_id"]
        patch_path = os.path.normcase(os.path.abspath(row["patch_path"]))
        if sample_id in base_sample_ids or sample_id in new_sample_ids:
            raise RuntimeError(f"Duplicate sample_id: {sample_id}")
        if patch_path in base_patch_paths or patch_path in new_patch_paths:
            raise RuntimeError(f"Duplicate patch_path: {row['patch_path']}")
        if not os.path.isfile(row["patch_path"]):
            raise FileNotFoundError(f"Patch missing: {row['patch_path']}")

        resnet_score = float(row.get("resnet_score") or 0.0)
        yolo_score = float(row.get("yolo_score") or 0.0)
        row["hard_negative_score"] = f"{resnet_score:.6f}"
        row["sample_weight"] = (
            f"{1.0 + args.hard_negative_boost * resnet_score + args.yolo_score_boost * yolo_score:.6f}"
        )
        row["source_type"] = "full_volume_hard_negative"
        new_rows.append(row)
        new_sample_ids.add(sample_id)
        new_patch_paths.add(patch_path)

    for row in base_rows:
        row["source_type"] = row.get("source_type") or "original_candidate"

    fieldnames = list(base_fields)
    for field in hard_fields + [
        "hard_negative_score",
        "sample_weight",
        "source_type",
    ]:
        if field not in fieldnames:
            fieldnames.append(field)

    merged_rows = base_rows + new_rows
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    base_counts = count_rows(base_rows)
    merged_counts = count_rows(merged_rows)
    new_weights = [float(row["sample_weight"]) for row in new_rows]
    unique_new_uids = {row.get("seriesuid", "") for row in new_rows}
    unique_new_uids.discard("")

    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_csv": base_csv,
        "hard_negative_csv": hard_csv,
        "output_csv": out_csv,
        "base_counts": base_counts,
        "new_hard_negative_count": len(new_rows),
        "available_hard_negative_count": len(hard_rows),
        "min_resnet_score": args.min_resnet_score,
        "max_new_samples": args.max_new_samples,
        "new_hard_negative_uid_count": len(unique_new_uids),
        "merged_counts": merged_counts,
        "validation_unchanged": (
            merged_counts["val_positive"] == base_counts["val_positive"]
            and merged_counts["val_negative"] == base_counts["val_negative"]
        ),
        "weight_formula": (
            "1 + hard_negative_boost * resnet_score "
            "+ yolo_score_boost * yolo_score"
        ),
        "hard_negative_boost": args.hard_negative_boost,
        "yolo_score_boost": args.yolo_score_boost,
        "new_sample_weight": {
            "minimum": min(new_weights),
            "mean": sum(new_weights) / len(new_weights),
            "maximum": max(new_weights),
        },
        "note": (
            "Only full-volume train negatives were appended. The validation "
            "set is byte-for-byte represented by the same original rows."
        ),
    }
    os.makedirs(os.path.dirname(out_summary), exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Base samples: {base_counts}")
    print(f"Added hard negatives: {len(new_rows)}")
    print(f"Merged samples: {merged_counts}")
    print(f"Validation unchanged: {summary['validation_unchanged']}")
    print(f"Output CSV: {out_csv}")
    print(f"Summary: {out_summary}")


if __name__ == "__main__":
    main()
