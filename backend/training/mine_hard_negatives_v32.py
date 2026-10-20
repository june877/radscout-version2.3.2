"""Score existing negative candidates and create a v3.2 weighted manifest.

This is the first hard-negative mining pass. It does not change train/val
splits or create new patches. Train negatives that the current ResNet scores
as nodules receive larger sampling weights in the v3.2 training script.
"""

import argparse
import csv
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fp_resnet18 import FPResNet18


class NegativePatchDataset(Dataset):
    def __init__(self, rows, csv_dir):
        self.items = []
        for row_index, row in enumerate(rows):
            if row["split"] != "train" or int(row["label"]) != 0:
                continue
            stored_path = row["patch_path"]
            local_path = os.path.join(
                csv_dir, "patches", row["split"], os.path.basename(stored_path)
            )
            patch_path = local_path if os.path.isfile(local_path) else stored_path
            if not os.path.isfile(patch_path):
                raise FileNotFoundError(f"Patch missing: {stored_path}")
            self.items.append((row_index, patch_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row_index, patch_path = self.items[index]
        patch = torch.load(patch_path, map_location="cpu", weights_only=False).float()
        if list(patch.shape) != [3, 96, 96]:
            raise RuntimeError(f"Bad patch shape {list(patch.shape)}: {patch_path}")
        if float(patch.min()) < -1e-4 or float(patch.max()) > 1.0001:
            raise RuntimeError(f"Patch outside [0,1]: {patch_path}")
        return row_index, patch


def parse_args():
    parser = argparse.ArgumentParser(description="Mine v3.2 hard negatives")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--resnet_weights", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_summary", default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hard_negative_boost", type=float, default=3.0)
    parser.add_argument("--yolo_score_boost", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested CUDA device, but torch.cuda.is_available() is False. "
            "Check D:/Python/python.exe before continuing."
        )
    device = torch.device(args.device)
    csv_path = os.path.abspath(args.csv)
    csv_dir = os.path.dirname(csv_path)
    output_csv = os.path.abspath(args.out_csv)
    output_summary = os.path.abspath(
        args.out_summary
        or os.path.join(os.path.dirname(output_csv), "hard_negative_summary_v32.json")
    )
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No samples found in {csv_path}")

    checkpoint = torch.load(
        os.path.abspath(args.resnet_weights),
        map_location=device,
        weights_only=False,
    )
    model_config = checkpoint.get("model_config", {})
    model = FPResNet18(
        pretrained=False,
        dropout=float(model_config.get("dropout", 0.0)),
        normalize_input=bool(model_config.get("normalize_input", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = NegativePatchDataset(rows, csv_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    scores = {}
    with torch.no_grad():
        for row_indices, patches in loader:
            patches = patches.to(device, non_blocking=True)
            probabilities = model.predict(patches).cpu().tolist()
            for row_index, probability in zip(row_indices.tolist(), probabilities):
                scores[row_index] = float(probability)

    for extra_field in ("hard_negative_score", "sample_weight"):
        if extra_field not in fieldnames:
            fieldnames.append(extra_field)

    train_negative_scores = []
    for row_index, row in enumerate(rows):
        # Make the copied v3.2 dataset self-contained when the local patch exists.
        local_patch = os.path.join(
            csv_dir, "patches", row["split"], os.path.basename(row["patch_path"])
        )
        if os.path.isfile(local_patch):
            row["patch_path"] = local_patch

        if row["split"] == "train" and int(row["label"]) == 0:
            hard_score = scores[row_index]
            yolo_score = float(row.get("yolo_score") or 0.0)
            sample_weight = (
                1.0
                + args.hard_negative_boost * hard_score
                + args.yolo_score_boost * yolo_score
            )
            row["hard_negative_score"] = f"{hard_score:.6f}"
            row["sample_weight"] = f"{sample_weight:.6f}"
            train_negative_scores.append(hard_score)
        else:
            row["hard_negative_score"] = ""
            row["sample_weight"] = "1.000000"

    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sorted_scores = sorted(train_negative_scores, reverse=True)
    summary = {
        "source_csv": csv_path,
        "resnet_weights": os.path.abspath(args.resnet_weights),
        "output_csv": output_csv,
        "train_negative_count": len(sorted_scores),
        "hard_negative_count_score_ge_0_5": sum(score >= 0.5 for score in sorted_scores),
        "hard_negative_count_score_ge_0_7": sum(score >= 0.7 for score in sorted_scores),
        "score_min": min(sorted_scores) if sorted_scores else None,
        "score_mean": (
            sum(sorted_scores) / len(sorted_scores) if sorted_scores else None
        ),
        "score_max": max(sorted_scores) if sorted_scores else None,
        "top_20_scores": sorted_scores[:20],
        "weight_formula": (
            "1 + hard_negative_boost * resnet_score "
            "+ yolo_score_boost * yolo_score"
        ),
        "hard_negative_boost": args.hard_negative_boost,
        "yolo_score_boost": args.yolo_score_boost,
        "note": (
            "Only train negatives are reweighted. Validation rows and the "
            "existing train/val split are unchanged."
        ),
    }
    with open(output_summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Scored train negatives: {len(sorted_scores)}")
    print(
        "Hard negatives: "
        f"score>=0.5: {summary['hard_negative_count_score_ge_0_5']}, "
        f"score>=0.7: {summary['hard_negative_count_score_ge_0_7']}"
    )
    print(f"Weighted manifest: {output_csv}")
    print(f"Summary: {output_summary}")


if __name__ == "__main__":
    main()
