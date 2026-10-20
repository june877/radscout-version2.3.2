"""Build only frozen train/val volumes; one (full seriesuid, center z) per image.

Reference boxes approximate a spherical annotation intersecting a slice slab.
They are derived boxes, not manual segmentations. Full CTs remain external.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct_preprocessing import make_rgb_hu, context_indices, PREPROCESS_VERSION


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def annotation_geometry(image, rows):
    direction = np.array(image.GetDirection()).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("Sphere/slab rule requires an orthonormal image direction")
    result = []
    for i, row in rows.iterrows():
        world = tuple(float(row[c]) for c in ["coordX", "coordY", "coordZ"])
        voxel = tuple(image.TransformPhysicalPointToContinuousIndex(world))
        back = tuple(image.TransformContinuousIndexToPhysicalPoint(voxel))
        error = float(np.max(np.abs(np.array(back) - world)))
        if error > 1e-6:
            raise ValueError("Physical/voxel round trip failed")
        if not all(-0.5 <= v <= n - 0.5 for v, n in zip(voxel, image.GetSize())):
            raise ValueError(f"Annotation {i} outside image: {voxel}")
        diameter = float(row.diameter_mm)
        if not math.isfinite(diameter) or diameter <= 0:
            raise ValueError("Invalid annotation diameter")
        result.append(dict(annotation_id=int(i), world_mm=world, voxel_continuous=voxel,
                           diameter_mm=diameter, roundtrip_max_error_mm=error))
    return result


def reference_boxes(image, nodules):
    """Sphere intersected by finite slice slab: use the largest circle in slab.

    dz=max(abs(z-cz)*sz - sz/2,0); diameter(z)=2*sqrt(r*r-dz*dz).
    Positive slabs satisfy dz<r, conservatively preserving thick-slice nodules.
    Box uses pixel EDGE coordinates (continuous voxel center + 0.5).
    """
    width, height, depth = image.GetSize()
    sx, sy, sz = image.GetSpacing()
    by_z = {}
    for n in nodules:
        cx, cy, cz = n["voxel_continuous"]
        r = n["diameter_mm"] / 2
        lo = max(0, math.floor(cz - r / sz - 0.5))
        hi = min(depth - 1, math.ceil(cz + r / sz + 0.5))
        for z in range(lo, hi + 1):
            dz = max(abs(z - cz) * sz - sz / 2, 0.0)
            if dz >= r:
                continue
            diameter = 2 * math.sqrt(max(0, r * r - dz * dz))
            x1, x2 = max(0, cx + 0.5 - diameter / (2 * sx)), min(width, cx + 0.5 + diameter / (2 * sx))
            y1, y2 = max(0, cy + 0.5 - diameter / (2 * sy)), min(height, cy + 0.5 + diameter / (2 * sy))
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Degenerate reference box")
            by_z.setdefault(z, []).append(dict(annotation_id=n["annotation_id"],
                 xyxy_original_edges=[x1, y1, x2, y2], reference_diameter_mm=diameter))
    return by_z


def background_layers(depth, positive_z, uid, cap):
    # Require all three channels plus one extra safety layer clear of target slabs.
    forbidden = {q for p in positive_z for q in range(max(0, p - 2), min(depth, p + 3))}
    candidates = [z for z in range(depth) if z not in forbidden]
    seed = int(hashlib.sha256(("day1-bg-v1:" + uid).encode()).hexdigest()[:16], 16)
    return sorted(random.Random(seed).sample(candidates, min(cap, len(candidates))))


def plan_volume(image, rows, uid):
    nodules = annotation_geometry(image, rows)
    positive = reference_boxes(image, nodules)
    cap = min(12, max(4, math.ceil(0.75 * len(positive)))) if nodules else 8
    negative = background_layers(image.GetSize()[2], positive, uid, cap)
    return nodules, positive, negative


def yolo_lines(boxes, width, height):
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy_original_edges"]
        lines.append(f"0 {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    return lines


def build(manifest, annotations, out, mode="25d", limit=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty output directory; do not mix runs")
    volumes = pd.read_csv(manifest, dtype={"seriesuid": str})
    volumes = volumes[volumes.split.isin(["train", "val"])].sort_values("seriesuid")
    if limit:
        # Pilot is explicitly separate and never substitutes for the frozen list.
        volumes = pd.concat([group.head(limit) for _, group in volumes.groupby("split")]).sort_values("seriesuid")
    anno = pd.read_csv(annotations, dtype={"seriesuid": str})
    grouped = {uid: rows for uid, rows in anno.groupby("seriesuid", sort=False)}
    empty = anno.iloc[0:0]
    for split in ["train", "val"]:
        for kind in ["images", "labels"]:
            (out / kind / split).mkdir(parents=True, exist_ok=True)
    records, stats, timings = [], Counter(), []
    started = time.perf_counter()
    for number, row in enumerate(volumes.itertuples(), 1):
        t = time.perf_counter()
        if sha256(row.mhd_path) != row.mhd_sha256:
            raise ValueError("Source header changed after freeze")
        image = sitk.ReadImage(row.mhd_path)
        volume = sitk.GetArrayViewFromImage(image)
        nodules, positive, negative = plan_volume(image, grouped.get(row.seriesuid, empty), row.seriesuid)
        width, height, depth = image.GetSize()
        for z in sorted(set(positive) | set(negative)):
            name = f"{row.seriesuid}_z{z:04d}.png"
            rgb = make_rgb_hu(volume, z, 512, mode)
            image_path = out / "images" / row.split / name
            rgb.save(image_path, compress_level=3)
            boxes = positive.get(z, [])
            label_path = out / "labels" / row.split / (Path(name).stem + ".txt")
            label_path.write_text("\n".join(yolo_lines(boxes, width, height)) + ("\n" if boxes else ""), encoding="utf-8")
            records.append(dict(seriesuid=row.seriesuid, split=row.split, z_original=z,
                context_z=json.dumps(context_indices(z, depth) if mode == "25d" else [z,z,z]),
                image=f"images/{row.split}/{name}", label=f"labels/{row.split}/{Path(name).stem}.txt",
                target_count=len(boxes), annotation_ids=json.dumps([b["annotation_id"] for b in boxes]),
                original_width=width, original_height=height, export_width=512, export_height=512,
                scale_x=512/width, scale_y=512/height, preprocessing=PREPROCESS_VERSION,
                image_bytes=image_path.stat().st_size, image_sha256=sha256(image_path), label_sha256=sha256(label_path)))
            stats[f"{row.split}_positive" if boxes else f"{row.split}_background"] += 1
            stats[f"{row.split}_boxes"] += len(boxes)
            stats["multi_target_slices"] += int(len(boxes) > 1)
        elapsed = time.perf_counter() - t
        timings.append(dict(seriesuid=row.seriesuid, split=row.split, seconds=elapsed,
                            positive=len(positive), background=len(negative)))
        if number % 10 == 0 or number == len(volumes):
            print(f"export {number}/{len(volumes)} volumes, {len(records)} samples, {time.perf_counter()-started:.1f}s", flush=True)
    frame = pd.DataFrame(records)
    if frame.duplicated(["seriesuid", "z_original"]).any():
        raise AssertionError("Duplicate volume/slice")
    frame.to_csv(out / "sample_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(timings).to_csv(out / "volume_timings.csv", index=False, encoding="utf-8-sig")
    # Relative paths remain valid after delivery to another directory.
    (out / "data.yaml").write_text("train: images/train\nval: images/val\nnc: 1\nnames:\n  0: nodule\n", encoding="utf-8")
    summary = dict(mode=mode, volumes=len(volumes), samples=len(records), stats=dict(stats),
        total_png_bytes=int(frame.image_bytes.sum()), elapsed_seconds=time.perf_counter()-started,
        manifest_sha256=sha256(manifest), annotations_sha256=sha256(annotations),
        sphere_slab_rule="dz=max(abs(z-cz)*sz-sz/2,0); d=2*sqrt(r^2-dz^2)",
        background_rule="UID SHA256 seeded; targets: min(12,max(4,ceil(.75*positive_slices))); no target:8; center +/-2 clear of target slabs",
        warnings=["Derived spherical reference boxes; not manual slice contours",
                  "Background means no INCLUDED target annotation; excluded/uncatalogued findings may remain"],
        errors=[])
    write_json(out / "build_summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["25d", "2d"], default="25d")
    p.add_argument("--limit-per-split", type=int)
    a = p.parse_args()
    build(a.manifest, a.annotations, a.out, a.mode, a.limit_per_split)
