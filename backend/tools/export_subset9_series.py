"""
export_subset9_series.py — 导出 subset9 结节病例连续切片供前端测试

用法:
    cd D:/python_gjw/radscout2/version2.3/backend
    D:/Python/python.exe tools/export_subset9_series.py \
        --data_dir D:/python_gjw/radscout2/data/luna16 \
        --out_dir D:/python_gjw/radscout2/version2.3/test_series \
        --window 40

输出:
    test_series/subset9_case/
        slice_000.png ... slice_xxx.png    (连续 axial 切片)
        metadata.json                       (病例信息)

注意: subset9 只用于前端测试演示，不参与训练。
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LUNG_MIN = -1000
LUNG_MAX = 400


def world_to_voxel(wx, wy, wz, origin, spacing):
    ox, oy, oz = origin
    sx, sy, sz = spacing
    return (int(round((wx - ox) / sx)),
            int(round((wy - oy) / sy)),
            int(round((wz - oz) / sz)))


def find_subset9_mhd(data_dir):
    """找到 subset9 实际 .mhd 路径（兼容嵌套目录结构）"""
    subset9 = os.path.join(data_dir, "subset9")
    if not os.path.isdir(subset9):
        raise FileNotFoundError(f"subset9 目录不存在: {subset9}")
    # 可能嵌套: subset9/subset9/
    for root, dirs, files in os.walk(subset9):
        for f in files:
            if f.endswith(".mhd"):
                return root
    raise FileNotFoundError(f"subset9 中没有 .mhd 文件")


def export_series(data_dir, out_dir, window=40):
    """主流程"""
    # ── 1. 加载标注 & 扫描 subset9 ──
    print("=" * 60)
    print("  Step 1: Loading data...")
    anno = pd.read_csv(os.path.join(data_dir, "annotations.csv"))
    mhd_root = find_subset9_mhd(data_dir)
    mhd_files = sorted([f for f in os.listdir(mhd_root) if f.endswith(".mhd")])
    mhd_uids = {f.replace(".mhd", ""): os.path.join(mhd_root, f) for f in mhd_files}
    print(f"  .mhd files in subset9: {len(mhd_uids)}")

    # ── 2. 匹配有标注的 UID ──
    anno_uids = set(anno["seriesuid"].unique())
    matched = sorted([u for u in mhd_uids if u in anno_uids])
    print(f"  Matched (has nodules): {len(matched)}")
    if not matched:
        print("ERROR: subset9 中没有匹配标注的 volume")
        sys.exit(1)

    # ── 3. 选第一个可用病例 ──
    chosen_uid = None
    chosen_row = None
    for uid in matched:
        nodules = anno[anno["seriesuid"] == uid]
        if len(nodules) > 0:
            chosen_uid = uid
            chosen_row = nodules.iloc[0]
            break

    if chosen_uid is None:
        print("ERROR: 没有找到有结节的病例")
        sys.exit(1)

    print(f"  Selected: {chosen_uid[:50]}...")
    print(f"    Nodules: {len(nodules)}")
    print(f"    World: ({chosen_row['coordX']:.1f}, {chosen_row['coordY']:.1f}, {chosen_row['coordZ']:.1f})")
    print(f"    Diameter: {chosen_row['diameter_mm']:.1f} mm")

    # ── 4. 读取 CT ──
    print("\n  Step 2: Loading CT volume...")
    mhd_path = mhd_uids[chosen_uid]
    img = sitk.ReadImage(mhd_path)
    volume = sitk.GetArrayFromImage(img)  # [Z, Y, X]
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    print(f"    Shape: {volume.shape}")
    print(f"    Spacing: ({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm")
    print(f"    Origin: ({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f})")

    # ── 5. 结节世界坐标 → 体素坐标 ──
    vx, vy, vz = world_to_voxel(
        chosen_row["coordX"], chosen_row["coordY"], chosen_row["coordZ"],
        origin, spacing)
    print(f"\n  Step 3: Nodule center voxel = ({vx}, {vy}, {vz})")

    # ── 6. 确定切片范围 ──
    start_z = max(0, vz - window)
    end_z = min(volume.shape[0] - 1, vz + window)
    print(f"    Export z range: [{start_z}, {end_z}] ({end_z - start_z + 1} slices)")

    # ── 7. 导出 PNG ──
    case_dir = os.path.join(out_dir, "subset9_case")
    os.makedirs(case_dir, exist_ok=True)

    print(f"\n  Step 4: Exporting slices to {case_dir}/...")
    count = 0
    for z in range(start_z, end_z + 1):
        slice_hu = volume[z].astype(np.float32)
        # HU clip + normalize
        clipped = np.clip(slice_hu, LUNG_MIN, LUNG_MAX)
        norm = ((clipped - LUNG_MIN) / (LUNG_MAX - LUNG_MIN) * 255).astype(np.uint8)
        # save
        fname = f"slice_{count:03d}.png"
        Image.fromarray(norm).save(os.path.join(case_dir, fname))
        count += 1

    # ── 8. metadata.json ──
    metadata = {
        "seriesuid": chosen_uid,
        "subset": 9,
        "nodule_center_world": {
            "x": float(chosen_row["coordX"]),
            "y": float(chosen_row["coordY"]),
            "z": float(chosen_row["coordZ"]),
        },
        "nodule_center_voxel": {"x": vx, "y": vy, "z": vz},
        "nodule_diameter_mm": float(chosen_row["diameter_mm"]),
        "exported_start_z": int(start_z),
        "exported_end_z": int(end_z),
        "nodule_relative_index": int(vz - start_z),
        "slice_count": count,
        "spacing_mm": list(map(float, spacing)),
        "origin_mm": list(map(float, origin)),
        "volume_shape": list(volume.shape),
        "note": "Test series from LUNA16 subset9 for v2.3 2.5D inference testing. NOT used for training.",
    }
    with open(os.path.join(case_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ── 9. 完成 ──
    print(f"\n{'='*60}")
    print(f"  EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Series:     {chosen_uid[:40]}...")
    print(f"  Slices:     {count} ({start_z} → {end_z})")
    print(f"  Nodule at:  slice {metadata['nodule_relative_index']} (z={vz})")
    print(f"  Output:     {case_dir}/")
    print(f"\n  提示: 在前端一次性选择 {case_dir}/slice_*.png")
    print(f"  进行连续切片 2.5D 检测。结节位于第 {metadata['nodule_relative_index']} 层。")
    print(f"{'='*60}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export subset9 nodule case for frontend testing")
    ap.add_argument("--data_dir", type=str,
                    default=r"D:/python_gjw/radscout2/data/luna16")
    ap.add_argument("--out_dir", type=str,
                    default=r"D:/python_gjw/radscout2/version2.3/test_series")
    ap.add_argument("--window", type=int, default=40,
                    help="Slice window radius (default 40 → 81 slices)")
    args = ap.parse_args()
    export_series(args.data_dir, args.out_dir, args.window)
