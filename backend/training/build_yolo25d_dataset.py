"""
build_yolo25d_dataset.py — LUNA16 → 3-slice 2.5D YOLO 检测数据集

核心思路:
  医生看 CT 是上下滑动观察——孤立结节只出现在少数层面，
  血管在多个层面连续走行。单层 2D 模型无法区分。

  v2.3 方案: 输入 3 个连续切片，堆叠为 RGB 3-channel PNG，
  YOLO11s 不改第一层，直接复用 3-channel 输入。

  通道:  R=z-1   G=z核心层  B=z+1
  标签:  仅中心层 G 上的 bbox

用法:
    cd D:/python_gjw/radscout2/version2.3/backend
    D:/Python/python.exe training/build_yolo25d_dataset.py \
        --data_dir D:/python_gjw/radscout2/data/luna16 \
        --out_dir ../datasets/luna_yolo_25d_3slice \
        --subsets 0 1 2 3 4 5 6 \
        --img_size 512 --neg_ratio 2 --debug_count 100
"""

import os, sys, json, random, time, argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── 常量 ──
LUNG_MIN, LUNG_MAX = -1000, 400
SLICE_RADIUS = 1  # 正样本: 结节中心 z ±1


def world_to_voxel(wx, wy, wz, origin, spacing):
    ox, oy, oz = origin; sx, sy, sz = spacing
    return (int(round((wx-ox)/sx)), int(round((wy-oy)/sy)), int(round((wz-oz)/sz)))


def hu_to_uint8(hu_arr):
    c = np.clip(hu_arr, LUNG_MIN, LUNG_MAX)
    return ((c - LUNG_MIN) / (LUNG_MAX - LUNG_MIN) * 255).astype(np.uint8)


def get_slice(volume, z, num_slices):
    """安全取切片: z 越界→复制最近有效层"""
    z = max(0, min(z, num_slices-1))
    return volume[z].astype(np.float32)


def make_3channel(z_minus, z_center, z_plus, target_size=512):
    """
    三张单通道 2D 切片 → 一张 3-channel RGB 图 (R=z-1, G=z, B=z+1)
    返回 (rgb_uint8_array, scale_w, scale_h)
    """
    h, w = z_minus.shape
    # HU→uint8
    r = hu_to_uint8(z_minus)
    g = hu_to_uint8(z_center)
    b = hu_to_uint8(z_plus)
    # 堆叠
    rgb = np.stack([r, g, b], axis=-1)  # [H, W, 3]
    # Resize
    pil = Image.fromarray(rgb)
    pil = pil.resize((target_size, target_size), Image.BILINEAR)
    out = np.array(pil)
    return out, target_size/w, target_size/h


def make_bbox_yolo(cx_v, cy_v, w_v, h_v, img_w, img_h):
    """体素 bbox → YOLO 归一化格式"""
    cx = np.clip(cx_v / img_w, 0, 1)
    cy = np.clip(cy_v / img_h, 0, 1)
    nw = np.clip(w_v / img_w, 0.001, 1)
    nh = np.clip(h_v / img_h, 0.001, 1)
    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def find_subset_dirs(data_dir, subset_ids):
    found = []
    for sid in subset_ids:
        d = os.path.join(data_dir, f"subset{sid}")
        if not os.path.isdir(d): continue
        inner = os.path.join(d, f"subset{sid}")
        actual = inner if os.path.isdir(inner) else d
        cnt = len([f for f in os.listdir(actual) if f.endswith('.mhd')])
        if cnt: found.append({"id": sid, "path": actual, "mhd_count": cnt})
        else:  print(f"  [WARN] subset{sid}: 0 .mhd files")
    return found


def scan_mhd(subset_dirs):
    m = {}
    for sd in subset_dirs:
        for f in os.listdir(sd["path"]):
            if f.endswith(".mhd"): m[f.replace(".mhd","")] = os.path.join(sd["path"], f)
    return m


def build_25d_dataset(data_dir, out_dir, subset_ids, img_size=512,
                       neg_ratio=2, debug_count=100):
    random.seed(42)  # 确保负样本采样可复现
    t0 = time.time()
    build_log = {"steps": [], "warnings": []}

    # 1. subset 扫描
    print("="*60 + "\n  Step 1: Scanning subsets...")
    sdirs = find_subset_dirs(data_dir, subset_ids)
    if not sdirs: raise FileNotFoundError("No valid subsets")
    print(f"  {len(sdirs)} subsets, {sum(s['mhd_count'] for s in sdirs)} .mhd files")

    # 2. 标注
    print("\n  Step 2: Loading annotations...")
    anno = pd.read_csv(os.path.join(data_dir, "annotations.csv"))
    print(f"  {len(anno)} nodules, {anno['seriesuid'].nunique()} volumes")

    # 3. 匹配
    print("\n  Step 3: Matching UIDs...")
    mhd_map = scan_mhd(sdirs)
    anno_uids = set(anno["seriesuid"].unique())
    matched = sorted([u for u in mhd_map if u in anno_uids])
    print(f"  Matched: {len(matched)}, Missed: {len(anno_uids - set(mhd_map))}")

    # 4. Split
    print("\n  Step 4: Splitting train/val...")
    train_uids, val_uids = train_test_split(matched, test_size=0.2, random_state=42)
    train_set, val_set = set(train_uids), set(val_uids)
    assert not (train_set & val_set), "Train/val overlap!"
    print(f"  Train: {len(train_set)} vol, Val: {len(val_set)} vol, Overlap: 0")

    # 5. 提取
    print("\n  Step 5: Extracting 3-slice 2.5D samples...")
    for d in ["images/train","images/val","labels/train","labels/val",
              "debug/train","debug/val"]:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    counters = {"train_img": 0, "val_img": 0,
                "train_pos": 0, "val_pos": 0,
                "train_neg": 0, "val_neg": 0,
                "train_bbox": 0, "val_bbox": 0}
    skipped = []
    uid_mapping = {}  # 文件名前缀(无后缀) → 完整 seriesuid，供后续反向查找

    # Debug: 收集 3-slice 对比图 (z-1 / z / z+1 三列)
    debug_3slice = {"train": [], "val": []}  # list of (img_r, img_g, img_b, uid, z)

    for uid in tqdm(sorted(matched), desc="Processing"):
        split = "train" if uid in train_set else "val"
        pfx = "train" if split == "train" else "val"

        try:
            im = sitk.ReadImage(mhd_map[uid])
            vol = sitk.GetArrayFromImage(im)
            origin, spacing = im.GetOrigin(), im.GetSpacing()
        except Exception as e:
            skipped.append({"uid": uid, "reason": str(e)})
            continue

        N = vol.shape[0]
        nodules = anno[anno["seriesuid"] == uid]

        # ── 正样本: 结节中心 z ±1 各出 1 张 3-slice 图 ──
        pos_z_set = set()
        for _, row in nodules.iterrows():
            vx, vy, vz = world_to_voxel(row["coordX"], row["coordY"], row["coordZ"], origin, spacing)
            dvx = row["diameter_mm"] / spacing[0]
            dvy = row["diameter_mm"] / spacing[1]
            for dz in range(-SLICE_RADIUS, SLICE_RADIUS+1):
                z = vz + dz
                if 0 <= z < N:
                    z1 = get_slice(vol, z-1, N)
                    zc = get_slice(vol, z,   N)
                    z2 = get_slice(vol, z+1, N)

                    rgb, sw, sh = make_3channel(z1, zc, z2, img_size)

                    # bbox: 体素 → 缩放后归一化
                    cx_s = vx * sw; cy_s = vy * sh
                    w_s = dvx * sw; h_s = dvy * sh
                    line = make_bbox_yolo(cx_s, cy_s, w_s, h_s, img_size, img_size)

                    fname = f"{uid[:30]}_{z:04d}_{counters[f'{pfx}_img']:06d}.png"
                    Image.fromarray(rgb).save(os.path.join(out_dir, "images", pfx, fname))
                    with open(os.path.join(out_dir, "labels", pfx, fname.replace(".png",".txt")), "w") as f:
                        f.write(line + "\n")
                    uid_mapping[fname.replace(".png","")] = uid  # 记录完整 UID
                    counters[f"{pfx}_img"] += 1
                    counters[f"{pfx}_pos"] += 1
                    counters[f"{pfx}_bbox"] += 1
                    pos_z_set.add(z)

                    # debug 3-slice comparison (limit)
                    if len(debug_3slice[split]) < debug_count // 2:
                        debug_3slice[split].append((
                            hu_to_uint8(z1), hu_to_uint8(zc), hu_to_uint8(z2),
                            uid, z, vx, vy, w_s, h_s
                        ))

        # ── 负样本: 无结节区域的 3-slice ──
        neg_cnt = int(max(len(nodules) * neg_ratio, 3))
        neg_cands = [z for z in range(1, N-1) if z not in pos_z_set]
        for z in random.sample(neg_cands, min(neg_cnt, len(neg_cands))) if neg_cands else []:
            z1 = get_slice(vol, z-1, N)
            zc = get_slice(vol, z,   N)
            z2 = get_slice(vol, z+1, N)
            rgb, sw, sh = make_3channel(z1, zc, z2, img_size)
            fname = f"{uid[:30]}_{z:04d}_{counters[f'{pfx}_img']:06d}.png"
            Image.fromarray(rgb).save(os.path.join(out_dir, "images", pfx, fname))
            with open(os.path.join(out_dir, "labels", pfx, fname.replace(".png",".txt")), "w") as f:
                f.write("")
            uid_mapping[fname.replace(".png","")] = uid  # 记录完整 UID
            counters[f"{pfx}_img"] += 1
            counters[f"{pfx}_neg"] += 1

    # ── 6. Debug: 3-slice 对比图 ──
    print("\n  Step 6: Generating 3-slice comparison debug images...")
    for split, items in debug_3slice.items():
        pfx = "train" if split == "train" else "val"
        for i, (r_img, g_img, b_img, uid, z, vx, vy, w_s, h_s) in enumerate(items):
            # 三列拼接: z-1 | z中心 | z+1，在中心层画 bbox
            panel = Image.new("RGB", (img_size * 3, img_size))
            for col_idx, (img_arr, label) in enumerate([
                (r_img, "z-1"), (g_img, f"z={z}"), (b_img, "z+1")
            ]):
                pil_slice = Image.fromarray(img_arr).resize((img_size, img_size))
                panel.paste(pil_slice, (col_idx * img_size, 0))
            # bbox on center column (column 1)
            draw = ImageDraw.Draw(panel)
            sc_w = img_size / g_img.shape[1]
            sc_h = img_size / g_img.shape[0]
            x1 = max(0, img_size + int((vx - w_s/2) * sc_w))
            y1 = max(0, int((vy - h_s/2) * sc_h))
            x2 = min(panel.width, img_size + int((vx + w_s/2) * sc_w))
            y2 = min(panel.height, int((vy + h_s/2) * sc_h))
            if x2 > x1 and y2 > y1:
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            try:
                panel.save(os.path.join(out_dir, "debug", pfx, f"3slice_{i:04d}_{uid[:15]}.png"))
            except:
                pass

    # Also generate standard bbox overlay debug (center slice only like v2.2)
    print("  Creating center-slice bbox debug images...")
    for split in ["train", "val"]:
        pfx = "train" if split == "train" else "val"
        img_dir = os.path.join(out_dir, "images", pfx)
        lbl_dir = os.path.join(out_dir, "labels", pfx)
        pngs = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        # pick files with nodules
        nod_pngs = []
        for pf in pngs:
            lf = pf.replace(".png", ".txt")
            if os.path.isfile(os.path.join(lbl_dir, lf)) and os.path.getsize(os.path.join(lbl_dir, lf)) > 0:
                nod_pngs.append(pf)
        random.shuffle(nod_pngs)
        cnt = min(debug_count // 2, len(nod_pngs))
        for dbg_i in range(cnt):
            pf = nod_pngs[dbg_i]
            img = Image.open(os.path.join(img_dir, pf)).convert("RGB")
            lf = pf.replace(".png", ".txt")
            with open(os.path.join(lbl_dir, lf)) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, cx, cy, w, h = parts
                        cx,cy,w,h = float(cx),float(cy),float(w),float(h)
                        x1 = int((cx-w/2)*img_size); y1 = int((cy-h/2)*img_size)
                        x2 = int((cx+w/2)*img_size); y2 = int((cy+h/2)*img_size)
                        ImageDraw.Draw(img).rectangle([x1,y1,x2,y2], outline=(0,255,0), width=2)
            img.save(os.path.join(out_dir, "debug", pfx, f"bbox_{dbg_i:04d}_{pf[:30]}.png"))

    # ── 7. data.yaml ──
    out_fwd = out_dir.replace("\\", "/")
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(f"# LUNA16 3-slice 2.5D YOLO Dataset\n")
        f.write(f"# R=z-1  G=z(center)  B=z+1\n")
        f.write(f"path: {out_fwd}\n")
        f.write(f"train: images/train\nval: images/val\nnc: 1\nnames:\n  0: nodule\n")

    # ── 8. Summaries ──
    summary = {
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "3-slice 2.5D (R=z-1, G=z, B=z+1)",
        "subsets": [s["id"] for s in sdirs],
        "total_volumes": len(mhd_map),
        "matched_volumes": len(matched),
        "train_volumes": len(train_set),
        "val_volumes": len(val_set),
        "overlap": 0,
        "positive_images": {"train": counters["train_pos"], "val": counters["val_pos"]},
        "negative_images": {"train": counters["train_neg"], "val": counters["val_neg"]},
        "total_images": {"train": counters["train_img"], "val": counters["val_img"]},
        "total_bboxes": {"train": counters["train_bbox"], "val": counters["val_bbox"]},
    }
    with open(os.path.join(out_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── uid_mapping.json ──
    with open(os.path.join(out_dir, "uid_mapping.json"), "w") as f:
        json.dump(uid_mapping, f, indent=2)
    print(f"\n  uid_mapping.json saved: {len(uid_mapping)} entries")

    build_log["elapsed"] = round(time.time()-t0, 1)
    with open(os.path.join(out_dir, "build_log.json"), "w") as f:
        json.dump(build_log, f, indent=2)

    print("\n" + "="*60)
    print(f"  BUILD COMPLETE ({build_log['elapsed']}s)")
    print(f"  Train: {counters['train_pos']}pos + {counters['train_neg']}neg = {counters['train_img']} imgs")
    print(f"  Val:   {counters['val_pos']}pos + {counters['val_neg']}neg = {counters['val_img']} imgs")
    print(f"  Bboxes: {counters['train_bbox']} train + {counters['val_bbox']} val")
    print("="*60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--subsets", type=int, nargs="+", default=list(range(7)))
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--neg_ratio", type=float, default=2.0)
    ap.add_argument("--debug_count", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
    build_25d_dataset(args.data_dir, os.path.abspath(args.out_dir),
                       args.subsets, args.img_size, args.neg_ratio, args.debug_count)
