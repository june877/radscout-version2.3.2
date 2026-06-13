"""
LUNA16 Dataset — 2D CT 切片数据集（v2：切片缓存系统）

核心优化:
  旧版: __getitem__ 每次都 load_mhd_volume() → 读几十 MB .raw 文件
  新版: 首次运行 build_slice_cache() 把所有切片预处理存为 .pt，
        之后 CachedSliceDataset 直接读 .pt（亚毫秒级），10-50x 加速。

用法:
  # 第一次：构建缓存（一次性，约 2-5 分钟）
  python training/luna_dataset.py --data_dir ... --cache_dir outputs/cache_slices

  # 之后训练：自动使用缓存
  # create_dataloaders(..., cache_dir="outputs/cache_slices") 自动检测
"""

import os
import sys
import json
import math
import random
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
# 配置常量
# ============================================================
LUNG_WINDOW_MIN = -1000
LUNG_WINDOW_MAX = 400
IMAGE_SIZE = 512
SLICE_RADIUS = 1  # 每个结节上下各取 N 张切片


# ============================================================
# 工具函数
# ============================================================

def world_to_voxel(world_xyz, origin, spacing, direction=None):
    """世界坐标 (mm) → 体素坐标 (voxel index)"""
    ox, oy, oz = origin
    sx, sy, sz = spacing
    wx, wy, wz = world_xyz
    vx = int(round((wx - ox) / sx))
    vy = int(round((wy - oy) / sy))
    vz = int(round((wz - oz) / sz))
    return vx, vy, vz


def normalize_hu(hu_array):
    """CT HU 值 → 肺窗 [-1000,400] → [0,1]"""
    clipped = np.clip(hu_array, LUNG_WINDOW_MIN, LUNG_WINDOW_MAX)
    normalized = (clipped - LUNG_WINDOW_MIN) / (LUNG_WINDOW_MAX - LUNG_WINDOW_MIN)
    return normalized.astype(np.float32)


def load_mhd_volume(mhd_path):
    """SimpleITK 读取 .mhd → (volume, origin, spacing, direction)"""
    image = sitk.ReadImage(mhd_path)
    volume = sitk.GetArrayFromImage(image)
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = (np.array(image.GetDirection()).reshape(3, 3)
                 if image.GetDirection() else np.eye(3))
    return volume, origin, spacing, direction


def _find_subset0_dir(data_dir):
    """找到 subset0 目录（兼容 nested/unnested 结构）"""
    for cand in [
        os.path.join(data_dir, "subset0", "subset0"),
        os.path.join(data_dir, "subset0"),
    ]:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"找不到 subset0 目录！尝试过 nested 和 flat 结构。")


def _get_seriesuid_to_mhd(data_dir):
    """扫描 subset0，返回 {seriesuid: mhd_path}"""
    subset0_dir = _find_subset0_dir(data_dir)
    mhd_files = sorted([f for f in os.listdir(subset0_dir) if f.endswith(".mhd")])
    if not mhd_files:
        raise FileNotFoundError(f"在 {subset0_dir} 中没有找到 .mhd 文件！")
    return {f.replace(".mhd", ""): os.path.join(subset0_dir, f) for f in mhd_files}


def _process_slice(image_2d, orig_h, orig_w):
    """
    处理单张 2D 切片: HU 归一化 → resize 512 → 转 tensor [1, 512, 512]
    返回 (tensor, scale_h, scale_w)
    """
    image = normalize_hu(image_2d.astype(np.float32))
    scale_h = IMAGE_SIZE / orig_h
    scale_w = IMAGE_SIZE / orig_w
    from PIL import Image
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    image_pil = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image = np.array(image_pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).unsqueeze(0)
    return tensor, scale_h, scale_w


# ============================================================
# 原始 Dataset（保留兼容，__getitem__ 内加了 volume 缓存）
# ============================================================

class Luna2DSliceDataset(Dataset):
    """
    从 .mhd/.raw 实时提取切片（无磁盘缓存时使用）。

    优化: __getitem__ 内缓存最近读取的 volume，避免同一 volume
          的多个切片反复读 .raw 文件。
    """

    def __init__(self, data_dir, annotations_path, seriesuids,
                 limit_samples=None):
        self.data_dir = data_dir
        self.limit_samples = limit_samples
        self.series_to_mhd = _get_seriesuid_to_mhd(data_dir)

        all_annotations = pd.read_csv(annotations_path)
        self.annotations = all_annotations[
            all_annotations["seriesuid"].isin(seriesuids)
        ].copy()

        self.samples = []
        self._build_samples()

        # 简单的 volume 缓存：存最近 N 个 CT
        self._vol_cache = {}
        self._vol_cache_max = 3

        print(f"Dataset: {len(self.samples)} slices "
              f"({sum(1 for s in self.samples if s['has_nodule'])} pos, "
              f"{sum(1 for s in self.samples if not s['has_nodule'])} neg) "
              f"[no cache]")

    def _build_samples(self):
        for i, (seriesuid, mhd_path) in enumerate(self.series_to_mhd.items()):
            nodules = self.annotations[self.annotations["seriesuid"] == seriesuid]
            try:
                volume, origin, spacing, _ = load_mhd_volume(mhd_path)
            except Exception as e:
                print(f"  [Warn] {seriesuid[:40]}...: {e}")
                continue

            num_slices = volume.shape[0]

            # 正样本
            for _, row in nodules.iterrows():
                vx, vy, vz = world_to_voxel(
                    (row["coordX"], row["coordY"], row["coordZ"]), origin, spacing)
                diam_vx = row["diameter_mm"] / spacing[0]
                diam_vy = row["diameter_mm"] / spacing[1]
                for dz in range(-SLICE_RADIUS, SLICE_RADIUS + 1):
                    sz = vz + dz
                    if 0 <= sz < num_slices:
                        self.samples.append({
                            "seriesuid": seriesuid, "slice_z": int(sz),
                            "has_nodule": 1, "bbox_cx": vx, "bbox_cy": vy,
                            "bbox_w": diam_vx, "bbox_h": diam_vy,
                            "diameter_mm": row["diameter_mm"],
                        })

            # 负样本
            pos_slices = {s["slice_z"] for s in self.samples if s["seriesuid"] == seriesuid}
            neg_count = min(max(len(nodules) * 2, 5), 30)
            neg_candidates = [z for z in range(num_slices) if z not in pos_slices]
            if neg_candidates:
                for z in random.sample(neg_candidates, min(neg_count, len(neg_candidates))):
                    self.samples.append({
                        "seriesuid": seriesuid, "slice_z": z,
                        "has_nodule": 0, "bbox_cx": 0, "bbox_cy": 0,
                        "bbox_w": 25, "bbox_h": 25, "diameter_mm": 0,
                    })

            if (i + 1) % 20 == 0:
                print(f"  Scanned {i+1}/{len(self.series_to_mhd)} volumes...")

        if self.limit_samples and len(self.samples) > self.limit_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:self.limit_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        uid = sample["seriesuid"]
        mhd_path = self.series_to_mhd[uid]

        # 缓存检查
        if uid in self._vol_cache:
            volume, _, _ = self._vol_cache[uid]
        else:
            volume, origin, spacing, _ = load_mhd_volume(mhd_path)
            # LRU 淘汰
            if len(self._vol_cache) >= self._vol_cache_max:
                self._vol_cache.pop(next(iter(self._vol_cache)))
            self._vol_cache[uid] = (volume, origin, spacing)

        # 提取切片
        sz = max(0, min(sample["slice_z"], volume.shape[0] - 1))
        image_2d = volume[int(sz)]
        orig_h, orig_w = image_2d.shape
        img_tensor, scale_h, scale_w = _process_slice(image_2d, orig_h, orig_w)

        # bbox
        if sample["has_nodule"]:
            cx = sample["bbox_cx"] * scale_w / IMAGE_SIZE
            cy = sample["bbox_cy"] * scale_h / IMAGE_SIZE
            bw = sample["bbox_w"] * scale_w / IMAGE_SIZE
            bh = sample["bbox_h"] * scale_h / IMAGE_SIZE
        else:
            cx = cy = bw = bh = 0.0

        target = {
            "has_nodule": torch.tensor(sample["has_nodule"], dtype=torch.float32),
            "bbox": torch.tensor([cx, cy, bw, bh], dtype=torch.float32),
            "seriesuid": uid,
        }
        return img_tensor, target


# ============================================================
# 切片缓存构建
# ============================================================

def build_slice_cache(data_dir, annotations_path, cache_dir, val_split=0.2,
                      limit_samples=None):
    """
    一次性遍历所有 CT volume，提取切片，保存为 .pt 文件。

    缓存结构:
      cache_dir/
        train/
          000000.pt  ← 每个文件包含 {"image": [1,512,512], "target": {...}}
          000001.pt
          ...
        val/
          ...
        metadata.json  ← {"num_train": N, "num_val": M, ...}
    """
    os.makedirs(cache_dir, exist_ok=True)
    train_dir = os.path.join(cache_dir, "train")
    val_dir = os.path.join(cache_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    t0 = time.time()

    # ── 获取所有 seriesuid 并按 split 分组 ──
    series_to_mhd = _get_seriesuid_to_mhd(data_dir)
    annotations = pd.read_csv(annotations_path)
    anno_uids = set(annotations["seriesuid"].unique())
    valid_uids = [u for u in series_to_mhd if u in anno_uids]
    train_uids, val_uids = train_test_split(valid_uids, test_size=val_split, random_state=42)

    print(f"Building slice cache...")
    print(f"  Train volumes: {len(train_uids)}, Val volumes: {len(val_uids)}")
    print(f"  Cache dir: {cache_dir}")

    train_samples = []
    val_samples = []

    # 扫描所有 volume 收集样本元数据
    all_mhd = series_to_mhd
    for split_name, uids in [("train", train_uids), ("val", val_uids)]:
        sample_list = train_samples if split_name == "train" else val_samples
        for seriesuid in uids:
            mhd_path = all_mhd[seriesuid]
            nodules = annotations[annotations["seriesuid"] == seriesuid]
            try:
                volume, origin, spacing, _ = load_mhd_volume(mhd_path)
            except Exception as e:
                print(f"  [Warn] {seriesuid[:40]}...: {e}")
                continue

            num_slices = volume.shape[0]
            # 正样本
            for _, row in nodules.iterrows():
                vx, vy, vz = world_to_voxel(
                    (row["coordX"], row["coordY"], row["coordZ"]), origin, spacing)
                diam_vx = row["diameter_mm"] / spacing[0]
                diam_vy = row["diameter_mm"] / spacing[1]
                for dz in range(-SLICE_RADIUS, SLICE_RADIUS + 1):
                    sz = vz + dz
                    if 0 <= sz < num_slices:
                        sample_list.append({
                            "seriesuid": seriesuid, "slice_z": int(sz),
                            "has_nodule": 1, "bbox_cx": vx, "bbox_cy": vy,
                            "bbox_w": diam_vx, "bbox_h": diam_vy,
                            "diameter_mm": float(row["diameter_mm"]), "split": split_name,
                        })
            # 负样本
            pos_slices = {s["slice_z"] for s in sample_list if s["seriesuid"] == seriesuid}
            neg_count = min(max(len(nodules) * 2, 5), 30)
            neg_candidates = [z for z in range(num_slices) if z not in pos_slices]
            if neg_candidates:
                for z in random.sample(neg_candidates, min(neg_count, len(neg_candidates))):
                    sample_list.append({
                        "seriesuid": seriesuid, "slice_z": z,
                        "has_nodule": 0, "bbox_cx": 0, "bbox_cy": 0,
                        "bbox_w": 25, "bbox_h": 25, "diameter_mm": 0,
                        "split": split_name,
                    })

    if limit_samples:
        random.shuffle(train_samples)
        train_samples = train_samples[:limit_samples]
        random.shuffle(val_samples)
        val_samples = val_samples[:max(1, limit_samples // 4)]

    # ── 第二遍：实际提取切片并保存 ──
    # 按 seriesuid 组织样本，这样每个 volume 只读一次
    all_samples = train_samples + val_samples
    by_uid = {}
    for s in all_samples:
        by_uid.setdefault(s["seriesuid"], []).append(s)

    train_count = 0
    val_count = 0

    for uid, samples in tqdm(by_uid.items(), desc="Extracting slices"):
        mhd_path = all_mhd[uid]
        try:
            volume, origin, spacing, _ = load_mhd_volume(mhd_path)
        except Exception:
            continue

        for s in samples:
            sz = max(0, min(s["slice_z"], volume.shape[0] - 1))
            image_2d = volume[int(sz)]
            orig_h, orig_w = image_2d.shape
            img_tensor, scale_h, scale_w = _process_slice(image_2d, orig_h, orig_w)

            # bbox 归一化
            if s["has_nodule"]:
                cx = s["bbox_cx"] * scale_w / IMAGE_SIZE
                cy = s["bbox_cy"] * scale_h / IMAGE_SIZE
                bw = s["bbox_w"] * scale_w / IMAGE_SIZE
                bh = s["bbox_h"] * scale_h / IMAGE_SIZE
            else:
                cx = cy = bw = bh = 0.0

            target = {
                "has_nodule": torch.tensor(s["has_nodule"], dtype=torch.float32),
                "bbox": torch.tensor([cx, cy, bw, bh], dtype=torch.float32),
                "seriesuid": uid,
            }

            item = {"image": img_tensor, "target": target}

            if s["split"] == "train":
                torch.save(item, os.path.join(train_dir, f"{train_count:06d}.pt"))
                train_count += 1
            else:
                torch.save(item, os.path.join(val_dir, f"{val_count:06d}.pt"))
                val_count += 1

    # 保存元信息
    metadata = {
        "num_train": train_count,
        "num_val": val_count,
        "image_size": IMAGE_SIZE,
        "lung_window": [LUNG_WINDOW_MIN, LUNG_WINDOW_MAX],
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nCache built: {train_count} train + {val_count} val slices "
          f"in {elapsed:.1f}s")
    return metadata


# ============================================================
# 缓存 Dataset（训练时使用）
# ============================================================

class CachedSliceDataset(Dataset):
    """
    从预处理的 .pt 文件读取切片。

    每个 .pt 文件是一个 dict: {"image": [1,512,512], "target": {...}}
    读取速度: 亚毫秒级（vs 读取 .mhd/.raw 的数百毫秒）
    """

    def __init__(self, split_dir, limit_samples=None):
        """
        参数:
            split_dir: cache 下的 "train/" 或 "val/" 目录
            limit_samples: 限制样本数（调试用）
        """
        self.split_dir = split_dir
        self.file_list = sorted([
            f for f in os.listdir(split_dir) if f.endswith(".pt")
        ])
        if limit_samples and len(self.file_list) > limit_samples:
            self.file_list = self.file_list[:limit_samples]
        print(f"CachedDataset: {len(self.file_list)} slices [{split_dir}]")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item = torch.load(
            os.path.join(self.split_dir, self.file_list[idx]),
            weights_only=False,
        )
        return item["image"], item["target"]


# ============================================================
# DataLoader 工厂
# ============================================================

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    has_nodule = torch.stack([item[1]["has_nodule"] for item in batch])
    bbox = torch.stack([item[1]["bbox"] for item in batch])
    seriesuids = [item[1]["seriesuid"] for item in batch]
    return images, {
        "has_nodule": has_nodule,
        "bbox": bbox,
        "seriesuid": seriesuids,
    }


def create_dataloaders(data_dir, annotations_path, limit_samples=None,
                       batch_size=8, num_workers=0, val_split=0.2,
                       pin_memory=True, cache_dir=None):
    """
    创建 train/val DataLoader。

    如果 cache_dir 存在且包含缓存数据，直接使用 CachedSliceDataset；
    否则回退到 Luna2DSliceDataset（实时读取 .mhd）。

    参数:
        cache_dir: 切片缓存目录（如 outputs/cache_slices）。
                   如果为 None，跳过缓存逻辑。
    """
    # ── 检查缓存 ──
    use_cache = False
    if cache_dir is not None:
        metadata_path = os.path.join(cache_dir, "metadata.json")
        train_cache = os.path.join(cache_dir, "train")
        val_cache = os.path.join(cache_dir, "val")
        if os.path.isfile(metadata_path) and os.path.isdir(train_cache) and os.path.isdir(val_cache):
            use_cache = True
            with open(metadata_path) as f:
                meta = json.load(f)
            print(f"[Cache] Found: {meta['num_train']} train + {meta['num_val']} val slices")
            print(f"[Cache] Built at: {meta.get('built_at', 'unknown')}")

    if use_cache:
        print("[Cache] Using CachedSliceDataset (fast path)")
        train_dataset = CachedSliceDataset(train_cache, limit_samples=limit_samples)
        val_dataset = CachedSliceDataset(
            val_cache,
            limit_samples=limit_samples // 4 if limit_samples else None,
        )
    else:
        print("[No cache] Falling back to Luna2DSliceDataset (slow — reading .mhd/.raw)")
        if cache_dir:
            print(f"[No cache] To build cache: run with --rebuild_cache or call build_slice_cache()")

        # 获取 split
        subset0_dir = _find_subset0_dir(data_dir)
        mhd_uids = [f.replace(".mhd", "") for f in os.listdir(subset0_dir) if f.endswith(".mhd")]
        annotations = pd.read_csv(annotations_path)
        anno_uids = set(annotations["seriesuid"].unique())
        valid_uids = [u for u in mhd_uids if u in anno_uids]
        train_uids, val_uids = train_test_split(valid_uids, test_size=val_split, random_state=42)
        print(f"Train seriesuids: {len(train_uids)}, Val seriesuids: {len(val_uids)}")

        train_dataset = Luna2DSliceDataset(
            data_dir, annotations_path, train_uids,
            limit_samples=limit_samples,
        )
        val_dataset = Luna2DSliceDataset(
            data_dir, annotations_path, val_uids,
            limit_samples=limit_samples // 4 if limit_samples else None,
        )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"数据集为空！train={len(train_dataset)}, val={len(val_dataset)}")

    # persistent_workers 只在 num_workers > 0 时启用
    persistent = num_workers > 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, persistent_workers=persistent,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, persistent_workers=persistent,
    )

    return train_loader, val_loader, train_dataset, val_dataset


# ============================================================
# 测试 / 命令行缓存构建入口
# ============================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build slice cache or test dataset")
    ap.add_argument("--data_dir", type=str, default=r"D:\python_gjw\radscout2\data\luna16")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--limit_samples", type=int, default=None)
    ap.add_argument("--build_cache", action="store_true")
    ap.add_argument("--test_load", action="store_true")
    args = ap.parse_args()

    anno_path = os.path.join(args.data_dir, "annotations.csv")

    if args.build_cache:
        if not args.cache_dir:
            args.cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "outputs", "cache_slices",
            )
        build_slice_cache(args.data_dir, anno_path, args.cache_dir,
                          limit_samples=args.limit_samples)

    if args.test_load:
        cache = args.cache_dir
        if not cache:
            cache = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "outputs", "cache_slices",
            )
        train_loader, val_loader, train_ds, val_ds = create_dataloaders(
            data_dir=args.data_dir, annotations_path=anno_path,
            limit_samples=args.limit_samples, batch_size=4,
            cache_dir=cache,
        )
        t0 = time.time()
        images, targets = next(iter(train_loader))
        print(f"\nBatch load time: {(time.time() - t0)*1000:.1f}ms")
        print(f"Images: {images.shape}, Pos: {targets['has_nodule'].sum().item()}")
