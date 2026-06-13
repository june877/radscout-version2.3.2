"""
ct_patch_utils.py — 从原始 CT volume 裁剪 2.5D patch

2.5D patch: 在 3 个连续 axial 层面 (z-1, z, z+1) 上，
按 bbox 区域裁剪 → resize 到 patch_size → 返回 [3, patch_size, patch_size] tensor。

边缘处理: z=0 用 [0,0,1], z=last 用 [last-1,last,last]。
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LUNG_MIN = -1000
LUNG_MAX = 400


# ── CT volume 访问（带缓存，避免反复读 .mhd） ──

_volume_cache = {}
_cache_max = 8


def _get_volume(mhd_path):
    """读取 CT volume（LRU 缓存）"""
    if mhd_path in _volume_cache:
        return _volume_cache[mhd_path]
    img = sitk.ReadImage(mhd_path)
    vol = sitk.GetArrayFromImage(img)  # [Z, Y, X]
    if len(_volume_cache) >= _cache_max:
        _volume_cache.pop(next(iter(_volume_cache)))
    _volume_cache[mhd_path] = vol
    return vol


def clear_volume_cache():
    _volume_cache.clear()


# ── 2.5D patch 裁剪 ──

def extract_25d_patch(mhd_path, slice_z, x1, y1, x2, y2, patch_size=96):
    """
    从原始 CT volume 裁剪 2.5D patch。

    参数:
        mhd_path:    .mhd 文件路径
        slice_z:     int, 中心层 axial index (0-based)
        x1,y1,x2,y2: int, 原始 CT 体素坐标的 bbox (像素, 左上+右下)
        patch_size:  输出 patch 尺寸 (正方形)

    返回:
        torch.Tensor: [3, patch_size, patch_size], float32, HU clip → [0,1]
                      如果 bbox 无效则返回 None
    """
    try:
        volume = _get_volume(mhd_path)
    except Exception as e:
        print(f"  [WARN] Cannot read {mhd_path}: {e}")
        return None

    N = volume.shape[0]
    H, W = volume.shape[1], volume.shape[2]

    # 边界保护
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(W - 1, int(x2))
    y2 = min(H - 1, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    # ── 裁剪三层 ──
    slices = []
    for dz in [-1, 0, 1]:
        z = slice_z + dz
        if z < 0:
            z = 0
        elif z >= N:
            z = N - 1

        # 从 volume 中取 patch
        patch = volume[z, y1:y2, x1:x2].astype(np.float32)

        # HU 归一化
        patch = np.clip(patch, LUNG_MIN, LUNG_MAX)
        patch = (patch - LUNG_MIN) / (LUNG_MAX - LUNG_MIN)

        # Resize 到 patch_size × patch_size
        from PIL import Image
        patch_u8 = (patch * 255).astype(np.uint8)
        patch_resized = np.array(
            Image.fromarray(patch_u8).resize((patch_size, patch_size), Image.BILINEAR)
        ).astype(np.float32) / 255.0

        slices.append(patch_resized)

    # 堆叠 [3, patch_size, patch_size]
    return torch.from_numpy(np.stack(slices, axis=0))


# ── 测试 ──
if __name__ == "__main__":
    # 测试：从 subset0 第一个 volume 裁剪一个 patch
    data_dir = r"D:/python_gjw/radscout2/data/luna16"
    # 找一个 .mhd
    import glob
    mhd_files = glob.glob(os.path.join(data_dir, "subset0/subset0/*.mhd"))
    if not mhd_files:
        mhd_files = glob.glob(os.path.join(data_dir, "subset0/*.mhd"))
    if mhd_files:
        test_mhd = mhd_files[0]
        print(f"Testing with: {os.path.basename(test_mhd)}")
        # 在中心层中心位置裁剪一个 64×64 patch
        vol = _get_volume(test_mhd)
        z_mid = vol.shape[0] // 2
        cx, cy = vol.shape[2] // 2, vol.shape[1] // 2
        r = 32
        patch = extract_25d_patch(test_mhd, z_mid, cx-r, cy-r, cx+r, cy+r, patch_size=96)
        if patch is not None:
            print(f"Patch shape: {patch.shape} (expected [3, 96, 96])")
            print(f"Value range: [{patch.min():.3f}, {patch.max():.3f}]")
            print("ct_patch_utils test passed!")
        else:
            print("extract_25d_patch returned None")
    else:
        print("No .mhd files found for testing")
