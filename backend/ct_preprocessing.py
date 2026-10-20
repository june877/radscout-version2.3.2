"""Shared pixel-only CT input processing. No annotations are accepted here.

RGB on disk and in PIL is [z-1,z,z+1]. Ultralytics' OpenCV reader uses
BGR internally, then Format(bgr=0) restores RGB before the network.
No lung/lesion crop: retain the whole slice, bilinear resize to a square.
"""
import numpy as np
from PIL import Image

HU_MIN, HU_MAX = -1000.0, 400.0
PREPROCESS_VERSION = "day1-hu1400-rgb-adjacent-v1"


def hu_to_uint8(array):
    clipped = np.clip(np.asarray(array, dtype=np.float32), HU_MIN, HU_MAX)
    return ((clipped - HU_MIN) / (HU_MAX - HU_MIN) * 255.0).astype(np.uint8)


def context_indices(z, depth):
    if depth < 1 or not 0 <= z < depth:
        raise ValueError("Center slice must be inside the volume")
    return [max(0, z - 1), z, min(depth - 1, z + 1)]


def stack_uint8_context(slices, z, target_size=None):
    indices = context_indices(z, len(slices))
    planes = [np.asarray(slices[i], dtype=np.uint8) for i in indices]
    if len({p.shape for p in planes}) != 1 or planes[0].ndim != 2:
        raise ValueError("Context planes must be equally sized grayscale slices")
    rgb = Image.fromarray(np.stack(planes, axis=-1))
    if target_size is not None and rgb.size != (target_size, target_size):
        rgb = rgb.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return rgb


def make_rgb_hu(volume, z, target_size=512, mode="25d"):
    indices = context_indices(z, len(volume))
    if mode == "2d":
        indices = [z, z, z]
    elif mode != "25d":
        raise ValueError("Mode must be 25d or 2d")
    rgb = Image.fromarray(np.stack([hu_to_uint8(volume[i]) for i in indices], axis=-1))
    if rgb.size != (target_size, target_size):
        rgb = rgb.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return rgb


def crop_classifier(rgb, xyxy, patch_size=96):
    """Use the same PIL bilinear crop/resize as the runtime classifier."""
    return rgb.crop(xyxy).resize((patch_size, patch_size), Image.Resampling.BILINEAR)
