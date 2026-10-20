"""Bounding-box helpers shared by v3.1 training and inference."""

import math


def yolo_to_xyxy(cx, cy, width, height, image_width, image_height):
    """Convert normalized YOLO cx/cy/w/h to pixel xyxy coordinates."""
    x1 = (cx - width / 2) * image_width
    y1 = (cy - height / 2) * image_height
    x2 = (cx + width / 2) * image_width
    y2 = (cy + height / 2) * image_height
    return x1, y1, x2, y2


def xyxy_to_yolo(x1, y1, x2, y2, image_width, image_height):
    """Convert pixel xyxy coordinates to normalized YOLO cx/cy/w/h."""
    cx = ((x1 + x2) / 2) / image_width
    cy = ((y1 + y2) / 2) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return cx, cy, width, height


def compute_iou(box1, box2):
    """Calculate IoU for two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def expand_bbox(x1, y1, x2, y2, expand_ratio=1.5):
    """Expand an xyxy box around its center."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = (x2 - x1) * expand_ratio
    height = (y2 - y1) * expand_ratio
    return (
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
    )


def clamp_bbox(x1, y1, x2, y2, image_width, image_height, min_size=8):
    """Clamp an xyxy box to image bounds and preserve a minimum size."""
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(image_width - 1, int(x2))
    y2 = min(image_height - 1, int(y2))

    if x2 - x1 < min_size:
        center = (x1 + x2) // 2
        x1 = max(0, center - min_size // 2)
        x2 = min(image_width - 1, x1 + min_size)
    if y2 - y1 < min_size:
        center = (y1 + y2) // 2
        y1 = max(0, center - min_size // 2)
        y2 = min(image_height - 1, y1 + min_size)

    return x1, y1, x2, y2


def center_distance(cx1, cy1, cx2, cy2):
    """Calculate Euclidean distance between two center points."""
    return math.hypot(cx1 - cx2, cy1 - cy2)
