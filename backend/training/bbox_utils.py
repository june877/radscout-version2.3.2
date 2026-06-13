"""
bbox_utils.py — YOLO bbox 工具集

YOLO 格式: [cx, cy, w, h] 归一化到 [0,1]（中心坐标 + 宽高）
xyxy 格式:  [x1, y1, x2, y2] 像素坐标（左上角 + 右下角）

所有函数同时支持 numpy array 和单个数值。
"""

import numpy as np


# ── 坐标转换 ──

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """
    YOLO normalized [cx,cy,w,h] → 像素 [x1,y1,x2,y2]

    参数:
        cx, cy, w, h: 归一化到 [0,1] 的中心坐标和宽高
        img_w, img_h:  图像像素宽高

    返回:
        (x1, y1, x2, y2) 像素坐标，左上右下
    """
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    像素 [x1,y1,x2,y2] → YOLO normalized [cx,cy,w,h]

    返回: (cx, cy, w, h) 归一化到 [0,1]
    """
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def yolo_to_xywh_pixels(cx, cy, w, h, img_w, img_h):
    """YOLO normalized → xywh 像素格式 (x=左上, y=左上, w=宽, h=高)"""
    x = (cx - w / 2) * img_w
    y = (cy - h / 2) * img_h
    pw = w * img_w
    ph = h * img_h
    return x, y, pw, ph


# ── IoU ──

def compute_iou(box1, box2):
    """
    计算两个 bbox 的 IoU (Intersection over Union)

    参数:
        box1, box2: (x1, y1, x2, y2) 像素坐标

    返回:
        float: IoU 值 [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_iou_batch(yolo_boxes, gt_boxes, img_w=512, img_h=512):
    """
    批量计算每个 YOLO pred box 与所有 GT 的最大 IoU

    参数:
        yolo_boxes: list of (cx, cy, w, h) 归一化
        gt_boxes:   list of (cx, cy, w, h) 归一化（同一张图的 GT）
        img_w, img_h: 图像尺寸

    返回:
        list of float: 每个 pred box 的最大 IoU
    """
    results = []
    for yb in yolo_boxes:
        px1, py1, px2, py2 = yolo_to_xyxy(*yb, img_w, img_h)
        max_iou = 0.0
        for gb in gt_boxes:
            gx1, gy1, gx2, gy2 = yolo_to_xyxy(*gb, img_w, img_h)
            iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            max_iou = max(max_iou, iou)
        results.append(max_iou)
    return results


# ── bbox 操作 ──

def expand_bbox(x1, y1, x2, y2, expand_ratio=1.5):
    """
    扩大 bbox（保持中心不变）

    参数:
        x1, y1, x2, y2: 像素坐标
        expand_ratio:    扩大倍数，1.0 不变，1.5 宽高各扩大 50%

    返回:
        (new_x1, new_y1, new_x2, new_y2)
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * expand_ratio
    h = (y2 - y1) * expand_ratio
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def clamp_bbox(x1, y1, x2, y2, img_w, img_h, min_size=8):
    """
    Clamp bbox 到图像范围内，确保宽高至少为 min_size

    返回:
        (clamped_x1, clamped_y1, clamped_x2, clamped_y2) 全部为 int
    """
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w - 1, int(x2))
    y2 = min(img_h - 1, int(y2))

    if x2 - x1 < min_size:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - min_size // 2)
        x2 = min(img_w - 1, x1 + min_size)
    if y2 - y1 < min_size:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - min_size // 2)
        y2 = min(img_h - 1, y1 + min_size)

    return x1, y1, x2, y2


def center_distance(cx1, cy1, cx2, cy2):
    """两个归一化中心点的欧氏距离"""
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


# ── 测试 ──
if __name__ == "__main__":
    # IoU 测试
    a = (10, 10, 50, 50)
    b = (30, 30, 70, 70)
    print(f"IoU({a}, {b}) = {compute_iou(a, b):.4f}")  # ~0.26

    # YOLO → xyxy 测试
    cx, cy, w, h = 0.5, 0.5, 0.2, 0.2
    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, 512, 512)
    print(f"yolo({cx}, {cy}, {w}, {h}) → xyxy({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    # 扩大
    ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, 1.5)
    print(f"expanded(1.5x): ({ex1:.0f}, {ey1:.0f}, {ex2:.0f}, {ey2:.0f})")

    # clamp
    cx1, cy1, cx2, cy2 = clamp_bbox(-5, -5, 520, 520, 512, 512)
    print(f"clamped: ({cx1}, {cy1}, {cx2}, {cy2})")

    print("All bbox_utils tests passed!")
