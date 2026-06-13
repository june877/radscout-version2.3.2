"""
yolo_inference.py — YOLO11s 推理模块

加载训练好的 YOLO11s 权重，对上传的单张 CT 切片做检测。

兼容格式: PNG/JPG/JPEG
输出格式: 与前端 DetectionViewer 兼容
"""

import os
import sys
import io
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class YOLOInference:
    """YOLO11s 推理器"""

    def __init__(self, weights_path, device="cuda", conf_threshold=0.25):
        """
        参数:
            weights_path:   yolo11s_luna_best.pt 路径
            device:         "cuda" / "cpu"
            conf_threshold: YOLO 置信度阈值
        """
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.model_name = "RadScout-YOLO11s-v2.2"
        self.model_loaded = False
        self.model = None
        self.device_str = device

        if not os.path.isfile(weights_path):
            print(f"[YOLOInference] WARNING: weights not found: {weights_path}")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.model_loaded = True
            print(f"[YOLOInference] Model loaded: {weights_path}")
            print(f"[YOLOInference] Device: {device}")
        except ImportError:
            print("[YOLOInference] ERROR: ultralytics not installed.")
            print("  Run: D:/Python/python.exe -m pip install ultralytics")
        except Exception as e:
            print(f"[YOLOInference] ERROR loading model: {e}")

    def detect(self, image_bytes: bytes) -> dict:
        """
        对单张上传图像做检测。

        参数:
            image_bytes: PNG/JPG 文件字节流

        返回:
            dict: 兼容前端 DetectionViewer 格式
        """
        t0 = time.time()

        # ── 模型未加载 ──
        if not self.model_loaded:
            return {
                "model_name": self.model_name,
                "detections": [],
                "total_count": 0,
                "inference_time_ms": 0.0,
                "image_info": {
                    "format": "uploaded_2d_slice",
                    "note": "YOLO weights not loaded. Train the model first.",
                },
                "error": (
                    f"YOLO11s 权重不存在: {self.weights_path}。"
                    "请先运行 train_yolo.py 训练模型。"
                ),
            }

        # ── 图像预处理 ──
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            orig_w, orig_h = image.size
        except Exception as e:
            return {
                "model_name": self.model_name,
                "detections": [],
                "total_count": 0,
                "inference_time_ms": 0.0,
                "image_info": {"format": "uploaded_2d_slice"},
                "error": f"无法解析图像: {e}",
            }

        # ── YOLO 推理 ──
        try:
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                device=self.device_str,
                verbose=False,
            )
        except Exception as e:
            return {
                "model_name": self.model_name,
                "detections": [],
                "total_count": 0,
                "inference_time_ms": round((time.time() - t0) * 1000, 1),
                "image_info": {"format": "uploaded_2d_slice"},
                "error": f"YOLO inference error: {e}",
            }

        inference_ms = round((time.time() - t0) * 1000, 1)

        # ── 解析 YOLO 输出 ──
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    # YOLO 返回 xyxy 格式（归一化或像素取决于实现）
                    # Ultralytics 默认返回 xyxy 像素坐标
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    x1, y1, x2, y2 = xyxy
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)

                    # clamp 到图像范围
                    x = max(0, x)
                    y = max(0, y)
                    w = max(1, min(w, orig_w - x))
                    h = max(1, min(h, orig_h - y))

                    label = "肺结节 (Nodule)" if cls_id == 0 else f"未知类别 {cls_id}"

                    detections.append({
                        "id": i + 1,
                        "bbox": {"x": x, "y": y, "width": w, "height": h},
                        "label": label,
                        "confidence": round(conf, 4),
                        "size_mm": {"width": None, "height": None},
                        "description": f"YOLO11s 检测到疑似肺结节区域 (prob={conf:.2f})",
                    })

        return {
            "model_name": self.model_name,
            "detections": detections,
            "total_count": len(detections),
            "inference_time_ms": inference_ms,
            "image_info": {
                "format": "uploaded_2d_slice",
                "note": (
                    f"Result generated by YOLO11s 2D CT slice detector. "
                    f"Confidence threshold: {self.conf_threshold}. "
                    f"size_mm unavailable for uploaded images without pixel spacing."
                ),
            },
        }


# ============================================================
# v2.3 2.5D 推理器 (R=z-1, G=z, B=z+1)
# ============================================================

class YOLOInference25D:
    """
    YOLO11s 3-slice 2.5D 推理器。

    模型输入: R=z-1, G=z(中心层), B=z+1
    Web Demo 限制: 用户只上传单张图片，无法获得上下层。
    降级策略: 将上传图片复制三份作为 z-1/z/z+1。
    这意味着 Web demo 实际降级为 2D 推理，但权重来自 2.5D 训练。
    """

    def __init__(self, weights_path, device="cuda", conf_threshold=0.25):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.model_name = "RadScout-YOLO11s-2.5D-v2.3"
        self.model_loaded = False
        self.model = None
        self.device_str = device

        if not os.path.isfile(weights_path):
            print(f"[YOLO25D] WARNING: weights not found: {weights_path}")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.model_loaded = True
            print(f"[YOLO25D] Model loaded: {weights_path}")
            print(f"[YOLO25D] Input: 3-slice 2.5D (R=z-1, G=z, B=z+1)")
        except ImportError:
            print("[YOLO25D] ERROR: ultralytics not installed.")
        except Exception as e:
            print(f"[YOLO25D] ERROR: {e}")

    def detect(self, image_bytes: bytes) -> dict:
        t0 = time.time()

        if not self.model_loaded:
            return {
                "model_name": self.model_name,
                "detections": [], "total_count": 0,
                "inference_time_ms": 0.0,
                "image_info": {"format": "uploaded_2d_slice (replicated to 3-slice 2.5D)"},
                "error": "YOLO 2.5D weights not loaded. Train first.",
            }

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("L")
            orig_w, orig_h = image.size
        except Exception as e:
            return {
                "model_name": self.model_name, "detections": [],
                "total_count": 0, "inference_time_ms": 0.0,
                "image_info": {}, "error": f"Image error: {e}",
            }

        # 3-slice 2.5D: 将上传图片复制三份模拟 z-1/z/z+1
        # 这等效于使用 2.5D 权重做 2D 推理
        img_arr = np.array(image)
        rgb = np.stack([img_arr, img_arr, img_arr], axis=-1)
        rgb_pil = Image.fromarray(rgb)

        try:
            results = self.model.predict(
                rgb_pil, conf=self.conf_threshold,
                device=self.device_str, verbose=False,
            )
        except Exception as e:
            return {
                "model_name": self.model_name, "detections": [],
                "total_count": 0,
                "inference_time_ms": round((time.time()-t0)*1000, 1),
                "image_info": {}, "error": f"Inference error: {e}",
            }

        inference_ms = round((time.time() - t0) * 1000, 1)
        detections = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    x1, y1, x2, y2 = xyxy
                    x = max(0, int(x1))
                    y = max(0, int(y1))
                    w = max(1, int(x2-x1))
                    h = max(1, int(y2-y1))

                    label = "肺结节 (Nodule)" if cls_id == 0 else f"类别{cls_id}"

                    detections.append({
                        "id": i+1,
                        "bbox": {"x": x, "y": y, "width": w, "height": h},
                        "label": label,
                        "confidence": round(conf, 4),
                        "size_mm": {"width": None, "height": None},
                        "description": f"YOLO11s 2.5D 检测 (prob={conf:.2f})",
                    })

        return {
            "model_name": self.model_name,
            "detections": detections,
            "total_count": len(detections),
            "inference_time_ms": inference_ms,
            "image_info": {
                "format": "uploaded_2d_slice (replicated to 3-slice)",
                "note": (
                    "Web demo uses single-upload image replicated 3x. "
                    "Full 2.5D benefit requires CT volume with adjacent slices. "
                    f"Threshold: {self.conf_threshold}."
                ),
            },
        }
