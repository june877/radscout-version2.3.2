"""
推理模块 — 加载训练好的 SimpleCTDetector 进行单张切片检测

用法:
    from models.inference import SimpleCTDetectorInference
    infer = SimpleCTDetectorInference(weights_path)
    result = infer.detect(image_bytes)

返回格式兼容前端 DetectionViewer。
"""

import os
import time
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn

# 把 backend 目录加入 path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.simple_detector import SimpleCTDetector


class SimpleCTDetectorInference:
    """
    推理器：加载模型权重，对单张上传的 CT 切片做检测

    处理流程:
      1. 接收 PNG/JPG 字节流
      2. PIL 解码 → 转灰度 → resize 512×512 → 归一化 [0,1]
      3. 输入模型
      4. sigmoid 得到结节概率
      5. 如果概率 > 阈值，还原 bbox 到原图坐标
      6. 返回兼容前端的 JSON 格式
    """

    def __init__(self, weights_path, device=None, threshold=0.5):
        """
        参数:
            weights_path: best_model.pt 路径
            device:       "cuda" / "cpu"，None 自动选择
            threshold:    分类阈值，sigmoid 概率 > threshold 才报告结节
        """
        self.threshold = threshold
        self.model_name = "RadScout-SimpleCTDetector-v2.1"
        self.model_loaded = False

        # 选择设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 加载模型
        if not os.path.isfile(weights_path):
            self.model_loaded = False
            self.model = None
            print(f"[Inference] 权重文件不存在: {weights_path}")
            return

        try:
            self.model = SimpleCTDetector().to(self.device)
            checkpoint = torch.load(weights_path, map_location=self.device)
            # 兼容两种保存格式
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.model_loaded = True
            print(f"[Inference] 模型已加载: {weights_path}")
            print(f"  val_accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
        except Exception as e:
            self.model_loaded = False
            self.model = None
            print(f"[Inference] 模型加载失败: {e}")

    def detect(self, image_bytes: bytes) -> dict:
        """
        对单张上传的 CT 切片做检测

        参数:
            image_bytes: PNG/JPG 文件字节流

        返回:
            dict，格式兼容前端 DetectionViewer
        """
        t0 = time.time()

        if not self.model_loaded:
            return {
                "model_name": self.model_name,
                "detections": [],
                "total_count": 0,
                "inference_time_ms": 0.0,
                "image_info": {
                    "format": "uploaded_2d_slice",
                    "note": "模型权重未加载，请先运行训练脚本。",
                },
                "error": "模型权重不存在，请先运行训练脚本生成 weights/best_model.pt",
            }

        # ── 步骤 1: 图像预处理 ──
        # PIL 解码
        image = Image.open(io.BytesIO(image_bytes))
        orig_w, orig_h = image.size                      # 原始尺寸

        # 转灰度（L 模式 = 单通道）
        if image.mode != "L":
            image = image.convert("L")

        # Resize 到 512×512
        image_resized = image.resize((512, 512), Image.BILINEAR)
        img_array = np.array(image_resized, dtype=np.float32) / 255.0   # [0, 1]

        # 转为 tensor [1, 1, 512, 512]
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # ── 步骤 2: 模型推理 ──
        with torch.no_grad():
            cls_logits, bbox_pred = self.model(img_tensor)

        cls_prob = torch.sigmoid(cls_logits).item()      # 结节概率
        bbox_norm = bbox_pred.squeeze().cpu().numpy()    # [cx, cy, w, h] 归一化

        inference_time = (time.time() - t0) * 1000

        # ── 步骤 3: 根据阈值构建检测结果 ──
        if cls_prob < self.threshold:
            return {
                "model_name": self.model_name,
                "detections": [],
                "total_count": 0,
                "inference_time_ms": round(inference_time, 1),
                "image_info": {
                    "format": "uploaded_2d_slice",
                    "note": f"No lesion detected (prob={cls_prob:.3f} < threshold={self.threshold})",
                },
            }

        # ── 步骤 4: bbox 还原到原图尺寸 ──
        # bbox_norm 是在 512×512 上的归一化坐标 [cx, cy, w, h]
        cx_norm, cy_norm, w_norm, h_norm = bbox_norm

        # 512×512 上的像素坐标
        cx_512 = cx_norm * 512
        cy_512 = cy_norm * 512
        w_512 = w_norm * 512
        h_512 = h_norm * 512

        # 转换回原图尺寸
        scale_w = orig_w / 512.0
        scale_h = orig_h / 512.0

        cx_orig = cx_512 * scale_w
        cy_orig = cy_512 * scale_h
        w_orig = w_512 * scale_w
        h_orig = h_512 * scale_h

        # 转为 xywh 格式（左上角坐标 + 宽高）
        x = int(cx_orig - w_orig / 2)
        y = int(cy_orig - h_orig / 2)
        w = int(w_orig)
        h = int(h_orig)

        # 边界保护
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)

        return {
            "model_name": self.model_name,
            "detections": [
                {
                    "id": 1,
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "label": "肺结节 (Nodule)",
                    "confidence": round(cls_prob, 4),
                    "size_mm": {"width": None, "height": None},  # 上传图片无 pixel spacing
                    "description": f"模型检测到疑似肺结节区域 (prob={cls_prob:.2f})",
                }
            ],
            "total_count": 1,
            "inference_time_ms": round(inference_time, 1),
            "image_info": {
                "format": "uploaded_2d_slice",
                "note": "Result generated by trained SimpleCTDetector model. "
                        "size_mm unavailable for uploaded 2D images without pixel spacing metadata.",
            },
        }


# 测试
if __name__ == "__main__":
    import os
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(backend_dir, "weights", "best_model.pt")

    infer = SimpleCTDetectorInference(weights_path)
    print(f"Model loaded: {infer.model_loaded}")
    print(f"Device: {infer.device}")

    # 生成一张假图测试
    dummy_img = Image.new("L", (512, 512), color=128)
    buf = io.BytesIO()
    dummy_img.save(buf, format="PNG")
    result = infer.detect(buf.getvalue())
    print(f"\nResult: {json.dumps(result, indent=2, ensure_ascii=False)}")
