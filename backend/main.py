"""
RadScout v2.3 — FastAPI 后端
YOLO11s 3-slice 2.5D (R=z-1, G=z, B=z+1) CT 肺结节检测

启动:
    cd backend
    python main.py
    然后访问 http://localhost:8000/docs 查看 API 文档
"""

import os, json, io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.yolo_inference import YOLOInference25D

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_25D_PATH = os.path.join(BACKEND_DIR, "weights", "yolo11s_25d_3slice_best.pt")
WEIGHTS_2D_PATH = os.path.join(BACKEND_DIR, "weights", "yolo11s_luna_best.pt")
METRICS_25D_PATH = os.path.join(BACKEND_DIR, "outputs", "yolo25d_3slice_metrics.json")
METRICS_2D_PATH = os.path.join(BACKEND_DIR, "outputs", "yolo_metrics.json")

app = FastAPI(
    title="RadScout v2.3 API",
    description="YOLO11s 3-slice 2.5D CT Nodule Detection — LUNA16 subset0-6",
    version="2.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ── 推理器: 2.5D 优先 ──
if os.path.isfile(WEIGHTS_25D_PATH):
    inference = YOLOInference25D(WEIGHTS_25D_PATH, device="cuda", conf_threshold=0.25)
elif os.path.isfile(WEIGHTS_2D_PATH):
    print("[main] 2.5D weights not found, falling back to v2.2 2D YOLO")
    from models.yolo_inference import YOLOInference as YOLO2D
    inference = YOLO2D(WEIGHTS_2D_PATH, device="cuda", conf_threshold=0.25)
else:
    inference = YOLOInference25D(WEIGHTS_25D_PATH, device="cuda", conf_threshold=0.25)


# ============================================================
# 辅助
# ============================================================

def _natural_sort_key(filename: str):
    """自然排序: 提取文件名中所有数字用于排序"""
    import re
    parts = re.split(r'(\d+)', filename)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _read_as_grayscale(file_bytes: bytes, target_size=None):
    """读取图像字节 → 灰度 numpy array [H, W]"""
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    if target_size:
        img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img).astype(np.float32), img.size


# ============================================================
# API
# ============================================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "model_name": inference.model_name,
        "yolo_model_loaded": inference.model_loaded,
        "device": inference.device_str,
        "weights_25d_exists": os.path.isfile(WEIGHTS_25D_PATH),
    }


@app.post("/api/detect-series")
async def detect_series(files: list[UploadFile] = File(...)):
    """
    ★ v2.3 主接口: 上传连续 CT 切片序列，逐层 2.5D 检测。

    要求至少 3 张 PNG/JPG，按文件名自然排序。
    每层构造 R=z-1, G=z, B=z+1 输入 YOLO11s。
    边缘层使用复制补齐。
    """
    if len(files) < 3:
        return JSONResponse(status_code=400, content={
            "error": f"2.5D 检测至少需要 3 张连续切片，当前仅 {len(files)} 张。"
        })

    # 校验文件类型
    allowed = (".png", ".jpg", ".jpeg")
    for f in files:
        if not f.filename.lower().endswith(allowed):
            return JSONResponse(status_code=400, content={
                "error": f"不支持的文件类型: {f.filename}，仅支持 PNG/JPG/JPEG"
            })

    # 按文件名自然排序
    sorted_files = sorted(files, key=lambda f: _natural_sort_key(f.filename))
    ordered_names = [f.filename for f in sorted_files]

    # 检查模型
    if not inference.model_loaded:
        return JSONResponse(status_code=503, content={
            "error": "YOLO 2.5D 权重不存在，请先训练 v2.3 模型。",
            "expected_path": WEIGHTS_25D_PATH,
            "model_loaded": False,
        })

    # 读取所有切片
    slices = []
    sizes = set()
    for f in sorted_files:
        content = await f.read()
        arr, (w, h) = _read_as_grayscale(content)
        slices.append(arr)
        sizes.add((w, h))

    if len(sizes) > 1:
        return JSONResponse(status_code=400, content={
            "error": f"所有切片尺寸必须一致，检测到 {len(sizes)} 种不同尺寸: {sizes}"
        })

    # ── 逐层 2.5D 推理（直接调 YOLO model.predict，不经过 detect() 避免 .convert("L")） ──
    import time
    t0 = time.time()
    N = len(slices)
    detections_by_slice = {}

    for i in range(N):
        # 构造 3-slice: R=z-1, G=z, B=z+1，边界复制补齐
        r = slices[i-1] if i > 0 else slices[0]
        g = slices[i]
        b = slices[i+1] if i < N-1 else slices[N-1]

        # 堆叠 RGB 直接给 YOLO
        rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
        rgb_pil = Image.fromarray(rgb, mode="RGB")

        # 直接调 YOLO model.predict——保留三通道（不经过 YOLOInference25D.detect 的 .convert("L")）
        if inference.model_loaded and inference.model is not None:
            results = inference.model.predict(
                rgb_pil, conf=inference.conf_threshold,
                device=inference.device_str, imgsz=512, verbose=False)
            dets = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    for j in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[j]
                        dets.append({
                            "id": j+1,
                            "bbox": {"x": int(x1), "y": int(y1), "width": int(x2-x1), "height": int(y2-y1)},
                            "label": "肺结节 (Nodule)",
                            "confidence": round(float(confs[j]), 4),
                            "size_mm": {"width": None, "height": None},
                            "description": f"2.5D YOLO (prob={float(confs[j]):.2f})",
                        })
            detections_by_slice[str(i)] = dets
        else:
            detections_by_slice[str(i)] = []

    elapsed = (time.time() - t0) * 1000

    return JSONResponse(content={
        "model_name": inference.model_name,
        "slice_count": N,
        "ordered_filenames": ordered_names,
        "detections_by_slice": detections_by_slice,
        "inference_time_ms": round(elapsed, 1),
        "note": "2.5D inference (R=z-1, G=z, B=z+1) using adjacent uploaded slices.",
    })


@app.post("/api/detect")
async def detect_lesions(file: UploadFile = File(...)):
    """
    单张检测（保留用于调试，v2.3 主功能为 /api/detect-series）
    """
    allowed_ext = (".png", ".jpg", ".jpeg")
    if not file.filename.lower().endswith(allowed_ext):
        return JSONResponse(status_code=400,
                            content={"error": "不支持的文件类型，请上传 PNG/JPG"})

    if not inference.model_loaded:
        return JSONResponse(status_code=503, content={
            "error": "模型权重未加载，请先训练",
            "model_loaded": False,
        })

    try:
        image_bytes = await file.read()
        result = inference.detect(image_bytes)
        if "error" in result:
            return JSONResponse(status_code=503, content=result)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": f"检测失败: {str(e)}"})


@app.get("/api/metrics")
async def get_metrics():
    metrics_file = None
    for p in [METRICS_25D_PATH, METRICS_2D_PATH]:
        if os.path.isfile(p):
            metrics_file = p
            break
    if metrics_file is None:
        return JSONResponse(content={
            "model_name": inference.model_name,
            "dataset": "LUNA16 subset0-6",
            "metrics": {},
            "note": "尚未完成训练。请运行 train_yolo25d.py。",
        })
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": f"读取 metrics 失败: {str(e)}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
