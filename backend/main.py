"""
RadScout v3.2 — FastAPI 后端
Two-stage detection: YOLO11s-2.5D + PNG-ResNet18-2.5D FP reduction

启动:
    cd backend
    D:/Python/python.exe main.py
    http://localhost:8000/docs
"""

import os, io, json, re
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BACKEND_DIR, "configs", "v32_inference_config.json")
os.environ.setdefault('YOLO_CONFIG_DIR', os.path.join(BACKEND_DIR, '.runtime', 'ultralytics'))

# ── Load config ──
with open(CONFIG_PATH, encoding='utf-8') as f:
    config = json.load(f)

YOLO_WEIGHTS = os.path.join(BACKEND_DIR, config["yolo_weights"])
RESNET_WEIGHTS = os.path.join(BACKEND_DIR, config["resnet_weights"])
if not os.path.isfile(RESNET_WEIGHTS) and config.get("fallback_resnet_weights"):
    fallback_path = os.path.join(BACKEND_DIR, config["fallback_resnet_weights"])
    if os.path.isfile(fallback_path):
        print(f"[v3.2] New ResNet checkpoint not found; using fallback: {fallback_path}")
        RESNET_WEIGHTS = fallback_path

DEMO_DIR = Path(BACKEND_DIR).parent / 'demo' / 'subset9_demo'
app = FastAPI(title="RadScout 三切片定位 API", version="3.3.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Init two-stage detector ──
from models.two_stage_inference import TwoStageDetector

detector = TwoStageDetector(
    yolo_weights=YOLO_WEIGHTS,
    resnet_weights=RESNET_WEIGHTS,
    device=config.get("device", "cuda"),
    yolo_conf=config["yolo_conf"],
    resnet_threshold=config["resnet_threshold"],
    imgsz=config["imgsz"],
    patch_size=config["patch_size"],
    expand_ratio=config["expand_ratio"],
    nms_iou_threshold=config.get("nms_iou_threshold", 0.5),
    containment_threshold=config.get("containment_threshold"),
    yolo_display_threshold=config.get('yolo_display_threshold'),
    merge_radius_mm=config.get('merge_radius_mm',3.0),
    expected_yolo_sha256=config.get('yolo_sha256'),
)


# ============================================================
@app.get("/api/health")
async def health():
    return {
        "status": "ok" if detector.yolo_loaded else "degraded",
        "device": detector.device,
        "load_errors": detector.load_errors,
        "model_name": detector.model_name,
        "yolo_loaded": detector.yolo_loaded,
        "resnet_loaded": detector.resnet_loaded,
        "config": {
            "yolo_conf": detector.yolo_conf,
            "resnet_threshold": detector.resnet_threshold,
            "default_mode": config["default_mode"],
            "yolo_weights":YOLO_WEIGHTS,
            "yolo_sha256":detector.yolo_sha256,
            "selected_epoch":config.get('selected_epoch'),
            "imgsz":detector.imgsz,
            "yolo_display_threshold":detector.yolo_display_threshold,
            "nms_iou_threshold":detector.nms_iou_threshold,
            "merge_radius_mm":detector.merge_radius_mm,
            "resnet_weights": RESNET_WEIGHTS,
        },
    }


@app.post("/api/v3/detect")
async def v3_detect(
    file: UploadFile = File(...),
    mode: str = Query(config['default_mode'], pattern="^(yolo_only|two_stage|all)$"),
    resnet_threshold: float | None = Query(None, ge=0, le=1),
    input_type: str = Query("yolo_25d_rgb", pattern="^(yolo_25d_rgb|ct_sequence)$"),
):
    """
    v3.2 二阶段检测 — 单张 2.5D RGB 图像 (R=z-1, G=z, B=z+1)

    mode: yolo_only | two_stage | all
    resnet_threshold: 不传则使用配置文件中的默认值
    input_type: yolo_25d_rgb (训练集2.5D RGB PNG) | ct_sequence (连续CT切片)
    """
    allowed = (".png", ".jpg", ".jpeg")
    if not file.filename.lower().endswith(allowed):
        return JSONResponse(status_code=400, content={"error": "仅支持 PNG/JPG"})

    image_bytes = await file.read()
    result = detector.detect_single(image_bytes, mode=mode, input_type=input_type, resnet_threshold=resnet_threshold)
    if "error" in result:
        return JSONResponse(status_code=503, content=result)
    return JSONResponse(content=result)


@app.post("/api/v3/detect-series")
async def v3_detect_series(
    files: list[UploadFile] = File(...),
    mode: str = Query(config['default_mode'], pattern="^(yolo_only|two_stage|all)$"),
    resnet_threshold: float | None = Query(None, ge=0, le=1),
    input_type: str = Query("ct_sequence", pattern="^(yolo_25d_rgb|ct_sequence)$"),
    metadata: str | None = Form(None),
):
    """v3.0 连续切片序列检测。input_type: ct_sequence (默认)"""
    if len(files) < 3:
        return JSONResponse(status_code=400,
                            content={"error": "至少需要 3 张连续切片（ct_sequence 模式）"})
    for f in files:
        if not f.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            return JSONResponse(status_code=400,
                                content={"error": f"不支持: {f.filename}"})

    try:
        geometry_metadata=json.loads(metadata) if metadata is not None else None
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={'error':'metadata 必须是有效的 JSON'})
    files_data = [(f.filename, await f.read()) for f in files]
    result = detector.detect_series(files_data, mode=mode, input_type=input_type,
                                    resnet_threshold=resnet_threshold, metadata=geometry_metadata)
    if "error" in result:
        return JSONResponse(status_code=503, content=result)
    return JSONResponse(content=result)


# ── Legacy compat ──
@app.post("/api/detect")
async def legacy_detect(file: UploadFile = File(...)):
    """兼容旧版单张检测 (默认 two_stage)"""
    return await v3_detect(file=file, mode=config['default_mode'], resnet_threshold=None, input_type="yolo_25d_rgb")


@app.post("/api/detect-series")
async def legacy_detect_series(files: list[UploadFile] = File(...)):
    """兼容旧版序列检测 (默认 two_stage)"""
    return await v3_detect_series(files=files, mode=config['default_mode'], resnet_threshold=None, input_type="ct_sequence", metadata=None)


@app.get('/api/demo')
async def get_demo():
    """A known demonstration case; references are returned only for display."""
    if not (DEMO_DIR/'metadata.json').is_file():
        return JSONResponse(status_code=404,content={'error':'未找到带标注演示包'})
    metadata=json.loads((DEMO_DIR/'metadata.json').read_text(encoding='utf-8'))
    reference=json.loads((DEMO_DIR/'reference/nodules.json').read_text(encoding='utf-8'))
    filenames=sorted(p.name for p in (DEMO_DIR/'images').glob('slice_*.png'))
    return {'metadata':metadata,'references':reference.get('nodules',[]),'filenames':filenames}


@app.get('/api/demo/images/{filename}')
async def get_demo_image(filename:str):
    if not re.fullmatch(r'slice_\d+\.png',filename):
        return JSONResponse(status_code=400,content={'error':'无效的演示切片文件名'})
    image=DEMO_DIR/'images'/filename
    if not image.is_file():
        return JSONResponse(status_code=404,content={'error':'演示切片不存在'})
    return FileResponse(image,media_type='image/png')


@app.get("/api/metrics")
async def get_metrics():
    """读取 YOLO 2.5D + ResNet18 指标"""
    result = {"yolo_metrics": None, "resnet_metrics": None, "two_stage_eval": None,
              "selected_epoch":config.get('selected_epoch'), 'legacy_resnet_comparison':True}

    yolo_path = os.path.join(BACKEND_DIR, "configs", "selected_model_validation.json")
    if os.path.isfile(yolo_path):
        with open(yolo_path,encoding='utf-8') as f:
            result["yolo_metrics"] = json.load(f)

    rn_path = os.path.join(BACKEND_DIR, "outputs", "fp_resnet18_v32", "metrics.json")
    if not os.path.isfile(rn_path):
        rn_path = os.path.join(BACKEND_DIR, "outputs", "fp_resnet18_png", "metrics.json")
    if os.path.isfile(rn_path):
        with open(rn_path) as f:
            result["resnet_metrics"] = json.load(f)

    ts_path = os.path.join(BACKEND_DIR, "outputs", "two_stage_eval_v321", "detection_eval_report.json")
    if not os.path.isfile(ts_path):
        ts_path = os.path.join(BACKEND_DIR, "outputs", "two_stage_eval_png", "detection_eval_report.json")
    if os.path.isfile(ts_path):
        with open(ts_path) as f:
            result["two_stage_eval"] = json.load(f)

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
