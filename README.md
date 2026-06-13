# RadScout v2.2 — YOLO11s CT Nodule Detection Prototype

基于 LUNA16 subset0-6 训练的 YOLO11s 2D CT 肺结节检测原型。

> **版本演进**: v2.1 使用 SimpleCTDetector（0.1M 参数轻量 CNN）作为 baseline；v2.2 升级为 YOLO11s（9.4M 参数），原生支持多框目标检测。当前系统是本科项目/作品集原型，不是临床系统。

## Current Status

### 已完成

| 功能 | 状态 | 说明 |
|------|:----:|------|
| LUNA16 subset0-6 数据加载 | ✅ | 7 个子集，428 个带标注 CT volume，852 个结节 |
| YOLO 数据集构建 | ✅ | .mhd/.raw → PNG + YOLO label 格式 |
| YOLO11s 训练 | ✅ | Ultralytics YOLO，GPU/AMP 支持 |
| YOLO11s 推理 | ✅ | 多框检测，前端兼容格式 |
| FastAPI 后端 | ✅ | `/api/health`, `/api/detect`, `/api/metrics` |
| React 前端 | ✅ | 上传、多框可视化、指标面板、免责声明 |

### 尚未完成

| 功能 | 状态 | 说明 |
|------|:----:|------|
| 全部 LUNA16 子集 | ❌ | 仅 subset0-6（subset7-9 需另外下载） |
| 3D CT volume 推理 | ❌ | 仅单张 2D 轴向切片 |
| DICOM 直接上传 | ❌ | 仅 PNG/JPG |
| size_mm 物理尺寸 | ❌ | 上传图片无 pixel spacing，显示"尺寸未知" |
| 临床级验证 | ❌ | 教学/作品集原型 |

## 项目结构

```
version2.2/
  backend/
    main.py                        # FastAPI 入口（v2.2：YOLO11s 推理）
    requirements.txt               # Python 依赖
    models/
      yolo_inference.py            # YOLO11s 推理器（v2.2 新增）
      inference.py                 # v2.1 SimpleCTDetector 推理（保留）
      simple_detector.py           # v2.1 SimpleCTDetector 模型（保留）
    training/
      build_yolo_dataset.py        # YOLO 数据集构建脚本（v2.2 新增）
      train_yolo.py                # YOLO11s 训练脚本（v2.2 新增）
      train.py                     # v2.1 SimpleCTDetector 训练（保留）
      luna_dataset.py              # v2.1 数据集（保留）
    tools/
      check_gpu.py                 # GPU 环境检查
    weights/
      yolo11s_luna_best.pt         # YOLO11s 最佳权重（训练后生成）
    outputs/
      yolo_metrics.json            # YOLO 评估指标（训练后生成）
      cache_slices/                # v2.1 缓存（保留）
    runs/
      yolo_luna/                   # YOLO 训练输出
  datasets/
    luna_yolo/                     # YOLO 格式数据集
      data.yaml                    # Ultralytics 配置
      images/  train/  val/        # PNG 切片
      labels/  train/  val/        # YOLO label txt
      debug/   train/  val/        # bbox 标注 debug 图
  frontend/                        # React + Vite 前端
  README.md
```

## 启动方式

### 0. 安装依赖

```bash
cd backend
D:/Python/python.exe -m pip install -r requirements.txt
D:/Python/python.exe -m pip install ultralytics
```

前端：

```bash
cd frontend
npm install
```

### 1. 构建 YOLO 数据集

```bash
cd backend
D:/Python/python.exe training/build_yolo_dataset.py \
    --data_dir D:/python_gjw/radscout2/data/luna16 \
    --out_dir ../datasets/luna_yolo \
    --subsets 0 1 2 3 4 5 6 \
    --img_size 512 \
    --neg_ratio 2 \
    --debug_count 100
```

输出：`datasets/luna_yolo/`（data.yaml + images + labels + debug）

### 2. 训练 YOLO11s

```bash
cd backend
D:/Python/python.exe training/train_yolo.py \
    --model yolo11s.pt \
    --data ../datasets/luna_yolo/data.yaml \
    --epochs 100 --imgsz 640 --batch 8 --device 0 --workers 2 --amp
```

如果显存不足：

```bash
D:/Python/python.exe training/train_yolo.py \
    --model yolo11s.pt \
    --data ../datasets/luna_yolo/data.yaml \
    --epochs 100 --imgsz 640 --batch 4 --device 0 --workers 2 --amp
```

训练输出：
- `weights/yolo11s_luna_best.pt` — 最佳模型权重
- `outputs/yolo_metrics.json` — 评估指标
- `runs/yolo_luna/` — 完整训练日志（Ultralytics 格式）

### 3. 启动后端

```bash
cd backend
D:/Python/python.exe main.py
```

后端运行在 http://localhost:8000。

### 4. 启动前端

```bash
cd frontend
npm run dev
```

前端运行在 http://localhost:5173。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 + YOLO 模型加载状态 |
| POST | `/api/detect` | 上传图像，YOLO11s 多框检测 |
| GET | `/api/metrics` | 读取 `outputs/yolo_metrics.json` |

## 评估指标说明

v2.2 使用 YOLO 标准目标检测指标（**不使用普通 accuracy**）：

| 指标 | 含义 |
|------|------|
| **Precision** | 模型预测为结节的框中，有多少是真的 |
| **Recall** (Sensitivity) | 真实结节中，模型找出了多少 |
| **mAP50** | IoU >= 0.5 时的 mean Average Precision |
| **mAP50-95** | IoU 0.5 到 0.95 平均的 mAP（更严格） |
| **F1 Score** | Precision 和 Recall 的调和平均 |

## 当前限制

- 只做 2D CT 切片检测，不是完整 3D CT volume 检测
- 不支持 DICOM 直接上传（需先转 PNG/JPG）
- bbox 物理尺寸 `size_mm` 对上传 PNG/JPG 为 null（无 pixel spacing）
- 当前仅训练 subset0-6，非全量 LUNA16
- 教具/作品集原型，不具备临床诊断意义
