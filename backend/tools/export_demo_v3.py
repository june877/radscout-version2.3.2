"""
export_demo_v3.py — 保存 v3.0 二阶段检测演示结果

用法:
    cd D:/python_gjw/radscout2/version3.0/backend
    D:/Python/python.exe tools/export_demo_v3.py \
        --input D:/python_gjw/radscout2/version2.3/test_series/subset9_case/slice_040.png \
        --out_dir outputs/demo_v3
"""

import os, sys, json, io, argparse
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.two_stage_inference import TwoStageDetector

# v2.3 bbox utils
sys.path.insert(0, r"D:/python_gjw/radscout2/version2.3/backend")


def draw_boxes(img, dets, dashed_ids=None):
    """在 RGB PIL 图像上画检测框。dashed_ids 用虚线。"""
    if dashed_ids is None: dashed_ids = set()
    draw = ImageDraw.Draw(img)
    for d in dets:
        b = d["bbox"]
        x1, y1, x2, y2 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
        is_dashed = d.get("keep") is False or d["id"] in dashed_ids
        color = (128, 128, 128) if is_dashed else (255, 60, 48)
        if is_dashed:
            # draw dashed manually
            import math
            dash_len = 8; gap = 4
            for side in [(x1, y1, x2, y1), (x1, y1, x1, y2), (x1, y2, x2, y2), (x2, y1, x2, y2)]:
                sx, sy, ex, ey = side
                length = math.hypot(ex - sx, ey - sy)
                if length == 0: continue
                steps = int(length / (dash_len + gap))
                for s in range(steps):
                    t0 = s * (dash_len + gap) / length
                    t1 = t0 + dash_len / length
                    draw.line([(sx + (ex - sx) * t0, sy + (ey - sy) * t0),
                               (sx + (ex - sx) * min(t1, 1), sy + (ey - sy) * min(t1, 1))],
                              fill=color, width=2)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # label
        label = f"Y:{d.get('yolo_score', 0) * 100:.0f}%"
        if d.get("resnet_score") is not None:
            label += f" R:{d['resnet_score'] * 100:.0f}%"
        if d.get("final_score") is not None:
            label += f" F:{d['final_score'] * 100:.0f}%"
        draw.text((x1, y1 - 16 if y1 > 16 else y1 + 2), label, fill=color)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Input image path")
    ap.add_argument("--out_dir", type=str, default="outputs/demo_v3")
    ap.add_argument("--yolo_weights", type=str, default="weights/yolo11s_25d_3slice_best.pt")
    ap.add_argument("--resnet_weights", type=str, default="outputs/fp_resnet18/best_fp_resnet18.pt")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yw = os.path.join(backend_dir, args.yolo_weights)
    rw = os.path.join(backend_dir, args.resnet_weights)

    detector = TwoStageDetector(yw, rw, device="cuda")

    with open(args.input, "rb") as f:
        img_bytes = f.read()

    base_img = Image.open(args.input)
    if base_img.mode != 'RGB': base_img = base_img.convert('RGB')
    base_img.save(os.path.join(args.out_dir, "demo_input.png"))

    results = {}
    for mode in ["yolo_only", "two_stage", "all"]:
        result = detector.detect_single(img_bytes, mode=mode if mode != "all" else "two_stage")
        if mode == "all":
            result["mode"] = "all"
            result["filtered_detections"] = [d for d in result.get("all_detections", []) if d.get("keep")]
        results[mode] = result

        # Draw
        img_out = base_img.copy()
        all_dets = result.get("all_detections", result.get("raw_detections", []))
        filtered_ids = {d["id"] for d in result.get("filtered_detections", [])}
        if mode == "yolo_only":
            draw_boxes(img_out, all_dets)
        elif mode == "two_stage":
            draw_boxes(img_out, [d for d in all_dets if d.get("keep")])
        else:  # all
            draw_boxes(img_out, all_dets, dashed_ids={d["id"] for d in all_dets if not d.get("keep")})
        img_out.save(os.path.join(args.out_dir, f"demo_result_{mode}.png"))

    # Save JSON
    with open(os.path.join(args.out_dir, "demo_result.json"), "w") as f:
        json.dump({m: {"summary": v["summary"], "detections": v.get("all_detections", v.get("raw_detections", []))}
                   for m, v in results.items()}, f, indent=2, ensure_ascii=False)

    print(f"Demo saved to {args.out_dir}/")
    for m in ["yolo_only", "two_stage", "all"]:
        s = results[m]["summary"]
        print(f"  {m}: raw={s['raw_count']} kept={s['kept_count']} filtered={s['filtered_count']}")


if __name__ == "__main__":
    main()
