"""YOLO11s 2.5D candidate detection and ResNet18 false-positive reduction."""
import io, os, re, time, hashlib
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from training.bbox_utils import expand_bbox, clamp_bbox, compute_iou
from models.fp_resnet18 import load_fp_resnet18_checkpoint
from ct_preprocessing import stack_uint8_context, crop_classifier
from series_geometry import validate_geometry, attach_geometry, group_yolo_candidates

def apply_nms(detections, iou_threshold=0.5, score_key="final_score", containment_threshold=None):
    ordered = sorted(detections, key=lambda d: d.get(score_key) or 0.0, reverse=True)
    kept = []
    for d in ordered:
        a = d["bbox"]
        ba = (a["x"], a["y"], a["x"]+a["width"], a["y"]+a["height"])
        suppress = False
        for k in kept:
            b = k["bbox"]
            bb = (b["x"], b["y"], b["x"]+b["width"], b["y"]+b["height"])
            overlap = compute_iou(ba, bb)
            intersection = max(0,min(ba[2],bb[2])-max(ba[0],bb[0])) * max(0,min(ba[3],bb[3])-max(ba[1],bb[1]))
            small_area = min(a["width"]*a["height"],b["width"]*b["height"])
            contained = containment_threshold is not None and small_area > 0 and intersection/small_area >= containment_threshold
            if overlap >= iou_threshold or contained:
                suppress = True
                break
        if not suppress:
            kept.append(d)
    return kept, len(detections)-len(kept)

class TwoStageDetector:
    def __init__(self, yolo_weights, resnet_weights, device="cuda", yolo_conf=0.10,
                 resnet_threshold=0.25, imgsz=512, patch_size=96, expand_ratio=1.5,
                 nms_iou_threshold=0.5, containment_threshold=None,
                 yolo_display_threshold=None, merge_radius_mm=3.0, expected_yolo_sha256=None):
        self.device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        self.yolo_conf, self.resnet_threshold = yolo_conf, resnet_threshold
        self.imgsz, self.patch_size, self.expand_ratio = imgsz, patch_size, expand_ratio
        self.nms_iou_threshold, self.containment_threshold = nms_iou_threshold, containment_threshold
        self.yolo_display_threshold = yolo_conf if yolo_display_threshold is None else yolo_display_threshold
        self.merge_radius_mm = merge_radius_mm
        self.model_name = "YOLO11s 三切片 · 第90轮"
        self.yolo_weights = os.path.abspath(yolo_weights)
        self.yolo_sha256 = None
        self.yolo = self.resnet = None
        self.yolo_loaded = self.resnet_loaded = False
        self.load_errors = []
        self.resnet_weights = os.path.abspath(resnet_weights)
        try:
            if not os.path.isfile(yolo_weights): raise FileNotFoundError(yolo_weights)
            with open(yolo_weights, 'rb') as source:
                self.yolo_sha256 = hashlib.sha256(source.read()).hexdigest()
            if expected_yolo_sha256 and self.yolo_sha256 != expected_yolo_sha256:
                raise ValueError('YOLO checkpoint checksum does not match the frozen model')
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_weights)
            self.yolo_loaded = True
        except Exception as exc: self.load_errors.append("YOLO: " + str(exc))
        try:
            if not os.path.isfile(resnet_weights): raise FileNotFoundError(resnet_weights)
            self.resnet, self.checkpoint = load_fp_resnet18_checkpoint(resnet_weights, torch.device(self.device))
            self.resnet_loaded = True
        except Exception as exc: self.load_errors.append("ResNet: " + str(exc))

    def _error(self, message):
        return {"model_name": self.model_name, "mode":"error","error":message,
                "raw_detections":[],"filtered_detections":[],"all_detections":[],
                "summary":{"raw_count":0,"kept_count":0,"filtered_count":0,"nms_removed_count":0},
                "inference_time_ms":0}

    def detect_single(self, image_bytes, mode="yolo_only", input_type="yolo_25d_rgb", resnet_threshold=None):
        threshold = self.resnet_threshold if resnet_threshold is None else resnet_threshold
        if mode not in ("yolo_only","two_stage","all"): return self._error("Unknown detection mode")
        if not 0 <= threshold <= 1: return self._error("ResNet threshold must be between 0 and 1")
        if input_type != "yolo_25d_rgb": return self._error("Single image requires yolo_25d_rgb input")
        if not self.yolo_loaded: return self._error("YOLO weights not loaded")
        if mode != "yolo_only" and not self.resnet_loaded: return self._error("ResNet weights not loaded")
        started = time.perf_counter()
        try:
            if isinstance(image_bytes, Image.Image):
                img = image_bytes.copy().convert("RGB")
            else:
                img = Image.open(io.BytesIO(image_bytes) if isinstance(image_bytes, bytes) else image_bytes).convert("RGB")
            # PIL preserves R=z-1, G=z, B=z+1. Match NMS with evaluation.
            rs = self.yolo.predict(img, conf=self.yolo_conf, iou=self.nms_iou_threshold,
                                   device=self.device, imgsz=self.imgsz, half=False,
                                   rect=False, max_det=300, verbose=False)[0]
            xyxy = rs.boxes.xyxy.cpu().numpy() if rs.boxes is not None else []
            confs = rs.boxes.conf.cpu().numpy() if rs.boxes is not None else []
        except Exception as exc: return self._error("Image/YOLO error: " + str(exc))
        raw = []
        for i,(b,score) in enumerate(zip(xyxy,confs)):
            # Keep subpixel coordinates so evaluation and production crop identically.
            x1,y1,x2,y2 = map(float,b)
            raw.append({"id":i+1,"bbox":{"x":x1,"y":y1,"width":x2-x1,"height":y2-y1},
                        "label":"Nodule","yolo_score":float(score),"resnet_score":None,
                        "final_score":float(score),"keep":float(score)>=self.yolo_display_threshold,
                        "filter_reason":"passed" if float(score)>=self.yolo_display_threshold else "below_yolo_threshold",
                        "size_mm":{"width":None,"height":None},"description":"YOLO candidate"})
        if mode != "yolo_only" and raw:
            patches, indices = [], []
            for i,d in enumerate(raw):
                b=d["bbox"]
                ex=clamp_bbox(*expand_bbox(b["x"],b["y"],b["x"]+b["width"],b["y"]+b["height"],self.expand_ratio),
                              img.width,img.height,min_size=16)
                crop=crop_classifier(img, ex, self.patch_size)
                patches.append(np.asarray(crop,dtype=np.float32).transpose(2,0,1)/255)
                indices.append(i)
            try:
                with torch.inference_mode():
                    scores=self.resnet.predict(torch.from_numpy(np.stack(patches)).to(self.device)).cpu().tolist()
            except Exception as exc: return self._error("ResNet inference error: "+str(exc))
            for i,score in zip(indices,scores):
                d=raw[i]; d["resnet_score"]=score; d["final_score"]=d["yolo_score"]*score
                d["keep"]=score>=threshold; d["filter_reason"]="passed" if d["keep"] else "low_resnet_score"
        accepted=[d for d in raw if d["keep"]]
        kept,removed=apply_nms(accepted,self.nms_iou_threshold,"final_score",self.containment_threshold)
        kept_ids={d["id"] for d in kept}
        for d in accepted:
            if d["id"] not in kept_ids:
                d["keep"]=False; d["filter_reason"]="nms_suppressed"
        return {"model_name":self.model_name,"mode":mode,"input_type":input_type,"resnet_threshold":threshold,
                "nms_iou_threshold":self.nms_iou_threshold,"containment_threshold":self.containment_threshold,
                "raw_detections":raw,"filtered_detections":kept,"all_detections":raw,
                "summary":{"raw_count":len(raw),"kept_count":len(kept),
                           "filtered_count":len(raw)-len(kept),"nms_removed_count":removed},
                "inference_time_ms":round((time.perf_counter()-started)*1000,1),
                "image_dims":{"width":img.width,"height":img.height}}

    def detect_series(self, files_data, mode="yolo_only", input_type="ct_sequence", resnet_threshold=None, metadata=None):
        if input_type != "ct_sequence": return self._error("Series requires ct_sequence input")
        if not self.yolo_loaded: return self._error("YOLO weights not loaded")
        if len(files_data)<3: return self._error("At least three consecutive slices are required")
        names=[name for name,_ in files_data]
        if len(names)!=len(set(names)): return self._error("Duplicate filenames are not supported")
        def natural_key(item):
            return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)",item[0])]
        ordered=sorted(files_data,key=natural_key)
        try:
            images=[]
            for name,data in ordered:
                source=Image.open(io.BytesIO(data))
                rgb=np.asarray(source.convert("RGB"),dtype=np.int16)
                chroma=float(np.abs(rgb[:,:,0]-rgb[:,:,1]).mean()+np.abs(rgb[:,:,1]-rgb[:,:,2]).mean())
                if chroma>2.0:
                    return self._error("序列输入需要连续灰度 CT 切片；"+name+" 是彩色图。请勿将已合成的三通道训练 PNG 当作灰度序列上传。")
                images.append(source.convert("L"))
            indexes=[re.fullmatch(r"(?:slice[_-]?)?(\d+)",os.path.splitext(name)[0],re.IGNORECASE) for name,_ in ordered]
            if all(indexes):
                numbers=[int(m.group(1)) for m in indexes]
                if any(b-a!=1 for a,b in zip(numbers,numbers[1:])):
                    return self._error("切片编号不连续；请上传同一序列的连续切片。")
            if len({im.size for im in images})!=1: return self._error("All slices must have identical dimensions")
            slices=[np.asarray(im) for im in images]
            geometry=validate_geometry(metadata, [name for name,_ in ordered], images[0].size)
        except Exception as exc: return self._error("Invalid image in series: "+str(exc))
        started=time.perf_counter(); n=len(slices); all_by={}; by={}; raw_total=removed_total=0
        for z in range(n):
            context=[max(0,z-1),z,min(n-1,z+1)]
            # Exported CT/demo slices are already 512 square. Preserve the
            # uploaded dimensions for ordinary PNG so returned boxes stay in
            # the viewer's coordinate system; YOLO handles its own imgsz resize.
            rgb=stack_uint8_context(slices, z)
            result=self.detect_single(rgb,mode,"yolo_25d_rgb",resnet_threshold)
            if "error" in result: return result
            for d in result["all_detections"]:
                d["slice_index"]=z; d["context_slices"]=context
                d['id']=f'{z}:{d["id"]}'
                attach_geometry(d, geometry)
            by[str(z)]=result["filtered_detections"]
            all_by[str(z)]=result["all_detections"]
            raw_total+=result["summary"]["raw_count"]; removed_total+=result["summary"]["nms_removed_count"]
        candidates=[]; group_count=None
        if mode=='yolo_only' and geometry is not None:
            candidates,group_count=group_yolo_candidates(all_by,self.yolo_display_threshold,self.merge_radius_mm)
            by={str(z):[] for z in range(n)}
            for d in candidates: by[str(d['slice_index'])].append(d)
        else:
            candidates=[d for values in by.values() for d in values]
        return {"model_name":self.model_name,"mode":mode,"input_type":input_type,"slice_count":n,
                "geometry":geometry, "candidates":candidates,
                "yolo_display_threshold":self.yolo_display_threshold,
                "grouping_applied":group_count is not None,
                "candidate_group_count":group_count,
                "resnet_threshold":self.resnet_threshold if resnet_threshold is None else resnet_threshold,
                "nms_iou_threshold":self.nms_iou_threshold,"containment_threshold":self.containment_threshold,
                "ordered_filenames":[name for name,_ in ordered],
                "image_dims":{"width":images[0].width,"height":images[0].height},
                "detections_by_slice":by,"all_by_slice":all_by,
                "summary":{"total_slices":n,"total_yolo_raw":raw_total,
                           "total_kept":sum(map(len,by.values())),"total_nms_removed":removed_total},
                "inference_time_ms":round((time.perf_counter()-started)*1000,1),
                "image_info":{"format":"ct_sequence 2.5D","note":"Counts are per-slice candidates, not distinct 3D nodules."}}
