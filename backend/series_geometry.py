"""Display geometry and annotation-independent candidate grouping for PNG series."""
import re
import numpy as np
from scipy.spatial import cKDTree


def validate_geometry(metadata, filenames, image_size):
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError('空间元数据必须是 JSON 对象')
    def vector(key, length):
        value = np.asarray(metadata.get(key), dtype=float)
        if value.shape != (length,) or not np.isfinite(value).all():
            raise ValueError(f'空间元数据中的 {key} 无效')
        return value
    size = vector('size_xyz', 3)
    spacing = vector('spacing_mm', 3)
    origin = vector('origin_mm', 3)
    direction = vector('direction', 9).reshape(3, 3)
    export = vector('export_size', 2)
    scale = vector('scale_xy', 2)
    if np.any(size <= 0) or not np.all(size == np.floor(size)) or np.any(spacing <= 0):
        raise ValueError('原始尺寸与像素间距必须为正值')
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError('方向矩阵必须为有效的正交矩阵')
    if not np.allclose(export, image_size) or not np.allclose(scale, export / size[:2]):
        raise ValueError('元数据中的导出尺寸或缩放比例与上传图像不一致')
    if metadata.get('center_index_base', 0) != 0:
        raise ValueError('原始层号须使用从 0 开始的编号')
    mapping = metadata.get('slice_mapping')
    if mapping is not None:
        if not isinstance(mapping, dict) or any(name not in mapping for name in filenames):
            raise ValueError('slice_mapping 必须包含每张上传图像的原始层号')
        zs = [mapping[name] for name in filenames]
    else:
        start = metadata.get('sequence_start_original_z')
        matches = [re.fullmatch(r'(?:slice[_-]?)?(\d+)\.(?:png|jpe?g)', name, re.I) for name in filenames]
        if not isinstance(start, int) or not all(matches):
            raise ValueError('无法确定原始层号；请提供 slice_mapping 或 slice_编号 格式及序列起始层号')
        zs = [start + int(match[1]) for match in matches]
    if any(not isinstance(z, int) or not 0 <= z < size[2] for z in zs):
        raise ValueError('上传图像的原始层号超出体积范围')
    if any(b-a != 1 for a, b in zip(zs, zs[1:])):
        raise ValueError('元数据对应的原始 CT 层号不连续')
    return {'seriesuid':str(metadata.get('seriesuid', '')), 'size_xyz':size.astype(int).tolist(),
            'spacing_mm':spacing.tolist(), 'origin_mm':origin.tolist(),
            'direction':direction.ravel().tolist(), 'export_size':export.astype(int).tolist(),
            'scale_xy':scale.tolist(), 'original_z_by_slice':zs,
            'complete_volume':zs == list(range(int(size[2]))),
            'hu_window':metadata.get('hu_window'), 'center_index_base':0}


def attach_geometry(detection, geometry):
    bbox = detection['bbox']
    cx, cy = bbox['x']+bbox['width']/2, bbox['y']+bbox['height']/2
    detection['center_px'] = [cx, cy]
    if geometry is None:
        return
    sx, sy = geometry['scale_xy']
    voxel = np.array([cx/sx-.5, cy/sy-.5,
                      geometry['original_z_by_slice'][detection['slice_index']]])
    spacing = np.array(geometry['spacing_mm'])
    world = np.array(geometry['origin_mm']) + np.array(geometry['direction']).reshape(3,3) @ (voxel*spacing)
    detection.update(original_z=int(voxel[2]), voxel_center=voxel.tolist(), world_mm=world.tolist(),
                     size_mm={'width':bbox['width']/sx*spacing[0], 'height':bbox['height']/sy*spacing[1]})


def group_yolo_candidates(all_by_slice, score_threshold, radius_mm=3.0):
    """Same stable YOLO-seed radius rule as frozen evaluation; no annotations."""
    candidates = [d for detections in all_by_slice.values() for d in detections]
    if not candidates:
        return [], 0
    coordinates = np.array([d['world_mm'] for d in candidates])
    tree = cKDTree(coordinates)
    assigned = np.zeros(len(candidates), dtype=bool)
    order = np.argsort([-d['yolo_score'] for d in candidates], kind='stable')
    groups = []
    for index in order:
        if assigned[index]:
            continue
        members = np.sort([i for i in tree.query_ball_point(coordinates[index], radius_mm) if not assigned[i]])
        assigned[members] = True
        group_id = len(groups)+1
        for member in members:
            candidate = candidates[member]
            candidate['group_id'] = group_id
            candidate['keep'] = False
            candidate['filter_reason'] = 'merged_context_candidate' if member != index else 'below_yolo_threshold'
        seed = candidates[index]
        seed['keep'] = seed['yolo_score'] >= score_threshold
        seed['filter_reason'] = 'passed' if seed['keep'] else 'below_yolo_threshold'
        seed['member_count'] = len(members)
        seed['first_original_z'] = min(candidates[i]['original_z'] for i in members)
        seed['last_original_z'] = max(candidates[i]['original_z'] for i in members)
        groups.append(seed)
    return [d for d in groups if d['keep']], len(groups)
