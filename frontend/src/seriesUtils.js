export const PREDICTION_COLOR = '#ff9c45'
export const REFERENCE_COLOR = '#47c9e8'
export const REGION_COLOR = '#ffd86a'

export function parseSizeCalibration(text, filename) {
  let spacing, nativeSize = null
  if (/\.mhd$/i.test(filename)) {
    const fields = Object.fromEntries(text.split(/\r?\n/).filter(line => line.includes('='))
      .map(line => { const i = line.indexOf('='); return [line.slice(0, i).trim(), line.slice(i + 1).trim()] }))
    spacing = (fields.ElementSpacing || '').split(/\s+/).map(Number).slice(0, 2)
    if (fields.DimSize) nativeSize = fields.DimSize.split(/\s+/).map(Number).slice(0, 2)
  } else {
    const metadata = JSON.parse(text)
    if (Array.isArray(metadata.spacing_mm)) spacing = metadata.spacing_mm.slice(0, 2)
    else if (Array.isArray(metadata.pixel_spacing_mm)) spacing = metadata.pixel_spacing_mm.slice(0, 2)
    else if (Array.isArray(metadata.PixelSpacing)) spacing = [Number(metadata.PixelSpacing[1]), Number(metadata.PixelSpacing[0])]
    if (Array.isArray(metadata.size_xyz)) nativeSize = metadata.size_xyz.slice(0, 2)
    else if (metadata.Columns && metadata.Rows) nativeSize = [Number(metadata.Columns), Number(metadata.Rows)]
  }
  if (!Array.isArray(spacing) || spacing.length !== 2 || !spacing.every(value => Number.isFinite(value) && value > 0)) {
    throw new Error('文件中缺少有效的像素间距。请提供 ElementSpacing、spacing_mm 或 PixelSpacing。')
  }
  if (nativeSize && (nativeSize.length !== 2 || !nativeSize.every(value => Number.isInteger(value) && value > 0))) {
    throw new Error('原始 CT 的宽高像素数无效')
  }
  return { spacing, nativeSize, source: filename }
}

export function effectivePixelSpacing(calibration, imageDims) {
  if (!calibration) return null
  return calibration.spacing.map((spacing, axis) => {
    const exported = axis === 0 ? imageDims.width : imageDims.height
    return calibration.nativeSize ? spacing * calibration.nativeSize[axis] / exported : spacing
  })
}

export function calibrateDetectionSize(detection, calibration, imageDims) {
  if (!detection._manual_size_calibration && Number.isFinite(detection.size_mm?.width) && Number.isFinite(detection.size_mm?.height)) return detection
  const spacing = effectivePixelSpacing(calibration, imageDims)
  return { ...detection, _manual_size_calibration: Boolean(spacing), size_mm: spacing ? {
    width: detection.bbox.width * spacing[0], height: detection.bbox.height * spacing[1],
  } : { width: null, height: null } }
}

export function estimateBoxSize(detection) {
  const measured = detection.size_mm
  const hasSpacing = Number.isFinite(measured?.width) && measured.width > 0
    && Number.isFinite(measured?.height) && measured.height > 0
  const width = hasSpacing ? measured.width : detection.bbox.width
  const height = hasSpacing ? measured.height : detection.bbox.height
  const diameter = Math.max(width, height)
  return { diameter, radius: diameter / 2, unit: hasSpacing ? 'mm' : 'px' }
}

export function contextSlices(center, count) {
  return [-1, 0, 1].map(offset => ({
    index: Math.max(0, Math.min(count - 1, center + offset)),
    offset, copied: center + offset < 0 || center + offset >= count,
  }))
}

export function hitTest(items, x, y) {
  return [...items].sort((a, b) => a.bbox.width * a.bbox.height - b.bbox.width * b.bbox.height)
    .find(({ bbox: b }) => x >= b.x && x <= b.x + b.width && y >= b.y && y <= b.y + b.height)
}

export function regionFor(bbox, width, height) {
  const size = Math.min(width, height, Math.max(96, bbox.width * 2.2, bbox.height * 2.2))
  return { x: Math.max(0, Math.min(width - size, bbox.x + bbox.width / 2 - size / 2)),
    y: Math.max(0, Math.min(height - size, bbox.y + bbox.height / 2 - size / 2)), width: size, height: size }
}

export function originalZ(geometry, index) {
  return geometry?.original_z_by_slice?.[index] ?? null
}

export function layerLabel(geometry, index) {
  const z = originalZ(geometry, index)
  return z == null ? '序列第 ' + (index + 1) + ' 层' : '原始 z=' + z + ' · 序列第 ' + (index + 1) + ' 层'
}

export function makeReferences(references, geometry) {
  if (!geometry) return []
  const { spacing_mm: spacing, origin_mm: origin, direction: d, scale_xy: scale } = geometry
  return references.map((item, index) => {
    const world = item.world_mm
    let voxel = item.voxel_continuous
    if (!voxel && Array.isArray(world) && world.length === 3) {
      const delta = world.map((v, i) => v - origin[i])
      voxel = [0, 1, 2].map(axis => [0, 1, 2].reduce((sum, row) => sum + d[row * 3 + axis] * delta[row], 0) / spacing[axis])
    }
    if (!Array.isArray(voxel) || voxel.length !== 3 || !voxel.every(Number.isFinite)
      || !Number.isFinite(item.diameter_mm) || item.diameter_mm <= 0) return null
    const calculatedWorld = [0, 1, 2].map(row => origin[row] + voxel.reduce((sum, v, axis) => sum + d[row * 3 + axis] * v * spacing[axis], 0))
    if (Array.isArray(world) && (world.length !== 3 || !world.every(Number.isFinite)
      || calculatedWorld.some((v, i) => Math.abs(v - world[i]) > 0.1))) return null
    return { ...item, id: 'reference-' + (item.annotation_id ?? index + 1), kind: 'reference',
      label: '参考结节 ' + (index + 1), voxel_center: voxel, world_mm: calculatedWorld,
      center_px: [(voxel[0] + 0.5) * scale[0], (voxel[1] + 0.5) * scale[1]],
      slice_index: geometry.original_z_by_slice.indexOf(Math.round(voxel[2])) }
  }).filter(Boolean)
}

export function referenceBoxesAt(references, geometry, sliceIndex) {
  if (!geometry) return []
  const z = originalZ(geometry, sliceIndex)
  return references.flatMap(item => {
    const dz = (z - item.voxel_center[2]) * geometry.spacing_mm[2]
    const radius = item.diameter_mm / 2
    if (Math.abs(dz) >= radius) return []
    const sectionRadius = Math.sqrt(radius * radius - dz * dz)
    const rx = sectionRadius / geometry.spacing_mm[0] * geometry.scale_xy[0]
    const ry = sectionRadius / geometry.spacing_mm[1] * geometry.scale_xy[1]
    return [{ ...item, bbox: { x: item.center_px[0] - rx, y: item.center_px[1] - ry, width: 2 * rx, height: 2 * ry } }]
  })
}

export function formatVector(values, digits = 2) {
  return values?.map(v => Number(v).toFixed(digits)).join(', ') ?? '未知'
}
