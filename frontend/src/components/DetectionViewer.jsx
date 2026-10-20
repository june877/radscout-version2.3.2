import { useRef, useEffect, useState } from 'react'
import { estimateBoxSize } from '../seriesUtils'

const LABEL_COLORS = { '肺结节 (Nodule)': '#FF3B30', 'Nodule': '#FF3B30' }

/**
 * v3.0 DetectionViewer — 单图二阶段检测结果展示
 *
 * Props:
 *   imageDataUrl: string
 *   allDetections: [det with keep/resnet_score/yolo_score/filter_reason]
 *   mode: "yolo_only" | "two_stage" | "all"
 *   modelName, inferenceTime, summary
 */
export default function DetectionViewer({ imageDataUrl, allDetections, mode, modelName, inferenceTime, summary }) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)

  const visibleDets = (allDetections || []).filter(d => {
    if (mode === 'yolo_only') return true
    if (mode === 'two_stage') return d.keep
    return true  // all
  }).map(d => ({
    ...d,
    _style: mode === 'all' && !d.keep ? 'dashed' : 'solid'
  }))

  useEffect(() => {
    const canvas = canvasRef.current; const container = containerRef.current
    if (!canvas || !container) return
    const img = new Image()
    img.onload = () => {
      const maxW = container.clientWidth; const scale = Math.min(1, maxW / img.naturalWidth)
      canvas.width = img.naturalWidth * scale; canvas.height = img.naturalHeight * scale
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      visibleDets.forEach(det => {
        const { x, y, width, height } = det.bbox
        const bx = x * scale, by = y * scale, bw = width * scale, bh = height * scale
        const color = det._style === 'dashed' ? '#666' : (LABEL_COLORS[det.label] || '#34C759')
        ctx.fillStyle = color + '18'; ctx.fillRect(bx, by, bw, bh)
        ctx.lineWidth = det._style === 'dashed' ? 1.5 : 2.5
        if (det._style === 'dashed') ctx.setLineDash([5, 3])
        ctx.strokeStyle = color; ctx.strokeRect(bx, by, bw, bh); ctx.setLineDash([])
        // Label
        const ypct = det.yolo_score != null ? (det.yolo_score * 100).toFixed(0) + '%' : (det.confidence * 100).toFixed(0) + '%'
        const rpct = det.resnet_score != null ? 'R:' + (det.resnet_score * 100).toFixed(0) + '%' : ''
        const label = ypct + (rpct ? ' ' + rpct : '')
        ctx.font = `bold ${Math.max(10, 11 * scale)}px sans-serif`
        const tm = ctx.measureText(label); const lx = bx, ly = by - 14 > 0 ? by - 14 : by + 2
        ctx.fillStyle = color; ctx.fillRect(lx, ly, tm.width + 8, 13)
        ctx.fillStyle = '#fff'; ctx.fillText(label, lx + 4, ly + 10)
      })
    }
    img.src = imageDataUrl
  }, [imageDataUrl, visibleDets])

  return (
    <div className="viewer">
      <div className="viewer-header">
        <span className="viewer-title">
          {visibleDets.length > 0 ? `${visibleDets.length} detections` : 'no nodule detected'}
        </span>
        <span className="viewer-meta">{modelName} · {inferenceTime?.toFixed(0)}ms</span>
      </div>
      <div className="canvas-container" ref={containerRef}>
        <canvas ref={canvasRef} style={{ maxWidth: '100%' }} />
      </div>

      {/* Summary */}
      {summary && (
        <div className="v3-summary-bar">
          <span>YOLO: {summary.raw_count} | Kept: {summary.kept_count} | Filtered: {summary.filtered_count}
            {summary.nms_removed_count != null ? ` | NMS: ${summary.nms_removed_count}` : ''}
          </span>
          <span style={{ color: '#5a6070', fontSize: '0.75rem' }}>
            {mode === 'yolo_only' ? 'YOLO only' : mode === 'two_stage' ? 'Two-stage' : 'All'}
          </span>
        </div>
      )}

      {visibleDets.length > 0 && (
        <ul className="detection-list">
          {visibleDets.map(det => (
            <li key={det.id} className={`detection-item ${det._style === 'dashed' ? 'filtered-out' : ''}`}>
              <span className="color-dot"
                style={{ background: det._style === 'dashed' ? '#555' : (LABEL_COLORS[det.label] || '#34C759') }} />
              <div className="det-info">
                <strong>{det.label} {det._style === 'dashed' ? '(filtered)' : ''}</strong>
                <span className="det-desc">
                  Y: {det.yolo_score != null ? (det.yolo_score * 100).toFixed(1) + '%' : '-'}
                  {' | '}R: {det.resnet_score != null ? (det.resnet_score * 100).toFixed(1) + '%' : '-'}
                  {' | '}Final: {det.final_score != null ? (det.final_score * 100).toFixed(1) + '%' : '-'}
                  {' | '}{det.filter_reason || ''}
                </span>
              </div>
              <div className="det-meta">
                <span className="det-confidence" style={det._style === 'dashed' ? { color: '#666' } : {}}>
                  {(det.yolo_score * 100).toFixed(0)}%
                </span>
                <span className="det-size">估计半径 {estimateBoxSize(det).radius.toFixed(2)} {estimateBoxSize(det).unit}</span>
              </div>
            </li>
          ))}
        </ul>
      )}

      {visibleDets.length === 0 && (
        <div style={{ padding: '16px', textAlign: 'center', color: '#5a6070', fontSize: '0.9rem' }}>
          no nodule detected
        </div>
      )}
    </div>
  )
}
