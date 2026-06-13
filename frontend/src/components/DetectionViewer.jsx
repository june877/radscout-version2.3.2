import { useRef, useEffect, useState } from 'react'

const LABEL_COLORS = {
  '肺结节 (Nodule)': '#FF3B30',
  '肿块 (Mass)': '#FF9500',
  '钙化 (Calcification)': '#007AFF',
}

export default function DetectionViewer({ imageDataUrl, detections, modelName, inferenceTime }) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const [activeId, setActiveId] = useState(null)

  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (!canvas || !container) return

      const maxW = container.clientWidth
      const scale = Math.min(1, maxW / img.naturalWidth)
      canvas.width = img.naturalWidth * scale
      canvas.height = img.naturalHeight * scale

      const ctx = canvas.getContext('2d')
      ctx.imageSmoothingEnabled = true
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

      detections.forEach((det) => {
        const { x, y, width, height } = det.bbox
        const bx = x * scale
        const by = y * scale
        const bw = width * scale
        const bh = height * scale
        const color = LABEL_COLORS[det.label] || '#34C759'
        const isActive = activeId === det.id

        // 半透明填充
        ctx.fillStyle = color + (isActive ? '28' : '12')
        ctx.fillRect(bx, by, bw, bh)

        // 边框
        ctx.lineWidth = isActive ? 3 : 2
        ctx.strokeStyle = color
        ctx.strokeRect(bx, by, bw, bh)

        // 四角准星
        const cs = Math.min(14, bw * 0.3, bh * 0.3)
        ctx.lineWidth = isActive ? 2.5 : 2
        drawCorner(ctx, bx, by, cs, 0)           // 左上
        drawCorner(ctx, bx + bw, by, cs, 1)       // 右上
        drawCorner(ctx, bx, by + bh, cs, 2)       // 左下
        drawCorner(ctx, bx + bw, by + bh, cs, 3)  // 右下

        // 标签
        const shortLabel = det.label.includes('结节')
          ? 'Nodule' : det.label.includes('肿块')
          ? 'Mass' : det.label.includes('钙化')
          ? 'Calc' : det.label
        const labelText = `${shortLabel} ${(det.confidence * 100).toFixed(0)}%`
        ctx.font = `${isActive ? 'bold ' : ''}${Math.max(11, 12.5 * scale)}px "Segoe UI", "PingFang SC", sans-serif`
        const tm = ctx.measureText(labelText)
        const tw = tm.width
        const th = Math.max(15, 17 * scale)
        const lx = bx
        const ly = by - th - 3 > 0 ? by - th - 3 : by + 2

        const padX = 8
        const padY = 3
        const rx = 5
        ctx.beginPath()
        ctx.moveTo(lx + rx, ly)
        ctx.lineTo(lx + tw + padX * 2 - rx, ly)
        ctx.arcTo(lx + tw + padX * 2, ly, lx + tw + padX * 2, ly + rx, rx)
        ctx.lineTo(lx + tw + padX * 2, ly + th + padY * 2 - rx)
        ctx.arcTo(lx + tw + padX * 2, ly + th + padY * 2, lx + tw + padX * 2 - rx, ly + th + padY * 2, rx)
        ctx.lineTo(lx + rx, ly + th + padY * 2)
        ctx.arcTo(lx, ly + th + padY * 2, lx, ly + th + padY * 2 - rx, rx)
        ctx.lineTo(lx, ly + rx)
        ctx.arcTo(lx, ly, lx + rx, ly, rx)
        ctx.closePath()
        ctx.fillStyle = color
        ctx.fill()
        ctx.fillStyle = '#FFFFFF'
        ctx.textBaseline = 'middle'
        ctx.fillText(labelText, lx + padX, ly + padY + th / 2)
      })
    }
    img.src = imageDataUrl
  }, [imageDataUrl, detections, activeId])

  return (
    <div className="viewer">
      <div className="viewer-header">
        <span className="viewer-title">
          检测结果 — {detections.length > 0
            ? `共 ${detections.length} 处疑似病灶`
            : '未检测到明显疑似结节'}
        </span>
        <span className="viewer-meta">{modelName} · 推理 {inferenceTime?.toFixed(0) || '—'} ms</span>
      </div>
      <div className="canvas-container" ref={containerRef}>
        <canvas ref={canvasRef} style={{ maxWidth: '100%', cursor: 'pointer' }} />
        {detections.length === 0 && (
          <div style={{
            position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
            color: '#5a6070', fontSize: '0.9rem', pointerEvents: 'none',
          }}>
            未检测到明显疑似结节
          </div>
        )}
      </div>
      <ul className="detection-list">
        {detections.map((det) => {
          const sizeW = det.size_mm?.width
          const sizeH = det.size_mm?.height
          const sizeText = (sizeW != null && sizeH != null)
            ? `${sizeW} × ${sizeH} mm`
            : '尺寸未知'
          return (
          <li
            key={det.id}
            className={`detection-item ${activeId === det.id ? 'active' : ''}`}
            onMouseEnter={() => setActiveId(det.id)}
            onMouseLeave={() => setActiveId(null)}
          >
            <span
              className="color-dot"
              style={{ background: LABEL_COLORS[det.label] || '#34C759' }}
            />
            <div className="det-info">
              <strong>{det.label}</strong>
              <span className="det-desc">{det.description}</span>
            </div>
            <div className="det-meta">
              <span className="det-confidence">{(det.confidence * 100).toFixed(0)}%</span>
              <span className="det-size">{sizeText}</span>
            </div>
          </li>
          )
        })}
      </ul>
    </div>
  )
}

function drawCorner(ctx, cx, cy, size, dir) {
  ctx.beginPath()
  switch (dir) {
    case 0: ctx.moveTo(cx, cy + size); ctx.lineTo(cx, cy); ctx.lineTo(cx + size, cy); break
    case 1: ctx.moveTo(cx - size, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + size); break
    case 2: ctx.moveTo(cx, cy - size); ctx.lineTo(cx, cy); ctx.lineTo(cx + size, cy); break
    case 3: ctx.moveTo(cx - size, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy - size); break
  }
  ctx.stroke()
}
