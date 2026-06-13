import { useRef, useEffect, useState, useMemo } from 'react'

const LABEL_COLORS = {
  '肺结节 (Nodule)': '#FF3B30',
  'nodule': '#FF3B30',
}

/**
 * SeriesViewer — 连续 CT 切片 2.5D 检测结果浏览器
 *
 * Props:
 *   slices: [{dataUrl, filename}] — 所有切片
 *   detectionsBySlice: {"0": [...], "1": [...]}
 *   modelName: string
 *   inferenceTime: number (ms)
 */
export default function SeriesViewer({ slices, detectionsBySlice, modelName, inferenceTime }) {
  const [currentIdx, setCurrentIdx] = useState(0)
  const canvasRef = useRef(null)
  const containerRef = useRef(null)

  const N = slices.length
  const currentDets = detectionsBySlice[String(currentIdx)] || []

  // 哪些层有检测
  const hasDetections = useMemo(() => {
    const set = new Set()
    Object.entries(detectionsBySlice).forEach(([key, dets]) => {
      if (dets.length > 0) set.add(Number(key))
    })
    return set
  }, [detectionsBySlice])

  // 初始化：跳到第一个有检测的层
  useEffect(() => {
    const first = [...hasDetections].sort((a, b) => a - b)[0]
    if (first != null) setCurrentIdx(first)
  }, [])

  // 画当前层 + bbox
  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const img = new Image()
    img.onload = () => {
      const maxW = container.clientWidth
      const scale = Math.min(1, maxW / img.naturalWidth)
      canvas.width = img.naturalWidth * scale
      canvas.height = img.naturalHeight * scale

      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

      currentDets.forEach((det) => {
        const { x, y, width, height } = det.bbox
        const bx = x * scale, by = y * scale
        const bw = width * scale, bh = height * scale
        const color = LABEL_COLORS[det.label] || '#34C759'

        ctx.fillStyle = color + '18'
        ctx.fillRect(bx, by, bw, bh)
        ctx.lineWidth = 2.5
        ctx.strokeStyle = color
        ctx.strokeRect(bx, by, bw, bh)

        // 角标
        const cs = Math.min(14, bw * 0.3, bh * 0.3)
        ctx.lineWidth = 2
        ;[[bx,by,1,0],[bx+bw,by,-1,0],[bx,by+bh,1,1],[bx+bw,by+bh,-1,1]]
          .forEach(([cx,cy,sx,sy]) => {
            ctx.beginPath()
            ctx.moveTo(cx, cy + (sy ? -cs : cs))
            ctx.lineTo(cx, cy)
            ctx.lineTo(cx + (sx < 0 ? -cs : cs), cy)
            ctx.stroke()
          })

        // 标签
        const pct = (det.confidence * 100).toFixed(0) + '%'
        ctx.font = `bold ${Math.max(11, 13*scale)}px sans-serif`
        const tm = ctx.measureText(pct)
        const lx = bx, ly = by - 19 > 0 ? by - 19 : by + 2
        ctx.fillStyle = color
        ctx.fillRect(lx, ly, tm.width + 10, 18)
        ctx.fillStyle = '#fff'
        ctx.textBaseline = 'middle'
        ctx.fillText(pct, lx + 5, ly + 9)
      })
    }
    img.src = slices[currentIdx]?.dataUrl
  }, [currentIdx, currentDets, slices])

  return (
    <div className="viewer">
      <div className="viewer-header">
        <span className="viewer-title">
          Slice {currentIdx + 1} / {N}
          {hasDetections.has(currentIdx)
            ? ` — ${currentDets.length} 处疑似病灶`
            : ' — 当前层未检测到疑似结节'}
        </span>
        <span className="viewer-meta">{modelName} · {inferenceTime?.toFixed(0)}ms</span>
      </div>

      <div className="canvas-container" ref={containerRef}>
        <canvas ref={canvasRef} style={{ maxWidth: '100%' }} />
      </div>

      {/* 滑动条 */}
      <div className="slider-container">
        <input
          type="range"
          min={0}
          max={N - 1}
          value={currentIdx}
          onChange={(e) => setCurrentIdx(Number(e.target.value))}
          className="slice-slider"
        />
        <div className="slider-ticks">
          {slices.map((_, i) => (
            <div
              key={i}
              className={`slider-tick ${i === currentIdx ? 'active' : ''} ${hasDetections.has(i) ? 'has-det' : ''}`}
              onClick={() => setCurrentIdx(i)}
            />
          ))}
        </div>
      </div>

      {/* 检测列表 */}
      {currentDets.length > 0 && (
        <ul className="detection-list">
          {currentDets.map((det) => (
            <li key={det.id} className="detection-item">
              <span className="color-dot"
                style={{ background: LABEL_COLORS[det.label] || '#34C759' }} />
              <div className="det-info">
                <strong>{det.label}</strong>
                <span className="det-desc">{det.description}</span>
              </div>
              <div className="det-meta">
                <span className="det-confidence">{(det.confidence * 100).toFixed(0)}%</span>
                <span className="det-size">
                  {det.size_mm?.width != null
                    ? `${det.size_mm.width} x ${det.size_mm.height} mm`
                    : '尺寸未知'}
                </span>
              </div>
            </li>
          ))}
        </ul>
      )}

      {currentDets.length === 0 && (
        <div style={{ padding: '16px', textAlign: 'center', color: '#5a6070', fontSize: '0.9rem' }}>
          当前层未检测到明显疑似结节
        </div>
      )}
    </div>
  )
}
