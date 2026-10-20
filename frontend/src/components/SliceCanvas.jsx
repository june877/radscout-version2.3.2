import { useEffect, useRef } from 'react'
import { hitTest, PREDICTION_COLOR, REFERENCE_COLOR, REGION_COLOR } from '../seriesUtils'

export default function SliceCanvas({ slice, predictions = [], references = [], region = null,
  crop = null, onSelect, label = 'CT 切片' }) {
  const canvasRef = useRef(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !slice) return
    let active = true
    const image = new Image()
    image.onload = () => {
      if (!active) return
      const view = crop || { x: 0, y: 0, width: image.naturalWidth, height: image.naturalHeight }
      canvas.width = crop ? 384 : image.naturalWidth
      canvas.height = crop ? 384 : image.naturalHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(image, view.x, view.y, view.width, view.height, 0, 0, canvas.width, canvas.height)
      const sx = canvas.width / view.width, sy = canvas.height / view.height
      const mark = (bbox, color, dashed = false, center = false) => {
        const x = (bbox.x - view.x) * sx, y = (bbox.y - view.y) * sy
        const w = bbox.width * sx, h = bbox.height * sy
        ctx.strokeStyle = color; ctx.lineWidth = crop ? 2 : 2.5
        ctx.setLineDash(dashed ? [7, 5] : [])
        ctx.strokeRect(x, y, w, h); ctx.setLineDash([])
        if (center) {
          const cx = x + w / 2, cy = y + h / 2
          ctx.beginPath(); ctx.moveTo(cx - 5, cy); ctx.lineTo(cx + 5, cy)
          ctx.moveTo(cx, cy - 5); ctx.lineTo(cx, cy + 5); ctx.stroke()
        }
      }
      predictions.forEach(d => mark(d.bbox, d.keep === false ? '#89909d' : PREDICTION_COLOR, d.keep === false, true))
      references.forEach(d => mark(d.bbox, REFERENCE_COLOR, false, true))
      if (region && !crop) mark(region, REGION_COLOR, true)
    }
    image.onerror = () => {
      if (!active) return
      canvas.width = 384; canvas.height = 384
      const ctx = canvas.getContext('2d')
      ctx.fillStyle = '#111'; ctx.fillRect(0, 0, 384, 384)
      ctx.fillStyle = '#ddd'; ctx.font = '16px sans-serif'; ctx.fillText('切片加载失败', 120, 190)
    }
    image.src = slice.dataUrl
    return () => { active = false }
  }, [slice, predictions, references, region, crop])

  function handleClick(event) {
    if (!onSelect || crop) return
    const canvas = canvasRef.current, rect = canvas.getBoundingClientRect()
    const x = (event.clientX - rect.left) * canvas.width / rect.width
    const y = (event.clientY - rect.top) * canvas.height / rect.height
    const prediction = hitTest(predictions, x, y), reference = hitTest(references, x, y)
    if (prediction) onSelect({ kind: 'prediction', target: prediction })
    else if (reference) onSelect({ kind: 'reference', target: reference })
  }
  return <canvas ref={canvasRef} className={'slice-canvas ' + (onSelect ? 'clickable' : '')}
    onClick={handleClick} aria-label={label} />
}
