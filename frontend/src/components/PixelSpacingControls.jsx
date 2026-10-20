import { useEffect, useState } from 'react'
import { effectivePixelSpacing, parseSizeCalibration } from '../seriesUtils'

export default function PixelSpacingControls({ calibration, onChange, imageDims }) {
  const [x, setX] = useState(''), [y, setY] = useState(''), [error, setError] = useState('')
  useEffect(() => {
    const spacing = effectivePixelSpacing(calibration, imageDims)
    setX(spacing ? String(spacing[0]) : ''); setY(spacing ? String(spacing[1]) : '')
  }, [calibration, imageDims])

  function apply(event) {
    event.preventDefault()
    const spacing = [Number(x), Number(y)]
    if (!spacing.every(value => Number.isFinite(value) && value > 0)) {
      setError('请填写两个大于 0 的像素间距'); return
    }
    onChange({ spacing, nativeSize: null, source: '手动填写的上传图像间距' }); setError('')
  }
  async function readFile(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    try {
      onChange(parseSizeCalibration(await file.text(), file.name)); setError('')
    } catch (err) { setError(err.message) }
  }
  const spacing = effectivePixelSpacing(calibration, imageDims)
  return <section className="spacing-panel">
    <h3>换算为毫米</h3>
    <p>导入当前病例的 CT 头文件或元数据，也可填写所上传图像的像素间距。导入原始宽高用于整幅切片缩放；局部截取图请手动填写当前间距。</p>
    <form onSubmit={apply} className="spacing-form">
      <label>水平方向（mm/px）<input aria-label="水平方向像素间距" type="number" min="0.000001" step="any"
        value={x} onChange={event => setX(event.target.value)} required /></label>
      <label>垂直方向（mm/px）<input aria-label="垂直方向像素间距" type="number" min="0.000001" step="any"
        value={y} onChange={event => setY(event.target.value)} required /></label>
      <button type="submit">换算成 mm</button>
    </form>
    <div className="spacing-actions"><label className="secondary-button">导入 .mhd / 元数据 JSON
      <input type="file" accept=".mhd,.json" onChange={readFile} hidden /></label>
      {calibration && <button onClick={() => onChange(null)} type="button">清除间距</button>}</div>
    {spacing && <p className="spacing-status">已按 {spacing[0].toFixed(6)} × {spacing[1].toFixed(6)} mm/px 换算 · {calibration.source}
      {calibration.nativeSize ? ' · 已按原始与上传图像的宽高比例调整' : ''}</p>}
    {!spacing && <p className="spacing-status">像素间距可从原始 CT 的 PixelSpacing 或 ElementSpacing 获取。仅凭 PNG 图像和检测框无法确定毫米长度。</p>}
    {error && <p className="spacing-error" role="alert">{error}</p>}
  </section>
}
