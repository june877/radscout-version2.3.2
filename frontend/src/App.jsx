import { useEffect, useRef, useState } from 'react'
import ImageUploader from './components/ImageUploader'
import SeriesViewer from './components/SeriesViewer'
import { detectSeriesV3, loadDemo } from './api'

export default function App() {
  const [slices, setSlices] = useState(null)
  const [seriesResult, setSeriesResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingText, setLoadingText] = useState('')
  const [error, setError] = useState(null)
  const [mode, setMode] = useState('yolo_only')
  const [resnetTh, setResnetTh] = useState(0.25)
  const [metadata, setMetadata] = useState(null)
  const [references, setReferences] = useState([])
  const slicesRef = useRef(null), requestId = useRef(0), abortRef = useRef(null)
  useEffect(() => () => {
    abortRef.current?.abort()
    slicesRef.current?.forEach(s => URL.revokeObjectURL(s.dataUrl))
  }, [])

  function replaceSlices(next) {
    slicesRef.current?.forEach(s => URL.revokeObjectURL(s.dataUrl))
    slicesRef.current = next; setSlices(next)
  }
  async function handleFilesSelect(files, selectedMetadata = metadata, selectedReferences = references) {
    const id = ++requestId.current
    abortRef.current?.abort()
    const controller = new AbortController(); abortRef.current = controller
    const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name, 'en', { numeric: true }))
    const previews = sorted.map(file => ({ dataUrl: URL.createObjectURL(file), filename: file.name }))
    replaceSlices(previews); setSeriesResult(null); setError(null); setLoading(true)
    setMetadata(selectedMetadata); setReferences(selectedReferences)
    setLoadingText('正在使用第 90 轮主模型检测连续切片…')
    try {
      const result = await detectSeriesV3(files, mode, mode === 'yolo_only' ? null : resnetTh, selectedMetadata, controller.signal)
      if (id !== requestId.current) return
      const byName = new Map(previews.map(slice => [slice.filename, slice]))
      const ordered = result.ordered_filenames.map(name => byName.get(name))
      if (ordered.some(slice => !slice)) throw new Error('服务器返回的切片文件名与上传文件不一致')
      slicesRef.current = ordered; setSlices(ordered); setSeriesResult(result)
    } catch (err) {
      if (id === requestId.current && err.name !== 'AbortError') setError(err.message || '检测失败')
    } finally {
      if (id === requestId.current) setLoading(false)
    }
  }

  async function handleDemo() {
    const id = ++requestId.current
    abortRef.current?.abort()
    const controller = new AbortController(); abortRef.current = controller
    setLoading(true); setError(null); setLoadingText('正在读取带标注演示病例…')
    try {
      const demo = await loadDemo(controller.signal, (done, total) => {
        if (id === requestId.current) setLoadingText('正在读取演示切片 ' + done + ' / ' + total)
      })
      if (id !== requestId.current) return
      await handleFilesSelect(demo.files, demo.metadata, demo.references)
    } catch (err) {
      if (id === requestId.current && err.name !== 'AbortError') { setError(err.message); setLoading(false) }
    }
  }

  async function handleJsonFiles(event) {
    const selected = Array.from(event.target.files || [])
    event.target.value = ''
    try {
      let meta = metadata, refs = references
      for (const file of selected) {
        const data = JSON.parse(await file.text())
        if (data.spacing_mm && data.origin_mm && data.direction) meta = data
        else if (Array.isArray(data.nodules)) refs = data.nodules
        else if (Array.isArray(data)) refs = data
        else throw new Error(file.name + ' 不是空间元数据或结节标注文件')
      }
      setMetadata(meta); setReferences(refs); setError(null)
    } catch (err) { setError('JSON 文件读取失败：' + err.message) }
  }
  function handleReset() {
    ++requestId.current; abortRef.current?.abort()
    replaceSlices(null); setSeriesResult(null); setError(null); setLoading(false)
    setMetadata(null); setReferences([])
  }

  return (
    <div className="app">
      <header className="app-header"><p className="eyebrow">CT 肺结节定位</p><h1>RadScout 三切片视图</h1>
        <p className="subtitle">连续灰度序列 · YOLO11s 第90轮 · 点击目标查看相邻三层</p></header>
      <main className="app-main">
        {!slices && !loading && <section className="upload-workspace">
          <button className="demo-button" onClick={handleDemo}>打开带标注演示病例</button>
          <p className="demo-description">已知病例包含完整灰度序列、空间信息与结节参考标注。</p>
          <div className="upload-section"><ImageUploader onFilesSelect={handleFilesSelect} disabled={loading} /></div>
          <div className="metadata-controls"><label className="secondary-button">
            选择元数据与标注 JSON（可选）
            <input type="file" accept=".json,application/json" multiple onChange={handleJsonFiles} hidden /></label>
            <span>{metadata ? '已加载空间元数据' : '未加载空间元数据'} · {references.length} 个参考标注</span>
          </div>
          <p className="upload-note">普通切片可直接检测，检测后可导入原 CT 的 .mhd / 元数据，或填写像素间距，将尺寸换算为 mm。显示参考标注时，请先选择对应病例的 metadata.json 和 nodules.json。</p>
          <div className="model-controls"><button className={'mode-button ' + (mode === 'yolo_only' ? 'active' : '')}
            onClick={() => setMode('yolo_only')}>YOLO 主模型</button>
            <details><summary>ResNet 扩展对照</summary>
              <p>沿用旧网页分类与裁剪流程，用于工程比较。</p>
              {['two_stage', 'all'].map(value => <button key={value} className={'mode-button ' + (mode === value ? 'active' : '')}
                onClick={() => setMode(value)}>{value === 'two_stage' ? '二阶段保留候选' : '全部候选（含已排除）'}</button>)}
              <label>ResNet 门槛 <input type="number" value={resnetTh} min="0" max="1" step="0.05"
                onChange={e => setResnetTh(Number(e.target.value))} /></label>
            </details>
          </div>
          <p className="selected-mode">当前模式：{mode === 'yolo_only' ? 'YOLO 主模型' : mode === 'two_stage' ? '二阶段扩展对照' : '扩展全部候选'}</p>
        </section>}
        {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError(null)} aria-label="关闭错误提示">×</button></div>}
        {loading && <div className="loading-indicator" role="status"><div className="spinner" /><span>{loadingText}</span></div>}
        {slices && seriesResult && !loading && <SeriesViewer slices={slices} result={seriesResult} references={references} mode={mode} />}
        {(slices || loading) && <button className="btn-reset" onClick={handleReset}>{loading ? '取消当前显示，重新选择' : '重新选择病例'}</button>}
      </main>
    </div>
  )
}
