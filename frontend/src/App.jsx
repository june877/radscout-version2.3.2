import { useState } from 'react'
import ImageUploader from './components/ImageUploader'
import SeriesViewer from './components/SeriesViewer'
import MetricsPanel from './components/MetricsPanel'
import { detectSeries } from './api'

export default function App() {
  const [slices, setSlices] = useState(null)         // [{dataUrl, filename}]
  const [seriesResult, setSeriesResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFilesSelect = async (files) => {
    // 自然排序 + 创建 dataUrl
    const naturalSort = (a, b) => {
      const re = /(\d+)/g
      const aParts = a.name.match(re) || []
      const bParts = b.name.match(re) || []
      for (let i = 0; i < Math.max(aParts.length, bParts.length); i++) {
        const an = parseInt(aParts[i]) || 0
        const bn = parseInt(bParts[i]) || 0
        if (an !== bn) return an - bn
      }
      return a.name.localeCompare(b.name)
    }
    const sorted = [...files].sort(naturalSort)

    const sliceData = sorted.map((f) => ({
      dataUrl: URL.createObjectURL(f),
      filename: f.name,
    }))

    setSlices(sliceData)
    setSeriesResult(null)
    setError(null)
    setLoading(true)

    try {
      const result = await detectSeries(files)
      setSeriesResult(result)
    } catch (err) {
      setError(err.message || '检测请求失败，请确认后端服务已启动')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    if (slices) slices.forEach((s) => URL.revokeObjectURL(s.dataUrl))
    setSlices(null)
    setSeriesResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>
          <span className="logo-icon">🫁</span>
          RadScout v2.3
        </h1>
        <p className="subtitle">YOLO11s 3-Slice 2.5D — Continuous CT Slice Nodule Detection</p>
      </header>

      <main className="app-main">
        {!slices && (
          <div className="upload-section">
            <ImageUploader onFilesSelect={handleFilesSelect} disabled={loading} />
          </div>
        )}

        {error && (
          <div className="error-banner">
            <span>{error}</span>
            <button onClick={() => setError(null)}>&times;</button>
          </div>
        )}

        {loading && (
          <div className="loading-indicator">
            <div className="spinner" />
            <span>正在逐层分析 CT 切片序列（2.5D YOLO）...</span>
          </div>
        )}

        {slices && seriesResult && (
          <div className="results-layout">
            <div className="viewer-panel">
              <SeriesViewer
                slices={slices}
                detectionsBySlice={seriesResult.detections_by_slice}
                modelName={seriesResult.model_name}
                inferenceTime={seriesResult.inference_time_ms}
              />
              <button className="btn-reset" onClick={handleReset}>
                重新上传切片序列
              </button>
            </div>

            <div className="info-panel">
              <MetricsPanel />
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
