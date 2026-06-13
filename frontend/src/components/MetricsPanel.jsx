import { useState, useEffect } from 'react'
import { fetchMetrics } from '../api'

export default function MetricsPanel() {
  const [metrics, setMetrics] = useState(null)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    fetchMetrics()
      .then(setMetrics)
      .catch(() => {})
  }, [])

  if (!metrics) return null

  const hasMetrics = metrics.metrics && Object.keys(metrics.metrics).length > 0


  return (
    <div className="panel metrics-panel">
      <h3 onClick={() => setOpen(!open)} className="panel-toggle">
        📊 模型评估指标
        <span className={`arrow ${open ? 'open' : ''}`}>&#9662;</span>
      </h3>
      {open && (
        <div className="panel-content">
          {!hasMetrics ? (
            <p style={{ color: '#8b92a0', fontSize: '0.85rem', padding: '8px 0' }}>
              尚未完成训练或未生成评估指标
            </p>
          ) : (
            <>
          <p className="metric-dataset">验证集: {metrics.dataset}</p>
          <div className="metric-grid">
            {[
              ['Precision', metrics.metrics.precision],
              ['Recall', metrics.metrics.recall],
              ['mAP50', metrics.metrics.mAP50],
              ['mAP50-95', metrics.metrics.mAP50_95],
              ['F1', metrics.metrics.f1_score],
            ].filter(([, val]) => val != null && val > 0).map(([label, val]) => (
              <div key={label} className="metric-item">
                <div className="metric-header">
                  <span className="metric-label">{label}</span>
                  <span className="metric-value">{typeof val === 'number' ? val.toFixed(4) : val}</span>
                </div>
                <div className="metric-bar">
                  <div className="metric-fill" style={{ width: `${Math.min(val * 100, 100)}%` }} />
                </div>
              </div>
            ))}
            {/* 兼容旧版 SimpleCTDetector 指标 */}
            {metrics.metrics.train_loss != null && metrics.metrics.val_loss != null && (
              <div style={{ marginTop: 12, borderTop: '1px solid #2a3040', paddingTop: 8 }}>
                <h4 style={{ fontSize: '0.8rem', color: '#5a6070', marginBottom: 6 }}>训练指标</h4>
                {['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'].map(label => {
                  const val = metrics.metrics[label]
                  return val != null ? (
                    <div key={label} className="per-label-row">
                      <span className="pl-name">{label}:</span>
                      <span>{val.toFixed(4)}</span>
                    </div>
                  ) : null
                })}
              </div>
            )}
          </div>
          {metrics.note && (
            <div className="per-label" style={{ marginTop: 12 }}>
              <p style={{ fontSize: '0.72rem', color: '#5a6070', lineHeight: 1.5 }}>
                {metrics.note}
              </p>
            </div>
          )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
