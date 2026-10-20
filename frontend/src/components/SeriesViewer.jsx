import { useState, useMemo } from 'react'
import SliceCanvas from './SliceCanvas'
import ThreeSliceDialog from './ThreeSliceDialog'
import PixelSpacingControls from './PixelSpacingControls'
import { makeReferences, referenceBoxesAt, layerLabel, formatVector, estimateBoxSize, calibrateDetectionSize } from '../seriesUtils'

export default function SeriesViewer({ slices, result, references: rawReferences = [], mode }) {
  const geometry = result.geometry
  const [calibration, setCalibration] = useState(null)
  const references = useMemo(() => makeReferences(rawReferences, geometry), [rawReferences, geometry])
  const rawBySlice = mode === 'all' ? result.all_by_slice : result.detections_by_slice
  const bySlice = useMemo(() => Object.fromEntries(Object.entries(rawBySlice || {}).map(([key, values]) =>
    [key, values.map(det => calibrateDetectionSize(det, calibration, result.image_dims))])), [rawBySlice, calibration, result.image_dims])
  const candidates = useMemo(() => Object.values(bySlice || {}).flat().map(d => ({ ...d, kind: 'prediction' }))
    .sort((a, b) => b.yolo_score - a.yolo_score), [bySlice])
  const [currentIdx, setCurrentIdx] = useState(candidates[0]?.slice_index ?? 0)
  const [selection, setSelection] = useState(null)
  const [displayMode, setDisplayMode] = useState(references.length ? 'overlay' : 'prediction')
  const currentDets = bySlice?.[String(currentIdx)] || []
  const currentReferences = useMemo(() => referenceBoxesAt(references, geometry, currentIdx), [references, geometry, currentIdx])
  const hasDetections = useMemo(() => new Set(candidates.map(d => d.slice_index)), [candidates])

  function select(kind, target, index = currentIdx) {
    if (index < 0) return
    let chosen = target
    if (kind === 'reference') {
      chosen = referenceBoxesAt([target], geometry, index)[0]
      if (!chosen) return
    }
    setCurrentIdx(index); setSelection({ kind, target: chosen, index })
  }
  function selectFromCanvas({ kind, target }) { select(kind, target, kind === 'reference' ? target.slice_index : currentIdx) }
  return (
    <div className="results-layout">
      <div className="viewer-panel">
        <div className="viewer">
          <div className="viewer-header"><div><h2 className="viewer-title">{layerLabel(geometry, currentIdx)} / {slices.length} 层</h2>
            <p className="viewer-meta">{slices[currentIdx]?.filename} · {result.model_name}</p></div>
            <span className="viewer-meta">{(result.inference_time_ms / 1000).toFixed(1)} 秒</span></div>
          <div className="view-controls">
            {['prediction', 'reference', 'overlay'].map(value => <button key={value}
              disabled={value !== 'prediction' && references.length === 0}
              className={displayMode === value ? 'active' : ''} onClick={() => setDisplayMode(value)}>
              {{ prediction: '预测', reference: '参考标注', overlay: '叠加比较' }[value]}</button>)}
          </div>
          <p className="legend"><span className="prediction-key">橙色：预测</span><span className="reference-key">蓝色：参考</span>
            <span>点击框或右侧列表，查看三切片</span></p>
          <div className="canvas-container"><SliceCanvas slice={slices[currentIdx]}
            predictions={displayMode === 'reference' ? [] : currentDets}
            references={displayMode === 'prediction' ? [] : currentReferences}
            onSelect={selectFromCanvas} label="CT 全图，点击候选框查看三切片定位" /></div>
          <div className="slider-container">
            <label className="slider-label" htmlFor="slice-slider">浏览切片</label>
            <input id="slice-slider" type="range" min={0} max={slices.length - 1} value={currentIdx}
              onChange={e => setCurrentIdx(Number(e.target.value))} className="slice-slider" />
            <div className="slider-ticks">{slices.map((_, index) => <button key={index}
              aria-label={'跳到第 ' + (index + 1) + ' 层'} className={'slider-tick ' + (index === currentIdx ? 'active ' : '') + (hasDetections.has(index) ? 'has-det' : '')}
              onClick={() => setCurrentIdx(index)} />)}</div>
            <div className="slice-navigation"><button disabled={currentIdx === 0} onClick={() => setCurrentIdx(currentIdx - 1)}>上一层</button>
              <span>第 {currentIdx + 1} / {slices.length} 层</span><button disabled={currentIdx === slices.length - 1} onClick={() => setCurrentIdx(currentIdx + 1)}>下一层</button></div>
          </div>
          <p className="v3-summary-bar">{result.grouping_applied ? '三维合并后保留 ' + result.candidates.length + ' 个候选' : '保留 ' + result.summary.total_kept + ' 个逐层候选框；相邻层可能重复'}
            {' · '}当前层 {currentDets.length} 个候选</p>
          {!geometry && <PixelSpacingControls calibration={calibration} onChange={setCalibration} imageDims={result.image_dims} />}
          <p className="viewer-note">{geometry ? '病例 ' + geometry.seriesuid + '。' + (geometry.complete_volume ? '已上传完整体积切片。' : '当前仅上传部分体积；边界上下文可能不完整。') :
            calibration ? '尺寸已按像素间距换算为 mm；层号按上传顺序显示。' : '请导入或填写像素间距，将估计直径和半径换算为 mm。'}</p>
        </div>
      </div>
      <aside className="info-panel">
        <section className="location-panel"><h2>预测候选</h2><p className="panel-intro">点击候选，直接定位并展开三切片。</p>
          <div className="target-list">{candidates.map((det, i) => {
            const size = estimateBoxSize(det)
            return <button key={det.id} className={'target-button ' + (det.keep === false ? 'filtered-out' : '')}
            onClick={() => select('prediction', det, det.slice_index)}>
            <span className="target-title">候选 {i + 1}<strong>{(det.yolo_score * 100).toFixed(1)}%</strong></span>
            <span>{layerLabel(geometry, det.slice_index)}{det.keep === false ? ' · 已排除' : ''}</span>
            <span>{size.unit === 'mm' ? '估计直径 ' + size.diameter.toFixed(2) + ' mm · 半径 ' + size.radius.toFixed(2) + ' mm' : '填写像素间距后显示毫米尺寸'}</span>
          </button>})}</div>
          {!candidates.length && <p className="empty-message">没有超过当前阈值的候选。仍可浏览切片或定位参考标注。</p>}
        </section>
        <section className="location-panel"><h2>参考结节</h2>
          <p className="panel-intro">由标注中心与直径派生参考框。</p>
          {references.map(item => <button key={item.id} className="target-button reference-target" disabled={item.slice_index < 0}
            onClick={() => select('reference', item, item.slice_index)}>
            <span className="target-title">{item.label}<strong>{item.diameter_mm.toFixed(2)} mm</strong></span>
            <span>{item.slice_index < 0 ? '标注中心不在当前上传范围' : layerLabel(geometry, item.slice_index)}</span>
            <span>世界坐标 {formatVector(item.world_mm)} mm</span>
          </button>)}
          {!references.length && <p className="empty-message">未加载可对应的参考标注。可打开带标注演示病例，或在上传切片前选择元数据和标注 JSON。</p>}
        </section>
      </aside>
      {selection && <ThreeSliceDialog selection={{ ...selection, target: selection.kind === 'prediction' ? calibrateDetectionSize(selection.target, calibration, result.image_dims) : selection.target }} slices={slices} detections={bySlice || {}}
        references={references} geometry={geometry} imageDims={result.image_dims}
        calibration={calibration} onCalibrationChange={setCalibration}
        displayMode={displayMode} onDisplayModeChange={setDisplayMode} onClose={() => setSelection(null)} />}
    </div>
  )
}
