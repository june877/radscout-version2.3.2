import { useEffect, useRef } from 'react'
import SliceCanvas from './SliceCanvas'
import PixelSpacingControls from './PixelSpacingControls'
import { contextSlices, layerLabel, regionFor, referenceBoxesAt, formatVector, estimateBoxSize } from '../seriesUtils'

export default function ThreeSliceDialog({ selection, slices, detections, references, geometry,
  imageDims, displayMode, onDisplayModeChange, onClose, calibration, onCalibrationChange }) {
  const dialogRef = useRef(null)
  const { target, kind, index } = selection
  const region = regionFor(target.bbox, imageDims.width, imageDims.height)
  const estimatedSize = kind === 'prediction' ? estimateBoxSize(target) : null
  useEffect(() => {
    const dialog = dialogRef.current, previous = document.activeElement
    dialog.showModal()
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      dialog.close(); document.body.style.overflow = previousOverflow
      if (previous?.isConnected) previous.focus()
    }
  }, [])

  function backdropClick(event) {
    if (event.target !== dialogRef.current) return
    const r = dialogRef.current.getBoundingClientRect()
    if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) onClose()
  }
  return (
    <dialog className="three-slice-dialog" ref={dialogRef} onCancel={onClose} onClick={backdropClick}
      aria-labelledby="three-slice-title">
      <header className="dialog-header">
        <div><p className="eyebrow">三切片定位</p>
          <h2 id="three-slice-title">{kind === 'reference' ? target.label : '预测候选'} · {layerLabel(geometry, index)}</h2>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="关闭三切片定位" autoFocus>×</button>
      </header>
      <div className="dialog-content">
        <div className="view-controls">
          {['prediction', 'reference', 'overlay'].map(value => <button key={value}
            disabled={value !== 'prediction' && references.length === 0}
            className={displayMode === value ? 'active' : ''} onClick={() => onDisplayModeChange(value)}>
            {{ prediction: '预测', reference: '参考标注', overlay: '叠加比较' }[value]}</button>)}
        </div>
        <p className="legend"><span className="prediction-key">橙色：预测框与中心</span>
          <span className="reference-key">蓝色：参考框与中心</span><span className="region-key">黄色虚线：同步观察区域</span></p>
        <div className="three-slice-grid">
          {contextSlices(index, slices.length).map(({ index: sourceIndex, offset, copied }) => {
            const refs = displayMode === 'prediction' ? [] : referenceBoxesAt(references, geometry, sourceIndex)
            const preds = displayMode === 'reference' ? [] : (detections[String(sourceIndex)] || [])
            return <section key={offset} className={'slice-column ' + (offset === 0 ? 'center-slice' : '')}>
              <h3>{offset === -1 ? '上一层' : offset === 0 ? '中心层' : '下一层'}</h3>
              <p className="layer-caption">{layerLabel(geometry, sourceIndex)}</p>
              {copied && <p className="boundary-note">{geometry?.complete_volume ? '体积边界，复制最近有效层' : '邻层未上传，复制当前边界层'}</p>}
              <div className="slice-frame"><SliceCanvas slice={slices[sourceIndex]} predictions={preds}
                references={refs} region={region} label={offset + ' 全图 ' + slices[sourceIndex].filename} /></div>
              <p className="crop-label">相同位置局部放大</p>
              <div className="slice-frame crop-frame"><SliceCanvas slice={slices[sourceIndex]} predictions={preds}
                references={refs} crop={region} label={offset + ' 局部放大'} /></div>
              <p className="filename-caption">{slices[sourceIndex].filename}</p>
            </section>
          })}
        </div>
        {kind === 'prediction' && !geometry && <PixelSpacingControls calibration={calibration}
          onChange={onCalibrationChange} imageDims={imageDims} />}
        <dl className="location-details">
          <div><dt>图像中心（像素）</dt><dd>{formatVector(target.center_px || [target.bbox.x + target.bbox.width / 2, target.bbox.y + target.bbox.height / 2])}</dd></div>
          <div><dt>世界中心（mm）</dt><dd>{formatVector(target.world_mm)}</dd></div>
          <div><dt>{kind === 'reference' ? '标注直径' : '框估计尺寸'}</dt>
            <dd>{kind === 'reference' ? target.diameter_mm.toFixed(2) + ' mm' :
              estimatedSize.unit === 'mm' ?
                <><span className="estimated-size-line">估计直径 {estimatedSize.diameter.toFixed(2)} mm</span>
                  <span className="estimated-size-line">估计半径 {estimatedSize.radius.toFixed(2)} mm</span></> :
                '请填写像素间距，换算成 mm'}</dd></div>
          {kind === 'prediction' && <div><dt>YOLO 评分</dt><dd>{(target.yolo_score * 100).toFixed(1)}%</dd></div>}
        </dl>
        <p className="viewer-note">框宽、高乘各自方向的像素间距得到毫米长度，取较长一边作为估计直径，估计半径为直径的一半。上下层的黄色区域用于观察同一位置，不代表额外检出。参考框由结节中心与直径按球体近似派生，不是精确人工轮廓。</p>
      </div>
    </dialog>
  )
}
