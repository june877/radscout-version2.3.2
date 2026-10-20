import { useRef, useState } from 'react'

/**
 * v2.3: 支持多文件上传（主要用于 /api/detect-series）
 * multiple=true 时选择多张连续 CT 切片
 */
export default function ImageUploader({ onFilesSelect, disabled, multiple = true, minFiles = 3 }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFiles = (fileList) => {
    if (!fileList || fileList.length === 0) return
    const files = Array.from(fileList)

    // 类型校验
    for (const f of files) {
      if (!f.type.startsWith('image/')) {
        alert(`文件 ${f.name} 不是图像文件，请上传 PNG/JPG`)
        return
      }
      if (f.size > 20 * 1024 * 1024) {
        alert(`文件 ${f.name} 超过 20MB 限制`)
        return
      }
    }

    if (multiple && files.length < minFiles) {
      alert(`2.5D 检测至少需要 ${minFiles} 张连续切片，当前仅选择了 ${files.length} 张。`)
      return
    }

    onFilesSelect(files)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleFiles(e.dataTransfer.files)
  }

  return (
    <div
      className={`uploader ${dragOver ? 'drag-over' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <div className="uploader-icon">📂</div>
      <h3>上传连续 CT 切片</h3>
      <p>请选择同一 CT 序列中连续导出的灰度 PNG/JPG 切片；不要上传训练集的三通道合成图</p>
      <p className="uploader-hint">
        文件名需按层面顺序排列（如 slice_001.png, slice_002.png...）。
        2.5D 检测至少需要 3 张连续切片。
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/jpg"
        multiple={multiple}
        onChange={(e) => handleFiles(e.target.files)}
        hidden
        disabled={disabled}
      />
    </div>
  )
}
