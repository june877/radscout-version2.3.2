const API_BASE = '/api'

/** v3.0: 单张二阶段检测 */
export async function detectV3(file, mode = 'yolo_only', resnetThreshold = null) {
  const formData = new FormData()
  formData.append('file', file)
  let url = `${API_BASE}/v3/detect?mode=${mode}`
  if (resnetThreshold != null) url += `&resnet_threshold=${resnetThreshold}`
  const res = await fetch(url, { method: 'POST', body: formData })
  if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.error || `Error ${res.status}`) }
  return res.json()
}

/** v3.0: 序列二阶段检测 */
export async function detectSeriesV3(files, mode = 'yolo_only', resnetThreshold = null, metadata = null, signal = null) {
  const formData = new FormData()
  files.forEach(f => formData.append('files', f))
  if (metadata) formData.append('metadata', JSON.stringify(metadata))
  let url = `${API_BASE}/v3/detect-series?mode=${mode}&input_type=ct_sequence`
  if (resnetThreshold != null) url += `&resnet_threshold=${resnetThreshold}`
  const res = await fetch(url, { method: 'POST', body: formData, signal })
  if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.error || `Error ${res.status}`) }
  return res.json()
}

/** Legacy compat */
export async function detectLesions(file) {
  return detectV3(file, 'yolo_only')
}

export async function detectSeries(files) {
  return detectSeriesV3(files, 'yolo_only')
}

export async function fetchMetrics() {
  const res = await fetch(`${API_BASE}/metrics`)
  if (!res.ok) throw new Error('Failed')
  return res.json()
}

export async function loadDemo(signal, onProgress) {
  const response = await fetch(API_BASE + '/demo', { signal })
  if (!response.ok) throw new Error('演示病例未找到')
  const demo = await response.json()
  const files = new Array(demo.filenames.length)
  let next = 0, completed = 0
  async function worker() {
    while (next < files.length) {
      const index = next++, name = demo.filenames[index]
      const image = await fetch(API_BASE + '/demo/images/' + encodeURIComponent(name), { signal })
      if (!image.ok) throw new Error('演示切片读取失败：' + name)
      files[index] = new File([await image.blob()], name, { type: 'image/png' })
      onProgress?.(++completed, files.length)
    }
  }
  await Promise.all(Array.from({ length: Math.min(6, files.length) }, worker))
  return { ...demo, files }
}
