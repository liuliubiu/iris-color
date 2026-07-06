<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, computed } from 'vue'
import { ElMessage } from 'element-plus'
import {
  analyzeIris,
  analyzeIrisManual,
  checkHealth,
  parseAnalysisError,
  type AnalysisResult,
  type DetectionInfo,
} from '../api/iris'
import { useLayoutMode } from '../composables/useLayoutMode'
import BrandLogo from '../components/BrandLogo.vue'

const { isMobile } = useLayoutMode()

const videoRef = ref<HTMLVideoElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const adjustCanvasRef = ref<HTMLCanvasElement | null>(null)
const cropCanvasRef = ref<HTMLCanvasElement | null>(null)
const fileInputRef = ref<HTMLInputElement | null>(null)
const resultCardRef = ref<HTMLElement | null>(null)
const captureCardRef = ref<HTMLElement | null>(null)
const diagnosisRef = ref<HTMLElement | null>(null)

const cameraActive = ref(false)
const previewUrl = ref<string | null>(null)
const originalPreviewUrl = ref<string | null>(null)
const loading = ref(false)
const manualLoading = ref(false)
const result = ref<AnalysisResult | null>(null)
const errorMsg = ref('')
const qualityCheckFailed = ref(false)
const currentFile = ref<File | null>(null)
const originalFile = ref<File | null>(null)
const manualMode = ref(false)
const cropMode = ref(false)
const manualParams = ref<DetectionInfo | null>(null)
const adjustCursor = ref('default')
const cropCursor = ref('crosshair')
const skipQuality = ref(false)
const settingsOpen = ref(false)
const cameraErrorOpen = ref(false)
const isCoarsePointer =
  typeof window !== 'undefined' &&
  typeof window.matchMedia === 'function' &&
  window.matchMedia('(pointer: coarse)').matches
const touchTolBoost = isCoarsePointer ? 1.8 : 1
const cameraFacing = ref<'environment' | 'user'>('environment')
const cameraSwitching = ref(false)
const appVersion = ref('')

interface CropRect {
  x: number
  y: number
  width: number
  height: number
}

const cropRect = ref<CropRect>({ x: 0, y: 0, width: 0, height: 0 })

let mediaStream: MediaStream | null = null
let adjustImage: HTMLImageElement | null = null
let cropImage: HTMLImageElement | null = null
/** 框选交互画布相对原图缩放比（display = source * scale），导出裁剪时用 inverse 还原 */
let cropDisplayScale = 1
let cropBaseCanvas: HTMLCanvasElement | null = null
let cropDrawRaf = 0
const CROP_INTERACTIVE_MAX_DIM = 1280
type DragMode = 'move' | 'pupil_radius' | 'inner_radius' | 'outer_radius'
type CropDragMode = 'move' | 'nw' | 'ne' | 'sw' | 'se'
let dragMode: DragMode | null = null
let cropDragMode: CropDragMode | null = null
let dragOffset = { x: 0, y: 0 }
let cropDragStart = { x: 0, y: 0, rect: { x: 0, y: 0, width: 0, height: 0 } }

const gradeLabels: Record<number, string> = {
  1: '最浅',
  2: '较浅',
  3: '中等',
  4: '较深',
  5: '最深',
}

function roundRectPath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const r = Math.min(radius, width / 2, height / 2)
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + width, y, x + width, y + height, r)
  ctx.arcTo(x + width, y + height, x, y + height, r)
  ctx.arcTo(x, y + height, x, y, r)
  ctx.arcTo(x, y, x + width, y, r)
  ctx.closePath()
}

function loadReportEyeImage(): Promise<HTMLImageElement | null> {
  const url = previewUrl.value
  if (!url) return Promise.resolve(null)
  return new Promise((resolve) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => resolve(null)
    image.src = url
  })
}

function drawClippedImage(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  ctx.save()
  roundRectPath(ctx, x, y, width, height, radius)
  ctx.clip()
  ctx.drawImage(image, x, y, width, height)
  ctx.restore()
  roundRectPath(ctx, x, y, width, height, radius)
  ctx.strokeStyle = '#dceaf2'
  ctx.lineWidth = 1
  ctx.stroke()
}

async function downloadResultReport() {
  if (!result.value) return

  const data = result.value
  const colorLabel = data.iris_color?.label ?? '未知'
  const grade = data.grade ?? '-'
  const gradeNote = gradeLabels[data.grade ?? 0] ?? '待判定'
  const labL = data.lab?.L?.toFixed(2) ?? '-'
  const labA = data.lab?.a?.toFixed(2) ?? '-'
  const labB = data.lab?.b?.toFixed(2) ?? '-'
  const eyeImage = await loadReportEyeImage()

  const width = 760
  const scale = 2
  const cardX = 36
  const cardPadTop = 32
  const cardPadBottom = 40
  const headerH = 88
  const blockX = 60
  const blockW = width - 120
  const sectionGap = 14
  const colorBlockH = 92
  const gradeBlockH = 112
  const metricBlockH = 88

  let eyeFrame: { x: number; y: number; w: number; h: number } | null = null
  let yCursor = cardPadTop + headerH + sectionGap

  if (eyeImage) {
    const maxEyeW = blockW
    const maxEyeH = 200
    const fitScale = Math.min(maxEyeW / eyeImage.naturalWidth, maxEyeH / eyeImage.naturalHeight)
    const drawW = eyeImage.naturalWidth * fitScale
    const drawH = eyeImage.naturalHeight * fitScale
    eyeFrame = {
      x: blockX + (blockW - drawW) / 2,
      y: yCursor,
      w: drawW,
      h: drawH,
    }
    yCursor += drawH + sectionGap
  }

  const colorY = yCursor
  yCursor += colorBlockH + sectionGap
  const gradeY = yCursor
  yCursor += gradeBlockH + sectionGap
  const metricY = yCursor
  yCursor += metricBlockH
  const cardHeight = yCursor - cardPadTop + cardPadBottom
  const height = cardPadTop + cardHeight + 32

  const canvas = document.createElement('canvas')
  canvas.width = width * scale
  canvas.height = height * scale
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.scale(scale, scale)
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'

  ctx.fillStyle = '#eef5f8'
  ctx.fillRect(0, 0, width, height)

  roundRectPath(ctx, cardX, cardPadTop, width - 72, cardHeight, 24)
  ctx.fillStyle = '#ffffff'
  ctx.fill()
  ctx.strokeStyle = 'rgba(32, 112, 171, 0.14)'
  ctx.lineWidth = 1
  ctx.stroke()

  roundRectPath(ctx, cardX, cardPadTop, width - 72, headerH, 24)
  ctx.fillStyle = '#146c9c'
  ctx.fill()
  ctx.fillStyle = '#ffffff'
  ctx.font = '600 13px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
  ctx.fillText('IRIS COLOR ANALYSIS', blockX, cardPadTop + 30)
  ctx.font = '700 24px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
  ctx.fillText('虹膜颜色分析报告', blockX, cardPadTop + 66)

  if (eyeImage && eyeFrame) {
    drawClippedImage(ctx, eyeImage, eyeFrame.x, eyeFrame.y, eyeFrame.w, eyeFrame.h, 14)
  }

  roundRectPath(ctx, blockX, colorY, blockW, colorBlockH, 16)
  ctx.fillStyle = '#f8fbfd'
  ctx.fill()
  ctx.strokeStyle = '#dceaf2'
  ctx.stroke()
  ctx.fillStyle = '#687d8f'
  ctx.font = '600 13px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText('虹膜颜色', blockX + blockW / 2, colorY + 30)
  ctx.fillStyle = '#17324a'
  ctx.font = '700 34px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
  ctx.fillText(colorLabel, blockX + blockW / 2, colorY + 74)

  roundRectPath(ctx, blockX, gradeY, blockW, gradeBlockH, 16)
  ctx.fillStyle = '#f8fbfd'
  ctx.fill()
  ctx.strokeStyle = '#dceaf2'
  ctx.stroke()
  ctx.fillStyle = '#687d8f'
  ctx.font = '600 13px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
  ctx.fillText('颜色等级', blockX + blockW / 2, gradeY + 30)
  ctx.fillStyle = '#146c9c'
  ctx.font = '800 52px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
  ctx.fillText(`Grade ${grade}`, blockX + blockW / 2, gradeY + 82)
  const noteW = ctx.measureText(gradeNote).width + 28
  const noteX = blockX + (blockW - noteW) / 2
  roundRectPath(ctx, noteX, gradeY + 94, noteW, 28, 14)
  ctx.fillStyle = '#eaf6fb'
  ctx.fill()
  ctx.fillStyle = '#23637f'
  ctx.font = '700 14px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
  ctx.fillText(gradeNote, blockX + blockW / 2, gradeY + 113)

  const metricW = (blockW - 24) / 3
  const metrics = [
    { label: 'L* 明度', value: labL },
    { label: 'a*', value: labA },
    { label: 'b*', value: labB },
  ]
  metrics.forEach((item, index) => {
    const x = blockX + index * (metricW + 12)
    roundRectPath(ctx, x, metricY, metricW, metricBlockH, 14)
    ctx.fillStyle = '#ffffff'
    ctx.fill()
    ctx.strokeStyle = '#dceaf2'
    ctx.stroke()
    ctx.fillStyle = '#687d8f'
    ctx.font = '600 12px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
    ctx.fillText(item.label, x + metricW / 2, metricY + 26)
    ctx.fillStyle = '#17324a'
    ctx.font = '700 28px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
    ctx.fillText(item.value, x + metricW / 2, metricY + 66)
  })

  ctx.textAlign = 'left'
  ctx.fillStyle = '#8aa3b5'
  ctx.font = '500 11px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
  const stamp = new Date().toLocaleString('zh-CN', { hour12: false })
  ctx.fillText(`Generated ${stamp}`, blockX, cardPadTop + cardHeight - 18)

  canvas.toBlob((blob) => {
    if (!blob) {
      ElMessage.error('报告生成失败，请重试')
      return
    }
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `iris-color-report-${Date.now()}.png`
    link.click()
    URL.revokeObjectURL(url)
    ElMessage.success('分析报告已保存至本地')
  }, 'image/png')
}

async function startCamera(options?: { showErrorOnFail?: boolean }): Promise<boolean> {
  stopCamera()
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: cameraFacing.value },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    })
    if (videoRef.value) {
      videoRef.value.srcObject = mediaStream
      await videoRef.value.play()
      cameraActive.value = true
    }
    return true
  } catch {
    if (options?.showErrorOnFail) {
      cameraErrorOpen.value = true
    }
    return false
  }
}

function closeCameraErrorDialog() {
  cameraErrorOpen.value = false
}

function openFilePickerFromCameraDialog() {
  cameraErrorOpen.value = false
  fileInputRef.value?.click()
}

async function switchCamera() {
  if (cameraSwitching.value) return
  cameraSwitching.value = true
  cameraFacing.value = cameraFacing.value === 'environment' ? 'user' : 'environment'
  try {
    await startCamera()
  } finally {
    cameraSwitching.value = false
  }
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop())
    mediaStream = null
  }
  cameraActive.value = false
}

async function capturePhoto() {
  if (!cameraActive.value || !videoRef.value?.videoWidth) {
    const ok = await startCamera({ showErrorOnFail: true })
    if (!ok) return
  }

  const video = videoRef.value
  const canvas = canvasRef.value
  if (!video || !canvas) return

  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.drawImage(video, 0, 0)
  canvas.toBlob((blob) => {
    if (!blob) return
    const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' })
    enterCropMode(file)
  }, 'image/jpeg', 0.92)
}

function onFileSelected(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  enterCropMode(file)
  input.value = ''
}

function enterCropMode(file: File) {
  result.value = null
  errorMsg.value = ''
  qualityCheckFailed.value = false
  manualMode.value = false
  manualParams.value = null
  adjustImage = null
  currentFile.value = null
  cropImage = null
  resetCropRenderCache()

  revokePreviewUrls()
  originalFile.value = file
  originalPreviewUrl.value = URL.createObjectURL(file)
  previewUrl.value = originalPreviewUrl.value
  cropMode.value = true
  nextTick(loadCropCanvas)
}

function revokePreviewUrls() {
  if (originalPreviewUrl.value) {
    URL.revokeObjectURL(originalPreviewUrl.value)
    originalPreviewUrl.value = null
  }
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = null
  }
}

function resetCropRenderCache() {
  cropBaseCanvas = null
  cropDisplayScale = 1
  if (cropDrawRaf) {
    cancelAnimationFrame(cropDrawRaf)
    cropDrawRaf = 0
  }
}

function defaultCropRect(width: number, height: number): CropRect {
  const size = Math.min(width, height) * 0.72
  return {
    x: (width - size) / 2,
    y: (height - size) / 2,
    width: size,
    height: size,
  }
}

function loadCropCanvas() {
  const canvas = cropCanvasRef.value
  const url = originalPreviewUrl.value
  if (!canvas || !url) return
  const image = new Image()
  image.onload = () => {
    cropImage = image
    resetCropRenderCache()

    const maxDim = Math.max(image.naturalWidth, image.naturalHeight)
    cropDisplayScale = maxDim > CROP_INTERACTIVE_MAX_DIM
      ? CROP_INTERACTIVE_MAX_DIM / maxDim
      : 1

    canvas.width = Math.max(1, Math.round(image.naturalWidth * cropDisplayScale))
    canvas.height = Math.max(1, Math.round(image.naturalHeight * cropDisplayScale))

    cropBaseCanvas = document.createElement('canvas')
    cropBaseCanvas.width = canvas.width
    cropBaseCanvas.height = canvas.height
    const baseCtx = cropBaseCanvas.getContext('2d')
    baseCtx?.drawImage(image, 0, 0, canvas.width, canvas.height)

    cropRect.value = defaultCropRect(canvas.width, canvas.height)
    drawCropCanvas()
  }
  image.src = url
}

function normalizeCropRect() {
  const canvas = cropCanvasRef.value
  if (!canvas) return
  const minSize = Math.max(80, Math.min(canvas.width, canvas.height) * 0.12)
  let { x, y, width, height } = cropRect.value
  width = Math.max(minSize, width)
  height = Math.max(minSize, height)
  x = Math.max(0, Math.min(x, canvas.width - minSize))
  y = Math.max(0, Math.min(y, canvas.height - minSize))
  if (x + width > canvas.width) x = canvas.width - width
  if (y + height > canvas.height) y = canvas.height - height
  cropRect.value = { x, y, width, height }
}

function scheduleDrawCropCanvas() {
  if (cropDrawRaf) return
  cropDrawRaf = requestAnimationFrame(() => {
    cropDrawRaf = 0
    drawCropCanvas()
  })
}

function drawCropCanvas() {
  const canvas = cropCanvasRef.value
  if (!canvas || !cropImage) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const r = cropRect.value
  const uiScale = canvasUiScale(cropCanvasRef.value)
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (cropBaseCanvas) {
    ctx.drawImage(cropBaseCanvas, 0, 0)
  } else {
    ctx.drawImage(cropImage, 0, 0, canvas.width, canvas.height)
  }
  ctx.fillStyle = 'rgba(7, 21, 33, 0.58)'
  ctx.fillRect(0, 0, canvas.width, r.y)
  ctx.fillRect(0, r.y + r.height, canvas.width, canvas.height - r.y - r.height)
  ctx.fillRect(0, r.y, r.x, r.height)
  ctx.fillRect(r.x + r.width, r.y, canvas.width - r.x - r.width, r.height)
  ctx.strokeStyle = '#62caff'
  ctx.lineWidth = Math.max(2, 2.5 * uiScale)
  ctx.setLineDash([8 * uiScale, 6 * uiScale])
  ctx.strokeRect(r.x, r.y, r.width, r.height)
  ctx.setLineDash([])
  const handle = Math.max(8, 7 * uiScale)
  ctx.fillStyle = '#ffffff'
  ctx.strokeStyle = '#1876a9'
  ctx.lineWidth = Math.max(1.5, 2 * uiScale)
  for (const corner of [
    [r.x, r.y],
    [r.x + r.width, r.y],
    [r.x + r.width, r.y + r.height],
    [r.x, r.y + r.height],
  ]) {
    ctx.fillRect(corner[0] - handle / 2, corner[1] - handle / 2, handle, handle)
    ctx.strokeRect(corner[0] - handle / 2, corner[1] - handle / 2, handle, handle)
  }
}

function hitCropMode(point: { x: number; y: number }) {
  const r = cropRect.value
  const uiScale = canvasUiScale(cropCanvasRef.value)
  const handleSize = Math.max(14 * uiScale, 12) * touchTolBoost
  const corners: Array<[CropDragMode, number, number]> = [
    ['nw', r.x, r.y],
    ['ne', r.x + r.width, r.y],
    ['se', r.x + r.width, r.y + r.height],
    ['sw', r.x, r.y + r.height],
  ]
  for (const [mode, cx, cy] of corners) {
    if (Math.abs(point.x - cx) <= handleSize && Math.abs(point.y - cy) <= handleSize) {
      return mode
    }
  }
  if (point.x >= r.x && point.x <= r.x + r.width && point.y >= r.y && point.y <= r.y + r.height) {
    return 'move'
  }
  return null
}

function updateCropCursor(event: PointerEvent) {
  if (cropDragMode) {
    cropCursor.value = cropDragMode === 'move' ? 'grabbing' : `${cropDragMode}-resize`
    return
  }
  const mode = hitCropMode(canvasPoint(event, cropCanvasRef.value))
  if (mode === 'move') cropCursor.value = 'grab'
  else if (mode) cropCursor.value = `${mode}-resize`
  else cropCursor.value = 'crosshair'
}

function onCropPointerDown(event: PointerEvent) {
  const point = canvasPoint(event, cropCanvasRef.value)
  cropDragMode = hitCropMode(point)
  if (!cropDragMode) return
  cropDragStart = { x: point.x, y: point.y, rect: { ...cropRect.value } }
  cropCursor.value = cropDragMode === 'move' ? 'grabbing' : `${cropDragMode}-resize`
  cropCanvasRef.value?.setPointerCapture(event.pointerId)
}

function onCropPointerMove(event: PointerEvent) {
  if (!cropDragMode) {
    updateCropCursor(event)
    return
  }
  const point = canvasPoint(event, cropCanvasRef.value)
  const start = cropDragStart
  let { x, y, width, height } = start.rect
  if (cropDragMode === 'move') {
    x = start.rect.x + (point.x - start.x)
    y = start.rect.y + (point.y - start.y)
  } else {
    if (cropDragMode.includes('e')) width = point.x - start.rect.x
    if (cropDragMode.includes('w')) {
      width = start.rect.x + start.rect.width - point.x
      x = point.x
    }
    if (cropDragMode.includes('s')) height = point.y - start.rect.y
    if (cropDragMode.includes('n')) {
      height = start.rect.y + start.rect.height - point.y
      y = point.y
    }
  }
  cropRect.value = { x, y, width, height }
  normalizeCropRect()
  scheduleDrawCropCanvas()
}

function onCropPointerEnd(event: PointerEvent) {
  cropDragMode = null
  updateCropCursor(event)
  try {
    cropCanvasRef.value?.releasePointerCapture(event.pointerId)
  } catch {}
}

async function applyCropAndAnalyze() {
  if (!cropImage) return
  const inv = 1 / cropDisplayScale
  const sx = Math.round(cropRect.value.x * inv)
  const sy = Math.round(cropRect.value.y * inv)
  const w = Math.max(1, Math.round(cropRect.value.width * inv))
  const h = Math.max(1, Math.round(cropRect.value.height * inv))
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.drawImage(cropImage, sx, sy, w, h, 0, 0, w, h)

  const blob = await new Promise<Blob | null>((resolve) => {
    canvas.toBlob(resolve, 'image/jpeg', 0.92)
  })
  if (!blob) {
    ElMessage.error('裁剪失败，请重试')
    return
  }

  cropMode.value = false
  cropImage = null
  resetCropRenderCache()
  if (previewUrl.value && previewUrl.value !== originalPreviewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
  }
  previewUrl.value = URL.createObjectURL(blob)
  const name = originalFile.value?.name?.replace(/(\.[^.]+)?$/, '_crop.jpg') ?? 'crop.jpg'
  await uploadFile(new File([blob], name, { type: 'image/jpeg' }))
}

function reenterCropMode() {
  if (!originalFile.value || !originalPreviewUrl.value) return
  result.value = null
  errorMsg.value = ''
  qualityCheckFailed.value = false
  manualMode.value = false
  manualParams.value = null
  adjustImage = null
  if (previewUrl.value && previewUrl.value !== originalPreviewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
  }
  previewUrl.value = originalPreviewUrl.value
  cropMode.value = true
  nextTick(() => {
    loadCropCanvas()
    scrollCaptureIntoView()
  })
}

async function reanalyze() {
  const file = currentFile.value
  if (!file) return
  await uploadFile(file)
}

function scrollCaptureIntoView() {
  if (typeof window === 'undefined') return
  if (!window.matchMedia?.('(max-width: 980px)').matches) return
  nextTick(() => {
    captureCardRef.value?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  })
}

function scrollResultIntoView(target: 'card' | 'diagnosis' = 'card') {
  // 仅在窄屏（结果卡片堆叠在采集卡片下方）时自动滚动
  if (typeof window === 'undefined') return
  if (!window.matchMedia?.('(max-width: 980px)').matches) return
  nextTick(() => {
    if (target === 'diagnosis' && diagnosisRef.value) {
      // 让「虹膜颜色 + 颜色等级」诊断区居中显示
      diagnosisRef.value.scrollIntoView({ behavior: 'smooth', block: 'center' })
    } else {
      resultCardRef.value?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  })
}

function applyAnalysisError(data: unknown) {
  const parsed = parseAnalysisError(data)
  errorMsg.value = parsed.message
  qualityCheckFailed.value = parsed.qualityCheckFailed
}

const errorNoticeTitle = computed(() => {
  if (qualityCheckFailed.value) return '图像质量未达标'
  return '分析未通过'
})

const errorNoticeDetail = computed(() => {
  const msg = errorMsg.value
  const prefix = '图像质量未达标：'
  if (qualityCheckFailed.value && msg.startsWith(prefix)) {
    return msg.slice(prefix.length)
  }
  return msg
})

async function skipQualityAndReanalyze() {
  skipQuality.value = true
  qualityCheckFailed.value = false
  await reanalyze()
}

async function uploadFile(file: File) {
  loading.value = true
  result.value = null
  errorMsg.value = ''
  qualityCheckFailed.value = false
  currentFile.value = file
  manualMode.value = false
  manualParams.value = null
  adjustImage = null
  scrollResultIntoView()

  try {
    const data = await analyzeIris(file, skipQuality.value)
    if (data.success === false) {
      applyAnalysisError(data)
      scrollResultIntoView('card')
    } else {
      result.value = data
      initManualParams(data)
      manualMode.value = false
      // 结果渲染后卡片高度变化，重新滚动让诊断区居中显示
      scrollResultIntoView('diagnosis')
    }
  } catch (err: unknown) {
    if (axiosIsError(err) && err.response?.data) {
      applyAnalysisError(err.response.data)
    } else {
      errorMsg.value = '请求失败，请确认 iris-api 与 iris-vision 已启动'
      qualityCheckFailed.value = false
    }
    scrollResultIntoView('card')
  } finally {
    loading.value = false
  }
}

function initManualParams(data: AnalysisResult) {
  if (!data.detection) return
  manualParams.value = { ...data.detection }
}

function startManualAdjust() {
  if (!previewUrl.value || !manualParams.value) return
  manualMode.value = true
  nextTick(() => {
    loadAdjustCanvas()
    scrollCaptureIntoView()
  })
}

function clearPreview() {
  revokePreviewUrls()
  currentFile.value = null
  originalFile.value = null
  manualMode.value = false
  cropMode.value = false
  manualParams.value = null
  adjustImage = null
  cropImage = null
  resetCropRenderCache()
  qualityCheckFailed.value = false
}

function loadAdjustCanvas() {
  const canvas = adjustCanvasRef.value
  if (!canvas || !previewUrl.value) return
  const image = new Image()
  image.onload = () => {
    adjustImage = image
    canvas.width = image.naturalWidth
    canvas.height = image.naturalHeight
    normalizeManualParams()
    drawAdjustCanvas()
  }
  image.src = previewUrl.value
}

function normalizeManualParams() {
  const p = manualParams.value
  const canvas = adjustCanvasRef.value
  if (!p || !canvas) return
  const minDim = Math.min(canvas.width || 1, canvas.height || 1)
  p.center_x = Math.max(0, Math.min(canvas.width - 1, p.center_x))
  p.center_y = Math.max(0, Math.min(canvas.height - 1, p.center_y))
  p.pupil_radius = Math.max(2, Math.min(p.pupil_radius, minDim * 0.45))
  p.inner_radius = Math.max(p.pupil_radius + 1, Math.min(p.inner_radius, minDim * 0.48))
  p.outer_radius = Math.max(p.inner_radius + 2, Math.min(p.outer_radius, minDim * 0.5))
}

function drawAdjustCanvas() {
  const canvas = adjustCanvasRef.value
  const p = manualParams.value
  if (!canvas || !p || !adjustImage) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const uiScale = canvasUiScale(canvas)
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(adjustImage, 0, 0)
  ctx.lineWidth = Math.max(2, 3 * uiScale)
  ctx.fillStyle = 'rgba(0, 255, 255, 0.18)'
  ctx.beginPath()
  ctx.arc(p.center_x, p.center_y, p.outer_radius, 0, Math.PI * 2)
  ctx.arc(p.center_x, p.center_y, p.inner_radius, 0, Math.PI * 2, true)
  ctx.fill()
  drawCircle(ctx, p.center_x, p.center_y, p.pupil_radius, '#2f81f7')
  drawCircle(ctx, p.center_x, p.center_y, p.inner_radius, '#ff4d4f')
  drawCircle(ctx, p.center_x, p.center_y, p.outer_radius, '#3fb950')
  const crossHalf = 12 * uiScale
  ctx.strokeStyle = '#ff4d4f'
  ctx.beginPath()
  ctx.moveTo(p.center_x - crossHalf, p.center_y)
  ctx.lineTo(p.center_x + crossHalf, p.center_y)
  ctx.moveTo(p.center_x, p.center_y - crossHalf)
  ctx.lineTo(p.center_x, p.center_y + crossHalf)
  ctx.stroke()
}

function drawCircle(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string) {
  ctx.strokeStyle = color
  ctx.beginPath()
  ctx.arc(x, y, r, 0, Math.PI * 2)
  ctx.stroke()
}

function canvasDisplayMetrics(canvas: HTMLCanvasElement | null) {
  if (!canvas) return null
  const rect = canvas.getBoundingClientRect()
  if (!rect.width || !rect.height || !canvas.width || !canvas.height) return null

  const canvasRatio = canvas.width / canvas.height
  const rectRatio = rect.width / rect.height
  let contentWidth = rect.width
  let contentHeight = rect.height
  let offsetX = 0
  let offsetY = 0

  if (getComputedStyle(canvas).objectFit === 'contain') {
    if (rectRatio > canvasRatio) {
      contentHeight = rect.height
      contentWidth = contentHeight * canvasRatio
      offsetX = (rect.width - contentWidth) / 2
    } else {
      contentWidth = rect.width
      contentHeight = contentWidth / canvasRatio
      offsetY = (rect.height - contentHeight) / 2
    }
  }

  return {
    left: rect.left + offsetX,
    top: rect.top + offsetY,
    width: contentWidth,
    height: contentHeight,
    scaleX: canvas.width / contentWidth,
    scaleY: canvas.height / contentHeight,
  }
}

function canvasPoint(event: PointerEvent, canvas: HTMLCanvasElement | null) {
  const metrics = canvasDisplayMetrics(canvas)
  if (!canvas || !metrics) return { x: 0, y: 0 }
  return {
    x: (event.clientX - metrics.left) * metrics.scaleX,
    y: (event.clientY - metrics.top) * metrics.scaleY,
  }
}

function canvasUiScale(canvas: HTMLCanvasElement | null) {
  const metrics = canvasDisplayMetrics(canvas)
  if (!metrics) return 1
  return (metrics.scaleX + metrics.scaleY) / 2
}

function hitMode(point: { x: number; y: number }) {
  const p = manualParams.value
  const canvas = adjustCanvasRef.value
  if (!p || !canvas) return null
  const dx = point.x - p.center_x
  const dy = point.y - p.center_y
  const dist = Math.hypot(dx, dy)
  const uiScale = canvasUiScale(canvas)
  const lineTol = Math.max(12 * uiScale, 8) * touchTolBoost
  const crossTol = Math.max(10 * uiScale, 6) * touchTolBoost
  const crossHalf = 12 * uiScale
  const onHorizontalCross = Math.abs(dy) <= crossTol && Math.abs(dx) <= crossHalf
  const onVerticalCross = Math.abs(dx) <= crossTol && Math.abs(dy) <= crossHalf
  if (dist <= crossTol) return 'move'
  if (onHorizontalCross || onVerticalCross) return 'move'
  if (Math.abs(dist - p.pupil_radius) <= lineTol) return 'pupil_radius'
  if (Math.abs(dist - p.inner_radius) <= lineTol) return 'inner_radius'
  if (Math.abs(dist - p.outer_radius) <= lineTol) return 'outer_radius'
  return null
}

function updateAdjustCursor(event: PointerEvent) {
  if (dragMode) {
    adjustCursor.value = 'grabbing'
    return
  }
  adjustCursor.value = hitMode(canvasPoint(event, adjustCanvasRef.value)) ? 'grab' : 'default'
}

function onAdjustPointerDown(event: PointerEvent) {
  const point = canvasPoint(event, adjustCanvasRef.value)
  dragMode = hitMode(point)
  if (dragMode) {
    const p = manualParams.value
    dragOffset = p && dragMode === 'move'
      ? { x: point.x - p.center_x, y: point.y - p.center_y }
      : { x: 0, y: 0 }
    adjustCursor.value = 'grabbing'
    adjustCanvasRef.value?.setPointerCapture(event.pointerId)
  }
}

function onAdjustPointerMove(event: PointerEvent) {
  const p = manualParams.value
  if (!p) return
  if (!dragMode) {
    updateAdjustCursor(event)
    return
  }
  const point = canvasPoint(event, adjustCanvasRef.value)
  if (dragMode === 'move') {
    p.center_x = point.x - dragOffset.x
    p.center_y = point.y - dragOffset.y
  } else {
    p[dragMode] = Math.hypot(point.x - p.center_x, point.y - p.center_y)
  }
  normalizeManualParams()
  drawAdjustCanvas()
}

function onAdjustPointerEnd(event: PointerEvent) {
  dragMode = null
  updateAdjustCursor(event)
  try {
    adjustCanvasRef.value?.releasePointerCapture(event.pointerId)
  } catch {}
}

function updateManualNumber(key: keyof DetectionInfo, value: string) {
  const p = manualParams.value
  if (!p || key === 'method') return
  p[key] = Number(value || p[key])
  normalizeManualParams()
  drawAdjustCanvas()
}

async function analyzeWithManualParams() {
  if (!currentFile.value || !manualParams.value) return
  manualLoading.value = true
  errorMsg.value = ''
  qualityCheckFailed.value = false
  try {
    const data = await analyzeIrisManual(currentFile.value, manualParams.value, skipQuality.value)
    if (data.success === false) {
      applyAnalysisError(data)
    } else {
      result.value = data
      initManualParams(data)
      manualMode.value = false
      ElMessage.success('已按人工调整区域重新识别')
      scrollResultIntoView('diagnosis')
    }
  } catch (err: unknown) {
    if (axiosIsError(err) && err.response?.data) {
      applyAnalysisError(err.response.data)
    } else {
      errorMsg.value = '人工调整分析请求失败'
      qualityCheckFailed.value = false
    }
  } finally {
    manualLoading.value = false
  }
}

function axiosIsError(err: unknown): err is { response?: { data?: unknown } } {
  return typeof err === 'object' && err !== null && 'response' in err
}

onMounted(() => {
  startCamera()
  if (!isMobile.value) {
    checkHealth()
      .then((h) => {
        const v = (h as { ui_version?: string })?.ui_version
        if (v) appVersion.value = v
      })
      .catch(() => {})
  }
})

onBeforeUnmount(() => {
  stopCamera()
  resetCropRenderCache()
  revokePreviewUrls()
})
</script>

<template>
  <main class="page-shell" :class="{ 'layout-desktop': !isMobile }">
    <!-- 桌面端：专业软件顶栏 -->
    <header v-if="!isMobile" class="desktop-titlebar">
      <div class="desktop-brand">
        <BrandLogo variant="desktop" />
        <div class="desktop-brand-text">
          <strong>豪赋-虹膜颜色识别</strong>
          <span>虹膜颜色分级检测系统</span>
        </div>
      </div>
      <div class="desktop-titlebar-tools">
        <div class="desktop-status-chip">
          <span class="status-dot" :class="{ active: cameraActive }"></span>
          <span>{{ cameraActive ? `摄像头已就绪（${cameraFacing === 'environment' ? '后摄' : '前摄'}）` : '摄像头未接入' }}</span>
        </div>
        <label class="desktop-setting-toggle">
          <el-switch v-model="skipQuality" size="small" />
          <span>跳过质量检测</span>
        </label>
      </div>
    </header>

    <!-- 移动端：原有顶栏 -->
    <section v-else class="hero">
      <div class="hero-copy">
        <BrandLogo variant="mobile" />
        <h1>虹膜颜色识别</h1>
      </div>
      <div class="hero-actions">
        <div class="hero-status">
          <span class="status-dot" :class="{ active: cameraActive }"></span>
          <span>{{ cameraActive ? `摄像头已就绪（${cameraFacing === 'environment' ? '后摄' : '前摄'}）` : '摄像头未接入' }}</span>
        </div>
        <button
          type="button"
          class="settings-btn"
          :class="{ active: settingsOpen }"
          aria-label="检测设置"
          @click="settingsOpen = !settingsOpen"
        >
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
      </div>
    </section>

    <transition name="settings-slide">
      <div v-if="settingsOpen" class="mobile-settings-panel">
        <div class="settings-row">
          <span class="settings-label">跳过质量检测</span>
          <el-switch v-model="skipQuality" />
        </div>
      </div>
    </transition>
    <div v-if="settingsOpen" class="mobile-settings-backdrop" @click="settingsOpen = false"></div>

    <div class="workspace-shell">
    <section class="workflow-grid">
      <article
        ref="captureCardRef"
        class="clinical-card capture-card"
        :class="{ 'manual-mode-active': manualMode }"
      >
        <header class="card-header">
          <div>
            <span v-if="isMobile" class="section-kicker">Step 01</span>
            <span v-else class="section-kicker section-kicker--desktop">采集模块</span>
            <h2>图像采集</h2>
          </div>
          <el-tag v-if="isMobile" effect="plain" type="success">眼部特写</el-tag>
        </header>

        <div class="camera-box">
          <video
            v-show="cameraActive && !previewUrl && !manualMode && !cropMode"
            ref="videoRef"
            class="camera-video"
            autoplay
            playsinline
            muted
          />
          <img
            v-if="previewUrl && !manualMode && !cropMode"
            :src="previewUrl"
            alt="预览"
            class="preview-img"
          />
          <canvas
            v-show="cropMode"
            ref="cropCanvasRef"
            class="crop-canvas"
            :style="{ cursor: cropCursor }"
            @pointerdown="onCropPointerDown"
            @pointermove="onCropPointerMove"
            @pointerup="onCropPointerEnd"
            @pointercancel="onCropPointerEnd"
            @pointerleave="cropCursor = 'crosshair'"
          />
          <canvas
            v-show="manualMode"
            ref="adjustCanvasRef"
            class="adjust-canvas"
            :style="{ cursor: adjustCursor }"
            @pointerdown="onAdjustPointerDown"
            @pointermove="onAdjustPointerMove"
            @pointerup="onAdjustPointerEnd"
            @pointercancel="onAdjustPointerEnd"
            @pointerleave="adjustCursor = 'default'"
          />
          <button
            v-if="cameraActive && !previewUrl && !manualMode && !cropMode"
            type="button"
            class="camera-flip-btn"
            :disabled="cameraSwitching"
            :aria-label="cameraFacing === 'environment' ? '切换至前摄' : '切换至后摄'"
            @click="switchCamera"
          >
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 7h4l2-3h6l2 3h4" />
              <rect x="3" y="7" width="18" height="13" rx="2" />
              <path d="M12 11v4" />
              <path d="M9.5 13.5 12 11l2.5 2.5" />
            </svg>
            <span>{{ cameraFacing === 'environment' ? '前摄' : '后摄' }}</span>
          </button>
          <div v-if="!cameraActive && !previewUrl && !manualMode && !cropMode" class="camera-placeholder">
            <div class="placeholder-icon">●</div>
            <p>请开启摄像头或上传眼部图片</p>
          </div>
          <canvas ref="canvasRef" class="hidden-canvas" />
        </div>

        <div class="actions">
          <input
            ref="fileInputRef"
            type="file"
            accept="image/*"
            class="hidden-input"
            @change="onFileSelected"
          />
          <template v-if="cropMode">
            <div class="action-row action-row--primary">
              <el-button type="primary" size="large" :loading="loading" @click="applyCropAndAnalyze">
                确认框选并识别
              </el-button>
              <el-button size="large" @click="clearPreview">取消</el-button>
            </div>
          </template>
          <template v-else>
            <div class="action-row action-row--primary">
              <el-button type="primary" size="large" @click="capturePhoto">
                拍照
              </el-button>
              <el-button size="large" @click="fileInputRef?.click()">选择文件</el-button>
            </div>
            <div v-if="previewUrl && !manualMode" class="action-row action-row--secondary">
              <el-button
                v-if="currentFile && previewUrl"
                size="large"
                :loading="loading"
                @click="reanalyze"
              >
                重新识别
              </el-button>
              <el-button
                v-if="originalFile"
                size="large"
                @click="reenterCropMode"
              >
                重新框选
              </el-button>
              <el-button
                size="large"
                :disabled="!previewUrl || !manualParams"
                @click="startManualAdjust"
              >
                人工校准
              </el-button>
              <el-button size="large" @click="clearPreview">清除预览</el-button>
            </div>
            <div
              v-if="previewUrl && manualMode && manualParams"
              class="action-row action-row--manual-mobile"
            >
              <el-button
                type="primary"
                size="large"
                :loading="manualLoading"
                @click="analyzeWithManualParams"
              >
                按校准区域重新识别
              </el-button>
            </div>
            <div class="action-settings" v-if="isMobile">
              <el-checkbox v-model="skipQuality" class="quality-toggle">
                跳过质量检测
              </el-checkbox>
            </div>
          </template>
        </div>

        <div v-if="cropMode" class="crop-panel">
          <div class="crop-heading">
            <strong>框选眼部区域</strong>
            <span>拖动框体移动位置，拖动四角调整大小，仅识别框内图像</span>
          </div>
        </div>

        <div v-if="manualMode && manualParams" class="manual-panel">
          <div class="manual-heading">
            <strong>人工校准</strong>
            <span>蓝圈=瞳孔，红圈=环带内缘，绿圈=环带外缘</span>
          </div>
          <div class="manual-fields">
            <label>
              中心 X
              <input
                type="number"
                :value="Math.round(manualParams.center_x)"
                @input="updateManualNumber('center_x', ($event.target as HTMLInputElement).value)"
              />
            </label>
            <label>
              中心 Y
              <input
                type="number"
                :value="Math.round(manualParams.center_y)"
                @input="updateManualNumber('center_y', ($event.target as HTMLInputElement).value)"
              />
            </label>
            <label>
              瞳孔半径
              <input
                type="number"
                :value="Math.round(manualParams.pupil_radius)"
                @input="updateManualNumber('pupil_radius', ($event.target as HTMLInputElement).value)"
              />
            </label>
            <label>
              内环半径
              <input
                type="number"
                :value="Math.round(manualParams.inner_radius)"
                @input="updateManualNumber('inner_radius', ($event.target as HTMLInputElement).value)"
              />
            </label>
            <label>
              外环半径
              <input
                type="number"
                :value="Math.round(manualParams.outer_radius)"
                @input="updateManualNumber('outer_radius', ($event.target as HTMLInputElement).value)"
              />
            </label>
          </div>
          <el-button
            class="manual-submit-btn"
            type="primary"
            :loading="manualLoading"
            @click="analyzeWithManualParams"
          >
            按校准区域重新识别
          </el-button>
        </div>

        <div class="capture-tips">
          <div class="tip-item">
            <span>01</span>
            <p>拍照或上传后，先框选单眼所在区域再识别。</p>
          </div>
          <div class="tip-item">
            <span>02</span>
            <p>框选范围尽量贴近眼部，瞳孔位于框内中心附近。</p>
          </div>
          <div class="tip-item">
            <span>03</span>
            <p>全脸或大图可先框选缩小范围，提高识别准确率。</p>
          </div>
        </div>
      </article>

      <article ref="resultCardRef" class="clinical-card result-card">
        <header class="card-header">
          <div>
            <span v-if="isMobile" class="section-kicker">Step 02</span>
            <span v-else class="section-kicker section-kicker--desktop">分析模块</span>
            <h2>分析结果</h2>
          </div>
          <button
            type="button"
            class="download-report-btn"
            :disabled="!result || loading"
            aria-label="下载分析报告"
            @click="downloadResultReport"
          >
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 3v12" />
              <path d="M7 10l5 5 5-5" />
              <path d="M5 21h14" />
            </svg>
          </button>
        </header>

        <div v-if="loading" class="result-loading">
          <div class="loader-ring"></div>
          <span>正在分析虹膜颜色，请稍候...</span>
        </div>

        <div v-else-if="errorMsg" class="analysis-notice" :class="{ 'is-quality': qualityCheckFailed }">
          <div class="analysis-notice-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8">
              <circle cx="12" cy="12" r="9" />
              <path d="M12 8v5" stroke-linecap="round" />
              <circle cx="12" cy="16.5" r="0.8" fill="currentColor" stroke="none" />
            </svg>
          </div>
          <div class="analysis-notice-content">
            <strong>{{ errorNoticeTitle }}</strong>
            <p>{{ errorNoticeDetail }}</p>
            <button
              v-if="qualityCheckFailed"
              type="button"
              class="analysis-notice-action"
              :disabled="loading"
              @click="skipQualityAndReanalyze"
            >
              {{ loading ? '正在重新识别…' : '跳过质量检测并重新识别' }}
            </button>
          </div>
        </div>

        <div v-else-if="result" class="result-panel">
          <div ref="diagnosisRef" class="diagnosis-summary">
            <div class="color-display">
              <span>虹膜颜色</span>
              <strong>{{ result.iris_color?.label ?? '未知' }}</strong>
            </div>
            <div class="grade-display">
              <span class="grade-label">颜色等级</span>
              <span class="grade-number">Grade {{ result.grade ?? '-' }}</span>
              <span class="grade-note">{{ gradeLabels[result.grade ?? 0] ?? '待判定' }}</span>
            </div>
          </div>

          <div class="metrics-grid">
            <div class="metric-card">
              <span>L* 明度</span>
              <strong>{{ result.lab?.L?.toFixed(2) ?? '-' }}</strong>
            </div>
            <div class="metric-card">
              <span>a*</span>
              <strong>{{ result.lab?.a?.toFixed(2) ?? '-' }}</strong>
            </div>
            <div class="metric-card">
              <span>b*</span>
              <strong>{{ result.lab?.b?.toFixed(2) ?? '-' }}</strong>
            </div>
          </div>

          <div v-if="result.debug_images" class="evidence-section">
            <div class="section-title">
              <h3>识别依据</h3>
              <span>定位、环带与有效采样区域</span>
            </div>
            <div class="evidence-grid">
              <div
                v-for="item in [
                  ['01_pupil_localization', '瞳孔定位'],
                  ['02_iris_ring', '虹膜环带'],
                  ['04_valid_samples', '最终取色像素'],
                ]"
                :key="item[0]"
                class="evidence-card"
              >
                <div class="evidence-title">{{ item[1] }}</div>
                <img
                  v-if="result.debug_images[item[0]]"
                  :src="`data:image/jpeg;base64,${result.debug_images[item[0]]}`"
                  :alt="item[1]"
                />
              </div>
            </div>
          </div>
        </div>

        <div v-else class="empty-state">
          <div class="empty-icon">+</div>
          <h3>等待检测图像</h3>
        </div>
      </article>
    </section>
    </div>

    <footer v-if="!isMobile" class="desktop-statusbar">
      <span>豪赋医疗 · 虹膜颜色识别<span v-if="appVersion" class="desktop-version"> · v{{ appVersion }}</span></span>
      <span>{{ loading ? '正在分析…' : result ? '分析完成' : '等待图像输入' }}</span>
    </footer>

    <!-- 摄像头不可用提示（仅用户点击拍照时触发） -->
    <teleport to="body">
      <div
        v-if="cameraErrorOpen"
        class="app-dialog-backdrop"
        @click.self="closeCameraErrorDialog"
      >
        <div class="app-dialog" role="alertdialog" aria-labelledby="camera-dialog-title">
          <header class="app-dialog-header">
            <div class="app-dialog-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8">
                <path d="M3 7h4l2-3h6l2 3h4" />
                <rect x="3" y="7" width="18" height="13" rx="2" />
                <path d="M12 11v4" stroke-linecap="round" />
              </svg>
            </div>
            <h3 id="camera-dialog-title">摄像头不可用</h3>
          </header>
          <p class="app-dialog-body">
            无法访问摄像头，请检查系统或浏览器权限设置；您也可以直接上传眼部图片进行分析。
          </p>
          <footer class="app-dialog-footer">
            <button type="button" class="app-dialog-btn app-dialog-btn--ghost" @click="closeCameraErrorDialog">
              知道了
            </button>
            <button type="button" class="app-dialog-btn app-dialog-btn--primary" @click="openFilePickerFromCameraDialog">
              选择文件
            </button>
          </footer>
        </div>
      </div>
    </teleport>
  </main>
</template>

<style scoped>
.page-shell {
  min-height: 100vh;
  padding: 32px 24px 56px;
  padding-left: max(24px, env(safe-area-inset-left));
  padding-right: max(24px, env(safe-area-inset-right));
  background:
    radial-gradient(circle at top left, rgba(38, 132, 255, 0.14), transparent 34%),
    linear-gradient(135deg, #f5f9fc 0%, #eef5f8 48%, #f9fbfd 100%);
  color: #1f2d3d;
}

.hero {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  max-width: 1180px;
  margin: 0 auto 24px;
  padding: 28px 32px;
  border: 1px solid rgba(32, 112, 171, 0.12);
  border-radius: 28px;
  background: rgba(255, 255, 255, 0.82);
  box-shadow: 0 24px 70px rgba(31, 76, 112, 0.1);
  backdrop-filter: blur(10px);
}

.hero-copy {
  max-width: 760px;
}

.eyebrow,
.section-kicker {
  display: inline-flex;
  margin-bottom: 10px;
  color: #1876a9;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.hero h1 {
  margin: 0 0 10px;
  color: #12263a;
  font-size: clamp(30px, 4vw, 46px);
  line-height: 1.12;
}

.hero p {
  margin: 0;
  color: #5f7083;
  font-size: 16px;
  line-height: 1.7;
}

.hero-status {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  flex: 0 0 auto;
  padding: 12px 16px;
  border: 1px solid #d6eaf3;
  border-radius: 999px;
  background: #f7fcff;
  color: #33556f;
  font-size: 14px;
  font-weight: 600;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #a8b8c7;
}

.status-dot.active {
  background: #19b57a;
  box-shadow: 0 0 0 6px rgba(25, 181, 122, 0.14);
}

.hero-actions {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  flex: 0 0 auto;
}

/* 顶栏设置按钮与展开面板默认仅移动端可见 */
.settings-btn,
.mobile-settings-panel,
.mobile-settings-backdrop {
  display: none;
}

.workflow-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.08fr) minmax(360px, 0.92fr);
  gap: 24px;
  max-width: 1180px;
  margin: 0 auto;
}

.clinical-card {
  min-width: 0;
  padding: 24px;
  border: 1px solid rgba(32, 112, 171, 0.12);
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.94);
  box-shadow: 0 18px 52px rgba(36, 82, 118, 0.09);
}

.result-card {
  /* 自动滚动到结果时顶部留出呼吸空间 */
  scroll-margin-top: 12px;
}

.card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 18px;
}

.card-header h2 {
  margin: 0;
  color: #1b3348;
  font-size: 22px;
}

.download-report-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 42px;
  height: 42px;
  border: 1px solid #d6eaf3;
  border-radius: 12px;
  background: #f7fcff;
  color: #146c9c;
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease, transform 0.12s ease;
}

.download-report-btn:hover:not(:disabled) {
  background: #eaf6fb;
  color: #0f5a82;
}

.download-report-btn:active:not(:disabled) {
  transform: scale(0.96);
}

.download-report-btn:disabled {
  color: #a8b8c7;
  background: #f6f9fb;
  border-color: #e3edf3;
  cursor: not-allowed;
}

.camera-box {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  min-height: 360px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 22px;
  background:
    linear-gradient(rgba(12, 28, 45, 0.42), rgba(12, 28, 45, 0.42)),
    repeating-linear-gradient(90deg, transparent, transparent 32px, rgba(255, 255, 255, 0.04) 33px),
    #071521;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.06);
}

.camera-box::after {
  position: absolute;
  inset: 16px;
  pointer-events: none;
  content: '';
  border: 1px solid rgba(98, 202, 255, 0.22);
  border-radius: 18px;
}

.camera-flip-btn {
  position: absolute;
  top: 14px;
  right: 14px;
  z-index: 2;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  border: 1px solid rgba(98, 202, 255, 0.35);
  border-radius: 999px;
  background: rgba(7, 21, 33, 0.72);
  color: #e8f6ff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  backdrop-filter: blur(8px);
  transition: background 0.15s ease, transform 0.12s ease;
}

.camera-flip-btn:hover:not(:disabled) {
  background: rgba(20, 108, 156, 0.88);
}

.camera-flip-btn:active:not(:disabled) {
  transform: scale(0.96);
}

.camera-flip-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.camera-video,
.preview-img,
.adjust-canvas,
.crop-canvas {
  display: block;
  width: 100%;
  height: auto;
  max-height: 560px;
  object-fit: contain;
}

.adjust-canvas,
.crop-canvas {
  cursor: default;
  touch-action: none;
}

.camera-placeholder {
  position: relative;
  z-index: 1;
  display: grid;
  place-items: center;
  gap: 10px;
  color: #c8d8e7;
  text-align: center;
}

.placeholder-icon {
  display: grid;
  place-items: center;
  width: 74px;
  height: 74px;
  border: 1px solid rgba(111, 204, 255, 0.38);
  border-radius: 50%;
  color: #62caff;
  font-size: 20px;
  box-shadow: 0 0 36px rgba(98, 202, 255, 0.18);
}

.hidden-canvas,
.hidden-input {
  display: none;
}

.actions {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 18px;
}

.action-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
}

.action-row :deep(.el-button) {
  margin-left: 0;
}

.action-row--manual-mobile {
  display: none;
}

.action-settings {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px 18px;
  padding: 12px 14px;
  border: 1px solid #e2eef5;
  border-radius: 14px;
  background: #f8fbfd;
}

.quality-toggle {
  min-height: 40px;
}

.manual-panel {
  display: flex;
  gap: 14px;
  padding: 16px 18px;
  border: 1px solid #d4dee8;
  border-radius: 10px;
  background: linear-gradient(180deg, #fff 0%, #f7fafc 100%);
  box-shadow: 0 4px 14px rgba(36, 82, 118, 0.06);
}

.analysis-notice.is-quality {
  border-color: #c8d9e8;
  background: linear-gradient(180deg, #f8fbfe 0%, #f2f7fb 100%);
}

.analysis-notice-icon {
  display: grid;
  place-items: center;
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border-radius: 10px;
  background: #e8f2f8;
  color: #1876a9;
}

.analysis-notice.is-quality .analysis-notice-icon {
  background: #e3edf5;
  color: #156592;
}

.analysis-notice-content {
  flex: 1;
  min-width: 0;
}

.analysis-notice-content strong {
  display: block;
  margin-bottom: 6px;
  color: #1a3348;
  font-size: 15px;
  font-weight: 700;
}

.analysis-notice-content p {
  margin: 0;
  color: #4a6278;
  font-size: 13px;
  line-height: 1.6;
}

.analysis-notice-action {
  margin-top: 12px;
  padding: 8px 14px;
  border: 1px solid #b8cedf;
  border-radius: 6px;
  background: #fff;
  color: #1876a9;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.analysis-notice-action:hover:not(:disabled) {
  border-color: #1876a9;
  background: #eef6fb;
  color: #125a7f;
}

.analysis-notice-action:disabled {
  opacity: 0.65;
  cursor: not-allowed;
}

/* 软件风格小弹窗（teleport 至 body，仍受 scoped 作用） */
.app-dialog-backdrop {
  position: fixed;
  inset: 0;
  z-index: 2000;
  display: grid;
  place-items: center;
  padding: 20px;
  background: rgba(10, 21, 32, 0.42);
  backdrop-filter: blur(4px);
}

.app-dialog {
  width: min(100%, 380px);
  overflow: hidden;
  border: 1px solid #c5d3df;
  border-radius: 10px;
  background: #fff;
  box-shadow: 0 18px 48px rgba(16, 40, 62, 0.22);
}

.app-dialog-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 16px;
  border-bottom: 1px solid #e2eaf1;
  background: linear-gradient(180deg, #f6f9fc 0%, #eef3f7 100%);
}

.app-dialog-icon {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: #e3edf5;
  color: #1876a9;
}

.app-dialog-header h3 {
  margin: 0;
  color: #1a3348;
  font-size: 15px;
  font-weight: 700;
}

.app-dialog-body {
  margin: 0;
  padding: 16px;
  color: #4a6278;
  font-size: 13px;
  line-height: 1.65;
}

.app-dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 12px 16px 14px;
  border-top: 1px solid #e8eef3;
  background: #fafbfc;
}

.app-dialog-btn {
  min-width: 88px;
  padding: 8px 14px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.app-dialog-btn--ghost {
  border: 1px solid #c5d3df;
  background: #fff;
  color: #4a6278;
}

.app-dialog-btn--ghost:hover {
  border-color: #1876a9;
  background: #f3f8fb;
  color: #1876a9;
}

.app-dialog-btn--primary {
  border: 1px solid #1876a9;
  background: #1876a9;
  color: #fff;
}

.app-dialog-btn--primary:hover {
  border-color: #125a7f;
  background: #125a7f;
}

.manual-panel {
  margin-top: 18px;
  padding: 18px;
  border: 1px solid #dbe9f1;
  border-radius: 18px;
  background: #f8fbfd;
}

.crop-panel {
  margin-top: 18px;
  padding: 16px 18px;
  border: 1px solid #cfe3f0;
  border-radius: 18px;
  background: linear-gradient(180deg, #f3faff 0%, #f8fbfd 100%);
}

.crop-heading {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  color: #536a7b;
  font-size: 13px;
}

.crop-heading strong {
  color: #1b3348;
  font-size: 15px;
}

.manual-heading {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 14px;
  color: #536a7b;
  font-size: 13px;
}

.manual-heading strong {
  color: #1b3348;
  font-size: 15px;
}

.manual-fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 12px;
  margin-bottom: 14px;
}

.manual-fields label {
  color: #516677;
  font-size: 12px;
  font-weight: 600;
}

.manual-fields input {
  box-sizing: border-box;
  width: 100%;
  margin-top: 6px;
  padding: 10px 12px;
  /* 16px 可避免 iOS 聚焦时自动放大页面 */
  font-size: 16px;
  border: 1px solid #cfdde7;
  border-radius: 10px;
  outline: none;
  background: #fff;
  color: #1f2d3d;
}

.manual-fields input:focus {
  border-color: #2f92c4;
  box-shadow: 0 0 0 3px rgba(47, 146, 196, 0.12);
}

.capture-tips {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 18px;
}

.tip-item {
  min-height: 88px;
  padding: 14px;
  border: 1px solid #e0edf4;
  border-radius: 16px;
  background: #fbfdff;
}

.tip-item span {
  color: #1876a9;
  font-size: 12px;
  font-weight: 800;
}

.tip-item p {
  margin: 8px 0 0;
  color: #5f7083;
  font-size: 13px;
  line-height: 1.55;
}

.result-loading,
.empty-state {
  display: grid;
  place-items: center;
  min-height: 360px;
  color: #5f7083;
  text-align: center;
}

.result-loading {
  gap: 16px;
}

.loader-ring {
  width: 42px;
  height: 42px;
  border: 4px solid #d9edf7;
  border-top-color: #1876a9;
  border-radius: 50%;
  animation: spin 0.9s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.empty-icon {
  display: grid;
  place-items: center;
  width: 70px;
  height: 70px;
  margin-bottom: 12px;
  border-radius: 22px;
  background: linear-gradient(135deg, #e8f5fb, #f7fbfd);
  color: #1876a9;
  font-size: 34px;
  font-weight: 300;
}

.empty-state h3 {
  margin: 0 0 8px;
  color: #1b3348;
}

.empty-state p {
  max-width: 320px;
  margin: 0;
  line-height: 1.65;
}

.diagnosis-summary {
  display: grid;
  grid-template-columns: 1fr;
  gap: 14px;
  margin-bottom: 16px;
}

.color-display,
.grade-display,
.metric-card {
  border: 1px solid #dceaf2;
  border-radius: 18px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
}

.color-display {
  padding: 18px;
  text-align: center;
}

.color-display span,
.grade-label,
.metric-card span {
  display: block;
  color: #687d8f;
  font-size: 13px;
  font-weight: 600;
}

.color-display strong {
  display: block;
  margin-top: 8px;
  color: #17324a;
  font-size: 26px;
  text-align: center;
}

.grade-display {
  padding: 20px;
  text-align: center;
}

.grade-number {
  display: block;
  margin-top: 8px;
  color: #146c9c;
  font-size: clamp(40px, 8vw, 58px);
  font-weight: 800;
  line-height: 1;
}

.grade-note {
  display: inline-flex;
  margin-top: 10px;
  padding: 6px 12px;
  border-radius: 999px;
  background: #eaf6fb;
  color: #23637f;
  font-size: 14px;
  font-weight: 700;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}

.metric-card {
  padding: 14px;
}

.metric-card strong {
  display: block;
  margin-top: 8px;
  color: #1b3348;
  font-size: 22px;
}

.reason-list {
  margin-bottom: 18px;
}

.section-title {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.section-title h3 {
  margin: 0;
  color: #1b3348;
  font-size: 16px;
}

.section-title span {
  color: #75889a;
  font-size: 12px;
}

.evidence-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.evidence-card {
  overflow: hidden;
  border: 1px solid #dfeaf2;
  border-radius: 16px;
  background: #fff;
}

.evidence-title {
  padding: 10px 12px;
  color: #526879;
  font-size: 12px;
  font-weight: 700;
  background: #f5f9fc;
}

.evidence-card img {
  display: block;
  width: 100%;
  background: #071521;
}

/* 顶栏设置面板展开/收起动画 */
.settings-slide-enter-active,
.settings-slide-leave-active {
  transition: transform 0.22s ease, opacity 0.22s ease;
}

.settings-slide-enter-from,
.settings-slide-leave-to {
  transform: translateY(-10px);
  opacity: 0;
}

/* ============ 桌面端 / Electron 专业软件布局（≥641px，不影响手机） ============ */
@media (min-width: 641px) {
  .page-shell.layout-desktop {
    display: flex;
    flex-direction: column;
    height: 100vh;
    min-height: 100vh;
    padding: 0;
    overflow: hidden;
    background: #dfe4ea;
  }

  .desktop-titlebar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    height: 52px;
    padding: 0 18px;
    border-bottom: 1px solid #0f2233;
    background: linear-gradient(180deg, #243b52 0%, #1a2f42 100%);
    color: #eef4f8;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
  }

  .desktop-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
  }

  .desktop-brand-mark {
    display: none;
  }

  .desktop-brand-text {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }

  .desktop-brand-text strong {
    color: #fff;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.02em;
    white-space: nowrap;
  }

  .desktop-brand-text span {
    color: rgba(220, 232, 240, 0.72);
    font-size: 11px;
    letter-spacing: 0.04em;
  }

  .desktop-titlebar-tools {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
  }

  .desktop-status-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 6px;
    background: rgba(0, 0, 0, 0.18);
    color: rgba(232, 241, 247, 0.92);
    font-size: 12px;
    font-weight: 600;
  }

  .desktop-setting-toggle {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: rgba(232, 241, 247, 0.88);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    user-select: none;
  }

  .workspace-shell {
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  .layout-desktop .workflow-grid {
    grid-template-columns: minmax(0, 1fr) 400px;
    gap: 0;
    max-width: none;
    height: 100%;
    margin: 0;
  }

  .layout-desktop .clinical-card {
    display: flex;
    flex-direction: column;
    min-height: 0;
    height: 100%;
    padding: 16px 18px;
    border: none;
    border-radius: 0;
    background: #f4f6f8;
    box-shadow: none;
    overflow: auto;
  }

  .layout-desktop .capture-card {
    border-right: 1px solid #c5ced8;
    background: #eef1f4;
  }

  .layout-desktop .result-card {
    background: #fafbfc;
    border-left: 1px solid #d8dee6;
  }

  .layout-desktop .card-header {
    align-items: center;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid #d8dee6;
  }

  .layout-desktop .card-header h2 {
    font-size: 16px;
    font-weight: 700;
    color: #1a3348;
  }

  .layout-desktop .section-kicker--desktop {
    margin-bottom: 4px;
    color: #5a7388;
    font-size: 10px;
    letter-spacing: 0.06em;
    text-transform: none;
  }

  .layout-desktop .camera-box {
    flex: 1;
    min-height: 320px;
    border: 1px solid #2a4055;
    border-radius: 8px;
    background: #0a1520;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
  }

  .layout-desktop .camera-box::after {
    inset: 10px;
    border-radius: 6px;
    border-color: rgba(98, 202, 255, 0.16);
  }

  .layout-desktop .camera-video,
  .layout-desktop .preview-img,
  .layout-desktop .adjust-canvas,
  .layout-desktop .crop-canvas {
    max-height: none;
    height: 100%;
    min-height: 280px;
  }

  .layout-desktop .actions {
    flex-shrink: 0;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid #d8dee6;
  }

  .layout-desktop .action-row :deep(.el-button) {
    min-height: 36px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
  }

  .layout-desktop .action-row--primary :deep(.el-button--primary) {
    background: #1876a9;
    border-color: #1876a9;
  }

  .layout-desktop .capture-tips {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid #d8dee6;
  }

  .layout-desktop .tip-item {
    padding: 10px 12px;
    border: 1px solid #d8dee6;
    border-radius: 6px;
    background: #fff;
  }

  .layout-desktop .tip-item span {
    color: #1876a9;
    font-size: 11px;
    font-weight: 800;
  }

  .layout-desktop .tip-item p {
    margin: 6px 0 0;
    color: #4a6278;
    font-size: 12px;
    line-height: 1.5;
  }

  .layout-desktop .manual-panel,
  .layout-desktop .crop-panel {
    border-radius: 6px;
    border-color: #cfd8e2;
    background: #fff;
  }

  .layout-desktop .diagnosis-summary {
    border-radius: 8px;
    border: 1px solid #cfd8e2;
    background: linear-gradient(180deg, #fff 0%, #f6f9fb 100%);
  }

  .layout-desktop .grade-number {
    font-size: 28px;
  }

  .layout-desktop .metrics-grid {
    gap: 10px;
  }

  .layout-desktop .metric-card {
    border-radius: 6px;
    border: 1px solid #d8dee6;
    background: #fff;
  }

  .layout-desktop .evidence-grid {
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }

  .layout-desktop .evidence-card {
    border-radius: 6px;
  }

  .layout-desktop .empty-state,
  .layout-desktop .result-loading {
    border: 1px dashed #c5ced8;
    border-radius: 8px;
    background: #fff;
  }

  .desktop-statusbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    height: 28px;
    padding: 0 14px;
    border-top: 1px solid #b8c4d0;
    background: linear-gradient(180deg, #e8edf2 0%, #dde4eb 100%);
    color: #4a6278;
    font-size: 11px;
    font-weight: 600;
  }
}

@media (max-width: 980px) {
  .page-shell {
    padding: 22px 16px 40px;
    padding-left: max(16px, env(safe-area-inset-left));
    padding-right: max(16px, env(safe-area-inset-right));
  }

  .hero {
    align-items: flex-start;
    flex-direction: column;
    padding: 24px;
  }

  .workflow-grid {
    grid-template-columns: 1fr;
  }

  .camera-box {
    min-height: 300px;
  }
}

/* ============ 移动端 App 化布局（固定顶栏 + 固定底部操作栏） ============ */
@media (max-width: 640px) {
  .page-shell {
    /* 为固定顶栏与底部操作栏预留空间，中间内容区独立滚动 */
    padding: calc(60px + env(safe-area-inset-top)) 0 calc(150px + env(safe-area-inset-bottom));
    padding-left: max(12px, env(safe-area-inset-left));
    padding-right: max(12px, env(safe-area-inset-right));
  }

  /* —— 顶部应用导航栏 —— */
  .hero {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 60;
    align-items: center;
    flex-direction: row;
    justify-content: space-between;
    gap: 12px;
    max-width: none;
    margin: 0;
    padding: 10px 16px;
    padding-top: calc(10px + env(safe-area-inset-top));
    padding-left: max(16px, env(safe-area-inset-left));
    padding-right: max(16px, env(safe-area-inset-right));
    border: none;
    border-bottom: 1px solid rgba(32, 112, 171, 0.14);
    border-radius: 0;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 4px 18px rgba(31, 76, 112, 0.08);
    backdrop-filter: saturate(1.4) blur(14px);
  }

  .hero-copy {
    display: flex;
    align-items: center;
    gap: 10px;
    max-width: none;
  }

  /* 手机顶栏 Logo 由 BrandLogo 组件提供 */
  .hero-copy::before {
    display: none;
  }

  .hero .eyebrow {
    display: none;
  }

  .hero h1 {
    margin: 0;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.01em;
  }

  .hero-status {
    flex: 0 0 auto;
    gap: 6px;
    padding: 6px 12px;
    font-size: 12px;
  }

  .workflow-grid {
    gap: 12px;
  }

  /* —— 内容卡片：更扁平、专业 —— */
  .clinical-card {
    padding: 16px 14px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(36, 82, 118, 0.06);
  }

  .card-header {
    align-items: center;
    margin-bottom: 14px;
  }

  .card-header h2 {
    font-size: 19px;
  }

  .section-kicker {
    margin-bottom: 6px;
    font-size: 11px;
  }

  /* 相机/预览区域使用视口高度，单眼特写更易对准 */
  .camera-box {
    min-height: min(60vh, 440px);
    border-radius: 14px;
  }

  .camera-video,
  .preview-img,
  .adjust-canvas,
  .crop-canvas {
    max-height: min(60vh, 440px);
  }

  /* —— 顶栏设置按钮 —— */
  .settings-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    padding: 0;
    border: 1px solid #d6eaf3;
    border-radius: 12px;
    background: #f7fcff;
    color: #2a5b78;
    cursor: pointer;
    transition: transform 0.12s ease, background 0.15s ease, color 0.15s ease;
  }

  .settings-btn:active {
    transform: scale(0.94);
  }

  .settings-btn.active {
    border-color: #2f92c4;
    background: #2f92c4;
    color: #fff;
  }

  /* —— 顶栏展开的设置面板 —— */
  .mobile-settings-panel {
    position: fixed;
    top: calc(56px + env(safe-area-inset-top));
    left: max(12px, env(safe-area-inset-left));
    right: max(12px, env(safe-area-inset-right));
    z-index: 59;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 16px;
    border: 1px solid rgba(32, 112, 171, 0.14);
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.98);
    box-shadow: 0 16px 40px rgba(31, 76, 112, 0.18);
    backdrop-filter: saturate(1.4) blur(14px);
  }

  .mobile-settings-backdrop {
    position: fixed;
    inset: 0;
    z-index: 58;
    display: block;
    background: rgba(7, 21, 33, 0.32);
  }

  .settings-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .settings-label {
    color: #2b4256;
    font-size: 14px;
    font-weight: 600;
  }

  /* —— 底部固定操作栏：主操作 + 子操作 —— */
  .actions {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 60;
    display: flex;
    flex-direction: column-reverse;
    gap: 8px;
    margin: 0;
    padding: 10px 14px;
    padding-bottom: calc(10px + env(safe-area-inset-bottom));
    padding-left: max(14px, env(safe-area-inset-left));
    padding-right: max(14px, env(safe-area-inset-right));
    border-top: 1px solid rgba(32, 112, 171, 0.14);
    background: rgba(255, 255, 255, 0.94);
    box-shadow: 0 -4px 18px rgba(31, 76, 112, 0.1);
    backdrop-filter: saturate(1.4) blur(14px);
  }

  /* 主操作：醒目 */
  .action-row--primary {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin: 0;
  }

  .action-row--primary :deep(.el-button) {
    width: 100%;
    min-height: 50px;
    margin-left: 0;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    transition: transform 0.12s ease;
  }

  /* 子操作：识别后出现，更小、视觉次一级 */
  .action-row--secondary {
    display: flex;
    gap: 8px;
    margin: 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(32, 112, 171, 0.1);
  }

  .action-row--secondary :deep(.el-button) {
    flex: 1;
    min-width: 0;
    min-height: 38px;
    margin-left: 0;
    padding: 0 6px;
    border-radius: 10px;
    color: #436076;
    background: #f3f8fb;
    border-color: #d9e7f0;
    font-size: 12px;
    font-weight: 500;
    transition: transform 0.12s ease;
  }

  .action-row--secondary :deep(.el-button.is-disabled) {
    color: #a7b8c5;
    background: #f6f9fb;
  }

  .action-row--manual-mobile {
    display: grid;
    grid-template-columns: 1fr;
    gap: 10px;
    margin: 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(32, 112, 171, 0.1);
  }

  .action-row--manual-mobile :deep(.el-button) {
    width: 100%;
    min-height: 50px;
    margin-left: 0;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    transition: transform 0.12s ease;
  }

  .capture-card.manual-mode-active .action-row--primary {
    display: none;
  }

  .manual-panel .manual-submit-btn {
    display: none;
  }

  /* 触控按压反馈，营造原生点击手感 */
  .actions :deep(.el-button:active) {
    transform: scale(0.96);
  }

  /* 卡片内的设置整块已移至顶栏，移动端隐藏 */
  .action-settings {
    display: none;
  }

  /* 移动端隐藏采集提示，节省纵向空间 */
  .capture-tips {
    display: none;
  }

  .evidence-grid {
    grid-template-columns: 1fr;
  }

  .metrics-grid {
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .metric-card {
    padding: 12px 8px;
    text-align: center;
  }

  .metric-card strong {
    font-size: 18px;
  }

  .manual-fields {
    grid-template-columns: 1fr 1fr;
  }

  .result-loading,
  .empty-state {
    min-height: 200px;
  }
}
</style>
