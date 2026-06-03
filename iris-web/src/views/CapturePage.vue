<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { analyzeIris, analyzeIrisManual, type AnalysisResult, type DetectionInfo } from '../api/iris'

const videoRef = ref<HTMLVideoElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const adjustCanvasRef = ref<HTMLCanvasElement | null>(null)
const fileInputRef = ref<HTMLInputElement | null>(null)

const cameraActive = ref(false)
const previewUrl = ref<string | null>(null)
const loading = ref(false)
const manualLoading = ref(false)
const result = ref<AnalysisResult | null>(null)
const errorMsg = ref('')
const currentFile = ref<File | null>(null)
const manualMode = ref(false)
const manualParams = ref<DetectionInfo | null>(null)
const adjustCursor = ref('default')
const skipQuality = ref(false)

let mediaStream: MediaStream | null = null
let adjustImage: HTMLImageElement | null = null
type DragMode = 'move' | 'pupil_radius' | 'inner_radius' | 'outer_radius'
let dragMode: DragMode | null = null
let dragOffset = { x: 0, y: 0 }

const gradeLabels: Record<number, string> = {
  1: '最浅',
  2: '较浅',
  3: '中等',
  4: '较深',
  5: '最深',
}

async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
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
  } catch {
    ElMessage.error('无法访问摄像头，请检查浏览器权限或使用文件上传')
  }
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop())
    mediaStream = null
  }
  cameraActive.value = false
}

function capturePhoto() {
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
    if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = URL.createObjectURL(blob)
    uploadBlob(blob, 'capture.jpg')
  }, 'image/jpeg', 0.92)
}

function onFileSelected(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return

  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = URL.createObjectURL(file)
  uploadFile(file)
  input.value = ''
}

async function uploadBlob(blob: Blob, filename: string) {
  const file = new File([blob], filename, { type: 'image/jpeg' })
  await uploadFile(file)
}

async function uploadFile(file: File) {
  loading.value = true
  result.value = null
  errorMsg.value = ''
  currentFile.value = file
  manualMode.value = false
  manualParams.value = null
  adjustImage = null

  try {
    const data = await analyzeIris(file, skipQuality.value)
    if (data.success === false) {
      errorMsg.value = data.error || '分析失败'
    } else {
      result.value = data
      initManualParams(data)
      manualMode.value = false
    }
  } catch (err: unknown) {
    if (axiosIsError(err) && err.response?.data) {
      const detail = err.response.data as AnalysisResult & { detail?: unknown }
      errorMsg.value =
        detail.error ||
        (detail.detail === 'no_iris_detected'
          ? '未识别到虹膜，请使用单眼特写（瞳孔居中、对焦清晰）'
          : typeof detail.detail === 'string'
            ? detail.detail
            : JSON.stringify(detail.detail ?? detail))
    } else {
      errorMsg.value = '请求失败，请确认 iris-api 与 iris-vision 已启动'
    }
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
  })
}

function clearPreview() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = null
  currentFile.value = null
  manualMode.value = false
  manualParams.value = null
  adjustImage = null
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
  const uiScale = canvasUiScale()
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

function canvasDisplayMetrics() {
  const canvas = adjustCanvasRef.value
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

function canvasPoint(event: PointerEvent) {
  const canvas = adjustCanvasRef.value
  const metrics = canvasDisplayMetrics()
  if (!canvas || !metrics) return { x: 0, y: 0 }
  return {
    x: (event.clientX - metrics.left) * metrics.scaleX,
    y: (event.clientY - metrics.top) * metrics.scaleY,
  }
}

function canvasUiScale() {
  const metrics = canvasDisplayMetrics()
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
  const uiScale = canvasUiScale()
  const lineTol = Math.max(12 * uiScale, 8)
  const crossTol = Math.max(10 * uiScale, 6)
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
  adjustCursor.value = hitMode(canvasPoint(event)) ? 'grab' : 'default'
}

function onAdjustPointerDown(event: PointerEvent) {
  const point = canvasPoint(event)
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
  const point = canvasPoint(event)
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
  try {
    const data = await analyzeIrisManual(currentFile.value, manualParams.value, skipQuality.value)
    if (data.success === false) {
      errorMsg.value = data.error || '人工调整分析失败'
    } else {
      result.value = data
      initManualParams(data)
      ElMessage.success('已按人工调整区域重新识别')
      nextTick(drawAdjustCanvas)
    }
  } catch (err: unknown) {
    if (axiosIsError(err) && err.response?.data) {
      const detail = err.response.data as AnalysisResult & { detail?: unknown }
      errorMsg.value = detail.error || JSON.stringify(detail.detail ?? detail)
    } else {
      errorMsg.value = '人工调整分析请求失败'
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
})

onBeforeUnmount(() => {
  stopCamera()
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
})
</script>

<template>
  <main class="page-shell">
    <section class="hero">
      <div class="hero-copy">
        <span class="eyebrow">IRIS COLOR ANALYSIS</span>
        <h1>虹膜颜色识别</h1>
      </div>
      <div class="hero-status">
        <span class="status-dot" :class="{ active: cameraActive }"></span>
        <span>{{ cameraActive ? '摄像头已就绪' : '摄像头未接入' }}</span>
      </div>
    </section>

    <section class="workflow-grid">
      <article class="clinical-card capture-card">
        <header class="card-header">
          <div>
            <span class="section-kicker">Step 01</span>
            <h2>图像采集</h2>
          </div>
          <el-tag effect="plain" type="success">眼部特写</el-tag>
        </header>

        <div class="camera-box">
          <video
            v-show="cameraActive && !previewUrl && !manualMode"
            ref="videoRef"
            class="camera-video"
            autoplay
            playsinline
            muted
          />
          <img v-if="previewUrl && !manualMode" :src="previewUrl" alt="预览" class="preview-img" />
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
          <div v-if="!cameraActive && !previewUrl && !manualMode" class="camera-placeholder">
            <div class="placeholder-icon">●</div>
            <p>请开启摄像头或上传眼部图片</p>
          </div>
          <canvas ref="canvasRef" class="hidden-canvas" />
        </div>

        <div class="actions">
          <el-button type="primary" size="large" :loading="loading" @click="capturePhoto">
            拍照分析
          </el-button>
          <el-button size="large" @click="fileInputRef?.click()">选择文件</el-button>
          <input
            ref="fileInputRef"
            type="file"
            accept="image/*"
            class="hidden-input"
            @change="onFileSelected"
          />
          <el-button v-if="previewUrl" size="large" @click="clearPreview">清除预览</el-button>
          <el-button v-if="result?.detection && previewUrl" size="large" @click="startManualAdjust">
            人工校准区域
          </el-button>
          <el-checkbox v-model="skipQuality" class="quality-toggle">
            跳过质量检测
          </el-checkbox>
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
          <el-button type="primary" :loading="manualLoading" @click="analyzeWithManualParams">
            按校准区域重新识别
          </el-button>
        </div>

        <div class="capture-tips">
          <div class="tip-item">
            <span>01</span>
            <p>单眼特写，瞳孔尽量位于画面中心。</p>
          </div>
          <div class="tip-item">
            <span>02</span>
            <p>保持对焦清晰，避免强反光和过暗环境。</p>
          </div>
          <div class="tip-item">
            <span>03</span>
            <p>可直接上传已有眼部照片，无需拍摄全脸。</p>
          </div>
        </div>
      </article>

      <article class="clinical-card result-card">
        <header class="card-header">
          <div>
            <span class="section-kicker">Step 02</span>
            <h2>分析结果</h2>
          </div>
          <el-tag effect="plain" type="primary">自动分级</el-tag>
        </header>

        <div v-if="loading" class="result-loading">
          <div class="loader-ring"></div>
          <span>正在分析虹膜颜色，请稍候...</span>
        </div>

        <el-alert v-else-if="errorMsg" type="error" :title="errorMsg" show-icon />

        <div v-else-if="result" class="result-panel">
          <div class="diagnosis-summary">
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

          <el-descriptions class="reason-list" :column="1" border>
            <el-descriptions-item label="颜色依据">
              {{ result.iris_color?.reason ?? '-' }}
            </el-descriptions-item>
          </el-descriptions>

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
  </main>
</template>

<style scoped>
.page-shell {
  min-height: 100vh;
  padding: 32px 24px 56px;
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

.camera-video,
.preview-img,
.adjust-canvas {
  display: block;
  width: 100%;
  height: auto;
  max-height: 560px;
  object-fit: contain;
}

.adjust-canvas {
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
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-top: 18px;
}

.quality-toggle {
  min-height: 40px;
  padding: 0 8px;
}

.manual-panel {
  margin-top: 18px;
  padding: 18px;
  border: 1px solid #dbe9f1;
  border-radius: 18px;
  background: #f8fbfd;
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
  padding: 9px 10px;
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

@media (max-width: 980px) {
  .page-shell {
    padding: 22px 16px 40px;
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

@media (max-width: 640px) {
  .page-shell {
    padding: 14px 12px 28px;
  }

  .hero,
  .clinical-card {
    border-radius: 20px;
  }

  .hero {
    padding: 22px 18px;
  }

  .hero h1 {
    font-size: 30px;
  }

  .hero p {
    font-size: 14px;
  }

  .hero-status {
    width: 100%;
    justify-content: center;
  }

  .clinical-card {
    padding: 16px;
  }

  .card-header {
    align-items: stretch;
    flex-direction: column;
  }

  .camera-box {
    min-height: 220px;
    border-radius: 18px;
  }

  .actions {
    display: grid;
    grid-template-columns: 1fr;
  }

  .actions :deep(.el-button) {
    width: 100%;
    margin-left: 0;
  }

  .capture-tips,
  .metrics-grid,
  .evidence-grid {
    grid-template-columns: 1fr;
  }

  .manual-fields {
    grid-template-columns: 1fr 1fr;
  }

  .result-loading,
  .empty-state {
    min-height: 260px;
  }
}
</style>
