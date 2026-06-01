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

let mediaStream: MediaStream | null = null
let adjustImage: HTMLImageElement | null = null
type DragMode = 'move' | 'pupil_radius' | 'inner_radius' | 'outer_radius'
let dragMode: DragMode | null = null

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
    const data = await analyzeIris(file)
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
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(adjustImage, 0, 0)
  ctx.lineWidth = Math.max(2, canvas.width / 240)
  ctx.fillStyle = 'rgba(0, 255, 255, 0.18)'
  ctx.beginPath()
  ctx.arc(p.center_x, p.center_y, p.outer_radius, 0, Math.PI * 2)
  ctx.arc(p.center_x, p.center_y, p.inner_radius, 0, Math.PI * 2, true)
  ctx.fill()
  drawCircle(ctx, p.center_x, p.center_y, p.pupil_radius, '#2f81f7')
  drawCircle(ctx, p.center_x, p.center_y, p.inner_radius, '#ff4d4f')
  drawCircle(ctx, p.center_x, p.center_y, p.outer_radius, '#3fb950')
  ctx.strokeStyle = '#ff4d4f'
  ctx.beginPath()
  ctx.moveTo(p.center_x - 8, p.center_y)
  ctx.lineTo(p.center_x + 8, p.center_y)
  ctx.moveTo(p.center_x, p.center_y - 8)
  ctx.lineTo(p.center_x, p.center_y + 8)
  ctx.stroke()
}

function drawCircle(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string) {
  ctx.strokeStyle = color
  ctx.beginPath()
  ctx.arc(x, y, r, 0, Math.PI * 2)
  ctx.stroke()
}

function canvasPoint(event: PointerEvent) {
  const canvas = adjustCanvasRef.value
  if (!canvas) return { x: 0, y: 0 }
  const rect = canvas.getBoundingClientRect()
  return {
    x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height,
  }
}

function hitMode(point: { x: number; y: number }) {
  const p = manualParams.value
  const canvas = adjustCanvasRef.value
  if (!p || !canvas) return null
  const dist = Math.hypot(point.x - p.center_x, point.y - p.center_y)
  const tol = Math.max(8, canvas.width / 80)
  if (dist < tol) return 'move'
  if (Math.abs(dist - p.pupil_radius) < tol) return 'pupil_radius'
  if (Math.abs(dist - p.inner_radius) < tol) return 'inner_radius'
  if (Math.abs(dist - p.outer_radius) < tol) return 'outer_radius'
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
  dragMode = hitMode(canvasPoint(event))
  if (dragMode) {
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
    p.center_x = point.x
    p.center_y = point.y
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
    const data = await analyzeIrisManual(currentFile.value, manualParams.value)
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
  <div class="page">
    <header class="header">
      <h1>虹膜颜色识别</h1>
      <p class="subtitle">上传或拍摄「眼部特写」— 眼睛占满画面，系统直接分析虹膜颜色</p>
    </header>

    <el-row :gutter="24">
      <el-col :xs="24" :md="12">
        <el-card shadow="hover">
          <template #header>拍照 / 上传</template>

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
            <canvas ref="canvasRef" class="hidden-canvas" />
          </div>

          <div class="actions">
            <el-button type="primary" :loading="loading" @click="capturePhoto">
              拍照分析
            </el-button>
            <el-button @click="fileInputRef?.click()">选择文件</el-button>
            <input
              ref="fileInputRef"
              type="file"
              accept="image/*"
              class="hidden-input"
              @change="onFileSelected"
            />
            <el-button v-if="previewUrl" @click="clearPreview">清除预览</el-button>
            <el-button v-if="result?.detection && previewUrl" @click="startManualAdjust">
              人工调整识别区域
            </el-button>
          </div>

          <div v-if="manualMode && manualParams" class="manual-panel">
            <div class="manual-tip">
              蓝圈=瞳孔，红圈=环带内缘，绿圈=环带外缘。拖动中心移动，拖动圆边缩放。
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
              按人工调整区域再次识别颜色
            </el-button>
          </div>

          <el-alert
            class="tip"
            type="info"
            :closable="false"
            title="拍摄提示：单眼特写、瞳孔居中、对焦清晰、光线均匀；无需拍全脸。上传已有眼部照片亦可。"
          />
        </el-card>
      </el-col>

      <el-col :xs="24" :md="12">
        <el-card shadow="hover">
          <template #header>分析结果</template>

          <div v-if="loading" class="result-loading">
            <el-icon class="is-loading"><span class="spin">⏳</span></el-icon>
            <span>分析中...</span>
          </div>

          <el-alert v-else-if="errorMsg" type="error" :title="errorMsg" show-icon />

          <div v-else-if="result" class="result-panel">
            <div class="color-display">
              虹膜颜色：{{ result.iris_color?.label ?? '未知' }}
            </div>

            <div class="grade-display">
              <span class="grade-number">Grade {{ result.grade }}</span>
              <span class="grade-label">{{ gradeLabels[result.grade ?? 0] ?? '' }}</span>
            </div>

            <el-descriptions :column="1" border>
              <el-descriptions-item label="颜色依据">
                {{ result.iris_color?.reason ?? '-' }}
              </el-descriptions-item>
              <el-descriptions-item label="L*（明度）">
                {{ result.lab?.L?.toFixed(2) }}
              </el-descriptions-item>
              <el-descriptions-item label="a*">
                {{ result.lab?.a?.toFixed(2) }}
              </el-descriptions-item>
              <el-descriptions-item label="b*">
                {{ result.lab?.b?.toFixed(2) }}
              </el-descriptions-item>
            </el-descriptions>

            <div v-if="result.debug_images" class="evidence-section">
              <h3>识别依据</h3>
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

            <p class="message">{{ result.message }}</p>
          </div>

          <el-empty v-else description="拍照或上传图片后显示结果" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.page {
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 16px 48px;
}

.header {
  text-align: center;
  margin-bottom: 24px;
}

.header h1 {
  margin: 0 0 8px;
  font-size: 28px;
  color: #303133;
}

.subtitle {
  margin: 0;
  color: #909399;
  font-size: 14px;
}

.camera-box {
  width: 100%;
  aspect-ratio: 4 / 3;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.camera-video,
.preview-img,
.adjust-canvas {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.adjust-canvas {
  cursor: default;
  touch-action: none;
}

.hidden-canvas,
.hidden-input {
  display: none;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 16px;
}

.tip {
  margin-top: 16px;
}

.manual-panel {
  margin-top: 16px;
  padding: 12px;
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  background: #fafafa;
}

.manual-tip {
  margin-bottom: 10px;
  font-size: 13px;
  color: #606266;
}

.manual-fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}

.manual-fields label {
  font-size: 12px;
  color: #606266;
}

.manual-fields input {
  width: 100%;
  margin-top: 4px;
  padding: 6px 8px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
}

.result-loading {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 24px;
  color: #606266;
}

.grade-display {
  text-align: center;
  margin-bottom: 20px;
}

.color-display {
  text-align: center;
  margin-bottom: 12px;
  font-size: 20px;
  font-weight: 600;
  color: #303133;
}

.grade-number {
  display: block;
  font-size: 48px;
  font-weight: 700;
  color: #409eff;
}

.grade-label {
  font-size: 18px;
  color: #606266;
}

.message {
  margin-top: 16px;
  font-size: 13px;
  color: #909399;
}

.evidence-section {
  margin-top: 18px;
}

.evidence-section h3 {
  margin: 0 0 10px;
  font-size: 15px;
  color: #303133;
}

.evidence-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}

.evidence-card {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  overflow: hidden;
  background: #fff;
}

.evidence-title {
  padding: 6px 8px;
  font-size: 12px;
  color: #606266;
  background: #f5f7fa;
}

.evidence-card img {
  display: block;
  width: 100%;
  background: #000;
}
</style>
