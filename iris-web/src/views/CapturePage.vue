<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { analyzeIris, type AnalysisResult } from '../api/iris'

const videoRef = ref<HTMLVideoElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const fileInputRef = ref<HTMLInputElement | null>(null)

const cameraActive = ref(false)
const previewUrl = ref<string | null>(null)
const loading = ref(false)
const result = ref<AnalysisResult | null>(null)
const errorMsg = ref('')

let mediaStream: MediaStream | null = null

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

  try {
    const data = await analyzeIris(file)
    if (data.success === false) {
      errorMsg.value = data.error || '分析失败'
    } else {
      result.value = data
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
              v-show="cameraActive && !previewUrl"
              ref="videoRef"
              class="camera-video"
              autoplay
              playsinline
              muted
            />
            <img v-if="previewUrl" :src="previewUrl" alt="预览" class="preview-img" />
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
            <el-button v-if="previewUrl" @click="previewUrl = null">清除预览</el-button>
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
              <el-descriptions-item label="颜色置信度">
                {{ ((result.iris_color?.confidence ?? 0) * 100).toFixed(0) }}%
              </el-descriptions-item>
              <el-descriptions-item label="颜色依据">
                {{ result.iris_color?.reason ?? '-' }}
              </el-descriptions-item>
              <el-descriptions-item label="置信度">
                {{ ((result.confidence ?? 0) * 100).toFixed(0) }}%
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
              <el-descriptions-item label="采样像素">
                {{ result.quality?.sample_pixel_count }}
              </el-descriptions-item>
              <el-descriptions-item label="清晰度">
                {{ result.quality?.blur_score?.toFixed(1) }}
              </el-descriptions-item>
            </el-descriptions>

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
.preview-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
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
</style>
