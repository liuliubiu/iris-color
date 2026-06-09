import axios from 'axios'

export interface LabValues {
  L: number
  a: number
  b: number
}

export interface IrisColorInfo {
  code: string
  label: string
  confidence: number
  reason: string
}

export interface QualityInfo {
  blur_score: number
  overexposed_ratio: number
  eye_open: boolean
  sample_pixel_count: number
  issues: string[]
}

export interface DetectionInfo {
  center_x: number
  center_y: number
  pupil_radius: number
  inner_radius: number
  outer_radius: number
  method: string
}

export interface AnalysisResult {
  success: boolean
  quality?: QualityInfo
  lab?: LabValues
  iris_color?: IrisColorInfo
  detection?: DetectionInfo
  debug_images?: Record<string, string>
  grade?: number
  confidence?: number
  detection_method?: string
  message?: string
  error?: string
  detail?: unknown
}

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
})

const QUALITY_ISSUE_LABELS: Record<string, string> = {
  image_too_blurry: '图像模糊，请重新对焦拍摄',
  image_overexposed: '图像过曝，请避免强光直射',
  eye_closed: '未检测到睁眼，请确保眼睛睁开',
}

const ERROR_LABELS: Record<string, string> = {
  quality_check_failed: '图像质量未达标',
  no_iris_detected: '未识别到虹膜，请使用单眼特写（瞳孔居中、对焦清晰）',
  empty_file: '上传的文件为空',
  invalid_image_format: '图片格式无效，请上传常见图片格式',
  file_must_be_image: '请上传图片文件',
}

function unwrapErrorPayload(data: unknown): Record<string, unknown> {
  if (!data || typeof data !== 'object') return {}
  const obj = data as Record<string, unknown>
  const inner = obj.detail
  if (inner && typeof inner === 'object' && !Array.isArray(inner)) {
    return inner as Record<string, unknown>
  }
  if (typeof inner === 'string') {
    return { error: inner, message: ERROR_LABELS[inner] ?? inner }
  }
  return obj
}

function formatQualityMessage(issues: unknown, backendMessage?: unknown): string {
  if (typeof backendMessage === 'string' && backendMessage.trim()) {
    return backendMessage
  }
  const list = Array.isArray(issues)
    ? issues.filter((item): item is string => typeof item === 'string')
    : []
  const parts = list.map((code) => QUALITY_ISSUE_LABELS[code] ?? code)
  if (parts.length === 0) {
    return '图像质量未达标，请重新拍摄更清晰的眼部特写'
  }
  return `图像质量未达标：${parts.join('；')}`
}

export interface ParsedAnalysisError {
  message: string
  qualityCheckFailed: boolean
}

/** 将 API 错误响应解析为用户可读提示。 */
export function parseAnalysisError(data: unknown): ParsedAnalysisError {
  const payload = unwrapErrorPayload(data)
  const errorCode = typeof payload.error === 'string' ? payload.error : undefined

  if (errorCode === 'quality_check_failed') {
    const quality = payload.quality as Record<string, unknown> | undefined
    return {
      message: formatQualityMessage(quality?.issues, payload.message),
      qualityCheckFailed: true,
    }
  }

  if (errorCode === 'no_iris_detected') {
    return { message: ERROR_LABELS.no_iris_detected, qualityCheckFailed: false }
  }

  if (errorCode && ERROR_LABELS[errorCode]) {
    return { message: ERROR_LABELS[errorCode], qualityCheckFailed: false }
  }

  if (typeof payload.message === 'string' && payload.message.trim()) {
    return { message: payload.message, qualityCheckFailed: false }
  }

  if (errorCode) {
    return { message: `分析失败（${errorCode}）`, qualityCheckFailed: false }
  }

  return { message: '分析失败，请重试', qualityCheckFailed: false }
}

export async function analyzeIris(
  file: File,
  skipQuality = false,
): Promise<AnalysisResult> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post<AnalysisResult>('/iris/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    params: { skip_quality: skipQuality },
  })
  return response.data
}

export async function analyzeIrisManual(
  file: File,
  manualParams: DetectionInfo,
  skipQuality = false,
): Promise<AnalysisResult> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('manual_params', JSON.stringify({
    center_x: manualParams.center_x,
    center_y: manualParams.center_y,
    pupil_radius: manualParams.pupil_radius,
    inner_radius: manualParams.inner_radius,
    outer_radius: manualParams.outer_radius,
  }))

  const response = await api.post<AnalysisResult>('/iris/analyze/manual', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    params: { skip_quality: skipQuality },
  })
  return response.data
}

export async function checkHealth(): Promise<unknown> {
  const response = await api.get('/health')
  return response.data
}
