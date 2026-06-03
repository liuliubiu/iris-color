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

export async function analyzeIris(file: File, skipQuality = false): Promise<AnalysisResult> {
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
