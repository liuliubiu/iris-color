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

export interface AnalysisResult {
  success: boolean
  quality?: QualityInfo
  lab?: LabValues
  iris_color?: IrisColorInfo
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

export async function analyzeIris(file: File): Promise<AnalysisResult> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post<AnalysisResult>('/iris/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

export async function checkHealth(): Promise<unknown> {
  const response = await api.get('/health')
  return response.data
}
