import { paths } from './api-types'

type ApiPaths = paths

type GetResponse<T extends keyof ApiPaths> = 
  ApiPaths[T] extends { get: { responses: { 200: { content: { 'application/json': infer R } } } } }
    ? R
    : never

type PostRequest<T extends keyof ApiPaths> = 
  ApiPaths[T] extends { post: { requestBody: { content: { 'application/json': infer R } } } }
    ? R
    : never

type PostResponse<T extends keyof ApiPaths> = 
  ApiPaths[T] extends { post: { responses: { 200: { content: { 'application/json': infer R } } } } }
    ? R
    : never

const API_BASE = 'http://localhost:8000'

class ApiError extends Error {
  constructor(public status: number, message: string, public response?: any) {
    super(message)
    this.name = 'ApiError'
  }
}

async function apiRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`
  
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`
    try {
      const errorData = await response.json()
      errorMessage = errorData.detail || errorData.message || errorMessage
    } catch {
      // Use default error message if response isn't JSON
    }
    throw new ApiError(response.status, errorMessage, response)
  }

  const data = await response.json()
  return data as T
}

// Type-safe API client
export const apiClient = {
  // Dashboard stats
  getDashboardStats(): Promise<GetResponse<'/api/dashboard/stats'>> {
    return apiRequest('/api/dashboard/stats')
  },

  // Predictions
  getPredictions(): Promise<GetResponse<'/api/predictions'>> {
    return apiRequest('/api/predictions')
  },

  // Bets/Tracking
  getBets(): Promise<GetResponse<'/api/bets'>> {
    return apiRequest('/api/bets')
  },

  trackPrediction(data: PostRequest<'/api/bets'>): Promise<PostResponse<'/api/bets'>> {
    return apiRequest('/api/bets', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },

  settleBet(betId: number, data: { result: 'Won' | 'Lost' }): Promise<PostResponse<'/api/bets/{bet_id}/settle'>> {
    return apiRequest(`/api/bets/${betId}/settle`, {
      method: 'PUT',
      body: JSON.stringify(data),
    })
  },

  // BetAI chat
  betaiQuery(data: PostRequest<'/api/betai/query'>): Promise<any> {
    return apiRequest('/api/betai/query', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },
}

export { ApiError }
export type { GetResponse, PostRequest, PostResponse }