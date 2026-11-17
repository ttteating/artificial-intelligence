const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '/api'

export class ApiError extends Error {
  status: number
  detail: string
  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit, timeout = 90_000): Promise<T> {
  const ctrl = new AbortController()
  const timer = window.setTimeout(() => ctrl.abort(), timeout)
  try {
    const res = await fetch(`${BASE}${path}`, {
      ...init,
      signal: init?.signal ?? ctrl.signal,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`
      try {
        const body = await res.json()
        if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      } catch {
        /* 非 JSON 错误体：保留状态行 */
      }
      throw new ApiError(res.status, detail)
    }
    if (res.status === 204) return undefined as T
    return (await res.json()) as T
  } catch (err) {
    if (err instanceof ApiError) throw err
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError(0, '请求超时或已取消')
    }
    throw new ApiError(0, err instanceof Error ? err.message : '网络请求失败')
  } finally {
    window.clearTimeout(timer)
  }
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal, timeout?: number) =>
    request<T>(path, { method: 'GET', signal }, timeout),
  post: <T>(path: string, body: unknown, signal?: AbortSignal, timeout?: number) =>
    request<T>(path, { method: 'POST', body: JSON.stringify(body), signal }, timeout),
}
