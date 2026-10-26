import type { Action, Row, Values } from './types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const detail = typeof body?.detail === 'string' ? body.detail : res.statusText
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json()
}

export const fetchRows = () => request<Row[]>('/api/rows')

export const review = (idx: number, action: Action, values?: Values) =>
  request<Row>(`/api/rows/${idx}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, values }),
  })

export const reset = (idx: number) => request<Row>(`/api/rows/${idx}/reset`, { method: 'POST' })

export const imageUrl = (idx: number) => `/api/rows/${idx}/image`
