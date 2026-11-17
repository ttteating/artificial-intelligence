import { api } from './client'
import type {
  Condition, CustomScenarioOut, Explanation, GraphOut, Health, Overview,
  ScenarioList, ScenarioOut, SearchResponse, TableOut,
} from './types'

export const endpoints = {
  health: (signal?: AbortSignal) => api.get<Health>('/health', signal, 10_000),

  conditions: (signal?: AbortSignal) => api.get<import('./types').ConditionInfo[]>('/conditions', signal),

  overview: (signal?: AbortSignal) => api.get<Overview>('/overview', signal),

  summary: (split: 'dev' | 'test', signal?: AbortSignal) =>
    api.get<Record<string, import('./types').SummaryRow>>(`/summary/${split}`, signal),

  table: (name: string, signal?: AbortSignal) =>
    api.get<TableOut>(`/table/${encodeURIComponent(name)}`, signal),

  scenarios: (params: {
    split?: string
    condition?: string
    correct?: boolean
    min_hops?: number
    max_hops?: number
    n_clauses?: number
    q?: string
    limit?: number
    offset?: number
  }, signal?: AbortSignal) => {
    const qs = new URLSearchParams()
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') qs.set(k, String(v))
    })
    return api.get<ScenarioList>(`/scenarios?${qs.toString()}`, signal)
  },

  scenario: (sid: string, signal?: AbortSignal) =>
    api.get<ScenarioOut>(`/scenario/${encodeURIComponent(sid)}`, signal),

  graph: (sid: string, signal?: AbortSignal) =>
    api.get<GraphOut>(`/graph/${encodeURIComponent(sid)}`, signal, 30_000),

  search: (payload: {
    sid: string
    condition: Condition
    explain?: boolean
    explain_mode?: 'template' | 'real' | 'auto'
    max_expansions?: number | null
    include_trace?: boolean
    include_tree?: boolean
  }, signal?: AbortSignal) =>
    api.post<SearchResponse>('/search', payload, signal, 120_000),

  explain: (payload: { sid: string; condition: Condition; mode: 'template' | 'real' | 'auto' },
            signal?: AbortSignal) =>
    api.post<{ sid: string; condition: string; answer: string; explanation: Explanation }>(
      '/explain', payload, signal, 120_000),

  trees: (signal?: AbortSignal) =>
    api.get<Record<string, { sid: string; condition: string; json: string; html: string; nodes: number }>>(
      '/trees', signal),

  tree: (name: string, signal?: AbortSignal) =>
    api.get<Record<string, unknown>>(`/tree/${encodeURIComponent(name)}`, signal),

  llmReport: (signal?: AbortSignal) =>
    api.get<{
      reject: { reference: number; rows: Array<Record<string, unknown>> }
      cost_md: string
      real_vs_mock_md: string
      real_vs_mock_a2_md: string
      smoke_md: string
    }>('/llm/report', signal),

  createScenario: (payload: {
    scenario_text: string
    clauses: Record<string, string>
    title?: string | null
    label?: string | null
    persist?: boolean
  }, signal?: AbortSignal) =>
    api.post<CustomScenarioOut>('/scenarios', payload, signal, 60_000),
}
