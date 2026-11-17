import type { GraphOut, SearchResponse } from '@/api/types'

export interface ChainStep {
  i: number
  from: string
  to: string
  edge: string
  clause: string | null
  role: string | null
  excerpt: string
  note: string
  g: number | null
  h: number | null
  f: number | null
  conf: number | null
  llm_reward: number | null
  clf_score: number | null
  rule_mask: string[]
}

export function buildChain(result: SearchResponse | null, graph: GraphOut | null): ChainStep[] {
  if (!result) return []
  if (result.explanation?.steps?.length) {
    return result.explanation.steps.map((s) => ({
      i: s.i, from: s.from, to: s.to, edge: s.edge, clause: s.clause, role: s.role,
      excerpt: s.excerpt, note: s.note, g: s.g, h: s.h, f: s.f, conf: s.conf,
      llm_reward: s.llm_reward, clf_score: s.clf_score, rule_mask: s.rule_mask ?? [],
    }))
  }
  const nodeMap = new Map((graph?.nodes ?? []).map((n) => [n.nid, n]))
  return (result.path ?? []).map((a, idx) => {
    const node = nodeMap.get(a.dst)
    return {
      i: idx + 1, from: a.src, to: a.dst, edge: a.edge_type,
      clause: node?.clause ?? null, role: node?.role ?? null,
      excerpt: node?.text ? node.text.slice(0, 90) : '',
      note: '', g: null, h: null, f: null, conf: null,
      llm_reward: null, clf_score: null, rule_mask: [],
    }
  })
}

export function ruleHits(result: SearchResponse | null): Array<[string, number]> {
  if (!result) return []
  const hits = result.explanation?.signals?.rule_hits
  if (hits && Object.keys(hits).length) return Object.entries(hits)
  const counts: Record<string, number> = {}
  for (const row of result.trace ?? []) {
    for (const rule of row.rule_mask ?? []) counts[rule] = (counts[rule] ?? 0) + 1
  }
  return Object.entries(counts)
}

export function lastSignal(result: SearchResponse | null, key: 'llm_reward' | 'clf_score' | 'risk'): number | null {
  if (!result) return null
  const signals = result.explanation?.signals
  if (signals && signals[key] !== undefined && signals[key] !== null) return Number(signals[key])
  const rows = result.trace ?? []
  for (let i = rows.length - 1; i >= 0; i--) {
    const value = rows[i][key === 'clf_score' ? 'clf_score' : key === 'llm_reward' ? 'llm_reward' : 'clf_score']
    if (value !== undefined && value !== null) return Number(value)
  }
  return null
}

export function aggregateRisk(result: SearchResponse | null): number | null {
  if (!result) return null
  const risk = result.explanation?.signals?.risk
  if (typeof risk === 'number') return risk
  const rows = result.trace ?? []
  for (let i = rows.length - 1; i >= 0; i--) {
    const r = rows[i] as unknown as { risk?: number }
    if (typeof r.risk === 'number') return r.risk
  }
  return null
}
