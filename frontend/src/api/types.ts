export type Split = 'dev' | 'test' | 'custom'
export type Condition = 'B0' | 'B1' | 'A0' | 'A1' | 'A2' | 'M0' | 'M1' | 'H'
export type Label = 'Compliant' | 'Non-Compliant' | 'Not-Applicable'

export interface Health {
  status: string
  ready: boolean
  uptime_sec: number
  n_scenarios: number
  llm_mode: string
  graph_version: string
  error: string | null
}

export interface ConditionInfo {
  id: Condition
  kernel: string
  rules: boolean
  h: string
  llm: string
  clf: boolean
  description: string
}

export interface Action {
  src: string
  edge_type: string
  dst: string
}

export interface TreeNode {
  tid: number
  parent: number | null
  children: number[]
  depth: number
  node_id: string | null
  clause: string | null
  role: string | null
  action: Action | null
  g: number
  h: number
  f: number
  visits: number
  value: number
  conf: number | null
  label: string | null
  status: 'expanded' | 'frontier' | 'pruned' | 'goal' | 'deadend'
  flags: string[]
}

export interface TraceRow {
  action: Action | null
  depth: number
  g: number
  h: number
  f: number
  label: string
  conf: number
  clauses: string[]
  clf_score?: number
  llm_reward?: number
  rule_mask?: string[]
  adjust?: number
  expanded?: boolean
}

export interface Sufficiency {
  covered_clauses: string[]
  relevant_clauses: string[]
  coverage_ok: boolean
  exception_required: boolean
  exception_ok: boolean
  confidence: number
  tau_conf: number
  confidence_ok: boolean
  reached_goal: boolean
}

export interface ExplainStep {
  i: number
  from: string
  to: string
  edge: string
  clause: string | null
  role: string | null
  excerpt: string
  g: number | null
  h: number | null
  f: number | null
  conf: number | null
  llm_reward: number | null
  clf_score: number | null
  rule_mask: string[]
  note: string
  llm_note?: string
}

export interface Explanation {
  text: string
  steps: ExplainStep[]
  citations: { clause: string; excerpt: string }[]
  signals: {
    llm_reward?: number
    clf_score?: number | null
    risk?: number
    rule_hits?: Record<string, number>
    expanded_nodes?: number
    pruned_by_rule?: number
    llm_rejected?: number
    llm_suspicious?: number
    llm_calls?: number
  }
  sufficiency: Sufficiency
  mode: string
  violation: boolean
  llm_calls: number
  raw?: unknown
}

export interface SearchMeta {
  sid: string
  condition: Condition
  gold: string
  pred: string
  steps: number
  expanded: number
  confidence: number
  llm_calls: number
  n_clauses: number
  llm_mode: string
  needs_exc: boolean
  n_relevant: number
  clauses: Record<string, string>
  scenario: string
  path: Action[]
}

export interface SearchResponse {
  sid: string
  condition: Condition
  answer: Label
  confidence: number
  steps: number
  expanded_nodes: number
  reached_goal: boolean
  llm_calls: number
  pruned_by_rule: number
  llm_rejected: number
  llm_mode: string
  tree: TreeNode[] | null
  n_nodes_raw: number
  trace: TraceRow[] | null
  path: Action[]
  meta: SearchMeta
  explanation: Explanation | null
  elapsed_ms: number
}

export interface ScenarioRun {
  condition: Condition
  answer: Label
  gold: string
  correct: boolean
  expanded: number
  steps: number
  redundant: number
  llm_calls: number
  llm_rejected: number
  conf: number
}

export interface ScenarioOut {
  sid: string
  split: Split
  gold: string
  n_clauses: number
  oracle_hops: number | null
  text: string
  clauses: Record<string, string>
  runs: ScenarioRun[]
}

export interface ScenarioListItem {
  sid: string
  split: Split
  gold: string
  n_clauses: number
  oracle_hops: number | null
  runs: Record<string, {
    answer: Label
    correct: boolean
    expanded: number
    steps: number
    redundant: number
    llm_calls: number
    conf: number
  }>
}

export interface ScenarioList {
  total: number
  offset: number
  limit: number
  items: ScenarioListItem[]
}

export interface GraphNode {
  nid: string
  clause: string
  role: string
  text: string
  sent_idx: number
}

export interface GraphEdge {
  src: string
  etype: string
  dst: string
  w: number
}

export interface GraphOut {
  sid: string
  graph_version: string
  n_nodes: number
  n_edges: number
  entry: string[]
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface TableOut {
  name: string
  kind: 'json' | 'csv' | 'markdown'
  n_rows: number
  columns: string[]
  data: unknown
  text: string | null
}

export interface SummaryRow {
  n: number
  macro_f1: number
  accuracy: number
  avg_expanded: number
  avg_steps: number
  redundant_ratio: number
  avg_llm_calls: number
  goal_rate: number
  avg_pruned: number
  avg_llm_rejected: number
  recall_Compliant: number
  'recall_Non-Compliant': number
  'recall_Not-Applicable': number
  confusion: Record<string, Record<string, number>>
}

export interface Overview {
  health: Health
  conditions: ConditionInfo[]
  summary_dev: Record<string, SummaryRow>
  summary_test: Record<string, SummaryRow>
  graph_stats: Record<string, unknown>
  trees: Record<string, { sid: string; condition: string; json: string; html: string; nodes: number }>
  llm_reject: { reference: number; rows: Array<Record<string, unknown>> }
  ga_summary: Record<string, unknown>
  lambda_best: Record<string, number> | null
  counts: { dev: number | null; test: number | null }
}

export interface CustomScenarioOut {
  sid: string
  title: string
  n_clauses: number
  n_nodes: number
  n_edges: number
  entry: string[]
  clause_sim: Record<string, number>
  relevant: string[]
  graph_version: string
  cached: boolean
}
